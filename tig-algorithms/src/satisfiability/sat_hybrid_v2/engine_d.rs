//
// Il remplace `engine_a::track3` qui est l'ANCÊTRE du même moteur probSAT, resté
// en amont des 4 optimisations KEPT de la lignée v9 :
//   i8  scan-fusion per-flip          (−20 % : 1 000 630 → 800 610 ms)
//   i9  branchless sad-scan           (−0.2 %)
//   i10 weights-buffer [0.0;256]→WBUF=8 (−2.5 % : 2 KB de memset mort par flip)
//   i15 2-bit L1-pack de num_good     (−6.2 % : 42 KB L2 → 10.7 KB L1d-résident)
// et sous-fuellé (180B ⇒ 6/32 ; le seuil du 7ᵉ nonce est dans ]275B, 315B]).
//
// trajectoire RNG tolérée : cf dead-list v9 « tout coup touchant la trajectoire
// RNG »). Seules adaptations : `super::Prepared`/`super::preprocess`/`super::Hparams`
// sont internalisés ici pour que le fichier soit auto-suffisant dans l'archi
// engine_* de sat_hybrid, et un wrapper `solve()` parse la Map JSON des HP.
use anyhow::Result;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tig_challenges::satisfiability::*;

#[derive(Default, Clone, Debug, Serialize, Deserialize)]
pub struct Hparams {
    pub check_interval: Option<usize>,
    pub max_fuel_high: Option<f64>,
}

pub(crate) struct Prepared {
    pub rng: SmallRng,
    pub nv: usize,
    pub nc: usize,
    pub density: f64,
    pub p_cnt: Vec<u32>,
    pub n_cnt: Vec<u32>,
    pub all_off: Vec<u32>,
    pub p_bound: Vec<u32>,
    pub all_data: Vec<u32>,
    pub cl: Vec<i32>,
    pub co: Vec<u32>,
}

#[inline(always)]
pub(crate) fn preprocess(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Prepared {
    let nv = challenge.num_variables;
    let _ = save_solution(&Solution { variables: vec![false; nv] });
    let rng = SmallRng::seed_from_u64(u64::from_le_bytes(
        challenge.seed[..8].try_into().unwrap(),
    ));

    let mut p_cnt = vec![0u32; nv];
    let mut n_cnt = vec![0u32; nv];
    let mut good_clauses = 0u32;

    for orig in &challenge.clauses {
        let (a, b, c) = (orig[0], orig[1], orig[2]);
        if a == -b || a == -c || b == -c { continue; }
        good_clauses += 1;
        let va = (a.abs() - 1) as usize;
        if a > 0 { p_cnt[va] += 1; } else { n_cnt[va] += 1; }
        if b != a {
            let vb = (b.abs() - 1) as usize;
            if b > 0 { p_cnt[vb] += 1; } else { n_cnt[vb] += 1; }
        }
        if c != a && c != b {
            let vc = (c.abs() - 1) as usize;
            if c > 0 { p_cnt[vc] += 1; } else { n_cnt[vc] += 1; }
        }
    }

    let nc = good_clauses as usize;

    let mut all_off = vec![0u32; nv + 1];
    for v in 0..nv {
        all_off[v + 1] = all_off[v] + p_cnt[v] + n_cnt[v];
    }
    let total_entries = all_off[nv] as usize;
    let mut all_data = vec![0u32; total_entries];
    let mut p_bound = vec![0u32; nv];
    let mut cl = Vec::with_capacity(nc * 3);
    let mut co = Vec::with_capacity(nc + 1);
    co.push(0u32);

    {
        let mut p_pos = vec![0u32; nv];
        let mut n_pos = vec![0u32; nv];
        for v in 0..nv {
            p_pos[v] = all_off[v];
            n_pos[v] = all_off[v] + p_cnt[v];
            p_bound[v] = n_pos[v];
        }
        let mut ci = 0u32;
        for orig in &challenge.clauses {
            let (a, b, c) = (orig[0], orig[1], orig[2]);
            if a == -b || a == -c || b == -c { continue; }
            let va = (a.abs() - 1) as usize;
            if a > 0 { all_data[p_pos[va] as usize] = ci; p_pos[va] += 1; }
            else { all_data[n_pos[va] as usize] = ci; n_pos[va] += 1; }
            if b != a {
                let vb = (b.abs() - 1) as usize;
                if b > 0 { all_data[p_pos[vb] as usize] = ci; p_pos[vb] += 1; }
                else { all_data[n_pos[vb] as usize] = ci; n_pos[vb] += 1; }
            }
            if c != a && c != b {
                let vc = (c.abs() - 1) as usize;
                if c > 0 { all_data[p_pos[vc] as usize] = ci; p_pos[vc] += 1; }
                else { all_data[n_pos[vc] as usize] = ci; n_pos[vc] += 1; }
            }
            cl.push(a);
            if b != a { cl.push(b); }
            if c != a && c != b { cl.push(c); }
            co.push(cl.len() as u32);
            ci += 1;
        }
    }

    let density = nc as f64 / nv as f64;

    Prepared { rng, nv, nc, density, p_cnt, n_cnt, all_off, p_bound, all_data, cl, co }
}

// Entry point used by `mod.rs` for the (nv=10000, nc=42670) track.
// `max_fuel_high` is baked to 315B at the dispatch site (see mod.rs).
pub fn solve(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let hp: Hparams = hyperparameters
        .as_ref()
        .and_then(|m| serde_json::from_value::<Hparams>(Value::Object(m.clone())).ok())
        .unwrap_or_default();
    solve_t1(challenge, save_solution, &hp)
}

// ── 2-bit L1-resident pack of `num_good` (i15) ──────────────────────────────
// t1 is pure random 3-SAT: every clause has exactly 3 literals, so the
// satisfied-literal count `num_good[c]` is ALWAYS in {0,1,2,3} = exactly 2 bits.
// The former `Vec<u8>` (nc = 42 670 B ≈ 42 KB > 32 KB L1d) forced every scattered
// `num_good[c]` gather to L1-miss → L2. Packing 4 cells per byte shrinks the
// footprint to (nc+3)/4 ≈ 10.7 KB < 32 KB ⇒ the whole array becomes L1d-resident
// and the gathers stop missing. BYTE-IDENTICAL: no logical value, no RNG draw, no
// scan/pick order changes — only the storage layout. (Distinct from the DEAD
// breaks[]/crit_sum family i12/i13: NO extra state, NO maintenance loop, NO
// crit_sum — pure footprint compression, exactly the 2-bit pack already proven a
// win on this algo, v6 t3/t38, Hindsight 09803f0a/18c70ac6.)
//
// In the add path a cell receiving +1 had that literal contributing 0 before, so
// its prior value is ≤2 ⇒ +1 ≤3 (never wraps the 2-bit field); in the sub path the
// flipped literal was contributing 1 ⇒ prior value ≥1 ⇒ −1 ≥0. The cap/floor below
// mirror the original `saturating_add/sub` exactly within this {0..3} regime.
#[inline(always)]
unsafe fn ng_get(p: &[u8], c: usize) -> u8 {
    (*p.get_unchecked(c >> 2) >> ((c & 3) << 1)) & 0x3
}
#[inline(always)]
unsafe fn ng_set(p: &mut [u8], c: usize, val: u8) {
    let sh = (c & 3) << 1;
    let b = p.get_unchecked_mut(c >> 2);
    *b = (*b & !(0x3u8 << sh)) | ((val & 0x3) << sh);
}
// increment cell c (cap 3); returns the OLD value (mirrors `if *ng == 0`).
#[inline(always)]
unsafe fn ng_inc(p: &mut [u8], c: usize) -> u8 {
    let sh = (c & 3) << 1;
    let b = p.get_unchecked_mut(c >> 2);
    let cur = (*b >> sh) & 0x3;
    let nv = if cur >= 3 { 3 } else { cur + 1 };
    *b = (*b & !(0x3u8 << sh)) | (nv << sh);
    cur
}
// decrement cell c (floor 0); returns the NEW value (mirrors `new_val`).
#[inline(always)]
unsafe fn ng_dec(p: &mut [u8], c: usize) -> u8 {
    let sh = (c & 3) << 1;
    let b = p.get_unchecked_mut(c >> 2);
    let cur = (*b >> sh) & 0x3;
    let nv = if cur == 0 { 0 } else { cur - 1 };
    *b = (*b & !(0x3u8 << sh)) | (nv << sh);
    nv
}

// T1 (n_vars=10000): self-contained solver providing the best reproducible valid Q.
fn solve_t1(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hp: &Hparams,
) -> Result<()> {
    let Prepared {
        mut rng,
        nv,
        nc,
        density,
        p_cnt,
        n_cnt,
        all_off,
        p_bound,
        all_data,
        mut cl,
        co,
    } = preprocess(challenge, save_solution);

    let default_fuel = if nv >= 10000 { 125_000_000_000.0 } else { 250_000_000_000.0 };
    let max_fuel = hp.max_fuel_high.unwrap_or(default_fuel);

    let avg_clause_size = cl.len() as f64 / nc as f64;
    let difficulty_factor = density * avg_clause_size.sqrt();
    let scale_factor = if nv > 25000 { 1.5 } else { 1.0 };
    let base_fuel = (2000.0 + 100.0 * difficulty_factor) * (nv as f64).sqrt() * scale_factor;
    let flip_fuel = (200.0 + difficulty_factor) / scale_factor;
    let remaining = (max_fuel - base_fuel).max(0.0);
    let max_flips = if flip_fuel > 0.0 { (remaining / flip_fuel) as usize } else { 0 };

    let mut vars = vec![false; nv];
    for v in 0..nv {
        let np = p_cnt[v] as usize;
        let nn = n_cnt[v] as usize;
        if np > nn {
            vars[v] = true;
        } else if np < nn {
            vars[v] = false;
        } else {
            vars[v] = rng.gen_bool(0.5);
        }
    }

    // 2-bit packed: 4 cells per byte ⇒ (nc + 3) / 4 bytes (≈10.7 KB, L1d-resident).
    let mut num_good = vec![0u8; (nc + 3) / 4];
    for i in 0..nc {
        let s = co[i] as usize;
        let e = co[i + 1] as usize;
        let mut good = 0u8;
        for j in s..e {
            let l = cl[j];
            let v = (l.abs() - 1) as usize;
            if (l > 0 && vars[v]) || (l < 0 && !vars[v]) {
                good += 1;
            }
        }
        // good ∈ {0..3} for 3-SAT; ng_set writes the 2-bit field (byte-identical).
        unsafe { ng_set(&mut num_good, i, good); }
    }

    let mut residual: Vec<u32> = Vec::with_capacity(nc);
    let mut true_unsat = 0usize;
    for i in 0..nc {
        if unsafe { ng_get(&num_good, i) } == 0 {
            residual.push(i as u32);
            true_unsat += 1;
        }
    }

    if true_unsat == 0 {
        let _ = save_solution(&Solution { variables: vars });
        return Ok(());
    }

    let mut best_unsat = true_unsat;
    let mut best_vars = vars.clone();

    let large_problem_scale = ((nv as f64 - 25000.0) / 35000.0).max(0.0).min(1.0);
    let base_interval = 60.0 - 30.0 * large_problem_scale;
    let min_interval = if large_problem_scale > 0.0 { 15.0 } else { 25.0 };
    let density_factor_ci = if density > 4.0 { 1.2 } else { 1.0 };
    let check_interval = hp.check_interval
        .unwrap_or((base_interval * density_factor_ci * (1.0 + (density / 3.0).ln().max(0.0))).max(min_interval) as usize);

    let mut probsat_weights = vec![0.0f64; nc + 1];
    if avg_clause_size <= 3.2 {
        let cb: f64 = 2.06;
        for i in 0..=nc {
            probsat_weights[i] = cb.powf(-(i as f64));
        }
    } else {
        let cb: f64 = if avg_clause_size <= 4.2 {
            2.85
        } else if avg_clause_size <= 5.2 {
            3.7
        } else if avg_clause_size <= 6.2 {
            5.1
        } else {
            5.4
        };
        for i in 0..=nc {
            probsat_weights[i] = (i as f64 + 1.0).powf(-cb);
        }
    }

    let mut last_check_unsat = true_unsat;
    let mut rounds = 0usize;
    let mut stagnation = 0usize;

    unsafe {
        loop {
            if rounds >= max_flips { break; }
            if true_unsat == 0 { break; }

            if rounds % check_interval == 0 && rounds > 0 {
                let progress = last_check_unsat as i64 - true_unsat as i64;
                let progress_ratio = progress as f64 / last_check_unsat.max(1) as f64;
                let progress_threshold = 0.15 + 0.05 * (density / 3.0).min(1.0);

                if progress <= 0 {
                    stagnation += 1;

                    if stagnation >= 4 {
                        if stagnation >= 15 {
                            vars.copy_from_slice(&best_vars);
                            let perturb_cnt = (nv / 20).max(1);
                            for _ in 0..perturb_cnt {
                                let v = rng.gen::<usize>() % nv;
                                *vars.get_unchecked_mut(v) = !*vars.get_unchecked(v);
                            }
                            
                            residual.clear();
                            true_unsat = 0;
                            for i in 0..nc {
                                let s = *co.get_unchecked(i) as usize;
                                let e = *co.get_unchecked(i + 1) as usize;
                                let mut good = 0u8;
                                for j in s..e {
                                    let l = *cl.get_unchecked(j);
                                    let v = (l.abs() - 1) as usize;
                                    if (l > 0 && *vars.get_unchecked(v)) || (l < 0 && !*vars.get_unchecked(v)) {
                                        good = good.saturating_add(1);
                                    }
                                }
                                ng_set(&mut num_good, i, good);
                                if good == 0 {
                                    residual.push(i as u32);
                                    true_unsat += 1;
                                }
                            }
                            stagnation = 0;
                        } else {
                            let kicks = if stagnation >= 8 { 6 } else { 3 };
                            for _ in 0..kicks {
                                if true_unsat == 0 { break; }
                                let rid = rng.gen::<usize>() % residual.len();
                                let pcid = *residual.get_unchecked(rid) as usize;
                                if ng_get(&num_good, pcid) > 0 {
                                    residual.swap_remove(rid);
                                    continue;
                                }
                                let pcs = *co.get_unchecked(pcid) as usize;
                                let pce = *co.get_unchecked(pcid + 1) as usize;
                                if pcs == pce { continue; }
                                let lit = *cl.get_unchecked(pcs + rng.gen::<usize>() % (pce - pcs));
                                let v = (lit.abs() - 1) as usize;

                                let was_true = *vars.get_unchecked(v);
                                let (is, ie) = if was_true {
                                    (*p_bound.get_unchecked(v) as usize, *all_off.get_unchecked(v + 1) as usize)
                                } else {
                                    (*all_off.get_unchecked(v) as usize, *p_bound.get_unchecked(v) as usize)
                                };
                                let (ds, de) = if was_true {
                                    (*all_off.get_unchecked(v) as usize, *p_bound.get_unchecked(v) as usize)
                                } else {
                                    (*p_bound.get_unchecked(v) as usize, *all_off.get_unchecked(v + 1) as usize)
                                };

                                for k in is..ie {
                                    let c = *all_data.get_unchecked(k) as usize;
                                    if ng_inc(&mut num_good, c) == 0 { true_unsat = true_unsat.saturating_sub(1); }
                                }
                                for k in ds..de {
                                    let c = *all_data.get_unchecked(k) as usize;
                                    if ng_dec(&mut num_good, c) == 0 {
                                        residual.push(c as u32);
                                        true_unsat += 1;
                                    }
                                }
                                *vars.get_unchecked_mut(v) = !was_true;
                            }
                            stagnation = 0;
                        }
                    }
                } else if progress_ratio > progress_threshold {
                    stagnation = 0;
                } else {
                    stagnation = 0;
                }

                last_check_unsat = true_unsat;
            }

            if true_unsat == 0 { break; }

            let mut cid = usize::MAX;
            let mut min_len = usize::MAX;
            for _ in 0..3 {
                while !residual.is_empty() {
                    let id = rng.gen::<usize>() % residual.len();
                    let cand = *residual.get_unchecked(id) as usize;
                    if ng_get(&num_good, cand) > 0 {
                        residual.swap_remove(id);
                    } else {
                        let c_s = *co.get_unchecked(cand) as usize;
                        let c_e = *co.get_unchecked(cand + 1) as usize;
                        let clen = c_e - c_s;
                        if clen < min_len {
                            min_len = clen;
                            cid = cand;
                        }
                        break;
                    }
                }
                if residual.is_empty() { break; }
            }
            if cid == usize::MAX { break; }

            let cs = *co.get_unchecked(cid) as usize;
            let ce = *co.get_unchecked(cid + 1) as usize;
            let clen = ce - cs;

            if clen > 1 {
                let ri = rng.gen::<usize>() % clen;
                cl.swap(cs, cs + ri);
            }

            // SCAN-FUSION (i8): single per-literal sad-scan replacing the former
            // zero_found pre-scan + roulette re-scan. Each literal's break-count
            // `sad` is computed ONCE. If a literal has sad==0 we short-circuit
            // immediately — this reproduces the original `zero_found` pick EXACTLY
            // (first literal in clause order with break==0, since the original loop
            // also broke at the first such literal). When no break-0 literal exists
            // we fall through to the roulette using the `sad` already accumulated.
            // RNG order is byte-identical: the only draws are `cl.swap` (above,
            // unchanged) and the roulette `gen::<f64>` (below, still reached ONLY
            // when no break-0 literal exists) => trajectory preserved => 7/32 kept.
            // NB: t1 is 3-SAT (clen<=3 < 256) so clen_actual==clen always and the
            // fused pass covers the full clause exactly like the original zero scan.
            //
            // WEIGHTS-BUFFER-SHRINK (i10, gisement #1): the former `[0.0; 256]`
            // forced a 2 KB f64 memset PER FLIP. LLVM cannot elide it: the buffer
            // is both written (`weights[idx] = w`) and read by a runtime index
            // (`r -= weights[idx]`), so alias-analysis keeps the full zero-init.
            // On billions of flips that is ~248 dead stores/flip. t1 is 3-SAT
            // (every clause has clen <= 3), so a const WBUF=8 majorant is provably
            // never truncating: clen_actual == clen == (ce-cs) exactly as before.
            // The scanned literals, their `sad`, every `weights[idx]` value and the
            // RNG draws (`cl.swap` + roulette `gen::<f64>`) are UNCHANGED => picks
            // are byte-identical => trajectory preserved => 7/32 conserved.
            // (If a clause ever exceeded WBUF the pick would change and Q would
            // drop below 218750 — the iter gate REJECTs that, so it is self-guarding.)
            const WBUF: usize = 8;
            let mut total_weight = 0.0;
            let mut weights = [0.0; WBUF];
            let clen_actual = (ce - cs).min(WBUF);
            let mut zero_found: Option<usize> = None;

            for idx in 0..clen_actual {
                let j = cs + idx;
                let l = *cl.get_unchecked(j);
                let abs_l = (l.abs() - 1) as usize;
                let (os, oe) = if *vars.get_unchecked(abs_l) {
                    (*all_off.get_unchecked(abs_l) as usize, *p_bound.get_unchecked(abs_l) as usize)
                } else {
                    (*p_bound.get_unchecked(abs_l) as usize, *all_off.get_unchecked(abs_l + 1) as usize)
                };

                // BRANCHLESS-SCAN (i9): the per-flip break-count is THE SLS
                // bottleneck (Hindsight b9541ce8; run ~3x more than the update
                // loop). Replace the data-dependent `if num_good[c]==1 { sad+=1 }`
                // branch by a branchless add — `num_good[c]==1` (clause is
                // critically satisfied) is poorly predictable, so the branch
                // mispredicts often. `sad` is numerically IDENTICAL to i8 and no
                // RNG draw moves => trajectory byte-identical => 7/32 conserved.
                let mut sad = 0usize;
                for k in os..oe {
                    let c = *all_data.get_unchecked(k) as usize;
                    sad += (ng_get(&num_good, c) == 1) as usize;
                }

                if sad == 0 {
                    zero_found = Some(abs_l);
                    break;
                }

                let w = *probsat_weights.get_unchecked(sad.min(nc));
                weights[idx] = w;
                total_weight += w;
            }

            let v_idx = if let Some(v) = zero_found {
                v
            } else {
                let mut r = rng.gen::<f64>() * total_weight;
                let mut v_min = (cl.get_unchecked(cs).abs() - 1) as usize;
                for idx in 0..clen_actual {
                    r -= weights[idx];
                    if r <= 0.0 {
                        v_min = (cl.get_unchecked(cs + idx).abs() - 1) as usize;
                        break;
                    }
                }
                v_min
            };

            let was_true = *vars.get_unchecked(v_idx);
            let (is, ie) = if was_true {
                (*p_bound.get_unchecked(v_idx) as usize, *all_off.get_unchecked(v_idx + 1) as usize)
            } else {
                (*all_off.get_unchecked(v_idx) as usize, *p_bound.get_unchecked(v_idx) as usize)
            };
            let (ds, de) = if was_true {
                (*all_off.get_unchecked(v_idx) as usize, *p_bound.get_unchecked(v_idx) as usize)
            } else {
                (*p_bound.get_unchecked(v_idx) as usize, *all_off.get_unchecked(v_idx + 1) as usize)
            };

            for k in is..ie {
                let c = *all_data.get_unchecked(k) as usize;
                if ng_inc(&mut num_good, c) == 0 { true_unsat = true_unsat.saturating_sub(1); }
            }
            for k in ds..de {
                let c = *all_data.get_unchecked(k) as usize;
                if ng_dec(&mut num_good, c) == 0 {
                    residual.push(c as u32);
                    true_unsat += 1;
                }
            }

            *vars.get_unchecked_mut(v_idx) = !was_true;
            
            if true_unsat < best_unsat {
                best_unsat = true_unsat;
                best_vars.copy_from_slice(&vars);
            }

            rounds += 1;
        }
    }

    let _ = save_solution(&Solution { variables: vars });
    Ok(())
}
