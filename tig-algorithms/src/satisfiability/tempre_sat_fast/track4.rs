use anyhow::Result;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use tig_challenges::satisfiability::*;

pub fn solve(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Result<()> {
    let nv = challenge.num_variables;
    let _ = save_solution(&Solution { variables: vec![false; nv] });
    let mut rng = SmallRng::seed_from_u64(u64::from_le_bytes(challenge.seed[..8].try_into().unwrap()));

    let mut p_cnt = vec![0u32; nv];
    let mut n_cnt = vec![0u32; nv];
    let mut nc = 0usize;
    for orig in &challenge.clauses {
        let (a, b, c) = (orig[0], orig[1], orig[2]);
        if a == -b || a == -c || b == -c { continue; }
        nc += 1;
        let va = (a.unsigned_abs() as usize) - 1;
        if a > 0 { p_cnt[va] += 1; } else { n_cnt[va] += 1; }
        if b != a { let vb = (b.unsigned_abs() as usize) - 1; if b > 0 { p_cnt[vb] += 1; } else { n_cnt[vb] += 1; } }
        if c != a && c != b { let vc = (c.unsigned_abs() as usize) - 1; if c > 0 { p_cnt[vc] += 1; } else { n_cnt[vc] += 1; } }
    }

    let mut p_off = vec![0u32; nv + 1];
    let mut n_off = vec![0u32; nv + 1];
    for v in 0..nv { p_off[v+1] = p_off[v] + p_cnt[v]; n_off[v+1] = n_off[v] + n_cnt[v]; }
    let mut p_data = vec![0u32; p_off[nv] as usize];
    let mut n_data = vec![0u32; n_off[nv] as usize];
    let mut cl: Vec<i32> = Vec::with_capacity(nc * 3);
    let mut co: Vec<u32> = Vec::with_capacity(nc + 1);
    co.push(0);
    {
        let mut pp = p_off[..nv].to_vec();
        let mut np = n_off[..nv].to_vec();
        let mut ci = 0u32;
        for orig in &challenge.clauses {
            let (a, b, c) = (orig[0], orig[1], orig[2]);
            if a == -b || a == -c || b == -c { continue; }
            cl.push(a);
            let va = (a.unsigned_abs() as usize) - 1;
            if a > 0 { p_data[pp[va] as usize] = ci; pp[va] += 1; } else { n_data[np[va] as usize] = ci; np[va] += 1; }
            if b != a {
                cl.push(b); let vb = (b.unsigned_abs() as usize) - 1;
                if b > 0 { p_data[pp[vb] as usize] = ci; pp[vb] += 1; } else { n_data[np[vb] as usize] = ci; np[vb] += 1; }
            }
            if c != a && c != b {
                cl.push(c); let vc = (c.unsigned_abs() as usize) - 1;
                if c > 0 { p_data[pp[vc] as usize] = ci; pp[vc] += 1; } else { n_data[np[vc] as usize] = ci; np[vc] += 1; }
            }
            co.push(cl.len() as u32);
            ci += 1;
        }
    }

    let nvf = nv as f64;
    let density = nc as f64 / nv as f64;
    let max_fuel = 150_000_000_000.0f64;
    let avg_cs = cl.len() as f64 / nc as f64;
    let diff = density * avg_cs.sqrt();
    let base_fuel = (2000.0 + 100.0 * diff) * nvf.sqrt() * 1.5;
    let flip_fuel = (200.0 + diff) / 1.5;
    let max_flips = ((max_fuel - base_fuel).max(0.0) / flip_fuel) as usize;

    let mut vars = vec![false; nv];
    let rt = 0.003 + 0.007 / (1.0 + (-(nvf - 30000.0) / 8000.0).exp());
    let steep = 0.35 / (1.0 + (density - 4.18).max(0.0) * 12.0);
    for v in 0..nv {
        let np = p_cnt[v] as f64; let nn = n_cnt[v] as f64;
        if nn == 0.0 && np > 0.0 { vars[v] = true; continue; }
        if np == 0.0 { continue; }
        let vad = np / nn;
        let bp = (np + 0.25) / (np + nn + 1.2);
        let s = 1.0 / (1.0 + (-(vad - 1.0) / steep).exp());
        vars[v] = rng.gen_bool((rt * (1.0 - s) + bp * s).clamp(0.0, 1.0));
    }

    let appearances: Vec<u8> = (0..nv).map(|v| ((p_cnt[v] + n_cnt[v]) as usize).min(255) as u8).collect();
    let ng_len = (nc + 3) >> 2;
    let mut num_good = vec![0u8; ng_len];
    for i in 0..nc {
        let s = co[i] as usize; let e = co[i + 1] as usize;
        let shift = (i & 3) << 1; let bi = i >> 2;
        for j in s..e {
            let l = cl[j]; let v = (l.unsigned_abs() as usize) - 1;
            if (l > 0 && vars[v]) || (l < 0 && !vars[v]) { num_good[bi] += 1u8 << shift; }
        }
    }

    let mut residual: Vec<u32> = Vec::with_capacity(nc);
    for i in 0..nc { if (num_good[i >> 2] >> ((i & 3) << 1)) & 3 == 0 { residual.push(i as u32); } }
    if residual.is_empty() { let _ = save_solution(&Solution { variables: vars }); return Ok(()); }

    let base_prob = 0.45 + 0.1 * (density / 5.0).min(1.0);
    let mut current_prob = base_prob;
    let lps = ((nvf - 25000.0) / 35000.0).clamp(0.0, 1.0);
    let check_interval = ((60.0 - 30.0 * lps) * (1.0 + 0.2 / (1.0 + (-(density - 4.0) / 0.5).exp())) * (1.0 + (density / 3.0).ln().max(0.0))).max(25.0 - 10.0 * lps) as usize;
    let ss = 1.0 / (1.0 + (-(nvf - 30000.0) / 7000.0).exp());
    let pert_flips = 1 + (2.0 * ss) as usize;
    let stag_lim = 2 + (2.0 * (1.0 - (density / 5.0).min(1.0))) as usize;

    let mut last_cr = residual.len();
    let mut stagnation = 0usize;
    let mut var_age = vec![0u8; nv];
    let mut countdown = check_interval;
    let mut rounds = 0usize;

    unsafe {
    loop {
        if residual.is_empty() || rounds >= max_flips { break; }
        countdown -= 1;
        if countdown == 0 {
            countdown = check_interval;
            let progress = last_cr as i64 - residual.len() as i64;
            let pr = progress as f64 / last_cr.max(1) as f64;
            if progress <= 0 {
                stagnation += 1;
                current_prob = (current_prob + 0.03 * (-progress as f64 / last_cr.max(1) as f64).min(1.0)).min(0.9);
                if stagnation >= stag_lim {
                    let kicks = if stagnation >= 5 { (pert_flips * 12).min(100) } else if stagnation >= 4 { (pert_flips * 6).min(50) } else if stagnation >= 3 { (pert_flips * 3).min(20) } else { (pert_flips + 2).min(10) };
                    for _ in 0..kicks {
                        if residual.is_empty() { break; }
                        let rid = rng.gen::<usize>() % residual.len();
                        let pcid = *residual.get_unchecked(rid) as usize;
                        if (*num_good.get_unchecked(pcid >> 2) >> ((pcid & 3) << 1)) & 3 > 0 { residual.swap_remove(rid); continue; }
                        let pcs = *co.get_unchecked(pcid) as usize; let pce = *co.get_unchecked(pcid + 1) as usize;
                        if pcs == pce { continue; }
                        let lit = *cl.get_unchecked(pcs + rng.gen::<usize>() % (pce - pcs));
                        let v = (lit.unsigned_abs() as usize) - 1;
                        let was = *vars.get_unchecked(v);
                        let (is, ie) = if was { (*n_off.get_unchecked(v), *n_off.get_unchecked(v+1)) } else { (*p_off.get_unchecked(v), *p_off.get_unchecked(v+1)) };
                        let ia = if was { &n_data } else { &p_data };
                        let (ds, de) = if was { (*p_off.get_unchecked(v), *p_off.get_unchecked(v+1)) } else { (*n_off.get_unchecked(v), *n_off.get_unchecked(v+1)) };
                        let da = if was { &p_data } else { &n_data };
                        for k in is..ie { let c = *ia.get_unchecked(k as usize) as usize; *num_good.get_unchecked_mut(c >> 2) += 1u8 << ((c & 3) << 1); }
                        for k in ds..de { let c = *da.get_unchecked(k as usize) as usize; let sh = (c & 3) << 1; let bi = c >> 2; let before = (*num_good.get_unchecked(bi) >> sh) & 3; *num_good.get_unchecked_mut(bi) -= 1u8 << sh; if before == 1 { residual.push(c as u32); } }
                        *vars.get_unchecked_mut(v) = !was; *var_age.get_unchecked_mut(v) = 0;
                    }
                    stagnation = 0;
                }
            } else if pr > 0.15 + 0.05 * (density / 3.0).min(1.0) { stagnation = 0; current_prob = base_prob; }
            else { stagnation = 0; current_prob = current_prob * 0.8 + base_prob * 0.2; }
            last_cr = residual.len();
        }

        let rv = rng.gen::<usize>();
        let mut cid = 0usize; let mut found = false;
        while !residual.is_empty() { let id = rv % residual.len(); let cand = *residual.get_unchecked(id) as usize; if (*num_good.get_unchecked(cand >> 2) >> ((cand & 3) << 1)) & 3 > 0 { residual.swap_remove(id); } else { cid = cand; found = true; break; } }
        if !found { break; }

        let cs = *co.get_unchecked(cid) as usize; let ce = *co.get_unchecked(cid + 1) as usize; let clen = ce - cs;
        if clen > 1 { cl.swap(cs, cs + rv % clen); }

        let mut zero_found = None;
        'outer: for j in cs..ce {
            let l = *cl.get_unchecked(j); let al = (l.unsigned_abs() as usize) - 1;
            let (os, oe) = if l > 0 { (*n_off.get_unchecked(al), *n_off.get_unchecked(al+1)) } else { (*p_off.get_unchecked(al), *p_off.get_unchecked(al+1)) };
            let arr = if l > 0 { &n_data } else { &p_data };
            for k in os..oe { let c = *arr.get_unchecked(k as usize) as usize; if (*num_good.get_unchecked(c >> 2) >> ((c & 3) << 1)) & 3 == 1 { continue 'outer; } }
            zero_found = Some(al); break;
        }

        let v_idx = if let Some(v) = zero_found { v }
        else if rng.gen::<f64>() < current_prob { (cl.get_unchecked(cs).unsigned_abs() as usize) - 1 }
        else {
            let mut ms = usize::MAX; let mut vm = (cl.get_unchecked(cs).unsigned_abs() as usize) - 1; let mut mw = usize::MAX;
            for j in cs..ce {
                let l = *cl.get_unchecked(j); let al = (l.unsigned_abs() as usize) - 1;
                let (os, oe) = if l > 0 { (*n_off.get_unchecked(al), *n_off.get_unchecked(al+1)) } else { (*p_off.get_unchecked(al), *p_off.get_unchecked(al+1)) };
                let arr = if l > 0 { &n_data } else { &p_data };
                let mut sad = 0usize;
                for k in os..oe { let c = *arr.get_unchecked(k as usize) as usize; if (*num_good.get_unchecked(c >> 2) >> ((c & 3) << 1)) & 3 == 1 { sad += 1; } if sad >= ms { break; } }
                if sad == 0 { let app = *appearances.get_unchecked(al) as usize; let ab = (*var_age.get_unchecked(al) as usize) / 4; let aw = app.saturating_sub(ab); if ms > 0 || aw < mw { ms = 0; mw = aw; vm = al; } }
                else if ms > 0 { let app = *appearances.get_unchecked(al) as usize; let ab = (*var_age.get_unchecked(al) as usize) / 2; let cw = sad * sad * 256 + app - ab.min(50); if cw < mw { ms = sad; mw = cw; vm = al; } if ms <= 1 { break; } }
            }
            vm
        };

        let was = *vars.get_unchecked(v_idx);
        let (is, ie, ia) = if was { (*n_off.get_unchecked(v_idx), *n_off.get_unchecked(v_idx+1), &n_data) } else { (*p_off.get_unchecked(v_idx), *p_off.get_unchecked(v_idx+1), &p_data) };
        let (ds, de, da) = if was { (*p_off.get_unchecked(v_idx), *p_off.get_unchecked(v_idx+1), &p_data) } else { (*n_off.get_unchecked(v_idx), *n_off.get_unchecked(v_idx+1), &n_data) };
        for k in is..ie { let c = *ia.get_unchecked(k as usize) as usize; *num_good.get_unchecked_mut(c >> 2) += 1u8 << ((c & 3) << 1); }
        for k in ds..de { let c = *da.get_unchecked(k as usize) as usize; let sh = (c & 3) << 1; let bi = c >> 2; let before = (*num_good.get_unchecked(bi) >> sh) & 3; *num_good.get_unchecked_mut(bi) -= 1u8 << sh; if before == 1 { residual.push(c as u32); } }
        *vars.get_unchecked_mut(v_idx) = !was; *var_age.get_unchecked_mut(v_idx) = 0;
        for j in cs..ce { let l = *cl.get_unchecked(j); let var = (l.unsigned_abs() as usize) - 1; let age = var_age.get_unchecked_mut(var); *age = age.saturating_add(1); }
        rounds += 1;
    }
    }

    let _ = save_solution(&Solution { variables: vars });
    Ok(())
}