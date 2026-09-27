// Adapted from a118 engine_iv by NVX, itself derived from sat_imp_v4. This v7 file is modified; historical verbatim/benchmark notes below describe the upstream origin.
// R38: preserve engine_iv search trajectory; use exact selection and packed literal order.
// engine_iv — PORT VERBATIM de sat_imp_v4 (algo 60, qualifier mainnet c001_a098),
// reduit au chemin t4 (nv<=5000, density>=4.25 => track1). PREUVE : bench 34816
// (v5496, t4 seed=local, hp={"max_fuel_high":400e9}) = {3,7,12,17,27,28} — le nonce 28
// est HORS de l'union de nos 585 benchs a map. Ce moteur est le run B du portfolio t4.
// Preprocessing + dispatch = mod.rs d'imp_v4 VERBATIM (memes lignes, seules les branches
// non-t4 sont retirees) ; track1 = module inline BYTE-IDENTIQUE a track1.rs.
use anyhow::Result;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tig_challenges::satisfiability::*;

#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {
    pub base_prob: Option<f64>,
    pub max_prob: Option<f64>,
    pub check_interval: Option<usize>,
    pub stagnation_limit: Option<usize>,
    pub perturbation_flips: Option<usize>,
    pub max_fuel_high: Option<f64>,
    pub max_fuel_low: Option<f64>,
}


pub fn solve(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let hp: Option<Hyperparameters> = hyperparameters.as_ref().and_then(|m| {
        serde_json::from_value(Value::Object(m.clone())).ok()
    });

    let nv = challenge.num_variables;
    let _ = save_solution(&Solution { variables: vec![false; nv] });
    let mut rng = SmallRng::seed_from_u64(u64::from_le_bytes(challenge.seed[..8].try_into().unwrap()));

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

    track1::solve(&hp, &mut rng, nv, nc, density, p_cnt, n_cnt, &all_off, &p_bound, &all_data, &mut cl, &co, save_solution)
}

mod track1 {
    use super::super::donor_kernel::exact_div;
    use super::super::track1_cached::explicit_roulette;
    use super::super::track1_cached::signed_order::{SignedOrder as ClauseOrder,LiteralRanges};
    use super::super::donor_kernel::subset_cache::{SmallBreakCache,Byte13BreakCache,CacheOps};
    use anyhow::Result;
    use rand::{rngs::SmallRng, Rng};
    use tig_challenges::satisfiability::*;
    use super::Hyperparameters;

    pub fn solve(
        hp: &Option<Hyperparameters>,
        rng: &mut SmallRng,
        nv: usize, nc: usize, density: f64,
        p_cnt: Vec<u32>, n_cnt: Vec<u32>,
        all_off: &[u32], p_bound: &[u32],
        all_data: &[u32],
        cl: &mut Vec<i32>, co: &[u32],
        save_solution: &dyn Fn(&Solution) -> Result<()>,
    ) -> Result<()> {
        let narrow = nv <= 8192 && p_cnt.iter().zip(&n_cnt).all(|(&p,&n)| p+n <= 255);
        if narrow {
            solve_impl::<Byte13BreakCache>(hp,rng,nv,nc,density,p_cnt,n_cnt,all_off,p_bound,all_data,cl,co,save_solution)
        } else {
            solve_impl::<SmallBreakCache>(hp,rng,nv,nc,density,p_cnt,n_cnt,all_off,p_bound,all_data,cl,co,save_solution)
        }
    }
    fn solve_impl<C:CacheOps>(
        hp: &Option<Hyperparameters>,
        rng: &mut SmallRng,
        nv: usize, nc: usize, density: f64,
        p_cnt: Vec<u32>, n_cnt: Vec<u32>,
        all_off: &[u32], p_bound: &[u32],
        all_data: &[u32],
        cl: &mut Vec<i32>, co: &[u32],
        save_solution: &dyn Fn(&Solution) -> Result<()>,
    ) -> Result<()> {
        let max_fuel = hp.as_ref().and_then(|h| h.max_fuel_high).unwrap_or(160_000_000_000.0);

        let avg_clause_size = cl.len() as f64 / nc as f64;
        let difficulty_factor = density * avg_clause_size.sqrt();
        let scale_factor = if nv > 25000 { 1.5 } else { 1.0 };
        let base_fuel = (2000.0 + 100.0 * difficulty_factor) * (nv as f64).sqrt() * scale_factor;
        let flip_fuel = (200.0 + difficulty_factor) / scale_factor;
        let remaining = (max_fuel - base_fuel).max(0.0);
        let max_flips = if flip_fuel > 0.0 { (remaining / flip_fuel) as usize } else { 0 };

        let nad = 1.0;
        let random_threshold = if nv >= 30000 { 0.01 } else { 0.003 };
        let mut vars = vec![false; nv];
        for v in 0..nv {
            let np = p_cnt[v] as usize;
            let nn = n_cnt[v] as usize;
            if nn == 0 && np > 0 { vars[v] = true; continue; }
            if np == 0 && nn > 0 { continue; }
            let vad = if nn > 0 { np as f64 / nn as f64 } else { nad + 1.0 };
            if vad <= nad {
                vars[v] = rng.gen_bool(random_threshold);
            } else {
                let prob = (np as f64 + 0.25) / ((np + nn) as f64 + 1.2);
                vars[v] = rng.gen_bool(prob);
            }
        }

        let mut unsat_list: Vec<u32> = Vec::with_capacity(nc);
        let mut unsat_pos = vec![u32::MAX; nc];

        for c in 0..nc {
            let s = co[c] as usize;
            let e = co[c + 1] as usize;
            let mut g = 0u8;
            for j in s..e {
                let l = cl[j];
                let v = (l.abs() - 1) as usize;
                if (l > 0 && vars[v]) || (l < 0 && !vars[v]) { g += 1; }
            }
            if g == 0 {
                unsat_pos[c] = unsat_list.len() as u32;
                unsat_list.push(c as u32);
            }
        }

        if unsat_list.is_empty() {
            let _ = save_solution(&Solution { variables: vars });
            return Ok(());
        }

        let large_problem_scale = ((nv as f64 - 25000.0) / 35000.0).max(0.0).min(1.0);
        let base_interval = 60.0 - 30.0 * large_problem_scale;
        let min_interval = if large_problem_scale > 0.0 { 15.0 } else { 25.0 };
        let density_factor_ci = if density > 4.0 { 1.2 } else { 1.0 };
        let check_interval = hp.as_ref().and_then(|h| h.check_interval)
            .unwrap_or((base_interval * density_factor_ci * (1.0 + (density / 3.0).ln().max(0.0))).max(min_interval) as usize);

        let mut last_check_residual = unsat_list.len();
        let mut rounds = 0usize;
        let mut stagnation = 0usize;
        let stagnation_limit_t4 = hp.as_ref().and_then(|h| h.stagnation_limit).unwrap_or(3);

        let probs_break: [u32; 16] = [2535, 551, 233, 127, 80, 55, 41, 30, 24, 19, 16, 13, 11, 9, 8, 7];

        let mut current_reinit_stagnation: usize = ((nv * nc / 1000) as usize).clamp(1000, 500000);
        const REINIT_MIN_UNSAT: usize = 10;
        const MAX_REINITS: usize = 50;

        let mut best_unsat = unsat_list.len();
        let mut best_vars = vars.clone();
        let mut stagnation_count: usize = 0;
        let mut reinit_count: usize = 0;

        // Scratch space covers every incidence of any flipped variable. The
        // cache reports changed clauses in the original occurrence-list order.
        let max_degree = all_off.windows(2).map(|w| (w[1]-w[0]) as usize).max().unwrap_or(0).max(1);
        let mut changed = Vec::<u32>::with_capacity(max_degree);
        let scratch = changed.as_mut_ptr();
        let mut order = ClauseOrder::new(cl,co);
        let ranges = LiteralRanges::new(all_off,p_bound);
        let mut next_check_interval = check_interval;
        let mut cache = C::new(nv, &cl, co, &vars);
        unsafe {
            // A best residual below ten is monotone from here: neither
            // restart nor max-restart termination can become active again.
            if best_unsat >= REINIT_MIN_UNSAT {
            loop {
                if rounds >= max_flips { break; }
                if unsat_list.is_empty() { break; }

                if reinit_count >= MAX_REINITS && stagnation_count >= current_reinit_stagnation && best_unsat >= REINIT_MIN_UNSAT {
                    break;
                }

                if stagnation_count >= current_reinit_stagnation && best_unsat >= REINIT_MIN_UNSAT && reinit_count < MAX_REINITS {
                    reinit_count += 1;
                    let reinit_factor = if density > 4.0 { 1.5 } else { 1.3 };
                    current_reinit_stagnation = ((current_reinit_stagnation as f64 * reinit_factor) as usize).clamp(1000, 500000);

                    for v in 0..nv { vars[v] = false; }
                    for v in 0..nv {
                        let np = p_cnt[v] as usize;
                        let nn = n_cnt[v] as usize;
                        if nn == 0 && np > 0 { vars[v] = true; continue; }
                        if np == 0 && nn > 0 { continue; }
                        let vad = if nn > 0 { np as f64 / nn as f64 } else { nad + 1.0 };
                        if vad <= nad {
                            vars[v] = rng.gen_bool(random_threshold);
                        } else {
                            let prob = (np as f64 + 0.25) / ((np + nn) as f64 + 1.2);
                            vars[v] = rng.gen_bool(prob);
                        }
                    }

                    unsat_list.clear();
                    unsat_pos.fill(u32::MAX);
                    for c in 0..nc {
                        let s = co[c] as usize;
                        let e = co[c + 1] as usize;
                        let mut g = 0u8;
                        for j in s..e {
                            let l = cl[j];
                            let v = (l.abs() - 1) as usize;
                            if (l > 0 && vars[v]) || (l < 0 && !vars[v]) { g += 1; }
                        }
                        if g == 0 {
                            unsat_pos[c] = unsat_list.len() as u32;
                            unsat_list.push(c as u32);
                        }
                    }

                    cache = C::new(nv, &cl, co, &vars);
                    best_unsat = unsat_list.len();
                    best_vars.copy_from_slice(&vars);
                    stagnation_count = 0;
                }

                if (rounds == next_check_interval || check_interval == 0) && rounds % check_interval == 0 && rounds > 0 {
                    next_check_interval = next_check_interval.wrapping_add(check_interval);
                    let progress = last_check_residual as i64 - unsat_list.len() as i64;

                    if progress <= 0 {
                        stagnation += 1;

                        if stagnation >= stagnation_limit_t4 {
                            let kicks = if stagnation >= 8 { 6 } else { 3 };
                            for _ in 0..kicks {
                                if unsat_list.is_empty() { break; }

                                let rid = exact_div::rem(rng.gen::<usize>(),unsat_list.len());
                                let pcid = *unsat_list.get_unchecked(rid) as usize;
                                let len = order.len(pcid);
                                if len == 0 { continue; }
                                let lit = order.lit(pcid,exact_div::clause_rem(rng.gen::<usize>(),len));
                                let v = (lit.abs()-1) as usize;
                                let was_true = lit < 0;
                                let (is,ie,ds,de) = ranges.of((v<<1)|(was_true as usize));

                                flip_cached(&mut cache,all_data,is,ie,ds,de,v,&mut unsat_list,&mut unsat_pos,scratch);
                                *vars.get_unchecked_mut(v) = !was_true;
                            }
                            stagnation = 0;
                        }
                    } else {
                        stagnation = 0;
                    }

                    last_check_residual = unsat_list.len();
                }

                if unsat_list.is_empty() { break; }

                let rand_val = rng.gen::<usize>();

                let cid = *unsat_list.get_unchecked(exact_div::rem32_nonzero_small(rand_val as u32,unsat_list.len())) as usize;
                let clen = order.len(cid);
                let chosen_code=if clen==3 {
                    super::super::track1_cached::shaped_choice::choose_three(&mut order,cid,&cache,rand_val)
                } else {
                if clen > 1 { order.swap(cid,0,exact_div::clause_rem(rand_val,clen)); }

                let ll=order.codes(cid);
                let vv=[ll[0]>>1,ll[1]>>1,ll[2]>>1];
                let b0=cache.breaks_of(vv[0]);
                let b1=if clen>1{cache.breaks_of(vv[1])}else{1};
                let b2=if clen>2{cache.breaks_of(vv[2])}else{1};
                let zero_mask=((b0==0)as usize)|(((b1==0)as usize)<<1)|(((b2==0)as usize)<<2);
                let chosen_code=if zero_mask!=0{ClauseOrder::choose_zero(ll,zero_mask,rand_val)}
                    else if clen==3{ll[explicit_roulette::integer_three(b0,b1,b2,rand_val as u32)]}
                    else{
                        let w0=*probs_break.get_unchecked(b0.min(15));
                        let w1=if clen>1{*probs_break.get_unchecked(b1.min(15))}else{0};
                        let total=w0+w1;
                        let r=(rand_val as u32)%total.max(1);
                        if r<w0{ll[0]}else if clen>1 && r-w0<w1{ll[1]}else{ll[0]}
                    };
                    chosen_code
                };
                let v_idx=chosen_code>>1;
                let was_true=chosen_code&1!=0;
                let (is,ie,ds,de)=ranges.of(chosen_code);

                flip_cached(&mut cache,all_data,is,ie,ds,de,v_idx,&mut unsat_list,&mut unsat_pos,scratch);
                *vars.get_unchecked_mut(v_idx) = !was_true;
                rounds += 1;

                let cur = unsat_list.len();
                if cur < best_unsat {
                    best_unsat = cur;
                    best_vars.copy_from_slice(&vars);
                    stagnation_count = 0;
                } else {
                    stagnation_count += 1;
                }
                if best_unsat < REINIT_MIN_UNSAT { break; }
            }
            }
            if best_unsat < REINIT_MIN_UNSAT {
            loop {
                if rounds >= max_flips || unsat_list.is_empty() { break; }
                if (rounds == next_check_interval || check_interval == 0) && rounds % check_interval == 0 && rounds > 0 {
                    next_check_interval = next_check_interval.wrapping_add(check_interval);
                    let progress = last_check_residual as i64 - unsat_list.len() as i64;

                    if progress <= 0 {
                        stagnation += 1;

                        if stagnation >= stagnation_limit_t4 {
                            let kicks = if stagnation >= 8 { 6 } else { 3 };
                            for _ in 0..kicks {
                                if unsat_list.is_empty() { break; }

                                let rid = exact_div::rem(rng.gen::<usize>(),unsat_list.len());
                                let pcid = *unsat_list.get_unchecked(rid) as usize;
                                let len = order.len(pcid);
                                if len == 0 { continue; }
                                let lit = order.lit(pcid,exact_div::clause_rem(rng.gen::<usize>(),len));
                                let v = (lit.abs()-1) as usize;
                                let was_true = lit < 0;
                                let (is,ie,ds,de) = ranges.of((v<<1)|(was_true as usize));

                                flip_cached(&mut cache,all_data,is,ie,ds,de,v,&mut unsat_list,&mut unsat_pos,scratch);
                                // Cache states retain the complete live assignment.
                            }
                            stagnation = 0;
                        }
                    } else {
                        stagnation = 0;
                    }

                    last_check_residual = unsat_list.len();
                }

                if unsat_list.is_empty() { break; }

                let rand_val = rng.gen::<usize>();

                let cid = *unsat_list.get_unchecked(exact_div::rem32_nonzero_small(rand_val as u32,unsat_list.len())) as usize;
                let clen = order.len(cid);
                let chosen_code=if clen==3 {
                    super::super::track1_cached::shaped_choice::choose_three(&mut order,cid,&cache,rand_val)
                } else {
                if clen > 1 { order.swap(cid,0,exact_div::clause_rem(rand_val,clen)); }

                let ll=order.codes(cid);
                let vv=[ll[0]>>1,ll[1]>>1,ll[2]>>1];
                let b0=cache.breaks_of(vv[0]);
                let b1=if clen>1{cache.breaks_of(vv[1])}else{1};
                let b2=if clen>2{cache.breaks_of(vv[2])}else{1};
                let zero_mask=((b0==0)as usize)|(((b1==0)as usize)<<1)|(((b2==0)as usize)<<2);
                let chosen_code=if zero_mask!=0{ClauseOrder::choose_zero(ll,zero_mask,rand_val)}
                    else if clen==3{ll[explicit_roulette::integer_three(b0,b1,b2,rand_val as u32)]}
                    else{
                        let w0=*probs_break.get_unchecked(b0.min(15));
                        let w1=if clen>1{*probs_break.get_unchecked(b1.min(15))}else{0};
                        let total=w0+w1;
                        let r=(rand_val as u32)%total.max(1);
                        if r<w0{ll[0]}else if clen>1 && r-w0<w1{ll[1]}else{ll[0]}
                    };
                    chosen_code
                };
                let v_idx=chosen_code>>1;
                let was_true=chosen_code&1!=0;
                let (is,ie,ds,de)=ranges.of(chosen_code);

                flip_cached(&mut cache,all_data,is,ie,ds,de,v_idx,&mut unsat_list,&mut unsat_pos,scratch);

                rounds += 1;

                let cur = unsat_list.len();
                if cur < best_unsat {
                    best_unsat = cur;
                    cache.restore_variables(cl,co,&mut vars);
                    best_vars.copy_from_slice(&vars);
                }
            }
            }

        }

        if unsat_list.is_empty() {cache.restore_variables(cl,co,&mut vars);}
        let final_vars = if unsat_list.is_empty() { vars } else { best_vars };
        let _ = save_solution(&Solution { variables: final_vars });

        Ok(())
    }

    #[inline(always)]
    unsafe fn flip_cached<C:CacheOps>(
        cache:&mut C, all_data:&[u32], is:usize,ie:usize,ds:usize,de:usize,
        variable:usize, unsat_list:&mut Vec<u32>,unsat_pos:&mut [u32],scratch:*mut u32,
    ) {
        let gains=cache.gain_collect(all_data.get_unchecked(is..ie),variable,scratch);
        for i in 0..gains {
            let c=*scratch.add(i) as usize;
            let pos=*unsat_pos.get_unchecked(c) as usize;
            let last_index=unsat_list.len()-1;
            let last_clause=*unsat_list.get_unchecked(last_index) as usize;
            *unsat_list.get_unchecked_mut(pos)=last_clause as u32;
            *unsat_pos.get_unchecked_mut(last_clause)=pos as u32;
            // Removed clauses have no readable position until the next insertion.
            unsat_list.set_len(last_index);
        }
        let losses=cache.lose_many(all_data.get_unchecked(ds..de),variable,scratch);
        let len=unsat_list.len();
        // At most one live entry per clause: len+losses <= nc <= capacity.
        for i in 0..losses {
            let c=*scratch.add(i);
            *unsat_pos.get_unchecked_mut(c as usize)=(len+i) as u32;
            unsat_list.as_mut_ptr().add(len+i).write(c);
        }
        unsat_list.set_len(len+losses);
        cache.set_break(variable,gains as u32);
    }
}
