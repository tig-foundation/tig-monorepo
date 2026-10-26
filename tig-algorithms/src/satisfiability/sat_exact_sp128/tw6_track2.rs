// vars is a deferred snapshot while an attempt searches. Gain/loss
// orientation, not vars, drives flips; restore only before original observations.
// In COMPACT mode all appearance/age penalties lie in [-50,255].
// After excluding zero counts and consuming the ORIGINAL noise word, the
// first break-one variable strictly beats every candidate whose count>=2.
// The original greedy loop stops on minimum one, so later ties are not read.
// Raw degrees>255 retain the original full arithmetic branch unchanged.
// Integer equivalence: x<best && x<=12 iff x<min(best,13).
// best changes only on strict baseline record events; cache this derived limit
// on every such write, including initial state, periodic checks and attempt end.
// Copies and best_unsat values stay at exactly the same events.
use super::super::phase_div;
use super::super::exact_coin::ExactCoin;
// Retained-clause degree<=nc<=32002 on this track, so u16 appearances
// preserve every value. Age is only observed through min(age/2,50); u8 and
// u16 saturation are indistinguishable after age 100 and reset identically.
use super::super::paged_flip_ranges::FlipRanges;
// Best snapshots are copied only on the baseline's strict-improvement events.
// best_unsat strictly decreases globally, so the total number of copies is
// bounded by its initial value. No per-flip history is necessary: the saved
// snapshot is not read by move selection within an attempt.
use super::super::paged_small_order::ClauseOrder;
use super::super::exact_div;
use super::super::paged_cache::{SmallBreakCache,Byte13BreakCache,CacheOps};
use anyhow::Result;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use tig_challenges::satisfiability::*;
use super::Hyperparameters;

#[inline(always)]
unsafe fn apply_flip<C:CacheOps>(
    v_idx: usize,
    all_data:&[u32],
    ranges:&mut FlipRanges,
    cache: &mut C,
    residual: &mut Vec<u32>,
    unsat_count: &mut usize,
    var_age: &mut [u8],
) {
    
    let (is,ie,ds,de)=ranges.flip(v_idx);
    let ia=all_data;let da=all_data;

    let gains=cache.gain_many(ia.get_unchecked(is..ie),v_idx);
    residual.reserve(de-ds);
    let old_len=residual.len();
    let appended=cache.lose_many(da.get_unchecked(ds..de),v_idx,residual.as_mut_ptr().add(old_len));
    residual.set_len(old_len+appended);
    cache.set_break(v_idx,gains);
    *unsat_count=*unsat_count+appended-gains as usize;

    *var_age.get_unchecked_mut(v_idx) = 0;

}

/// Solves the SAT problem using SLS with Adaptive Phase Saving on Restarts.
/// Variables with higher appearance counts or polarity skew have a higher
/// probability of maintaining their phase (`best_vars`) during restarts.
pub fn solve(
    challenge: &Challenge,
    hp: &Option<Hyperparameters>,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Result<()> {
    let nv=challenge.num_variables;
    let mut degree=vec![0u32;nv];
    for orig in &challenge.clauses {for &l in orig {degree[(l.abs()-1)as usize]+=1;}}
    if nv<=8192 && degree.iter().all(|&d|d<=255) {
        solve_impl::<Byte13BreakCache,true>(challenge,hp,save_solution)
    } else {solve_impl::<SmallBreakCache,false>(challenge,hp,save_solution)}
}
fn solve_impl<C:CacheOps,const COMPACT:bool>(
    challenge: &Challenge,
    hp: &Option<Hyperparameters>,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Result<()> {
    let nv = challenge.num_variables;
    let _ = save_solution(&Solution { variables: vec![false; nv] });
    let mut rng = SmallRng::seed_from_u64(u64::from_le_bytes(challenge.seed[..8].try_into().unwrap()));

    let mut p_cnt = vec![0u32; nv];
    let mut n_cnt = vec![0u32; nv];
    let mut compact_clauses = Vec::with_capacity(challenge.clauses.len());
    let mut compact_len = Vec::with_capacity(challenge.clauses.len());
    let mut total_lits = 0usize;

    for orig in &challenge.clauses {
        let (a, b, c) = (orig[0], orig[1], orig[2]);
        if a == -b || a == -c || b == -c { continue; }

        let mut lits = [0i32; 3];
        let mut len = 1usize;
        lits[0] = a;

        let va = (a.abs() - 1) as usize;
        if a > 0 { p_cnt[va] += 1; } else { n_cnt[va] += 1; }

        if b != a {
            lits[len] = b;
            len += 1;
            let vb = (b.abs() - 1) as usize;
            if b > 0 { p_cnt[vb] += 1; } else { n_cnt[vb] += 1; }
        }
        if c != a && c != b {
            lits[len] = c;
            len += 1;
            let vc = (c.abs() - 1) as usize;
            if c > 0 { p_cnt[vc] += 1; } else { n_cnt[vc] += 1; }
        }

        total_lits += len;
        compact_clauses.push(lits);
        compact_len.push(len as u8);
    }

    let nc = compact_clauses.len();

    let mut p_off = vec![0u32; nv + 1];
    let mut n_off = vec![0u32; nv + 1];
    for v in 0..nv {
        p_off[v + 1] = p_off[v] + p_cnt[v];
        n_off[v + 1] = n_off[v] + n_cnt[v];
    }
    let mut p_data = vec![0u32; p_off[nv] as usize];
    let mut n_data = vec![0u32; n_off[nv] as usize];

    let mut p_pos = p_off[..nv].to_vec();
    let mut n_pos = n_off[..nv].to_vec();
    let mut cl = Vec::with_capacity(total_lits);
    let mut co = Vec::with_capacity(nc + 1);
    co.push(0u32);
    for (ci, lits) in compact_clauses.iter().enumerate() {
        let ci_u32 = ci as u32;
        let len = compact_len[ci] as usize;
        for j in 0..len {
            let lit = lits[j];
            let v = (lit.abs() - 1) as usize;
            if lit > 0 {
                p_data[p_pos[v] as usize] = ci_u32;
                p_pos[v] += 1;
            } else {
                n_data[n_pos[v] as usize] = ci_u32;
                n_pos[v] += 1;
            }
        }
        cl.extend_from_slice(&lits[..len]);
        co.push(cl.len() as u32);
    }

    let density = nc as f64 / nv as f64;
    let max_fuel = hp.as_ref().and_then(|h| h.max_fuel_high).unwrap_or(250_000_000_000.0);

    let var_appearances: Vec<u16> = (0..nv)
        .map(|v| (p_cnt[v] + n_cnt[v]) as u16)
        .collect();

    let max_app = *var_appearances.iter().max().unwrap_or(&1) as f64;
    let base_keep_prob = if nv <= 20000 { 0.15 } else { 0.09 };
    
    let phase_save_threshold: Vec<u32> = (0..nv).map(|v| {
        let app = var_appearances[v] as f64;
        let np = p_cnt[v] as f64;
        let nn = n_cnt[v] as f64;
        let skew = if np + nn > 0.0 { (np - nn).abs() / (np + nn) } else { 0.0 };
        let prob = (base_keep_prob + 0.35 * (app / max_app) + 0.25 * skew).clamp(0.0, 0.90);
        (prob * 4294967295.0) as u32
    }).collect();

    let avg_clause_size = cl.len() as f64 / nc as f64;
    let difficulty_factor = density * avg_clause_size.sqrt();
    let scale_factor = if nv > 25000 { 1.5 } else { 1.0 };
    let base_fuel = (2000.0 + 100.0 * difficulty_factor) * (nv as f64).sqrt() * scale_factor;
    let flip_fuel = (200.0 + difficulty_factor) / scale_factor;
    let remaining = (max_fuel - base_fuel).max(0.0);
    let max_flips = if flip_fuel > 0.0 { (remaining / flip_fuel) as usize } else { 0 };

    let nad = 1.0;
    let random_threshold = if nv >= 30000 { 0.01 } else { 0.003 };

    let base_prob: f64 = 0.52;
    let max_random_prob: f64 = 0.9;
    let smoothing_factor: f64 = 0.8;

    let large_problem_scale = ((nv as f64 - 25000.0) / 35000.0).max(0.0).min(1.0);
    let base_interval = 60.0 - 30.0 * large_problem_scale;
    let min_interval = if large_problem_scale > 0.0 { 15.0 } else { 25.0 };
    let density_factor_ci = if density > 4.0 { 1.2 } else { 1.0 };
    let check_interval = (base_interval * density_factor_ci * (1.0 + (density / 3.0).ln().max(0.0))).max(min_interval) as usize;
    let variance_interval = 1000usize;

    let raw_restarts = if nv <= 12000 { 4usize } else if nv <= 40000 { 3usize } else { 2usize };
    let restart_attempts = raw_restarts
        .saturating_sub(if density > 5.0 { 1 } else { 0 })
        .max(1)
        .min(max_flips.max(1));

    let mut best_unsat = nc + 1;
    let mut record_limit=best_unsat.min(13);
    let mut best_vars = vec![false; nv];
    let mut vars = vec![false; nv];
    let mut num_good = vec![0u8; nc];
    let mut residual: Vec<u32> = Vec::with_capacity(nc);
    let mut var_age = vec![0u8; nv];

    let mut order = ClauseOrder::new(&cl, &co);
    let mut all_off=Vec::with_capacity(nv+1);let mut p_bound=Vec::with_capacity(nv);
    let mut all_data=Vec::with_capacity(p_data.len()+n_data.len());all_off.push(0u32);
    for v in 0..nv {
        all_data.extend_from_slice(&p_data[p_off[v]as usize..p_off[v+1]as usize]);
        p_bound.push(all_data.len()as u32);
        all_data.extend_from_slice(&n_data[n_off[v]as usize..n_off[v+1]as usize]);
        all_off.push(all_data.len()as u32);
    }
    let all_data=super::super::large_buffer::from_slice(&all_data);
    for attempt in 0..restart_attempts {
        let attempt_budget = if restart_attempts == 1 {
            max_flips
        } else {
            max_flips / restart_attempts
                + if attempt < (max_flips % restart_attempts) { 1 } else { 0 }
        };
        let attempt_random_threshold = if attempt == 0 {
            random_threshold
        } else {
            (random_threshold * (1.0 + 0.45 * attempt as f64)).min(0.08)
        };
        let attempt_relax = if attempt == 0 {
            0.0
        } else {
            (0.06 * attempt as f64).min(0.18)
        };

        vars.fill(false);
        for v in 0..nv {
            let np = p_cnt[v] as usize;
            let nn = n_cnt[v] as usize;
            if nn == 0 && np > 0 { vars[v] = true; continue; }
            if np == 0 && nn > 0 { continue; }
            let vad = if nn > 0 { np as f64 / nn as f64 } else { nad + 1.0 };
            if vad <= nad {
                vars[v] = rng.gen_bool(attempt_random_threshold);
            } else {
                let bias_prob = (np as f64 + 0.25) / ((np + nn) as f64 + 1.2);
                let prob = (bias_prob * (1.0 - attempt_relax) + 0.5 * attempt_relax).clamp(0.0, 1.0);
                vars[v] = rng.gen_bool(prob);
            }
        }

        if attempt > 0 && best_unsat < nc {
            for v in 0..nv {
                if p_cnt[v] > 0 && n_cnt[v] > 0 && rng.gen::<u32>() < phase_save_threshold[v] {
                    vars[v] = best_vars[v];
                }
            }
        }


        num_good.fill(0);
        residual.clear();
        for i in 0..nc {
            let s = co[i] as usize;
            let e = co[i + 1] as usize;
            let mut ng = 0u8;
            for j in s..e {
                let l = cl[j];
                let v = (l.abs() - 1) as usize;
                if (l > 0 && vars[v]) || (l < 0 && !vars[v]) {
                    ng += 1;
                }
            }
            num_good[i] = ng;
            if ng == 0 {
                residual.push(i as u32);
            }
        }
        let mut unsat_count = residual.len();

        if unsat_count < best_unsat {
            best_unsat = unsat_count;
                        record_limit=best_unsat.min(13);
            best_vars.copy_from_slice(&vars);
        }
        if unsat_count == 0 {
            let _ = save_solution(&Solution { variables: vars });
            return Ok(());
        }

        let mut current_prob = base_prob;
        var_age.fill(0);
        let mut rounds = 0usize;
        let mut stagnation = 0usize;
        let mut max_unsat_window = unsat_count;
        let mut min_unsat_window = unsat_count;

        let mut coin=ExactCoin::new(current_prob);
        let mut cache = C::new(nv, &cl, &co, &vars);
        let mut ranges=FlipRanges::new(&all_off,&p_bound,&vars);
        let mut next_check_interval = check_interval;
        let mut next_variance_interval = variance_interval;
        unsafe {
            // The control events occur before the flip at their exact
            // original round. Capacity-only boundaries do not trigger events.
            if rounds<attempt_budget && unsat_count>0{
                if check_interval==0{let _=rounds%check_interval;}
                if variance_interval==0{let _=rounds%variance_interval;}
            }
            let max_loss_per_flip=(0..nv).map(|v|(p_bound[v]-all_off[v]).max(all_off[v+1]-p_bound[v])as usize).max().unwrap_or(0);
            let capacity_batch_limit=(4096usize/max_loss_per_flip.max(1)).max(1).min(64);
            'attempt_search: loop{
                let event_end=next_check_interval.min(next_variance_interval).min(attempt_budget);
                let steps=(event_end-rounds).min(capacity_batch_limit);
                residual.reserve(steps*max_loss_per_flip);
                let batch_end=rounds+steps;
                // Events/kicks are outside this bounded main phase.
                // At most steps*max_loss_per_flip new IDs can be appended.
                let small=residual.len().checked_add(steps*max_loss_per_flip).map_or(false,|n|n<phase_div::LIMIT);
                if small{
                while (rounds<batch_end) & (unsat_count!=0){
                if unsat_count > max_unsat_window { max_unsat_window = unsat_count; }
                if unsat_count < min_unsat_window { min_unsat_window = unsat_count; }

                let rand_val = rng.gen::<usize>();
                // Positive exact unsat_count at main-loop entry proves
                // at least one live residual ID. Removing only satisfied IDs
                // cannot empty the list before a valid clause is selected.
                let cid=loop{
                    let id=phase_div::rem_nonzero_small(rand_val,residual.len());
                    let c=*residual.get_unchecked(id)as usize;
                    if cache.good(c)>0{
                        let last=residual.len()-1;let ptr=residual.as_mut_ptr();ptr.add(id).write(*ptr.add(last));residual.set_len(last);
                    }else{break c;}
                };


                let cs = 0usize;
                let ce = order.len(cid);
                let clen = ce - cs;

                if clen>1{order.swap(cid,0,exact_div::clause_rem(rand_val,clen));}



            let vv=order.variables(cid);
            let v_idx=if clen==3{choose_shape::<C,COMPACT,3>(vv,clen,&cache,&coin,&mut rng,&var_appearances,&var_age)}else{choose_short::<C,COMPACT>(vv,clen,&cache,&coin,&mut rng,&var_appearances,&var_age)};
                apply_flip_main(
                    v_idx,
                    &all_data,&mut ranges,

                    &mut cache,
                    &mut residual,
                    &mut unsat_count,
                    &mut var_age,
                    
                );
                *var_age.get_unchecked_mut(v_idx)=1;
                if clen==3{
                    let a=if vv[0]==v_idx{vv[1]}else{vv[0]};let b=if vv[2]==v_idx{vv[1]}else{vv[2]};
                    let age=var_age.get_unchecked_mut(a);*age=age.saturating_add(1);
                    let age=var_age.get_unchecked_mut(b);*age=age.saturating_add(1);
                }else if clen==2{
                    let a=if vv[0]==v_idx{vv[1]}else{vv[0]};let age=var_age.get_unchecked_mut(a);*age=age.saturating_add(1);
                }


                rounds += 1;
                if unsat_count < record_limit {
                    best_unsat = unsat_count;
                        record_limit=best_unsat.min(13);
                    ranges.restore_variables(&all_off,&p_bound,&mut vars);
                    best_vars.copy_from_slice(&vars);
                }

                }
                }else{
                while (rounds<batch_end) & (unsat_count!=0){
                if unsat_count > max_unsat_window { max_unsat_window = unsat_count; }
                if unsat_count < min_unsat_window { min_unsat_window = unsat_count; }

                let rand_val = rng.gen::<usize>();
                // Positive exact unsat_count at main-loop entry proves
                // at least one live residual ID. Removing only satisfied IDs
                // cannot empty the list before a valid clause is selected.
                let cid=loop{
                    let id=exact_div::rem(rand_val,residual.len());
                    let c=*residual.get_unchecked(id)as usize;
                    if cache.good(c)>0{
                        let last=residual.len()-1;let ptr=residual.as_mut_ptr();ptr.add(id).write(*ptr.add(last));residual.set_len(last);
                    }else{break c;}
                };


                let cs = 0usize;
                let ce = order.len(cid);
                let clen = ce - cs;

                if clen>1{order.swap(cid,0,exact_div::clause_rem(rand_val,clen));}



            let vv=order.variables(cid);
            let v_idx=if clen==3{choose_shape::<C,COMPACT,3>(vv,clen,&cache,&coin,&mut rng,&var_appearances,&var_age)}else{choose_short::<C,COMPACT>(vv,clen,&cache,&coin,&mut rng,&var_appearances,&var_age)};
                apply_flip_main(
                    v_idx,
                    &all_data,&mut ranges,

                    &mut cache,
                    &mut residual,
                    &mut unsat_count,
                    &mut var_age,
                    
                );
                *var_age.get_unchecked_mut(v_idx)=1;
                if clen==3{
                    let a=if vv[0]==v_idx{vv[1]}else{vv[0]};let b=if vv[2]==v_idx{vv[1]}else{vv[2]};
                    let age=var_age.get_unchecked_mut(a);*age=age.saturating_add(1);
                    let age=var_age.get_unchecked_mut(b);*age=age.saturating_add(1);
                }else if clen==2{
                    let a=if vv[0]==v_idx{vv[1]}else{vv[0]};let age=var_age.get_unchecked_mut(a);*age=age.saturating_add(1);
                }


                rounds += 1;
                if unsat_count < record_limit {
                    best_unsat = unsat_count;
                        record_limit=best_unsat.min(13);
                    ranges.restore_variables(&all_off,&p_bound,&mut vars);
                    best_vars.copy_from_slice(&vars);
                }

                }
                }
                if unsat_count==0{break 'attempt_search;}
                if rounds<event_end{continue;}
                if rounds>=attempt_budget||unsat_count==0{break;}
                if unsat_count > max_unsat_window { max_unsat_window = unsat_count; }
                if unsat_count < min_unsat_window { min_unsat_window = unsat_count; }


                if rounds==next_check_interval{
                    next_check_interval=next_check_interval.checked_add(check_interval).unwrap_or(usize::MAX);


                    if unsat_count < best_unsat {
                        best_unsat = unsat_count;
                        record_limit=best_unsat.min(13);
                        ranges.restore_variables(&all_off,&p_bound,&mut vars);
                    best_vars.copy_from_slice(&vars);
                    }
                
                }
                if rounds==next_variance_interval{
                    next_variance_interval=next_variance_interval.checked_add(variance_interval).unwrap_or(usize::MAX);


                    let variance = max_unsat_window.saturating_sub(min_unsat_window);

                    if variance <= 2 {
                        stagnation += 1;
                        current_prob = (current_prob + 0.15).min(max_random_prob);
                    } else if variance <= 6 {
                        stagnation += 1;
                        current_prob = (current_prob + 0.05).min(max_random_prob);
                    } else if variance >= 20 {
                        stagnation = 0;
                        current_prob = base_prob;
                    } else {
                        stagnation = 0;
                        current_prob = current_prob * smoothing_factor + base_prob * (1.0 - smoothing_factor);
                    }

                    if stagnation >= 3 {
                        let kicks = if stagnation >= 6 { 8 } else { 4 };
                        for _ in 0..kicks {
                            if residual.is_empty() || unsat_count == 0 { break; }
                            let rid = exact_div::rem(rng.gen::<usize>(), residual.len());
                            let pcid = *residual.get_unchecked(rid) as usize;
                            if cache.good(pcid) > 0 {
                                residual.swap_remove(rid);
                                continue;
                            }
                            let pcs = 0usize;
                            let pce = order.len(pcid);
                            if pcs == pce { continue; }
                            let lit = order.lit(pcid, pcs + exact_div::clause_rem(rng.gen::<usize>(), pce - pcs));
                            let v = (lit.abs() - 1) as usize;

                            apply_flip(
                                v,
                                &all_data,&mut ranges,

                                &mut cache,
                                &mut residual,
                                &mut unsat_count,
                                &mut var_age,
                                
                            );
                        }
                        stagnation = 0;
                    }

                    coin=ExactCoin::new(current_prob);
                    max_unsat_window = unsat_count;
                    min_unsat_window = unsat_count;
                
                }
                // The original post-kick terminal cleanup consumes this one
                // word but makes no further flip or RNG draw when already solved.
                if unsat_count==0{let _:usize=rng.gen();break;}
            }
        }

        if unsat_count < best_unsat {
            best_unsat = unsat_count;
                        record_limit=best_unsat.min(13);
            ranges.restore_variables(&all_off,&p_bound,&mut vars);
                    best_vars.copy_from_slice(&vars);
        }
        if unsat_count == 0 {
            ranges.restore_variables(&all_off,&p_bound,&mut vars);
            let _ = save_solution(&Solution { variables: vars });
            return Ok(());
        }

    }

    let _ = save_solution(&Solution { variables: best_vars });
    Ok(())
}

#[inline(always)]
unsafe fn apply_flip_main<C:CacheOps>(
    v_idx: usize,
    all_data:&[u32],
    ranges:&mut FlipRanges,
    cache: &mut C,
    residual: &mut Vec<u32>,
    unsat_count: &mut usize,
    var_age: &mut [u8],
) {
    
    let (is,ie,ds,de)=ranges.flip(v_idx);
    let ia=all_data;let da=all_data;

    let gains=cache.gain_many(ia.get_unchecked(is..ie),v_idx);
    // Main-flip capacity is proven once per bounded phase batch.
    let old_len=residual.len();
    let appended=cache.lose_many(da.get_unchecked(ds..de),v_idx,residual.as_mut_ptr().add(old_len));
    residual.set_len(old_len+appended);
    cache.set_break(v_idx,gains);
    *unsat_count=*unsat_count+appended-gains as usize;

    // Main path commits the effective post-increment age of one.

}

#[inline(always)]
unsafe fn choose_shape<C:CacheOps,const COMPACT:bool,const LENGTH:usize>(vv:[usize;3],length:usize,cache:&C,coin:&ExactCoin,rng:&mut SmallRng,var_appearances:&[u16],var_age:&[u8])->usize{
 let clen=if LENGTH==0{length}else{LENGTH};
            let b0=cache.breaks_of(vv[0]);
            let b1=if clen>1{cache.breaks_of(vv[1])}else{1};
            let b2=if clen>2{cache.breaks_of(vv[2])}else{1};
            let zero_mask=((b0==0)as usize)|(((b1==0)as usize)<<1)|(((b2==0)as usize)<<2);
            let chosen_result=if zero_mask!=0 {
                if zero_mask&1!=0{vv[0]}else if zero_mask&2!=0{vv[1]}else{vv[2]}
            }else if b0==1{let _:u64=rng.gen();vv[0]}else if coin.draw(rng){vv[0]}
            else if COMPACT && clen>1 && b1==1{vv[1]}
            else if COMPACT && clen>2 && b2==1{vv[2]}else{
                // COMPACT is selected only when every raw degree<=255. Then
                // 1000*(one count) dominates the full [-50,255] penalty range.
                if COMPACT {
                let mut selected=vv[0];
                let mut min_sad=b0;
                let mut penalty=(*var_appearances.get_unchecked(vv[0])as i32)-(((*var_age.get_unchecked(vv[0])as usize)/2).min(50)as i32);
                if clen>1 && min_sad>1 {
                    let p=(*var_appearances.get_unchecked(vv[1])as i32)-(((*var_age.get_unchecked(vv[1])as usize)/2).min(50)as i32);
                    if b1<min_sad || p<penalty {selected=vv[1];min_sad=b1.min(min_sad);penalty=p;}
                }
                if clen>2 && min_sad>1 {
                    let p=(*var_appearances.get_unchecked(vv[2])as i32)-(((*var_age.get_unchecked(vv[2])as usize)/2).min(50)as i32);
                    if b2<min_sad || p<penalty {selected=vv[2];min_sad=b2.min(min_sad);penalty=p;}
                }
                selected
                }else{
let mut selected=vv[0];
                let mut min_sad=b0;
                let mut min_weight=(b0)*1000+(*var_appearances.get_unchecked(vv[0]) as usize)-(((*var_age.get_unchecked(vv[0]) as usize)/2).min(50));
                if clen>1 && min_sad>1 {
                    let sad=b1.min(min_sad);
                    let weight=(sad)*1000+(*var_appearances.get_unchecked(vv[1]) as usize)-(((*var_age.get_unchecked(vv[1]) as usize)/2).min(50));
                    if weight<min_weight {selected=vv[1];min_sad=sad;min_weight=weight;}
                }
                if clen>2 && min_sad>1 {
                    let sad=b2.min(min_sad);
                    let weight=(sad)*1000+(*var_appearances.get_unchecked(vv[2]) as usize)-(((*var_age.get_unchecked(vv[2]) as usize)/2).min(50));
                    if weight<min_weight {selected=vv[2];min_sad=sad;min_weight=weight;}
                }
                selected
                }
            };


 chosen_result
}
#[cold]#[inline(never)]
unsafe fn choose_short<C:CacheOps,const COMPACT:bool>(vv:[usize;3],length:usize,cache:&C,coin:&ExactCoin,rng:&mut SmallRng,var_appearances:&[u16],var_age:&[u8])->usize{
 choose_shape::<C,COMPACT,0>(vv,length,cache,coin,rng,var_appearances,var_age)
}

#[cfg(test)]mod one_tests{
 fn reference(b:[usize;3],p:[i32;3],len:usize)->usize{let mut chosen=0;let mut low=b[0];let mut w=1000*(low as i32)+p[0];for i in 1..len{if low<=1{break;}let n=b[i].min(low);let nw=1000*(n as i32)+p[i];if nw<w{chosen=i;low=n;w=nw;}}chosen}
 #[test]fn positive_score_one_shortcut_all_penalty_extrema(){for a in 1..8{for b in 1..8{for c in 1..8{for p in [-50,-1,0,1,127,255]{for q in [-50,0,255]{for r in [-50,0,255]{for len in 1..=3{let scores=[a,b,c];if let Some(i)=(0..len).find(|&i|scores[i]==1){assert_eq!(reference(scores,[p,q,r],len),i);}}}}}}}}
 }
}
