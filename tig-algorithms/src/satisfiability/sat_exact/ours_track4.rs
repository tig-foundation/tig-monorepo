use super::super::phase_div;
// A normal flip never shrinks the lazy residual list: it updates clause
// state and appends newly false clauses, leaving old entries in place. Checking
// non-emptiness once after each control phase therefore replaces the per-round
// check. During cleanup, removing the final stale entry exits on the same RNG
// word and assignment, with no heuristic/control update moved across that exit.
// Initial vars are retained only for absent-variable values and callbacks.
// During search, exact clause counts/XOR owners represent all incident values;
// selected false literal signs determine the same gain/loss orientation.
use super::super::paged_order::SignedOrder;
use super::super::compact_ranges::CompactRanges;
use super::super::exact_coin::ExactCoin;
use super::super::clause_order::ClauseOrder;
use super::super::exact_div;
use super::super::paged_cache::{ExactBreakCache, NarrowLargeBreakCache, CacheOps};
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
    let narrow = p_cnt.iter().zip(&n_cnt).all(|(&p,&n)| p+n <= 255);
    if narrow { solve_impl::<NarrowLargeBreakCache,true>(hp, rng, nv, nc, density, p_cnt, n_cnt, all_off, p_bound, all_data, cl, co, save_solution) }
    else { solve_impl::<ExactBreakCache,false>(hp, rng, nv, nc, density, p_cnt, n_cnt, all_off, p_bound, all_data, cl, co, save_solution) }
}

fn solve_impl<C:CacheOps,const PACKED:bool>(
    hp: &Option<Hyperparameters>,
    rng: &mut SmallRng,
    nv: usize, nc: usize, density: f64,
    p_cnt: Vec<u32>, n_cnt: Vec<u32>,
    all_off: &[u32], p_bound: &[u32],
    all_data: &[u32],
    cl: &mut Vec<i32>, co: &[u32],
    save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Result<()> {
    let page_data=super::super::large_buffer::from_slice(all_data);
    let all_data:&[u32]=&page_data;

    let max_loss_per_flip=p_cnt.iter().chain(n_cnt.iter()).copied().max().unwrap_or(0)as usize;
    let capacity_batch_limit=64usize.min((4096/max_loss_per_flip.max(1)).max(1));
    let nvf = nv as f64;
    let max_fuel = hp.as_ref().and_then(|h| h.max_fuel_low).unwrap_or(150_000_000_000.0);
    let avg_clause_size = cl.len() as f64 / nc as f64;
    let difficulty_factor = density * avg_clause_size.sqrt();
    let scale_factor = if nv > 25000 { 1.5 } else { 1.0 };
    let base_fuel = (2000.0 + 100.0 * difficulty_factor) * (nv as f64).sqrt() * scale_factor;
    let flip_fuel = (200.0 + difficulty_factor) / scale_factor;
    let remaining = (max_fuel - base_fuel).max(0.0);
    let max_flips = if flip_fuel > 0.0 { (remaining / flip_fuel) as usize } else { 0 };

    let mut vars = vec![false; nv];
    let nad = 1.0;
    let random_threshold = 0.003 + 0.007 / (1.0 + (-(nvf - 30000.0) / 8000.0).exp());
    let steep = 0.35 / (1.0 + (density - 4.18).max(0.0) * 12.0);
    for v in 0..nv {
        let np = p_cnt[v] as f64;
        let nn = n_cnt[v] as f64;
        if nn == 0.0 && np > 0.0 { vars[v] = true; continue; }
        if np == 0.0 { continue; }
        let vad = np / nn;
        let bias_prob = (np + 0.25) / (np + nn + 1.2);
        let s = 1.0 / (1.0 + (-(vad - nad) / steep).exp());
        let prob = (random_threshold * (1.0 - s) + bias_prob * s).max(0.0).min(1.0);
        vars[v] = rng.gen_bool(prob);
    }

    let appearances: Vec<u8> = (0..nv).map(|v| {
        ((p_cnt[v] + n_cnt[v]) as usize).min(255) as u8
    }).collect();
    drop(p_cnt);
    drop(n_cnt);

    let ng_len = nc; // one byte per clause; count is in 0..=3
    let mut num_good = vec![0u8; ng_len];

    for i in 0..nc {
        let s = co[i] as usize;
        let e = co[i + 1] as usize;
        let byte_idx = i;
        for j in s..e {
            let l = cl[j];
            let v = (l.abs() - 1) as usize;
            if (l > 0 && vars[v]) || (l < 0 && !vars[v]) {
                num_good[byte_idx] += 1u8;
            }
        }
    }

    let mut residual: Vec<u32> = Vec::with_capacity(nc);
    for i in 0..nc {
        if num_good[i] == 0 {
            residual.push(i as u32);
        }
    }

    if residual.is_empty() {
        let _ = save_solution(&Solution { variables: vars });
        return Ok(());
    }

    let base_prob = hp.as_ref().and_then(|h| h.base_prob)
        .unwrap_or(0.45 + 0.1 * (density / 5.0).min(1.0));
    let mut current_prob = base_prob;
    let mut coin = ExactCoin::new(current_prob);

    let large_problem_scale = ((nvf - 25000.0) / 35000.0).max(0.0).min(1.0);
    let base_interval = 60.0 - 30.0 * large_problem_scale;
    let min_interval = 25.0 - 10.0 * large_problem_scale;
    let density_s = 1.0 / (1.0 + (-(density - 4.0) / 0.5).exp());
    let density_factor = 1.0 + 0.2 * density_s;
    let check_interval = hp.as_ref().and_then(|h| h.check_interval)
        .unwrap_or((base_interval * density_factor * (1.0 + (density / 3.0).ln().max(0.0))).max(min_interval) as usize);
    let max_random_prob = hp.as_ref().and_then(|h| h.max_prob).unwrap_or(0.9);
    let prob_adjustment_factor = 0.03;
    let smoothing_factor = 0.8;
    let progress_threshold = 0.15 + 0.05 * (density / 3.0).min(1.0);

    let size_scale = 1.0 / (1.0 + (-(nvf - 30000.0) / 7000.0).exp());
    let perturbation_flips = hp.as_ref().and_then(|h| h.perturbation_flips)
        .unwrap_or(1 + (2.0 * size_scale) as usize);
    let stagnation_limit = hp.as_ref().and_then(|h| h.stagnation_limit)
        .unwrap_or(2 + (2.0 * (1.0 - (density / 5.0).min(1.0))) as usize);

    let mut last_check_residual = residual.len();
    let mut stagnation = 0usize;
    let mut var_age = vec![0u8; nv];
    let mut countdown = check_interval;
    let mut rounds = 0usize;

    let _probs_break: [u32; 16] = [2535, 551, 233, 127, 80, 55, 41, 30, 24, 19, 16, 13, 11, 9, 8, 7];

    let mut order=SignedOrder::new(&cl,&co);
    let ranges=CompactRanges::<PACKED>::new(all_off,p_bound);
    let mut cache = C::new(nv, &cl, &co, &vars);
    unsafe {
        let mut end=if check_interval==0{max_flips}else{(check_interval-1).min(max_flips)};
        'search: loop{
            if residual.is_empty() {break 'search;}
            let capacity_steps=(end-rounds).min(capacity_batch_limit);
            residual.reserve(capacity_steps*max_loss_per_flip);
            let capacity_end=rounds+capacity_steps;
            // This is an exact range guard, not a budget or heuristic
            // change. Phase capacity bounds every intermediate residual length.
            let small=residual.len().checked_add(capacity_steps*max_loss_per_flip).map_or(false,|n|n<phase_div::LIMIT);
            let done=if small{run_hot_phase::<C,PACKED,true>(&mut cache,&mut order,&ranges,&mut residual,rng,&coin,&appearances,&mut var_age,all_data,&mut rounds,capacity_end)}else{run_hot_phase::<C,PACKED,false>(&mut cache,&mut order,&ranges,&mut residual,rng,&coin,&appearances,&mut var_age,all_data,&mut rounds,capacity_end)};
            if done{break 'search;}

            if rounds<end{continue;}

            if residual.is_empty()||rounds>=max_flips{break;}

                let progress = last_check_residual as i64 - residual.len() as i64;
                let progress_ratio = progress as f64 / last_check_residual.max(1) as f64;

                if progress <= 0 {
                    stagnation += 1;
                    let prob_adjustment = prob_adjustment_factor
                        * (-progress as f64 / last_check_residual.max(1) as f64).min(1.0);
                    current_prob = (current_prob + prob_adjustment).min(max_random_prob);

                    if stagnation >= stagnation_limit {
                        let kicks = if stagnation >= 5 {
                            (perturbation_flips * 12).min(100)
                        } else if stagnation >= 4 {
                            (perturbation_flips * 6).min(50)
                        } else if stagnation >= 3 {
                            (perturbation_flips * 3).min(20)
                        } else {
                            (perturbation_flips + 2).min(10)
                        };

                        for _ in 0..kicks {
                            if residual.is_empty() { break; }
                            let rid = exact_div::rem(rng.gen::<usize>(), residual.len());
                            let pcid = *residual.get_unchecked(rid) as usize;
                            let ng_val = cache.good(pcid);
                            if ng_val > 0 {
                                residual.swap_remove(rid);
                                continue;
                            }
                            let pcs = 0usize;
                            let pce = order.len_bounded(pcid);
                            if pcs == pce { continue; }
                            let lit = order.lit(pcid, pcs + exact_div::clause_rem(rng.gen::<usize>(), pce - pcs));
                            let v = (lit.abs() - 1) as usize;

                            
    let (is,ie,ds,de)=ranges.of((v<<1)|((lit<0)as usize));
    let gains=cache.gain_many(all_data.get_unchecked(is..ie),v);
    residual.reserve(de-ds);
    let old_len=residual.len();
    let appended=cache.lose_many(all_data.get_unchecked(ds..de),v,residual.as_mut_ptr().add(old_len));
    residual.set_len(old_len+appended);
    cache.set_break(v,gains);

    // Assignment of v is represented exactly in the clause cache.
                            *var_age.get_unchecked_mut(v) = 0;
                        }
                        stagnation = 0;
                    }
                } else if progress_ratio > progress_threshold {
                    stagnation = 0;
                    current_prob = base_prob;
                } else {
                    stagnation = 0;
                    current_prob = current_prob * smoothing_factor + base_prob * (1.0 - smoothing_factor);
                }

                coin=ExactCoin::new(current_prob);
                last_check_residual = residual.len();
            
            end=rounds.saturating_add(check_interval).min(max_flips);
        }
    }

    cache.restore_variables(&cl,&co,&mut vars);
    let _ = save_solution(&Solution { variables: vars });
    Ok(())
}

// Called only at deterministic phase/capacity boundaries. No allocation,
// progress controller, kick loop or save callback is present in this hot loop.
#[inline(always)]unsafe fn run_hot_phase<C:CacheOps,const PACKED:bool,const SMALL:bool>(
 cache:&mut C,order:&mut SignedOrder,ranges:&CompactRanges<PACKED>,
 residual:&mut Vec<u32>,rng:&mut SmallRng,coin:&super::super::exact_coin::ExactCoin,
 appearances:&[u8],var_age:&mut[u8],all_data:&[u32],
 rounds_ref:&mut usize,capacity_end:usize
)->bool{
 let mut rounds=*rounds_ref;
            while rounds<capacity_end{
                // Phase-entry and no-shrink flip invariant prove non-emptiness here.
            let rand_val = rng.gen::<usize>();
            let cid=loop {
                let len=residual.len();
                let id=if SMALL{phase_div::rem_nonzero_small(rand_val,len)}else{exact_div::rem(rand_val,len)};
                let c=*residual.get_unchecked(id) as usize;
                if cache.good(c)==0 {break c;}
                // len>0 and id<len are proven by the phase invariant and exact
                // remainder. Keep every original swap-removal in the same order.
                let p=residual.as_mut_ptr();
                p.add(id).write(*p.add(len-1));
                residual.set_len(len-1);
                if len==1 {*rounds_ref=rounds;return true;}
            };

            let cs = 0usize;
            let ce = order.len_bounded(cid);
            let clen = ce - cs;

            let permutation_rank=if clen>1{
                let rank=exact_div::clause_rem(rand_val,clen);order.swap(cid,0,rank);rank
            }else{0};


            let ll=order.codes(cid);
            let vv=[ll[0]>>1,ll[1]>>1,ll[2]>>1];
            let b0=cache.breaks_of(vv[0]);
            let b1=if clen>1{cache.breaks_of(vv[1])}else{1};
            let b2=if clen>2{cache.breaks_of(vv[2])}else{1};
            let zero_mask=((b0==0)as usize)|(((b1==0)as usize)<<1)|(((b2==0)as usize)<<2);
            let chosen_code=if zero_mask!=0 {
                ClauseOrder::choose_zero_rank(ll,zero_mask,rand_val,permutation_rank)
            }else if coin.draw(rng){ll[0]}else{
                let mut selected=ll[0];let mut min_sad=b0;
                let mut penalty=(*appearances.get_unchecked(vv[0])as i32)-(((*var_age.get_unchecked(vv[0])as usize)/2).min(50)as i32);
                if clen>1 && min_sad>1 {
                    let p=(*appearances.get_unchecked(vv[1])as i32)-(((*var_age.get_unchecked(vv[1])as usize)/2).min(50)as i32);
                    if b1<min_sad || p<penalty {selected=ll[1];min_sad=b1.min(min_sad);penalty=p;}
                }
                if clen>2 && min_sad>1 {
                    let p=(*appearances.get_unchecked(vv[2])as i32)-(((*var_age.get_unchecked(vv[2])as usize)/2).min(50)as i32);
                    if b2<min_sad || p<penalty {selected=ll[2];}
                }
                selected
            };

            let v_idx=chosen_code>>1;
            
    let (is,ie,ds,de)=ranges.of(chosen_code);
    let gains=cache.gain_many(all_data.get_unchecked(is..ie),v_idx);
    // Covered by phase capacity bound.
    let old_len=residual.len();
    let appended=cache.lose_many(all_data.get_unchecked(ds..de),v_idx,residual.as_mut_ptr().add(old_len));
    residual.set_len(old_len+appended);
    cache.set_break(v_idx,gains);

    // Assignment of v_idx is represented exactly in the clause cache.
            // The selected variable occurs exactly once in this clause:
            // reset-to-zero followed by its own increment is exactly one.
            // Update the other two variables directly instead of rereading and
            // incrementing the just-reset selected variable in a three-item loop.
            *var_age.get_unchecked_mut(v_idx)=1;
            if clen==3 {
                let a=if vv[0]==v_idx{vv[1]}else{vv[0]};
                let b=if vv[2]==v_idx{vv[1]}else{vv[2]};
                let x=var_age.get_unchecked_mut(a);*x=x.saturating_add(1);
                let x=var_age.get_unchecked_mut(b);*x=x.saturating_add(1);
            }else if clen==2 {
                let a=if vv[0]==v_idx{vv[1]}else{vv[0]};
                let x=var_age.get_unchecked_mut(a);*x=x.saturating_add(1);
            }
            rounds += 1;

            }
 *rounds_ref=rounds;
 false
}
