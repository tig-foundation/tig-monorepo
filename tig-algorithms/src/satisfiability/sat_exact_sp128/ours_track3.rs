// The give-up latch is monotone: once true, it is never cleared. A phase
// beginning latched may omit repeated true|=predicate assignments. Unlatched
// phases retain every original post-flip latch event, and controller checks
// observe exactly the same value at every phase boundary. No deadline changes.
use super::super::short_mask::ShortMask;
use super::super::phase_div;
// Initial vars are retained only for absent-variable values and callbacks.
// During search, exact clause counts/XOR owners represent all incident values;
// selected false literal signs determine the same gain/loss orientation.
use super::super::raw_roulette::RawRoulette;
// Inductive invariant: stagnation is 0..3 at each periodic-check entry.
// A non-progress check increments it once; reaching 4 takes exactly 3 kicks
// and resets it to zero. Positive progress also resets it. Hence >=8, >=10,
// and >=15 branches are unreachable, including all reads of best_vars.
// The give-up latch is updated at exactly the same post-flip events as before.
use super::super::paged_small_signed::{SignedOrder as ClauseOrder,LiteralRanges};
use super::super::exact_div;
use super::super::paged_cache::{SmallBreakCache,Byte14BreakCache,CacheOps};
use anyhow::Result;
use rand::{rngs::SmallRng, Rng};
use tig_challenges::satisfiability::*;
use super::Hyperparameters;

// ---- grafted give-up (n_vars=10000,ratio=4267). Same construction as track2.
// L is an ABSOLUTE count: the quasi-stationary unsat count of a well-tuned SLS
// solver at its algorithmic threshold is O(1), not extensive (Lee/Ha/Jeon/Jeong
// arXiv:1005.0251 Table I; Ardelius & Aurell PRE 74 037702). It is 6 here rather
// than 16 because the level was fitted to this track's own first-passage trace.
// Deadline 5/12 = 41.7% of the BASE budget carries 2.66x margin over the worst
// traced solver (15.69%), against the 2.58x required at S=3 seeds. Anchored on
// base_max_flips, not max_flips: track3 is the only kernel that grows its own
// budget (up to 2x), so the mutable value is not a stable denominator.
const GIVEUP_LEVEL: usize = 6;
// ABSOLUTE flip count, not a fraction of the budget. First passage to the latch
// level is a property of the trajectory and is fuel-invariant; base_max_flips is
// linear in max_fuel_high, so a fractional deadline silently shrinks its own
// safety margin whenever a benchmarker runs at lower fuel. Live sat_imp_v4 runs
// this track at 180e9 and the source default is 125e9, against a 280e9
// calibration -- a fractional 5/12 would give 1.71x and 1.19x margin instead of
// the 2.58x required. 563M = 2.66x the worst traced first passage (2.119e8).
// If the budget is smaller than this the gate simply never fires and the kernel
// degrades to the parent, which is the safe direction.
const GIVEUP_DEADLINE_FLIPS: usize = 563_000_000;
const GIVEUP_ENABLE_NV_MAX: usize = 25000;

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
    let narrow=nv<=16384 && p_cnt.iter().zip(&n_cnt).all(|(&p,&n)|p+n<=255);
    if narrow {solve_impl::<Byte14BreakCache>(hp,rng,nv,nc,density,p_cnt,n_cnt,all_off,p_bound,all_data,cl,co,save_solution)}else{solve_impl::<SmallBreakCache>(hp,rng,nv,nc,density,p_cnt,n_cnt,all_off,p_bound,all_data,cl,co,save_solution)}
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
    // Each main flip appends at most the losing polarity's occurrence count.
    // Reserve for <=64 iterations, capped to ~4096 extra slots, at a phase
    // boundary. Heuristic control events still occur only at the original end.
    let max_loss_per_flip=p_cnt.iter().chain(n_cnt.iter()).copied().max().unwrap_or(0) as usize;
    let capacity_batch_limit=64usize.min((4096/max_loss_per_flip.max(1)).max(1));


    let default_fuel = if nv >= 10000 { 125_000_000_000.0 } else { 250_000_000_000.0 };
    let max_fuel = hp.as_ref().and_then(|h| h.max_fuel_high).unwrap_or(default_fuel);

    let avg_clause_size = cl.len() as f64 / nc as f64;
    let difficulty_factor = density * avg_clause_size.sqrt();
    let scale_factor = if nv > 25000 { 1.5 } else { 1.0 };
    let base_fuel = (2000.0 + 100.0 * difficulty_factor) * (nv as f64).sqrt() * scale_factor;
    let flip_fuel = (200.0 + difficulty_factor) / scale_factor;
    let remaining = (max_fuel - base_fuel).max(0.0);
    let mut max_flips = if flip_fuel > 0.0 { (remaining / flip_fuel) as usize } else { 0 };
    let base_max_flips = max_flips;

    let mut vars = vec![false; nv];
    // Compute clause lengths
    let mut max_len = 0usize;
    let mut lengths = vec![0usize; nc];
    for i in 0..nc {
        let len = (co[i + 1] - co[i]) as usize;
        lengths[i] = len;
        if len > max_len {
            max_len = len;
        }
    }
    // Bucket sort clauses by length
    let mut buckets: Vec<Vec<usize>> = (0..=max_len).map(|_| Vec::new()).collect();
    for i in 0..nc {
        buckets[lengths[i]].push(i);
    }

    // Greedy assignment: satisfy shortest clauses first
    for l in 1..=max_len {
        for &cid in buckets[l].iter() {
            let s = co[cid] as usize;
            let e = co[cid + 1] as usize;
            // Check if clause already satisfied
            let mut already = false;
            for j in s..e {
                let lit = cl[j];
                let v = (lit.abs() - 1) as usize;
                if (lit > 0 && vars[v]) || (lit < 0 && !vars[v]) {
                    already = true;
                    break;
                }
            }
            if already {
                continue;
            }

            // Choose literal with highest literal count (p_cnt for positive, n_cnt for negative)
            let mut best_score: u32 = 0;
            let mut best_v = 0usize;
            let mut best_target = false;
            let mut count = 0;
            for j in s..e {
                let lit = cl[j];
                let v = (lit.abs() - 1) as usize;
                let target_val = lit > 0;
                if vars[v] == target_val {
                    // already satisfied, can't happen due to check above
                    continue;
                }
                let score = if target_val { p_cnt[v] } else { n_cnt[v] };
                if score > best_score {
                    best_score = score;
                    best_v = v;
                    best_target = target_val;
                    count = 1;
                } else if score == best_score {
                    count += 1;
                    if rng.gen::<usize>() % count == 0 {
                        best_v = v;
                        best_target = target_val;
                    }
                }
            }
            if best_score == 0 {
                // All scores zero, pick any random literal
                let idx = rng.gen::<usize>() % (e - s);
                let lit = cl[s + idx];
                best_v = (lit.abs() - 1) as usize;
                best_target = lit > 0;
            }
            vars[best_v] = best_target;
        }
    }

    // Build num_good and residual from final assignment
    let mut num_good = vec![0u8; nc];
    for i in 0..nc {
        let s = co[i] as usize;
        let e = co[i + 1] as usize;
        let mut good = 0u8;
        for j in s..e {
            let l = cl[j];
            let v = (l.abs() - 1) as usize;
            if (l > 0 && vars[v]) || (l < 0 && !vars[v]) {
                good = good + 1;
            }
        }
        num_good[i] = good;
    }

    let mut residual: Vec<u32> = Vec::with_capacity(nc);
    let mut true_unsat = 0usize;
    for i in 0..nc {
        if num_good[i] == 0 {
            residual.push(i as u32);
            true_unsat += 1;
        }
    }

    if true_unsat == 0 {
        let _ = save_solution(&Solution { variables: vars });
        return Ok(());
    }


    let giveup_deadline: usize = if nv <= GIVEUP_ENABLE_NV_MAX {
        GIVEUP_DEADLINE_FLIPS
    } else {
        usize::MAX
    };
    // Seeded from the post-greedy-init minimum, so a nonce starting below the
    // level is latched immediately and can never be abandoned.
    let mut giveup_latched: bool = true_unsat <= GIVEUP_LEVEL;

    let large_problem_scale = ((nv as f64 - 25000.0) / 35000.0).max(0.0).min(1.0);
    let base_interval = 60.0 - 30.0 * large_problem_scale;
    let min_interval = if large_problem_scale > 0.0 { 15.0 } else { 25.0 };
    let density_factor_ci = if density > 4.0 { 1.2 } else { 1.0 };
    let check_interval = hp.as_ref().and_then(|h| h.check_interval)
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

    let roulette=RawRoulette::new(&probsat_weights);
    let mut last_check_unsat = true_unsat;
    let mut rounds = 0usize;
    let mut stagnation = 0usize;

    let paged_data=super::super::large_buffer::from_slice(all_data);
    let all_data=&*paged_data;
    let mut order = ClauseOrder::new(&cl, &co);
    let ranges=LiteralRanges::new(all_off,p_bound);
    let mut cache = C::new(nv, &cl, &co, &vars);
    let mut short_mask=ShortMask::new(nv,&cl,&co,&vars);
    let mut next_check_interval = check_interval;
    unsafe {
        if check_interval==0 && max_flips>0{let _=rounds%check_interval;}
        let mut end=check_interval.min(max_flips);
        'search: loop{
            let capacity_steps=(end-rounds).min(capacity_batch_limit);
            residual.reserve(capacity_steps*max_loss_per_flip);
            let capacity_end=rounds+capacity_steps;
            let small=residual.len().checked_add(capacity_steps*max_loss_per_flip).map_or(false,|n|n<phase_div::LIMIT);
            let done=if small{if giveup_latched{run_sample_phase::<C,true,true>(&mut cache,&mut order,&ranges,&mut residual,rng,&roulette,&probsat_weights,nc,all_data,&mut rounds,capacity_end,&mut true_unsat,&mut giveup_latched,&mut short_mask)}else{run_sample_phase::<C,true,false>(&mut cache,&mut order,&ranges,&mut residual,rng,&roulette,&probsat_weights,nc,all_data,&mut rounds,capacity_end,&mut true_unsat,&mut giveup_latched,&mut short_mask)}}else{if giveup_latched{run_sample_phase::<C,false,true>(&mut cache,&mut order,&ranges,&mut residual,rng,&roulette,&probsat_weights,nc,all_data,&mut rounds,capacity_end,&mut true_unsat,&mut giveup_latched,&mut short_mask)}else{run_sample_phase::<C,false,false>(&mut cache,&mut order,&ranges,&mut residual,rng,&roulette,&probsat_weights,nc,all_data,&mut rounds,capacity_end,&mut true_unsat,&mut giveup_latched,&mut short_mask)}};
            if done{break 'search;}
            // A capacity-only boundary is not a search-control event. Continue
            // the same phase without changing noise, RNG, ages, or budgets.
            if rounds<end {continue;}

            if rounds>=max_flips||true_unsat==0{break;}

                if !giveup_latched && rounds >= giveup_deadline {break 'search;}
                let progress = last_check_unsat as i64 - true_unsat as i64;
                let progress_ratio = progress as f64 / last_check_unsat.max(1) as f64;
                let progress_threshold = 0.15 + 0.05 * (density / 3.0).min(1.0);

                if progress <= 0 {
                    stagnation += 1;

                    if stagnation >= 4 {

                            let kicks = 3; // stagnation reaches at most 4 before this reset
                            for _ in 0..kicks {
                                if true_unsat == 0 { break; }
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

                                
    let (is,ie,ds,de)=ranges.of((v<<1)|((lit<0)as usize));

    let gains=cache.gain_many(all_data.get_unchecked(is..ie),v);
    residual.reserve(de-ds);
    let old_len=residual.len();
    let appended=cache.lose_many(all_data.get_unchecked(ds..de),v,residual.as_mut_ptr().add(old_len));
    residual.set_len(old_len+appended);
    cache.set_break(v,gains);
    short_mask.flip(v);
    true_unsat=true_unsat+appended-gains as usize;

    // Assignment of v is represented exactly in the clause cache.
                            }
                            stagnation = 0;
                        
                    }
                } else if progress_ratio > progress_threshold {
                    stagnation = 0;
                } else {
                    stagnation = 0;
                }

                last_check_unsat = true_unsat;

                // Dynamic budget adjustment
                if progress_ratio > 0.2 {
                    let increase = (base_max_flips / 100).max(1);
                    max_flips = (max_flips + increase).min(2 * base_max_flips);
                }

            
            end=rounds.saturating_add(check_interval).min(max_flips);
        }
    }

    cache.restore_variables(&cl,&co,&mut vars);
    let _ = save_solution(&Solution { variables: vars });
    Ok(())
}
#[inline(always)]
unsafe fn choose_shape<const LENGTH:usize,C:CacheOps>(ll:[usize;3],length:usize,cache:&C,roulette:&RawRoulette,rng:&mut SmallRng,probsat_weights:&[f64],nc:usize)->usize{
    let clen=if LENGTH==0{length}else{LENGTH};let vv=[ll[0]>>1,ll[1]>>1,ll[2]>>1];
            let b0=cache.breaks_of(vv[0]);
            let b1=if clen>1{cache.breaks_of(vv[1])}else{1};
            let b2=if clen>2{cache.breaks_of(vv[2])}else{1};
            let zero_mask=((b0==0)as usize)|(((b1==0)as usize)<<1)|(((b2==0)as usize)<<2);
            let chosen_code=if zero_mask!=0 {
                if zero_mask&1!=0{ll[0]}else if zero_mask&2!=0{ll[1]}else{ll[2]}
            }else{
                let word=rng.gen::<u64>();
                if clen==3 && (b0|b1|b2)<8 {
                    ll[roulette.choose(b0,b1,b2,word)]
                }else{
                let u=RawRoulette::sample(word);
                let w0=*probsat_weights.get_unchecked(b0.min(nc));
                let w1=if clen>1{*probsat_weights.get_unchecked(b1.min(nc))}else{0.0};
                let w2=if clen>2{*probsat_weights.get_unchecked(b2.min(nc))}else{0.0};
                let mut total=0.0;total+=w0;if clen>1{total+=w1;}if clen>2{total+=w2;}
                let mut r=u*total;
                r-=w0;
                if r<=0.0{ll[0]}else if clen>1{
                    r-=w1;
                    if r<=0.0{ll[1]}else if clen>2{r-=w2;if r<=0.0{ll[2]}else{ll[0]}}else{ll[0]}
                }else{ll[0]}
                }
            };


    chosen_code
 }
#[cold]
#[inline(never)]
unsafe fn choose_short<C:CacheOps>(ll:[usize;3],length:usize,cache:&C,roulette:&RawRoulette,rng:&mut SmallRng,probsat_weights:&[f64],nc:usize)->usize{
    choose_shape::<0,C>(ll,length,cache,roulette,rng,probsat_weights,nc)
}

#[inline(always)]
unsafe fn run_sample_phase<C:CacheOps,const SMALL:bool,const LATCHED:bool>(cache:&mut C,order:&mut ClauseOrder,ranges:&LiteralRanges,residual:&mut Vec<u32>,rng:&mut SmallRng,roulette:&RawRoulette,probsat_weights:&[f64],nc:usize,all_data:&[u32],rounds:&mut usize,end:usize,true_unsat:&mut usize,giveup_latched:&mut bool,short_mask:&mut ShortMask)->bool{
 while *rounds<end{

            if *true_unsat==0{return true;}

            let cid=if short_mask.all_satisfied(){
            let mut cid=0usize;
            for sample in 0..3{
                let candidate=loop{
                    let word=rng.gen::<usize>();let len=residual.len();
                    let id=if SMALL{phase_div::rem_nonzero_small(word,len)}else{exact_div::rem(word,len)};
                    let c=*residual.get_unchecked(id)as usize;
                    if cache.good(c)>0{
                        let last=residual.len()-1;let ptr=residual.as_mut_ptr();ptr.add(id).write(*ptr.add(last));residual.set_len(last);
                    }else{break c;}
                };
                if sample==0{cid=candidate;}
            }

                cid
            }else{
            let mut cid=0usize;let mut min_len=usize::MAX;
            for _ in 0..3{
                let candidate=loop{
                    let word=rng.gen::<usize>();let len=residual.len();
                    let id=if SMALL{phase_div::rem_nonzero_small(word,len)}else{exact_div::rem(word,len)};
                    let c=*residual.get_unchecked(id)as usize;
                    if cache.good(c)>0{
                        let last=residual.len()-1;let ptr=residual.as_mut_ptr();ptr.add(id).write(*ptr.add(last));residual.set_len(last);
                    }else{break c;}
                };
                let len=order.len(candidate);if len<min_len{min_len=len;cid=candidate;}
            }

                cid
            };
            let cs = 0usize;
            let ce = order.len(cid);
            let clen = ce - cs;

            if clen > 1 {
                let ri = exact_div::clause_rem(rng.gen::<usize>(), clen);
                order.swap(cid, 0, ri);
            }


            let ll=order.codes(cid);
            let vv=[ll[0]>>1,ll[1]>>1,ll[2]>>1];
            let chosen_code=if clen==3{choose_shape::<3,C>(ll,clen,cache,roulette,rng,probsat_weights,nc)}else{choose_short(ll,clen,cache,roulette,rng,probsat_weights,nc)};

            let v_idx=chosen_code>>1;
            
    let (is,ie,ds,de)=ranges.of(chosen_code);

    let gains=cache.gain_many(all_data.get_unchecked(is..ie),v_idx);
    // Capacity for all main-flip appends was reserved at this batch entry.
    let old_len=residual.len();
    let appended=cache.lose_many(all_data.get_unchecked(ds..de),v_idx,residual.as_mut_ptr().add(old_len));
    residual.set_len(old_len+appended);
    cache.set_break(v_idx,gains);
    short_mask.flip(v_idx);
    *true_unsat=*true_unsat+appended-gains as usize;

    // Assignment of v_idx is represented exactly in the clause cache.
            
            if !LATCHED{*giveup_latched |= *true_unsat <= GIVEUP_LEVEL;}

            *rounds += 1;

            
 }
 false
}
