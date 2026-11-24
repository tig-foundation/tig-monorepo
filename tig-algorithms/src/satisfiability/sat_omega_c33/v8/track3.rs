// Derived from a112/a125 Track3; v7-family majority rescue and implementation changes are described in README.md.
// The give-up latch is monotone: once true, it is never cleared. A phase
// beginning latched may omit repeated true|=predicate assignments. Unlatched
// phases retain every original post-flip latch event, and controller checks
// observe exactly the same value at every phase boundary. No deadline changes.
use super::donor_kernel::short_mask::ShortMask;
use super::donor_kernel::phase_div;
// Initial vars are retained only for absent-variable values and callbacks.
// During search, exact clause counts/XOR owners represent all incident values;
// selected false literal signs determine the same gain/loss orientation.
use super::donor_kernel::raw_roulette::RawRoulette;
// Inductive invariant: stagnation is 0..3 at each periodic-check entry.
// A non-progress check increments it once; reaching 4 takes exactly 3 kicks
// and resets it to zero. Positive progress also resets it. Hence >=8, >=10,
// and >=15 branches are unreachable, including all reads of best_vars.
// The give-up latch is updated at exactly the same post-flip events as before.
use super::donor_kernel::paged_small_signed::{SignedOrder as ClauseOrder,LiteralRanges};
use super::donor_kernel::exact_div;
use super::donor_kernel::paged_cache::{SmallBreakCache,Byte14BreakCache,CacheOps};
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


// Threshold 0 is the original baseline, 1 reproduces the existing gated
// majority route, and 2 tests the observed missing trigger on all three R18 cases.
const RESCUE_MAX_UNSAT: usize = 2;

#[derive(Clone,Copy,Debug)]
struct ContinuationGate { flips:usize, best:usize, hard:usize, early:usize,early_best:usize,extend:bool,capture:bool }
#[derive(Default,Clone,Debug)]
pub struct GateStats { pub flips:usize,pub gate_flips:usize,pub gate_best:Option<usize>,pub extended:bool,pub rescue_flips:usize,pub early_best:Option<usize>,pub near_u:usize,pub near_x:Vec<bool> }
fn apply_c15_gate(g:ContinuationGate,st:&mut GateStats,rounds:usize,best:usize,max:&mut usize)->bool{
 if rounds>=g.early&&st.early_best.is_none(){st.early_best=Some(best);if best>g.early_best{return false;}}
 if rounds>=g.flips&&st.gate_best.is_none(){st.gate_best=Some(best);if g.extend&&best<=g.best{st.extended=true;*max=g.hard;}}
 true
}
fn fuel_flips(nv:usize,nc:usize,nl:usize,fuel:f64)->usize{
 let density=nc as f64/nv as f64;let difficulty=density*(nl as f64/nc as f64).sqrt();let scale=if nv>25000{1.5}else{1.};
 ((fuel-(2000.+100.*difficulty)*(nv as f64).sqrt()*scale).max(0.)/((200.+difficulty)/scale))as usize
}
pub(super) fn solve_c15_gated(
 rng:&mut SmallRng,nv:usize,nc:usize,density:f64,p_cnt:Vec<u32>,n_cnt:Vec<u32>,all_off:&[u32],p_bound:&[u32],all_data:&[u32],cl:&mut Vec<i32>,co:&[u32],save:&dyn Fn(&Solution)->Result<()>,extend:bool,
)->Result<GateStats>{
 assert_eq!(nv,10000);let flips=fuel_flips(nv,nc,cl.len(),6_000_000_000.);let saved_rng=rng.clone();let saved_cl=cl.clone();
 let hp=Some(Hyperparameters{base_prob:None,max_prob:None,check_interval:None,stagnation_limit:None,perturbation_flips:None,max_fuel_high:Some(6_000_000_000.),max_fuel_low:None});
 let mut stats=GateStats{gate_flips:flips,..GateStats::default()};
 let best=solve_impl_gated::<Byte14BreakCache,false>(&hp,rng,nv,nc,density,&p_cnt,&n_cnt,all_off,p_bound,all_data,cl,co,save,false,Some(ContinuationGate{flips,best:9,hard:if extend{95_000_000}else{2*flips},early:12_000_000,early_best:18,extend,capture:extend}),&mut stats)?;
 if best>0&&best<=RESCUE_MAX_UNSAT{
  let mut rr=saved_rng;let mut cc=saved_cl;let mut rescue=GateStats::default();
  solve_impl_gated::<Byte14BreakCache,true>(&hp,&mut rr,nv,nc,density,&p_cnt,&n_cnt,all_off,p_bound,all_data,&mut cc,co,save,true,Some(ContinuationGate{flips:usize::MAX,best:usize::MAX,hard:flips,early:usize::MAX,early_best:usize::MAX,extend:false,capture:extend}),&mut rescue)?;
  stats.rescue_flips=rescue.flips;
  if !rescue.near_x.is_empty()&&(stats.near_x.is_empty()||rescue.near_u<stats.near_u){stats.near_u=rescue.near_u;stats.near_x=rescue.near_x;}
 }Ok(stats)
}

pub fn solve(
    hp: &Option<Hyperparameters>, rng: &mut SmallRng,
    nv: usize, nc: usize, density: f64,
    p_cnt: Vec<u32>, n_cnt: Vec<u32>,
    all_off: &[u32], p_bound: &[u32], all_data: &[u32],
    cl: &mut Vec<i32>, co: &[u32],
    save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Result<()> {
    let narrow = nv <= 16384 && p_cnt.iter().zip(&n_cnt).all(|(&p,&n)| p+n<=255);
    if narrow {
        solve_pair::<Byte14BreakCache>(hp,rng,nv,nc,density,p_cnt,n_cnt,all_off,p_bound,all_data,cl,co,save_solution)
    } else {
        solve_pair::<SmallBreakCache>(hp,rng,nv,nc,density,p_cnt,n_cnt,all_off,p_bound,all_data,cl,co,save_solution)
    }
}

fn solve_pair<C:CacheOps>(
    hp: &Option<Hyperparameters>, rng: &mut SmallRng,
    nv: usize, nc: usize, density: f64,
    p_cnt: Vec<u32>, n_cnt: Vec<u32>,
    all_off: &[u32], p_bound: &[u32], all_data: &[u32],
    cl: &mut Vec<i32>, co: &[u32],
    save_solution: &dyn Fn(&Solution) -> Result<()>,
) -> Result<()> {
    let saved_rng = rng.clone();
    let saved_cl = cl.clone();
    // The larger default primary budget contains the original primary prefix.
    // Keep the majority rescue at its original 125e9 default: extending both
    // routes paid for a longer failed rescue without adding solutions in replay.
    // Explicit max_fuel_high still configures both routes as before.
    let mut primary_hp=hp.clone();
    if nv==10_000 && hp.as_ref().and_then(|h|h.max_fuel_high).is_none() {
        let h=primary_hp.get_or_insert(Hyperparameters {
            base_prob:None,max_prob:None,check_interval:None,stagnation_limit:None,
            perturbation_flips:None,max_fuel_high:None,max_fuel_low:None,
        });
        h.max_fuel_high=Some(180_000_000_000.0);
    }
    let best = solve_impl::<C,false>(&primary_hp,rng,nv,nc,density,&p_cnt,&n_cnt,
        all_off,p_bound,all_data,cl,co,save_solution,false)?;
    if best > 0 && best <= RESCUE_MAX_UNSAT {
        let mut rescue_rng = saved_rng;
        let mut rescue_cl = saved_cl;
        let verify_cl = rescue_cl.clone();
        let save_complete = |solution: &Solution| -> Result<()> {
            if solution.variables.len() == nv && (0..nc).all(|cid| {
                verify_cl[co[cid] as usize..co[cid+1] as usize].iter().any(|&lit| {
                    solution.variables[(lit.abs()-1) as usize] == (lit>0)
                })
            }) { save_solution(solution)?; }
            Ok(())
        };
        solve_impl::<C,true>(hp,&mut rescue_rng,nv,nc,density,&p_cnt,&n_cnt,
            all_off,p_bound,all_data,&mut rescue_cl,co,&save_complete,true)?;
    }
    Ok(())
}
fn solve_impl<C:CacheOps,const SINGLE_SAMPLE:bool>(
    hp: &Option<Hyperparameters>,
    rng: &mut SmallRng,
    nv: usize, nc: usize, density: f64,
    p_cnt: &[u32], n_cnt: &[u32],
    all_off: &[u32], p_bound: &[u32],
    all_data: &[u32],
    cl: &mut Vec<i32>, co: &[u32],
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    majority_init: bool,
) -> Result<usize> {
    solve_impl_gated::<C,SINGLE_SAMPLE>(hp,rng,nv,nc,density,p_cnt,n_cnt,all_off,p_bound,all_data,cl,co,save_solution,majority_init,None,&mut GateStats::default())
}
fn solve_impl_gated<C:CacheOps,const SINGLE_SAMPLE:bool>(
    hp:&Option<Hyperparameters>,rng:&mut SmallRng,nv:usize,nc:usize,density:f64,p_cnt:&[u32],n_cnt:&[u32],all_off:&[u32],p_bound:&[u32],all_data:&[u32],cl:&mut Vec<i32>,co:&[u32],save_solution:&dyn Fn(&Solution)->Result<()>,majority_init:bool,gate:Option<ContinuationGate>,gate_stats:&mut GateStats,
) -> Result<usize> {
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
    if let Some(g)=gate{max_flips=max_flips.min(g.hard);}

    let mut vars = vec![false; nv];
    if majority_init {
        for v in 0..nv {
            if p_cnt[v] > n_cnt[v] { vars[v] = true; }
            else if p_cnt[v] < n_cnt[v] { vars[v] = false; }
            else { vars[v] = rng.gen_bool(0.5); }
        }
    } else {
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
        return Ok(0);
    }

    if gate.map_or(false,|g|g.capture)&&true_unsat>0&&true_unsat<=3{gate_stats.near_u=true_unsat;gate_stats.near_x=vars.clone();}
    let mut best_unsat = true_unsat;
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
        for i in 0..=nc.min(1080) {
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

    let paged_data=super::donor_kernel::large_buffer::from_slice(all_data);
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
            if let Some(g)=gate{
                if !apply_c15_gate(g,gate_stats,rounds,best_unsat,&mut max_flips){break 'search;}
            }
            let capacity_steps=(end-rounds).min(capacity_batch_limit);
            residual.reserve(capacity_steps*max_loss_per_flip);
            let mut capacity_end=rounds+capacity_steps;
            if let Some(g)=gate{if rounds<g.flips{capacity_end=capacity_end.min(g.flips);}if rounds<g.early{capacity_end=capacity_end.min(g.early);}}
            let small=residual.len().checked_add(capacity_steps*max_loss_per_flip).map_or(false,|n|n<phase_div::LIMIT);
            let done=if small{if giveup_latched{run_sample_phase::<C,true,true,SINGLE_SAMPLE>(&mut cache,&mut order,&ranges,&mut residual,rng,&roulette,&probsat_weights,nc,all_data,&mut rounds,capacity_end,&mut true_unsat,&mut best_unsat,&mut giveup_latched,&mut short_mask,gate.map_or(false,|g|g.capture),gate_stats,&cl,&co,&mut vars)}else{run_sample_phase::<C,true,false,SINGLE_SAMPLE>(&mut cache,&mut order,&ranges,&mut residual,rng,&roulette,&probsat_weights,nc,all_data,&mut rounds,capacity_end,&mut true_unsat,&mut best_unsat,&mut giveup_latched,&mut short_mask,gate.map_or(false,|g|g.capture),gate_stats,&cl,&co,&mut vars)}}else{if giveup_latched{run_sample_phase::<C,false,true,SINGLE_SAMPLE>(&mut cache,&mut order,&ranges,&mut residual,rng,&roulette,&probsat_weights,nc,all_data,&mut rounds,capacity_end,&mut true_unsat,&mut best_unsat,&mut giveup_latched,&mut short_mask,gate.map_or(false,|g|g.capture),gate_stats,&cl,&co,&mut vars)}else{run_sample_phase::<C,false,false,SINGLE_SAMPLE>(&mut cache,&mut order,&ranges,&mut residual,rng,&roulette,&probsat_weights,nc,all_data,&mut rounds,capacity_end,&mut true_unsat,&mut best_unsat,&mut giveup_latched,&mut short_mask,gate.map_or(false,|g|g.capture),gate_stats,&cl,&co,&mut vars)}};
            if done{break 'search;}
            // A capacity-only boundary is not a search-control event. Continue
            // the same phase without changing noise, RNG, ages, or budgets.
            if rounds<end {continue;}

            if let Some(g)=gate{if !apply_c15_gate(g,gate_stats,rounds,best_unsat,&mut max_flips){break 'search;}}
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
                if progress_ratio > 0.2 && !gate_stats.extended {
                    let increase = (base_max_flips / 100).max(1);
                    max_flips = (max_flips + increase).min(2 * base_max_flips);
                    if let Some(g)=gate{max_flips=max_flips.min(g.hard);}
                }

            
            end=rounds.saturating_add(check_interval).min(max_flips);
        }
    }

    cache.restore_variables(&cl,&co,&mut vars);
    gate_stats.flips=rounds;
    let _ = save_solution(&Solution { variables: vars });
    Ok(best_unsat)
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
unsafe fn run_sample_phase<C:CacheOps,const SMALL:bool,const LATCHED:bool,const SINGLE_SAMPLE:bool>(cache:&mut C,order:&mut ClauseOrder,ranges:&LiteralRanges,residual:&mut Vec<u32>,rng:&mut SmallRng,roulette:&RawRoulette,probsat_weights:&[f64],nc:usize,all_data:&[u32],rounds:&mut usize,end:usize,true_unsat:&mut usize,best_unsat:&mut usize,giveup_latched:&mut bool,short_mask:&mut ShortMask,capture:bool,near:&mut GateStats,cl:&[i32],co:&[u32],vars:&mut Vec<bool>)->bool{
 while *rounds<end{

            if *true_unsat==0{return true;}

            let cid=if short_mask.all_satisfied(){
            let mut cid=0usize;
            // Keep the primary trajectory; use one sample only in majority rescue.
            for sample in 0..if SINGLE_SAMPLE {1} else {3}{
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
            
            if capture&&*true_unsat>0&&*true_unsat<=3&&*true_unsat<*best_unsat{cache.restore_variables(cl,co,vars);near.near_u=*true_unsat;near.near_x=vars.clone();}
            *best_unsat = (*best_unsat).min(*true_unsat);
            if !LATCHED{*giveup_latched |= *true_unsat <= GIVEUP_LEVEL;}

            *rounds += 1;

            
 }
 false
}

#[cfg(test)]mod c15_tests{
 use super::*;
 #[test]fn cutoff_matches_every_f64_bit_and_integer_roulette_choice(){let old:Vec<_>=(0..=42670).map(|i|2.06f64.powf(-(i as f64))).collect();let mut new=vec![0.;42671];for i in 0..=1080{new[i]=2.06f64.powf(-(i as f64));}for i in 0..old.len(){assert_eq!(old[i].to_bits(),new[i].to_bits(),"index{}",i);}let a=RawRoulette::new(&old);let b=RawRoulette::new(&new);for x in 0..8{for y in 0..8{for z in 0..8{for r in [0,1,u64::MAX,0x123456789abcdef0]{assert_eq!(a.choose(x,y,z,r),b.choose(x,y,z,r));}}}}}
 #[test]fn early_gate_rejects_hopeless_but_extends_before_the_old_budget_exit(){let g=ContinuationGate{early:12,early_best:18,flips:29,best:9,hard:95,extend:true,capture:true};let mut s=GateStats::default();let mut max=29;assert!(apply_c15_gate(g,&mut s,11,55,&mut max));assert!(!apply_c15_gate(g,&mut s,12,19,&mut max));let mut s=GateStats::default();assert!(apply_c15_gate(g,&mut s,12,18,&mut max));assert!(apply_c15_gate(g,&mut s,29,9,&mut max));assert!(s.extended);assert_eq!(max,95);assert_eq!(s.gate_best,Some(9));}
}


const C26_STAGES:[(usize,usize);5]=[(2_000_000,85),(6_000_000,43),(8_000_000,31),(12_000_000,18),(16_000_000,12)];
fn apply_c26_gate(g:ContinuationGate,st:&mut GateStats,rounds:usize,best:usize,max:&mut usize,stage:&mut usize)->bool{
 if g.early==usize::MAX{return apply_c15_gate(g,st,rounds,best,max);}
 while *stage<C26_STAGES.len() && rounds>=C26_STAGES[*stage].0{let (at,lim)=C26_STAGES[*stage];*stage+=1;if at==g.early{st.early_best=Some(best);}if best>lim{return false;}}
 if rounds>=g.flips && st.gate_best.is_none(){st.gate_best=Some(best);if g.extend && best<=g.best{st.extended=true;*max=g.hard;}else{return false;}}
 true
}
#[cfg(test)]mod c26_gate_tests{use super::*;
 #[test]fn milestones_and_budget_reject_only_at_boundaries(){
  let g=ContinuationGate{early:12_000_000,early_best:18,flips:29_430_000,best:9,hard:95_000_000,extend:true,capture:true};
  for i in 0..C26_STAGES.len(){let(at,lim)=C26_STAGES[i];let mut s=GateStats::default();let mut max=g.flips;let mut stage=i;assert!(apply_c26_gate(g,&mut s,at-1,lim+1,&mut max,&mut stage));assert!(!apply_c26_gate(g,&mut s,at,lim+1,&mut max,&mut stage));}
  let mut s=GateStats::default();let mut stage=5;let mut max=g.flips;assert!(!apply_c26_gate(g,&mut s,g.flips,10,&mut max,&mut stage));assert_eq!(s.gate_best,Some(10));
  let mut s=GateStats::default();let mut stage=5;let mut max=g.flips;assert!(apply_c26_gate(g,&mut s,g.flips,9,&mut max,&mut stage));assert!(s.extended);assert_eq!(max,g.hard);
 }
 #[test]fn passing_trajectory_matches_original_rng_flips_and_assignment(){
  use rand::{RngCore,SeedableRng};use std::cell::RefCell;
  let nv=4usize;let nc=8usize;let clauses:Vec<Vec<i32>>=(0..8).map(|m|(0..3).map(|v|if m&(1<<v)==0{v+1}else{-(v+1)}).collect()).collect();
  let(mut pc,mut nn)=(vec![0u32;nv],vec![0u32;nv]);for cl in &clauses{for &l in cl{let v=(l.abs()-1)as usize;if l>0{pc[v]+=1}else{nn[v]+=1;}}}
  let mut off=vec![0u32;nv+1];for v in 0..nv{off[v+1]=off[v]+pc[v]+nn[v];}let mut bound=vec![0u32;nv];let(mut pp,mut np)=(vec![0u32;nv],vec![0u32;nv]);for v in 0..nv{pp[v]=off[v];np[v]=off[v]+pc[v];bound[v]=np[v];}
  let mut data=vec![0u32;off[nv]as usize];let(mut cl,mut co)=(Vec::new(),vec![0u32]);for(cid,c)in clauses.iter().enumerate(){for &l in c{let v=(l.abs()-1)as usize;let ix=if l>0{&mut pp[v]}else{&mut np[v]};data[*ix as usize]=cid as u32;*ix+=1;cl.push(l);}co.push(cl.len()as u32);}
  let hp=Some(Hyperparameters{base_prob:None,max_prob:None,check_interval:None,stagnation_limit:None,perturbation_flips:None,max_fuel_high:Some(2_000_000.0),max_fuel_low:None});let g=ContinuationGate{early:12_000_000,early_best:18,flips:usize::MAX,best:9,hard:usize::MAX,extend:true,capture:true};
  let(mut a,mut b)=(SmallRng::seed_from_u64(51),SmallRng::seed_from_u64(51));let(mut x,mut y)=(GateStats::default(),GateStats::default());let(mut cl2,sa,sb)=(cl.clone(),RefCell::new(Vec::new()),RefCell::new(Vec::new()));
  let one=solve_impl_gated::<Byte14BreakCache,false>(&hp,&mut a,nv,nc,2.0,&pc,&nn,&off,&bound,&data,&mut cl,&co,&|s|{sa.borrow_mut().push(s.variables.clone());Ok(())},false,Some(g),&mut x).unwrap();
  let two=solve_c26_impl_gated::<Byte14BreakCache,false>(&hp,&mut b,nv,nc,2.0,&pc,&nn,&off,&bound,&data,&mut cl2,&co,&|s|{sb.borrow_mut().push(s.variables.clone());Ok(())},false,Some(g),&mut y).unwrap();
  assert_eq!((one,x.flips,x.near_u,x.near_x,a.next_u64()),(two,y.flips,y.near_u,y.near_x,b.next_u64()));assert_eq!(*sa.borrow(),*sb.borrow());
 }
}
fn solve_c26_impl_gated<C:CacheOps,const SINGLE_SAMPLE:bool>(
    hp:&Option<Hyperparameters>,rng:&mut SmallRng,nv:usize,nc:usize,density:f64,p_cnt:&[u32],n_cnt:&[u32],all_off:&[u32],p_bound:&[u32],all_data:&[u32],cl:&mut Vec<i32>,co:&[u32],save_solution:&dyn Fn(&Solution)->Result<()>,majority_init:bool,gate:Option<ContinuationGate>,gate_stats:&mut GateStats,
) -> Result<usize> {
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
    if let Some(g)=gate{max_flips=max_flips.min(g.hard);}

    let mut vars = vec![false; nv];
    if majority_init {
        for v in 0..nv {
            if p_cnt[v] > n_cnt[v] { vars[v] = true; }
            else if p_cnt[v] < n_cnt[v] { vars[v] = false; }
            else { vars[v] = rng.gen_bool(0.5); }
        }
    } else {
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
        return Ok(0);
    }

    if gate.map_or(false,|g|g.capture)&&true_unsat>0&&true_unsat<=3{gate_stats.near_u=true_unsat;gate_stats.near_x=vars.clone();}
    let mut best_unsat = true_unsat;
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
        for i in 0..=nc.min(1080) {
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

    let paged_data=super::donor_kernel::large_buffer::from_slice(all_data);
    let all_data=&*paged_data;
    let mut order = ClauseOrder::new(&cl, &co);
    let ranges=LiteralRanges::new(all_off,p_bound);
    let mut cache = C::new(nv, &cl, &co, &vars);
    let mut short_mask=ShortMask::new(nv,&cl,&co,&vars);
    let mut next_check_interval = check_interval;
    unsafe {
        if check_interval==0 && max_flips>0{let _=rounds%check_interval;}
        let mut end=check_interval.min(max_flips);
        let mut stage_idx=0usize;
        'search: loop{
            if let Some(g)=gate{
                if !apply_c26_gate(g,gate_stats,rounds,best_unsat,&mut max_flips,&mut stage_idx){break 'search;}
            }
            let capacity_steps=(end-rounds).min(capacity_batch_limit);
            residual.reserve(capacity_steps*max_loss_per_flip);
            let mut capacity_end=rounds+capacity_steps;
            if let Some(g)=gate{if rounds<g.flips{capacity_end=capacity_end.min(g.flips);}if rounds<g.early{capacity_end=capacity_end.min(g.early);}}
            if gate.map_or(false,|g|g.early<usize::MAX) && stage_idx<C26_STAGES.len(){capacity_end=capacity_end.min(C26_STAGES[stage_idx].0);}
            let small=residual.len().checked_add(capacity_steps*max_loss_per_flip).map_or(false,|n|n<phase_div::LIMIT);
            let done=if small{if giveup_latched{run_sample_phase::<C,true,true,SINGLE_SAMPLE>(&mut cache,&mut order,&ranges,&mut residual,rng,&roulette,&probsat_weights,nc,all_data,&mut rounds,capacity_end,&mut true_unsat,&mut best_unsat,&mut giveup_latched,&mut short_mask,gate.map_or(false,|g|g.capture),gate_stats,&cl,&co,&mut vars)}else{run_sample_phase::<C,true,false,SINGLE_SAMPLE>(&mut cache,&mut order,&ranges,&mut residual,rng,&roulette,&probsat_weights,nc,all_data,&mut rounds,capacity_end,&mut true_unsat,&mut best_unsat,&mut giveup_latched,&mut short_mask,gate.map_or(false,|g|g.capture),gate_stats,&cl,&co,&mut vars)}}else{if giveup_latched{run_sample_phase::<C,false,true,SINGLE_SAMPLE>(&mut cache,&mut order,&ranges,&mut residual,rng,&roulette,&probsat_weights,nc,all_data,&mut rounds,capacity_end,&mut true_unsat,&mut best_unsat,&mut giveup_latched,&mut short_mask,gate.map_or(false,|g|g.capture),gate_stats,&cl,&co,&mut vars)}else{run_sample_phase::<C,false,false,SINGLE_SAMPLE>(&mut cache,&mut order,&ranges,&mut residual,rng,&roulette,&probsat_weights,nc,all_data,&mut rounds,capacity_end,&mut true_unsat,&mut best_unsat,&mut giveup_latched,&mut short_mask,gate.map_or(false,|g|g.capture),gate_stats,&cl,&co,&mut vars)}};
            if done{break 'search;}
            // A capacity-only boundary is not a search-control event. Continue
            // the same phase without changing noise, RNG, ages, or budgets.
            if rounds<end {continue;}

            if let Some(g)=gate{if !apply_c26_gate(g,gate_stats,rounds,best_unsat,&mut max_flips,&mut stage_idx){break 'search;}}
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
                if progress_ratio > 0.2 && !gate_stats.extended {
                    let increase = (base_max_flips / 100).max(1);
                    max_flips = (max_flips + increase).min(2 * base_max_flips);
                    if let Some(g)=gate{max_flips=max_flips.min(g.hard);}
                }

            
            end=rounds.saturating_add(check_interval).min(max_flips);
        }
    }

    cache.restore_variables(&cl,&co,&mut vars);
    gate_stats.flips=rounds;
    let _ = save_solution(&Solution { variables: vars });
    Ok(best_unsat)
}
pub(super) fn solve_c26_gated(
 rng:&mut SmallRng,nv:usize,nc:usize,density:f64,p_cnt:Vec<u32>,n_cnt:Vec<u32>,all_off:&[u32],p_bound:&[u32],all_data:&[u32],cl:&mut Vec<i32>,co:&[u32],save:&dyn Fn(&Solution)->Result<()>,extend:bool,
)->Result<GateStats>{
 assert_eq!(nv,10000);let flips=fuel_flips(nv,nc,cl.len(),6_000_000_000.);let saved_rng=rng.clone();let saved_cl=cl.clone();
 let hp=Some(Hyperparameters{base_prob:None,max_prob:None,check_interval:None,stagnation_limit:None,perturbation_flips:None,max_fuel_high:Some(6_000_000_000.),max_fuel_low:None});
 let mut stats=GateStats{gate_flips:flips,..GateStats::default()};
 let best=solve_c26_impl_gated::<Byte14BreakCache,false>(&hp,rng,nv,nc,density,&p_cnt,&n_cnt,all_off,p_bound,all_data,cl,co,save,false,Some(ContinuationGate{flips,best:9,hard:if extend{95_000_000}else{2*flips},early:12_000_000,early_best:18,extend,capture:extend}),&mut stats)?;
 if best>0&&best<=RESCUE_MAX_UNSAT{
  let mut rr=saved_rng;let mut cc=saved_cl;let mut rescue=GateStats::default();
  solve_c26_impl_gated::<Byte14BreakCache,true>(&hp,&mut rr,nv,nc,density,&p_cnt,&n_cnt,all_off,p_bound,all_data,&mut cc,co,save,true,Some(ContinuationGate{flips:usize::MAX,best:usize::MAX,hard:flips,early:usize::MAX,early_best:usize::MAX,extend:false,capture:extend}),&mut rescue)?;
  stats.rescue_flips=rescue.flips;
  if !rescue.near_x.is_empty()&&(stats.near_x.is_empty()||rescue.near_u<stats.near_u){stats.near_u=rescue.near_u;stats.near_x=rescue.near_x;}
 }Ok(stats)
}

const C29_STAGES:[(usize,usize);8]=[(1_000_000,112),(2_000_000,83),(4_000_000,77),(6_000_000,42),(8_000_000,30),(10_000_000,27),(12_000_000,18),(16_000_000,12)];
fn apply_c29_gate(g:ContinuationGate,st:&mut GateStats,rounds:usize,best:usize,max:&mut usize,stage:&mut usize)->bool{
 let stages=&C29_STAGES;
 if g.early==usize::MAX{return apply_c15_gate(g,st,rounds,best,max);}
 while *stage<stages.len() && rounds>=stages[*stage].0{let (at,lim)=stages[*stage];*stage+=1;if at==g.early{st.early_best=Some(best);}if best>lim{return false;}}
 if rounds>=g.flips && st.gate_best.is_none(){st.gate_best=Some(best);if g.extend && best<=g.best{st.extended=true;*max=g.hard;}else{return false;}}
 true
}
fn solve_c29_impl_gated<C:CacheOps,const SINGLE_SAMPLE:bool>(
    hp:&Option<Hyperparameters>,rng:&mut SmallRng,nv:usize,nc:usize,density:f64,p_cnt:&[u32],n_cnt:&[u32],all_off:&[u32],p_bound:&[u32],all_data:&[u32],cl:&mut Vec<i32>,co:&[u32],save_solution:&dyn Fn(&Solution)->Result<()>,majority_init:bool,gate:Option<ContinuationGate>,gate_stats:&mut GateStats,
) -> Result<usize> {
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
    if let Some(g)=gate{max_flips=max_flips.min(g.hard);}

    let mut vars = vec![false; nv];
    if majority_init {
        for v in 0..nv {
            if p_cnt[v] > n_cnt[v] { vars[v] = true; }
            else if p_cnt[v] < n_cnt[v] { vars[v] = false; }
            else { vars[v] = rng.gen_bool(0.5); }
        }
    } else {
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
        return Ok(0);
    }

    if gate.map_or(false,|g|g.capture)&&true_unsat>0&&true_unsat<=3{gate_stats.near_u=true_unsat;gate_stats.near_x=vars.clone();}
    let mut best_unsat = true_unsat;
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
        for i in 0..=nc.min(1080) {
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

    let paged_data=super::donor_kernel::large_buffer::from_slice(all_data);
    let all_data=&*paged_data;
    let mut order = ClauseOrder::new(&cl, &co);
    let ranges=LiteralRanges::new(all_off,p_bound);
    let mut cache = C::new(nv, &cl, &co, &vars);
    let mut short_mask=ShortMask::new(nv,&cl,&co,&vars);
    let mut next_check_interval = check_interval;
    unsafe {
        if check_interval==0 && max_flips>0{let _=rounds%check_interval;}
        let mut end=check_interval.min(max_flips);
        let mut stage_idx=0usize;
        'search: loop{
            if let Some(g)=gate{
                if !apply_c29_gate(g,gate_stats,rounds,best_unsat,&mut max_flips,&mut stage_idx){break 'search;}
            }
            let capacity_steps=(end-rounds).min(capacity_batch_limit);
            residual.reserve(capacity_steps*max_loss_per_flip);
            let mut capacity_end=rounds+capacity_steps;
            if let Some(g)=gate{if rounds<g.flips{capacity_end=capacity_end.min(g.flips);}if rounds<g.early{capacity_end=capacity_end.min(g.early);}}
            if gate.map_or(false,|g|g.early<usize::MAX) && stage_idx<8{capacity_end=capacity_end.min(C29_STAGES[stage_idx].0);}
            let small=residual.len().checked_add(capacity_steps*max_loss_per_flip).map_or(false,|n|n<phase_div::LIMIT);
            let done=if small{if giveup_latched{run_sample_phase::<C,true,true,SINGLE_SAMPLE>(&mut cache,&mut order,&ranges,&mut residual,rng,&roulette,&probsat_weights,nc,all_data,&mut rounds,capacity_end,&mut true_unsat,&mut best_unsat,&mut giveup_latched,&mut short_mask,gate.map_or(false,|g|g.capture),gate_stats,&cl,&co,&mut vars)}else{run_sample_phase::<C,true,false,SINGLE_SAMPLE>(&mut cache,&mut order,&ranges,&mut residual,rng,&roulette,&probsat_weights,nc,all_data,&mut rounds,capacity_end,&mut true_unsat,&mut best_unsat,&mut giveup_latched,&mut short_mask,gate.map_or(false,|g|g.capture),gate_stats,&cl,&co,&mut vars)}}else{if giveup_latched{run_sample_phase::<C,false,true,SINGLE_SAMPLE>(&mut cache,&mut order,&ranges,&mut residual,rng,&roulette,&probsat_weights,nc,all_data,&mut rounds,capacity_end,&mut true_unsat,&mut best_unsat,&mut giveup_latched,&mut short_mask,gate.map_or(false,|g|g.capture),gate_stats,&cl,&co,&mut vars)}else{run_sample_phase::<C,false,false,SINGLE_SAMPLE>(&mut cache,&mut order,&ranges,&mut residual,rng,&roulette,&probsat_weights,nc,all_data,&mut rounds,capacity_end,&mut true_unsat,&mut best_unsat,&mut giveup_latched,&mut short_mask,gate.map_or(false,|g|g.capture),gate_stats,&cl,&co,&mut vars)}};
            if done{break 'search;}
            // A capacity-only boundary is not a search-control event. Continue
            // the same phase without changing noise, RNG, ages, or budgets.
            if rounds<end {continue;}

            if let Some(g)=gate{if !apply_c29_gate(g,gate_stats,rounds,best_unsat,&mut max_flips,&mut stage_idx){break 'search;}}
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
                if progress_ratio > 0.2 && !gate_stats.extended {
                    let increase = (base_max_flips / 100).max(1);
                    max_flips = (max_flips + increase).min(2 * base_max_flips);
                    if let Some(g)=gate{max_flips=max_flips.min(g.hard);}
                }

            
            end=rounds.saturating_add(check_interval).min(max_flips);
        }
    }

    cache.restore_variables(&cl,&co,&mut vars);
    gate_stats.flips=rounds;
    let _ = save_solution(&Solution { variables: vars });
    Ok(best_unsat)
}
pub(super) fn solve_c29_gated(
 rng:&mut SmallRng,nv:usize,nc:usize,density:f64,p_cnt:Vec<u32>,n_cnt:Vec<u32>,all_off:&[u32],p_bound:&[u32],all_data:&[u32],cl:&mut Vec<i32>,co:&[u32],save:&dyn Fn(&Solution)->Result<()>,extend:bool,
)->Result<GateStats>{
 assert_eq!(nv,10000);let flips=fuel_flips(nv,nc,cl.len(),6_000_000_000.);let saved_rng=rng.clone();let saved_cl=cl.clone();
 let hp=Some(Hyperparameters{base_prob:None,max_prob:None,check_interval:None,stagnation_limit:None,perturbation_flips:None,max_fuel_high:Some(6_000_000_000.),max_fuel_low:None});
 let mut stats=GateStats{gate_flips:flips,..GateStats::default()};
 let best=solve_c29_impl_gated::<Byte14BreakCache,false>(&hp,rng,nv,nc,density,&p_cnt,&n_cnt,all_off,p_bound,all_data,cl,co,save,false,Some(ContinuationGate{flips,best:9,hard:if extend{95_000_000}else{2*flips},early:12_000_000,early_best:18,extend,capture:extend}),&mut stats)?;
 if best>0&&best<=RESCUE_MAX_UNSAT{
  let mut rr=saved_rng;let mut cc=saved_cl;let mut rescue=GateStats::default();
  solve_c29_impl_gated::<Byte14BreakCache,true>(&hp,&mut rr,nv,nc,density,&p_cnt,&n_cnt,all_off,p_bound,all_data,&mut cc,co,save,true,Some(ContinuationGate{flips:usize::MAX,best:usize::MAX,hard:flips,early:usize::MAX,early_best:usize::MAX,extend:false,capture:extend}),&mut rescue)?;
  stats.rescue_flips=rescue.flips;
  if !rescue.near_x.is_empty()&&(stats.near_x.is_empty()||rescue.near_u<stats.near_u){stats.near_u=rescue.near_u;stats.near_x=rescue.near_x;}
 }Ok(stats)
}
#[cfg(test)]mod c29_gate_b_tests{use super::*;
 #[test]fn milestones_and_budget_reject_only_at_boundaries(){
  let g=ContinuationGate{early:12_000_000,early_best:18,flips:29_430_000,best:9,hard:95_000_000,extend:true,capture:true};
  for i in 0..C29_STAGES.len(){let(at,lim)=C29_STAGES[i];let mut s=GateStats::default();let mut max=g.flips;let mut stage=i;assert!(apply_c29_gate(g,&mut s,at-1,lim+1,&mut max,&mut stage));assert!(!apply_c29_gate(g,&mut s,at,lim+1,&mut max,&mut stage));}
  let mut s=GateStats::default();let mut stage=8;let mut max=g.flips;assert!(!apply_c29_gate(g,&mut s,g.flips,10,&mut max,&mut stage));assert_eq!(s.gate_best,Some(10));
  let mut s=GateStats::default();let mut stage=8;let mut max=g.flips;assert!(apply_c29_gate(g,&mut s,g.flips,9,&mut max,&mut stage));assert!(s.extended);assert_eq!(max,g.hard);
 }
 #[test]fn passing_trajectory_matches_original_rng_flips_and_assignment(){
  use rand::{RngCore,SeedableRng};use std::cell::RefCell;
  let nv=4usize;let nc=8usize;let clauses:Vec<Vec<i32>>=(0..8).map(|m|(0..3).map(|v|if m&(1<<v)==0{v+1}else{-(v+1)}).collect()).collect();
  let(mut pc,mut nn)=(vec![0u32;nv],vec![0u32;nv]);for cl in &clauses{for &l in cl{let v=(l.abs()-1)as usize;if l>0{pc[v]+=1}else{nn[v]+=1;}}}
  let mut off=vec![0u32;nv+1];for v in 0..nv{off[v+1]=off[v]+pc[v]+nn[v];}let mut bound=vec![0u32;nv];let(mut pp,mut np)=(vec![0u32;nv],vec![0u32;nv]);for v in 0..nv{pp[v]=off[v];np[v]=off[v]+pc[v];bound[v]=np[v];}
  let mut data=vec![0u32;off[nv]as usize];let(mut cl,mut co)=(Vec::new(),vec![0u32]);for(cid,c)in clauses.iter().enumerate(){for &l in c{let v=(l.abs()-1)as usize;let ix=if l>0{&mut pp[v]}else{&mut np[v]};data[*ix as usize]=cid as u32;*ix+=1;cl.push(l);}co.push(cl.len()as u32);}
  let hp=Some(Hyperparameters{base_prob:None,max_prob:None,check_interval:None,stagnation_limit:None,perturbation_flips:None,max_fuel_high:Some(2_000_000.0),max_fuel_low:None});let g=ContinuationGate{early:12_000_000,early_best:18,flips:usize::MAX,best:9,hard:usize::MAX,extend:true,capture:true};
  let(mut a,mut b)=(SmallRng::seed_from_u64(51),SmallRng::seed_from_u64(51));let(mut x,mut y)=(GateStats::default(),GateStats::default());let(mut cl2,sa,sb)=(cl.clone(),RefCell::new(Vec::new()),RefCell::new(Vec::new()));
  let one=solve_impl_gated::<Byte14BreakCache,false>(&hp,&mut a,nv,nc,2.0,&pc,&nn,&off,&bound,&data,&mut cl,&co,&|s|{sa.borrow_mut().push(s.variables.clone());Ok(())},false,Some(g),&mut x).unwrap();
  let two=solve_c29_impl_gated::<Byte14BreakCache,false>(&hp,&mut b,nv,nc,2.0,&pc,&nn,&off,&bound,&data,&mut cl2,&co,&|s|{sb.borrow_mut().push(s.variables.clone());Ok(())},false,Some(g),&mut y).unwrap();
  assert_eq!((one,x.flips,x.near_u,x.near_x,a.next_u64()),(two,y.flips,y.near_u,y.near_x,b.next_u64()));assert_eq!(*sa.borrow(),*sb.borrow());
 }
}

#[cfg(target_arch="x86_64")]
#[target_feature(enable="avx2,bmi1,bmi2,lzcnt,popcnt")]
unsafe fn run_sample_phase_avx2<C:CacheOps,const SMALL:bool,const LATCHED:bool,const SINGLE_SAMPLE:bool>(cache:&mut C,order:&mut ClauseOrder,ranges:&LiteralRanges,residual:&mut Vec<u32>,rng:&mut SmallRng,roulette:&RawRoulette,probsat_weights:&[f64],nc:usize,all_data:&[u32],rounds:&mut usize,end:usize,true_unsat:&mut usize,best_unsat:&mut usize,giveup_latched:&mut bool,short_mask:&mut ShortMask,capture:bool,near:&mut GateStats,cl:&[i32],co:&[u32],vars:&mut Vec<bool>)->bool{
 while *rounds<end{

            if *true_unsat==0{return true;}

            let cid=if short_mask.all_satisfied(){
            let mut cid=0usize;
            // Keep the primary trajectory; use one sample only in majority rescue.
            for sample in 0..if SINGLE_SAMPLE {1} else {3}{
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
            
            if capture&&*true_unsat>0&&*true_unsat<=3&&*true_unsat<*best_unsat{cache.restore_variables(cl,co,vars);near.near_u=*true_unsat;near.near_x=vars.clone();}
            *best_unsat = (*best_unsat).min(*true_unsat);
            if !LATCHED{*giveup_latched |= *true_unsat <= GIVEUP_LEVEL;}

            *rounds += 1;

            
 }
 false
}
unsafe fn run_sample_phase_dispatch<C:CacheOps,const SMALL:bool,const LATCHED:bool,const SINGLE_SAMPLE:bool>(cache:&mut C,order:&mut ClauseOrder,ranges:&LiteralRanges,residual:&mut Vec<u32>,rng:&mut SmallRng,roulette:&RawRoulette,probsat_weights:&[f64],nc:usize,all_data:&[u32],rounds:&mut usize,end:usize,true_unsat:&mut usize,best_unsat:&mut usize,giveup_latched:&mut bool,short_mask:&mut ShortMask,capture:bool,near:&mut GateStats,cl:&[i32],co:&[u32],vars:&mut Vec<bool>)->bool{
 #[cfg(target_arch="x86_64")]
 if std::arch::is_x86_feature_detected!("avx2")&&std::arch::is_x86_feature_detected!("bmi1")&&std::arch::is_x86_feature_detected!("bmi2")&&std::arch::is_x86_feature_detected!("lzcnt")&&std::arch::is_x86_feature_detected!("popcnt"){return run_sample_phase_avx2::<C,SMALL,LATCHED,SINGLE_SAMPLE>(cache,order,ranges,residual,rng,roulette,probsat_weights,nc,all_data,rounds,end,true_unsat,best_unsat,giveup_latched,short_mask,capture,near,cl,co,vars);}
 run_sample_phase::<C,SMALL,LATCHED,SINGLE_SAMPLE>(cache,order,ranges,residual,rng,roulette,probsat_weights,nc,all_data,rounds,end,true_unsat,best_unsat,giveup_latched,short_mask,capture,near,cl,co,vars)
}
const C32_STAGES:[(usize,usize);10]=[(1_000_000,106),(2_000_000,79),(3_000_000,79),(4_000_000,73),(6_000_000,39),(8_000_000,28),(10_000_000,25),(12_000_000,18),(16_000_000,10),(20_000_000,9)];
fn apply_c32_gate(g:ContinuationGate,st:&mut GateStats,rounds:usize,best:usize,max:&mut usize,stage:&mut usize)->bool{
 let stages=&C32_STAGES;
 if g.early==usize::MAX{return apply_c15_gate(g,st,rounds,best,max);}
 while *stage<stages.len() && rounds>=stages[*stage].0{let (at,lim)=stages[*stage];*stage+=1;if at==g.early{st.early_best=Some(best);}if best>lim{return false;}}
 if rounds>=g.flips && st.gate_best.is_none(){st.gate_best=Some(best);if g.extend && best<=g.best{st.extended=true;*max=g.hard;}else{return false;}}
 true
}
fn solve_c32_impl_gated<C:CacheOps,const SINGLE_SAMPLE:bool>(
    hp:&Option<Hyperparameters>,rng:&mut SmallRng,nv:usize,nc:usize,density:f64,p_cnt:&[u32],n_cnt:&[u32],all_off:&[u32],p_bound:&[u32],all_data:&[u32],cl:&mut Vec<i32>,co:&[u32],save_solution:&dyn Fn(&Solution)->Result<()>,majority_init:bool,gate:Option<ContinuationGate>,gate_stats:&mut GateStats,
) -> Result<usize> {
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
    if let Some(g)=gate{max_flips=max_flips.min(g.hard);}

    let mut vars = vec![false; nv];
    if majority_init {
        for v in 0..nv {
            if p_cnt[v] > n_cnt[v] { vars[v] = true; }
            else if p_cnt[v] < n_cnt[v] { vars[v] = false; }
            else { vars[v] = rng.gen_bool(0.5); }
        }
    } else {
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
        return Ok(0);
    }

    if gate.map_or(false,|g|g.capture)&&true_unsat>0&&true_unsat<=3{gate_stats.near_u=true_unsat;gate_stats.near_x=vars.clone();}
    let mut best_unsat = true_unsat;
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
        for i in 0..=nc.min(1080) {
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

    let paged_data=super::donor_kernel::large_buffer::from_slice(all_data);
    let all_data=&*paged_data;
    let mut order = ClauseOrder::new(&cl, &co);
    let ranges=LiteralRanges::new(all_off,p_bound);
    let mut cache = C::new(nv, &cl, &co, &vars);
    let mut short_mask=ShortMask::new(nv,&cl,&co,&vars);
    let mut next_check_interval = check_interval;
    unsafe {
        if check_interval==0 && max_flips>0{let _=rounds%check_interval;}
        let mut end=check_interval.min(max_flips);
        let mut stage_idx=0usize;
        'search: loop{
            if let Some(g)=gate{
                if !apply_c32_gate(g,gate_stats,rounds,best_unsat,&mut max_flips,&mut stage_idx){break 'search;}
            }
            let capacity_steps=(end-rounds).min(capacity_batch_limit);
            residual.reserve(capacity_steps*max_loss_per_flip);
            let mut capacity_end=rounds+capacity_steps;
            if let Some(g)=gate{if rounds<g.flips{capacity_end=capacity_end.min(g.flips);}if rounds<g.early{capacity_end=capacity_end.min(g.early);}}
            if gate.map_or(false,|g|g.early<usize::MAX) && stage_idx<10{capacity_end=capacity_end.min(C32_STAGES[stage_idx].0);}
            let small=residual.len().checked_add(capacity_steps*max_loss_per_flip).map_or(false,|n|n<phase_div::LIMIT);
            let done=if small{if giveup_latched{run_sample_phase_dispatch::<C,true,true,SINGLE_SAMPLE>(&mut cache,&mut order,&ranges,&mut residual,rng,&roulette,&probsat_weights,nc,all_data,&mut rounds,capacity_end,&mut true_unsat,&mut best_unsat,&mut giveup_latched,&mut short_mask,gate.map_or(false,|g|g.capture),gate_stats,&cl,&co,&mut vars)}else{run_sample_phase_dispatch::<C,true,false,SINGLE_SAMPLE>(&mut cache,&mut order,&ranges,&mut residual,rng,&roulette,&probsat_weights,nc,all_data,&mut rounds,capacity_end,&mut true_unsat,&mut best_unsat,&mut giveup_latched,&mut short_mask,gate.map_or(false,|g|g.capture),gate_stats,&cl,&co,&mut vars)}}else{if giveup_latched{run_sample_phase_dispatch::<C,false,true,SINGLE_SAMPLE>(&mut cache,&mut order,&ranges,&mut residual,rng,&roulette,&probsat_weights,nc,all_data,&mut rounds,capacity_end,&mut true_unsat,&mut best_unsat,&mut giveup_latched,&mut short_mask,gate.map_or(false,|g|g.capture),gate_stats,&cl,&co,&mut vars)}else{run_sample_phase_dispatch::<C,false,false,SINGLE_SAMPLE>(&mut cache,&mut order,&ranges,&mut residual,rng,&roulette,&probsat_weights,nc,all_data,&mut rounds,capacity_end,&mut true_unsat,&mut best_unsat,&mut giveup_latched,&mut short_mask,gate.map_or(false,|g|g.capture),gate_stats,&cl,&co,&mut vars)}};
            if done{break 'search;}
            // A capacity-only boundary is not a search-control event. Continue
            // the same phase without changing noise, RNG, ages, or budgets.
            if rounds<end {continue;}

            if let Some(g)=gate{if !apply_c32_gate(g,gate_stats,rounds,best_unsat,&mut max_flips,&mut stage_idx){break 'search;}}
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
                if progress_ratio > 0.2 && !gate_stats.extended {
                    let increase = (base_max_flips / 100).max(1);
                    max_flips = (max_flips + increase).min(2 * base_max_flips);
                    if let Some(g)=gate{max_flips=max_flips.min(g.hard);}
                }

            
            end=rounds.saturating_add(check_interval).min(max_flips);
        }
    }

    cache.restore_variables(&cl,&co,&mut vars);
    gate_stats.flips=rounds;
    let _ = save_solution(&Solution { variables: vars });
    Ok(best_unsat)
}
pub(super) fn solve_c32_gated(
 rng:&mut SmallRng,nv:usize,nc:usize,density:f64,p_cnt:Vec<u32>,n_cnt:Vec<u32>,all_off:&[u32],p_bound:&[u32],all_data:&[u32],cl:&mut Vec<i32>,co:&[u32],save:&dyn Fn(&Solution)->Result<()>,extend:bool,
)->Result<GateStats>{
 assert_eq!(nv,10000);let flips=fuel_flips(nv,nc,cl.len(),6_000_000_000.);let saved_rng=rng.clone();let saved_cl=cl.clone();
 let hp=Some(Hyperparameters{base_prob:None,max_prob:None,check_interval:None,stagnation_limit:None,perturbation_flips:None,max_fuel_high:Some(6_000_000_000.),max_fuel_low:None});
 let mut stats=GateStats{gate_flips:flips,..GateStats::default()};
 let best=solve_c32_impl_gated::<Byte14BreakCache,false>(&hp,rng,nv,nc,density,&p_cnt,&n_cnt,all_off,p_bound,all_data,cl,co,save,false,Some(ContinuationGate{flips,best:9,hard:if extend{95_000_000}else{2*flips},early:12_000_000,early_best:18,extend,capture:extend}),&mut stats)?;
 if best>0&&best<=RESCUE_MAX_UNSAT && !(true && stats.near_u==1 && !stats.near_x.is_empty()){
  let mut rr=saved_rng;let mut cc=saved_cl;let mut rescue=GateStats::default();
  solve_c32_impl_gated::<Byte14BreakCache,true>(&hp,&mut rr,nv,nc,density,&p_cnt,&n_cnt,all_off,p_bound,all_data,&mut cc,co,save,true,Some(ContinuationGate{flips:usize::MAX,best:usize::MAX,hard:flips,early:usize::MAX,early_best:usize::MAX,extend:false,capture:extend}),&mut rescue)?;
  stats.rescue_flips=rescue.flips;
  if !rescue.near_x.is_empty()&&(stats.near_x.is_empty()||rescue.near_u<stats.near_u){stats.near_u=rescue.near_u;stats.near_x=rescue.near_x;}
 }Ok(stats)
}
#[cfg(test)]mod c32_gate_tests{use super::*;
 #[test]fn milestones_and_budget_reject_only_at_boundaries(){
  let g=ContinuationGate{early:12_000_000,early_best:18,flips:29_430_000,best:9,hard:95_000_000,extend:true,capture:true};
  for i in 0..C32_STAGES.len(){let(at,lim)=C32_STAGES[i];let mut s=GateStats::default();let mut max=g.flips;let mut stage=i;assert!(apply_c32_gate(g,&mut s,at-1,lim+1,&mut max,&mut stage));assert!(!apply_c32_gate(g,&mut s,at,lim+1,&mut max,&mut stage));}
  let mut s=GateStats::default();let mut stage=10;let mut max=g.flips;assert!(!apply_c32_gate(g,&mut s,g.flips,10,&mut max,&mut stage));assert_eq!(s.gate_best,Some(10));
  let mut s=GateStats::default();let mut stage=10;let mut max=g.flips;assert!(apply_c32_gate(g,&mut s,g.flips,9,&mut max,&mut stage));assert!(s.extended);assert_eq!(max,g.hard);
 }
 #[test]fn passing_trajectory_matches_original_rng_flips_and_assignment(){
  use rand::{RngCore,SeedableRng};use std::cell::RefCell;
  let nv=4usize;let nc=8usize;let clauses:Vec<Vec<i32>>=(0..8).map(|m|(0..3).map(|v|if m&(1<<v)==0{v+1}else{-(v+1)}).collect()).collect();
  let(mut pc,mut nn)=(vec![0u32;nv],vec![0u32;nv]);for cl in &clauses{for &l in cl{let v=(l.abs()-1)as usize;if l>0{pc[v]+=1}else{nn[v]+=1;}}}
  let mut off=vec![0u32;nv+1];for v in 0..nv{off[v+1]=off[v]+pc[v]+nn[v];}let mut bound=vec![0u32;nv];let(mut pp,mut np)=(vec![0u32;nv],vec![0u32;nv]);for v in 0..nv{pp[v]=off[v];np[v]=off[v]+pc[v];bound[v]=np[v];}
  let mut data=vec![0u32;off[nv]as usize];let(mut cl,mut co)=(Vec::new(),vec![0u32]);for(cid,c)in clauses.iter().enumerate(){for &l in c{let v=(l.abs()-1)as usize;let ix=if l>0{&mut pp[v]}else{&mut np[v]};data[*ix as usize]=cid as u32;*ix+=1;cl.push(l);}co.push(cl.len()as u32);}
  let hp=Some(Hyperparameters{base_prob:None,max_prob:None,check_interval:None,stagnation_limit:None,perturbation_flips:None,max_fuel_high:Some(2_000_000.0),max_fuel_low:None});let g=ContinuationGate{early:12_000_000,early_best:18,flips:usize::MAX,best:9,hard:usize::MAX,extend:true,capture:true};
  let(mut a,mut b)=(SmallRng::seed_from_u64(51),SmallRng::seed_from_u64(51));let(mut x,mut y)=(GateStats::default(),GateStats::default());let(mut cl2,sa,sb)=(cl.clone(),RefCell::new(Vec::new()),RefCell::new(Vec::new()));
  let one=solve_impl_gated::<Byte14BreakCache,false>(&hp,&mut a,nv,nc,2.0,&pc,&nn,&off,&bound,&data,&mut cl,&co,&|s|{sa.borrow_mut().push(s.variables.clone());Ok(())},false,Some(g),&mut x).unwrap();
  let two=solve_c32_impl_gated::<Byte14BreakCache,false>(&hp,&mut b,nv,nc,2.0,&pc,&nn,&off,&bound,&data,&mut cl2,&co,&|s|{sb.borrow_mut().push(s.variables.clone());Ok(())},false,Some(g),&mut y).unwrap();
  assert_eq!((one,x.flips,x.near_u,x.near_x,a.next_u64()),(two,y.flips,y.near_u,y.near_x,b.next_u64()));assert_eq!(*sa.borrow(),*sb.borrow());
 }
}
