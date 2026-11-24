// v6 Track2 lineage with a112 give-up reference, a125 cache helpers and v7-family controller/rescue changes.
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
use super::donor_kernel::phase_div;
use super::donor_kernel::exact_coin::ExactCoin;
// Retained-clause degree<=nc<=32002 on this track, so u16 appearances
// preserve every value. Age is only observed through min(age/2,50); u8 and
// u16 saturation are indistinguishable after age 100 and reset identically.
use super::donor_kernel::paged_flip_ranges::FlipRanges;
// Best snapshots are copied only on the baseline's strict-improvement events.
// best_unsat strictly decreases globally, so the total number of copies is
// bounded by its initial value. No per-flip history is necessary: the saved
// snapshot is not read by move selection within an attempt.
use super::donor_kernel::paged_small_order::ClauseOrder;
use super::donor_kernel::exact_div;
use super::donor_kernel::paged_cache::{SmallBreakCache,Byte13BreakCache,CacheOps};
use anyhow::Result;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use tig_challenges::satisfiability::*;
use super::Hyperparameters;

// Capture only at the first original segment boundary. The live baseline
// continues from the same state; a failed best-unsat==1 case may later resume
// the independent fixed-restart branch without replaying its common prefix.
struct RestartCheckpoint {
    rng: SmallRng,
    best_vars: Vec<bool>,
    best_unsat: usize,
    order: ClauseOrder,
    consumed_rounds: usize,
    giveup_latched: bool,
}
const ENABLE_RESCUE: bool = true;
const RESCUE_MAX_UNSAT: usize = usize::MAX;


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
        solve_impl::<Byte13BreakCache,true,false>(challenge,hp,save_solution,None)
    } else {solve_impl::<SmallBreakCache,false,false>(challenge,hp,save_solution,None)}
}
fn solve_impl<C:CacheOps,const COMPACT:bool,const WEIGHTED_RESCUE:bool>(
    challenge: &Challenge,
    hp: &Option<Hyperparameters>,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    resume: Option<RestartCheckpoint>,
) -> Result<()> {
    solve_policy::<C,COMPACT,WEIGHTED_RESCUE,{usize::MAX},{usize::MAX},false>(challenge,hp,save_solution,resume,&mut GateStats::default())
}
#[derive(Default,Debug,Clone,PartialEq,Eq)]
pub struct GateStats {pub attempts:Vec<(usize,usize,usize,u64)>,pub gated:bool}
/// Only registered opt-in gates are instantiated. Constants add no live gate
/// values to the search kernel; diagnostics clone RNG only at attempt boundaries.
pub fn solve_gated(ch:&Challenge,hp:&Option<Hyperparameters>,save:&dyn Fn(&Solution)->Result<()>,gate:Option<(usize,usize)>)->Result<()> {
    match gate {None=>solve(ch,hp,save),Some((0,16))=>gated::<0,16,false>(ch,hp,save,&mut GateStats::default()),Some((1,16))=>gated::<1,16,false>(ch,hp,save,&mut GateStats::default()),Some((1,18))=>gated::<1,18,false>(ch,hp,save,&mut GateStats::default()),_=>anyhow::bail!("unregistered attempt gate")}
}
pub fn solve_gated_diagnostic(ch:&Challenge,hp:&Option<Hyperparameters>,save:&dyn Fn(&Solution)->Result<()>,gate:Option<(usize,usize)>)->Result<GateStats> {
    let mut st=GateStats::default();match gate {None=>gated::<{usize::MAX},{usize::MAX},true>(ch,hp,save,&mut st),Some((0,16))=>gated::<0,16,true>(ch,hp,save,&mut st),Some((1,16))=>gated::<1,16,true>(ch,hp,save,&mut st),Some((1,18))=>gated::<1,18,true>(ch,hp,save,&mut st),_=>anyhow::bail!("unregistered attempt gate")}?;Ok(st)
}
fn gated<const AFTER:usize,const LIMIT:usize,const TRACE:bool>(ch:&Challenge,hp:&Option<Hyperparameters>,save:&dyn Fn(&Solution)->Result<()>,stats:&mut GateStats)->Result<()> {
    let nv=ch.num_variables;let mut degree=vec![0u32;nv];for cl in &ch.clauses{for &l in cl{degree[(l.abs()-1)as usize]+=1;}}
    if nv<=8192&&degree.iter().all(|&d|d<=255){solve_policy::<Byte13BreakCache,true,false,AFTER,LIMIT,TRACE>(ch,hp,save,None,stats)}else{solve_policy::<SmallBreakCache,false,false,AFTER,LIMIT,TRACE>(ch,hp,save,None,stats)}
}
fn solve_policy<C:CacheOps,const COMPACT:bool,const WEIGHTED_RESCUE:bool,const AFTER:usize,const LIMIT:usize,const TRACE:bool>(
    challenge:&Challenge,hp:&Option<Hyperparameters>,save_solution:&dyn Fn(&Solution)->Result<()>,resume:Option<RestartCheckpoint>,trace:&mut GateStats,
)->Result<()> {
    let resumed = resume.is_some();
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
    let all_data=super::donor_kernel::large_buffer::from_slice(&all_data);
    let mut giveup_latched = false;
    let mut giveup_abandon = false;
    let mut consumed_rounds = 0usize;
    let mut checkpoint: Option<RestartCheckpoint> = None;
    let start_attempt = if let Some(saved) = resume {
        rng = saved.rng;
        best_vars = saved.best_vars;
        best_unsat = saved.best_unsat;
        record_limit = best_unsat.min(13);
        order = saved.order;
        consumed_rounds = saved.consumed_rounds;
        giveup_latched = saved.giveup_latched;
        1
    } else { 0 };
    for attempt in start_attempt..restart_attempts {
        if consumed_rounds >= max_flips { break; }
        let mut attempt_budget = if restart_attempts == 1 {
            max_flips
        } else {
            max_flips / restart_attempts
                + if attempt < (max_flips % restart_attempts) { 1 } else { 0 }
        };
        attempt_budget = attempt_budget.min(max_flips - consumed_rounds);
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
        let mut extension_best = unsat_count;
        // The old loop checks control once before its periodic events. A kick
        // can change unsat at the same round without a second control check.
        // Batching must not accidentally observe that intermediate state.
        let mut observed_round = usize::MAX;

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
            macro_rules! observe_original_round {
                ($exit:lifetime) => {
                    if observed_round != rounds {
                        if unsat_count == 0 { break $exit; }
                        extension_best = extension_best.min(unsat_count);
                        if rounds >= attempt_budget {
                            if resumed { break $exit; }
                            if ENABLE_RESCUE && attempt == 0 && checkpoint.is_none()
                                && extension_best <= 16 && consumed_rounds + rounds < max_flips {
                                let phase = if unsat_count < best_unsat {
                                    ranges.restore_variables(&all_off,&p_bound,&mut vars);
                                    vars.clone()
                                } else { best_vars.clone() };
                                checkpoint = Some(RestartCheckpoint {
                                    rng: rng.clone(), best_vars: phase,
                                    best_unsat: best_unsat.min(unsat_count), order: order.clone(),
                                    consumed_rounds: consumed_rounds + rounds, giveup_latched,
                                });
                            }
                            let remaining = max_flips - consumed_rounds - rounds;
                            if remaining == 0 || extension_best > 16 { break $exit; }
                            attempt_budget += (max_flips / restart_attempts).max(1).min(remaining);
                            extension_best = unsat_count;
                        }
                        if !giveup_latched {
                            if unsat_count <= 16 { giveup_latched = true; }
                            else if rounds >= 67_500_000 {
                                giveup_abandon = true;
                                break $exit;
                            }
                        }
                        observed_round = rounds;
                    }
                }
            }
            'attempt_search: loop{
                observe_original_round!('attempt_search);
                // Every budget/deadline event is a batch boundary. Inside the
                // phase, only unsat minima and the monotone latch can change.
                let mut event_end=next_check_interval.min(next_variance_interval).min(attempt_budget);
                if !giveup_latched { event_end=event_end.min(67_500_000); }
                let phase_start=rounds;
                let steps=(event_end-rounds).min(capacity_batch_limit);
                residual.reserve(steps*max_loss_per_flip);
                let batch_end=rounds+steps;
                // Events/kicks are outside this bounded main phase.
                // At most steps*max_loss_per_flip new IDs can be appended.
                let small=residual.len().checked_add(steps*max_loss_per_flip).map_or(false,|n|n<phase_div::LIMIT);
                if small{
                while (rounds<batch_end) & (unsat_count!=0){
                // The boundary controller already observed phase_start, before
                // any same-round kick. Observe only subsequent main-loop entries.
                if rounds!=phase_start {
                    extension_best=extension_best.min(unsat_count);
                    if !giveup_latched && unsat_count<=16 { giveup_latched=true; }
                }
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
            let v_idx=if WEIGHTED_RESCUE {choose_weighted::<C,COMPACT,0>(vv,clen,&cache,&coin,&mut rng,&var_appearances,&var_age)} else {if clen==3{choose_shape::<C,COMPACT,3>(vv,clen,&cache,&coin,&mut rng,&var_appearances,&var_age)}else{choose_short::<C,COMPACT>(vv,clen,&cache,&coin,&mut rng,&var_appearances,&var_age)}};
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
                // The boundary controller already observed phase_start, before
                // any same-round kick. Observe only subsequent main-loop entries.
                if rounds!=phase_start {
                    extension_best=extension_best.min(unsat_count);
                    if !giveup_latched && unsat_count<=16 { giveup_latched=true; }
                }
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
            let v_idx=if WEIGHTED_RESCUE {choose_weighted::<C,COMPACT,0>(vv,clen,&cache,&coin,&mut rng,&var_appearances,&var_age)} else {if clen==3{choose_shape::<C,COMPACT,3>(vv,clen,&cache,&coin,&mut rng,&var_appearances,&var_age)}else{choose_short::<C,COMPACT>(vv,clen,&cache,&coin,&mut rng,&var_appearances,&var_age)}};
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
                observe_original_round!('attempt_search);
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

        consumed_rounds += rounds;
        if unsat_count < best_unsat {
            best_unsat = unsat_count;
                        record_limit=best_unsat.min(13);
            ranges.restore_variables(&all_off,&p_bound,&mut vars);
                    best_vars.copy_from_slice(&vars);
        }
        if TRACE {use rand::RngCore;let mut copy=rng.clone();trace.attempts.push((attempt,best_unsat,consumed_rounds,copy.next_u64()));}
        if unsat_count == 0 {
            ranges.restore_variables(&all_off,&p_bound,&mut vars);
            let _ = save_solution(&Solution { variables: vars });
            return Ok(());
        }
        if !resumed && attempt >= AFTER && best_unsat > LIMIT {if TRACE {trace.gated=true;}break;}
        if giveup_abandon { break; }
    }

    let _ = save_solution(&Solution { variables: best_vars });
    if ENABLE_RESCUE && !resumed && best_unsat > 0 && best_unsat <= RESCUE_MAX_UNSAT {
        if let Some(saved) = checkpoint {
            let save_complete = |solution: &Solution| -> Result<()> {
                if solution.variables.len() == nv && challenge.clauses.iter().all(|clause| {
                    clause.iter().any(|&lit| solution.variables[(lit.abs()-1) as usize] == (lit>0))
                }) { save_solution(solution)?; }
                Ok(())
            };
            solve_impl::<C,COMPACT,true>(challenge,hp,&save_complete,Some(saved))?;
        }
    }
    Ok(())
}

#[cfg(test)]mod c18_gate_tests {
 use super::*;use std::cell::RefCell;
 fn contradiction()->Challenge{let mut clauses=Vec::new();for g in 0..32{for m in 0..8{clauses.push((0..3).map(|v|{let l=(3*g+v+1)as i32;if m&(1<<v)==0{l}else{-l}}).collect());}}Challenge{seed:[7;32],num_variables:128,clauses}}
 fn hp()->Option<Hyperparameters>{Some(Hyperparameters{base_prob:None,max_prob:None,check_interval:None,stagnation_limit:None,perturbation_flips:None,max_fuel_high:Some(50_000_000.0),max_fuel_low:None})}
 #[test]fn opt_in_none_preserves_all_default_callbacks(){let ch=contradiction();let a=RefCell::new(Vec::new());let b=RefCell::new(Vec::new());solve(&ch,&hp(),&|s|{a.borrow_mut().push(s.variables.clone());Ok(())}).unwrap();let st=solve_gated_diagnostic(&ch,&hp(),&|s|{b.borrow_mut().push(s.variables.clone());Ok(())},None).unwrap();assert_eq!(*a.borrow(),*b.borrow());assert!(!st.gated);assert_eq!(st.attempts.len(),4);}
 #[test]fn first_attempt_gate_preserves_exact_rng_and_flip_prefix(){let ch=contradiction();let full=solve_gated_diagnostic(&ch,&hp(),&|_|Ok(()),None).unwrap();let cut=solve_gated_diagnostic(&ch,&hp(),&|_|Ok(()),Some((0,16))).unwrap();assert!(cut.gated);assert_eq!(cut.attempts,full.attempts[..1]);assert!(cut.attempts[0].1>16);}
 #[test]fn second_attempt_gates_preserve_exact_rng_and_flip_prefix(){let ch=contradiction();let full=solve_gated_diagnostic(&ch,&hp(),&|_|Ok(()),None).unwrap();for limit in[16,18]{let cut=solve_gated_diagnostic(&ch,&hp(),&|_|Ok(()),Some((1,limit))).unwrap();assert!(cut.gated);assert_eq!(cut.attempts,full.attempts[..2]);assert!(cut.attempts[1].1>limit);}}
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
#[inline(always)]
unsafe fn choose_weighted<C:CacheOps,const COMPACT:bool,const LENGTH:usize>(vv:[usize;3],length:usize,cache:&C,coin:&ExactCoin,rng:&mut SmallRng,var_appearances:&[u16],var_age:&[u8])->usize{
 // Development: positive break values select by probability, not hard ranking.
 let clen=if LENGTH==0{length}else{LENGTH};
 if clen<=1{return vv[0];}
 let b0=cache.breaks_of(vv[0]);
 let b1=cache.breaks_of(vv[1]);
 let b2=if clen>2{cache.breaks_of(vv[2])}else{1};
 if b0==0{return vv[0];} if b1==0{return vv[1];} if clen>2 && b2==0{return vv[2];}
 let word=rng.gen::<u32>();
 if clen==3{
   vv[super::track1_cached::explicit_roulette::integer_three(b0,b1,b2,word)]
 }else{
   const W:[u32;16]=[2535,551,233,127,80,55,41,30,24,19,16,13,11,9,8,7];
   let w0=W[b0.min(15)];let w1=W[b1.min(15)];
   if word%(w0+w1)<w0{vv[0]}else{vv[1]}
 }
}

#[cold]#[inline(never)]
unsafe fn choose_short<C:CacheOps,const COMPACT:bool>(vv:[usize;3],length:usize,cache:&C,coin:&ExactCoin,rng:&mut SmallRng,var_appearances:&[u16],var_age:&[u8])->usize{
 choose_shape::<C,COMPACT,0>(vv,length,cache,coin,rng,var_appearances,var_age)
}







fn solve_policy2<C:CacheOps,const COMPACT:bool,const WEIGHTED_RESCUE:bool,const LIMIT0:usize,const AFTER:usize,const LIMIT:usize,const TRACE:bool>(
    challenge:&Challenge,hp:&Option<Hyperparameters>,save_solution:&dyn Fn(&Solution)->Result<()>,resume:Option<RestartCheckpoint>,trace:&mut GateStats,
)->Result<()> {
    let resumed = resume.is_some();
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
    let all_data=super::donor_kernel::large_buffer::from_slice(&all_data);
    let mut giveup_latched = false;
    let mut giveup_abandon = false;
    let mut consumed_rounds = 0usize;
    let mut checkpoint: Option<RestartCheckpoint> = None;
    let start_attempt = if let Some(saved) = resume {
        rng = saved.rng;
        best_vars = saved.best_vars;
        best_unsat = saved.best_unsat;
        record_limit = best_unsat.min(13);
        order = saved.order;
        consumed_rounds = saved.consumed_rounds;
        giveup_latched = saved.giveup_latched;
        1
    } else { 0 };
    for attempt in start_attempt..restart_attempts {
        if consumed_rounds >= max_flips { break; }
        let mut attempt_budget = if restart_attempts == 1 {
            max_flips
        } else {
            max_flips / restart_attempts
                + if attempt < (max_flips % restart_attempts) { 1 } else { 0 }
        };
        attempt_budget = attempt_budget.min(max_flips - consumed_rounds);
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
        let mut extension_best = unsat_count;
        // The old loop checks control once before its periodic events. A kick
        // can change unsat at the same round without a second control check.
        // Batching must not accidentally observe that intermediate state.
        let mut observed_round = usize::MAX;

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
            macro_rules! observe_original_round {
                ($exit:lifetime) => {
                    if observed_round != rounds {
                        if unsat_count == 0 { break $exit; }
                        extension_best = extension_best.min(unsat_count);
                        if rounds >= attempt_budget {
                            if resumed { break $exit; }
                            if ENABLE_RESCUE && attempt == 0 && checkpoint.is_none()
                                && extension_best <= 16 && consumed_rounds + rounds < max_flips {
                                let phase = if unsat_count < best_unsat {
                                    ranges.restore_variables(&all_off,&p_bound,&mut vars);
                                    vars.clone()
                                } else { best_vars.clone() };
                                checkpoint = Some(RestartCheckpoint {
                                    rng: rng.clone(), best_vars: phase,
                                    best_unsat: best_unsat.min(unsat_count), order: order.clone(),
                                    consumed_rounds: consumed_rounds + rounds, giveup_latched,
                                });
                            }
                            let remaining = max_flips - consumed_rounds - rounds;
                            if remaining == 0 || extension_best > 16 { break $exit; }
                            attempt_budget += (max_flips / restart_attempts).max(1).min(remaining);
                            extension_best = unsat_count;
                        }
                        if !giveup_latched {
                            if unsat_count <= 16 { giveup_latched = true; }
                            else if rounds >= 67_500_000 {
                                giveup_abandon = true;
                                break $exit;
                            }
                        }
                        observed_round = rounds;
                    }
                }
            }
            'attempt_search: loop{
                observe_original_round!('attempt_search);
                // Every budget/deadline event is a batch boundary. Inside the
                // phase, only unsat minima and the monotone latch can change.
                let mut event_end=next_check_interval.min(next_variance_interval).min(attempt_budget);
                if !giveup_latched { event_end=event_end.min(67_500_000); }
                let phase_start=rounds;
                let steps=(event_end-rounds).min(capacity_batch_limit);
                residual.reserve(steps*max_loss_per_flip);
                let batch_end=rounds+steps;
                // Events/kicks are outside this bounded main phase.
                // At most steps*max_loss_per_flip new IDs can be appended.
                let small=residual.len().checked_add(steps*max_loss_per_flip).map_or(false,|n|n<phase_div::LIMIT);
                if small{
                while (rounds<batch_end) & (unsat_count!=0){
                // The boundary controller already observed phase_start, before
                // any same-round kick. Observe only subsequent main-loop entries.
                if rounds!=phase_start {
                    extension_best=extension_best.min(unsat_count);
                    if !giveup_latched && unsat_count<=16 { giveup_latched=true; }
                }
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
            let v_idx=if WEIGHTED_RESCUE {choose_weighted::<C,COMPACT,0>(vv,clen,&cache,&coin,&mut rng,&var_appearances,&var_age)} else {if clen==3{choose_shape::<C,COMPACT,3>(vv,clen,&cache,&coin,&mut rng,&var_appearances,&var_age)}else{choose_short::<C,COMPACT>(vv,clen,&cache,&coin,&mut rng,&var_appearances,&var_age)}};
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
                // The boundary controller already observed phase_start, before
                // any same-round kick. Observe only subsequent main-loop entries.
                if rounds!=phase_start {
                    extension_best=extension_best.min(unsat_count);
                    if !giveup_latched && unsat_count<=16 { giveup_latched=true; }
                }
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
            let v_idx=if WEIGHTED_RESCUE {choose_weighted::<C,COMPACT,0>(vv,clen,&cache,&coin,&mut rng,&var_appearances,&var_age)} else {if clen==3{choose_shape::<C,COMPACT,3>(vv,clen,&cache,&coin,&mut rng,&var_appearances,&var_age)}else{choose_short::<C,COMPACT>(vv,clen,&cache,&coin,&mut rng,&var_appearances,&var_age)}};
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
                observe_original_round!('attempt_search);
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

        consumed_rounds += rounds;
        if unsat_count < best_unsat {
            best_unsat = unsat_count;
                        record_limit=best_unsat.min(13);
            ranges.restore_variables(&all_off,&p_bound,&mut vars);
                    best_vars.copy_from_slice(&vars);
        }
        if TRACE {use rand::RngCore;let mut copy=rng.clone();trace.attempts.push((attempt,best_unsat,consumed_rounds,copy.next_u64()));}
        if unsat_count == 0 {
            ranges.restore_variables(&all_off,&p_bound,&mut vars);
            let _ = save_solution(&Solution { variables: vars });
            return Ok(());
        }
        if !resumed && ((attempt==0 && best_unsat>LIMIT0)||(attempt>=AFTER && best_unsat>LIMIT)) {if TRACE {trace.gated=true;}break;}
        if giveup_abandon { break; }
    }

    let _ = save_solution(&Solution { variables: best_vars });
    if ENABLE_RESCUE && !resumed && best_unsat > 0 && best_unsat <= RESCUE_MAX_UNSAT {
        if let Some(saved) = checkpoint {
            let save_complete = |solution: &Solution| -> Result<()> {
                if solution.variables.len() == nv && challenge.clauses.iter().all(|clause| {
                    clause.iter().any(|&lit| solution.variables[(lit.abs()-1) as usize] == (lit>0))
                }) { save_solution(solution)?; }
                Ok(())
            };
            solve_impl::<C,COMPACT,true>(challenge,hp,&save_complete,Some(saved))?;
        }
    }
    Ok(())
}
pub fn solve_two_stage<const LIMIT0:usize,const TRACE:bool>(ch:&Challenge,hp:&Option<Hyperparameters>,save:&dyn Fn(&Solution)->Result<()>,stats:&mut GateStats)->Result<()> {
    let nv=ch.num_variables;let mut degree=vec![0u32;nv];for cl in &ch.clauses{for &l in cl{degree[(l.abs()-1)as usize]+=1;}}
    if nv<=8192&&degree.iter().all(|&d|d<=255){solve_policy2::<Byte13BreakCache,true,false,LIMIT0,1,18,TRACE>(ch,hp,save,None,stats)}else{solve_policy2::<SmallBreakCache,false,false,LIMIT0,1,18,TRACE>(ch,hp,save,None,stats)}
}

#[cfg(test)]mod c20_tests{use super::*;use std::cell::RefCell;
fn fixture(cores:usize)->Challenge{let mut clauses=Vec::new();for g in 0..cores{for mask in 0..8{clauses.push((0..3).map(|v|{let l=(3*g+v+1)as i32;if mask&(1<<v)==0{l}else{-l}}).collect());}}Challenge{seed:[7;32],num_variables:3*cores+1,clauses}}
fn hp()->Option<Hyperparameters>{Some(Hyperparameters{base_prob:None,max_prob:None,check_interval:None,stagnation_limit:None,perturbation_flips:None,max_fuel_high:Some(50_000_000.0),max_fuel_low:None})}
#[test]fn two_stage_gates_preserve_exact_attempt0_prefix(){let ch=fixture(40);let full=solve_gated_diagnostic(&ch,&hp(),&|_|Ok(()),Some((1,18))).unwrap();let mut a=GateStats::default();let mut b=GateStats::default();solve_two_stage::<28,true>(&ch,&hp(),&|_|Ok(()),&mut a).unwrap();solve_two_stage::<32,true>(&ch,&hp(),&|_|Ok(()),&mut b).unwrap();assert!(a.gated&&b.gated);assert_eq!(a.attempts,full.attempts[..1]);assert_eq!(b.attempts,full.attempts[..1]);}
#[test]fn passing_attempt0_preserves_attempt1_and_default_callbacks(){let ch=fixture(24);let a=RefCell::new(Vec::new());let b=RefCell::new(Vec::new());let full=solve_gated_diagnostic(&ch,&hp(),&|s|{a.borrow_mut().push(s.variables.clone());Ok(())},Some((1,18))).unwrap();let mut cut=GateStats::default();solve_two_stage::<28,true>(&ch,&hp(),&|s|{b.borrow_mut().push(s.variables.clone());Ok(())},&mut cut).unwrap();assert_eq!(cut,full);assert_eq!(*a.borrow(),*b.borrow());assert_eq!(cut.attempts.len(),2);assert!(cut.attempts[0].1<=28);}
#[test]fn disabled_first_gate_matches_c18_callbacks_and_rng(){let ch=fixture(40);let a=RefCell::new(Vec::new());let b=RefCell::new(Vec::new());let full=solve_gated_diagnostic(&ch,&hp(),&|s|{a.borrow_mut().push(s.variables.clone());Ok(())},Some((1,18))).unwrap();let mut cut=GateStats::default();solve_two_stage::<{usize::MAX},true>(&ch,&hp(),&|s|{b.borrow_mut().push(s.variables.clone());Ok(())},&mut cut).unwrap();assert_eq!(full,cut);assert_eq!(*a.borrow(),*b.borrow());}}

fn c26_abort_attempt0(resumed:bool,attempt:usize,rounds:usize,att_min:usize)->bool{
 !resumed && attempt==0 && ((rounds==2_000_000 && att_min>69)||(rounds==3_500_000 && att_min>44))
}
fn solve_c26_policy2<C:CacheOps,const COMPACT:bool,const WEIGHTED_RESCUE:bool,const LIMIT0:usize,const AFTER:usize,const LIMIT:usize,const TRACE:bool>(
    challenge:&Challenge,hp:&Option<Hyperparameters>,save_solution:&dyn Fn(&Solution)->Result<()>,resume:Option<RestartCheckpoint>,trace:&mut GateStats,
)->Result<()> {
    let resumed = resume.is_some();
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
    let all_data=super::donor_kernel::large_buffer::from_slice(&all_data);
    let mut giveup_latched = false;
    let mut giveup_abandon = false;
    let mut consumed_rounds = 0usize;
    let mut checkpoint: Option<RestartCheckpoint> = None;
    let start_attempt = if let Some(saved) = resume {
        rng = saved.rng;
        best_vars = saved.best_vars;
        best_unsat = saved.best_unsat;
        record_limit = best_unsat.min(13);
        order = saved.order;
        consumed_rounds = saved.consumed_rounds;
        giveup_latched = saved.giveup_latched;
        1
    } else { 0 };
    for attempt in start_attempt..restart_attempts {
        if consumed_rounds >= max_flips { break; }
        let mut attempt_budget = if restart_attempts == 1 {
            max_flips
        } else {
            max_flips / restart_attempts
                + if attempt < (max_flips % restart_attempts) { 1 } else { 0 }
        };
        attempt_budget = attempt_budget.min(max_flips - consumed_rounds);
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
        let mut att_min=unsat_count;

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
        let mut extension_best = unsat_count;
        // The old loop checks control once before its periodic events. A kick
        // can change unsat at the same round without a second control check.
        // Batching must not accidentally observe that intermediate state.
        let mut observed_round = usize::MAX;

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
            macro_rules! observe_original_round {
                ($exit:lifetime) => {
                    if observed_round != rounds {
                        if unsat_count == 0 { break $exit; }
                        extension_best = extension_best.min(unsat_count);
                        if rounds >= attempt_budget {
                            if resumed { break $exit; }
                            if ENABLE_RESCUE && attempt == 0 && checkpoint.is_none()
                                && extension_best <= 16 && consumed_rounds + rounds < max_flips {
                                let phase = if unsat_count < best_unsat {
                                    ranges.restore_variables(&all_off,&p_bound,&mut vars);
                                    vars.clone()
                                } else { best_vars.clone() };
                                checkpoint = Some(RestartCheckpoint {
                                    rng: rng.clone(), best_vars: phase,
                                    best_unsat: best_unsat.min(unsat_count), order: order.clone(),
                                    consumed_rounds: consumed_rounds + rounds, giveup_latched,
                                });
                            }
                            let remaining = max_flips - consumed_rounds - rounds;
                            if remaining == 0 || extension_best > 16 { break $exit; }
                            attempt_budget += (max_flips / restart_attempts).max(1).min(remaining);
                            extension_best = unsat_count;
                        }
                        if !giveup_latched {
                            if unsat_count <= 16 { giveup_latched = true; }
                            else if rounds >= 67_500_000 {
                                giveup_abandon = true;
                                break $exit;
                            }
                        }
                        observed_round = rounds;
                    }
                }
            }
            'attempt_search: loop{
                observe_original_round!('attempt_search);
                // Every budget/deadline event is a batch boundary. Inside the
                // phase, only unsat minima and the monotone latch can change.
                let mut event_end=next_check_interval.min(next_variance_interval).min(attempt_budget);
                if !giveup_latched { event_end=event_end.min(67_500_000); }
                let phase_start=rounds;
                let steps=(event_end-rounds).min(capacity_batch_limit);
                residual.reserve(steps*max_loss_per_flip);
                let batch_end=rounds+steps;
                // Events/kicks are outside this bounded main phase.
                // At most steps*max_loss_per_flip new IDs can be appended.
                let small=residual.len().checked_add(steps*max_loss_per_flip).map_or(false,|n|n<phase_div::LIMIT);
                if small{
                while (rounds<batch_end) & (unsat_count!=0){
                // The boundary controller already observed phase_start, before
                // any same-round kick. Observe only subsequent main-loop entries.
                if rounds!=phase_start {
                    extension_best=extension_best.min(unsat_count);
                    if !giveup_latched && unsat_count<=16 { giveup_latched=true; }
                }
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
            let v_idx=if WEIGHTED_RESCUE {choose_weighted::<C,COMPACT,0>(vv,clen,&cache,&coin,&mut rng,&var_appearances,&var_age)} else {if clen==3{choose_shape::<C,COMPACT,3>(vv,clen,&cache,&coin,&mut rng,&var_appearances,&var_age)}else{choose_short::<C,COMPACT>(vv,clen,&cache,&coin,&mut rng,&var_appearances,&var_age)}};
                apply_flip_main(
                    v_idx,
                    &all_data,&mut ranges,

                    &mut cache,
                    &mut residual,
                    &mut unsat_count,
                    &mut var_age,
                    
                );
                att_min=att_min.min(unsat_count);
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
                // The boundary controller already observed phase_start, before
                // any same-round kick. Observe only subsequent main-loop entries.
                if rounds!=phase_start {
                    extension_best=extension_best.min(unsat_count);
                    if !giveup_latched && unsat_count<=16 { giveup_latched=true; }
                }
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
            let v_idx=if WEIGHTED_RESCUE {choose_weighted::<C,COMPACT,0>(vv,clen,&cache,&coin,&mut rng,&var_appearances,&var_age)} else {if clen==3{choose_shape::<C,COMPACT,3>(vv,clen,&cache,&coin,&mut rng,&var_appearances,&var_age)}else{choose_short::<C,COMPACT>(vv,clen,&cache,&coin,&mut rng,&var_appearances,&var_age)}};
                apply_flip_main(
                    v_idx,
                    &all_data,&mut ranges,

                    &mut cache,
                    &mut residual,
                    &mut unsat_count,
                    &mut var_age,
                    
                );
                att_min=att_min.min(unsat_count);
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
                observe_original_round!('attempt_search);
                if c26_abort_attempt0(resumed,attempt,rounds,att_min){if TRACE{trace.gated=true;}giveup_abandon=true;break 'attempt_search;}
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
                            att_min=att_min.min(unsat_count);
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

        consumed_rounds += rounds;
        if unsat_count < best_unsat {
            best_unsat = unsat_count;
                        record_limit=best_unsat.min(13);
            ranges.restore_variables(&all_off,&p_bound,&mut vars);
                    best_vars.copy_from_slice(&vars);
        }
        if TRACE {use rand::RngCore;let mut copy=rng.clone();trace.attempts.push((attempt,best_unsat,consumed_rounds,copy.next_u64()));}
        if unsat_count == 0 {
            ranges.restore_variables(&all_off,&p_bound,&mut vars);
            let _ = save_solution(&Solution { variables: vars });
            return Ok(());
        }
        if !resumed && ((attempt==0 && best_unsat>LIMIT0)||(attempt>=AFTER && best_unsat>LIMIT)) {if TRACE {trace.gated=true;}break;}
        if giveup_abandon { break; }
    }

    let _ = save_solution(&Solution { variables: best_vars });
    if ENABLE_RESCUE && !resumed && best_unsat > 0 && best_unsat <= RESCUE_MAX_UNSAT {
        if let Some(saved) = checkpoint {
            let save_complete = |solution: &Solution| -> Result<()> {
                if solution.variables.len() == nv && challenge.clauses.iter().all(|clause| {
                    clause.iter().any(|&lit| solution.variables[(lit.abs()-1) as usize] == (lit>0))
                }) { save_solution(solution)?; }
                Ok(())
            };
            solve_impl::<C,COMPACT,true>(challenge,hp,&save_complete,Some(saved))?;
        }
    }
    Ok(())
}
pub fn solve_c26_two_stage<const LIMIT0:usize,const TRACE:bool>(ch:&Challenge,hp:&Option<Hyperparameters>,save:&dyn Fn(&Solution)->Result<()>,stats:&mut GateStats)->Result<()> {
    let nv=ch.num_variables;let mut degree=vec![0u32;nv];for cl in &ch.clauses{for &l in cl{degree[(l.abs()-1)as usize]+=1;}}
    if nv<=8192&&degree.iter().all(|&d|d<=255){solve_c26_policy2::<Byte13BreakCache,true,false,LIMIT0,1,18,TRACE>(ch,hp,save,None,stats)}else{solve_c26_policy2::<SmallBreakCache,false,false,LIMIT0,1,18,TRACE>(ch,hp,save,None,stats)}
}
#[cfg(test)]mod c26_tests{use super::*;use std::cell::RefCell;
 #[test]fn early_gate_has_exact_boundaries_and_is_attempt0_only(){assert!(!c26_abort_attempt0(false,0,1_999_999,70));assert!(c26_abort_attempt0(false,0,2_000_000,70));assert!(!c26_abort_attempt0(false,0,2_000_000,69));assert!(c26_abort_attempt0(false,0,3_500_000,45));assert!(!c26_abort_attempt0(false,0,3_500_000,44));assert!(!c26_abort_attempt0(true,0,2_000_000,100));assert!(!c26_abort_attempt0(false,1,2_000_000,100));}
 #[test]fn passing_short_prefix_preserves_trace_rng_and_callback(){let mut clauses=Vec::new();for g in 0..24{for mask in 0..8{clauses.push((0..3).map(|v|{let l=(3*g+v+1)as i32;if mask&(1<<v)==0{l}else{-l}}).collect());}}let ch=Challenge{seed:[7;32],num_variables:73,clauses};let hp=Some(Hyperparameters{base_prob:None,max_prob:None,check_interval:None,stagnation_limit:None,perturbation_flips:None,max_fuel_high:Some(50_000_000.0),max_fuel_low:None});let a=RefCell::new(Vec::new());let b=RefCell::new(Vec::new());let(mut x,mut y)=(GateStats::default(),GateStats::default());solve_two_stage::<23,true>(&ch,&hp,&|s|{a.borrow_mut().push(s.variables.clone());Ok(())},&mut x).unwrap();solve_c26_two_stage::<23,true>(&ch,&hp,&|s|{b.borrow_mut().push(s.variables.clone());Ok(())},&mut y).unwrap();assert_eq!(x,y);assert_eq!(*a.borrow(),*b.borrow());}
}
