use anyhow::{anyhow, Result};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use tig_challenges::job_scheduling::*;
use super::types::*;
use super::infra::*;

// i126: Cross-nonce elite pool — nonces 5-32 warm-start from best prior elite + repair TS
static NONCE_COUNTER: AtomicUsize = AtomicUsize::new(0);
static ELITE_POOL: Mutex<Vec<(Solution, u64)>> = Mutex::new(Vec::new());
const MAX_POOL_SIZE: usize = 8;
const COLD_START_NONCES: usize = 4;
const REPAIR_ITERS: usize = 20_000;

#[inline]
fn slack_urgency_js(pre: &Pre, target_mk: Option<u32>, time: u32, product: usize, op_idx: usize) -> f64 {
    let Some(tgt) = target_mk else { return 0.0 };
    let lb = (time as u64).saturating_add(pre.product_suf_min[product][op_idx] as u64);
    let slack = (tgt as i64) - (lb as i64);
    let scale = (0.70 * pre.avg_op_min).max(1.0);
    let pos = (slack.max(0) as f64) / scale; let neg = ((-slack).max(0) as f64) / scale;
    (1.0 / (1.0 + pos)).clamp(0.0, 1.0) + (0.35 * neg).min(3.0)
}

#[inline]
fn route_pref_bonus_js(rp: Option<&RoutePrefLite>, product: usize, op_idx: usize, machine: usize) -> f64 {
    let Some(rp) = rp else { return 0.0 };
    if product >= rp.len() || op_idx >= rp[product].len() { return 0.0 }
    let r = rp[product][op_idx]; let mu = machine.min(255) as u8;
    if mu == r.best_m { (r.best_w as f64) / 255.0 } else if mu == r.second_m { (r.second_w as f64) / 255.0 } else { 0.0 }
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn score_candidate(
    pre: &Pre, rule: Rule, job: usize, product: usize, op_idx: usize,
    ops_rem: usize, op: &OpInfo, machine: usize, pt: u32, time: u32,
    target_mk: Option<u32>, best_end: u32, second_end: u32, best_cnt_total: usize,
    progress: f64, job_bias: f64, machine_penalty: f64, dynamic_load: f64,
    route_pref: Option<&RoutePrefLite>, route_w: f64, jitter: f64,
) -> f64 {
    let rem_min = pre.product_suf_min[product][op_idx] as f64;
    let rem_avg = pre.product_suf_avg[product][op_idx];
    let rem_bn = pre.product_suf_bn[product][op_idx];
    let flex_f = (op.flex as f64).max(1.0); let flex_inv = 1.0 / flex_f;
    let rem_min_n = rem_min / pre.horizon.max(1.0); let rem_avg_n = rem_avg / pre.max_job_avg_work.max(1e-9);
    let bn_n = rem_bn / pre.max_job_bn.max(1e-9); let ops_n = (ops_rem as f64) / (pre.max_ops as f64).max(1.0);
    let load_n = dynamic_load / pre.avg_machine_load.max(1e-9); let scar_n = pre.machine_scarcity[machine] / pre.avg_machine_scarcity.max(1e-9);
    let end_n = (best_end as f64) / pre.time_scale.max(1.0); let proc_n = (pt as f64) / pre.avg_op_min.max(1.0);
    let regret = if second_end >= INF { pre.avg_op_min * 2.6 } else { (second_end - best_end) as f64 };
    let reg_n = (regret / pre.avg_op_min.max(1.0)).clamp(0.0, 6.0);
    let scarcity_urg = 1.0 / (best_cnt_total as f64).max(1.0);
    let density_n = ((rem_min / (ops_rem as f64).max(1.0)) / pre.avg_op_min.max(1.0)).clamp(0.0, 4.0);
    let next_min = pre.product_next_min[product][op_idx] as f64; let next_min_n = next_min / pre.horizon.max(1.0);
    let next_flex_inv = pre.product_next_flex_inv[product][op_idx];
    let p2 = progress * progress; let next_w_base = 0.12 + p2 * 0.28;
    let next_term_raw = (0.55*next_min_n + 0.45*next_flex_inv) * (1.0 + 0.30*density_n*pre.high_flex);
    let js = pre.jobshopness; let fl = 1.0 - js;
    let avg_flex_inv = 1.0 / pre.flex_avg.max(1.0); let scarce_match = scar_n * (flex_inv - avg_flex_inv);
    let mpen = machine_penalty.clamp(0.0, 1.0); let mpen_gain = 1.0 + 0.85*pre.high_flex;
    let flow_term = pre.flow_w * pre.job_flow_pref[job] * (0.65 + 0.70*(1.0-progress));
    let slack_u = slack_urgency_js(pre, target_mk, time, product, op_idx);
    let slack_w = pre.slack_base * (0.25 + 0.75*progress); let slack_reg_boost = 1.0 + 0.40*reg_n*progress;
    let pop_pen = if pre.chaotic_like && op.flex >= 2 { let pop = pre.machine_best_pop[machine]; (0.07+0.15*(1.0-progress)).clamp(0.05,0.24)*pop*pre.flex_factor } else { 0.0 };
    let route_gain = (0.70 + 0.80*(1.0-progress)).clamp(0.70, 1.40);
    let route_term = if route_w > 0.0 && op.flex >= 2 { route_w*route_gain*route_pref_bonus_js(route_pref, product, op_idx, machine) } else { 0.0 };
    match rule {
        Rule::CriticalPath => { let next_term = next_w_base*0.30*next_term_raw; let slack_term = slack_w*slack_u*slack_reg_boost; (1.03*rem_min_n)+(0.10*ops_n)+(0.24*scarcity_urg)+(0.20*pre.flex_factor)*flex_inv+next_term+0.10*slack_term-(0.70*end_n)-pop_pen+(0.45*job_bias)+flow_term+route_term+jitter }
        Rule::MostWork => { let next_term = next_w_base*0.25*next_term_raw; (1.00*rem_avg_n)+(0.12*ops_n)+(0.18*scarcity_urg)+(0.15*pre.flex_factor)*flex_inv+next_term-(0.62*end_n)-pop_pen+(0.45*job_bias)+flow_term+route_term+jitter }
        Rule::LeastFlex => { let next_term = next_w_base*0.20*next_term_raw; (1.00*flex_inv)+(0.28*rem_min_n)+(0.22*scarcity_urg)+next_term-(0.55*end_n)-pop_pen+(0.35*job_bias)+flow_term+route_term+jitter }
        Rule::ShortestProc => { let next_term = next_w_base*0.20*next_term_raw; (-1.00*proc_n)+(0.25*rem_min_n)+(0.12*scarcity_urg)+next_term-(0.20*end_n)-pop_pen+(0.25*job_bias)+flow_term+route_term+jitter }
        Rule::Regret => { let next_term = next_w_base*0.25*next_term_raw; (1.05*reg_n)+(0.55*rem_min_n)+(0.22*scarcity_urg)+next_term-(0.68*end_n)-pop_pen+(0.35*job_bias)+flow_term+route_term+jitter }
        Rule::EndTight => { let end_w=1.10+1.00*progress+0.35*pre.high_flex; let cp_w=1.15+0.30*js; let reg_w=(0.55+0.20*(1.0-progress))*(0.85+0.60*js); let mpen_w=(0.10+0.45*pre.high_flex)*pre.flex_factor; let next_term=next_w_base*(0.45+0.55*js)*next_term_raw; let slack_term=slack_w*(0.70+0.40*js)*slack_u*slack_reg_boost; (cp_w*rem_min_n)+0.12*rem_avg_n+0.08*ops_n+0.18*scarcity_urg+(0.30*pre.flex_factor)*flex_inv+(0.20*pre.flex_factor)*scarce_match+(reg_w*pre.flex_factor)*reg_n+next_term+slack_term-end_w*end_n-0.22*proc_n-pop_pen-(mpen_gain*mpen_w)*mpen+0.55*job_bias+flow_term+route_term+jitter }
        Rule::BnHeavy => { let bn_w=(0.90+0.55*js)*pre.bn_focus; let end_w=0.65+0.70*progress; let reg_w=(0.60+0.25*(1.0-progress))*(0.85+0.35*js); let load_w=if pre.hi_flex{-0.35}else{0.55+0.25*js}; let mpen_w=(0.12+0.30*js)*pre.flex_factor*(0.95+0.65*pre.high_flex); let next_term=next_w_base*(0.55+0.75*js)*next_term_raw; let slack_term=slack_w*(0.45+0.55*js)*slack_u*slack_reg_boost; (0.95*rem_min_n)+(0.30*rem_avg_n)+(bn_w*bn_n)+(0.22*density_n)+(0.10*ops_n)+(0.65*pre.flex_factor)*flex_inv+(0.35*pre.flex_factor)*scarce_match+load_w*pre.flex_factor*load_n+(reg_w*pre.flex_factor)*reg_n+0.18*scarcity_urg+next_term+slack_term-end_w*end_n-0.18*proc_n-pop_pen-(mpen_gain*mpen_w)*mpen+0.60*job_bias+flow_term+route_term+jitter }
        Rule::Adaptive => { let end_w=(0.90*fl+0.72*js)+(0.62+0.12*fl)*progress+0.18*pre.high_flex; let reg_w=(0.50*fl+0.78*js)+0.18*(1.0-progress); let bn_w=((0.45+0.40*js)+0.25*(1.0-progress))*pre.bn_focus; let load_sign=if pre.hi_flex{-1.0}else{1.0}; let load_w=load_sign*(0.45*fl+0.75*js)*pre.flex_factor; let density_w=0.08*fl+0.20*js; let next_term=next_w_base*(0.50*fl+1.50*js)*next_term_raw; let mpen_w=(0.08*fl+0.28*js)*pre.flex_factor*(1.0+0.85*pre.high_flex); let slack_term=slack_w*(0.55*fl+0.85*js)*slack_u*slack_reg_boost; (1.05*rem_min_n)+(0.48*rem_avg_n)+(bn_w*bn_n)+density_w*density_n+(0.08*ops_n)+(0.62*pre.flex_factor)*flex_inv+(0.55*pre.flex_factor)*scarce_match+load_w*load_n+(reg_w*pre.flex_factor)*reg_n+0.20*pre.flex_factor*scarcity_urg+next_term+slack_term-end_w*end_n-(0.18*fl+0.12*js)*proc_n-pop_pen-(mpen_gain*mpen_w)*mpen+(0.62+0.06*js)*job_bias+flow_term+route_term+jitter }
        Rule::FlexBalance => { let end_w=(0.85+0.70*progress+0.15*js).clamp(0.85,1.75); let cp_w=(1.00+0.30*js+0.15*(1.0-progress)).clamp(0.95,1.45); let load_w=(0.55+0.35*pre.high_flex).clamp(0.55,0.95)*pre.flex_factor; let mpen_w=(0.55+0.65*pre.high_flex).clamp(0.55,1.15); let reg_w=(0.35+0.25*(1.0-progress)).clamp(0.35,0.70); let next_term=next_w_base*0.40*next_term_raw; (cp_w*rem_min_n)+0.55*rem_avg_n+0.08*ops_n+0.06*density_n+0.08*scarcity_urg+next_term+(0.70*slack_w)*slack_u-end_w*end_n-0.16*proc_n-pop_pen-load_w*load_n-(mpen_w*(1.0+0.85*pre.high_flex))*mpen+(reg_w*pre.flex_factor)*reg_n+(0.58+0.10*pre.high_flex)*job_bias+flow_term+route_term+jitter }
    }
}

#[inline]
fn rule_idx(r: Rule) -> usize {
    match r { Rule::Adaptive=>0, Rule::BnHeavy=>1, Rule::EndTight=>2, Rule::CriticalPath=>3, Rule::MostWork=>4, Rule::LeastFlex=>5, Rule::Regret=>6, Rule::ShortestProc=>7, Rule::FlexBalance=>8 }
}

fn choose_rule_bandit(rng: &mut SmallRng, rules: &[Rule], rule_best: &[u32], rule_tries: &[u32], global_best: u32, margin: u32, stuck: usize, chaos_like: bool, late_phase: bool) -> Rule {
    if rules.is_empty() { return Rule::Adaptive; }
    let mut best_seen = global_best; for &mk in rule_best { if mk < best_seen { best_seen = mk; } }
    let scale = (margin as f64).max(1.0); let s = ((stuck as f64)/140.0).clamp(0.0,1.0); let explore_mix = (0.10+0.55*s).clamp(0.10,0.65);
    let mut w = vec![0.0f64; rules.len()];
    for (i, &r) in rules.iter().enumerate() {
        let mk = rule_best[rule_idx(r)]; let t = rule_tries[rule_idx(r)].max(1) as f64;
        let delta = mk.saturating_sub(best_seen) as f64; let exploit = (-delta/scale).exp(); let explore = (1.0/t).sqrt();
        let mut ww = (1.0-explore_mix)*exploit+explore_mix*explore; ww = ww.max(1e-6);
        if chaos_like { ww = ww.powf(0.70); } else if late_phase { ww = ww.powf(1.18); }
        w[i] = ww;
    }
    let mut sum = 0.0; for &ww in &w { sum += ww.max(0.0); }
    if !(sum > 0.0) { return rules[rng.gen_range(0..rules.len())]; }
    let mut r = rng.gen::<f64>() * sum;
    for (i, &ww) in w.iter().enumerate() { r -= ww.max(0.0); if r <= 0.0 { return rules[i]; } }
    rules[rules.len()-1]
}

fn construct_solution_conflict(
    challenge: &Challenge, pre: &Pre, rule: Rule, k: usize, target_mk: Option<u32>,
    rng: &mut SmallRng, job_bias: Option<&[f64]>, machine_penalty: Option<&[f64]>,
    route_pref: Option<&RoutePrefLite>, route_w: f64,
) -> Result<(Solution, u32)> {
    let num_jobs = challenge.num_jobs; let num_machines = challenge.num_machines;
    let mut job_next_op = vec![0usize; num_jobs]; let mut job_ready_time = vec![0u32; num_jobs];
    let mut machine_avail = vec![0u32; num_machines]; let mut machine_load = pre.machine_load0.clone();
    let mut job_schedule: Vec<Vec<(usize, u32)>> = pre.job_ops_len.iter().map(|&len| Vec::with_capacity(len)).collect();
    let mut remaining_ops = pre.total_ops; let mut time = 0u32;
    let mut demand: Vec<u16> = vec![0u16; num_machines];
    let mut raw_by_machine: Vec<Vec<RawCand>> = (0..num_machines).map(|_| Vec::with_capacity(12)).collect();
    let mut idle_machines: Vec<usize> = Vec::with_capacity(num_machines);
    let chaotic_like = pre.chaotic_like;
    let mut machine_work: Vec<u64> = if chaotic_like { vec![0u64; num_machines] } else { vec![] };
    let mut sum_work: u64 = 0;
    while remaining_ops > 0 {
        loop {
            idle_machines.clear();
            for m in 0..num_machines { if machine_avail[m] <= time { idle_machines.push(m); } }
            if idle_machines.is_empty() { break; }
            for &m in &idle_machines { demand[m] = 0; raw_by_machine[m].clear(); }
            let progress = 1.0 - (remaining_ops as f64) / (pre.total_ops as f64).max(1.0);
            let cap_per_machine = if k == 0 { 12usize } else { (k+6).min(12) };
            for job in 0..num_jobs {
                let op_idx = job_next_op[job];
                if op_idx >= pre.job_ops_len[job] || job_ready_time[job] > time { continue; }
                let product = pre.job_products[job]; let op = &pre.product_ops[product][op_idx];
                if op.flex == 0 || op.machines.is_empty() || op.min_pt >= INF { continue; }
                let (best_end, second_end, best_cnt_total, best_cnt_idle) = best_second_and_counts(time, &machine_avail, op);
                if best_end >= INF || best_cnt_idle == 0 { continue; }
                let ops_rem = pre.job_ops_len[job] - op_idx; let jb = job_bias.map(|v| v[job]).unwrap_or(0.0);
                let flex_inv = 1.0/(op.flex as f64).max(1.0); let scarcity_urg = 1.0/(best_cnt_total as f64).max(1.0);
                let regret = if second_end >= INF { pre.avg_op_min*2.6 } else { (second_end-best_end) as f64 };
                let regn = (regret/pre.avg_op_min.max(1.0)).clamp(0.0,6.0); let rigidity = (0.60*flex_inv+0.40*scarcity_urg).clamp(0.0,2.5);
                for &(m, pt) in &op.machines {
                    if machine_avail[m] > time { continue; }
                    let end = time.saturating_add(pt); if end != best_end { continue; }
                    demand[m] = demand[m].saturating_add(1);
                    let mp = machine_penalty.map(|v| v[m]).unwrap_or(0.0); let jitter = if k > 0 { rng.gen::<f64>()*1e-9 } else { 0.0 };
                    let base = score_candidate(pre, rule, job, product, op_idx, ops_rem, op, m, pt, time, target_mk, best_end, second_end, best_cnt_total, progress, jb, mp, machine_load[m], route_pref, route_w, jitter);
                    push_top_k_raw(&mut raw_by_machine[m], RawCand { job, machine: m, pt, base_score: base, rigidity, reg_n: regn }, cap_per_machine);
                }
            }
            let denom = (idle_machines.len() as f64).max(1.0);
            let (conflict_w, conflict_scale) = if chaotic_like { (-(0.05+0.08*(1.0-progress)).clamp(0.04,0.14), (0.95+0.20*pre.flex_factor).clamp(0.90,1.20)) } else { ((0.09+0.26*pre.jobshopness+0.11*pre.high_flex+0.16*(1.0-progress)).clamp(0.05,0.45), (0.90+0.40*pre.flex_factor).clamp(0.85,1.75)) };
            let (bal_w, avg_work) = if chaotic_like { ((0.030+0.070*(1.0-progress)).clamp(0.025,0.11), (sum_work as f64)/(num_machines as f64).max(1.0)) } else { (0.0, 0.0) };
            let mut best: Option<Cand> = None; let mut top: Vec<Cand> = if k > 0 { Vec::with_capacity(k) } else { Vec::new() };
            for &m in &idle_machines {
                let dem = demand[m] as f64; if dem <= 0.0 || raw_by_machine[m].is_empty() { continue; }
                let dem_n = ((dem-1.0)/denom).clamp(0.0,2.5);
                let bal_pen = if chaotic_like && bal_w > 0.0 { let denomw=(avg_work+(pre.avg_op_min*3.0).max(1.0)).max(1.0); let r=(machine_work[m] as f64)/denomw; let done_n=(r/(r+1.0)).clamp(0.0,1.0); -bal_w*done_n } else { 0.0 };
                for rc in &raw_by_machine[m] {
                    let rig=rc.rigidity.clamp(0.0,2.5); let regc=rc.reg_n.clamp(0.0,4.5);
                    let mut boost=conflict_w*conflict_scale*dem_n*(1.15*rig+0.85*regc);
                    if chaotic_like { boost=boost.max(-0.26); }
                    let c = Cand { job: rc.job, machine: rc.machine, pt: rc.pt, score: rc.base_score+boost+bal_pen };
                    if k == 0 { if best.map_or(true, |bb| c.score > bb.score) { best = Some(c); } } else { push_top_k(&mut top, c, k); }
                }
            }
            let chosen = if k == 0 { match best { Some(c) => c, None => break } } else { if top.is_empty() { break; } choose_from_top_weighted(rng, &top) };
            let job = chosen.job; let machine = chosen.machine; let pt = chosen.pt;
            let product = pre.job_products[job]; let op_idx = job_next_op[job]; let op = &pre.product_ops[product][op_idx];
            let (best_end_now,_,_,_) = best_second_and_counts(time, &machine_avail, op);
            let end_check = time.max(machine_avail[machine]).saturating_add(pt);
            if machine_avail[machine] > time || end_check != best_end_now { break; }
            let end_time = time.saturating_add(pt);
            job_schedule[job].push((machine, time)); job_next_op[job]+=1; job_ready_time[job]=end_time; machine_avail[machine]=end_time; remaining_ops-=1;
            if chaotic_like { machine_work[machine]=machine_work[machine].saturating_add(pt as u64); sum_work=sum_work.saturating_add(pt as u64); }
            if op.min_pt < INF && op.flex > 0 && !op.machines.is_empty() { let delta=(op.min_pt as f64)/(op.flex as f64).max(1.0); if delta>0.0 { for &(mm,_) in &op.machines { let v=machine_load[mm]-delta; machine_load[mm]=if v>0.0{v}else{0.0}; } } }
            if remaining_ops == 0 { break; }
        }
        if remaining_ops == 0 { break; }
        let mut next_time: Option<u32> = None;
        for &t in &machine_avail { if t > time { next_time = Some(next_time.map_or(t, |b| b.min(t))); } }
        for j in 0..num_jobs { let op_idx=job_next_op[j]; if op_idx < pre.job_ops_len[j] && job_ready_time[j] > time { let t=job_ready_time[j]; next_time=Some(next_time.map_or(t,|b|b.min(t))); } }
        time = next_time.ok_or_else(|| anyhow!("Stalled"))?;
    }
    let mk = machine_avail.into_iter().max().unwrap_or(0);
    Ok((Solution { job_schedule }, mk))
}

pub fn tabu_search_phase(pre: &Pre, challenge: &Challenge, base_sol: &Solution, max_iterations: usize, tenure_base: usize) -> Result<Option<(Solution, u32)>> {
    let mut ds = build_disj_from_solution(pre, challenge, base_sol)?;
    let mut buf = EvalBuf::new(ds.n); let n = ds.n;
    let Some(init_eval) = eval_disj(&ds, &mut buf) else { return Ok(None) };
    let initial_mk = init_eval.0; let mut best_global_mk = initial_mk; let mut best_global_ds = ds.clone();
    let tenure = tenure_base.max(5); let tenure_delta = (tenure/3).max(2); let max_no_improve = (max_iterations/2).max(60);
    let mut tabu: HashMap<(usize,usize),usize> = HashMap::with_capacity(tenure*8);
    let mut crit = vec![false; n]; let mut no_improve = 0usize;
    let mut pseed: u64 = (challenge.seed[0] as u64).wrapping_mul(0x9E3779B97F4A7C15) ^ (initial_mk as u64).wrapping_shl(16) ^ (n as u64).wrapping_mul(0x517CC1B727220A95);
    let mut tail = vec![0u32; n]; let mut back_deg = vec![0u16; n]; let mut back_stack: Vec<usize> = Vec::with_capacity(n);
    let mut machine_pred_node = vec![NONE_USIZE; n]; let mut job_pred_node = vec![NONE_USIZE; n];
    for j in 0..ds.num_jobs { let base = ds.job_offsets[j]; let end = ds.job_offsets[j+1]; for k in (base+1)..end { job_pred_node[k] = k-1; } }
    let kick_threshold = (max_no_improve*2/3).max(40); let mut kicks_left = 5usize;
    // Pre-compute machine_pred_node once; O(1) incremental updates per swap eliminate O(N) per-iter rebuild
    machine_pred_node.fill(NONE_USIZE);
    for seq in &ds.machine_seq { for i in 1..seq.len() { machine_pred_node[seq[i]] = seq[i-1]; } }
    let mut need_pred_rebuild = false;
    for iter in 0..max_iterations {
        if no_improve >= max_no_improve {
            if kicks_left == 0 { break; }
            ds = best_global_ds.clone(); no_improve = 0; kicks_left -= 1; tabu.clear();
            need_pred_rebuild = true; continue;
        }
        if no_improve > 0 && no_improve % kick_threshold == 0 && kicks_left > 0 {
            let Some((_, kick_mk_node)) = eval_disj(&ds, &mut buf) else { break };
            crit.fill(false); let mut u = kick_mk_node; while u != NONE_USIZE { crit[u]=true; u=buf.best_pred[u]; }
            let mut kick_swaps: Vec<(usize,usize)> = Vec::new();
            for m in 0..ds.num_machines { if ds.machine_seq[m].len() <= 1 { continue; } for i in 0..(ds.machine_seq[m].len()-1) { if crit[ds.machine_seq[m][i]] || crit[ds.machine_seq[m][i+1]] { kick_swaps.push((m,i)); } } }
            if !kick_swaps.is_empty() {
                let num_kicks = (3 + (no_improve / kick_threshold)).min(5);
                for _ in 0..num_kicks { pseed^=pseed.wrapping_shl(13); pseed^=pseed.wrapping_shr(7); pseed^=pseed.wrapping_shl(17); let idx=(pseed as usize)%kick_swaps.len(); let (m,pos)=kick_swaps[idx]; if pos+1<ds.machine_seq[m].len() { ds.machine_seq[m].swap(pos,pos+1); } }
            }
            kicks_left -= 1; need_pred_rebuild = true; continue;
        }
        let Some((cur_mk, mk_node)) = eval_disj(&ds, &mut buf) else { break };
        if iter > 0 { if cur_mk < best_global_mk { best_global_mk = cur_mk; best_global_ds = ds.clone(); no_improve = 0; } else { no_improve += 1; } }
        // Rebuild machine_pred_node only after kicks (raw swaps) — otherwise maintained incrementally O(1)
        if need_pred_rebuild {
            machine_pred_node.fill(NONE_USIZE);
            for seq in &ds.machine_seq { for i in 1..seq.len() { machine_pred_node[seq[i]] = seq[i-1]; } }
            need_pred_rebuild = false;
        }
        tail.fill(0); back_deg.fill(0);
        for i in 0..n { if ds.job_succ[i] != NONE_USIZE { back_deg[i] += 1; } if buf.machine_succ[i] != NONE_USIZE { back_deg[i] += 1; } }
        back_stack.clear(); for i in 0..n { if back_deg[i] == 0 { back_stack.push(i); } }
        while let Some(nd) = back_stack.pop() {
            let contrib = ds.node_pt[nd].saturating_add(tail[nd]);
            let jp = job_pred_node[nd]; if jp != NONE_USIZE { if contrib > tail[jp] { tail[jp] = contrib; } back_deg[jp] = back_deg[jp].saturating_sub(1); if back_deg[jp] == 0 { back_stack.push(jp); } }
            let mp = machine_pred_node[nd]; if mp != NONE_USIZE { if contrib > tail[mp] { tail[mp] = contrib; } back_deg[mp] = back_deg[mp].saturating_sub(1); if back_deg[mp] == 0 { back_stack.push(mp); } }
        }
        crit.fill(false); let mut u = mk_node; while u != NONE_USIZE { crit[u]=true; u=buf.best_pred[u]; }
        let mut best_move: Option<(usize,usize,u32)> = None; let mut best_move_mk = u32::MAX;
        let mut fallback_move: Option<(usize,usize,u32)> = None; let mut fallback_mk = u32::MAX;
        for m in 0..ds.num_machines {
            if ds.machine_seq[m].len() <= 1 { continue; }
            let mut blocks: Vec<(usize,usize)> = Vec::new(); let mut i = 0;
            while i < ds.machine_seq[m].len() {
                if !crit[ds.machine_seq[m][i]] { i+=1; continue; }
                let bstart=i; let mut bend=i;
                while bend+1 < ds.machine_seq[m].len() { let x=ds.machine_seq[m][bend]; let y=ds.machine_seq[m][bend+1]; if !crit[y] { break; } let end_x=buf.start[x].saturating_add(ds.node_pt[x]); if buf.start[y]!=end_x { break; } bend+=1; }
                if bend > bstart { blocks.push((bstart,bend)); }
                i = bend+1;
            }
            for &(bstart,bend) in &blocks {
                let block_len = bend-bstart+1;
                let mut swap_positions = [bstart,NONE_USIZE]; let num_swaps = if block_len>=3 { swap_positions[1]=bend-1; 2 } else { 1 };
                for si in 0..num_swaps {
                    let pos=swap_positions[si]; if pos+1>=ds.machine_seq[m].len() { continue; }
                    let node_u=ds.machine_seq[m][pos]; let node_v=ds.machine_seq[m][pos+1];
                    let est_mk = estimate_swap_mk(node_u, node_v, &buf.start, &tail, &ds.node_pt, &job_pred_node, &ds.job_succ, &machine_pred_node, &buf.machine_succ);
                    let key=(node_u.min(node_v),node_u.max(node_v)); let is_tabu=tabu.get(&key).map_or(false,|&exp| iter<exp); let aspiration=est_mk<best_global_mk;
                    if (!is_tabu||aspiration) && est_mk<best_move_mk { best_move_mk=est_mk; best_move=Some((m,pos,est_mk)); }
                    if est_mk<fallback_mk { fallback_mk=est_mk; fallback_move=Some((m,pos,est_mk)); }
                }
            }
        }
        let chosen = best_move.or(fallback_move);
        match chosen {
            Some((m,pos,_est)) => {
                let node_a=ds.machine_seq[m][pos]; let node_b=ds.machine_seq[m][pos+1];
                // Save before swap for O(1) machine_pred update
                let pred_a = machine_pred_node[node_a];
                let succ_b_node = buf.machine_succ[node_b]; // fresh from eval_disj this iter
                ds.machine_seq[m].swap(pos,pos+1);
                pseed^=pseed.wrapping_shl(13); pseed^=pseed.wrapping_shr(7); pseed^=pseed.wrapping_shl(17);
                let offset=(pseed%((2*tenure_delta+1) as u64)) as usize;
                let progress=(iter as f64)/(max_iterations as f64); let late_bonus=if progress>0.6{((progress-0.6)*10.0) as usize}else{0};
                let this_tenure=(tenure+offset+late_bonus).saturating_sub(tenure_delta);
                let key=(node_a.min(node_b),node_a.max(node_b)); tabu.insert(key,iter+this_tenure);
                // O(1) machine_pred_node update: pred_a → node_b → node_a → succ_b_node
                machine_pred_node[node_b] = pred_a;
                machine_pred_node[node_a] = node_b;
                if succ_b_node != NONE_USIZE { machine_pred_node[succ_b_node] = node_a; }
            }
            None => break,
        }
    }
    let Some((final_mk,_)) = eval_disj(&ds, &mut buf) else { return Ok(None) };
    if final_mk < best_global_mk { best_global_mk = final_mk; best_global_ds = ds.clone(); }
    if best_global_mk >= initial_mk { return Ok(None); }
    ds = best_global_ds;
    let Some((mk_final,_)) = eval_disj(&ds, &mut buf) else { return Ok(None) };
    let sol = disj_to_solution(pre, &ds, &buf.start)?;
    Ok(Some((sol, mk_final)))
}

#[inline]
fn estimate_swap_mk(u: usize, v: usize, heads: &[u32], tails: &[u32], pt: &[u32], job_pred: &[usize], job_succ: &[usize], machine_pred: &[usize], machine_succ: &[usize]) -> u32 {
    let mp_u=machine_pred[u]; let ms_v=machine_succ[v]; let jp_v=job_pred[v]; let jp_u=job_pred[u]; let js_u=job_succ[u]; let js_v=job_succ[v];
    let r_jp_v=if jp_v!=NONE_USIZE{heads[jp_v].saturating_add(pt[jp_v])}else{0}; let r_mp_u=if mp_u!=NONE_USIZE{heads[mp_u].saturating_add(pt[mp_u])}else{0};
    let new_r_v=r_jp_v.max(r_mp_u); let r_jp_u=if jp_u!=NONE_USIZE{heads[jp_u].saturating_add(pt[jp_u])}else{0}; let new_r_u=r_jp_u.max(new_r_v.saturating_add(pt[v]));
    let q_js_u=if js_u!=NONE_USIZE{pt[js_u].saturating_add(tails[js_u])}else{0}; let q_ms_v=if ms_v!=NONE_USIZE{pt[ms_v].saturating_add(tails[ms_v])}else{0};
    let new_q_u=q_js_u.max(q_ms_v); let q_js_v=if js_v!=NONE_USIZE{pt[js_v].saturating_add(tails[js_v])}else{0}; let new_q_v=q_js_v.max(pt[u].saturating_add(new_q_u));
    let path_v=new_r_v.saturating_add(pt[v]).saturating_add(new_q_v); let path_u=new_r_u.saturating_add(pt[u]).saturating_add(new_q_u);
    path_v.max(path_u)
}

// ============================================================================
// N7 INSERTION ESTIMATOR — same-machine move, O(1) makespan approximation
// ============================================================================
#[inline]
fn estimate_insert_mk_disj(
    u: usize, m: usize, old_pos: usize, new_pos_adj: usize,
    ds: &DisjSchedule, heads: &[u32], tails: &[u32], job_pred: &[usize],
) -> u32 {
    let seq = &ds.machine_seq[m];
    let slen = seq.len();
    if slen < 2 || old_pos >= slen { return u32::MAX; }
    let pt_u = ds.node_pt[u];
    let at = |i: usize| -> usize { if i < old_pos { seq[i] } else { seq[i + 1] } };
    let cleaned_len = slen - 1;
    let mp = if new_pos_adj > 0 { at(new_pos_adj - 1) } else { NONE_USIZE };
    let ms = if new_pos_adj < cleaned_len { at(new_pos_adj) } else { NONE_USIZE };
    let old_mp = if old_pos > 0 { seq[old_pos - 1] } else { NONE_USIZE };
    let old_ms = if old_pos + 1 < slen { seq[old_pos + 1] } else { NONE_USIZE };
    let jp_u = job_pred[u]; let js_u = ds.job_succ[u];
    let r_j = if jp_u != NONE_USIZE { heads[jp_u].saturating_add(ds.node_pt[jp_u]) } else { 0 };
    let r_m = if mp != NONE_USIZE { heads[mp].saturating_add(ds.node_pt[mp]) } else { 0 };
    let new_r = r_j.max(r_m);
    let q_j = if js_u != NONE_USIZE { ds.node_pt[js_u].saturating_add(tails[js_u]) } else { 0 };
    let q_m = if ms != NONE_USIZE { ds.node_pt[ms].saturating_add(tails[ms]) } else { 0 };
    let new_q = q_j.max(q_m);
    let path_u = new_r.saturating_add(pt_u).saturating_add(new_q);
    let path_gap = if old_ms != NONE_USIZE {
        let jp2 = job_pred[old_ms];
        let r_j2 = if jp2 != NONE_USIZE { heads[jp2].saturating_add(ds.node_pt[jp2]) } else { 0 };
        let r_m2 = if old_mp != NONE_USIZE { heads[old_mp].saturating_add(ds.node_pt[old_mp]) } else { 0 };
        r_j2.max(r_m2).saturating_add(ds.node_pt[old_ms]).saturating_add(tails[old_ms])
    } else { 0 };
    let path_ms = if ms != NONE_USIZE {
        let jp3 = job_pred[ms];
        let r_j3 = if jp3 != NONE_USIZE { heads[jp3].saturating_add(ds.node_pt[jp3]) } else { 0 };
        let r_u_end = new_r.saturating_add(pt_u);
        r_j3.max(r_u_end).saturating_add(ds.node_pt[ms]).saturating_add(tails[ms])
    } else { 0 };
    path_u.max(path_gap).max(path_ms)
}

// ============================================================================
// LAZY N5+N7 TABU SEARCH — N7 insertions activate only after N7_LAZY_THRESHOLD
// consecutive non-improving iterations. Before threshold: pure N5 (fast).
// After threshold: N5+N7 combined (richer neighborhood to escape local optimum).
// N7 candidates: move first/last op of critical block across block boundary.
// N7_LAZY_THRESHOLD=2500, max_no_improve=5000 → max 5000 N7 iters worst-case.
// ============================================================================
const N7_LAZY_THRESHOLD: usize = 2500;
const N7_ITER_CAP: usize = 350;

fn tabu_n7_phase(
    pre: &Pre, challenge: &Challenge, base_sol: &Solution,
    max_iterations: usize, tenure_base: usize,
) -> Result<Option<(Solution, u32)>> {
    let mut ds = build_disj_from_solution(pre, challenge, base_sol)?;
    let mut buf = EvalBuf::new(ds.n); let n = ds.n;
    let Some(init_eval) = eval_disj(&ds, &mut buf) else { return Ok(None) };
    let initial_mk = init_eval.0; let mut best_mk = initial_mk; let mut best_ds = ds.clone();
    let tenure = tenure_base.max(5); let tenure_delta = (tenure/3).max(2);
    let max_no_improve = (max_iterations/2).max(60);
    let mut tabu: HashMap<(usize,usize),usize> = HashMap::with_capacity(tenure*8);
    let mut ins_tabu: HashMap<usize,usize> = HashMap::with_capacity(32);
    let mut crit = vec![false; n]; let mut no_improve = 0usize;
    let mut tail = vec![0u32; n]; let mut back_deg = vec![0u16; n]; let mut back_stack: Vec<usize> = Vec::with_capacity(n);
    let mut machine_pred_node = vec![NONE_USIZE; n]; let mut job_pred_node = vec![NONE_USIZE; n];
    for j in 0..ds.num_jobs { let base = ds.job_offsets[j]; let end = ds.job_offsets[j+1]; for k in (base+1)..end { job_pred_node[k] = k-1; } }
    machine_pred_node.fill(NONE_USIZE);
    for seq in &ds.machine_seq { for i in 1..seq.len() { machine_pred_node[seq[i]] = seq[i-1]; } }
    let mut need_rebuild = false;
    let mut pseed: u64 = 0x9E3779B9_u64.wrapping_mul(initial_mk as u64).wrapping_add(n as u64);
    let n7_active = N7_LAZY_THRESHOLD < max_no_improve;
    let mut n7_eval_count = 0usize;
    'outer: for iter in 0..max_iterations {
        if no_improve >= max_no_improve { break; }
        let Some((cur_mk, mk_node)) = eval_disj(&ds, &mut buf) else { break };
        if iter > 0 { if cur_mk < best_mk { best_mk = cur_mk; best_ds = ds.clone(); no_improve = 0; } else { no_improve += 1; } }
        if need_rebuild {
            machine_pred_node.fill(NONE_USIZE);
            for seq in &ds.machine_seq { for i in 1..seq.len() { machine_pred_node[seq[i]] = seq[i-1]; } }
            need_rebuild = false;
        }
        tail.fill(0); back_deg.fill(0);
        for i in 0..n { if ds.job_succ[i] != NONE_USIZE { back_deg[i] += 1; } if buf.machine_succ[i] != NONE_USIZE { back_deg[i] += 1; } }
        back_stack.clear(); for i in 0..n { if back_deg[i] == 0 { back_stack.push(i); } }
        while let Some(nd) = back_stack.pop() {
            let c = ds.node_pt[nd].saturating_add(tail[nd]);
            let jp = job_pred_node[nd]; if jp != NONE_USIZE { if c > tail[jp] { tail[jp] = c; } back_deg[jp] = back_deg[jp].saturating_sub(1); if back_deg[jp] == 0 { back_stack.push(jp); } }
            let mp = machine_pred_node[nd]; if mp != NONE_USIZE { if c > tail[mp] { tail[mp] = c; } back_deg[mp] = back_deg[mp].saturating_sub(1); if back_deg[mp] == 0 { back_stack.push(mp); } }
        }
        crit.fill(false); let mut u = mk_node; while u != NONE_USIZE { crit[u]=true; u=buf.best_pred[u]; }
        tabu.retain(|_,&mut exp| exp > iter);
        let use_n7 = n7_active && no_improve >= N7_LAZY_THRESHOLD && n7_eval_count < N7_ITER_CAP;
        if use_n7 { n7_eval_count += 1; ins_tabu.retain(|_,&mut exp| exp > iter); }
        let mut best_swap: Option<(usize,usize)> = None; let mut best_swap_mk = u32::MAX;
        let mut fb_swap: Option<(usize,usize)> = None; let mut fb_swap_mk = u32::MAX;
        let mut best_ins: Option<(usize,usize,usize)> = None;
        let mut best_ins_mk = u32::MAX;
        for m in 0..ds.num_machines {
            let slen = ds.machine_seq[m].len();
            if slen <= 1 { continue; }
            let mut i = 0;
            while i < slen {
                let seq = &ds.machine_seq[m];
                if !crit[seq[i]] { i += 1; continue; }
                let bs = i; let mut be = i;
                while be + 1 < slen {
                    let x=seq[be]; let y=seq[be+1];
                    if !crit[y] { break; }
                    if buf.start[y] != buf.start[x].saturating_add(ds.node_pt[x]) { break; }
                    be += 1;
                }
                if bs + 1 < slen {
                    let seq2 = &ds.machine_seq[m];
                    let na = seq2[bs]; let nb = seq2[bs+1];
                    let est = estimate_swap_mk(na, nb, &buf.start, &tail, &ds.node_pt, &job_pred_node, &ds.job_succ, &machine_pred_node, &buf.machine_succ);
                    let key=(na.min(nb),na.max(nb)); let is_tabu=tabu.get(&key).map_or(false,|&e| iter<e); let asp=est<best_mk;
                    if (!is_tabu||asp) && est<best_swap_mk { best_swap_mk=est; best_swap=Some((m,bs)); }
                    if est<fb_swap_mk { fb_swap_mk=est; fb_swap=Some((m,bs)); }
                }
                if be > bs && be < slen {
                    let seq2 = &ds.machine_seq[m];
                    let pos = be - 1;
                    let na = seq2[pos]; let nb = seq2[pos+1];
                    let est = estimate_swap_mk(na, nb, &buf.start, &tail, &ds.node_pt, &job_pred_node, &ds.job_succ, &machine_pred_node, &buf.machine_succ);
                    let key=(na.min(nb),na.max(nb)); let is_tabu=tabu.get(&key).map_or(false,|&e| iter<e); let asp=est<best_mk;
                    if (!is_tabu||asp) && est<best_swap_mk { best_swap_mk=est; best_swap=Some((m,pos)); }
                    if est<fb_swap_mk { fb_swap_mk=est; fb_swap=Some((m,pos)); }
                }
                // N7: lazy — only when no_improve >= N7_LAZY_THRESHOLD
                if use_n7 && be > bs {
                    let seq2 = &ds.machine_seq[m];
                    let u_a = seq2[bs];
                    if !ins_tabu.contains_key(&u_a) {
                        let est_a = estimate_insert_mk_disj(u_a, m, bs, be, &ds, &buf.start, &tail, &job_pred_node);
                        if est_a < best_mk && est_a < best_ins_mk { best_ins_mk=est_a; best_ins=Some((m,bs,be)); }
                    }
                    let u_b = seq2[be];
                    if !ins_tabu.contains_key(&u_b) {
                        let est_b = estimate_insert_mk_disj(u_b, m, be, bs, &ds, &buf.start, &tail, &job_pred_node);
                        if est_b < best_mk && est_b < best_ins_mk { best_ins_mk=est_b; best_ins=Some((m,be,bs)); }
                    }
                }
                i = be + 1;
            }
        }
        pseed^=pseed.wrapping_shl(13); pseed^=pseed.wrapping_shr(7); pseed^=pseed.wrapping_shl(17);
        let off = (pseed % ((2*tenure_delta+1) as u64)) as usize;
        if best_swap_mk <= best_ins_mk {
            if let Some((m, pos)) = best_swap {
                let na=ds.machine_seq[m][pos]; let nb=ds.machine_seq[m][pos+1];
                let pred_a=machine_pred_node[na]; let succ_b=buf.machine_succ[nb];
                ds.machine_seq[m].swap(pos, pos+1);
                let key=(na.min(nb),na.max(nb)); tabu.insert(key, iter+tenure.saturating_sub(tenure_delta)+off);
                machine_pred_node[nb]=pred_a; machine_pred_node[na]=nb; if succ_b!=NONE_USIZE { machine_pred_node[succ_b]=na; }
            } else if let Some((m, pos)) = fb_swap {
                let na=ds.machine_seq[m][pos]; let nb=ds.machine_seq[m][pos+1];
                let pred_a=machine_pred_node[na]; let succ_b=buf.machine_succ[nb];
                ds.machine_seq[m].swap(pos, pos+1);
                let key=(na.min(nb),na.max(nb)); tabu.insert(key, iter+tenure.saturating_sub(tenure_delta)+off);
                machine_pred_node[nb]=pred_a; machine_pred_node[na]=nb; if succ_b!=NONE_USIZE { machine_pred_node[succ_b]=na; }
            } else {
                break 'outer;
            }
        } else if let Some((m, old_pos, new_pos_adj)) = best_ins {
            let u_ins = ds.machine_seq[m][old_pos];
            apply_insert(&mut ds.machine_seq[m], old_pos, new_pos_adj);
            ins_tabu.insert(u_ins, iter+tenure.saturating_sub(tenure_delta)+off);
            need_rebuild = true;
        } else if let Some((m, pos)) = fb_swap {
            let na=ds.machine_seq[m][pos]; let nb=ds.machine_seq[m][pos+1];
            let pred_a=machine_pred_node[na]; let succ_b=buf.machine_succ[nb];
            ds.machine_seq[m].swap(pos, pos+1);
            let key=(na.min(nb),na.max(nb)); tabu.insert(key, iter+tenure.saturating_sub(tenure_delta)+off);
            machine_pred_node[nb]=pred_a; machine_pred_node[na]=nb; if succ_b!=NONE_USIZE { machine_pred_node[succ_b]=na; }
        } else {
            break 'outer;
        }
    }
    let Some((final_mk,_)) = eval_disj(&ds, &mut buf) else { return Ok(None) };
    if final_mk < best_mk { best_mk = final_mk; best_ds = ds.clone(); }
    if best_mk >= initial_mk { return Ok(None); }
    let Some((mk_final,_)) = eval_disj(&best_ds, &mut buf) else { return Ok(None) };
    let sol = disj_to_solution(pre, &best_ds, &buf.start)?;
    Ok(Some((sol, mk_final)))
}

// ============================================================================
// PATH RELINKING — directed search between elite solutions.
// For each pair (source, target) in the elite pool, apply adjacent swaps that
// reduce the permutation distance to target (inversion moves). Select the swap
// with the lowest estimated makespan at each step. Track the best makespan seen
// anywhere along the path. Returns the best improvement found, if any.
// ============================================================================
fn path_relinking_phase(
    pre: &Pre, challenge: &Challenge,
    pool: &[(Solution, u32)],
    initial_best_mk: u32,
    steps_per_pair: usize,
) -> Result<Option<(Solution, u32)>> {
    if pool.len() < 2 || steps_per_pair == 0 { return Ok(None); }
    let n = pre.total_ops;
    let mut best_mk = initial_best_mk;
    let mut best_found: Option<Solution> = None;

    let mut buf = EvalBuf::new(n);
    let mut tail = vec![0u32; n];
    let mut back_deg = vec![0u16; n];
    let mut back_stack: Vec<usize> = Vec::with_capacity(n);
    let mut machine_pred_node = vec![NONE_USIZE; n];
    let mut job_pred_node = vec![NONE_USIZE; n];
    let mut target_pos = vec![0usize; n];

    // Static job_pred_node (same for all pairs)
    {
        let proto = build_disj_from_solution(pre, challenge, &pool[0].0)?;
        for j in 0..proto.num_jobs {
            let base_j = proto.job_offsets[j]; let end_j = proto.job_offsets[j + 1];
            for k in (base_j + 1)..end_j { job_pred_node[k] = k - 1; }
        }
    }

    // Pair best solution against up to 5 diverse pool members
    let num_pairs = pool.len().min(6).saturating_sub(1);
    let steps = (steps_per_pair / num_pairs.max(1)).max(1);

    for target_idx in 1..=num_pairs {
        if target_idx >= pool.len() { break; }
        let mut current = build_disj_from_solution(pre, challenge, &pool[0].0)?;
        let target_ds = build_disj_from_solution(pre, challenge, &pool[target_idx].0)?;

        // target_pos[op] = position of op in its machine's target sequence
        for m in 0..target_ds.num_machines {
            for (pos, &op) in target_ds.machine_seq[m].iter().enumerate() {
                if op < n { target_pos[op] = pos; }
            }
        }

        for _step in 0..steps {
            let Some((cur_mk, _)) = eval_disj(&current, &mut buf) else { break };
            if cur_mk < best_mk {
                best_mk = cur_mk;
                if let Ok(s) = disj_to_solution(pre, &current, &buf.start) { best_found = Some(s); }
            }

            // Rebuild machine_pred_node
            machine_pred_node.fill(NONE_USIZE);
            for seq in &current.machine_seq { for i in 1..seq.len() { machine_pred_node[seq[i]] = seq[i - 1]; } }

            // Compute tails
            tail.fill(0); back_deg.fill(0);
            for i in 0..n {
                if current.job_succ[i] != NONE_USIZE { back_deg[i] += 1; }
                if buf.machine_succ[i] != NONE_USIZE { back_deg[i] += 1; }
            }
            back_stack.clear(); for i in 0..n { if back_deg[i] == 0 { back_stack.push(i); } }
            while let Some(nd) = back_stack.pop() {
                let c = current.node_pt[nd].saturating_add(tail[nd]);
                let jp = job_pred_node[nd];
                if jp != NONE_USIZE { if c > tail[jp] { tail[jp] = c; } back_deg[jp] = back_deg[jp].saturating_sub(1); if back_deg[jp] == 0 { back_stack.push(jp); } }
                let mp = machine_pred_node[nd];
                if mp != NONE_USIZE { if c > tail[mp] { tail[mp] = c; } back_deg[mp] = back_deg[mp].saturating_sub(1); if back_deg[mp] == 0 { back_stack.push(mp); } }
            }

            // Find best inversion move: adjacent swap in current that reduces distance to target
            let mut best_move: Option<(usize, usize)> = None;
            let mut best_est = u32::MAX;
            let mut any_inversion = false;

            for m in 0..current.num_machines {
                if current.machine_seq[m].len() <= 1 { continue; }
                for i in 0..(current.machine_seq[m].len() - 1) {
                    let op_a = current.machine_seq[m][i];
                    let op_b = current.machine_seq[m][i + 1];
                    if op_a >= n || op_b >= n { continue; }
                    // Inversion: in target, op_b appears before op_a on this machine
                    if target_pos[op_a] > target_pos[op_b] {
                        any_inversion = true;
                        let est = estimate_swap_mk(op_a, op_b, &buf.start, &tail, &current.node_pt,
                            &job_pred_node, &current.job_succ, &machine_pred_node, &buf.machine_succ);
                        if est < best_est { best_est = est; best_move = Some((m, i)); }
                    }
                }
            }

            if !any_inversion { break; } // path complete: current reached target

            // Apply best inversion (or first inversion if all estimated as u32::MAX)
            let (m, pos) = best_move.unwrap_or_else(|| {
                for m in 0..current.num_machines {
                    for i in 0..(current.machine_seq[m].len().saturating_sub(1)) {
                        let a = current.machine_seq[m][i]; let b = current.machine_seq[m][i + 1];
                        if a < n && b < n && target_pos[a] > target_pos[b] { return (m, i); }
                    }
                }
                (0, 0)
            });
            current.machine_seq[m].swap(pos, pos + 1);
        }
    }

    if best_mk >= initial_best_mk { return Ok(None); }
    Ok(best_found.map(|s| (s, best_mk)))
}

// ============================================================================
// CRITICAL-JOB RUIN-AND-RECREATE — exact reinsertion, no stale buffers
// ============================================================================

fn critical_job_ruin_recreate(
    pre: &Pre,
    challenge: &Challenge,
    base_sol: &Solution,
    _rng: &mut SmallRng,
) -> Result<Solution> {
    let mut ds = build_disj_from_solution(pre, challenge, base_sol)?;
    let mut buf = EvalBuf::new(ds.n);
    let (_mk, mk_node) = eval_disj(&ds, &mut buf).ok_or_else(|| anyhow!("Invalid base sol"))?;

    // Identify top-2 critical jobs by op count on critical path
    let mut on_cp = vec![false; ds.n];
    let mut u = mk_node;
    while u != NONE_USIZE {
        on_cp[u] = true;
        u = buf.best_pred[u];
    }

    let mut job_score = vec![0usize; challenge.num_jobs];
    for nd in 0..ds.n {
        if on_cp[nd] {
            job_score[ds.node_job[nd]] += 1;
        }
    }

    let mut ranked: Vec<(usize, usize)> = job_score
        .into_iter()
        .enumerate()
        .filter(|(_, s)| *s > 0)
        .collect();
    if ranked.is_empty() {
        return Ok(base_sol.clone());
    }
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    let ruin_jobs: Vec<usize> = ranked.iter().take(2).map(|(j, _)| *j).collect();
    let mut is_ruined = vec![false; challenge.num_jobs];
    for &j in &ruin_jobs {
        is_ruined[j] = true;
    }

    // RUIN: remove all ops of ruined jobs from machine sequences
    let mut removed: Vec<usize> = Vec::with_capacity(pre.total_ops / 15);
    for m in 0..ds.num_machines {
        let mut i = 0;
        while i < ds.machine_seq[m].len() {
            let node = ds.machine_seq[m][i];
            if is_ruined[ds.node_job[node]] {
                removed.push(node);
                ds.machine_seq[m].remove(i);
            } else {
                i += 1;
            }
        }
    }

    // RECREATE: reinsert in topological order (job, op_idx) — preserves precedence
    removed.sort_by_key(|&node| (ds.node_job[node], ds.node_op[node]));

    for node in removed {
        let machine = ds.node_machine[node];
        let mut best_mk = u32::MAX;
        let mut best_pos = 0usize;
        let max_pos = ds.machine_seq[machine].len();

        for pos in 0..=max_pos {
            ds.machine_seq[machine].insert(pos, node);
            if let Some((mk2, _)) = eval_disj(&ds, &mut buf) {
                if mk2 < best_mk {
                    best_mk = mk2;
                    best_pos = pos;
                }
            }
            ds.machine_seq[machine].remove(pos);
        }
        ds.machine_seq[machine].insert(best_pos, node);
    }

    let (_, _) = eval_disj(&ds, &mut buf).ok_or_else(|| anyhow!("Invalid after ruin"))?;
    disj_to_solution(pre, &ds, &buf.start)
}

// ============================================================================
// SOLVE
// ============================================================================

pub fn solve(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    pre: &Pre,
    effort: &EffortConfig,
) -> Result<()> {
    let nonce_id = NONCE_COUNTER.fetch_add(1, Ordering::SeqCst);
    let (greedy_sol, greedy_mk) = run_simple_greedy_baseline(challenge)?;
    save_solution(&greedy_sol)?;

    let mut rng = SmallRng::from_seed(challenge.seed);
    let allow_flex_balance = pre.high_flex > 0.60 && pre.jobshopness > 0.38;
    let mut rules: Vec<Rule> = vec![Rule::Adaptive, Rule::BnHeavy, Rule::EndTight, Rule::CriticalPath, Rule::MostWork, Rule::LeastFlex, Rule::Regret, Rule::ShortestProc];
    if allow_flex_balance { rules.push(Rule::FlexBalance); }

    let mut best_makespan = greedy_mk; let mut best_solution: Option<Solution> = Some(greedy_sol.clone()); let mut top_solutions: Vec<(Solution, u32)> = Vec::new();
    // Diversity pool for Phase 6 seeding: (solution, makespan, machine_signature)
    let mut diverse_pool: Vec<(Solution, u32, Vec<u8>)> = Vec::with_capacity(30);
    push_top_solutions(&mut top_solutions, &greedy_sol, greedy_mk, 15);
    push_diverse(&mut diverse_pool, greedy_sol.clone(), greedy_mk, machine_signature(&greedy_sol, pre), 25);
    let target_margin: u32 = ((pre.avg_op_min * (0.9 + 0.9*pre.high_flex + 0.6*pre.jobshopness)).max(1.0)) as u32;
    let route_w_base: f64 = if pre.chaotic_like { 0.0 } else { (0.040 + 0.10*pre.high_flex + 0.08*pre.jobshopness).clamp(0.04, 0.22) };

    // i126: moved before warm/cold branch so Phase 6-7 can access regardless of path
    let ts_iters = effort.job_shop_iters;
    let ts_tenure = ((pre.total_ops as f64).sqrt() as usize * (100 + (pre.load_cv * 60.0) as usize) / 100).clamp(8, 24);

    let warm_start = nonce_id >= COLD_START_NONCES && {
        let pool = ELITE_POOL.lock().unwrap();
        !pool.is_empty()
    };

    if warm_start {
        // Nonces 5-32: skip Phase 1-5, warm-start from best global elite + repair TS
        let seed_sol = {
            let pool = ELITE_POOL.lock().unwrap();
            pool[0].0.clone()  // pool is sorted by makespan ascending
        };
        if let Ok(Some((rep, mk))) = tabu_search_phase(pre, challenge, &seed_sol, REPAIR_ITERS, ts_tenure) {
            if mk < best_makespan { best_makespan = mk; best_solution = Some(rep.clone()); save_solution(&rep)?; }
            push_top_solutions(&mut top_solutions, &rep, mk, 15);
            push_diverse(&mut diverse_pool, rep.clone(), mk, machine_signature(&rep, pre), 25);
        }
    } else {
    // ---- Phase 1-5: full cold start (nonces 0-3) ----
    if pre.flow_route.is_some() && pre.flow_pt_by_job.is_some() {
        if let Ok((sol, mk)) = neh_reentrant_flow_solution(pre, challenge.num_jobs, challenge.num_machines) {
            if mk < best_makespan { best_makespan=mk; best_solution=Some(sol.clone()); save_solution(&sol)?; }
            push_top_solutions(&mut top_solutions, &sol, mk, 15);
        }
    }

    let mut ranked: Vec<(Rule,u32,Solution)> = Vec::with_capacity(rules.len());
    for &rule in &rules {
        let (sol, mk) = construct_solution_conflict(challenge, pre, rule, 0, None, &mut rng, None, None, None, 0.0)?;
        if mk < best_makespan { best_makespan=mk; best_solution=Some(sol.clone()); save_solution(&sol)?; }
        push_top_solutions(&mut top_solutions, &sol, mk, 15); ranked.push((rule, mk, sol));
    }
    ranked.sort_by_key(|x| x.1);
    let r0=ranked[0].0; let r1=ranked.get(1).map(|x|x.0).unwrap_or(r0); let r2=ranked.get(2).map(|x|x.0).unwrap_or(r1);

    let mut rule_best: Vec<u32> = vec![u32::MAX; 9]; let mut rule_tries: Vec<u32> = vec![0u32; 9];
    for (rr,mk,_) in &ranked { let idx=rule_idx(*rr); rule_best[idx]=rule_best[idx].min(*mk); rule_tries[idx]=rule_tries[idx].saturating_add(1); }

    let base = &ranked[0].2;
    let mut learned_jb = Some(job_bias_from_solution(pre, base)?);
    let mut learned_mp = Some(machine_penalty_from_solution(pre, base, challenge.num_machines)?);
    let mut learned_rp = if route_w_base > 0.0 { Some(route_pref_from_solution_lite(pre, base, challenge)?) } else { None };
    let mut learn_updates_left = 4usize;
    let num_restarts = 450usize;

    let mut k_hi = if pre.flex_avg > 8.0 { 6 } else if pre.flex_avg > 6.5 { 4 } else if pre.flex_avg > 4.0 { 5 } else { 6 };
    if pre.jobshopness > 0.60 && k_hi < 6 { k_hi += 1; }
    k_hi = k_hi.min(6).max(2);
    let mut stuck: usize = 0;

    for r in 0..num_restarts {
        let late = r >= (num_restarts*2)/3;
        let (k_min,k_max) = if stuck>170 { (4usize,6usize.min(k_hi)) } else if stuck>90 { (3usize,6usize.min(k_hi.max(4))) } else if stuck>35 { (2usize,k_hi) } else { (2usize,k_hi.min(4)) };
        let rule = if r < 35 { let u: f64=rng.gen(); if allow_flex_balance&&pre.high_flex>0.82&&u<0.10{Rule::FlexBalance}else if u<0.52{r0}else if u<0.80{r1}else if u<0.92{r2}else{rules[rng.gen_range(0..rules.len())]} }
            else { choose_rule_bandit(&mut rng, &rules, &rule_best, &rule_tries, best_makespan, target_margin, stuck, pre.chaotic_like, late) };
        let k = if k_max<=k_min { k_min } else { rng.gen_range(k_min..=k_max) };
        let learn_base = if pre.chaotic_like { 0.0 } else { (0.08+0.22*pre.jobshopness+0.18*pre.high_flex).clamp(0.05,0.42) };
        let learn_boost = (1.0+0.35*((stuck as f64)/120.0).clamp(0.0,1.0)).clamp(1.0,1.35);
        let learn_p = (learn_base*learn_boost).clamp(0.0,0.60);
        let use_learn = learned_jb.is_some() && learned_mp.is_some() && rng.gen::<f64>()<learn_p && (route_w_base==0.0||learned_rp.is_some());
        let target = if best_makespan < (u32::MAX/2) { Some(best_makespan.saturating_add(target_margin)) } else { None };
        let (sol, mk) = if use_learn {
            construct_solution_conflict(challenge, pre, rule, k, target, &mut rng, learned_jb.as_deref(), learned_mp.as_deref(), learned_rp.as_ref(), route_w_base)?
        } else {
            construct_solution_conflict(challenge, pre, rule, k, target, &mut rng, None, None, None, 0.0)?
        };
        let ridx=rule_idx(rule); rule_tries[ridx]=rule_tries[ridx].saturating_add(1); rule_best[ridx]=rule_best[ridx].min(mk);
        if mk < best_makespan {
            best_makespan=mk; best_solution=Some(sol.clone()); save_solution(&sol)?; stuck=0;
            if learn_updates_left > 0 && !pre.chaotic_like {
                learned_jb=Some(job_bias_from_solution(pre,&sol)?); learned_mp=Some(machine_penalty_from_solution(pre,&sol,challenge.num_machines)?);
                if route_w_base>0.0 { learned_rp=Some(route_pref_from_solution_lite(pre,&sol,challenge)?); }
                learn_updates_left-=1;
            }
        } else { stuck=stuck.saturating_add(1); }
        push_top_solutions(&mut top_solutions, &sol, mk, 15);
        // Collect diverse solutions for Phase 6 seeding (add every 15th restart)
        if r % 15 == 0 {
            push_diverse(&mut diverse_pool, sol.clone(), mk, machine_signature(&sol, pre), 25);
        }
    }

    let route_w_ls: f64 = if route_w_base>0.0 { (route_w_base*1.40).clamp(route_w_base,0.40) } else { 0.0 };
    let mut refine_results: Vec<(Solution,u32)> = Vec::new();
    for (base_sol, _) in top_solutions.iter() {
        let jb = job_bias_from_solution(pre, base_sol)?;
        let mp = machine_penalty_from_solution(pre, base_sol, challenge.num_machines)?;
        let rp = if route_w_ls>0.0 { Some(route_pref_from_solution_lite(pre, base_sol, challenge)?) } else { None };
        let target_ls = if best_makespan < (u32::MAX/2) { Some(best_makespan.saturating_add(target_margin/2)) } else { None };
        for attempt in 0..10 {
            let rule = if pre.chaotic_like { match attempt%4 { 0=>Rule::Adaptive, 1=>Rule::ShortestProc, 2=>Rule::MostWork, _=>Rule::Regret } }
                else { match attempt { 0=>r0, 1=>Rule::Adaptive, 2=>Rule::BnHeavy, 3=>Rule::EndTight, 4=>Rule::Regret, 5=>Rule::CriticalPath, 6=>Rule::LeastFlex, 7=>Rule::MostWork, 8=>if allow_flex_balance{Rule::FlexBalance}else{r1}, _=>r1 } };
            let k = match attempt%4 { 0=>2, 1=>3, 2=>4, _=>2 }.min(k_hi);
            let (sol, mk) = construct_solution_conflict(challenge, pre, rule, k, target_ls, &mut rng, Some(&jb), Some(&mp), rp.as_ref(), if rp.is_some(){route_w_ls}else{0.0})?;
            if mk < best_makespan { best_makespan=mk; best_solution=Some(sol.clone()); save_solution(&sol)?; }
            refine_results.push((sol, mk));
        }
    }
    for (sol, mk) in refine_results { push_top_solutions(&mut top_solutions, &sol, mk, 15); }

    let ts_starts = top_solutions.len().min(10);
    {
        let ts_pool: Vec<Solution> = top_solutions.iter().take(ts_starts).map(|(s,_)| s.clone()).collect();
        for base_sol in &ts_pool {
            if let Some((sol2, mk2)) = tabu_search_phase(pre, challenge, base_sol, ts_iters, ts_tenure)? {
                if mk2 < best_makespan { best_makespan=mk2; best_solution=Some(sol2.clone()); save_solution(&sol2)?; }
                push_top_solutions(&mut top_solutions, &sol2, mk2, 20);
            }
        }
    }

    {
        let bn_pool: Vec<Solution> = top_solutions.iter().take(8).map(|(s,_)| s.clone()).collect();
        for base_sol in &bn_pool {
            let mut ds = match build_disj_from_solution(pre, challenge, base_sol) { Ok(d) => d, Err(_) => continue };
            let mut buf = EvalBuf::new(ds.n);
            let Some((mut cur_mk, _)) = eval_disj(&ds, &mut buf) else { continue };
            let mut machine_total_pt = vec![0u64; challenge.num_machines];
            for m in 0..challenge.num_machines {
                if m < ds.machine_seq.len() {
                    for &nd in &ds.machine_seq[m] {
                        if nd < ds.node_pt.len() {
                            machine_total_pt[m] = machine_total_pt[m].saturating_add(ds.node_pt[nd] as u64);
                        }
                    }
                }
            }
            let mut m_rank: Vec<usize> = (0..challenge.num_machines).filter(|&m| m < ds.machine_seq.len() && ds.machine_seq[m].len() > 1).collect();
            m_rank.sort_by(|&a, &b| machine_total_pt[b].cmp(&machine_total_pt[a]));
            let num_bn_ls = m_rank.len().min(3);
            let mut any_improved = false;
            for bi in 0..num_bn_ls {
                let m = m_rank[bi];
                let seq_cap = ds.machine_seq[m].len().min(18);
                let mut found_improvement = true;
                while found_improvement {
                    found_improvement = false;
                    'swap_loop: for i in 0..seq_cap.saturating_sub(1) {
                        for j in (i+1)..seq_cap {
                            if j >= ds.machine_seq[m].len() { break; }
                            ds.machine_seq[m].swap(i, j);
                            if let Some((new_mk, _)) = eval_disj(&ds, &mut buf) {
                                if new_mk < cur_mk {
                                    cur_mk = new_mk;
                                    found_improvement = true;
                                    any_improved = true;
                                    break 'swap_loop;
                                }
                            }
                            ds.machine_seq[m].swap(i, j);
                        }
                    }
                }
            }
            if any_improved {
                if let Some((mk_bn, _)) = eval_disj(&ds, &mut buf) {
                    if let Ok(sol_bn) = disj_to_solution(pre, &ds, &buf.start) {
                        if mk_bn < best_makespan { best_makespan=mk_bn; best_solution=Some(sol_bn.clone()); save_solution(&sol_bn)?; }
                        push_top_solutions(&mut top_solutions, &sol_bn, mk_bn, 20);
                    }
                }
            }
        }
    }

    {
        let ils_pool: Vec<Solution> = top_solutions.iter().take(6).map(|(s,_)| s.clone()).collect();
        for base_sol in &ils_pool {
            if let Ok(Some((ls_sol, ls_mk))) = critical_block_move_local_search_ex(pre, challenge, base_sol, 5, 400, 120) {
                if ls_mk < best_makespan { best_makespan=ls_mk; best_solution=Some(ls_sol.clone()); save_solution(&ls_sol)?; }
                push_top_solutions(&mut top_solutions, &ls_sol, ls_mk, 20);
                if let Some((sol3, mk3)) = tabu_search_phase(pre, challenge, &ls_sol, ts_iters/2, ts_tenure)? {
                    if mk3 < best_makespan { best_makespan=mk3; best_solution=Some(sol3.clone()); save_solution(&sol3)?; }
                    push_top_solutions(&mut top_solutions, &sol3, mk3, 20);
                }
            }
        }
    }
    } // end Phase 1-5 cold start (else branch)

    {
        let num_machines = challenge.num_machines;

        let mut machine_rank: Vec<(usize, f64)> = (0..num_machines).map(|m| {
            let scar = if m < pre.machine_scarcity.len() { pre.machine_scarcity[m] } else { 1.0 };
            (m, scar)
        }).collect();
        machine_rank.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let num_bn = (num_machines / 5).max(3).min(8);
        let mut is_bottleneck = vec![false; num_machines];
        for i in 0..num_bn { is_bottleneck[machine_rank[i].0] = true; }

        let pop_cap = 10usize;
        // Diversity-seeded initial population: 2 best + up to 8 maximally diverse
        let mut mem_pop: Vec<(Solution, u32)> = Vec::with_capacity(pop_cap);
        for (s, mk) in top_solutions.iter().take(2) {
            mem_pop.push((s.clone(), *mk));
        }
        if !diverse_pool.is_empty() {
            let best_sig = machine_signature(&mem_pop[0].0, pre);
            let mut sorted_diverse: Vec<(usize, usize)> = diverse_pool.iter().enumerate()
                .map(|(i, (_, _, sig))| (i, hamming_distance(sig, &best_sig)))
                .collect();
            sorted_diverse.sort_unstable_by(|a, b| b.1.cmp(&a.1));
            for (idx, _dist) in sorted_diverse.iter().take(pop_cap - mem_pop.len()) {
                let (s, mk, _) = &diverse_pool[*idx];
                if !mem_pop.iter().any(|(_, pmk)| *pmk == *mk) {
                    mem_pop.push((s.clone(), *mk));
                }
            }
        }
        // Fill remaining slots from top_solutions
        for (s, mk) in top_solutions.iter() {
            if mem_pop.len() >= pop_cap { break; }
            if !mem_pop.iter().any(|(_, pmk)| *pmk == *mk) {
                mem_pop.push((s.clone(), *mk));
            }
        }

        let num_generations = 22usize;
        let mut gen_no_improve = 0usize;
        let max_gen_no_improve = 12usize;
        // Phase-isolated 4000ms budget: reduced from 5500ms to free ~48s for Phase 7 ILS
        let phase6_start = std::time::Instant::now();
        let phase6_budget_ms = 4000u128;

        for gen in 0..num_generations {
            if phase6_start.elapsed().as_millis() >= phase6_budget_ms { break; }
            if gen_no_improve >= max_gen_no_improve { break; }
            let cur_pop = mem_pop.len();
            if cur_pop < 2 { break; }

            let use_mutation = gen % 5 == 4;

            let ia = {
                let a = rng.gen_range(0..cur_pop);
                let b = rng.gen_range(0..cur_pop);
                if mem_pop[a].1 <= mem_pop[b].1 { a } else { b }
            };
            let ib = {
                let mut b = rng.gen_range(0..cur_pop);
                if b == ia { b = (b + 1) % cur_pop; }
                let c = rng.gen_range(0..cur_pop);
                let c = if c == ia { (c + 1) % cur_pop } else { c };
                if mem_pop[b].1 <= mem_pop[c].1 { b } else { c }
            };

            let (sol_a, mk_a) = mem_pop[ia].clone();
            let (sol_b, mk_b) = mem_pop[ib].clone();

            let ds_a = match build_disj_from_solution(pre, challenge, &sol_a) { Ok(d) => d, Err(_) => { gen_no_improve += 1; continue; } };
            let ds_b = match build_disj_from_solution(pre, challenge, &sol_b) { Ok(d) => d, Err(_) => { gen_no_improve += 1; continue; } };

            let (better_ds, worse_ds) = if mk_a <= mk_b { (&ds_a, &ds_b) } else { (&ds_b, &ds_a) };
            let mut child_ds = better_ds.clone();

            for m in 0..num_machines {
                if is_bottleneck[m] { continue; }
                if m >= worse_ds.machine_seq.len() || m >= child_ds.machine_seq.len() { continue; }
                if worse_ds.machine_seq[m].len() == child_ds.machine_seq[m].len() {
                    if rng.gen::<f64>() < 0.65 {
                        child_ds.machine_seq[m] = worse_ds.machine_seq[m].clone();
                    }
                }
            }

            if use_mutation {
                let non_bn_machines: Vec<usize> = (0..num_machines).filter(|&m| !is_bottleneck[m] && child_ds.machine_seq[m].len() > 1).collect();
                if !non_bn_machines.is_empty() {
                    for _ in 0..3 {
                        let m = non_bn_machines[rng.gen_range(0..non_bn_machines.len())];
                        let seq_len = child_ds.machine_seq[m].len();
                        if seq_len > 1 {
                            let pos = rng.gen_range(0..seq_len - 1);
                            child_ds.machine_seq[m].swap(pos, pos + 1);
                        }
                    }
                }
                let bn_machines: Vec<usize> = (0..num_machines).filter(|&m| is_bottleneck[m] && child_ds.machine_seq[m].len() > 1).collect();
                if !bn_machines.is_empty() {
                    let m = bn_machines[rng.gen_range(0..bn_machines.len())];
                    let seq_len = child_ds.machine_seq[m].len();
                    if seq_len > 1 {
                        let pos = rng.gen_range(0..seq_len - 1);
                        child_ds.machine_seq[m].swap(pos, pos + 1);
                    }
                }
            }

            let mut child_buf = EvalBuf::new(child_ds.n);
            let child_mk = match eval_disj(&child_ds, &mut child_buf) {
                Some((mk, _)) => mk,
                None => { gen_no_improve += 1; continue; }
            };
            let child_sol = match disj_to_solution(pre, &child_ds, &child_buf.start) {
                Ok(s) => s,
                Err(_) => { gen_no_improve += 1; continue; }
            };

            let (mut ls_sol, mut ls_mk) = match critical_block_move_local_search_ex(pre, challenge, &child_sol, 4, 250, 100) {
                Ok(Some((s, mk))) => (s, mk),
                _ => (child_sol, child_mk),
            };

            // Elite check: within 2% of the global best makespan
            let is_elite = ls_mk <= best_makespan + (best_makespan / 50);

            if is_elite {
                if let Ok(mut ds) = build_disj_from_solution(pre, challenge, &ls_sol) {
                    let mut buf = EvalBuf::new(ds.n);
                    if let Some((_, terminal)) = eval_disj(&ds, &mut buf) {
                        let mut crit = vec![false; ds.n];
                        let mut u = terminal;
                        while u != NONE_USIZE {
                            crit[u] = true;
                            u = buf.best_pred[u];
                        }
                        let mut blocks: Vec<Vec<usize>> = Vec::new();
                        for m in 0..num_machines {
                            if m >= ds.machine_seq.len() { continue; }
                            let seq = &ds.machine_seq[m];
                            let mut i = 0;
                            while i < seq.len() {
                                if !crit[seq[i]] { i += 1; continue; }
                                let block_start = i;
                                while i < seq.len() && crit[seq[i]] { i += 1; }
                                if i - block_start >= 2 {
                                    blocks.push(seq[block_start..i].to_vec());
                                }
                            }
                        }
                        if !blocks.is_empty() {
                            blocks.sort_by_key(|b| std::cmp::Reverse(b.len()));
                            let target_block = &blocks[rng.gen_range(0..blocks.len().min(2))];
                            let m = ds.node_machine[target_block[0]];
                            let target_start = target_block[0];
                            let target_end = *target_block.last().unwrap();
                            if let (Some(ps), Some(pe)) = (
                                ds.machine_seq[m].iter().position(|&x| x == target_start),
                                ds.machine_seq[m].iter().position(|&x| x == target_end)
                            ) {
                                if ps != pe {
                                    ds.machine_seq[m].swap(ps, pe);
                                    if eval_disj(&ds, &mut buf).is_some() {
                                        if let Ok(perturbed_sol) = disj_to_solution(pre, &ds, &buf.start) {
                                            if let Ok(Some((re_sol, re_mk))) = critical_block_move_local_search_ex(pre, challenge, &perturbed_sol, 3, 300, 120) {
                                                if re_mk < ls_mk {
                                                    ls_sol = re_sol;
                                                    ls_mk = re_mk;
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if ls_mk < best_makespan {
                best_makespan = ls_mk;
                best_solution = Some(ls_sol.clone());
                save_solution(&ls_sol)?;
                gen_no_improve = 0;
            } else {
                gen_no_improve += 1;
            }

            push_top_solutions(&mut top_solutions, &ls_sol, ls_mk, 20);

            if cur_pop >= pop_cap {
                let worst_idx = mem_pop.iter().enumerate().max_by_key(|(_, (_, mk))| *mk).map(|(i, _)| i).unwrap_or(cur_pop - 1);
                if ls_mk < mem_pop[worst_idx].1 {
                    mem_pop[worst_idx] = (ls_sol, ls_mk);
                }
            } else {
                mem_pop.push((ls_sol, ls_mk));
            }
        }

        let mem_best: Vec<Solution> = {
            let mut sorted = mem_pop.clone();
            sorted.sort_by_key(|(_, mk)| *mk);
            sorted.into_iter().take(3).map(|(s, _)| s).collect()
        };
        for base_sol in &mem_best {
            if let Some((ts_sol, ts_mk)) = tabu_search_phase(pre, challenge, base_sol, ts_iters / 3, ts_tenure)? {
                if ts_mk < best_makespan {
                    best_makespan = ts_mk;
                    best_solution = Some(ts_sol.clone());
                    save_solution(&ts_sol)?;
                }
                push_top_solutions(&mut top_solutions, &ts_sol, ts_mk, 20);
            }
        }
    }

    // Phase 7: Critical-Job Ruin-and-Recreate ILS (CJR-ILS)
    // Replaces the double-bridge phase.  Stronger structural perturbation with
    // exact reinsertion evaluation, repaired by CBMLS, intensified by tabu(20k).
    {
        let ils7_start = std::time::Instant::now();
        let ils7_budget_ms = 3500u128;
        let mut no_global_improve = 0usize;
        let pool_size = top_solutions.len().min(8);

        while ils7_start.elapsed().as_millis() < ils7_budget_ms && no_global_improve < 25 {
            let base = if pool_size > 2 && rng.gen_bool(0.30) {
                top_solutions[rng.gen_range(0..pool_size)].0.clone()
            } else {
                match best_solution.as_ref() { Some(s) => s.clone(), None => break }
            };

            let perturbed = match critical_job_ruin_recreate(pre, challenge, &base, &mut rng) {
                Ok(s) => s,
                Err(_) => { no_global_improve += 1; continue; }
            };

            // Fast repair: critical-block local search
            let (repaired_sol, _) = match critical_block_move_local_search_ex(pre, challenge, &perturbed, 4, 250, 100) {
                Ok(Some((s, _mk))) => (s, 0u32),
                _ => (perturbed, 0u32),
            };

            // Deep intensification with the proven tabu search (20k iterations)
            match tabu_search_phase(pre, challenge, &repaired_sol, 30000, ts_tenure) {
                Ok(Some((improved, mk))) => {
                    if mk < best_makespan {
                        best_makespan = mk;
                        best_solution = Some(improved.clone());
                        save_solution(&improved)?;
                        no_global_improve = 0;
                        push_top_solutions(&mut top_solutions, &improved, mk, 20);
                    } else {
                        no_global_improve += 1;
                    }
                }
                _ => { no_global_improve += 1; }
            }
        }
    }

    // Post-CJR controlled perturbation + short tabu (basin-hopping kick)
    // Apply 2 adjacent CP swaps to force out of CJR attractor, then tabu(10k)
    // Uses isolated post_rng to avoid polluting the main rng state (i90 regression fix)
    {
        if let Some(ref curr_best) = best_solution.clone() {
            if let Ok(mut ds) = build_disj_from_solution(pre, challenge, curr_best) {
                let mut buf = EvalBuf::new(ds.n);
                if let Some((_mk, terminal)) = eval_disj(&ds, &mut buf) {
                    let mut pos: Vec<usize> = vec![0; ds.n];
                    for m in 0..ds.machine_seq.len() {
                        for (i, &nd) in ds.machine_seq[m].iter().enumerate() {
                            if nd < pos.len() { pos[nd] = i; }
                        }
                    }
                    let mut swap_cands: Vec<(usize, usize)> = Vec::new();
                    let mut u = terminal;
                    while u != NONE_USIZE {
                        let m = ds.node_machine[u];
                        let p = pos[u];
                        if p > 0 { swap_cands.push((m, p - 1)); }
                        if p + 1 < ds.machine_seq[m].len() { swap_cands.push((m, p)); }
                        u = buf.best_pred[u];
                    }
                    swap_cands.sort_unstable();
                    swap_cands.dedup();
                    let num_swaps = 2.min(swap_cands.len());
                    let mut post_rng = SmallRng::from_seed([best_makespan as u8; 32]);
                    for _ in 0..num_swaps {
                        if swap_cands.is_empty() { break; }
                        let idx = post_rng.gen_range(0..swap_cands.len());
                        let (m, left_pos) = swap_cands.swap_remove(idx);
                        if left_pos + 1 < ds.machine_seq[m].len() {
                            ds.machine_seq[m].swap(left_pos, left_pos + 1);
                        }
                    }
                    if eval_disj(&ds, &mut buf).is_some() {
                        if let Ok(perturbed_sol) = disj_to_solution(pre, &ds, &buf.start) {
                            let post_tenure = (ds.n as f64 * 0.15) as usize;
                            if let Ok(Some((improved_sol, improved_mk))) =
                                tabu_n7_phase(pre, challenge, &perturbed_sol, 10_000, post_tenure)
                            {
                                if improved_mk < best_makespan {
                                    best_makespan = improved_mk;
                                    best_solution = Some(improved_sol.clone());
                                    save_solution(&improved_sol)?;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    if let Some(ref final_best) = best_solution.clone() {
        if let Some((sol4, mk4)) = tabu_search_phase(pre, challenge, final_best, ts_iters, ts_tenure)? {
            if mk4 < best_makespan {
                best_makespan = mk4;
                best_solution = Some(sol4.clone());
                save_solution(&sol4)?;
                push_top_solutions(&mut top_solutions, &sol4, mk4, 20);
            }
        }
    }

    // Path relinking: directed search between elite pool members.
    // 4k steps converge as well as 15k (quality plateau confirmed in i109).
    if let Ok(Some((pr_sol, pr_mk))) = path_relinking_phase(pre, challenge, &top_solutions, best_makespan, 4_000) {
        if pr_mk < best_makespan {
            best_makespan = pr_mk;
            best_solution = Some(pr_sol.clone());
            save_solution(&pr_sol)?;
            push_top_solutions(&mut top_solutions, &pr_sol, pr_mk, 20);
        }
    }

    // Kicked lazy N7 tabu: escapes N5 basin of PR output.
    // N5-only second tabu adds zero quality after PR (proven i108/i109) — replace with N7.
    if let Some(ref best_after_pr) = best_solution.clone() {
        if let Ok(mut ds) = build_disj_from_solution(pre, challenge, best_after_pr) {
            let mut buf = EvalBuf::new(ds.n);
            if let Some((_mk, terminal)) = eval_disj(&ds, &mut buf) {
                // Find critical blocks
                let mut crit = vec![false; ds.n];
                let mut u = terminal;
                while u != NONE_USIZE { crit[u] = true; u = buf.best_pred[u]; }
                let mut blocks: Vec<(usize, usize, usize)> = Vec::new();
                for m in 0..ds.num_machines {
                    let mut i = 0;
                    while i < ds.machine_seq[m].len() {
                        let node = ds.machine_seq[m][i];
                        if !crit[node] { i += 1; continue; }
                        let bs = i;
                        let mut be = i;
                        while be + 1 < ds.machine_seq[m].len() {
                            let x = ds.machine_seq[m][be];
                            let y = ds.machine_seq[m][be + 1];
                            if !crit[y] { break; }
                            if buf.start[y] != buf.start[x].saturating_add(ds.node_pt[x]) { break; }
                            be += 1;
                        }
                        if be > bs { blocks.push((m, bs, be)); }
                        i = be + 1;
                    }
                }
                // Perturb: swap ends of up to 3 critical blocks (deterministic seed)
                if !blocks.is_empty() {
                    let mut kick_rng = SmallRng::seed_from_u64(best_makespan as u64);
                    let n_kicks = 3.min(blocks.len());
                    for _ in 0..n_kicks {
                        if blocks.is_empty() { break; }
                        let idx = kick_rng.gen_range(0..blocks.len());
                        let (m, s, e) = blocks.swap_remove(idx);
                        if s < e { ds.machine_seq[m].swap(s, e); }
                    }
                }
                // Convert to solution and run lazy N7 tabu (15k iters — reduced from 20k to fit cycle)
                if let Ok(perturbed_sol) = disj_to_solution(pre, &ds, &buf.start) {
                    let n7_tenure = ((ds.n as f64 * 0.15) as usize).clamp(8, 24);
                    if let Ok(Some((n7_sol, n7_mk))) = tabu_n7_phase(pre, challenge, &perturbed_sol, 15_000, n7_tenure) {
                        if n7_mk < best_makespan {
                            best_makespan = n7_mk;
                            best_solution = Some(n7_sol.clone());
                            save_solution(&n7_sol)?;
                        }
                    }
                }
            }
        }
    }

    // D-ILS Schrage phase: perturb + full machine Schrage sweep to escape N5+N7 basin
    // Fuel-based (no wall-clock). num_sweeps = effort.job_shop_iters / 2000 ≈ 50 at default.
    {
        let num_sweeps = effort.job_shop_iters / 2000;
        if num_sweeps > 0 {
            if let Some(ref current_best) = best_solution.clone() {
                if let Ok(Some((dils_sol, dils_mk))) = dils_schrage_phase(pre, challenge, current_best, num_sweeps, &mut rng) {
                    if dils_mk < best_makespan {
                        best_makespan = dils_mk;
                        best_solution = Some(dils_sol.clone());
                        save_solution(&dils_sol)?;
                    }
                }
            }
        }
    }

    // i126: Store best solution in cross-nonce elite pool for subsequent nonces
    if let Some(ref best) = best_solution {
        let mk = best_makespan as u64;
        let mut pool = ELITE_POOL.lock().unwrap();
        if !pool.iter().any(|(_, m)| *m == mk) {
            pool.push((best.clone(), mk));
            pool.sort_by_key(|(_, m)| *m);
            pool.truncate(MAX_POOL_SIZE);
        }
    }

    if let Some(sol) = best_solution { save_solution(&sol)?; }
    Ok(())
}
