use anyhow::{anyhow, Result};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use tig_challenges::job_scheduling::*;
use super::types::*;
use super::infra_shared::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Rule {
    Adaptive, BnHeavy, EndTight, CriticalPath, MostWork, LeastFlex, Regret, ShortestProc, FlexBalance,
}

#[inline]
fn slack_urgency_fh(pre: &Pre, target_mk: Option<u32>, time: u32, product: usize, op_idx: usize) -> f64 {
    let Some(tgt) = target_mk else { return 0.0 };
    let lb = (time as u64).saturating_add(pre.product_suf_min[product][op_idx] as u64);
    let slack = (tgt as i64) - (lb as i64);
    let scale = (0.70 * pre.avg_op_min).max(1.0);
    let pos = (slack.max(0) as f64) / scale; let neg = ((-slack).max(0) as f64) / scale;
    (1.0 / (1.0 + pos)).clamp(0.0, 1.0) + (0.35 * neg).min(3.0)
}

struct RoutePrefCounts {
    ops: Vec<Vec<RouteOpPref>>,
}

struct RouteOpPref {
    total: u32,
    log_total_plus1: f64,
    counts: Vec<(usize, u32)>,
}

fn build_route_pref_counts_from_full(counts: &[Vec<Vec<u32>>]) -> RoutePrefCounts {
    let mut ops: Vec<Vec<RouteOpPref>> = Vec::with_capacity(counts.len());
    for prod_counts in counts.iter() {
        let mut prod_ops = Vec::with_capacity(prod_counts.len());
        for op_counts in prod_counts.iter() {
            let total = op_counts.iter().sum::<u32>();
            let log_total_plus1 = (total as f64 + 1.0).ln();
            let mut sparse: Vec<(usize, u32)> = Vec::new();
            for (m, &c) in op_counts.iter().enumerate() {
                if c > 0 {
                    sparse.push((m, c));
                }
            }
            prod_ops.push(RouteOpPref { total, log_total_plus1, counts: sparse });
        }
        ops.push(prod_ops);
    }
    RoutePrefCounts { ops }
}

fn build_single_solution_pref_counts(
    pre: &Pre,
    challenge: &Challenge,
    sol: &Solution,
) -> RoutePrefCounts {
    let num_products = pre.product_ops.len();
    let num_machines = challenge.num_machines;
    let mut counts: Vec<Vec<Vec<u32>>> = pre.product_ops.iter()
        .map(|ops| ops.iter().map(|_| vec![0u32; num_machines]).collect())
        .collect();
    for job in 0..challenge.num_jobs {
        let product = pre.job_products[job];
        if product >= num_products { continue; }
        let sched = &sol.job_schedule[job];
        let ops_len = counts[product].len();
        for (op_idx, &(m, _)) in sched.iter().enumerate() {
            if op_idx >= ops_len { break; }
            if m < num_machines {
                counts[product][op_idx][m] = counts[product][op_idx][m].saturating_add(1);
            }
        }
    }
    build_route_pref_counts_from_full(&counts)
}

#[inline]
fn route_pref_bonus_fh(pref: Option<&RoutePrefCounts>, product: usize, op_idx: usize, machine: usize) -> f64 {
    let Some(pref) = pref else { return 0.0 };
    if product >= pref.ops.len() || op_idx >= pref.ops[product].len() { return 0.0; }
    let op_pref = &pref.ops[product][op_idx];
    if op_pref.total == 0 { return 0.0; }
    let count = match op_pref.counts.binary_search_by_key(&machine, |&(m, _)| m) {
        Ok(i) => op_pref.counts[i].1,
        Err(_) => 0,
    };
    (count as f64 + 1.0).ln() / op_pref.log_total_plus1
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn score_candidate(
    pre: &Pre, rule: Rule, job: usize, product: usize, op_idx: usize,
    ops_rem: usize, op: &OpInfo, machine: usize, pt: u32, time: u32,
    target_mk: Option<u32>, best_end: u32, second_end: u32, best_cnt_total: usize,
    progress: f64, job_bias: f64, machine_penalty: f64, dynamic_load: f64,
    route_pref_counts: Option<&RoutePrefCounts>, route_w: f64, jitter: f64,
) -> f64 {
    let rem_min = pre.product_suf_min[product][op_idx] as f64;
    let rem_avg = pre.product_suf_avg[product][op_idx]; let rem_bn = pre.product_suf_bn[product][op_idx];
    let flex_f = (op.flex as f64).max(1.0); let flex_inv = 1.0/flex_f;
    let rem_min_n = rem_min/pre.horizon.max(1.0); let rem_avg_n = rem_avg/pre.max_job_avg_work.max(1e-9);
    let bn_n = rem_bn/pre.max_job_bn.max(1e-9); let ops_n = (ops_rem as f64)/(pre.max_ops as f64).max(1.0);
    let load_n = dynamic_load/pre.avg_machine_load.max(1e-9); let scar_n = pre.machine_scarcity[machine]/pre.avg_machine_scarcity.max(1e-9);
    let end_n = (best_end as f64)/pre.time_scale.max(1.0); let proc_n = (pt as f64)/pre.avg_op_min.max(1.0);
    let regret = if second_end >= INF { pre.avg_op_min*2.6 } else { (second_end-best_end) as f64 };
    let reg_n = (regret/pre.avg_op_min.max(1.0)).clamp(0.0,6.0);
    let scarcity_urg = 1.0/(best_cnt_total as f64).max(1.0);
    let density_n = ((rem_min/(ops_rem as f64).max(1.0))/pre.avg_op_min.max(1.0)).clamp(0.0,4.0);
    let next_min = pre.product_next_min[product][op_idx] as f64; let next_min_n = next_min/pre.horizon.max(1.0);
    let next_flex_inv = pre.product_next_flex_inv[product][op_idx];
    let p2 = progress*progress; let next_w_base = 0.12+p2*0.28;
    let next_term_raw = (0.55*next_min_n+0.45*next_flex_inv)*(1.0+0.30*density_n*pre.high_flex);
    let js = pre.jobshopness; let fl = 1.0-js;
    let avg_flex_inv = 1.0/pre.flex_avg.max(1.0); let scarce_match = scar_n*(flex_inv-avg_flex_inv);
    let mpen = machine_penalty.clamp(0.0,1.0); let mpen_gain = 1.0+0.85*pre.high_flex;
    let flow_term = pre.flow_w*pre.job_flow_pref[job]*(0.65+0.70*(1.0-progress));
    let slack_u = slack_urgency_fh(pre, target_mk, time, product, op_idx);
    let slack_w = pre.slack_base*(0.25+0.75*progress);
    let slack_reg_boost = 1.0 + 0.50 * reg_n / (1.0 + slack_u);
    let pop_pen = if pre.chaotic_like && op.flex >= 2 { let pop=pre.machine_best_pop[machine]; (0.07+0.15*(1.0-progress)).clamp(0.05,0.24)*pop*pre.flex_factor } else { 0.0 };
    let route_gain = (0.70+0.80*(1.0-progress)).clamp(0.70,1.40);
    let route_term = if route_w!=0.0 && op.flex>=2 { route_w*route_gain*route_pref_bonus_fh(route_pref_counts,product,op_idx,machine) } else { 0.0 };
    match rule {
        Rule::CriticalPath => { let next_term=next_w_base*0.30*next_term_raw; let slack_term=slack_w*slack_u*slack_reg_boost; (1.03*rem_min_n)+(0.10*ops_n)+(0.24*scarcity_urg)+(0.20*pre.flex_factor)*flex_inv+next_term+0.10*slack_term-(0.70*end_n)-pop_pen+(0.45*job_bias)+flow_term+route_term+jitter }
        Rule::MostWork => { let next_term=next_w_base*0.25*next_term_raw; (1.00*rem_avg_n)+(0.12*ops_n)+(0.18*scarcity_urg)+(0.15*pre.flex_factor)*flex_inv+next_term-(0.62*end_n)-pop_pen+(0.45*job_bias)+flow_term+route_term+jitter }
        Rule::LeastFlex => { let next_term=next_w_base*0.20*next_term_raw; (1.00*flex_inv)+(0.28*rem_min_n)+(0.22*scarcity_urg)+next_term-(0.55*end_n)-pop_pen+(0.35*job_bias)+flow_term+route_term+jitter }
        Rule::ShortestProc => { let next_term=next_w_base*0.20*next_term_raw; (-1.00*proc_n)+(0.25*rem_min_n)+(0.12*scarcity_urg)+next_term-(0.20*end_n)-pop_pen+(0.25*job_bias)+flow_term+route_term+jitter }
        Rule::Regret => { let next_term=next_w_base*0.25*next_term_raw; (1.05*reg_n)+(0.55*rem_min_n)+(0.22*scarcity_urg)+next_term-(0.68*end_n)-pop_pen+(0.35*job_bias)+flow_term+route_term+jitter }
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

#[inline]
fn bandit_context_idx(late_phase: bool, use_learn: bool) -> usize {
    ((late_phase as usize) << 1) | (use_learn as usize)
}

fn choose_rule_bandit(
    _rng: &mut SmallRng, rules: &[Rule], rule_sum_ctx: &[Vec<f64>], rule_tries_ctx: &[Vec<u32>],
    rule_sum_global: &[f64], rule_tries_global: &[u32], ctx: usize,
    _global_best: u32, margin: u32, _stuck: usize, _chaos_like: bool, _late_phase: bool,
) -> Rule {
    if rules.is_empty() { return Rule::Adaptive; }
    let ctx_total_tries: u32 = if ctx < rule_sum_ctx.len() && ctx < rule_tries_ctx.len() {
        rules.iter().map(|&r| rule_tries_ctx[ctx][rule_idx(r)]).sum()
    } else { 0 };
    let active_ctx = ctx < rule_sum_ctx.len() && ctx < rule_tries_ctx.len() && ctx_total_tries >= 4;
    let (sums, tries): (&[f64], &[u32]) = if active_ctx {
        (&rule_sum_ctx[ctx], &rule_tries_ctx[ctx])
    } else {
        (rule_sum_global, rule_tries_global)
    };
    let total_tries = tries.iter().fold(0u32, |a, &b| a.saturating_add(b)).max(1) as f64;

    let c = 0.4 * (margin as f64).max(1.0);

    let mut best_score = f64::NEG_INFINITY;
    let mut best_rule = rules[0];
    let mut any_valid = false;

    for &r in rules {
        let idx = rule_idx(r);
        let n = tries[idx].max(1) as f64;
        let avg = if tries[idx] > 0 { sums[idx] / n } else { f64::MAX };
        let ucb = -avg + c * (2.0 * total_tries.ln().max(0.0) / n).sqrt();
        if ucb > best_score || !any_valid {
            best_score = ucb;
            best_rule = r;
            any_valid = true;
        }
    }

    best_rule
}

fn construct_solution_conflict(
    challenge: &Challenge, pre: &Pre, rule: Rule, k: usize, target_mk: Option<u32>,
    rng: &mut SmallRng, job_bias: Option<&[f64]>, machine_penalty: Option<&[f64]>,
    route_pref_counts: Option<&RoutePrefCounts>, route_w: f64, diversify: bool,
) -> Result<(Solution, u32)> {
    let num_jobs=challenge.num_jobs; let num_machines=challenge.num_machines;
    let mut job_next_op=vec![0usize;num_jobs]; let mut job_ready_time=vec![0u32;num_jobs];
    let mut machine_avail=vec![0u32;num_machines]; let mut machine_load=pre.machine_load0.clone();
    let mut job_schedule: Vec<Vec<(usize,u32)>>=pre.job_ops_len.iter().map(|&len|Vec::with_capacity(len)).collect();
    let mut remaining_ops=pre.total_ops; let mut time=0u32;
    let mut demand: Vec<u16>=vec![0u16;num_machines];
    let mut raw_rigid_by_machine: Vec<Vec<RawCand>>=(0..num_machines).map(|_|Vec::with_capacity(3)).collect();
    let mut raw_gen_by_machine: Vec<Vec<RawCand>>=(0..num_machines).map(|_|Vec::with_capacity(9)).collect();
    let mut idle_machines: Vec<usize>=Vec::with_capacity(num_machines);
    let chaotic_like=pre.chaotic_like;
    let mut machine_work: Vec<u64>=if chaotic_like{vec![0u64;num_machines]}else{vec![]};
    let mut sum_work: u64=0;
    while remaining_ops > 0 {
        loop {
            idle_machines.clear();
            for m in 0..num_machines { if machine_avail[m]<=time { idle_machines.push(m); } }
            if idle_machines.is_empty() { break; }
            for &m in &idle_machines { demand[m]=0; raw_rigid_by_machine[m].clear(); raw_gen_by_machine[m].clear(); }
            let progress=1.0-(remaining_ops as f64)/(pre.total_ops as f64).max(1.0);
            let cap_per_machine=if k==0{12usize}else{(k+6).min(12)};
            let rigid_cap=if cap_per_machine>=10{3usize}else{2usize};
            let gen_cap=cap_per_machine-rigid_cap;
            for job in 0..num_jobs {
                let op_idx=job_next_op[job]; if op_idx>=pre.job_ops_len[job]||job_ready_time[job]>time{continue;}
                let product=pre.job_products[job]; let op=&pre.product_ops[product][op_idx];
                if op.flex==0||op.machines.is_empty()||op.min_pt>=INF{continue;}
                let (best_end,second_end,best_cnt_total,best_cnt_idle)=best_second_and_counts(time,&machine_avail,op);
                if best_end>=INF||best_cnt_idle==0{continue;}
                let ops_rem=pre.job_ops_len[job]-op_idx; let jb=job_bias.map(|v|v[job]).unwrap_or(0.0);
                let flex_inv=1.0/(op.flex as f64).max(1.0); let scarcity_urg=1.0/(best_cnt_total as f64).max(1.0);
                let regret=if second_end>=INF{pre.avg_op_min*2.6}else{(second_end-best_end) as f64};
                let regn=(regret/pre.avg_op_min.max(1.0)).clamp(0.0,6.0); let rigidity=(0.60*flex_inv+0.40*scarcity_urg).clamp(0.0,2.5);
                for &(m,pt) in &op.machines {
                    if machine_avail[m]>time{continue;}
                    let end=time.saturating_add(pt); if end!=best_end{continue;}
                    demand[m]=demand[m].saturating_add(1);
                    let mp = if diversify {
                        machine_penalty.map(|v| v[m] * 1.5).unwrap_or(0.0)
                    } else {
                        machine_penalty.map(|v| v[m]).unwrap_or(0.0)
                    };
                    let jitter=if k>0{rng.gen::<f64>()*1e-9}else{0.0};
                    let base=score_candidate(pre,rule,job,product,op_idx,ops_rem,op,m,pt,time,target_mk,best_end,second_end,best_cnt_total,progress,jb,mp,machine_load[m],route_pref_counts,route_w,jitter);
                    let rc=RawCand{job,machine:m,pt,base_score:base,rigidity,reg_n:regn};
                    if op.flex<=2||best_cnt_total<=2||rigidity>=0.62{push_top_k_raw(&mut raw_rigid_by_machine[m],rc,rigid_cap);}else{push_top_k_raw(&mut raw_gen_by_machine[m],rc,gen_cap);}
                }
            }
            let denom=(idle_machines.len() as f64).max(1.0);
            let (mut conflict_w,conflict_scale)=if chaotic_like{(-(0.05+0.08*(1.0-progress)).clamp(0.04,0.14),(0.95+0.20*pre.flex_factor).clamp(0.90,1.20))}else{((0.09+0.26*pre.jobshopness+0.11*pre.high_flex+0.16*(1.0-progress)).clamp(0.05,0.45),(0.90+0.40*pre.flex_factor).clamp(0.85,1.75))};
            if diversify {
                conflict_w += 0.2 * (1.0 - progress);
            }
            let (bal_w,avg_work)=if chaotic_like{((0.030+0.070*(1.0-progress)).clamp(0.025,0.11),(sum_work as f64)/(num_machines as f64).max(1.0))}else{(0.0,0.0)};
            let mut best: Option<Cand>=None; let mut top: Vec<Cand>=if k>0{Vec::with_capacity(k)}else{Vec::new()};
            for &m in &idle_machines {
                let dem=demand[m] as f64; if dem<=0.0||(raw_rigid_by_machine[m].is_empty()&&raw_gen_by_machine[m].is_empty()){continue;}
                let dem_n=((dem-1.0)/denom).clamp(0.0,2.5);
                let bal_pen=if chaotic_like&&bal_w>0.0{let denomw=(avg_work+(pre.avg_op_min*3.0).max(1.0)).max(1.0); let r=(machine_work[m] as f64)/denomw; let done_n=(r/(r+1.0)).clamp(0.0,1.0); -bal_w*done_n}else{0.0};
                for rc in raw_rigid_by_machine[m].iter().chain(raw_gen_by_machine[m].iter()) {
                    let rig=rc.rigidity.clamp(0.0,2.5); let regc=rc.reg_n.clamp(0.0,4.5);
                    let mut boost=conflict_w*conflict_scale*dem_n*(1.15*rig+0.85*regc);
                    if chaotic_like{boost=boost.max(-0.26);}
                    let c=Cand{job:rc.job,machine:rc.machine,pt:rc.pt,score:rc.base_score+boost+bal_pen};
                    if k==0{if best.map_or(true,|bb|c.score>bb.score){best=Some(c);}}else{push_top_k(&mut top,c,k);}
                }
            }
            let chosen=if k==0{match best{Some(c)=>c,None=>break}}else{if top.is_empty(){break;}choose_from_top_weighted(rng,&top)};
            let job=chosen.job; let machine=chosen.machine; let pt=chosen.pt;
            let product=pre.job_products[job]; let op_idx=job_next_op[job]; let op=&pre.product_ops[product][op_idx];
            let (best_end_now,_,_,_)=best_second_and_counts(time,&machine_avail,op);
            let end_check=time.max(machine_avail[machine]).saturating_add(pt);
            if machine_avail[machine]>time||end_check!=best_end_now{break;}
            let end_time=time.saturating_add(pt);
            job_schedule[job].push((machine,time)); job_next_op[job]+=1; job_ready_time[job]=end_time; machine_avail[machine]=end_time; remaining_ops-=1;
            if chaotic_like{machine_work[machine]=machine_work[machine].saturating_add(pt as u64);sum_work=sum_work.saturating_add(pt as u64);}
            if op.min_pt<INF&&op.flex>0&&!op.machines.is_empty(){let delta=(op.min_pt as f64)/(op.flex as f64).max(1.0);if delta>0.0{for &(mm,_) in &op.machines{let v=machine_load[mm]-delta;machine_load[mm]=if v>0.0{v}else{0.0};}}}
            if remaining_ops==0{break;}
        }
        if remaining_ops==0{break;}
        let mut next_time: Option<u32>=None;
        for &t in &machine_avail{if t>time{next_time=Some(next_time.map_or(t,|b|b.min(t)));}}
        for j in 0..num_jobs{let op_idx=job_next_op[j];if op_idx<pre.job_ops_len[j]&&job_ready_time[j]>time{let t=job_ready_time[j];next_time=Some(next_time.map_or(t,|b|b.min(t)));}}
        time=next_time.ok_or_else(||anyhow!("Stalled"))?;
    }
    let mk=machine_avail.into_iter().max().unwrap_or(0);
    Ok((Solution{job_schedule},mk))
}

#[inline]
fn boundary_insertion_positions(
    ds: &DisjSchedule,
    starts: &[u32],
    job_op_to_node: &[Vec<usize>],
    node: usize,
    target_m: usize,
    cap: usize,
) -> Vec<usize> {
    if cap == 0 { return Vec::new(); }
    let job = ds.node_job[node];
    let op_idx = ds.node_op[node];
    let tlen = ds.machine_seq[target_m].len();
    let cur_start = starts[node];
    let node_pt = ds.node_pt[node];

    let mut pred_finish = 0u32;
    if op_idx > 0 && op_idx - 1 < job_op_to_node[job].len() {
        let pred = job_op_to_node[job][op_idx - 1];
        if pred != usize::MAX { pred_finish = starts[pred].saturating_add(ds.node_pt[pred]); }
    }

    let mut latest_start = u32::MAX;
    if op_idx + 1 < job_op_to_node[job].len() {
        let succ = job_op_to_node[job][op_idx + 1];
        if succ != usize::MAX { latest_start = starts[succ].saturating_sub(node_pt); }
    }

    let mut left_pos = 0usize;
    let mut right_pos = tlen;
    let mut cur_anchor = 0usize;
    for (k, &nd) in ds.machine_seq[target_m].iter().enumerate() {
        let finish = starts[nd].saturating_add(ds.node_pt[nd]);
        if finish <= pred_finish { left_pos = k + 1; }
        if finish <= cur_start { cur_anchor = k + 1; }
        if right_pos == tlen && latest_start != u32::MAX && finish > latest_start { right_pos = k; }
    }

    let lo = left_pos.min(right_pos);
    let hi = left_pos.max(right_pos);
    if cur_anchor < lo { cur_anchor = lo; } else if cur_anchor > hi { cur_anchor = hi; }

    let mut positions: Vec<usize> = Vec::with_capacity(cap.min(3));
    for &p in &[left_pos, right_pos, cur_anchor] {
        if p <= tlen && !positions.contains(&p) {
            positions.push(p);
            if positions.len() >= cap { break; }
        }
    }
    positions
}

#[inline]
fn bottleneck_zone_nodes(
    pre: &Pre,
    ds: &DisjSchedule,
    starts: &[u32],
    job_op_to_node: &[Vec<usize>],
    current_mk: u32,
) -> Vec<usize> {
    let n=ds.n;
    if n<=80 { return (0..n).collect(); }
    let num_machines=ds.machine_seq.len();
    let mut finish=vec![0u32;n];
    let mut machine_end=vec![0u32;num_machines];
    for nd in 0..n {
        let f=starts[nd].saturating_add(ds.node_pt[nd]);
        finish[nd]=f;
        let m=ds.node_machine[nd];
        if f>machine_end[m]{machine_end[m]=f;}
    }
    let tail_slack=((pre.avg_op_min*(1.80+1.20*pre.high_flex+0.85*pre.jobshopness)).max(1.0)) as u32;
    let edge=current_mk.saturating_sub(tail_slack);
    let mut zone_machines: Vec<usize>=(0..num_machines).filter(|&m| machine_end[m]>=edge).collect();
    if zone_machines.is_empty() {
        if let Some((m,_))=machine_end.iter().enumerate().max_by_key(|&(_, &end)| end) { zone_machines.push(m); }
    }
    let mut seed=vec![false;n]; let mut seed_nodes: Vec<usize>=Vec::with_capacity(zone_machines.len()*4);
    for &m in &zone_machines {
        let seq=&ds.machine_seq[m]; let mut added=0usize;
        for &nd in seq.iter().rev() {
            if finish[nd]>=edge||starts[nd]>=edge {
                if !seed[nd]{seed[nd]=true;seed_nodes.push(nd);}
                added+=1;
                if added>=4{break;}
            }
        }
        if added==0 {
            for &nd in seq.iter().rev().take(2) {
                if !seed[nd]{seed[nd]=true;seed_nodes.push(nd);}
            }
        }
    }
    let mut seen=vec![false;n]; let mut nodes: Vec<usize>=Vec::with_capacity(seed_nodes.len()*3);
    for &nd in &seed_nodes {
        if !seen[nd]{seen[nd]=true;nodes.push(nd);}
        let job=ds.node_job[nd]; let op_idx=ds.node_op[nd];
        if op_idx>0&&op_idx-1<job_op_to_node[job].len() {
            let pred=job_op_to_node[job][op_idx-1];
            if pred!=usize::MAX&&!seen[pred]{seen[pred]=true;nodes.push(pred);}
        }
        if op_idx+1<job_op_to_node[job].len() {
            let succ=job_op_to_node[job][op_idx+1];
            if succ!=usize::MAX&&!seen[succ]{seen[succ]=true;nodes.push(succ);}
        }
    }
    if nodes.is_empty() { return (0..n).collect(); }
    nodes.sort_by(|&a,&b| finish[b].cmp(&finish[a]).then_with(|| starts[b].cmp(&starts[a])));
    nodes
}

#[inline]
fn bottleneck_completion_profile(
    ds: &DisjSchedule,
    starts: &[u32],
    tail_span: u32,
) -> (u32, u32, u32, u32) {
    let mut first_end = 0u32;
    let mut second_end = 0u32;
    let mut first_m = usize::MAX;
    let mut second_m = usize::MAX;

    for (machine, seq) in ds.machine_seq.iter().enumerate() {
        let end = match seq.last() {
            Some(&node) => starts[node].saturating_add(ds.node_pt[node]),
            None => 0,
        };
        if end > first_end {
            second_end = first_end;
            second_m = first_m;
            first_end = end;
            first_m = machine;
        } else if end > second_end {
            second_end = end;
            second_m = machine;
        }
    }

    let tail_edge = first_end.saturating_sub(tail_span);
    let mut tail_count = 0u32;
    let mut tail_work = 0u32;
    for machine in [first_m, second_m] {
        if machine == usize::MAX || machine == second_m && second_m == first_m {
            continue;
        }
        for &node in ds.machine_seq[machine].iter().rev().take(24) {
            let pt = ds.node_pt[node];
            if starts[node].saturating_add(pt) >= tail_edge {
                tail_count = tail_count.saturating_add(1);
                tail_work = tail_work.saturating_add(pt);
            }
        }
    }

    (
        first_end,
        first_end.saturating_sub(second_end),
        tail_count,
        tail_work,
    )
}

#[inline]
fn insertion_makespan_lb(
    pre: &Pre,
    ds: &DisjSchedule,
    starts: &[u32],
    job_op_to_node: &[Vec<usize>],
    node: usize,
    target_m: usize,
    pos: usize,
    new_pt: u32,
) -> u32 {
    let job = ds.node_job[node];
    let op_idx = ds.node_op[node];
    let seq = &ds.machine_seq[target_m];
    let mut pred_finish = 0u32;
    if op_idx > 0 && op_idx - 1 < job_op_to_node[job].len() {
        let pred = job_op_to_node[job][op_idx - 1];
        if pred != usize::MAX {
            pred_finish = starts[pred].saturating_add(ds.node_pt[pred]);
        }
    }
    let left_finish = if pos == 0 || seq.is_empty() {
        0u32
    } else {
        let li = (pos - 1).min(seq.len() - 1);
        let left = seq[li];
        starts[left].saturating_add(ds.node_pt[left])
    };
    let est_start = pred_finish.max(left_finish);
    let est_end = est_start.saturating_add(new_pt);
    let mut lb = est_end;
    if pos < seq.len() {
        let right = seq[pos];
        let right_finish_lb = est_end.saturating_add(ds.node_pt[right]);
        if right_finish_lb > lb {
            lb = right_finish_lb;
        }
    }
    let product = pre.job_products[job];
    if product < pre.product_suf_min.len() {
        let suf = &pre.product_suf_min[product];
        if op_idx + 1 < suf.len() {
            let chain = est_end.saturating_add(suf[op_idx + 1]);
            if chain > lb {
                lb = chain;
            }
        }
    }
    lb
}

fn pair_option_prefilter(
    pre: &Pre,
    ds: &DisjSchedule,
    starts: &[u32],
    job_op_to_node: &[Vec<usize>],
    node: usize,
    bottleneck_m: usize,
    max_alternatives: usize,
) -> Vec<(usize, u32)> {
    let cur_m = ds.node_machine[node];
    let cur_pt = ds.node_pt[node];
    let mut keep = Vec::with_capacity(max_alternatives.saturating_add(1));
    keep.push((cur_m, cur_pt));
    if max_alternatives == 0 {
        return keep;
    }

    let job = ds.node_job[node];
    let op_idx = ds.node_op[node];
    let product = pre.job_products[job];
    if product >= pre.product_ops.len() || op_idx >= pre.product_ops[product].len() {
        return keep;
    }
    let op_info = &pre.product_ops[product][op_idx];
    if op_info.machines.len() <= 1 {
        return keep;
    }

    let mut pred_finish = 0u32;
    if op_idx > 0 && op_idx - 1 < job_op_to_node[job].len() {
        let pred = job_op_to_node[job][op_idx - 1];
        if pred != usize::MAX {
            pred_finish = starts[pred].saturating_add(ds.node_pt[pred]);
        }
    }
    let mut succ_start = u32::MAX;
    if op_idx + 1 < job_op_to_node[job].len() {
        let succ = job_op_to_node[job][op_idx + 1];
        if succ != usize::MAX {
            succ_start = starts[succ];
        }
    }

    let mut scored: Vec<(u64, u32, u32, usize, u32, u32)> =
        Vec::with_capacity(op_info.machines.len());
    for &(machine, pt) in &op_info.machines {
        if machine == cur_m {
            continue;
        }
        let positions = boundary_insertion_positions(
            ds,
            starts,
            job_op_to_node,
            node,
            machine,
            2,
        );
        if positions.is_empty() {
            continue;
        }

        let mut best_score = u64::MAX;
        let mut best_est_end = u32::MAX;
        let scarcity_penalty = (pre.avg_op_min.max(1.0)
            * (pre.machine_scarcity[machine] / pre.avg_machine_scarcity.max(1e-9))
                .clamp(0.0, 3.0)) as u32;
        let bottleneck_burden = if machine == bottleneck_m { 1 } else { 0 };
        for pos in positions {
            let lb = insertion_makespan_lb(
                pre,
                ds,
                starts,
                job_op_to_node,
                node,
                machine,
                pos,
                pt,
            );
            let seq = &ds.machine_seq[machine];
            let left_finish = if pos == 0 {
                0
            } else if let Some(&left) = seq.get(pos - 1) {
                starts[left].saturating_add(ds.node_pt[left])
            } else {
                0
            };
            let est_end = pred_finish.max(left_finish).saturating_add(pt);
            let successor_pressure = if succ_start == u32::MAX {
                0
            } else {
                est_end.saturating_sub(succ_start)
            };
            let score = (lb as u64)
                .saturating_mul(8)
                .saturating_add((successor_pressure as u64).saturating_mul(5))
                .saturating_add((bottleneck_burden as u64).saturating_mul(pre.avg_op_min.max(1.0) as u64))
                .saturating_add(scarcity_penalty as u64);
            if score < best_score || score == best_score && est_end < best_est_end {
                best_score = score;
                best_est_end = est_end;
            }
        }
        if best_score != u64::MAX {
            scored.push((
                best_score,
                best_est_end,
                pt,
                machine,
                scarcity_penalty,
                bottleneck_burden,
            ));
        }
    }

    let mut nondominated: Vec<(u64, u32, u32, usize, u32, u32)> =
        Vec::with_capacity(scored.len());
    for i in 0..scored.len() {
        let candidate = scored[i];
        let dominated = scored.iter().enumerate().any(|(j, other)| {
            j != i
                && other.1 <= candidate.1
                && other.2 <= candidate.2
                && other.4 <= candidate.4
                && other.5 <= candidate.5
                && (other.1 < candidate.1
                    || other.2 < candidate.2
                    || other.4 < candidate.4
                    || other.5 < candidate.5)
        });
        if !dominated {
            nondominated.push(candidate);
        }
    }
    nondominated.sort_by(|a, b| {
        a.0.cmp(&b.0)
            .then_with(|| a.1.cmp(&b.1))
            .then_with(|| a.3.cmp(&b.3))
            .then_with(|| a.2.cmp(&b.2))
    });
    for (_, _, pt, machine, _, _) in nondominated.into_iter().take(max_alternatives) {
        keep.push((machine, pt));
    }
    keep
}

#[inline]
fn removable_delay_scores(
    ds: &DisjSchedule,
    starts: &[u32],
    job_op_to_node: &[Vec<usize>],
    makespan: u32,
) -> Vec<u64> {
    let n = ds.n;
    let mut machine_next = vec![usize::MAX; n];
    let mut tail_rank = vec![0u32; n];
    for seq in &ds.machine_seq {
        let len = seq.len();
        for (pos, &node) in seq.iter().enumerate() {
            if pos + 1 < len {
                machine_next[node] = seq[pos + 1];
            }
            tail_rank[node] = (len.saturating_sub(pos).min(16)) as u32;
        }
    }

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| starts[b].cmp(&starts[a]).then_with(|| b.cmp(&a)));
    let mut latest_start = vec![makespan; n];
    let mut slack = vec![0u32; n];
    let mut scores = vec![0u64; n];

    for node in order {
        let job = ds.node_job[node];
        let op_idx = ds.node_op[node];
        let mut latest_finish = makespan;
        let mut tight_successors = 0u32;

        if op_idx + 1 < job_op_to_node[job].len() {
            let succ = job_op_to_node[job][op_idx + 1];
            if succ != usize::MAX {
                latest_finish = latest_finish.min(latest_start[succ]);
                if slack[succ] == 0 {
                    tight_successors = tight_successors.saturating_add(1);
                }
            }
        }
        let machine_succ = machine_next[node];
        if machine_succ != usize::MAX {
            latest_finish = latest_finish.min(latest_start[machine_succ]);
            if slack[machine_succ] == 0 {
                tight_successors = tight_successors.saturating_add(1);
            }
        }

        let latest = latest_finish.saturating_sub(ds.node_pt[node]);
        latest_start[node] = latest;
        let node_slack = latest.saturating_sub(starts[node]);
        slack[node] = node_slack;
        let urgency = makespan.saturating_sub(node_slack) as u64;
        scores[node] = (tight_successors as u64)
            .saturating_mul(1_000_000)
            .saturating_add((tail_rank[node] as u64).saturating_mul(10_000))
            .saturating_add((ds.node_pt[node] as u64).saturating_mul(10))
            .saturating_add(urgency);
    }
    scores
}

fn iterative_cp_descent(
    pre: &Pre,
    challenge: &Challenge,
    base_sol: &Solution,
    max_iters: usize,
) -> Result<Option<(Solution, u32)>> {
    let mut ds = build_disj_from_solution(pre, challenge, base_sol)?;
    let mut buf = EvalBuf::new(ds.n);
    let Some((mut current_mk, _)) = eval_disj(&ds, &mut buf) else { return Ok(None); };
    let initial_mk = current_mk;
    let n = ds.n;

    let mut job_op_to_node: Vec<Vec<usize>> = vec![vec![]; challenge.num_jobs];
    for nd in 0..n {
        let job = ds.node_job[nd];
        let op_idx = ds.node_op[nd];
        if op_idx >= job_op_to_node[job].len() {
            job_op_to_node[job].resize(op_idx + 1, usize::MAX);
        }
        job_op_to_node[job][op_idx] = nd;
    }


    let use_zone = n > 80;
    let mut greedy_passes = 0;
    let max_greedy_passes = if use_zone { 3 } else { 1 };
    while greedy_passes < max_greedy_passes {
        greedy_passes += 1;
        let candidate_nodes: Vec<usize> = if use_zone {
            bottleneck_zone_nodes(pre, &ds, &buf.start, &job_op_to_node, current_mk)
        } else {
            (0..n).collect()
        };
        for node in candidate_nodes {
            let job = ds.node_job[node];
            let op_idx = ds.node_op[node];
            let product = pre.job_products[job];
            if op_idx >= pre.product_ops[product].len() { continue; }
            let op_info = &pre.product_ops[product][op_idx];
            if op_info.machines.len() <= 1 { continue; }
            let cur_m = ds.node_machine[node];
            let cur_pt = ds.node_pt[node];
            let mut best_m = cur_m;
            let mut best_pt = cur_pt;
            let mut best_mk = current_mk;
            let mut best_ins = 0usize;

            for &(new_m, new_pt) in &op_info.machines {
                if new_m == cur_m { continue; }
                let Some(old_pos) = ds.machine_seq[cur_m].iter().position(|&x| x == node) else { continue };
                ds.machine_seq[cur_m].remove(old_pos);
                ds.node_machine[node] = new_m;
                ds.node_pt[node] = new_pt;
                let positions = boundary_insertion_positions(&ds, &buf.start, &job_op_to_node, node, new_m, 3);
                for &pos in &positions {
                    let lb = insertion_makespan_lb(pre, &ds, &buf.start, &job_op_to_node, node, new_m, pos, new_pt);
                    if lb >= best_mk { continue; }
                    ds.machine_seq[new_m].insert(pos, node);
                    if let Some((tmk, _)) = eval_disj(&ds, &mut buf) {
                        if tmk < best_mk {
                            best_mk = tmk;
                            best_m = new_m;
                            best_pt = new_pt;
                            best_ins = pos;
                        }
                    }
                    ds.machine_seq[new_m].remove(pos);
                }
                ds.machine_seq[cur_m].insert(old_pos, node);
                ds.node_machine[node] = cur_m;
                ds.node_pt[node] = cur_pt;
            }

            let Some(old_pos) = ds.machine_seq[cur_m].iter().position(|&x| x == node) else { continue };
            ds.machine_seq[cur_m].remove(old_pos);
            let positions = boundary_insertion_positions(&ds, &buf.start, &job_op_to_node, node, cur_m, 4);
            for &pos in &positions {
                let lb = insertion_makespan_lb(pre, &ds, &buf.start, &job_op_to_node, node, cur_m, pos, cur_pt);
                if lb >= best_mk { continue; }
                ds.machine_seq[cur_m].insert(pos, node);
                if let Some((tmk, _)) = eval_disj(&ds, &mut buf) {
                    if tmk < best_mk {
                        best_mk = tmk;
                        best_m = cur_m;
                        best_pt = cur_pt;
                        best_ins = pos;
                    }
                }
                ds.machine_seq[cur_m].remove(pos);
            }
            ds.machine_seq[cur_m].insert(old_pos, node);

            if best_mk < current_mk {
                let Some(old_pos) = ds.machine_seq[cur_m].iter().position(|&x| x == node) else { continue };
                ds.machine_seq[cur_m].remove(old_pos);
                if best_m != cur_m {
                    let ins = best_ins.min(ds.machine_seq[best_m].len());
                    ds.machine_seq[best_m].insert(ins, node);
                    ds.node_machine[node] = best_m;
                    ds.node_pt[node] = best_pt;
                } else {
                    let ins = best_ins.min(ds.machine_seq[cur_m].len());
                    ds.machine_seq[cur_m].insert(ins, node);
                }
                if let Some((mk_after, _)) = eval_disj(&ds, &mut buf) {
                    current_mk = mk_after;
                } else {
                    current_mk = best_mk;
                }
                if use_zone { break; }
            }
        }
    }

    for _iter in 0..max_iters {
        let mut finish = vec![0u32; n];
        for nd in 0..n { finish[nd] = buf.start[nd].saturating_add(ds.node_pt[nd]); }
        let makespan = current_mk;

        let mut on_cp = vec![false; n];
        for nd in 0..n { if finish[nd] == makespan { on_cp[nd] = true; } }

        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| buf.start[b].cmp(&buf.start[a]));

        for &nd in &order {
            if !on_cp[nd] { continue; }
            let job = ds.node_job[nd];
            let op_idx = ds.node_op[nd];
            if op_idx > 0 {
                let pred_op = op_idx - 1;
                if pred_op < job_op_to_node[job].len() {
                    let pred = job_op_to_node[job][pred_op];
                    if pred != usize::MAX && finish[pred] == buf.start[nd] { on_cp[pred] = true; }
                }
            }
            let machine = ds.node_machine[nd];
            let seq = &ds.machine_seq[machine];
            if let Some(pos) = seq.iter().position(|&x| x == nd) {
                if pos > 0 {
                    let pred = seq[pos - 1];
                    if finish[pred] == buf.start[nd] { on_cp[pred] = true; }
                }
            }
        }

        let mut cp_flex: Vec<usize> = (0..n).filter(|&nd| {
            if !on_cp[nd] { return false; }
            let job = ds.node_job[nd]; let op_idx = ds.node_op[nd];
            let product = pre.job_products[job];
            op_idx < pre.product_ops[product].len() && pre.product_ops[product][op_idx].machines.len() > 1
        }).collect();
        cp_flex.sort_by_key(|&nd| buf.start[nd]);

        if cp_flex.is_empty() { break; }
        let delay_score = removable_delay_scores(&ds, &buf.start, &job_op_to_node, current_mk);
        let cp_pair = cp_flex.clone();
        let mut machine_cp_count = vec![0usize; challenge.num_machines];
        for &nd in &cp_flex { machine_cp_count[ds.node_machine[nd]] += 1; }
        let plateau_bottleneck_m = machine_cp_count.iter().enumerate()
            .max_by_key(|&(_, &c)| c).map(|(m, _)| m).unwrap_or(0);
        cp_flex.sort_by(|&a, &b| {
            let ja = ds.node_job[a]; let oa = ds.node_op[a]; let pa = pre.job_products[ja];
            let jb = ds.node_job[b]; let ob = ds.node_op[b]; let pb = pre.job_products[jb];
            let fa = pre.product_ops[pa][oa].machines.len();
            let fb = pre.product_ops[pb][ob].machines.len();
            machine_cp_count[ds.node_machine[b]].cmp(&machine_cp_count[ds.node_machine[a]])
                .then_with(|| ds.node_pt[b].cmp(&ds.node_pt[a]))
                .then_with(|| fa.cmp(&fb))
                .then_with(|| buf.start[a].cmp(&buf.start[b]))
        });

        let mut single_cp_flex = cp_flex.clone();
        if n > 80 && single_cp_flex.len() > 32 {
            single_cp_flex.sort_by(|&a, &b| {
                let leverage = |nd: usize| {
                    let job = ds.node_job[nd];
                    let op_idx = ds.node_op[nd];
                    let product = pre.job_products[job];
                    let alternatives = pre.product_ops[product][op_idx]
                        .machines
                        .len()
                        .saturating_sub(1);
                    let successor_pressure = if op_idx + 1 < job_op_to_node[job].len() {
                        let succ = job_op_to_node[job][op_idx + 1];
                        if succ != usize::MAX && on_cp[succ] {
                            ds.node_pt[succ]
                        } else {
                            0
                        }
                    } else {
                        0
                    };
                    (
                        machine_cp_count[ds.node_machine[nd]],
                        ds.node_pt[nd],
                        finish[nd],
                        successor_pressure,
                        alternatives,
                    )
                };
                delay_score[b].cmp(&delay_score[a])
                    .then_with(|| leverage(b).cmp(&leverage(a)))
                    .then_with(|| buf.start[b].cmp(&buf.start[a]))
                    .then_with(|| a.cmp(&b))
            });
            single_cp_flex.truncate(32);
        }

        let mut iter_improved = false;
        let mut start_snapshot = buf.start.clone();
        let tail_slack = ((pre.avg_op_min * (1.40 + 0.80 * pre.high_flex + 0.60 * pre.jobshopness)).max(1.0)) as u32;
        let mut current_bottleneck_profile = bottleneck_completion_profile(&ds, &start_snapshot, tail_slack);
        let mut neutral_left = if n > 140 { 1usize } else { 2usize };

        for &nd in &single_cp_flex {
            let job = ds.node_job[nd]; let op_idx = ds.node_op[nd];
            let product = pre.job_products[job];
            if op_idx >= pre.product_ops[product].len() { continue; }
            let op_info = &pre.product_ops[product][op_idx];
            let cur_m = ds.node_machine[nd]; let cur_pt = ds.node_pt[nd];
            let mut best_m = cur_m; let mut best_pt = cur_pt;
            let mut best_mk = current_mk; let mut best_ins = 0usize;
            let consider_neutral = neutral_left > 0
                && (cur_m == plateau_bottleneck_m
                    || machine_cp_count[cur_m].saturating_add(1) >= machine_cp_count[plateau_bottleneck_m]);
            let mut best_neutral: Option<(usize, u32, usize, (u32, u32, u32, u32))> = None;

            for &(new_m, new_pt) in &op_info.machines {
                if new_m == cur_m { continue; }
                let old_pos = match ds.machine_seq[cur_m].iter().position(|&x| x == nd) { Some(p)=>p, None=>continue };
                ds.machine_seq[cur_m].remove(old_pos);
                ds.node_machine[nd] = new_m; ds.node_pt[nd] = new_pt;
                let positions = boundary_insertion_positions(&ds, &start_snapshot, &job_op_to_node, nd, new_m, 4);
                for &pos in &positions {
                    let lb = insertion_makespan_lb(pre, &ds, &start_snapshot, &job_op_to_node, nd, new_m, pos, new_pt);
                    if lb >= best_mk && !(consider_neutral && lb == current_mk) { continue; }
                    ds.machine_seq[new_m].insert(pos, nd);
                    if let Some((tmk, _)) = eval_disj(&ds, &mut buf) {
                        if tmk < best_mk {
                            best_mk = tmk; best_m = new_m; best_pt = new_pt; best_ins = pos;
                        } else if consider_neutral && tmk == current_mk {
                            let cand_profile = bottleneck_completion_profile(&ds, &buf.start, tail_slack);
                            if cand_profile < current_bottleneck_profile
                                && best_neutral.as_ref().map_or(true, |bn| cand_profile < bn.3)
                            {
                                best_neutral = Some((new_m, new_pt, pos, cand_profile));
                            }
                        }
                    }
                    ds.machine_seq[new_m].remove(pos);
                }
                ds.machine_seq[cur_m].insert(old_pos, nd);
                ds.node_machine[nd] = cur_m; ds.node_pt[nd] = cur_pt;
            }
            {
                let Some(old_pos) = ds.machine_seq[cur_m].iter().position(|&x| x == nd) else { continue };
                ds.machine_seq[cur_m].remove(old_pos);
                let positions = boundary_insertion_positions(&ds, &start_snapshot, &job_op_to_node, nd, cur_m, 4);
                for &pos in &positions {
                    let lb = insertion_makespan_lb(pre, &ds, &start_snapshot, &job_op_to_node, nd, cur_m, pos, cur_pt);
                    if lb >= best_mk && !(consider_neutral && lb == current_mk) { continue; }
                    ds.machine_seq[cur_m].insert(pos, nd);
                    if let Some((tmk, _)) = eval_disj(&ds, &mut buf) {
                        if tmk < best_mk {
                            best_mk = tmk; best_m = cur_m; best_pt = cur_pt; best_ins = pos;
                        } else if consider_neutral && tmk == current_mk {
                            let cand_profile = bottleneck_completion_profile(&ds, &buf.start, tail_slack);
                            if cand_profile < current_bottleneck_profile
                                && best_neutral.as_ref().map_or(true, |bn| cand_profile < bn.3)
                            {
                                best_neutral = Some((cur_m, cur_pt, pos, cand_profile));
                            }
                        }
                    }
                    ds.machine_seq[cur_m].remove(pos);
                }
                ds.machine_seq[cur_m].insert(old_pos, nd);
            }
            if best_m != cur_m {
                let Some(old_pos) = ds.machine_seq[cur_m].iter().position(|&x| x == nd) else { continue };
                ds.machine_seq[cur_m].remove(old_pos);
                let ins = best_ins.min(ds.machine_seq[best_m].len());
                ds.machine_seq[best_m].insert(ins, nd);
                ds.node_machine[nd] = best_m; ds.node_pt[nd] = best_pt;
                current_mk = best_mk;
                let _ = eval_disj(&ds, &mut buf);
                start_snapshot.clone_from(&buf.start);
                neutral_left = 0;
                iter_improved = true;
            } else if let Some((neutral_m, neutral_pt, neutral_ins, _)) = best_neutral {
                let Some(old_pos) = ds.machine_seq[cur_m].iter().position(|&x| x == nd) else { continue };
                ds.machine_seq[cur_m].remove(old_pos);
                let ins = neutral_ins.min(ds.machine_seq[neutral_m].len());
                ds.machine_seq[neutral_m].insert(ins, nd);
                ds.node_machine[nd] = neutral_m; ds.node_pt[nd] = neutral_pt;
                let _ = eval_disj(&ds, &mut buf);
                start_snapshot.clone_from(&buf.start);
                current_bottleneck_profile = bottleneck_completion_profile(&ds, &buf.start, tail_slack);
                neutral_left = neutral_left.saturating_sub(1);
                iter_improved = true;
            }
        }

        if cp_pair.len() >= 2 {
            let enhanced_pairs = n <= 180 || cp_pair.len() <= 40;
            let mut pairs: Vec<(usize, usize)> = Vec::with_capacity(if enhanced_pairs { 48 } else { 16 });
            let mut add_pair = |a: usize, b: usize| {
                if a == b || pairs.len() >= 48 {
                    return;
                }
                let pair = if a < b { (a, b) } else { (b, a) };
                if !pairs.contains(&pair) {
                    pairs.push(pair);
                }
            };

            if enhanced_pairs {
                let mut cp_member = vec![false; n];
                let mut cp_rank = vec![usize::MAX; n];
                for (rank, &nd) in cp_pair.iter().enumerate() {
                    cp_member[nd] = true;
                    cp_rank[nd] = rank;
                }

                for w in ds.machine_seq[plateau_bottleneck_m].windows(2).rev() {
                    if cp_member[w[0]] && cp_member[w[1]] {
                        add_pair(w[0], w[1]);
                    }
                }

                for &nd in cp_flex.iter().take(24) {
                    let job = ds.node_job[nd];
                    let op_idx = ds.node_op[nd];
                    if op_idx > 0 && op_idx - 1 < job_op_to_node[job].len() {
                        let pred = job_op_to_node[job][op_idx - 1];
                        if pred != usize::MAX && cp_member[pred] {
                            add_pair(nd, pred);
                        }
                    }
                    if op_idx + 1 < job_op_to_node[job].len() {
                        let succ = job_op_to_node[job][op_idx + 1];
                        if succ != usize::MAX && cp_member[succ] {
                            add_pair(nd, succ);
                        }
                    }
                }

                for &nd in cp_flex.iter().take(12) {
                    let rank = cp_rank[nd];
                    for &delta in &[1usize, 2usize] {
                        if rank >= delta {
                            add_pair(nd, cp_pair[rank - delta]);
                        }
                        if rank.saturating_add(delta) < cp_pair.len() {
                            add_pair(nd, cp_pair[rank + delta]);
                        }
                    }
                }

                for w in cp_pair.windows(2).rev() {
                    add_pair(w[0], w[1]);
                }
            } else {
                for w in cp_pair.windows(2).rev().take(16) {
                    add_pair(w[0], w[1]);
                }
            }

            let pair_start = buf.start.clone();

            let cand_positions = |ds: &DisjSchedule, m: usize, nd: usize| -> Vec<usize> {
                boundary_insertion_positions(ds, &pair_start, &job_op_to_node, nd, m, 3)
            };

            let mut global_best_mk = current_mk;
            let mut global_best_config: Option<(usize, usize, usize, u32, usize, usize, u32, usize, u8)> = None;

            for (nd1, nd2) in pairs {
                let job1 = ds.node_job[nd1];
                let op1 = ds.node_op[nd1];
                let job2 = ds.node_job[nd2];
                let op2 = ds.node_op[nd2];
                let prod1 = pre.job_products[job1];
                let prod2 = pre.job_products[job2];
                if op1 >= pre.product_ops[prod1].len() || op2 >= pre.product_ops[prod2].len() {
                    continue;
                }
                let op_info1 = &pre.product_ops[prod1][op1];
                let op_info2 = &pre.product_ops[prod2][op2];
                if op_info1.machines.len() <= 1 && op_info2.machines.len() <= 1 {
                    continue;
                }

                let cur_m1 = ds.node_machine[nd1];
                let cur_pt1 = ds.node_pt[nd1];
                let cur_m2 = ds.node_machine[nd2];
                let cur_pt2 = ds.node_pt[nd2];

                let opts1 = pair_option_prefilter(
                    pre,
                    &ds,
                    &pair_start,
                    &job_op_to_node,
                    nd1,
                    plateau_bottleneck_m,
                    3,
                );
                let opts2 = pair_option_prefilter(
                    pre,
                    &ds,
                    &pair_start,
                    &job_op_to_node,
                    nd2,
                    plateau_bottleneck_m,
                    3,
                );

                let pos1 = match ds.machine_seq[cur_m1].iter().position(|&x| x == nd1) {
                    Some(p) => p,
                    None => continue,
                };
                let pos2 = match ds.machine_seq[cur_m2].iter().position(|&x| x == nd2) {
                    Some(p) => p,
                    None => continue,
                };

                if cur_m1 == cur_m2 {
                    if pos1 > pos2 {
                        ds.machine_seq[cur_m1].remove(pos1);
                        ds.machine_seq[cur_m2].remove(pos2);
                    } else {
                        ds.machine_seq[cur_m2].remove(pos2);
                        ds.machine_seq[cur_m1].remove(pos1);
                    }
                } else {
                    ds.machine_seq[cur_m1].remove(pos1);
                    ds.machine_seq[cur_m2].remove(pos2);
                }

                let mut best_mk_pair = current_mk;
                let mut best_config: Option<(usize, u32, usize, usize, u32, usize, u8)> = None;

                for &(m1, pt1) in &opts1 {
                    for &(m2, pt2) in &opts2 {
                        if m1 != m2 {
                            ds.node_machine[nd1] = m1;
                            ds.node_pt[nd1] = pt1;
                            ds.node_machine[nd2] = m2;
                            ds.node_pt[nd2] = pt2;

                            let p1s = cand_positions(&ds, m1, nd1);
                            let p2s = cand_positions(&ds, m2, nd2);

                            for &p1i in &p1s {
                                let lb1 = insertion_makespan_lb(pre, &ds, &pair_start, &job_op_to_node, nd1, m1, p1i, pt1);
                                if lb1 >= best_mk_pair { continue; }
                                ds.machine_seq[m1].insert(p1i, nd1);
                                for &p2i in &p2s {
                                    let lb2 = insertion_makespan_lb(pre, &ds, &pair_start, &job_op_to_node, nd2, m2, p2i, pt2);
                                    if lb1.max(lb2) >= best_mk_pair {
                                        continue;
                                    }
                                    ds.machine_seq[m2].insert(p2i, nd2);
                                    if let Some((tmk, _)) = eval_disj(&ds, &mut buf) {
                                        if tmk < best_mk_pair {
                                            best_mk_pair = tmk;
                                            best_config = Some((m1, pt1, p1i, m2, pt2, p2i, 0));
                                        }
                                    }
                                    ds.machine_seq[m2].remove(p2i);
                                }
                                ds.machine_seq[m1].remove(p1i);
                            }
                        } else {
                            let m = m1;
                            ds.node_machine[nd1] = m;
                            ds.node_pt[nd1] = pt1;
                            ds.node_machine[nd2] = m;
                            ds.node_pt[nd2] = pt2;

                            for order_flag in 0u8..=1u8 {
                                let (a, b, pta, ptb) = if order_flag == 0 { (nd1, nd2, pt1, pt2) } else { (nd2, nd1, pt2, pt1) };

                                let pa = cand_positions(&ds, m, a);
                                for &pai in &pa {
                                    let lba = insertion_makespan_lb(pre, &ds, &pair_start, &job_op_to_node, a, m, pai, pta);
                                    if lba >= best_mk_pair { continue; }
                                    ds.machine_seq[m].insert(pai, a);
                                    let pb = cand_positions(&ds, m, b);
                                    for &pbi in &pb {
                                        let lbb = insertion_makespan_lb(pre, &ds, &pair_start, &job_op_to_node, b, m, pbi, ptb);
                                        if lba.max(lbb) >= best_mk_pair {
                                            continue;
                                        }
                                        ds.machine_seq[m].insert(pbi, b);
                                        if let Some((tmk, _)) = eval_disj(&ds, &mut buf) {
                                            if tmk < best_mk_pair {
                                                best_mk_pair = tmk;
                                                if order_flag == 0 {
                                                    best_config = Some((m, pt1, pai, m, pt2, pbi, 0));
                                                } else {
                                                    best_config = Some((m, pt1, pbi, m, pt2, pai, 1));
                                                }
                                            }
                                        }
                                        ds.machine_seq[m].remove(pbi);
                                    }
                                    ds.machine_seq[m].remove(pai);
                                }
                            }
                        }
                    }
                }

                if let Some((bm1, bpt1, bp1, bm2, bpt2, bp2, order_flag)) = best_config {
                    if best_mk_pair < global_best_mk {
                        global_best_mk = best_mk_pair;
                        global_best_config = Some((nd1, nd2, bm1, bpt1, bp1, bm2, bpt2, bp2, order_flag));
                    }
                }

                ds.node_machine[nd1] = cur_m1;
                ds.node_pt[nd1] = cur_pt1;
                ds.node_machine[nd2] = cur_m2;
                ds.node_pt[nd2] = cur_pt2;

                if cur_m1 != cur_m2 {
                    let ins1 = pos1.min(ds.machine_seq[cur_m1].len());
                    ds.machine_seq[cur_m1].insert(ins1, nd1);
                    let ins2 = pos2.min(ds.machine_seq[cur_m2].len());
                    ds.machine_seq[cur_m2].insert(ins2, nd2);
                } else {
                    let m = cur_m1;
                    if pos1 <= pos2 {
                        let ins1 = pos1.min(ds.machine_seq[m].len());
                        ds.machine_seq[m].insert(ins1, nd1);
                        let ins2 = pos2.min(ds.machine_seq[m].len());
                        ds.machine_seq[m].insert(ins2, nd2);
                    } else {
                        let ins2 = pos2.min(ds.machine_seq[m].len());
                        ds.machine_seq[m].insert(ins2, nd2);
                        let ins1 = pos1.min(ds.machine_seq[m].len());
                        ds.machine_seq[m].insert(ins1, nd1);
                    }
                }

                let _ = eval_disj(&ds, &mut buf);
            }

            if let Some((nd1, nd2, bm1, bpt1, bp1, bm2, bpt2, bp2, order_flag)) = global_best_config {
                let m_cur1 = ds.node_machine[nd1];
                let m_cur2 = ds.node_machine[nd2];
                let pos1_opt = ds.machine_seq[m_cur1].iter().position(|&x| x == nd1);
                let pos2_opt = ds.machine_seq[m_cur2].iter().position(|&x| x == nd2);
                if let (Some(pos1), Some(pos2)) = (pos1_opt, pos2_opt) {
                    ds.machine_seq[m_cur1].remove(pos1);
                    let pos2_adj = if m_cur1 == m_cur2 && pos2 > pos1 {
                        ds.machine_seq[m_cur1].iter().position(|&x| x == nd2)
                    } else {
                        ds.machine_seq[m_cur2].iter().position(|&x| x == nd2)
                    };
                    if let Some(pos2_adj) = pos2_adj {
                        ds.machine_seq[m_cur2].remove(pos2_adj);

                        ds.node_machine[nd1] = bm1;
                        ds.node_pt[nd1] = bpt1;
                        ds.node_machine[nd2] = bm2;
                        ds.node_pt[nd2] = bpt2;

                        if bm1 != bm2 {
                            let ins1 = bp1.min(ds.machine_seq[bm1].len());
                            ds.machine_seq[bm1].insert(ins1, nd1);
                            let ins2 = bp2.min(ds.machine_seq[bm2].len());
                            ds.machine_seq[bm2].insert(ins2, nd2);
                        } else {
                            let m = bm1;
                            if order_flag == 0 {
                                let ins1 = bp1.min(ds.machine_seq[m].len());
                                ds.machine_seq[m].insert(ins1, nd1);
                                let ins2 = bp2.min(ds.machine_seq[m].len());
                                ds.machine_seq[m].insert(ins2, nd2);
                            } else {
                                let ins2 = bp2.min(ds.machine_seq[m].len());
                                ds.machine_seq[m].insert(ins2, nd2);
                                let ins1 = bp1.min(ds.machine_seq[m].len());
                                ds.machine_seq[m].insert(ins1, nd1);
                            }
                        }

                        current_mk = global_best_mk;
                        let _ = eval_disj(&ds, &mut buf);
                        iter_improved = true;
                    } else {
                        ds.machine_seq[m_cur1].insert(pos1, nd1);
                    }
                }
            }
        }

        if !iter_improved { break; }
    }

    if current_mk >= initial_mk { return Ok(None); }
    let Some((final_mk, _)) = eval_disj(&ds, &mut buf) else { return Ok(None); };
    if final_mk >= initial_mk { return Ok(None); }
    let sol = disj_to_solution(pre, &ds, &buf.start)?;
    Ok(Some((sol, final_mk)))
}

#[inline]
fn cp_window_option_prefilter(
    pre: &Pre,
    ds: &DisjSchedule,
    starts: &[u32],
    job_op_to_node: &[Vec<usize>],
    node: usize,
    route_pref_counts: Option<&RoutePrefCounts>,
    max_keep: usize,
) -> Vec<(usize, u32)> {
    if max_keep == 0 { return Vec::new(); }
    let cur_m = ds.node_machine[node];
    let cur_pt = ds.node_pt[node];
    let mut keep: Vec<(usize, u32)> = Vec::with_capacity(max_keep.min(3));
    keep.push((cur_m, cur_pt));

    let job = ds.node_job[node];
    let op_idx = ds.node_op[node];
    let product = pre.job_products[job];
    if max_keep <= 1 || product >= pre.product_ops.len() || op_idx >= pre.product_ops[product].len() {
        return keep;
    }
    let op_info = &pre.product_ops[product][op_idx];
    if op_info.machines.len() <= 1 { return keep; }

    let mut pred_finish = 0u32;
    if op_idx > 0 && op_idx - 1 < job_op_to_node[job].len() {
        let pred = job_op_to_node[job][op_idx - 1];
        if pred != usize::MAX { pred_finish = starts[pred].saturating_add(ds.node_pt[pred]); }
    }
    let mut succ_start = u32::MAX;
    if op_idx + 1 < job_op_to_node[job].len() {
        let succ = job_op_to_node[job][op_idx + 1];
        if succ != usize::MAX { succ_start = starts[succ]; }
    }

    let mut scored: Vec<(f64, u32, u32, usize, u32, u32)> =
        Vec::with_capacity(op_info.machines.len().saturating_sub(1));
    for &(m, pt) in &op_info.machines {
        if m == cur_m { continue; }
        let positions = boundary_insertion_positions(ds, starts, job_op_to_node, node, m, 2);
        let seq = &ds.machine_seq[m];
        let latest_start = if succ_start != u32::MAX { succ_start.saturating_sub(pt) } else { u32::MAX };
        let scarcity_pen = 0.12 * pre.avg_op_min * (pre.machine_scarcity[m] / pre.avg_machine_scarcity.max(1e-9)).clamp(0.0, 3.0);
        let scarcity_burden = (pre.machine_scarcity[m] * 1_000.0 / pre.avg_machine_scarcity.max(1e-9))
            .clamp(0.0, u32::MAX as f64) as u32;
        let tail_burden = seq.last()
            .map(|&last| starts[last].saturating_add(ds.node_pt[last]))
            .unwrap_or(0);
        let route_bonus = route_pref_bonus_fh(route_pref_counts, product, op_idx, m) * (0.16 * pre.avg_op_min);
        let mut best_score = f64::INFINITY;
        let mut best_est_end = u32::MAX;

        if positions.is_empty() {
            let est_end = pred_finish.saturating_add(pt);
            let succ_pen = if latest_start != u32::MAX && pred_finish > latest_start { (pred_finish - latest_start) as f64 } else { 0.0 };
            best_score = est_end as f64 + 1.2 * succ_pen + scarcity_pen - route_bonus;
            best_est_end = est_end;
        } else {
            for &pos in &positions {
                let left_finish = if pos == 0 { 0 } else {
                    let prev = seq[pos - 1];
                    starts[prev].saturating_add(ds.node_pt[prev])
                };
                let est_start = pred_finish.max(left_finish);
                let est_end = est_start.saturating_add(pt);
                let right_start = if pos < seq.len() { starts[seq[pos]] } else { u32::MAX };
                let overlap_pen = if right_start != u32::MAX && est_end > right_start { (est_end - right_start) as f64 } else { 0.0 };
                let succ_pen = if latest_start != u32::MAX && est_start > latest_start { (est_start - latest_start) as f64 } else { 0.0 };
                let score = est_end as f64 + 1.7 * overlap_pen + 1.2 * succ_pen + scarcity_pen - route_bonus;
                if score < best_score || score == best_score && est_end < best_est_end {
                    best_score = score;
                    best_est_end = est_end;
                }
            }
        }

        scored.push((best_score, best_est_end, pt, m, scarcity_burden, tail_burden));
    }

    let mut nondominated: Vec<(f64, u32, u32, usize, u32, u32)> =
        Vec::with_capacity(scored.len());
    for i in 0..scored.len() {
        let candidate = scored[i];
        let dominated = scored.iter().enumerate().any(|(j, other)| {
            j != i
                && other.1 <= candidate.1
                && other.2 <= candidate.2
                && other.4 <= candidate.4
                && other.5 <= candidate.5
                && (other.1 < candidate.1
                    || other.2 < candidate.2
                    || other.4 < candidate.4
                    || other.5 < candidate.5)
        });
        if !dominated {
            nondominated.push(candidate);
        }
    }
    nondominated.sort_by(|a, b| {
        a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.cmp(&b.1))
            .then_with(|| a.3.cmp(&b.3))
            .then_with(|| a.2.cmp(&b.2))
    });
    for (_, _, pt, m, _, _) in nondominated.into_iter().take(max_keep.saturating_sub(1)) {
        keep.push((m, pt));
    }
    keep
}

fn cp_window_exhaustive(
    pre: &Pre,
    challenge: &Challenge,
    base_sol: &Solution,
    max_iters: usize,
) -> Result<Option<(Solution, u32)>> {
    let mut ds = build_disj_from_solution(pre, challenge, base_sol)?;
    let mut buf = EvalBuf::new(ds.n);
    let Some((mut current_mk, _)) = eval_disj(&ds, &mut buf) else { return Ok(None); };
    let initial_mk = current_mk;
    let n = ds.n;

    let mut job_op_to_node: Vec<Vec<usize>> = vec![vec![]; challenge.num_jobs];
    for nd in 0..n {
        let job = ds.node_job[nd];
        let op_idx = ds.node_op[nd];
        if op_idx >= job_op_to_node[job].len() {
            job_op_to_node[job].resize(op_idx + 1, usize::MAX);
        }
        job_op_to_node[job][op_idx] = nd;
    }
    let base_route_pref = Some(build_single_solution_pref_counts(pre, challenge, base_sol));

    #[derive(Clone, Copy)]
    struct MoveRec {
        node: usize,
        new_m: usize,
        new_pt: u32,
        new_ins: usize,
        old_m: usize,
        old_pt: u32,
        old_pos: usize,
    }

    struct BeamState {
        mk: u32,
        tail: (u32, u32),
        moves: Vec<MoveRec>,
    }

    #[inline]
    fn beam_tail_potential(
        ds: &DisjSchedule,
        starts: &[u32],
        machine: usize,
        mk: u32,
        tail_span: u32,
    ) -> (u32, u32) {
        let Some(seq) = ds.machine_seq.get(machine) else { return (0, 0) };
        let edge = mk.saturating_sub(tail_span);
        let mut count = 0u32;
        let mut work = 0u32;
        for &node in seq.iter().rev().take(24) {
            let Some(&start) = starts.get(node) else { continue };
            let Some(&pt) = ds.node_pt.get(node) else { continue };
            if start.saturating_add(pt) >= edge {
                count = count.saturating_add(1);
                work = work.saturating_add(pt);
            }
        }
        (count, work)
    }

    #[inline]
    fn apply_rec(ds: &mut DisjSchedule, rec: &MoveRec) {
        let seq_old = &mut ds.machine_seq[rec.old_m];
        debug_assert!(rec.old_pos < seq_old.len() && seq_old[rec.old_pos] == rec.node);
        seq_old.remove(rec.old_pos);
        let ins = rec.new_ins.min(ds.machine_seq[rec.new_m].len());
        ds.machine_seq[rec.new_m].insert(ins, rec.node);
        ds.node_machine[rec.node] = rec.new_m;
        ds.node_pt[rec.node] = rec.new_pt;
    }

    #[inline]
    fn undo_rec(ds: &mut DisjSchedule, rec: &MoveRec) {
        let Some(&cur_m) = ds.node_machine.get(rec.node) else { return };
        let Some(seq) = ds.machine_seq.get(cur_m) else { return };
        let Some(pos) = seq.iter().position(|&x| x == rec.node) else { return };
        ds.machine_seq[cur_m].remove(pos);
        let ins = rec.old_pos.min(ds.machine_seq[rec.old_m].len());
        ds.machine_seq[rec.old_m].insert(ins, rec.node);
        ds.node_machine[rec.node] = rec.old_m;
        ds.node_pt[rec.node] = rec.old_pt;
    }

    #[inline]
    fn make_rec(ds: &DisjSchedule, node: usize, new_m: usize, new_pt: u32, pos: usize) -> Option<MoveRec> {
        let old_m = ds.node_machine[node];
        let old_pt = ds.node_pt[node];
        let old_pos = ds.machine_seq[old_m].iter().position(|&x| x == node)?;
        let tlen = if new_m == old_m {
            ds.machine_seq[new_m].len().saturating_sub(1)
        } else {
            ds.machine_seq[new_m].len()
        };
        let new_ins = pos.min(tlen);
        Some(MoveRec { node, new_m, new_pt, new_ins, old_m, old_pt, old_pos })
    }

    #[inline]
    fn move_path_lcp(a: &[MoveRec], b: &[MoveRec]) -> usize {
        let n = a.len().min(b.len());
        let mut i = 0usize;
        while i < n {
            let x = &a[i];
            let y = &b[i];
            if x.node != y.node
                || x.new_m != y.new_m
                || x.new_pt != y.new_pt
                || x.new_ins != y.new_ins
                || x.old_m != y.old_m
                || x.old_pt != y.old_pt
                || x.old_pos != y.old_pos
            {
                break;
            }
            i += 1;
        }
        i
    }

    #[inline]
    fn transition_path(
        ds: &mut DisjSchedule,
        applied: &mut Vec<MoveRec>,
        target: &[MoveRec],
    ) {
        let lcp = move_path_lcp(applied, target);
        while applied.len() > lcp {
            if let Some(rec) = applied.pop() {
                undo_rec(ds, &rec);
            }
        }
        for rec in target.iter().skip(lcp) {
            apply_rec(ds, rec);
            applied.push(*rec);
        }
    }

    #[inline]
    fn path_sort_key(moves: &[MoveRec]) -> Vec<(usize, usize, u32, usize)> {
        moves
            .iter()
            .map(|r| (r.node, r.new_m, r.new_pt, r.new_ins))
            .collect()
    }

    let mut work_buf = EvalBuf::new(n);
    let mut root_starts: Vec<u32> = Vec::new();
    let mut applied_path: Vec<MoveRec> = Vec::with_capacity(8);

    for _iter in 0..max_iters {
        let mut finish = vec![0u32; n];
        for nd in 0..n { finish[nd] = buf.start[nd].saturating_add(ds.node_pt[nd]); }
        let makespan = current_mk;

        let mut on_cp = vec![false; n];
        for nd in 0..n { if finish[nd] == makespan { on_cp[nd] = true; } }

        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| buf.start[b].cmp(&buf.start[a]));
        for &nd in &order {
            if !on_cp[nd] { continue; }
            let job = ds.node_job[nd];
            let op_idx = ds.node_op[nd];
            if op_idx > 0 {
                let pred_op = op_idx - 1;
                if pred_op < job_op_to_node[job].len() {
                    let pred = job_op_to_node[job][pred_op];
                    if pred != usize::MAX && finish[pred] == buf.start[nd] { on_cp[pred] = true; }
                }
            }
            let machine = ds.node_machine[nd];
            let seq = &ds.machine_seq[machine];
            if let Some(pos) = seq.iter().position(|&x| x == nd) {
                if pos > 0 {
                    let pred_nd = seq[pos - 1];
                    if finish[pred_nd] == buf.start[nd] { on_cp[pred_nd] = true; }
                }
            }
        }

        let mut cp_flex: Vec<usize> = (0..n).filter(|&nd| {
            if !on_cp[nd] { return false; }
            let job = ds.node_job[nd]; let op_idx = ds.node_op[nd];
            let product = pre.job_products[job];
            op_idx < pre.product_ops[product].len() && pre.product_ops[product][op_idx].machines.len() > 1
        }).collect();
        cp_flex.sort_by_key(|&nd| buf.start[nd]);

        if cp_flex.is_empty() { break; }

        let mut machine_cp_count = vec![0usize; challenge.num_machines];
        for &nd in &cp_flex { machine_cp_count[ds.node_machine[nd]] += 1; }
        let bottleneck_m = machine_cp_count.iter().enumerate()
            .max_by_key(|&(_, &c)| c).map(|(m, _)| m).unwrap_or(0);
        let delay_score = removable_delay_scores(&ds, &buf.start, &job_op_to_node, current_mk);

        let use_tail_potential = n <= 160 && cp_flex.len() <= 80;
        let tail_span = ((pre.avg_op_min * (1.35 + 0.75 * pre.high_flex + 0.45 * pre.jobshopness)).max(1.0)) as u32;
        let cp_len = cp_flex.len();
        let window_sizes: &[usize] = if cp_len <= 20 {
            &[5, 4, 3]
        } else if cp_len <= 60 {
            &[4, 3, 2]
        } else {
            &[2, 1]
        };

        let mut best_window_improvement = false;
        root_starts.clone_from(&buf.start);

        'window_search: for &wsize in window_sizes {
            if cp_flex.len() < wsize { continue; }
            let bottleneck_ops: Vec<usize> = cp_flex.iter().copied()
                .filter(|&nd| ds.node_machine[nd] == bottleneck_m)
                .collect();
            if bottleneck_ops.is_empty() { continue; }

            let num_windows = cp_flex.len().saturating_sub(wsize) + 1;
            let mut window_order: Vec<(u64, usize)> = Vec::with_capacity(num_windows);
            for w_start in 0..num_windows {
                let mut score = 0u64;
                let mut has_bottleneck = false;
                for &node in &cp_flex[w_start..w_start + wsize] {
                    let machine = ds.node_machine[node];
                    let job = ds.node_job[node];
                    let op_idx = ds.node_op[node];
                    let product = pre.job_products[job];
                    let flexibility = pre.product_ops[product][op_idx].machines.len().max(1);
                    let bottleneck_bonus = if machine == bottleneck_m {
                        32u64
                    } else {
                        0u64
                    };
                    if machine == bottleneck_m {
                        has_bottleneck = true;
                    }
                    let machine_pressure = (machine_cp_count[machine] as u64).saturating_mul(8);
                    let tail_proximity = finish[node] as u64;
                    let rigidity = (16usize / flexibility) as u64;
                    score = score
                        .saturating_add(if n > 80 { delay_score[node] } else { 0 })
                        .saturating_add(bottleneck_bonus)
                        .saturating_add(machine_pressure)
                        .saturating_add(ds.node_pt[node] as u64)
                        .saturating_add(tail_proximity)
                        .saturating_add(rigidity);
                }
                if has_bottleneck {
                    window_order.push((score, w_start));
                }
            }
            window_order.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
            for (_, w_start) in window_order {
                let window: Vec<usize> = cp_flex[w_start..w_start+wsize].to_vec();

                transition_path(&mut ds, &mut applied_path, &[]);

                let mut beam: Vec<BeamState> = Vec::with_capacity(4);
                beam.push(BeamState {
                    mk: current_mk,
                    tail: if use_tail_potential {
                        beam_tail_potential(&ds, &root_starts, bottleneck_m, current_mk, tail_span)
                    } else {
                        (0, 0)
                    },
                    moves: vec![],
                });
                let beam_width = 4usize;
                let max_evals = 80usize;
                let mut evals = 0usize;

                for &node in &window {
                    beam.sort_by(|a, b| {
                        a.mk.cmp(&b.mk)
                            .then_with(|| a.tail.cmp(&b.tail))
                            .then_with(|| path_sort_key(&a.moves).cmp(&path_sort_key(&b.moves)))
                    });

                    let mut all_candidates: Vec<BeamState> = Vec::with_capacity(beam.len() * 4);
                    for state in &beam {
                        all_candidates.push(BeamState {
                            mk: state.mk,
                            tail: state.tail,
                            moves: state.moves.clone(),
                        });

                        if evals >= max_evals { break; }

                        transition_path(&mut ds, &mut applied_path, &state.moves);

                        let job = ds.node_job[node];
                        let op_idx = ds.node_op[node];
                        let product = pre.job_products[job];
                        if product < pre.product_ops.len() && op_idx < pre.product_ops[product].len() {
                            let cur_m = ds.node_machine[node];
                            let cur_pt = ds.node_pt[node];

                            let alternatives = cp_window_option_prefilter(
                                pre, &ds, &root_starts, &job_op_to_node, node,
                                base_route_pref.as_ref(), 2,
                            );
                            for &(new_m, new_pt) in &alternatives {
                                if new_m == cur_m { continue; }
                                if evals >= max_evals { break; }
                                let positions = boundary_insertion_positions(
                                    &ds, &root_starts, &job_op_to_node, node, new_m, 2,
                                );
                                for &pos in &positions {
                                    if evals >= max_evals { break; }
                                    let Some(rec) = make_rec(&ds, node, new_m, new_pt, pos) else { continue };
                                    apply_rec(&mut ds, &rec);
                                    if let Some((tmk, _)) = eval_disj(&ds, &mut work_buf) {
                                        evals += 1;
                                        let tail = if use_tail_potential {
                                            beam_tail_potential(&ds, &work_buf.start, bottleneck_m, tmk, tail_span)
                                        } else {
                                            (0, 0)
                                        };
                                        if tmk < state.mk || (use_tail_potential && tmk == state.mk && tail < state.tail) {
                                            let mut new_moves = state.moves.clone();
                                            new_moves.push(rec);
                                            all_candidates.push(BeamState {
                                                mk: tmk,
                                                tail,
                                                moves: new_moves,
                                            });
                                        }
                                    }
                                    undo_rec(&mut ds, &rec);
                                }
                            }

                            if evals < max_evals {
                                let positions = boundary_insertion_positions(
                                    &ds, &root_starts, &job_op_to_node, node, cur_m, 2,
                                );
                                for &pos in &positions {
                                    if evals >= max_evals { break; }
                                    let Some(rec) = make_rec(&ds, node, cur_m, cur_pt, pos) else { continue };
                                    if rec.new_ins == rec.old_pos && rec.new_m == rec.old_m {
                                        continue;
                                    }
                                    apply_rec(&mut ds, &rec);
                                    if let Some((tmk, _)) = eval_disj(&ds, &mut work_buf) {
                                        evals += 1;
                                        let tail = if use_tail_potential {
                                            beam_tail_potential(&ds, &work_buf.start, bottleneck_m, tmk, tail_span)
                                        } else {
                                            (0, 0)
                                        };
                                        if tmk < state.mk || (use_tail_potential && tmk == state.mk && tail < state.tail) {
                                            let mut new_moves = state.moves.clone();
                                            new_moves.push(rec);
                                            all_candidates.push(BeamState {
                                                mk: tmk,
                                                tail,
                                                moves: new_moves,
                                            });
                                        }
                                    }
                                    undo_rec(&mut ds, &rec);
                                }
                            }
                        }
                    }

                    all_candidates.sort_by(|a, b| {
                        a.mk.cmp(&b.mk)
                            .then_with(|| a.tail.cmp(&b.tail))
                            .then_with(|| path_sort_key(&a.moves).cmp(&path_sort_key(&b.moves)))
                    });
                    all_candidates.truncate(beam_width);
                    beam = all_candidates;
                    if evals >= max_evals { break; }
                }

                if let Some(best_state) = beam.first() {
                    if best_state.mk < current_mk {
                        transition_path(&mut ds, &mut applied_path, &best_state.moves);
                        applied_path.clear();
                        if let Some((tmk, _)) = eval_disj(&ds, &mut buf) {
                            current_mk = tmk;
                        } else {
                            current_mk = best_state.mk;
                        }
                        best_window_improvement = true;
                        break 'window_search;
                    }
                }

                transition_path(&mut ds, &mut applied_path, &[]);
            }
            if best_window_improvement { break; }
        }

        if !best_window_improvement { break; }
    }

    transition_path(&mut ds, &mut applied_path, &[]);

    if current_mk >= initial_mk { return Ok(None); }
    let Some((final_mk, _)) = eval_disj(&ds, &mut buf) else { return Ok(None); };
    if final_mk >= initial_mk { return Ok(None); }
    let sol = disj_to_solution(pre, &ds, &buf.start)?;
    Ok(Some((sol, final_mk)))
}

#[inline]
fn solution_route_sequence_signature(
    pre: &Pre,
    challenge: &Challenge,
    sol: &Solution,
) -> Vec<(u16, u8)> {
    let num_machines = challenge.num_machines;
    let mut routes = Vec::with_capacity(pre.total_ops);
    let mut ranks = vec![u8::MAX; pre.total_ops];
    let mut machine_ops: Vec<Vec<(u32, usize)>> = (0..num_machines)
        .map(|_| Vec::new())
        .collect();
    let mut node = 0usize;

    for job in 0..challenge.num_jobs {
        let lim = pre.job_ops_len[job].min(sol.job_schedule[job].len());
        for op_idx in 0..pre.job_ops_len[job] {
            if op_idx < lim {
                let (machine, start) = sol.job_schedule[job][op_idx];
                if machine < num_machines {
                    routes.push(machine as u16);
                    machine_ops[machine].push((start, node));
                } else {
                    routes.push(u16::MAX);
                }
            } else {
                routes.push(u16::MAX);
            }
            node += 1;
        }
    }

    for ops in &mut machine_ops {
        ops.sort_unstable_by_key(|&(start, node)| (start, node));
        let len = ops.len();
        if len == 0 {
            continue;
        }
        for (rank, &(_, node)) in ops.iter().enumerate() {
            ranks[node] = ((rank.saturating_mul(8) / len).min(7)) as u8;
        }
    }

    routes.into_iter().zip(ranks).collect()
}

#[inline]
fn composite_distance_sig(a: &[(u16, u8)], b: &[(u16, u8)]) -> u32 {
    let mut distance = 0u32;
    let n = a.len().min(b.len());
    for i in 0..n {
        if a[i].0 != b[i].0 {
            distance = distance.saturating_add(4);
        } else if a[i].1 != b[i].1 {
            distance = distance.saturating_add(1);
        }
    }
    distance
}

fn push_top_solutions_diverse(
    pre: &Pre,
    challenge: &Challenge,
    pool: &mut Vec<(Solution, u32, Vec<(u16, u8)>)>,
    sol: &Solution,
    mk: u32,
    cap: usize,
) {
    let sig = solution_route_sequence_signature(pre, challenge, sol);
    pool.push((sol.clone(), mk, sig));

    while pool.len() > cap {
        let len = pool.len();
        let mut min_nn = vec![u32::MAX; len];
        for i in 0..len {
            for j in (i + 1)..len {
                let d = composite_distance_sig(&pool[i].2, &pool[j].2);
                if d < min_nn[i] {
                    min_nn[i] = d;
                }
                if d < min_nn[j] {
                    min_nn[j] = d;
                }
            }
        }

        let mut worst_mk = pool[0].1;
        for i in 1..len {
            if pool[i].1 > worst_mk {
                worst_mk = pool[i].1;
            }
        }

        let mut drop_idx: Option<usize> = None;
        let mut drop_min_nn = u32::MAX;
        for i in 0..len {
            if pool[i].1 != worst_mk {
                continue;
            }
            let mnn = min_nn[i];
            if drop_idx.is_none() || mnn < drop_min_nn {
                drop_idx = Some(i);
                drop_min_nn = mnn;
            }
        }
        let di = drop_idx.unwrap_or(0);
        pool.swap_remove(di);
    }
}

fn consensus_learning_from_elites(
    pre: &Pre,
    challenge: &Challenge,
    elites: &[(Solution, u32, Vec<(u16, u8)>)],
) -> Result<(Vec<f64>, Vec<f64>, RoutePrefCounts)> {
    if elites.is_empty() {
        return Err(anyhow!("No elites for consensus learning"));
    }
    let num_jobs = challenge.num_jobs;
    let num_machines = challenge.num_machines;

    let best_elite_mk = elites.iter().map(|(_, mk, _)| *mk).min().unwrap_or(0);
    let worst_elite_mk = elites.iter().map(|(_, mk, _)| *mk).max().unwrap_or(best_elite_mk);
    let weight_scale = pre.avg_op_min.max(1.0) as u32;
    let use_quality_weights = elites.len() >= 3
        && worst_elite_mk.saturating_sub(best_elite_mk) >= weight_scale;
    let elite_weight = |mk: u32| -> u32 {
        if !use_quality_weights {
            return 1;
        }
        1 + (worst_elite_mk.saturating_sub(mk) / weight_scale).min(3)
    };
    let denom = elites.iter()
        .map(|(_, mk, _)| elite_weight(*mk) as f64)
        .sum::<f64>()
        .max(1.0);

    let mut jb_sum = vec![0.0f64; num_jobs];
    for (sol, mk, _) in elites.iter() {
        let jb = job_bias_from_solution(pre, sol)?;
        let weight = elite_weight(*mk) as f64;
        for j in 0..num_jobs {
            jb_sum[j] += weight * jb[j];
        }
    }
    for j in 0..num_jobs {
        jb_sum[j] /= denom;
    }

    let mut bottleneck_cnt = vec![0u32; num_machines];
    let mut machine_end = vec![0u32; num_machines];
    for (sol, mk, _) in elites.iter() {
        let mk = *mk;
        machine_end.fill(0);
        for job in 0..num_jobs {
            let product = pre.job_products[job];
            if product >= pre.product_ops.len() {
                continue;
            }
            let sched = &sol.job_schedule[job];
            let ops = &pre.product_ops[product];
            let lim = sched.len().min(ops.len());
            for op_idx in 0..lim {
                let (m, st) = sched[op_idx];
                if m >= num_machines {
                    continue;
                }
                let op_info = &ops[op_idx];
                let mut pt = None;
                for &(mm, p) in &op_info.machines {
                    if mm == m {
                        pt = Some(p);
                        break;
                    }
                }
                let pt = pt.unwrap_or(op_info.min_pt);
                if pt >= INF {
                    continue;
                }
                let end = st.saturating_add(pt);
                if end > machine_end[m] {
                    machine_end[m] = end;
                }
            }
        }
        let weight = elite_weight(mk);
        for m in 0..num_machines {
            if machine_end[m] == mk {
                bottleneck_cnt[m] = bottleneck_cnt[m].saturating_add(weight);
            }
        }
    }
    let mut machine_penalty = vec![0.0f64; num_machines];
    for m in 0..num_machines {
        machine_penalty[m] = (bottleneck_cnt[m] as f64) / denom;
    }

    let num_products = pre.product_ops.len();
    let mut counts: Vec<Vec<Vec<u32>>> = pre
        .product_ops
        .iter()
        .map(|ops| ops.iter().map(|_| vec![0u32; num_machines]).collect())
        .collect();

    for (sol, mk, _) in elites.iter() {
        let weight = elite_weight(*mk);
        for job in 0..num_jobs {
            let product = pre.job_products[job];
            if product >= num_products {
                continue;
            }
            let sched = &sol.job_schedule[job];
            let ops_len = counts[product].len();
            for (op_idx, (m, _)) in sched.iter().enumerate() {
                if op_idx >= ops_len {
                    break;
                }
                if *m < num_machines {
                    counts[product][op_idx][*m] = counts[product][op_idx][*m].saturating_add(weight);
                }
            }
        }
    }

    let route_counts = build_route_pref_counts_from_full(&counts);
    Ok((jb_sum, machine_penalty, route_counts))
}

fn incumbent_directed_restart_guides(
    pre: &Pre,
    challenge: &Challenge,
    sol: &Solution,
    base_job_bias: Option<&[f64]>,
    base_machine_penalty: Option<&[f64]>,
    pattern: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut job_bias = base_job_bias
        .map(|v| v.to_vec())
        .unwrap_or_else(|| vec![0.0; challenge.num_jobs]);
    job_bias.resize(challenge.num_jobs, 0.0);
    let mut machine_penalty = base_machine_penalty
        .map(|v| v.to_vec())
        .unwrap_or_else(|| vec![0.0; challenge.num_machines]);
    machine_penalty.resize(challenge.num_machines, 0.0);

    let mut machine_end = vec![0u32; challenge.num_machines];
    let mut job_end = vec![0u32; challenge.num_jobs];
    for job in 0..challenge.num_jobs {
        let Some(&product) = pre.job_products.get(job) else { continue };
        let Some(ops) = pre.product_ops.get(product) else { continue };
        let Some(sched) = sol.job_schedule.get(job) else { continue };
        for (op_idx, &(machine, start)) in sched.iter().enumerate() {
            let Some(op) = ops.get(op_idx) else { break };
            if machine >= challenge.num_machines {
                continue;
            }
            let Some(&(_, pt)) = op.machines.iter().find(|&&(m, _)| m == machine) else {
                continue;
            };
            let end = start.saturating_add(pt);
            if end > machine_end[machine] {
                machine_end[machine] = end;
            }
            if end > job_end[job] {
                job_end[job] = end;
            }
        }
    }

    let makespan = machine_end.iter().copied().max().unwrap_or(0);
    let mut machines: Vec<usize> = (0..challenge.num_machines).collect();
    machines.sort_by(|&a, &b| machine_end[b].cmp(&machine_end[a]).then_with(|| a.cmp(&b)));
    let bottleneck_span = (pre.avg_op_min
        * (2.0 + 0.70 * pre.high_flex + 0.40 * pre.jobshopness))
        .max(1.0) as u32;

    let penalized = match pattern {
        0 => 1usize,
        1 => 2usize,
        _ => 0usize,
    };
    for &machine in machines.iter().take(penalized) {
        if machine_end[machine] == 0 {
            continue;
        }
        let closeness = machine_end[machine] as f64 / makespan.max(1) as f64;
        let penalty = (0.24 + 0.28 * closeness).min(0.58);
        machine_penalty[machine] = (machine_penalty[machine] + penalty).min(1.0);
    }

    let tail_edge = makespan.saturating_sub(bottleneck_span);
    let late_weight = if pattern == 2 { 0.52 } else { 0.22 };
    for job in 0..challenge.num_jobs {
        if job_end[job] < tail_edge {
            continue;
        }
        let lateness = job_end[job]
            .saturating_sub(tail_edge) as f64
            / bottleneck_span.max(1) as f64;
        job_bias[job] += late_weight * (0.35 + lateness.min(1.0));
    }

    (job_bias, machine_penalty)
}

fn target_failure_cause(
    pre: &Pre,
    challenge: &Challenge,
    sol: &Solution,
    makespan: u32,
) -> usize {
    let mut machine_end = vec![0u32; challenge.num_machines];
    let mut tail_jobs = 0usize;
    let tail_span = (pre.avg_op_min * (1.0 + 0.45 * pre.jobshopness)).max(1.0) as u32;
    let tail_edge = makespan.saturating_sub(tail_span);

    for job in 0..challenge.num_jobs {
        let Some(&product) = pre.job_products.get(job) else { continue };
        let Some(ops) = pre.product_ops.get(product) else { continue };
        let Some(sched) = sol.job_schedule.get(job) else { continue };
        let mut job_end = 0u32;
        for (op_idx, &(machine, start)) in sched.iter().enumerate() {
            let Some(op) = ops.get(op_idx) else { break };
            if machine >= challenge.num_machines {
                continue;
            }
            let Some((_, pt)) = op.machines.iter().find(|&&(m, _)| m == machine) else { continue };
            let end = start.saturating_add(*pt);
            if end > machine_end[machine] {
                machine_end[machine] = end;
            }
            if end > job_end {
                job_end = end;
            }
        }
        if job_end >= tail_edge {
            tail_jobs = tail_jobs.saturating_add(1);
        }
    }

    let tail_machines = machine_end.iter().filter(|&&end| end >= tail_edge).count();
    if pre.hi_flex && tail_machines >= 2 && tail_machines >= tail_jobs {
        0
    } else if pre.flex_avg <= 2.5 {
        1
    } else {
        2
    }
}

pub fn solve(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    pre: &Pre,
    effort: &EffortConfig,
) -> Result<()> {
    let mut rng = SmallRng::from_seed(challenge.seed);
    let rules: Vec<Rule> = vec![Rule::Adaptive,Rule::BnHeavy,Rule::EndTight,Rule::CriticalPath,Rule::MostWork,Rule::LeastFlex,Rule::Regret,Rule::ShortestProc,Rule::FlexBalance];
    let mut best_makespan = u32::MAX;
    let mut best_solution: Option<Solution> = None;
    let mut rule_sum_tmp = vec![0.0f64; 10];
    let mut rule_tries_tmp = vec![0u32; 10];
    let mut rule_sum_ctx_tmp = vec![vec![0.0f64; 10]; 4];
    let mut rule_tries_ctx_tmp = vec![vec![0u32; 10]; 4];
    let dummy_ctx = bandit_context_idx(false, false);
    for _ in 0..3 {
        let rule = choose_rule_bandit(
            &mut rng, &rules, &rule_sum_ctx_tmp, &rule_tries_ctx_tmp,
            &rule_sum_tmp, &rule_tries_tmp, dummy_ctx, best_makespan, 0, 0, false, false,
        );
        let (sol, mk) = construct_solution_conflict(
            challenge, pre, rule, 0, None, &mut rng, None, None, None, 0.0, false,
        )?;
        if mk < best_makespan {
            best_makespan = mk;
            best_solution = Some(sol.clone());
            save_solution(&sol)?;
        }
        let ridx = rule_idx(rule);
        rule_tries_tmp[ridx] = rule_tries_tmp[ridx].saturating_add(1);
        rule_sum_tmp[ridx] += mk as f64;
        rule_tries_ctx_tmp[dummy_ctx][ridx] = rule_tries_ctx_tmp[dummy_ctx][ridx].saturating_add(1);
        rule_sum_ctx_tmp[dummy_ctx][ridx] += mk as f64;
    }
    if best_solution.is_none() {
        let (sol, mk) = construct_solution_conflict(
            challenge, pre, Rule::Adaptive, 0, None, &mut rng, None, None, None, 0.0, false,
        )?;
        best_makespan = mk;
        save_solution(&sol)?;
        best_solution = Some(sol);
    }
    let mut top_solutions: Vec<(Solution, u32, Vec<(u16, u8)>)> = Vec::new();
    let target_margin: u32 = ((pre.avg_op_min*(0.9+0.9*pre.high_flex+0.6*pre.jobshopness)).max(1.0)) as u32;
    let route_w_base: f64 = (0.040+0.10*pre.high_flex+0.08*pre.jobshopness).clamp(0.04,0.22);

    if pre.flow_route.is_some()&&pre.flow_pt_by_job.is_some() {
        if let Ok((sol,mk))=neh_reentrant_flow_solution(pre,challenge.num_jobs,challenge.num_machines) {
            if mk<best_makespan{best_makespan=mk;best_solution=Some(sol.clone());save_solution(&sol)?;}
            push_top_solutions_diverse(pre, challenge, &mut top_solutions, &sol, mk, 15);
        }
    }
    let mut ranked: Vec<(Rule,u32,Solution)>=Vec::with_capacity(rules.len());
    for &rule in &rules {
        let (sol,mk)=construct_solution_conflict(challenge,pre,rule,0,None,&mut rng,None,None,None,0.0,false)?;
        if mk<best_makespan{best_makespan=mk;best_solution=Some(sol.clone());save_solution(&sol)?;}
        push_top_solutions_diverse(pre, challenge, &mut top_solutions, &sol, mk, 15);
        ranked.push((rule,mk,sol));
    }
    ranked.sort_by_key(|x|x.1);
    let r0=ranked[0].0; let r1=ranked.get(1).map(|x|x.0).unwrap_or(r0); let r2=ranked.get(2).map(|x|x.0).unwrap_or(r1);
    let mut rule_tries: Vec<u32>=vec![0u32;10];
    let mut rule_sum: Vec<f64>=vec![0.0;10];
    let mut rule_tries_ctx: Vec<Vec<u32>>=vec![vec![0u32;10];4];
    let mut rule_sum_ctx: Vec<Vec<f64>>=vec![vec![0.0;10];4];
    let base_ctx=bandit_context_idx(false,false);
    for (rr,mk,_) in &ranked{
        let idx=rule_idx(*rr);
        rule_tries[idx]=rule_tries[idx].saturating_add(1);
        rule_sum[idx]+=*mk as f64;
        rule_tries_ctx[base_ctx][idx]=rule_tries_ctx[base_ctx][idx].saturating_add(1);
        rule_sum_ctx[base_ctx][idx]+=*mk as f64;
    }
    let (jb0, mp0, route_counts0) = consensus_learning_from_elites(pre, challenge, &top_solutions)?;
    let mut learned_jb=Some(jb0);
    let mut learned_mp=Some(mp0);
    let mut learned_route_counts=Some(route_counts0);
    let mut learn_updates_left=10usize;
    let num_restarts=effort.fjsp_high_iters;
    let k_hi=if pre.flex_avg>8.0{8}else if pre.flex_avg>6.5{7}else{6};
    let mut stuck: usize=0;
    let mut stuck_ema: f64 = 0.0;
    let mut target_miss_feedback = [0u8; 3];
    let mut directed_restarts_left = 3usize;
    for r in 0..num_restarts {
        let late=r>=(num_restarts*2)/3;
        let k_min = 2usize;
        let k_max = 4usize.min(k_hi);
        let incumbent_before = best_makespan;
        let mut nearest_route_distance = u32::MAX;
        for i in 0..top_solutions.len() {
            for j in (i + 1)..top_solutions.len() {
                let d = composite_distance_sig(&top_solutions[i].2, &top_solutions[j].2);
                if d < nearest_route_distance {
                    nearest_route_distance = d;
                }
            }
        }
        let archive_clustered = top_solutions.len() >= 3
            && nearest_route_distance <= ((pre.total_ops as u32) / 40).max(1);
        let exploration_pressure = if archive_clustered { 0.12 } else { 0.0 }
            + 0.16 * ((stuck as f64) / 100.0).clamp(0.0, 1.0)
            + 0.06 * stuck_ema;
        let exploration_p = if pre.high_flex > 0.0 && top_solutions.len() >= 3 {
            exploration_pressure.clamp(0.0, 0.28)
        } else {
            0.0
        };
        let exploratory = exploration_p > 0.0 && rng.gen::<f64>() < exploration_p;
        let overload_feedback = pre.hi_flex
            && target_miss_feedback[0] >= 4
            && target_miss_feedback[0] > target_miss_feedback[1]
            && target_miss_feedback[0] > target_miss_feedback[2];
        let rigid_feedback = pre.flex_avg <= 2.5
            && target_miss_feedback[1] >= 4
            && target_miss_feedback[1] >= target_miss_feedback[2];
        let chain_feedback = !overload_feedback && !rigid_feedback
            && target_miss_feedback[2] >= 5;
        let use_learn = learned_jb.is_some() && learned_mp.is_some() && learned_route_counts.is_some();
        let ctx=bandit_context_idx(late || overload_feedback,use_learn);
        let rule=if exploratory {
                if rng.gen::<f64>() < 0.45 {
                    rules[rng.gen_range(0..rules.len())]
                } else {
                    choose_rule_bandit(&mut rng,&rules,&rule_sum_ctx,&rule_tries_ctx,&rule_sum,&rule_tries,ctx,best_makespan,target_margin,stuck,false,late)
                }
            } else if r<35 {
                let u: f64=rng.gen();
                if u<0.12{Rule::FlexBalance}else if u<0.50{r0}else if u<0.75{r1}else if u<0.90{r2}else{rules[rng.gen_range(0..rules.len())]}
            } else {
                choose_rule_bandit(&mut rng,&rules,&rule_sum_ctx,&rule_tries_ctx,&rule_sum,&rule_tries,ctx,best_makespan,target_margin,stuck,false,late)
            };
        let (k, base_diversify) = if pre.total_ops < 30 {
            let k_val = if k_max <= k_min { k_min } else { rng.gen_range(k_min..=k_max) };
            let div = stuck > 120 && rng.gen::<f64>() < 0.4;
            (k_val, div)
        } else {
            let p_diversify = 1.0 / (1.0 + (-(stuck_ema - 0.05) * 20.0).exp());
            let k_val = ((k_min as f64 + (k_max as f64 - k_min as f64) * p_diversify).floor() as usize)
                .clamp(k_min, k_max);
            let div = rng.gen::<f64>() < p_diversify;
            (k_val, div)
        };
        let feedback_diversify = overload_feedback && !exploratory && pre.total_ops >= 30;
        let diversify = base_diversify || exploratory || feedback_diversify;
        let target = if incumbent_before < (u32::MAX / 2) {
            if exploratory && stuck > 100 {
                None
            } else if exploratory {
                Some(incumbent_before.saturating_add(target_margin.saturating_mul(6)))
            } else {
                let pressure_slack = if overload_feedback {
                    target_margin.saturating_mul(3) / 4
                } else if rigid_feedback || chain_feedback {
                    target_margin / 3
                } else {
                    target_margin / 2
                };
                Some(incumbent_before.saturating_add(pressure_slack.max(1)))
            }
        } else {
            None
        };
        let route_w = if exploratory { -route_w_base * 0.70 } else { route_w_base };
        let directed_guides = if directed_restarts_left > 0
            && stuck >= 45
            && pre.hi_flex
            && pre.total_ops >= 30
            && !exploratory
            && r % 7 == 0
        {
            if let Some(ref incumbent) = best_solution {
                let pattern = 3usize.saturating_sub(directed_restarts_left);
                directed_restarts_left = directed_restarts_left.saturating_sub(1);
                Some(incumbent_directed_restart_guides(
                    pre,
                    challenge,
                    incumbent,
                    learned_jb.as_deref(),
                    learned_mp.as_deref(),
                    pattern,
                ))
            } else {
                None
            }
        } else {
            None
        };
        let (sol,mk)=if let Some((directed_jb, directed_mp)) = directed_guides {
                construct_solution_conflict(
                    challenge,pre,rule,k,target,&mut rng,
                    Some(&directed_jb),Some(&directed_mp),
                    learned_route_counts.as_ref(),route_w, true,
                )?
            } else if use_learn {
                construct_solution_conflict(
                    challenge,pre,rule,k,target,&mut rng,
                    if exploratory { None } else { learned_jb.as_deref() },
                    learned_mp.as_deref(),learned_route_counts.as_ref(),route_w,diversify,
                )?
            } else {
                construct_solution_conflict(challenge,pre,rule,k,target,&mut rng,None,None,None,0.0,diversify)?
            };
        let ridx=rule_idx(rule);rule_tries[ridx]=rule_tries[ridx].saturating_add(1);rule_sum[ridx]+=mk as f64;rule_tries_ctx[ctx][ridx]=rule_tries_ctx[ctx][ridx].saturating_add(1);rule_sum_ctx[ctx][ridx]+=mk as f64;
        if let Some(target_mk) = target {
            if mk > target_mk {
                let cause = target_failure_cause(pre, challenge, &sol, mk);
                for (idx, feedback) in target_miss_feedback.iter_mut().enumerate() {
                    if idx == cause {
                        *feedback = feedback.saturating_add(3).min(12);
                    } else {
                        *feedback = feedback.saturating_sub(1);
                    }
                }
            } else {
                for feedback in &mut target_miss_feedback {
                    *feedback = feedback.saturating_sub(2);
                }
            }
        }
        let exploration_archive_limit = incumbent_before.saturating_add(
            target_margin.saturating_mul(10).max((pre.avg_op_min * 8.0).max(1.0) as u32),
        );
        if !exploratory || incumbent_before >= (u32::MAX / 2) || mk <= exploration_archive_limit {
            push_top_solutions_diverse(pre, challenge, &mut top_solutions, &sol, mk, 15);
        }
        if mk<best_makespan{best_makespan=mk;best_solution=Some(sol.clone());save_solution(&sol)?;stuck=0;if learn_updates_left>0{let (jb,mp,route_counts)=consensus_learning_from_elites(pre,challenge,&top_solutions)?;learned_jb=Some(jb);learned_mp=Some(mp);learned_route_counts=Some(route_counts);learn_updates_left-=1;}}else{stuck=stuck.saturating_add(1);}
        stuck_ema = 0.9 * stuck_ema + 0.1 * ((stuck as f64) / 200.0).min(1.0).max(0.0);
    }
    let route_w_ls: f64=(route_w_base*1.40).clamp(route_w_base,0.40);
    let use_learn_refine = learned_jb.is_some() && learned_mp.is_some() && learned_route_counts.is_some();
    let ctx_refine = bandit_context_idx(true, use_learn_refine);
    let mut refine_results: Vec<(Solution,u32)>=Vec::new();
    let mut sorted_elites: Vec<(u32, Solution)> = top_solutions.iter()
        .map(|(s,mk,_)| (*mk, s.clone()))
        .collect();
    sorted_elites.sort_by_key(|(mk, _)| *mk);
    let top_n = sorted_elites.len().min(5);
    for _ in 0..top_n {
        let target_ls = if best_makespan < (u32::MAX / 2) {
            Some(best_makespan.saturating_add(target_margin / 2))
        } else {
            None
        };
        let rule_ref = choose_rule_bandit(
            &mut rng, &rules, &rule_sum_ctx, &rule_tries_ctx,
            &rule_sum, &rule_tries, ctx_refine, best_makespan,
            target_margin, stuck, false, true,
        );
        let k_ref = if k_hi >= 2 { rng.gen_range(2..=k_hi) } else { k_hi };
        let (sol1, mk1) = construct_solution_conflict(
            challenge, pre, rule_ref, k_ref, target_ls, &mut rng,
            learned_jb.as_deref(), learned_mp.as_deref(),
            learned_route_counts.as_ref(), route_w_ls, false,
        )?;
        if mk1 < best_makespan {
            best_makespan = mk1;
            best_solution = Some(sol1.clone());
            save_solution(&sol1)?;
        }
        refine_results.push((sol1, mk1));
        if mk1 >= best_makespan {
            let rule_div = Rule::Adaptive;
            let k_div = if k_hi >= 2 { rng.gen_range(2..=k_hi) } else { k_hi };
            let (sol2, mk2) = construct_solution_conflict(
                challenge, pre, rule_div, k_div, target_ls, &mut rng,
                learned_jb.as_deref(), learned_mp.as_deref(),
                learned_route_counts.as_ref(), route_w_ls, false,
            )?;
            if mk2 < best_makespan {
                best_makespan = mk2;
                best_solution = Some(sol2.clone());
                save_solution(&sol2)?;
            }
            refine_results.push((sol2, mk2));
        }
    }
    for (sol,mk) in refine_results{push_top_solutions_diverse(pre, challenge, &mut top_solutions, &sol, mk, 15);}
    let flow_like = pre.flow_route.is_some() || (pre.jobshopness < 0.28 && pre.flex_avg <= 3.0);
    let route_rich = !flow_like && pre.hi_flex && pre.jobshopness >= 0.35;
    let (ls_run_cap, cp_run_cap, cpw_run_cap, tabu_primary_iters, tabu_final_iters) = if route_rich {
        (10usize, 15usize, 12usize, 75_000usize, 35_000usize)
    } else if flow_like || pre.flex_avg <= 2.5 {
        (15usize, 10usize, 5usize, 100_000usize, 55_000usize)
    } else {
        (15usize, 12usize, 10usize, 80_000usize, 40_000usize)
    };

    top_solutions.sort_by_key(|x| x.1);
    let ls_runs=top_solutions.len().min(ls_run_cap);
    let ls_seeds: Vec<Solution>=top_solutions.iter().take(ls_runs).map(|x|x.0.clone()).collect();
    for base_sol in &ls_seeds {
        if let Some((sol2,mk2))=critical_block_move_local_search_ex(pre,challenge,base_sol,8,128,24)?{
            if mk2<best_makespan{best_makespan=mk2;best_solution=Some(sol2.clone());save_solution(&sol2)?;}
            push_top_solutions_diverse(pre, challenge, &mut top_solutions, &sol2, mk2, 15);
        }
    }

    top_solutions.sort_by_key(|x| x.1);
    let cp_runs = top_solutions.len().min(cp_run_cap);
    let cp_seeds: Vec<Solution> = top_solutions.iter().take(cp_runs).map(|x| x.0.clone()).collect();
    for base_sol in cp_seeds {
        if let Ok(Some((sol2, mk2))) = iterative_cp_descent(pre, challenge, &base_sol, 8) {
            if mk2 < best_makespan {
                best_makespan = mk2;
                best_solution = Some(sol2.clone());
                save_solution(&sol2)?;
            }
            push_top_solutions_diverse(pre, challenge, &mut top_solutions, &sol2, mk2, 15);
        }
    }

    top_solutions.sort_by_key(|x| x.1);
    let cpw_runs = top_solutions.len().min(cpw_run_cap);
    let cpw_seeds: Vec<Solution> = top_solutions.iter().take(cpw_runs).map(|x| x.0.clone()).collect();
    for base_sol in cpw_seeds {
        if let Ok(Some((sol2, mk2))) = cp_window_exhaustive(pre, challenge, &base_sol, 6) {
            if mk2 < best_makespan {
                best_makespan = mk2;
                best_solution = Some(sol2.clone());
                save_solution(&sol2)?;
            }
        }
    }

    if let Some(sol) = best_solution.clone() {
        if let Ok(Some((sol2, mk2))) = iterative_cp_descent(pre, challenge, &sol, 15) {
            if mk2 < best_makespan {
                best_makespan = mk2;
                best_solution = Some(sol2.clone());
                save_solution(&sol2)?;
            }
            let mut cur = sol2;
            if let Ok(Some((sol3, mk3))) = cp_window_exhaustive(pre, challenge, &cur, 4) {
                if mk3 < best_makespan {
                    best_makespan = mk3;
                    best_solution = Some(sol3.clone());
                    save_solution(&sol3)?;
                }
                cur = sol3;
            }
            if let Ok(Some((sol4, mk4))) = iterative_cp_descent(pre, challenge, &cur, 6) {
                if mk4 < best_makespan {
                    best_makespan = mk4;
                    best_solution = Some(sol4.clone());
                    save_solution(&sol4)?;
                }
            }
        }
    }

    {
        let tenure = 8 + challenge.num_jobs / 4;
        let mut elites: Vec<(Solution, u32)> = Vec::with_capacity(3);
        if let Some(ref sol) = best_solution {
            elites.push((sol.clone(), best_makespan));
        }
        top_solutions.sort_by_key(|x| x.1);
        for (sol, mk, _) in top_solutions.iter() {
            if elites.len() >= 3 {
                break;
            }
            if elites.iter().any(|(s, m)| *m == *mk && s.job_schedule == sol.job_schedule) {
                continue;
            }
            elites.push((sol.clone(), *mk));
        }
        for (seed_sol, seed_mk) in &elites {
            if let Some((s, m)) = super::fjsp_medium::tabu_run_mg(
                pre, challenge, seed_sol, *seed_mk, tabu_primary_iters, tenure, true, 8, &mut rng, 0, true,
            )? {
                if m < best_makespan {
                    best_makespan = m;
                    best_solution = Some(s.clone());
                    save_solution(&s)?;
                }
            }
        }
        if let Some(sol) = best_solution.clone() {
            if let Some((s, m)) = super::fjsp_medium::tabu_run_mg(
                pre, challenge, &sol, best_makespan, tabu_final_iters, tenure, true, 6, &mut rng, 0, true,
            )? {
                if m < best_makespan {
                    best_solution = Some(s.clone());
                    save_solution(&s)?;
                }
            }
        }
    }

    if let Some(sol)=best_solution{save_solution(&sol)?;}
    Ok(())
}