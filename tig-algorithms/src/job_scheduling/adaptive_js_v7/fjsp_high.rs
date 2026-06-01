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

#[inline]
fn route_pref_bonus_fh(rp: Option<&RoutePrefLite>, product: usize, op_idx: usize, machine: usize) -> f64 {
    let Some(rp) = rp else { return 0.0 };
    if product >= rp.len() || op_idx >= rp[product].len() { return 0.0; }
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
    let slack_w = pre.slack_base*(0.25+0.75*progress); let slack_reg_boost = 1.0+0.40*reg_n*progress;
    let pop_pen = if pre.chaotic_like && op.flex >= 2 { let pop=pre.machine_best_pop[machine]; (0.07+0.15*(1.0-progress)).clamp(0.05,0.24)*pop*pre.flex_factor } else { 0.0 };
    let route_gain = (0.70+0.80*(1.0-progress)).clamp(0.70,1.40);
    let route_term = if route_w>0.0 && op.flex>=2 { route_w*route_gain*route_pref_bonus_fh(route_pref,product,op_idx,machine) } else { 0.0 };
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
    "Map the restart regime into four contextual bandit buckets: early/late × plain/learned.";
    ((late_phase as usize) << 1) | (use_learn as usize)
}

fn choose_rule_bandit(
    rng: &mut SmallRng, rules: &[Rule], rule_best_ctx: &[Vec<u32>], rule_tries_ctx: &[Vec<u32>],
    rule_best_global: &[u32], rule_tries_global: &[u32], ctx: usize,
    global_best: u32, margin: u32, stuck: usize, chaos_like: bool, late_phase: bool,
) -> Rule {
    "Contextual rule bandit that uses statistics from the active early/late × plain/learned regime, with a global fallback only when that regime is still sparse.";
    if rules.is_empty() { return Rule::Adaptive; }
    let ctx_total_tries: u32=if ctx<rule_best_ctx.len()&&ctx<rule_tries_ctx.len(){rules.iter().map(|&r|rule_tries_ctx[ctx][rule_idx(r)]).sum()}else{0};
    let ctx_min_tries=(rules.len() as u32).min(6).max(4);
    let active_ctx=ctx<rule_best_ctx.len()&&ctx<rule_tries_ctx.len()&&ctx_total_tries>=ctx_min_tries;
    let (active_best,active_tries): (&[u32], &[u32])=if active_ctx{(&rule_best_ctx[ctx],&rule_tries_ctx[ctx])}else{(rule_best_global,rule_tries_global)};
    let mut best_seen=u32::MAX; for &r in rules { let mk=active_best[rule_idx(r)]; if mk<best_seen { best_seen=mk; } }
    if best_seen==u32::MAX { best_seen=global_best; }
    let scale = (margin as f64).max(1.0); let s = ((stuck as f64)/140.0).clamp(0.0,1.0); let explore_mix = (0.10+0.55*s).clamp(0.10,0.65);
    let mut w = vec![0.0f64; rules.len()];
    for (i, &r) in rules.iter().enumerate() {
        let idx=rule_idx(r); let mk=active_best[idx]; let t=active_tries[idx].max(1) as f64;
        let exploit=if mk==u32::MAX{0.45}else{(-(mk.saturating_sub(best_seen) as f64)/scale).exp()}; let explore=(1.0/t).sqrt();
        let mut ww=(1.0-explore_mix)*exploit+explore_mix*explore; ww=ww.max(1e-6);
        if chaos_like{ww=ww.powf(0.70);}else if late_phase{ww=ww.powf(1.18);}
        w[i]=ww;
    }
    let mut sum=0.0; for &ww in &w { sum+=ww.max(0.0); }
    if !(sum>0.0) { return rules[rng.gen_range(0..rules.len())]; }
    let mut r=rng.gen::<f64>()*sum;
    for (i,&ww) in w.iter().enumerate() { r-=ww.max(0.0); if r<=0.0 { return rules[i]; } }
    rules[rules.len()-1]
}

fn construct_solution_conflict(
    challenge: &Challenge, pre: &Pre, rule: Rule, k: usize, target_mk: Option<u32>,
    rng: &mut SmallRng, job_bias: Option<&[f64]>, machine_penalty: Option<&[f64]>,
    route_pref: Option<&RoutePrefLite>, route_w: f64,
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
                    let mp=machine_penalty.map(|v|v[m]).unwrap_or(0.0); let jitter=if k>0{rng.gen::<f64>()*1e-9}else{0.0};
                    let base=score_candidate(pre,rule,job,product,op_idx,ops_rem,op,m,pt,time,target_mk,best_end,second_end,best_cnt_total,progress,jb,mp,machine_load[m],route_pref,route_w,jitter);
                    let rc=RawCand{job,machine:m,pt,base_score:base,rigidity,reg_n:regn};
                    if op.flex<=2||best_cnt_total<=2||rigidity>=0.62{push_top_k_raw(&mut raw_rigid_by_machine[m],rc,rigid_cap);}else{push_top_k_raw(&mut raw_gen_by_machine[m],rc,gen_cap);}
                }
            }
            let denom=(idle_machines.len() as f64).max(1.0);
            let (conflict_w,conflict_scale)=if chaotic_like{(-(0.05+0.08*(1.0-progress)).clamp(0.04,0.14),(0.95+0.20*pre.flex_factor).clamp(0.90,1.20))}else{((0.09+0.26*pre.jobshopness+0.11*pre.high_flex+0.16*(1.0-progress)).clamp(0.05,0.45),(0.90+0.40*pre.flex_factor).clamp(0.85,1.75))};
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
    "Return a tight critical-interval insertion set: after predecessor-feasible work, before successor-breaking work, and one clamped anchor near the node's current time.";
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

fn greedy_reassign_pass(pre: &Pre, challenge: &Challenge, base_sol: &Solution) -> Result<Option<(Solution, u32)>> {
    let mut ds=build_disj_from_solution(pre,challenge,base_sol)?; let mut buf=EvalBuf::new(ds.n); let n=ds.n;
    let mut job_op_to_node: Vec<Vec<usize>>=vec![vec![];challenge.num_jobs];
    for nd in 0..n {
        let job=ds.node_job[nd]; let op_idx=ds.node_op[nd];
        if op_idx>=job_op_to_node[job].len(){job_op_to_node[job].resize(op_idx+1,usize::MAX);}
        job_op_to_node[job][op_idx]=nd;
    }
    let Some((mut current_mk,_))=eval_disj(&ds,&mut buf) else{return Ok(None)};
    let initial_mk=current_mk; let use_zone=n>80; let mut improved=true; let mut passes=0; let max_passes=if use_zone{6}else{3};
    while improved&&passes<max_passes {
        improved=false; passes+=1;
        let candidate_nodes: Vec<usize>=if use_zone{bottleneck_zone_nodes(pre,&ds,&buf.start,&job_op_to_node,current_mk)}else{(0..n).collect()};
        for node in candidate_nodes {
            let job=ds.node_job[node]; let op_idx=ds.node_op[node]; let product=pre.job_products[job];
            let op_info=&pre.product_ops[product][op_idx]; if op_info.machines.len()<=1{continue;}
            let cur_machine=ds.node_machine[node]; let cur_pt=ds.node_pt[node];
            let mut best_m=cur_machine; let mut best_pt=cur_pt; let mut best_mk=current_mk; let mut best_ins_pos=0usize;
            for &(new_m,new_pt) in &op_info.machines {
                if new_m==cur_machine{continue;}
                let old_pos=match ds.machine_seq[cur_machine].iter().position(|&x|x==node){Some(p)=>p,None=>continue};
                ds.machine_seq[cur_machine].remove(old_pos); ds.node_machine[node]=new_m; ds.node_pt[node]=new_pt;
                let positions=boundary_insertion_positions(&ds,&buf.start,&job_op_to_node,node,new_m,3);
                for &pos in &positions {
                    ds.machine_seq[new_m].insert(pos,node);
                    if let Some((test_mk,_))=eval_disj(&ds,&mut buf){if test_mk<best_mk{best_mk=test_mk;best_m=new_m;best_pt=new_pt;best_ins_pos=pos;}}
                    ds.machine_seq[new_m].remove(pos);
                }
                ds.machine_seq[cur_machine].insert(old_pos,node); ds.node_machine[node]=cur_machine; ds.node_pt[node]=cur_pt;
            }
            if best_m!=cur_machine {
                let old_pos=ds.machine_seq[cur_machine].iter().position(|&x|x==node).unwrap();
                ds.machine_seq[cur_machine].remove(old_pos);
                let ins=best_ins_pos.min(ds.machine_seq[best_m].len());
                ds.machine_seq[best_m].insert(ins,node); ds.node_machine[node]=best_m; ds.node_pt[node]=best_pt;
                if let Some((mk_after,_))=eval_disj(&ds,&mut buf){current_mk=mk_after;}else{current_mk=best_mk;}
                improved=true;
                if use_zone{break;}
            }
        }
    }
    if current_mk>=initial_mk{return Ok(None);}
    let Some((_,_))=eval_disj(&ds,&mut buf) else{return Ok(None)};
    let sol=disj_to_solution(pre,&ds,&buf.start)?; Ok(Some((sol,current_mk)))
}

#[inline]
fn machine_tail_signature(
    ds: &DisjSchedule,
    starts: &[u32],
    machine: usize,
    tail_edge: u32,
) -> (u32, usize, u32) {
    "Return a lexicographic tail signature (machine end, near-tail count, near-tail work) used to accept only structurally improving plateau moves.";
    let mut end = 0u32;
    let mut tail_count = 0usize;
    let mut tail_work = 0u32;
    for &nd in &ds.machine_seq[machine] {
        let finish = starts[nd].saturating_add(ds.node_pt[nd]);
        if finish > end { end = finish; }
        if finish >= tail_edge {
            tail_count += 1;
            tail_work = tail_work.saturating_add(ds.node_pt[nd]);
        }
    }
    (end, tail_count, tail_work)
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
    let base_route_pref = route_pref_from_solution_lite(pre, base_sol, challenge).ok();

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

        let mut iter_improved = false;
        let mut start_snapshot = buf.start.clone();
        let tail_slack = ((pre.avg_op_min * (1.40 + 0.80 * pre.high_flex + 0.60 * pre.jobshopness)).max(1.0)) as u32;
        let tail_edge = current_mk.saturating_sub(tail_slack);
        let mut current_tail_sig = machine_tail_signature(&ds, &start_snapshot, plateau_bottleneck_m, tail_edge);
        let mut neutral_left = if n > 140 { 1usize } else { 2usize };

        for &nd in &cp_flex {
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
            let mut best_neutral: Option<(usize, u32, usize, (u32, usize, u32))> = None;

            for &(new_m, new_pt) in &op_info.machines {
                if new_m == cur_m { continue; }
                let old_pos = match ds.machine_seq[cur_m].iter().position(|&x| x == nd) { Some(p)=>p, None=>continue };
                ds.machine_seq[cur_m].remove(old_pos);
                ds.node_machine[nd] = new_m; ds.node_pt[nd] = new_pt;
                let positions = boundary_insertion_positions(&ds, &start_snapshot, &job_op_to_node, nd, new_m, 4);
                for &pos in &positions {
                    ds.machine_seq[new_m].insert(pos, nd);
                    if let Some((tmk, _)) = eval_disj(&ds, &mut buf) {
                        if tmk < best_mk {
                            best_mk = tmk; best_m = new_m; best_pt = new_pt; best_ins = pos;
                        } else if consider_neutral && tmk == current_mk {
                            let cand_sig = machine_tail_signature(&ds, &buf.start, plateau_bottleneck_m, tail_edge);
                            if cand_sig < current_tail_sig && best_neutral.as_ref().map_or(true, |bn| cand_sig < bn.3) {
                                best_neutral = Some((new_m, new_pt, pos, cand_sig));
                            }
                        }
                    }
                    ds.machine_seq[new_m].remove(pos);
                }
                ds.machine_seq[cur_m].insert(old_pos, nd);
                ds.node_machine[nd] = cur_m; ds.node_pt[nd] = cur_pt;
            }
            if best_m != cur_m {
                let old_pos = ds.machine_seq[cur_m].iter().position(|&x| x == nd).unwrap();
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
                let old_pos = ds.machine_seq[cur_m].iter().position(|&x| x == nd).unwrap();
                ds.machine_seq[cur_m].remove(old_pos);
                let ins = neutral_ins.min(ds.machine_seq[neutral_m].len());
                ds.machine_seq[neutral_m].insert(ins, nd);
                ds.node_machine[nd] = neutral_m; ds.node_pt[nd] = neutral_pt;
                let _ = eval_disj(&ds, &mut buf);
                start_snapshot.clone_from(&buf.start);
                current_tail_sig = machine_tail_signature(&ds, &buf.start, plateau_bottleneck_m, tail_edge);
                neutral_left = neutral_left.saturating_sub(1);
                iter_improved = true;
            }
        }

        if cp_pair.len() >= 2 {
            let pairs: Vec<(usize, usize)> = cp_pair.windows(2).map(|w| (w[0], w[1])).collect();
            let pair_start = buf.start.clone();

            let cand_positions = |ds: &DisjSchedule, m: usize, nd: usize| -> Vec<usize> {
                boundary_insertion_positions(ds, &pair_start, &job_op_to_node, nd, m, 3)
            };

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

                let mut opts1: Vec<(usize, u32)> = Vec::new();
                opts1.push((cur_m1, cur_pt1));
                opts1.extend(
                    op_info1
                        .machines
                        .iter()
                        .copied()
                        .filter(|&(m, _)| m != cur_m1)
                        .take(4),
                );

                let mut opts2: Vec<(usize, u32)> = Vec::new();
                opts2.push((cur_m2, cur_pt2));
                opts2.extend(
                    op_info2
                        .machines
                        .iter()
                        .copied()
                        .filter(|&(m, _)| m != cur_m2)
                        .take(4),
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
                                ds.machine_seq[m1].insert(p1i, nd1);
                                for &p2i in &p2s {
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
                                let (a, b) = if order_flag == 0 { (nd1, nd2) } else { (nd2, nd1) };

                                let pa = cand_positions(&ds, m, a);
                                for &pai in &pa {
                                    ds.machine_seq[m].insert(pai, a);
                                    let pb = cand_positions(&ds, m, b);
                                    for &pbi in &pb {
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
                    if best_mk_pair < current_mk {
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

                        current_mk = best_mk_pair;
                        let _ = eval_disj(&ds, &mut buf);
                        iter_improved = true;
                        continue;
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
    route_pref: Option<&RoutePrefLite>,
    max_keep: usize,
) -> Vec<(usize, u32)> {
    "Keep the current machine plus up to two escape machines ranked by cheap base-start insertion end, mild route bias, and scarce-machine protection.";
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

    let mut scored: Vec<(f64, usize, u32)> = Vec::with_capacity(op_info.machines.len().saturating_sub(1));
    for &(m, pt) in &op_info.machines {
        if m == cur_m { continue; }
        let positions = boundary_insertion_positions(ds, starts, job_op_to_node, node, m, 2);
        let seq = &ds.machine_seq[m];
        let latest_start = if succ_start != u32::MAX { succ_start.saturating_sub(pt) } else { u32::MAX };
        let scarcity_pen = 0.12 * pre.avg_op_min * (pre.machine_scarcity[m] / pre.avg_machine_scarcity.max(1e-9)).clamp(0.0, 3.0);
        let route_bonus = route_pref_bonus_fh(route_pref, product, op_idx, m) * (0.16 * pre.avg_op_min);
        let mut best_score = f64::INFINITY;

        if positions.is_empty() {
            let est_end = pred_finish.saturating_add(pt);
            let succ_pen = if latest_start != u32::MAX && pred_finish > latest_start { (pred_finish - latest_start) as f64 } else { 0.0 };
            best_score = est_end as f64 + 1.2 * succ_pen + scarcity_pen - route_bonus;
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
                if score < best_score { best_score = score; }
            }
        }

        scored.push((best_score, m, pt));
    }

    scored.sort_by(|a, b| {
        a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.cmp(&b.1))
            .then_with(|| a.2.cmp(&b.2))
    });
    for (_, m, pt) in scored.into_iter().take(max_keep.saturating_sub(1)) {
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
    let base_route_pref = route_pref_from_solution_lite(pre, base_sol, challenge).ok();

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

        let window_sizes = [4usize, 3, 2];
        let mut best_window_improvement = false;

        'window_search: for &wsize in &window_sizes {
            if cp_flex.len() < wsize { continue; }
            let bottleneck_ops: Vec<usize> = cp_flex.iter().copied()
                .filter(|&nd| ds.node_machine[nd] == bottleneck_m)
                .collect();
            if bottleneck_ops.is_empty() { continue; }

            let num_windows = cp_flex.len().saturating_sub(wsize) + 1;
            for w_start in 0..num_windows {
                let window: Vec<usize> = cp_flex[w_start..w_start+wsize].to_vec();
                let has_bottleneck = window.iter().any(|&nd| ds.node_machine[nd] == bottleneck_m);
                if !has_bottleneck { continue; }

                let orig: Vec<(usize, u32, usize)> = window.iter().map(|&nd| {
                    let cur_m = ds.node_machine[nd];
                    let cur_pt = ds.node_pt[nd];
                    let cur_pos = ds.machine_seq[cur_m].iter().position(|&x| x == nd).unwrap_or(0);
                    (cur_m, cur_pt, cur_pos)
                }).collect();
                let base_start = buf.start.clone();
                let opt_cap = if pre.total_ops > 420 || pre.flex_avg > 7.0 || pre.high_flex > 0.55 { 2usize } else { 3usize };

                let options: Vec<Vec<(usize, u32)>> = window.iter().map(|&nd| {
                    cp_window_option_prefilter(pre, &ds, &base_start, &job_op_to_node, nd, base_route_pref.as_ref(), opt_cap)
                }).collect();
                if options.iter().any(|opts| opts.is_empty()) { continue; }
                let mut total_alts = 1usize;
                for opts in &options {
                    total_alts = total_alts.saturating_mul(opts.len().max(1));
                    if total_alts > 128 { break; }
                }
                if total_alts > 128 { continue; }

                let mut best_combo_mk = current_mk;
                let mut best_combo: Option<Vec<(usize, u32)>> = None;

                let total_nodes = window.len();
                let mut combo_indices = vec![0usize; total_nodes];
                'combo_loop: loop {
                    let mut apply_ok = true;
                    for (wi, &nd) in window.iter().enumerate() {
                        if options[wi].is_empty() { apply_ok = false; break; }
                        let (new_m, new_pt) = options[wi][combo_indices[wi]];
                        let cur_m = ds.node_machine[nd];
                        if new_m == cur_m {
                            continue;
                        }
                        let old_pos = match ds.machine_seq[cur_m].iter().position(|&x| x == nd) {
                            Some(p) => p, None => { apply_ok = false; break; }
                        };
                        ds.machine_seq[cur_m].remove(old_pos);
                        ds.node_machine[nd] = new_m;
                        ds.node_pt[nd] = new_pt;
                        ds.machine_seq[new_m].push(nd);
                    }

                    if apply_ok {
                        let affected_machines: Vec<usize> = {
                            let mut ms: Vec<usize> = window.iter().map(|&nd| ds.node_machine[nd]).collect();
                            ms.extend(orig.iter().map(|&(m,_,_)| m));
                            ms.sort(); ms.dedup(); ms
                        };
                        for &m in &affected_machines {
                            let seq = &mut ds.machine_seq[m];
                            seq.sort_by_key(|&nd2| base_start[nd2]);
                        }

                        if let Some((tmk, _)) = eval_disj(&ds, &mut buf) {
                            if tmk < best_combo_mk {
                                best_combo_mk = tmk;
                                best_combo = Some(window.iter().map(|&nd| {
                                    (ds.node_machine[nd], ds.node_pt[nd])
                                }).collect());
                            }
                        }
                    }

                    for (wi, &nd) in window.iter().enumerate() {
                        let (orig_m, orig_pt, orig_pos) = orig[wi];
                        let cur_m = ds.node_machine[nd];
                        if cur_m != orig_m {
                            if let Some(p) = ds.machine_seq[cur_m].iter().position(|&x| x == nd) {
                                ds.machine_seq[cur_m].remove(p);
                            }
                            ds.node_machine[nd] = orig_m;
                            ds.node_pt[nd] = orig_pt;
                            let ins = orig_pos.min(ds.machine_seq[orig_m].len());
                            ds.machine_seq[orig_m].insert(ins, nd);
                        }
                    }
                    buf.start.clone_from(&base_start);

                    let mut carry = true;
                    for wi in (0..total_nodes).rev() {
                        if carry {
                            combo_indices[wi] += 1;
                            if combo_indices[wi] < options[wi].len() {
                                carry = false;
                            } else {
                                combo_indices[wi] = 0;
                            }
                        }
                    }
                    if carry { break 'combo_loop; }
                }

                if let Some(best_assign) = best_combo {
                    if best_combo_mk < current_mk {
                        for (wi, &nd) in window.iter().enumerate() {
                            let (new_m, new_pt) = best_assign[wi];
                            let cur_m = ds.node_machine[nd];
                            if new_m != cur_m {
                                let old_pos = ds.machine_seq[cur_m].iter().position(|&x| x == nd).unwrap();
                                ds.machine_seq[cur_m].remove(old_pos);
                                ds.node_machine[nd] = new_m;
                                ds.node_pt[nd] = new_pt;
                                ds.machine_seq[new_m].push(nd);
                            }
                        }
                        let affected_machines: Vec<usize> = {
                            let mut ms: Vec<usize> = window.iter().map(|&nd| ds.node_machine[nd]).collect();
                            ms.extend(orig.iter().map(|&(m,_,_)| m));
                            ms.sort(); ms.dedup(); ms
                        };
                        for &m in &affected_machines {
                            ds.machine_seq[m].sort_by_key(|&nd2| base_start[nd2]);
                        }
                        let _ = eval_disj(&ds, &mut buf);
                        current_mk = best_combo_mk;
                        best_window_improvement = true;
                        break 'window_search;
                    }
                }
            }
            if best_window_improvement { break; }
        }

        if !best_window_improvement { break; }
    }

    if current_mk >= initial_mk { return Ok(None); }
    let Some((final_mk, _)) = eval_disj(&ds, &mut buf) else { return Ok(None); };
    if final_mk >= initial_mk { return Ok(None); }
    let sol = disj_to_solution(pre, &ds, &buf.start)?;
    Ok(Some((sol, final_mk)))
}

#[inline]
fn solution_machine_signature(pre: &Pre, challenge: &Challenge, sol: &Solution) -> Vec<u16> {
    let num_machines = challenge.num_machines;
    let mut sig: Vec<u16> = Vec::with_capacity(pre.total_ops);
    for job in 0..challenge.num_jobs {
        let lim = pre.job_ops_len[job].min(sol.job_schedule[job].len());
        for op_idx in 0..pre.job_ops_len[job] {
            let m_u16 = if op_idx < lim {
                let m = sol.job_schedule[job][op_idx].0;
                if m < num_machines { m as u16 } else { u16::MAX }
            } else {
                u16::MAX
            };
            sig.push(m_u16);
        }
    }
    sig
}

#[inline]
fn hamming_distance_sig(a: &[u16], b: &[u16]) -> u32 {
    let mut d = 0u32;
    let n = a.len().min(b.len());
    for i in 0..n {
        if a[i] != b[i] {
            d = d.saturating_add(1);
        }
    }
    d
}

fn push_top_solutions_diverse(
    pre: &Pre,
    challenge: &Challenge,
    pool: &mut Vec<(Solution, u32, Vec<u16>)>,
    sol: &Solution,
    mk: u32,
    cap: usize,
) {
    let sig = solution_machine_signature(pre, challenge, sol);
    pool.push((sol.clone(), mk, sig));

    while pool.len() > cap {
        let len = pool.len();
        let mut min_nn = vec![u32::MAX; len];
        for i in 0..len {
            for j in (i + 1)..len {
                let d = hamming_distance_sig(&pool[i].2, &pool[j].2);
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
    elites: &[(Solution, u32, Vec<u16>)],
) -> Result<(Vec<f64>, Vec<f64>, RoutePrefLite)> {
    if elites.is_empty() {
        return Err(anyhow!("No elites for consensus learning"));
    }
    let num_jobs = challenge.num_jobs;
    let num_machines = challenge.num_machines;

    let mut jb_sum = vec![0.0f64; num_jobs];
    for (sol, _, _) in elites.iter() {
        let jb = job_bias_from_solution(pre, sol)?;
        for j in 0..num_jobs {
            jb_sum[j] += jb[j];
        }
    }
    let denom = elites.len() as f64;
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
        for m in 0..num_machines {
            if machine_end[m] == mk {
                bottleneck_cnt[m] = bottleneck_cnt[m].saturating_add(1);
            }
        }
    }
    let mut machine_penalty = vec![0.0f64; num_machines];
    for m in 0..num_machines {
        machine_penalty[m] = (bottleneck_cnt[m] as f64) / denom;
    }

    let mut rp = route_pref_from_solution_lite(pre, &elites[0].0, challenge)?;
    let num_products = pre.product_ops.len();
    let mut counts: Vec<Vec<Vec<u32>>> = pre
        .product_ops
        .iter()
        .map(|ops| ops.iter().map(|_| vec![0u32; num_machines]).collect())
        .collect();

    for (sol, _, _) in elites.iter() {
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
                    counts[product][op_idx][*m] = counts[product][op_idx][*m].saturating_add(1);
                }
            }
        }
    }

    for product in 0..num_products {
        let ops_len = counts[product].len();
        if product >= rp.len() {
            break;
        }
        for op_idx in 0..ops_len {
            if op_idx >= rp[product].len() {
                break;
            }
            let row = &counts[product][op_idx];
            let mut total = 0u32;
            for &c in row.iter() {
                total = total.saturating_add(c);
            }
            if total == 0 {
                continue;
            }

            let mut best_m = 0usize;
            let mut best_c = 0u32;
            let mut second_m = 0usize;
            let mut second_c = 0u32;
            for (m, &c) in row.iter().enumerate() {
                if c > best_c {
                    second_m = best_m;
                    second_c = best_c;
                    best_m = m;
                    best_c = c;
                } else if m != best_m && c > second_c {
                    second_m = m;
                    second_c = c;
                }
            }
            if second_c == 0 {
                second_m = best_m;
            }

            let total64 = total as u64;
            let best_w = ((best_c as u64) * 255u64 / total64) as u8;
            let second_w = ((second_c as u64) * 255u64 / total64) as u8;

            rp[product][op_idx].best_m = (best_m.min(255)) as u8;
            rp[product][op_idx].best_w = best_w;
            rp[product][op_idx].second_m = (second_m.min(255)) as u8;
            rp[product][op_idx].second_w = second_w;
        }
    }

    Ok((jb_sum, machine_penalty, rp))
}

pub fn solve(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    pre: &Pre,
    effort: &EffortConfig,
) -> Result<()> {
    "Run constructor/restart search, sort and snapshot elites before expensive CP phases, remove per-window CP follow-up descents, and keep one stronger final global-best intensification.";
    let (greedy_sol, greedy_mk) = run_simple_greedy_baseline(challenge)?;
    save_solution(&greedy_sol)?;

    let mut rng = SmallRng::from_seed(challenge.seed);
    let rules: Vec<Rule> = vec![Rule::Adaptive,Rule::BnHeavy,Rule::EndTight,Rule::CriticalPath,Rule::MostWork,Rule::LeastFlex,Rule::Regret,Rule::ShortestProc,Rule::FlexBalance];
    let mut best_makespan = greedy_mk;
    let mut best_solution: Option<Solution> = Some(greedy_sol);
    let mut top_solutions: Vec<(Solution, u32, Vec<u16>)> = Vec::new();
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
        let (sol,mk)=construct_solution_conflict(challenge,pre,rule,0,None,&mut rng,None,None,None,0.0)?;
        if mk<best_makespan{best_makespan=mk;best_solution=Some(sol.clone());save_solution(&sol)?;}
        push_top_solutions_diverse(pre, challenge, &mut top_solutions, &sol, mk, 15);
        ranked.push((rule,mk,sol));
    }
    ranked.sort_by_key(|x|x.1);
    let r0=ranked[0].0; let r1=ranked.get(1).map(|x|x.0).unwrap_or(r0); let r2=ranked.get(2).map(|x|x.0).unwrap_or(r1);
    let mut rule_best: Vec<u32>=vec![u32::MAX;10]; let mut rule_tries: Vec<u32>=vec![0u32;10];
    let mut rule_best_ctx: Vec<Vec<u32>>=vec![vec![u32::MAX;10];4]; let mut rule_tries_ctx: Vec<Vec<u32>>=vec![vec![0u32;10];4];
    let base_ctx=bandit_context_idx(false,false);
    for (rr,mk,_) in &ranked{let idx=rule_idx(*rr);rule_best[idx]=rule_best[idx].min(*mk);rule_tries[idx]=rule_tries[idx].saturating_add(1);rule_best_ctx[base_ctx][idx]=rule_best_ctx[base_ctx][idx].min(*mk);rule_tries_ctx[base_ctx][idx]=rule_tries_ctx[base_ctx][idx].saturating_add(1);}
    let (jb0, mp0, rp0) = consensus_learning_from_elites(pre, challenge, &top_solutions)?;
    let mut learned_jb=Some(jb0);
    let mut learned_mp=Some(mp0);
    let mut learned_rp=Some(rp0);
    let mut learn_updates_left=10usize;
    let num_restarts=effort.fjsp_high_iters;
    let k_hi=if pre.flex_avg>8.0{8}else if pre.flex_avg>6.5{7}else{6};
    let mut stuck: usize=0;
    for r in 0..num_restarts {
        let late=r>=(num_restarts*2)/3;
        let (k_min,k_max)=if stuck>170{(4usize,6usize.min(k_hi))}else if stuck>90{(3usize,6usize.min(k_hi))}else if stuck>35{(2usize,6usize.min(k_hi))}else{(2usize,4usize.min(k_hi))};
        let learn_base=(0.08+0.22*pre.jobshopness+0.18*pre.high_flex).clamp(0.05,0.42);
        let learn_boost=(1.0+0.35*((stuck as f64)/120.0).clamp(0.0,1.0)).clamp(1.0,1.35);
        let learn_p=(learn_base*learn_boost).clamp(0.0,0.60);
        let use_learn=learned_jb.is_some()&&learned_mp.is_some()&&rng.gen::<f64>()<learn_p&&learned_rp.is_some();
        let ctx=bandit_context_idx(late,use_learn);
        let rule=if r<35{let u: f64=rng.gen();if u<0.12{Rule::FlexBalance}else if u<0.50{r0}else if u<0.75{r1}else if u<0.90{r2}else{rules[rng.gen_range(0..rules.len())]}}
            else{choose_rule_bandit(&mut rng,&rules,&rule_best_ctx,&rule_tries_ctx,&rule_best,&rule_tries,ctx,best_makespan,target_margin,stuck,false,late)};
        let k=if k_max<=k_min{k_min}else{rng.gen_range(k_min..=k_max)};
        let target=if best_makespan<(u32::MAX/2){Some(best_makespan.saturating_add(target_margin))}else{None};
        let (sol,mk)=if use_learn{construct_solution_conflict(challenge,pre,rule,k,target,&mut rng,learned_jb.as_deref(),learned_mp.as_deref(),learned_rp.as_ref(),route_w_base)?}
            else{construct_solution_conflict(challenge,pre,rule,k,target,&mut rng,None,None,None,0.0)?};
        let ridx=rule_idx(rule);rule_tries[ridx]=rule_tries[ridx].saturating_add(1);rule_best[ridx]=rule_best[ridx].min(mk);rule_tries_ctx[ctx][ridx]=rule_tries_ctx[ctx][ridx].saturating_add(1);rule_best_ctx[ctx][ridx]=rule_best_ctx[ctx][ridx].min(mk);
        push_top_solutions_diverse(pre, challenge, &mut top_solutions, &sol, mk, 15);
        if mk<best_makespan{best_makespan=mk;best_solution=Some(sol.clone());save_solution(&sol)?;stuck=0;if learn_updates_left>0{let (jb,mp,rp)=consensus_learning_from_elites(pre,challenge,&top_solutions)?;learned_jb=Some(jb);learned_mp=Some(mp);learned_rp=Some(rp);learn_updates_left-=1;}}else{stuck=stuck.saturating_add(1);}
    }
    let route_w_ls: f64=(route_w_base*1.40).clamp(route_w_base,0.40);
    let mut refine_results: Vec<(Solution,u32)>=Vec::new();
    for (base_sol,_,_) in top_solutions.iter() {
        let jb=job_bias_from_solution(pre,base_sol)?; let mp=machine_penalty_from_solution(pre,base_sol,challenge.num_machines)?;
        let rp=Some(route_pref_from_solution_lite(pre,base_sol,challenge)?);
        let target_ls=if best_makespan<(u32::MAX/2){Some(best_makespan.saturating_add(target_margin/2))}else{None};
        for attempt in 0..10 {
            let rule=match attempt{0=>r0,1=>Rule::Adaptive,2=>Rule::BnHeavy,3=>Rule::EndTight,4=>Rule::Regret,5=>Rule::CriticalPath,6=>Rule::LeastFlex,7=>Rule::MostWork,8=>Rule::FlexBalance,_=>r1};
            let k=match attempt%4{0=>2,1=>3,2=>4,_=>2}.min(k_hi);
            let (sol,mk)=construct_solution_conflict(challenge,pre,rule,k,target_ls,&mut rng,Some(&jb),Some(&mp),rp.as_ref(),route_w_ls)?;
            if mk<best_makespan{best_makespan=mk;best_solution=Some(sol.clone());save_solution(&sol)?;}
            refine_results.push((sol,mk));
        }
    }
    for (sol,mk) in refine_results{push_top_solutions_diverse(pre, challenge, &mut top_solutions, &sol, mk, 15);}
    top_solutions.sort_by_key(|x| x.1);
    let ls_runs=top_solutions.len().min(15);
    let ls_seeds: Vec<Solution>=top_solutions.iter().take(ls_runs).map(|x|x.0.clone()).collect();
    for base_sol in &ls_seeds {
        if let Some((sol2,mk2))=critical_block_move_local_search_ex(pre,challenge,base_sol,8,128,24)?{
            if mk2<best_makespan{best_makespan=mk2;best_solution=Some(sol2.clone());save_solution(&sol2)?;}
            push_top_solutions_diverse(pre, challenge, &mut top_solutions, &sol2, mk2, 15);
        }
    }
    if let Some(ref sol)=best_solution.clone(){if let Ok(Some((sol2,mk2)))=greedy_reassign_pass(pre,challenge,sol){if mk2<best_makespan{best_makespan=mk2;best_solution=Some(sol2.clone());save_solution(&sol2)?;push_top_solutions_diverse(pre, challenge, &mut top_solutions, &sol2, mk2, 15);}}}

    top_solutions.sort_by_key(|x| x.1);
    let cp_runs = top_solutions.len().min(12);
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
    let cpw_runs = top_solutions.len().min(10);
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

    if let Some(ref sol) = best_solution.clone() {
        if let Ok(Some((sol2, mk2))) = iterative_cp_descent(pre, challenge, sol, 15) {
            if mk2 < best_makespan {
                best_makespan = mk2;
                best_solution = Some(sol2.clone());
                save_solution(&sol2)?;
                if let Ok(Some((sol3, mk3))) = greedy_reassign_pass(pre, challenge, &sol2) {
                    if mk3 < best_makespan {
                        best_makespan = mk3;
                        best_solution = Some(sol3.clone());
                        save_solution(&sol3)?;
                    }
                }
            }
        }
    }

    if let Some(sol)=best_solution{save_solution(&sol)?;}
    Ok(())
}