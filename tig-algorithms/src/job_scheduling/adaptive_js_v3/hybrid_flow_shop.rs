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
fn slack_urgency_hfs(pre: &Pre, target_mk: Option<u32>, time: u32, product: usize, op_idx: usize) -> f64 {
    let Some(tgt) = target_mk else { return 0.0 };
    let lb = (time as u64).saturating_add(pre.product_suf_min[product][op_idx] as u64);
    let slack = (tgt as i64) - (lb as i64);
    let scale = (0.70 * pre.avg_op_min).max(1.0);
    let pos = (slack.max(0) as f64) / scale; let neg = ((-slack).max(0) as f64) / scale;
    (1.0 / (1.0 + pos)).clamp(0.0, 1.0) + (0.35 * neg).min(3.0)
}

#[inline]
fn route_pref_bonus_hfs(rp: Option<&RoutePrefLite>, product: usize, op_idx: usize, machine: usize) -> f64 {
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
    route_pref: Option<&RoutePrefLite>, route_w: f64, jitter: f64, horizon: f64, time_scale: f64,
) -> f64 {
    let rem_min = pre.product_suf_min[product][op_idx] as f64;
    let rem_avg = pre.product_suf_avg[product][op_idx]; let rem_bn = pre.product_suf_bn[product][op_idx];
    let flex_f = (op.flex as f64).max(1.0); let flex_inv = 1.0/flex_f;
    let rem_min_n = rem_min/horizon.max(1.0); let rem_avg_n = rem_avg/pre.max_job_avg_work.max(1e-9);
    let bn_n = rem_bn/pre.max_job_bn.max(1e-9); let ops_n = (ops_rem as f64)/(pre.max_ops as f64).max(1.0);
    let load_n = dynamic_load/pre.avg_machine_load.max(1e-9); let scar_n = pre.machine_scarcity[machine]/pre.avg_machine_scarcity.max(1e-9);
    let end_n = (best_end as f64)/time_scale.max(1.0); let proc_n = (pt as f64)/pre.avg_op_min.max(1.0);
    let regret = if second_end >= INF { pre.avg_op_min*2.6 } else { (second_end-best_end) as f64 };
    let reg_n = (regret/pre.avg_op_min.max(1.0)).clamp(0.0,6.0);
    let scarcity_urg = 1.0/(best_cnt_total as f64).max(1.0);
    let density_n = ((rem_min/(ops_rem as f64).max(1.0))/pre.avg_op_min.max(1.0)).clamp(0.0,4.0);
    let next_min = pre.product_next_min[product][op_idx] as f64; let next_min_n = next_min/horizon.max(1.0);
    let next_flex_inv = pre.product_next_flex_inv[product][op_idx];
    let p2 = progress*progress; let next_w_base = 0.12+p2*0.28;
    let next_term_raw = (0.55*next_min_n+0.45*next_flex_inv)*(1.0+0.30*density_n*pre.high_flex);
    let js = pre.jobshopness; let fl = 1.0-js;
    let avg_flex_inv = 1.0/pre.flex_avg.max(1.0); let scarce_match = scar_n*(flex_inv-avg_flex_inv);
    let mpen = machine_penalty.clamp(0.0,1.0); let mpen_gain = 1.0+0.85*pre.high_flex;
    let flow_term = pre.flow_w*pre.job_flow_pref[job]*(0.65+0.70*(1.0-progress));
    let slack_u = slack_urgency_hfs(pre, target_mk, time, product, op_idx);
    let slack_w = pre.slack_base*(0.25+0.75*progress); let slack_reg_boost = 1.0+0.40*reg_n*progress;
    let pop_pen = if pre.chaotic_like && op.flex >= 2 { let pop=pre.machine_best_pop[machine]; (0.07+0.15*(1.0-progress)).clamp(0.05,0.24)*pop*pre.flex_factor } else { 0.0 };
    let route_gain = (0.70+0.80*(1.0-progress)).clamp(0.70,1.40);
    let route_term = if route_w>0.0 && op.flex>=2 { route_w*route_gain*route_pref_bonus_hfs(route_pref,product,op_idx,machine) } else { 0.0 };
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

fn choose_rule_bandit(rng: &mut SmallRng, rules: &[Rule], rule_best: &[u32], rule_tries: &[u32], global_best: u32, margin: u32, stuck: usize, chaos_like: bool, late_phase: bool) -> Rule {
    if rules.is_empty() { return Rule::Adaptive; }
    let mut best_seen = global_best; for &mk in rule_best { if mk < best_seen { best_seen = mk; } }
    let scale = (margin as f64).max(1.0); let s = ((stuck as f64)/140.0).clamp(0.0,1.0); let explore_mix = (0.10+0.55*s).clamp(0.10,0.65);
    let mut w = vec![0.0f64; rules.len()];
    for (i, &r) in rules.iter().enumerate() {
        let mk=rule_best[rule_idx(r)]; let t=rule_tries[rule_idx(r)].max(1) as f64;
        let delta=mk.saturating_sub(best_seen) as f64; let exploit=(-delta/scale).exp(); let explore=(1.0/t).sqrt();
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
    route_pref: Option<&RoutePrefLite>, route_w: f64, horizon: f64, time_scale: f64,
) -> Result<(Solution, u32)> {
    let num_jobs=challenge.num_jobs; let num_machines=challenge.num_machines;
    let mut job_next_op=vec![0usize;num_jobs]; let mut job_ready_time=vec![0u32;num_jobs];
    let mut machine_avail=vec![0u32;num_machines]; let mut machine_load=pre.machine_load0.clone();
    let mut job_schedule: Vec<Vec<(usize,u32)>>=pre.job_ops_len.iter().map(|&len|Vec::with_capacity(len)).collect();
    let mut remaining_ops=pre.total_ops; let mut time=0u32;
    let mut demand: Vec<u16>=vec![0u16;num_machines];
    let mut raw_by_machine: Vec<Vec<RawCand>>=(0..num_machines).map(|_|Vec::with_capacity(12)).collect();
    let mut idle_machines: Vec<usize>=Vec::with_capacity(num_machines);
    let chaotic_like=pre.chaotic_like;
    let mut machine_work: Vec<u64>=if chaotic_like{vec![0u64;num_machines]}else{vec![]};
    let mut sum_work: u64=0;
    while remaining_ops > 0 {
        loop {
            idle_machines.clear();
            for m in 0..num_machines { if machine_avail[m]<=time { idle_machines.push(m); } }
            if idle_machines.is_empty() { break; }
            for &m in &idle_machines { demand[m]=0; raw_by_machine[m].clear(); }
            let progress=1.0-(remaining_ops as f64)/(pre.total_ops as f64).max(1.0);
            let cap_per_machine=if k==0{12usize}else{(k+6).min(12)};
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
                    let base=score_candidate(pre,rule,job,product,op_idx,ops_rem,op,m,pt,time,target_mk,best_end,second_end,best_cnt_total,progress,jb,mp,machine_load[m],route_pref,route_w,jitter,horizon,time_scale);
                    push_top_k_raw(&mut raw_by_machine[m],RawCand{job,machine:m,pt,base_score:base,rigidity,reg_n:regn},cap_per_machine);
                }
            }
            let denom=(idle_machines.len() as f64).max(1.0);
            let (conflict_w,conflict_scale)=if chaotic_like{(-(0.05+0.08*(1.0-progress)).clamp(0.04,0.14),(0.95+0.20*pre.flex_factor).clamp(0.90,1.20))}else{((0.09+0.26*pre.jobshopness+0.11*pre.high_flex+0.16*(1.0-progress)).clamp(0.05,0.45),(0.90+0.40*pre.flex_factor).clamp(0.85,1.75))};
            let (bal_w,avg_work)=if chaotic_like{((0.030+0.070*(1.0-progress)).clamp(0.025,0.11),(sum_work as f64)/(num_machines as f64).max(1.0))}else{(0.0,0.0)};
            let mut best: Option<Cand>=None; let mut top: Vec<Cand>=if k>0{Vec::with_capacity(k)}else{Vec::new()};
            for &m in &idle_machines {
                let dem=demand[m] as f64; if dem<=0.0||raw_by_machine[m].is_empty(){continue;}
                let dem_n=((dem-1.0)/denom).clamp(0.0,2.5);
                let bal_pen=if chaotic_like&&bal_w>0.0{let denomw=(avg_work+(pre.avg_op_min*3.0).max(1.0)).max(1.0); let r=(machine_work[m] as f64)/denomw; let done_n=(r/(r+1.0)).clamp(0.0,1.0); -bal_w*done_n}else{0.0};
                for rc in &raw_by_machine[m] {
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

#[derive(Clone)]
struct EliteParams {
    jb: Vec<f64>,
    mp: Vec<f64>,
    rp: RoutePrefLite,
    score: u32,
}

fn normalize_elite(elite: &mut Vec<EliteParams>, cap: usize) {
    elite.sort_by_key(|e| e.score);
    if elite.len() > cap { elite.truncate(cap); }
}

fn pick_elite_idx(rng: &mut SmallRng, elite: &[EliteParams]) -> usize {
    let len = elite.len(); if len <= 1 { return 0; }
    let a = rng.gen_range(0..len); let b = rng.gen_range(0..len);
    if elite[a].score <= elite[b].score { a } else { b }
}

fn pick_top_idx(rng: &mut SmallRng, top: &[(Solution, u32)]) -> usize {
    let len = top.len(); if len <= 1 { return 0; }
    let a = rng.gen_range(0..len); let b = rng.gen_range(0..len);
    if top[a].1 <= top[b].1 { a } else { b }
}

fn maybe_add_elite(elite: &mut Vec<EliteParams>, cand: EliteParams, cap: usize) {
    if elite.is_empty() { elite.push(cand); return; }
    if elite.len() < cap { elite.push(cand); normalize_elite(elite, cap); return; }
    normalize_elite(elite, cap);
    let worst = elite.last().map(|e| e.score).unwrap_or(u32::MAX);
    if cand.score < worst {
        if let Some(last) = elite.last_mut() { *last = cand; } else { elite.push(cand); }
        normalize_elite(elite, cap);
    }
}

fn commit_best(save_solution: &dyn Fn(&Solution) -> Result<()>, best_mk: &mut u32, best_sol: &mut Option<Solution>, sol: &Solution, mk: u32) -> Result<bool> {
    if mk < *best_mk { *best_mk = mk; *best_sol = Some(sol.clone()); save_solution(sol)?; Ok(true) } else { Ok(false) }
}

fn maybe_intensify_ls(pre: &Pre, challenge: &Challenge, rng: &mut SmallRng, sol: &Solution, mk: u32, best_mk: u32, target_margin: u32, stuck: usize, late: bool) -> Result<Option<(Solution, u32)>> {
    let flex = (pre.high_flex + pre.jobshopness).clamp(0.0, 1.5);
    let near_best = mk <= best_mk.saturating_add((target_margin / 3).max(1));
    let very_near_best = mk <= best_mk.saturating_add((target_margin / 6).max(1));
    let do_ls = if mk < best_mk { late || stuck > 20 || flex >= 0.12 || rng.gen::<f64>() < 0.55 }
        else if very_near_best && (late || stuck > 80) { rng.gen::<f64>() < (0.05 + 0.05 * flex).clamp(0.04, 0.11) }
        else if near_best && stuck > 140 { rng.gen::<f64>() < (0.035 + 0.045 * flex).clamp(0.03, 0.085) }
        else { false };
    if !do_ls { return Ok(None); }
    let (p1, p2, p3) = if mk < best_mk { let bump = if flex > 0.60 { 1.0 } else { 0.0 }; (38 + (6.0 * bump) as usize, 60 + (10.0 * bump) as usize, 12) }
        else if stuck > 180 { (30, 48, 10) } else { (24, 36, 8) };
    critical_block_move_local_search_ex(pre, challenge, sol, p1, p2, p3)
}

fn maybe_escape_ls(pre: &Pre, challenge: &Challenge, rng: &mut SmallRng, top_solutions: &[(Solution, u32)], stuck: usize, flex01: f64) -> Result<Option<(Solution, u32)>> {
    if top_solutions.is_empty() || stuck < 60 { return Ok(None); }
    let p = (0.040 + 0.060 * flex01 + 0.040 * ((stuck as f64) / 160.0).clamp(0.0, 1.0)).clamp(0.04, 0.14);
    if rng.gen::<f64>() >= p { return Ok(None); }
    let idx = pick_top_idx(rng, top_solutions); let base = &top_solutions[idx].0;
    let bump = if flex01 > 0.55 { 1.0 } else { 0.0 };
    let p1 = (34.0 + 8.0 * bump) as usize; let p2 = (56.0 + 10.0 * bump) as usize; let p3 = (10.0 + 2.0 * bump) as usize;
    critical_block_move_local_search_ex(pre, challenge, base, p1, p2, p3)
}

fn greedy_reassign_pass(pre: &Pre, challenge: &Challenge, base_sol: &Solution) -> Result<Option<(Solution, u32)>> {
    let mut ds=build_disj_from_solution(pre,challenge,base_sol)?; let mut buf=EvalBuf::new(ds.n); let n=ds.n;
    let Some((mut current_mk,_))=eval_disj(&ds,&mut buf) else{return Ok(None)};
    let initial_mk=current_mk; let mut improved=true; let mut passes=0; let max_passes=3;
    while improved&&passes<max_passes {
        improved=false; passes+=1;
        for node in 0..n {
            let job=ds.node_job[node]; let op_idx=ds.node_op[node]; let product=pre.job_products[job];
            let op_info=&pre.product_ops[product][op_idx]; if op_info.machines.len()<=1{continue;}
            let cur_machine=ds.node_machine[node]; let cur_pt=ds.node_pt[node];
            let mut best_m=cur_machine; let mut best_pt=cur_pt; let mut best_mk=current_mk; let mut best_ins_pos=0usize;
            for &(new_m,new_pt) in &op_info.machines {
                if new_m==cur_machine{continue;}
                let old_pos=match ds.machine_seq[cur_machine].iter().position(|&x|x==node){Some(p)=>p,None=>continue};
                ds.machine_seq[cur_machine].remove(old_pos); ds.node_machine[node]=new_m; ds.node_pt[node]=new_pt;
                let target_len=ds.machine_seq[new_m].len(); let cur_start=buf.start[node]; let mut sorted_pos=target_len;
                for (k,&nd) in ds.machine_seq[new_m].iter().enumerate(){if buf.start[nd]>=cur_start{sorted_pos=k;break;}}
                let mut positions: Vec<usize>=Vec::with_capacity(3);
                for &p in &[sorted_pos,sorted_pos.saturating_sub(1),target_len]{if p<=target_len&&!positions.contains(&p){positions.push(p);}}
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
                current_mk=best_mk; improved=true;
            }
        }
    }
    if current_mk>=initial_mk{return Ok(None);}
    let Some((_,_))=eval_disj(&ds,&mut buf) else{return Ok(None)};
    let sol=disj_to_solution(pre,&ds,&buf.start)?; Ok(Some((sol,current_mk)))
}

pub fn solve(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    pre: &Pre,
    effort: &EffortConfig,
) -> Result<()> {
    let (greedy_sol, greedy_mk) = run_simple_greedy_baseline(challenge)?;
    save_solution(&greedy_sol)?;

    let mut rng = SmallRng::from_seed(challenge.seed);
    let horizon = pre.machine_load0.iter().cloned().fold(0.0f64, f64::max).max(pre.horizon);
    let time_scale = (horizon * (2.65 + 0.15 * pre.load_cv + 0.10 * pre.jobshopness + 0.10 * pre.high_flex)).max(1.0);
    let rules: Vec<Rule> = vec![Rule::Adaptive,Rule::BnHeavy,Rule::EndTight,Rule::CriticalPath,Rule::MostWork,Rule::LeastFlex,Rule::Regret,Rule::ShortestProc,Rule::FlexBalance];
    let flex01 = (pre.high_flex + pre.jobshopness).clamp(0.0, 1.0);
    let mut best_makespan = greedy_mk;
    let mut best_solution: Option<Solution> = Some(greedy_sol.clone());
    let mut top_solutions: Vec<(Solution,u32)> = Vec::new();
    push_top_solutions(&mut top_solutions, &greedy_sol, greedy_mk, 15);
    let target_margin: u32 = ((pre.avg_op_min*(0.9+0.9*pre.high_flex+0.6*pre.jobshopness)).max(1.0)) as u32;
    let route_w_base: f64 = (0.040+0.10*pre.high_flex+0.08*pre.jobshopness).clamp(0.04,0.22);

    if pre.flow_route.is_some()&&pre.flow_pt_by_job.is_some() {
        if let Ok((sol,mk))=neh_reentrant_flow_solution(pre,challenge.num_jobs,challenge.num_machines) {
            commit_best(save_solution,&mut best_makespan,&mut best_solution,&sol,mk)?;
            push_top_solutions(&mut top_solutions,&sol,mk,15);
        }
    }

    let mut ranked: Vec<(Rule,u32,Solution)>=Vec::with_capacity(rules.len());
    for &rule in &rules {
        let (sol,mk)=construct_solution_conflict(challenge,pre,rule,0,None,&mut rng,None,None,None,0.0,horizon,time_scale)?;
        commit_best(save_solution,&mut best_makespan,&mut best_solution,&sol,mk)?;
        push_top_solutions(&mut top_solutions,&sol,mk,15); ranked.push((rule,mk,sol));
    }
    ranked.sort_by_key(|x|x.1);
    let r0=ranked[0].0; let r1=ranked.get(1).map(|x|x.0).unwrap_or(r0); let r2=ranked.get(2).map(|x|x.0).unwrap_or(r1);
    let mut rule_best: Vec<u32>=vec![u32::MAX;10]; let mut rule_tries: Vec<u32>=vec![0u32;10];
    for (rr,mk,_) in &ranked{let idx=rule_idx(*rr);rule_best[idx]=rule_best[idx].min(*mk);rule_tries[idx]=rule_tries[idx].saturating_add(1);}

    let elite_cap: usize = (6usize + (2.0 * flex01).round() as usize).clamp(6, 8);
    let mut elite: Vec<EliteParams> = Vec::new();
    for i in 0..ranked.len().min(3) {
        let sol=&ranked[i].2; let mk=ranked[i].1;
        let jb=job_bias_from_solution(pre,sol)?; let mp=machine_penalty_from_solution(pre,sol,challenge.num_machines)?; let rp=route_pref_from_solution_lite(pre,sol,challenge)?;
        elite.push(EliteParams{jb,mp,rp,score:mk});
    }
    {
        let jb=job_bias_from_solution(pre,&greedy_sol)?; let mp=machine_penalty_from_solution(pre,&greedy_sol,challenge.num_machines)?; let rp=route_pref_from_solution_lite(pre,&greedy_sol,challenge)?;
        elite.push(EliteParams{jb,mp,rp,score:greedy_mk});
    }
    normalize_elite(&mut elite, elite_cap);

    let num_restarts=effort.hybrid_flow_shop_iters;
    let k_hi=6usize;
    let mut stuck: usize=0;
    let mut escape_cooldown: usize=0;

    for r in 0..num_restarts {
        if escape_cooldown > 0 { escape_cooldown -= 1; }

        if escape_cooldown == 0 {
            if let Some((sol2,mk2)) = maybe_escape_ls(pre,challenge,&mut rng,&top_solutions,stuck,flex01)? {
                escape_cooldown = 14;
                let improved = commit_best(save_solution,&mut best_makespan,&mut best_solution,&sol2,mk2)?;
                if improved {
                    stuck = 0;
                    let jb=job_bias_from_solution(pre,&sol2)?; let mp=machine_penalty_from_solution(pre,&sol2,challenge.num_machines)?; let rp=route_pref_from_solution_lite(pre,&sol2,challenge)?;
                    maybe_add_elite(&mut elite,EliteParams{jb,mp,rp,score:mk2},elite_cap);
                } else { stuck = stuck.saturating_add(1); }
                push_top_solutions(&mut top_solutions,&sol2,mk2,15);
                continue;
            } else if stuck > 150 { escape_cooldown = 6; }
        }

        let late = r >= (num_restarts*2)/3;
        let (k_min,k_max) = if stuck>170{(4usize,6usize)}else if stuck>90{(3usize,6usize)}else if stuck>35{(2usize,6usize)}else{(2usize,4usize)};

        let rule = if r < 35 {
            let u: f64=rng.gen();
            if u<0.11{Rule::FlexBalance}else if u<0.18{Rule::ShortestProc}else if u<0.50{r0}else if u<0.75{r1}else if u<0.90{r2}else{rules[rng.gen_range(0..rules.len())]}
        } else {
            choose_rule_bandit(&mut rng,&rules,&rule_best,&rule_tries,best_makespan,target_margin,stuck,false,late)
        };

        let k = if k_max<=k_min{k_min}else if stuck>120&&rng.gen::<f64>()<0.55{k_max}else{rng.gen_range(k_min..=k_max)}.min(k_hi);
        let learn_base = (0.09+0.24*pre.jobshopness+0.20*pre.high_flex).clamp(0.06,0.44);
        let learn_boost = (1.0+0.38*((stuck as f64)/120.0).clamp(0.0,1.0)).clamp(1.0,1.38);
        let learn_p = (learn_base*learn_boost).clamp(0.0,0.65);

        if stuck > 80 && !top_solutions.is_empty() && rng.gen::<f64>() < 0.04 {
            let idx=pick_top_idx(&mut rng,&top_solutions); let (sref,mkref)=(&top_solutions[idx].0,top_solutions[idx].1);
            let jb=job_bias_from_solution(pre,sref)?; let mp=machine_penalty_from_solution(pre,sref,challenge.num_machines)?; let rp=route_pref_from_solution_lite(pre,sref,challenge)?;
            maybe_add_elite(&mut elite,EliteParams{jb,mp,rp,score:mkref},elite_cap);
        }

        let use_learn = !elite.is_empty() && rng.gen::<f64>() < learn_p;
        let target = if best_makespan<(u32::MAX/2){Some(best_makespan.saturating_add(target_margin))}else{None};

        let (mut sol, mut mk) = if use_learn {
            let mix_p=(0.055+0.10*pre.high_flex+0.09*pre.jobshopness+0.16*((stuck as f64)/160.0).clamp(0.0,1.0)).clamp(0.05,0.40);
            let base_idx=pick_elite_idx(&mut rng,&elite); let mut mp_idx=base_idx; let mut rp_idx=base_idx;
            if elite.len()>1&&rng.gen::<f64>()<mix_p{mp_idx=pick_elite_idx(&mut rng,&elite);}
            if elite.len()>1&&rng.gen::<f64>()<mix_p{rp_idx=pick_elite_idx(&mut rng,&elite);}
            let drop_mp_p=(0.030+0.060*pre.high_flex).clamp(0.03,0.10); let drop_rp_p=(0.030+0.070*pre.jobshopness).clamp(0.03,0.12);
            let mp_opt=if rng.gen::<f64>()<drop_mp_p{None}else{Some(&elite[mp_idx].mp)};
            let rp_opt=if rng.gen::<f64>()<drop_rp_p{None}else{Some(&elite[rp_idx].rp)};
            let jitter=(0.80+0.70*rng.gen::<f64>()).clamp(0.65,1.55);
            let route_w=if rp_opt.is_some(){(route_w_base*jitter).clamp(route_w_base*0.55,0.45)}else{0.0};
            construct_solution_conflict(challenge,pre,rule,k,target,&mut rng,Some(&elite[base_idx].jb),mp_opt.map(|v|&**v),rp_opt,route_w,horizon,time_scale)?
        } else {
            construct_solution_conflict(challenge,pre,rule,k,target,&mut rng,None,None,None,0.0,horizon,time_scale)?
        };

        if let Some((sol2,mk2))=maybe_intensify_ls(pre,challenge,&mut rng,&sol,mk,best_makespan,target_margin,stuck,late)?{sol=sol2;mk=mk2;}

        let ridx=rule_idx(rule); rule_tries[ridx]=rule_tries[ridx].saturating_add(1); rule_best[ridx]=rule_best[ridx].min(mk);
        let improved=commit_best(save_solution,&mut best_makespan,&mut best_solution,&sol,mk)?;

        if improved {
            stuck=0;
            let jb=job_bias_from_solution(pre,&sol)?; let mp=machine_penalty_from_solution(pre,&sol,challenge.num_machines)?; let rp=route_pref_from_solution_lite(pre,&sol,challenge)?;
            maybe_add_elite(&mut elite,EliteParams{jb,mp,rp,score:mk},elite_cap);
        } else {
            stuck=stuck.saturating_add(1);
            let add_p=(0.075+0.025*flex01).clamp(0.07,0.11);
            if mk<=best_makespan.saturating_add(target_margin/2)&&rng.gen::<f64>()<add_p {
                let jb=job_bias_from_solution(pre,&sol)?; let mp=machine_penalty_from_solution(pre,&sol,challenge.num_machines)?; let rp=route_pref_from_solution_lite(pre,&sol,challenge)?;
                maybe_add_elite(&mut elite,EliteParams{jb,mp,rp,score:mk},elite_cap);
            }
        }
        push_top_solutions(&mut top_solutions,&sol,mk,15);
    }

    let route_w_ls: f64=(route_w_base*1.40).clamp(route_w_base,0.40);
    let mut refine_results: Vec<(Solution,u32)>=Vec::new();
    for (base_sol,_) in top_solutions.iter() {
        let jb=job_bias_from_solution(pre,base_sol)?; let mp_base=machine_penalty_from_solution(pre,base_sol,challenge.num_machines)?; let rp_base=route_pref_from_solution_lite(pre,base_sol,challenge)?;
        let target_ls=if best_makespan<(u32::MAX/2){Some(best_makespan.saturating_add(target_margin/2))}else{None};
        let mix_ref_p=(0.045+0.10*pre.high_flex+0.09*pre.jobshopness).clamp(0.04,0.22);
        for attempt in 0..10 {
            let rule=match attempt{0=>r0,1=>Rule::Adaptive,2=>Rule::BnHeavy,3=>Rule::EndTight,4=>Rule::Regret,5=>Rule::CriticalPath,6=>Rule::LeastFlex,7=>Rule::MostWork,8=>Rule::FlexBalance,_=>Rule::ShortestProc};
            let k=match attempt%6{0=>2,1=>3,2=>4,3=>5,4=>3,_=>2}.min(k_hi);
            let mut mp_ref: Option<&Vec<f64>>=Some(&mp_base); let mut rp_ref: Option<&RoutePrefLite>=Some(&rp_base);
            if !elite.is_empty()&&rng.gen::<f64>()<mix_ref_p {
                let eidx=pick_elite_idx(&mut rng,&elite);
                if rng.gen::<f64>()<0.62{mp_ref=Some(&elite[eidx].mp);}
                if rng.gen::<f64>()<0.72{rp_ref=Some(&elite[eidx].rp);}
                if rng.gen::<f64>()<0.055{rp_ref=None;}
            }
            let rw_j=if rp_ref.is_some(){(route_w_ls*(0.86+0.50*rng.gen::<f64>())).clamp(route_w_ls*0.70,0.45)}else{0.0};
            let (mut sol,mut mk)=construct_solution_conflict(challenge,pre,rule,k,target_ls,&mut rng,Some(&jb),mp_ref.map(|v|&**v),rp_ref,rw_j,horizon,time_scale)?;
            if let Some((sol2,mk2))=maybe_intensify_ls(pre,challenge,&mut rng,&sol,mk,best_makespan,target_margin,attempt,true)?{sol=sol2;mk=mk2;}
            if commit_best(save_solution,&mut best_makespan,&mut best_solution,&sol,mk)? {
                let jb2=job_bias_from_solution(pre,&sol)?; let mp2=machine_penalty_from_solution(pre,&sol,challenge.num_machines)?; let rp2=route_pref_from_solution_lite(pre,&sol,challenge)?;
                maybe_add_elite(&mut elite,EliteParams{jb:jb2,mp:mp2,rp:rp2,score:mk},elite_cap);
            }
            refine_results.push((sol,mk));
        }
    }
    for (sol,mk) in refine_results { push_top_solutions(&mut top_solutions,&sol,mk,15); }

    let ls_runs=top_solutions.len().min(15);
    for i in 0..ls_runs {
        let base_sol=&top_solutions[i].0;
        if let Some((sol2,mk2))=critical_block_move_local_search_ex(pre,challenge,base_sol,40,64,12)?{
            commit_best(save_solution,&mut best_makespan,&mut best_solution,&sol2,mk2)?;
            push_top_solutions(&mut top_solutions,&sol2,mk2,15);
        }
    }

    if let Some(ref sol)=best_solution.clone() {
        if pre.high_flex+pre.jobshopness > 0.55 {
            if let Some((sol2,mk2))=critical_block_move_local_search_ex(pre,challenge,sol,50,80,14)?{
                commit_best(save_solution,&mut best_makespan,&mut best_solution,&sol2,mk2)?;
            }
        }
    }

    if let Some(ref sol)=best_solution.clone() {
        if let Ok(Some((sol2,mk2)))=greedy_reassign_pass(pre,challenge,sol) {
            if mk2 < best_makespan { best_solution=Some(sol2.clone()); save_solution(&sol2)?; }
        }
    }

    if let Some(sol)=best_solution { save_solution(&sol)?; }

    Ok(())
}
