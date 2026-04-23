use anyhow::{anyhow, Result};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use tig_challenges::job_scheduling::*;
use super::types::*;
use super::infra_shared::*;
use crate::{seeded_hasher, HashMap};

type DetCache = HashMap<(u64, usize, usize, usize, u8), Option<(Solution, u32)>>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Rule {
    Adaptive, BnHeavy, EndTight, CriticalPath, MostWork, LeastFlex, Regret, ShortestProc, FlexBalance,
}

#[allow(dead_code)]
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


#[inline]
fn rule_idx(r: Rule) -> usize {
    match r { Rule::Adaptive=>0, Rule::BnHeavy=>1, Rule::EndTight=>2, Rule::CriticalPath=>3, Rule::MostWork=>4, Rule::LeastFlex=>5, Rule::Regret=>6, Rule::ShortestProc=>7, Rule::FlexBalance=>8 }
}

fn choose_rule_bandit(rng: &mut SmallRng, rules: &[Rule], rule_best: &[u32], rule_tries: &[u32], global_best: u32, margin: u32, stuck: usize, chaos_like: bool, late_phase: bool) -> Rule {
    if rules.is_empty() { return Rule::Adaptive; }
    let mut best_seen = global_best; for &mk in rule_best { if mk < best_seen { best_seen = mk; } }
    let scale = (margin as f64).max(1.0); let s = ((stuck as f64)/140.0).clamp(0.0,1.0); let explore_mix = (0.10+0.55*s).clamp(0.10,0.65);

    let mut sum = 0.0;
    for &r in rules.iter() {
        let mk=rule_best[rule_idx(r)]; let t=rule_tries[rule_idx(r)].max(1) as f64;
        let delta=mk.saturating_sub(best_seen) as f64; let exploit=(-delta/scale).exp(); let explore=(1.0/t).sqrt();
        let mut ww=(1.0-explore_mix)*exploit+explore_mix*explore; ww=ww.max(1e-6);
        if chaos_like{ww=ww.powf(0.70);}else if late_phase{ww=ww.powf(1.18);}
        sum += ww.max(0.0);
    }

    if !(sum>0.0) { return rules[rng.gen_range(0..rules.len())]; }

    let mut r=rng.gen::<f64>()*sum;
    for &rule in rules.iter() {
        let mk=rule_best[rule_idx(rule)]; let t=rule_tries[rule_idx(rule)].max(1) as f64;
        let delta=mk.saturating_sub(best_seen) as f64; let exploit=(-delta/scale).exp(); let explore=(1.0/t).sqrt();
        let mut ww=(1.0-explore_mix)*exploit+explore_mix*explore; ww=ww.max(1e-6);
        if chaos_like{ww=ww.powf(0.70);}else if late_phase{ww=ww.powf(1.18);}
        r-=ww.max(0.0); if r<=0.0 { return rule; }
    }
    rules[rules.len()-1]
}

fn construct_solution_conflict_mode<const HAS_TARGET: bool, const HAS_JOB_BIAS: bool, const HAS_MACHINE_PENALTY: bool, const USE_ROUTE_PREF: bool>(
    challenge: &Challenge, pre: &Pre, rule: Rule, k: usize, target_mk: u32,
    rng: &mut SmallRng, job_bias: &[f64], machine_penalty: &[f64],
    route_pref: Option<&RoutePrefLite>, horizon: f64, time_scale: f64,
) -> Result<(Solution, u32)> {
    let num_jobs=challenge.num_jobs; let num_machines=challenge.num_machines;
    let mut job_next_op=vec![0usize;num_jobs]; let mut job_ready_time=vec![0u32;num_jobs];
    let mut machine_avail=vec![0u32;num_machines]; let mut machine_load=pre.machine_load0.clone();
    let mut job_schedule: Vec<Vec<(usize,u32)>>=pre.job_ops_len.iter().map(|&len|Vec::with_capacity(len)).collect();
    let mut remaining_ops=pre.total_ops; let mut time=0u32;

    let avg_op_min_scale=pre.avg_op_min.max(1.0);
    let horizon_scale=horizon.max(1.0);
    let time_scale_scale=time_scale.max(1.0);
    let max_job_avg_work_scale=pre.max_job_avg_work.max(1e-9);
    let max_job_bn_scale=pre.max_job_bn.max(1e-9);
    let avg_machine_load_scale=pre.avg_machine_load.max(1e-9);
    let flex_factor_nonneg=pre.flex_factor.max(0.0);
    let bn_focus_u=if pre.bn_focus<=0.0{0.0}else{pre.bn_focus/(1.0+pre.bn_focus)};
    let slack_scale=(0.70 * pre.avg_op_min).max(1.0);
    let job_product=&pre.job_products;

    let mut demand: Vec<u16>=vec![0u16;num_machines];
    let mut raw_by_machine: Vec<Vec<RawCand>>=(0..num_machines).map(|_|Vec::with_capacity(12)).collect();
    let chaotic_like=pre.chaotic_like;
    let mut machine_work: Vec<u64>=if chaotic_like{vec![0u64;num_machines]}else{vec![]};
    let mut sum_work: u64=0;

    let mut job_ops_rem=vec![0usize;num_jobs];
    let mut job_op_ptr: Vec<*const OpInfo>=vec![std::ptr::null();num_jobs];
    let mut job_op_flex=vec![0usize;num_jobs];
    let mut job_op_has_machines=vec![false;num_jobs];
    let mut job_op_min_pt=vec![INF;num_jobs];
    let mut job_rem_min_raw=vec![0u64;num_jobs];
    let mut job_rem_min_u=vec![0.0;num_jobs];
    let mut job_rem_avg_u=vec![0.0;num_jobs];
    let mut job_bn_u=vec![0.0;num_jobs];
    let mut job_dens_u=vec![0.0;num_jobs];
    let mut job_next_u=vec![0.0;num_jobs];
    let mut job_flex_inv=vec![0.0;num_jobs];
    let mut job_flex_u=vec![0.0;num_jobs];

    let mut ready_jobs: Vec<usize> = Vec::with_capacity(num_jobs);
    let mut ready_pos: Vec<usize> = vec![usize::MAX; num_jobs];
    let mut in_ready: Vec<bool> = vec![false; num_jobs];
    let mut ready_heap: std::collections::BinaryHeap<std::cmp::Reverse<(u32, usize)>> =
        std::collections::BinaryHeap::new();
    let mut idle_machines: Vec<usize> = (0..num_machines).collect();
    let mut idle_pos: Vec<usize> = (0..num_machines).collect();
    let mut machine_heap: std::collections::BinaryHeap<std::cmp::Reverse<(u32, usize, u32)>> =
        std::collections::BinaryHeap::new();
    let mut machine_gen: Vec<u32> = vec![0u32; num_machines];
    let mut touched_machines: Vec<usize> = Vec::with_capacity(num_machines);
    let mut touched_gen: Vec<u32> = vec![0u32; num_machines];
    let mut cur_gen: u32 = 0;
    let mut top: Vec<Cand> = if k > 0 { Vec::with_capacity(k) } else { Vec::new() };

    for j in 0..num_jobs {
        let job_len=pre.job_ops_len[j];
        if job_len == 0 { continue; }
        let product=job_product[j];
        let op=&pre.product_ops[product][0];
        let rem_min_raw=pre.product_suf_min[product][0] as u64;
        let rem_min=rem_min_raw as f64;
        let rem_min_n=rem_min/horizon_scale;
        let rem_avg_n=pre.product_suf_avg[product][0]/max_job_avg_work_scale;
        let bn_n=pre.product_suf_bn[product][0]/max_job_bn_scale;
        let density_n=((rem_min/(job_len as f64).max(1.0))/avg_op_min_scale).clamp(0.0,4.0);
        let next_min_n=(pre.product_next_min[product][0] as f64)/horizon_scale;
        let next_term_raw=(0.55*next_min_n+0.45*pre.product_next_flex_inv[product][0])*(1.0+0.30*density_n*pre.high_flex);
        let flex_inv=1.0/(op.flex as f64).max(1.0);
        let flex_term=flex_inv*flex_factor_nonneg;

        job_ops_rem[j]=job_len;
        job_op_ptr[j]=op as *const OpInfo;
        job_op_flex[j]=op.flex as usize;
        job_op_has_machines[j]=!op.machines.is_empty();
        job_op_min_pt[j]=op.min_pt;
        job_rem_min_raw[j]=rem_min_raw;
        job_rem_min_u[j]=if rem_min_n<=0.0{0.0}else{rem_min_n/(1.0+rem_min_n)};
        job_rem_avg_u[j]=if rem_avg_n<=0.0{0.0}else{rem_avg_n/(1.0+rem_avg_n)};
        job_bn_u[j]=if bn_n<=0.0{0.0}else{bn_n/(1.0+bn_n)};
        job_dens_u[j]=if density_n<=0.0{0.0}else{density_n/(1.0+density_n)};
        job_next_u[j]=if next_term_raw<=0.0{0.0}else{next_term_raw/(1.0+next_term_raw)};
        job_flex_inv[j]=flex_inv;
        job_flex_u[j]=if flex_term<=0.0{0.0}else{flex_term/(1.0+flex_term)};

        in_ready[j] = true;
        ready_pos[j] = ready_jobs.len();
        ready_jobs.push(j);
    }

    let advance_frontier = |time: &mut u32,
                                    ready_jobs: &mut Vec<usize>,
                                    ready_pos: &mut [usize],
                                    in_ready: &mut [bool],
                                    ready_heap: &mut std::collections::BinaryHeap<std::cmp::Reverse<(u32, usize)>>,
                                    job_next_op: &[usize],
                                    job_ready_time: &[u32],
                                    idle_machines: &mut Vec<usize>,
                                    idle_pos: &mut [usize],
                                    machine_heap: &mut std::collections::BinaryHeap<std::cmp::Reverse<(u32, usize, u32)>>,
                                    machine_avail: &[u32],
                                    machine_gen: &[u32]| -> Option<u32> {
        while let Some(std::cmp::Reverse((t, m, g))) = machine_heap.peek().copied() {
            if t > *time { break; }
            machine_heap.pop();
            if m >= idle_pos.len() || g != machine_gen[m] || machine_avail[m] != t || idle_pos[m] != usize::MAX { continue; }
            idle_pos[m] = idle_machines.len();
            idle_machines.push(m);
        }

        while let Some(std::cmp::Reverse((t, j))) = ready_heap.peek().copied() {
            if t > *time { break; }
            ready_heap.pop();
            if j >= in_ready.len() || in_ready[j] || job_next_op[j] >= pre.job_ops_len[j] || job_ready_time[j] != t { continue; }
            in_ready[j] = true;
            ready_pos[j] = ready_jobs.len();
            ready_jobs.push(j);
        }

        let next_machine_time = loop {
            let Some(std::cmp::Reverse((t, m, g))) = machine_heap.peek().copied() else { break None; };
            if t <= *time || m >= idle_pos.len() || g != machine_gen[m] || machine_avail[m] != t || idle_pos[m] != usize::MAX {
                machine_heap.pop();
                continue;
            }
            break Some(t);
        };

        let next_ready_time = loop {
            let Some(std::cmp::Reverse((t, j))) = ready_heap.peek().copied() else { break None; };
            if t <= *time || j >= in_ready.len() || in_ready[j] || job_next_op[j] >= pre.job_ops_len[j] || job_ready_time[j] != t {
                ready_heap.pop();
                continue;
            }
            break Some(t);
        };

        let nt = match (next_machine_time, next_ready_time) {
            (Some(a), Some(b)) => a.min(b),
            (Some(a), None) => a,
            (None, Some(b)) => b,
            (None, None) => return None,
        };
        *time = nt;

        while let Some(std::cmp::Reverse((t, m, g))) = machine_heap.peek().copied() {
            if t > nt { break; }
            machine_heap.pop();
            if m >= idle_pos.len() || g != machine_gen[m] || machine_avail[m] != t || idle_pos[m] != usize::MAX { continue; }
            idle_pos[m] = idle_machines.len();
            idle_machines.push(m);
        }

        while let Some(std::cmp::Reverse((t, j))) = ready_heap.peek().copied() {
            if t > nt { break; }
            ready_heap.pop();
            if j >= in_ready.len() || in_ready[j] || job_next_op[j] >= pre.job_ops_len[j] || job_ready_time[j] != t { continue; }
            in_ready[j] = true;
            ready_pos[j] = ready_jobs.len();
            ready_jobs.push(j);
        }

        Some(nt)
    };

    while remaining_ops > 0 {
        if idle_machines.is_empty() {
            advance_frontier(
                &mut time,
                &mut ready_jobs,
                &mut ready_pos,
                &mut in_ready,
                &mut ready_heap,
                &job_next_op,
                &job_ready_time,
                &mut idle_machines,
                &mut idle_pos,
                &mut machine_heap,
                &machine_avail,
                &machine_gen,
            ).ok_or_else(||anyhow!("Stalled"))?;
            continue;
        }

        touched_machines.clear();
        cur_gen=cur_gen.wrapping_add(1);
        if cur_gen==0{
            touched_gen.fill(0);
            cur_gen=1;
        }
        let progress=1.0-(remaining_ops as f64)/(pre.total_ops as f64).max(1.0);
        let cap_per_machine=if k==0{12usize}else{(k+6).min(12)};

        for &job in &ready_jobs {
            let op_ptr=job_op_ptr[job];
            if op_ptr.is_null(){continue;}
            let op=unsafe{&*op_ptr};
            let op_flex=job_op_flex[job];
            if op_flex==0||!job_op_has_machines[job]||job_op_min_pt[job]>=INF{continue;}
            let (best_end,second_end,best_cnt_total,best_cnt_idle)=best_second_and_counts(time,&machine_avail,op);
            if best_end>=INF||best_cnt_idle==0{continue;}

            let _op_idx=job_next_op[job];
            let _ops_rem=job_ops_rem[job]; let jb=if HAS_JOB_BIAS { job_bias[job] } else { 0.0 };
            let flex_inv=job_flex_inv[job]; let scarcity_urg=1.0/(best_cnt_total as f64).max(1.0);
            let regret=if second_end>=INF{pre.avg_op_min*2.6}else{(second_end-best_end) as f64};
            let regn=(regret/avg_op_min_scale).clamp(0.0,6.0); let rigidity=(0.60*flex_inv+0.40*scarcity_urg).clamp(0.0,2.5);

            let flow_term=pre.flow_w*pre.job_flow_pref[job]*(0.65+0.70*(1.0-progress));
            let slack_u=if HAS_TARGET {
                let lb=(time as u64).saturating_add(job_rem_min_raw[job]);
                let slack=(target_mk as i64) - (lb as i64);
                let pos=(slack.max(0) as f64) / slack_scale; let neg=((-slack).max(0) as f64) / slack_scale;
                (1.0 / (1.0 + pos)).clamp(0.0, 1.0) + (0.35 * neg).min(3.0)
            } else { 0.0 };

            let rem_min_u=job_rem_min_u[job];
            let rem_avg_u=job_rem_avg_u[job];
            let bn_u=job_bn_u[job];
            let reg_u=if regn<=0.0{0.0}else{regn/(1.0+regn)};
            let dens_u=job_dens_u[job];
            let next_u=job_next_u[job];
            let end_n=(best_end as f64)/time_scale_scale;
            let end_u=if end_n<=0.0{0.0}else{end_n/(1.0+end_n)};
            let flex_u=job_flex_u[job];
            let sat_scarcity=if scarcity_urg<=0.0{0.0}else{scarcity_urg/(1.0+scarcity_urg)};
            let scarce_slack=scarcity_urg*slack_u;
            let scarce_reg=scarcity_urg*reg_u;
            let prog_gate=if progress<=0.0{0.0}else{progress/(1.0+progress)};
            let base_bias0=jb+flow_term;

            for &(m,pt) in &op.machines {
                if idle_pos[m]==usize::MAX{continue;}
                let end=time.saturating_add(pt); if end!=best_end{continue;}
                if touched_gen[m]!=cur_gen{
                    touched_gen[m]=cur_gen;
                    touched_machines.push(m);
                    demand[m]=0;
                    raw_by_machine[m].clear();
                }
                demand[m]=demand[m].saturating_add(1);
                let mp=if HAS_MACHINE_PENALTY { machine_penalty[m] } else { 0.0 }; let jitter=if k>0{rng.gen::<f64>()*1e-9}else{0.0};
                let load_n=machine_load[m]/avg_machine_load_scale;
                let proc_n=(pt as f64)/avg_op_min_scale;
                let mpen=mp.clamp(0.0,1.0);
                let pop_pen=if pre.chaotic_like&&op_flex>=2{
                    let pop=pre.machine_best_pop[m];
                    (0.07+0.15*(1.0-progress)).clamp(0.05,0.24)*pop*pre.flex_factor
                }else{
                    0.0
                };
                let load_u=if load_n<=0.0{0.0}else{load_n/(1.0+load_n)};
                let proc_u=if proc_n<=0.0{0.0}else{proc_n/(1.0+proc_n)};
                let mpen_u=if mpen<=0.0{0.0}else{mpen/(1.0+mpen)};
                let base_bias=base_bias0+jitter;
                let base=match rule {
                    Rule::CriticalPath => {
                        let chain=rem_min_u*(1.0+next_u);
                        let urgent=scarce_slack*(1.0+scarce_reg*prog_gate);
                        chain+urgent+base_bias-end_u-pop_pen
                    }
                    Rule::MostWork => {
                        let work=rem_avg_u*(1.0+dens_u);
                        let smooth=work*(1.0+load_u);
                        smooth+base_bias-end_u-pop_pen
                    }
                    Rule::LeastFlex => {
                        let rigid=flex_u*(1.0+sat_scarcity);
                        rigid+rem_min_u+next_u+base_bias-end_u-pop_pen
                    }
                    Rule::ShortestProc => {
                        let short=0.0-proc_u;
                        short+rem_min_u*(1.0+next_u)+sat_scarcity+base_bias-end_u-pop_pen
                    }
                    Rule::Regret => {
                        let regret_focus=reg_u*(1.0+sat_scarcity)*(1.0+prog_gate);
                        regret_focus+rem_min_u+next_u+base_bias-end_u-pop_pen
                    }
                    Rule::EndTight => {
                        let tight=scarce_slack*(1.0+scarce_reg);
                        let chain=rem_min_u*(1.0+prog_gate)*(1.0+next_u);
                        let penal=end_u*(1.0+prog_gate)+proc_u+mpen_u*pre.flex_factor;
                        chain+tight+base_bias-penal-pop_pen
                    }
                    Rule::BnHeavy => {
                        let bn_focus=bn_u*(1.0+dens_u)*(1.0+bn_focus_u);
                        let chain=rem_min_u*(1.0+next_u);
                        let penal=end_u+proc_u+load_u*pre.flex_factor+mpen_u*pre.flex_factor;
                        bn_focus+chain+scarce_slack+reg_u+flex_u+base_bias-penal-pop_pen
                    }
                    Rule::Adaptive => {
                        let js=pre.jobshopness;
                        let fl=1.0-js;
                        if js>=fl{
                            let hard=reg_u*(1.0+scarce_reg)+flex_u+rem_min_u*(1.0+next_u);
                            hard+base_bias-(end_u+mpen_u*pre.flex_factor)-pop_pen
                        }else{
                            let flow=rem_avg_u*(1.0+dens_u)+(0.0-proc_u)+slack_u;
                            flow+base_bias-(end_u+load_u*pre.flex_factor)-pop_pen
                        }
                    }
                    Rule::FlexBalance => {
                        let flexible=flex_u*(1.0+sat_scarcity);
                        let chain=(rem_avg_u+rem_min_u)*(1.0+next_u);
                        let penal=end_u+load_u*pre.flex_factor+mpen_u*(1.0+pre.flex_factor);
                        flexible+chain+base_bias-penal-pop_pen
                    }
                };
                push_top_k_raw(&mut raw_by_machine[m],RawCand{job,machine:m,pt,base_score:base,rigidity,reg_n:regn},cap_per_machine);
            }
        }

        let denom=(idle_machines.len() as f64).max(1.0);
        let (conflict_w,conflict_scale)=if chaotic_like{(-(0.05+0.08*(1.0-progress)).clamp(0.04,0.14),(0.95+0.20*pre.flex_factor).clamp(0.90,1.20))}else{((0.09+0.26*pre.jobshopness+0.11*pre.high_flex+0.16*(1.0-progress)).clamp(0.05,0.45),(0.90+0.40*pre.flex_factor).clamp(0.85,1.75))};
        let (bal_w,avg_work)=if chaotic_like{((0.030+0.070*(1.0-progress)).clamp(0.025,0.11),(sum_work as f64)/(num_machines as f64).max(1.0))}else{(0.0,0.0)};

        let mut best: Option<Cand>=None; top.clear();
        if touched_machines.len()>1{touched_machines.sort_unstable();}
        for &m in &touched_machines {
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

        let chosen=if k==0{
            match best{
                Some(c)=>c,
                None=>{
                    advance_frontier(
                        &mut time,
                        &mut ready_jobs,
                        &mut ready_pos,
                        &mut in_ready,
                        &mut ready_heap,
                        &job_next_op,
                        &job_ready_time,
                        &mut idle_machines,
                        &mut idle_pos,
                        &mut machine_heap,
                        &machine_avail,
                        &machine_gen,
                    ).ok_or_else(||anyhow!("Stalled"))?;
                    continue;
                }
            }
        }else{
            if top.is_empty(){
                advance_frontier(
                    &mut time,
                    &mut ready_jobs,
                    &mut ready_pos,
                    &mut in_ready,
                    &mut ready_heap,
                    &job_next_op,
                    &job_ready_time,
                    &mut idle_machines,
                    &mut idle_pos,
                    &mut machine_heap,
                    &machine_avail,
                    &machine_gen,
                ).ok_or_else(||anyhow!("Stalled"))?;
                continue;
            }
            if USE_ROUTE_PREF{
                let rp=route_pref.unwrap();
                let mut best_rb: Option<f64>=None;
                let mut best_idx=0usize;
                let mut keep_cnt=0usize;
                for (i,c) in top.iter().enumerate(){
                    let job=c.job;
                    if job_op_ptr[job].is_null(){continue;}
                    let op_idx=job_next_op[job];
                    let product=job_product[job];
                    let rb=route_pref_bonus_hfs(Some(rp),product,op_idx,c.machine);
                    match best_rb{
                        None=>{best_rb=Some(rb);best_idx=i;keep_cnt=1;}
                        Some(b)=>{
                            if rb>b{best_rb=Some(rb);best_idx=i;keep_cnt=1;}
                            else if rb==b{keep_cnt+=1;}
                        }
                    }
                }
                if keep_cnt==0{
                    choose_from_top_weighted(rng,&top)
                }else if keep_cnt==1{
                    top[best_idx]
                }else{
                    let best_rb=best_rb.unwrap();
                    let mut write=0usize;
                    for i in 0..top.len(){
                        let c=top[i];
                        let job=c.job;
                        if job_op_ptr[job].is_null(){continue;}
                        let op_idx=job_next_op[job];
                        let product=job_product[job];
                        if route_pref_bonus_hfs(Some(rp),product,op_idx,c.machine)==best_rb{
                            top[write]=c;
                            write+=1;
                        }
                    }
                    top.truncate(write);
                    choose_from_top_weighted(rng,&top)
                }
            }else{
                choose_from_top_weighted(rng,&top)
            }
        };

        let job=chosen.job; let machine=chosen.machine; let pt=chosen.pt;
        let product=job_product[job]; let _op_idx=job_next_op[job]; let op=unsafe{&*job_op_ptr[job]};
        let end_time=time.saturating_add(pt);

        in_ready[job] = false;
        let pos = ready_pos[job];
        if pos < ready_jobs.len() && ready_jobs[pos] == job {
            ready_jobs.swap_remove(pos);
            if pos < ready_jobs.len() {
                let moved = ready_jobs[pos];
                ready_pos[moved] = pos;
            }
        }
        ready_pos[job] = usize::MAX;

        let machine_pos = idle_pos[machine];
        if machine_pos < idle_machines.len() {
            idle_machines.swap_remove(machine_pos);
            if machine_pos < idle_machines.len() {
                let moved = idle_machines[machine_pos];
                idle_pos[moved] = machine_pos;
            }
        }
        idle_pos[machine] = usize::MAX;

        job_schedule[job].push((machine,time)); job_next_op[job]+=1; job_ready_time[job]=end_time; machine_avail[machine]=end_time; remaining_ops-=1;

        if job_next_op[job] < pre.job_ops_len[job] {
            let new_op_idx=job_next_op[job];
            let next_op=&pre.product_ops[product][new_op_idx];
            let rem_min_raw=pre.product_suf_min[product][new_op_idx] as u64;
            let rem_min=rem_min_raw as f64;
            let rem_min_n=rem_min/horizon_scale;
            let rem_avg_n=pre.product_suf_avg[product][new_op_idx]/max_job_avg_work_scale;
            let bn_n=pre.product_suf_bn[product][new_op_idx]/max_job_bn_scale;
            let ops_rem=pre.job_ops_len[job]-new_op_idx;
            let density_n=((rem_min/(ops_rem as f64).max(1.0))/avg_op_min_scale).clamp(0.0,4.0);
            let next_min_n=(pre.product_next_min[product][new_op_idx] as f64)/horizon_scale;
            let next_term_raw=(0.55*next_min_n+0.45*pre.product_next_flex_inv[product][new_op_idx])*(1.0+0.30*density_n*pre.high_flex);
            let flex_inv=1.0/(next_op.flex as f64).max(1.0);
            let flex_term=flex_inv*flex_factor_nonneg;

            job_ops_rem[job]=ops_rem;
            job_op_ptr[job]=next_op as *const OpInfo;
            job_op_flex[job]=next_op.flex as usize;
            job_op_has_machines[job]=!next_op.machines.is_empty();
            job_op_min_pt[job]=next_op.min_pt;
            job_rem_min_raw[job]=rem_min_raw;
            job_rem_min_u[job]=if rem_min_n<=0.0{0.0}else{rem_min_n/(1.0+rem_min_n)};
            job_rem_avg_u[job]=if rem_avg_n<=0.0{0.0}else{rem_avg_n/(1.0+rem_avg_n)};
            job_bn_u[job]=if bn_n<=0.0{0.0}else{bn_n/(1.0+bn_n)};
            job_dens_u[job]=if density_n<=0.0{0.0}else{density_n/(1.0+density_n)};
            job_next_u[job]=if next_term_raw<=0.0{0.0}else{next_term_raw/(1.0+next_term_raw)};
            job_flex_inv[job]=flex_inv;
            job_flex_u[job]=if flex_term<=0.0{0.0}else{flex_term/(1.0+flex_term)};

            if end_time==time{
                in_ready[job]=true;
                ready_pos[job]=ready_jobs.len();
                ready_jobs.push(job);
            }else{
                ready_heap.push(std::cmp::Reverse((end_time, job)));
            }
        } else {
            job_ops_rem[job]=0;
            job_op_ptr[job]=std::ptr::null();
            job_op_flex[job]=0;
            job_op_has_machines[job]=false;
            job_op_min_pt[job]=INF;
            job_rem_min_raw[job]=0;
            job_rem_min_u[job]=0.0;
            job_rem_avg_u[job]=0.0;
            job_bn_u[job]=0.0;
            job_dens_u[job]=0.0;
            job_next_u[job]=0.0;
            job_flex_inv[job]=0.0;
            job_flex_u[job]=0.0;
        }

        machine_gen[machine]=machine_gen[machine].wrapping_add(1);
        if end_time==time{
            idle_pos[machine]=idle_machines.len();
            idle_machines.push(machine);
        }else{
            machine_heap.push(std::cmp::Reverse((end_time, machine, machine_gen[machine])));
        }

        if chaotic_like{machine_work[machine]=machine_work[machine].saturating_add(pt as u64);sum_work=sum_work.saturating_add(pt as u64);}
        if op.min_pt<INF&&op.flex>0&&!op.machines.is_empty(){let delta=(op.min_pt as f64)/(op.flex as f64).max(1.0);if delta>0.0{for &(mm,_) in &op.machines{let v=machine_load[mm]-delta;machine_load[mm]=if v>0.0{v}else{0.0};}}}
    }

    let mk=machine_avail.into_iter().max().unwrap_or(0);
    Ok((Solution{job_schedule},mk))
}

fn construct_solution_conflict(
    challenge: &Challenge, pre: &Pre, rule: Rule, k: usize, target_mk: Option<u32>,
    rng: &mut SmallRng, job_bias: Option<&[f64]>, machine_penalty: Option<&[f64]>,
    route_pref: Option<&RoutePrefLite>, route_w: f64, horizon: f64, time_scale: f64,
) -> Result<(Solution, u32)> {
    let empty: &[f64] = &[];
    let routed = if route_w > 0.0 { route_pref } else { None };

    if let Some(tgt) = target_mk {
        if let Some(jb) = job_bias {
            if let Some(mp) = machine_penalty {
                if let Some(rp) = routed {
                    construct_solution_conflict_mode::<true,true,true,true>(challenge,pre,rule,k,tgt,rng,jb,mp,Some(rp),horizon,time_scale)
                } else {
                    construct_solution_conflict_mode::<true,true,true,false>(challenge,pre,rule,k,tgt,rng,jb,mp,None,horizon,time_scale)
                }
            } else if let Some(rp) = routed {
                construct_solution_conflict_mode::<true,true,false,true>(challenge,pre,rule,k,tgt,rng,jb,empty,Some(rp),horizon,time_scale)
            } else {
                construct_solution_conflict_mode::<true,true,false,false>(challenge,pre,rule,k,tgt,rng,jb,empty,None,horizon,time_scale)
            }
        } else if let Some(mp) = machine_penalty {
            if let Some(rp) = routed {
                construct_solution_conflict_mode::<true,false,true,true>(challenge,pre,rule,k,tgt,rng,empty,mp,Some(rp),horizon,time_scale)
            } else {
                construct_solution_conflict_mode::<true,false,true,false>(challenge,pre,rule,k,tgt,rng,empty,mp,None,horizon,time_scale)
            }
        } else if let Some(rp) = routed {
            construct_solution_conflict_mode::<true,false,false,true>(challenge,pre,rule,k,tgt,rng,empty,empty,Some(rp),horizon,time_scale)
        } else {
            construct_solution_conflict_mode::<true,false,false,false>(challenge,pre,rule,k,tgt,rng,empty,empty,None,horizon,time_scale)
        }
    } else if let Some(jb) = job_bias {
        if let Some(mp) = machine_penalty {
            if let Some(rp) = routed {
                construct_solution_conflict_mode::<false,true,true,true>(challenge,pre,rule,k,0,rng,jb,mp,Some(rp),horizon,time_scale)
            } else {
                construct_solution_conflict_mode::<false,true,true,false>(challenge,pre,rule,k,0,rng,jb,mp,None,horizon,time_scale)
            }
        } else if let Some(rp) = routed {
            construct_solution_conflict_mode::<false,true,false,true>(challenge,pre,rule,k,0,rng,jb,empty,Some(rp),horizon,time_scale)
        } else {
            construct_solution_conflict_mode::<false,true,false,false>(challenge,pre,rule,k,0,rng,jb,empty,None,horizon,time_scale)
        }
    } else if let Some(mp) = machine_penalty {
        if let Some(rp) = routed {
            construct_solution_conflict_mode::<false,false,true,true>(challenge,pre,rule,k,0,rng,empty,mp,Some(rp),horizon,time_scale)
        } else {
            construct_solution_conflict_mode::<false,false,true,false>(challenge,pre,rule,k,0,rng,empty,mp,None,horizon,time_scale)
        }
    } else if let Some(rp) = routed {
        construct_solution_conflict_mode::<false,false,false,true>(challenge,pre,rule,k,0,rng,empty,empty,Some(rp),horizon,time_scale)
    } else {
        construct_solution_conflict_mode::<false,false,false,false>(challenge,pre,rule,k,0,rng,empty,empty,None,horizon,time_scale)
    }
}

#[derive(Clone)]
struct EliteParams {
    jb: Vec<f64>,
    mp: Vec<f64>,
    rp: RoutePrefLite,
    score: u32,
}

fn normalize_elite(elite: &mut Vec<EliteParams>, cap: usize) {
    if elite.is_empty() {
        return;
    }
    if elite.len() <= 1 {
        if elite.len() > cap {
            elite.truncate(cap);
        }
        return;
    }

    #[inline]
    fn elite_sig64(e: &EliteParams) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();

        for (i, v) in e.jb.iter().enumerate() {
            (i as u32).hash(&mut hasher);
            v.to_bits().hash(&mut hasher);
        }
        for (i, v) in e.mp.iter().enumerate() {
            (i as u32).hash(&mut hasher);
            v.to_bits().hash(&mut hasher);
        }
        for (p, ops) in e.rp.iter().enumerate() {
            for (o, r) in ops.iter().enumerate() {
                (p as u32).hash(&mut hasher);
                (o as u32).hash(&mut hasher);
                r.best_m.hash(&mut hasher);
                r.second_m.hash(&mut hasher);
                r.best_w.hash(&mut hasher);
                r.second_w.hash(&mut hasher);
            }
        }

        hasher.finish()
    }

    #[inline]
    fn dist(a: u64, b: u64) -> u32 {
        (a ^ b).count_ones()
    }

    let mut view: Vec<(u32, u64, usize)> = elite
        .iter()
        .enumerate()
        .map(|(idx, e)| (e.score, elite_sig64(e), idx))
        .collect();
    view.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

    let best_sig = view[0].1;

    let cand: Vec<(u32, u32, u64, usize)> = view
        .iter()
        .map(|(s, sg, idx)| (*s, dist(*sg, best_sig), *sg, *idx))
        .collect();

    let mut keep = vec![true; cand.len()];
    for i in 0..cand.len() {
        if !keep[i] {
            continue;
        }
        for j in 0..cand.len() {
            if i == j || !keep[i] {
                continue;
            }
            let (si, di, sgi, _) = cand[i];
            let (sj, dj, sgj, _) = cand[j];

            let no_worse = sj <= si && dj >= di;
            let strictly_better = sj < si || dj > di;
            if no_worse && strictly_better {
                if sj == si && dj == di && sgj > sgi {
                    continue;
                }
                keep[i] = false;
            }
        }
    }

    let mut skyline: Vec<(u32, u64, usize)> = Vec::new();
    for (i, k) in keep.iter().copied().enumerate() {
        if k {
            let (s, _d, sg, idx) = cand[i];
            skyline.push((s, sg, idx));
        }
    }
    skyline.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

    let mut selected: Vec<(u32, u64, usize)> = Vec::new();
    selected.push(view[0]);

    #[inline]
    fn already_selected(selected: &[(u32, u64, usize)], idx: usize) -> bool {
        selected.iter().any(|x| x.2 == idx)
    }

    while selected.len() < cap && selected.len() < elite.len() {
        let mut best_pick: Option<(u32, u64, usize, u32)> = None;
        for &(s, sg, idx) in &skyline {
            if already_selected(&selected, idx) {
                continue;
            }
            let mut md = u32::MAX;
            for &(_ss, ssg, _ii) in &selected {
                md = md.min(dist(sg, ssg));
            }
            match best_pick {
                None => best_pick = Some((s, sg, idx, md)),
                Some((bs, bsg, _bidx, bmd)) => {
                    if md > bmd || (md == bmd && (s < bs || (s == bs && sg < bsg))) {
                        best_pick = Some((s, sg, idx, md));
                    }
                }
            }
        }
        if let Some((s, sg, idx, _)) = best_pick {
            selected.push((s, sg, idx));
        } else {
            break;
        }
    }

    while selected.len() < cap && selected.len() < elite.len() {
        let mut best_pick: Option<(u32, u64, usize, u32)> = None;
        for &(s, sg, idx) in &view {
            if already_selected(&selected, idx) {
                continue;
            }
            let mut md = u32::MAX;
            for &(_ss, ssg, _ii) in &selected {
                md = md.min(dist(sg, ssg));
            }
            match best_pick {
                None => best_pick = Some((s, sg, idx, md)),
                Some((bs, bsg, _bidx, bmd)) => {
                    if md > bmd || (md == bmd && (s < bs || (s == bs && sg < bsg))) {
                        best_pick = Some((s, sg, idx, md));
                    }
                }
            }
        }
        if let Some((s, sg, idx, _)) = best_pick {
            selected.push((s, sg, idx));
        } else {
            break;
        }
    }

    let mut new_elite: Vec<EliteParams> = selected
        .into_iter()
        .map(|(_s, _sg, idx)| elite[idx].clone())
        .collect();
    new_elite.sort_by_key(|e| e.score);

    if new_elite.len() > cap {
        new_elite.truncate(cap);
    }
    *elite = new_elite;
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

#[inline]
fn solution_sig64(sol: &Solution) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for (j, ops) in sol.job_schedule.iter().enumerate() {
        for (o, (m, _t)) in ops.iter().enumerate() {
            std::hash::Hash::hash(&(j, o, *m), &mut hasher);
        }
    }
    std::hash::Hasher::finish(&hasher)
}

#[inline]
fn pick_diverse_top_solution(best: &Solution, top: &[(Solution, u32)], scan: usize) -> Option<usize> {
    if top.is_empty() || scan == 0 { return None; }
    let sig_best = solution_sig64(best);
    let lim = scan.min(top.len());
    let mut best_idx: Option<usize> = None;
    let mut best_dist: u32 = 0;
    for i in 0..lim {
        let sig_i = solution_sig64(&top[i].0);
        let dist = (sig_best ^ sig_i).count_ones();
        if best_idx.is_none() || dist > best_dist {
            best_idx = Some(i);
            best_dist = dist;
        }
    }
    best_idx
}

#[inline]
fn exact_solution_sig64(sol: &Solution) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for (j, ops) in sol.job_schedule.iter().enumerate() {
        for (o, (m, t)) in ops.iter().enumerate() {
            std::hash::Hash::hash(&(j, o, *m, *t), &mut hasher);
        }
    }
    std::hash::Hasher::finish(&hasher)
}

#[allow(dead_code)]
#[inline]
fn unique_deterministic_phase_indices(top: &[(Solution, u32)], limit: usize) -> Vec<usize> {
    let lim = limit.min(top.len());
    let mut best_by_sig: std::collections::HashMap<u64, usize> =
        std::collections::HashMap::with_capacity(lim.saturating_mul(2).max(1));
    for i in 0..lim {
        let sig = exact_solution_sig64(&top[i].0);
        match best_by_sig.get_mut(&sig) {
            Some(best_i) => {
                if top[i].1 < top[*best_i].1 {
                    *best_i = i;
                }
            }
            None => {
                best_by_sig.insert(sig, i);
            }
        }
    }
    let mut out: Vec<usize> = best_by_sig.into_values().collect();
    out.sort_by(|&a, &b| top[a].1.cmp(&top[b].1).then_with(|| a.cmp(&b)));
    out
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

fn cached_cbm(pre: &Pre, challenge: &Challenge, sol: &Solution, p1: usize, p2: usize, p3: usize, cache: &mut DetCache) -> Result<Option<(Solution, u32)>> {
    let key = (exact_solution_sig64(sol), p1, p2, p3, 0u8);
    if let Some(hit) = cache.get(&key) { return Ok(hit.clone()); }
    let res = critical_block_move_local_search_ex(pre, challenge, sol, p1, p2, p3)?;
    if cache.len() >= 1024 { cache.clear(); }
    cache.insert(key, res.clone());
    Ok(res)
}

fn cached_gr(pre: &Pre, challenge: &Challenge, sol: &Solution, cache: &mut DetCache) -> Result<Option<(Solution, u32)>> {
    let key = (exact_solution_sig64(sol), 0usize, 0usize, 0usize, 1u8);
    if let Some(hit) = cache.get(&key) { return Ok(hit.clone()); }
    let res = greedy_reassign_pass(pre, challenge, sol)?;
    if cache.len() >= 1024 { cache.clear(); }
    cache.insert(key, res.clone());
    Ok(res)
}

fn maybe_intensify_ls(pre: &Pre, challenge: &Challenge, rng: &mut SmallRng, sol: &Solution, mk: u32, best_mk: u32, target_margin: u32, stuck: usize, late: bool, cache: &mut DetCache) -> Result<Option<(Solution, u32)>> {
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
    cached_cbm(pre, challenge, sol, p1, p2, p3, cache)
}

fn maybe_escape_ls(
    pre: &Pre,
    challenge: &Challenge,
    rng: &mut SmallRng,
    top_solutions: &[(Solution, u32)],
    best_sol: &Solution,
    scan: usize,
    stuck: usize,
    flex01: f64,
    cache: &mut DetCache,
) -> Result<Option<(Solution, u32)>> {
    if top_solutions.is_empty() || stuck < 60 {
        return Ok(None);
    }
    let p = (0.040 + 0.060 * flex01 + 0.040 * ((stuck as f64) / 160.0).clamp(0.0, 1.0)).clamp(0.04, 0.14);
    if rng.gen::<f64>() >= p {
        return Ok(None);
    }

    let idx = pick_diverse_top_solution(best_sol, top_solutions, scan)
        .unwrap_or_else(|| pick_top_idx(rng, top_solutions));
    let base = &top_solutions[idx].0;

    let bump = if flex01 > 0.55 { 1.0 } else { 0.0 };
    let p1 = (34.0 + 8.0 * bump) as usize;
    let p2 = (56.0 + 10.0 * bump) as usize;
    let p3 = (10.0 + 2.0 * bump) as usize;
    cached_cbm(pre, challenge, base, p1, p2, p3, cache)
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
            let old_pos=match ds.machine_seq[cur_machine].iter().position(|&x|x==node){Some(p)=>p,None=>continue};
            let mut best_m=cur_machine; let mut best_pt=cur_pt; let mut best_mk=current_mk; let mut best_ins_pos=0usize;

            {
                let seq=&mut ds.machine_seq[cur_machine];
                seq[old_pos..].rotate_left(1);
                seq.pop();
            }

            for &(new_m,new_pt) in &op_info.machines {
                if new_m==cur_machine{continue;}
                ds.node_machine[node]=new_m; ds.node_pt[node]=new_pt;
                let target_len=ds.machine_seq[new_m].len(); let cur_start=buf.start[node]; let mut sorted_pos=target_len;
                for (k,&nd) in ds.machine_seq[new_m].iter().enumerate(){if buf.start[nd]>=cur_start{sorted_pos=k;break;}}

                let pos0=sorted_pos;
                let pos1=sorted_pos.saturating_sub(1);
                let has_pos1=pos1!=pos0&&pos1<=target_len;
                let pos2=target_len;
                let has_pos2=pos2!=pos0&&pos2!=pos1&&pos2<=target_len;

                let mut cur_pos=pos0;
                {
                    let seq=&mut ds.machine_seq[new_m];
                    seq.push(node);
                    seq[pos0..].rotate_right(1);
                }
                if let Some((test_mk,_))=eval_disj(&ds,&mut buf){if test_mk<best_mk{best_mk=test_mk;best_m=new_m;best_pt=new_pt;best_ins_pos=pos0;}}

                if has_pos1{
                    {
                        let seq=&mut ds.machine_seq[new_m];
                        seq[pos1..=cur_pos].rotate_right(1);
                    }
                    cur_pos=pos1;
                    if let Some((test_mk,_))=eval_disj(&ds,&mut buf){if test_mk<best_mk{best_mk=test_mk;best_m=new_m;best_pt=new_pt;best_ins_pos=pos1;}}
                }

                if has_pos2{
                    {
                        let seq=&mut ds.machine_seq[new_m];
                        seq[cur_pos..].rotate_left(1);
                    }
                    cur_pos=pos2;
                    if let Some((test_mk,_))=eval_disj(&ds,&mut buf){if test_mk<best_mk{best_mk=test_mk;best_m=new_m;best_pt=new_pt;best_ins_pos=pos2;}}
                }

                {
                    let seq=&mut ds.machine_seq[new_m];
                    if cur_pos<seq.len()-1{seq[cur_pos..].rotate_left(1);}
                    seq.pop();
                }
            }

            if best_m!=cur_machine {
                let ins=best_ins_pos.min(ds.machine_seq[best_m].len());
                {
                    let seq=&mut ds.machine_seq[best_m];
                    seq.push(node);
                    seq[ins..].rotate_right(1);
                }
                ds.node_machine[node]=best_m; ds.node_pt[node]=best_pt;
                current_mk=best_mk; improved=true;
            } else {
                let seq=&mut ds.machine_seq[cur_machine];
                seq.push(node);
                seq[old_pos..].rotate_right(1);
                ds.node_machine[node]=cur_machine; ds.node_pt[node]=cur_pt;
            }
        }
    }
    if current_mk>=initial_mk{return Ok(None);}
    let Some((_,_))=eval_disj(&ds,&mut buf) else{return Ok(None)};
    let sol=disj_to_solution(pre,&ds,&buf.start)?; Ok(Some((sol,current_mk)))
}

fn ensure_solution_features(
    pre: &Pre,
    challenge: &Challenge,
    sol: &Solution,
    feature_cache: &mut std::collections::HashMap<u64, (Vec<f64>, Vec<f64>, RoutePrefLite)>,
) -> Result<u64> {
    let sig = exact_solution_sig64(sol);
    match feature_cache.entry(sig) {
        std::collections::hash_map::Entry::Occupied(_) => {}
        std::collections::hash_map::Entry::Vacant(e) => {
            let jb=job_bias_from_solution(pre,sol)?;
            let mp=machine_penalty_from_solution(pre,sol,challenge.num_machines)?;
            let rp=route_pref_from_solution_lite(pre,sol,challenge)?;
            e.insert((jb,mp,rp));
        }
    }
    Ok(sig)
}

pub fn solve(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    pre: &Pre,
    effort: &EffortConfig,
) -> Result<()> {
    let (greedy_sol, greedy_mk) = run_simple_greedy_baseline(challenge)?;
    save_solution(&greedy_sol)?;
    let mut cache: DetCache = HashMap::with_hasher(seeded_hasher(&challenge.seed));
    let mut rng = SmallRng::from_seed(challenge.seed);
    let horizon = pre.machine_load0.iter().cloned().fold(0.0f64, f64::max).max(pre.horizon);
    let time_scale = (horizon * (2.65 + 0.15 * pre.load_cv + 0.10 * pre.jobshopness + 0.10 * pre.high_flex)).max(1.0);
    let rules: Vec<Rule> = vec![Rule::Adaptive,Rule::BnHeavy,Rule::EndTight,Rule::CriticalPath,Rule::MostWork,Rule::LeastFlex,Rule::Regret,Rule::ShortestProc,Rule::FlexBalance];
    let flex01 = (pre.high_flex + pre.jobshopness).clamp(0.0, 1.0);
    let mut best_makespan = greedy_mk;
    let mut best_solution: Option<Solution> = Some(greedy_sol.clone());
    let mut top_solutions: Vec<(Solution,u32)> = Vec::new();
    push_top_solutions(&mut top_solutions, &greedy_sol, greedy_mk, 15);
    let mut feature_cache: std::collections::HashMap<u64, (Vec<f64>, Vec<f64>, RoutePrefLite)> =
        std::collections::HashMap::with_capacity(64);
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
        let sig=ensure_solution_features(pre,challenge,sol,&mut feature_cache)?;
        let cached=feature_cache.get(&sig).unwrap();
        elite.push(EliteParams{jb:cached.0.clone(),mp:cached.1.clone(),rp:cached.2.clone(),score:mk});
    }
    {
        let sig=ensure_solution_features(pre,challenge,&greedy_sol,&mut feature_cache)?;
        let cached=feature_cache.get(&sig).unwrap();
        elite.push(EliteParams{jb:cached.0.clone(),mp:cached.1.clone(),rp:cached.2.clone(),score:greedy_mk});
    }
    normalize_elite(&mut elite, elite_cap);

    let num_restarts=effort.hybrid_flow_shop_iters;
    let k_hi=6usize;
    let mut stuck: usize=0;
    let mut escape_cooldown: usize=0;

    for r in 0..num_restarts {
        if escape_cooldown > 0 { escape_cooldown -= 1; }

        if escape_cooldown == 0 {
            if let Some((sol2,mk2)) = maybe_escape_ls(pre,challenge,&mut rng,&top_solutions,best_solution.as_ref().unwrap(),top_solutions.len().min(15),stuck,flex01,&mut cache)? {
                escape_cooldown = 14;
                let improved = commit_best(save_solution,&mut best_makespan,&mut best_solution,&sol2,mk2)?;
                if improved {
                    stuck = 0;
                    let sig=ensure_solution_features(pre,challenge,&sol2,&mut feature_cache)?;
                    let cached=feature_cache.get(&sig).unwrap();
                    maybe_add_elite(&mut elite,EliteParams{jb:cached.0.clone(),mp:cached.1.clone(),rp:cached.2.clone(),score:mk2},elite_cap);
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
            let sig=ensure_solution_features(pre,challenge,sref,&mut feature_cache)?;
            let cached=feature_cache.get(&sig).unwrap();
            maybe_add_elite(&mut elite,EliteParams{jb:cached.0.clone(),mp:cached.1.clone(),rp:cached.2.clone(),score:mkref},elite_cap);
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

        if let Some((sol2,mk2))=maybe_intensify_ls(pre,challenge,&mut rng,&sol,mk,best_makespan,target_margin,stuck,late,&mut cache)?{sol=sol2;mk=mk2;}

        let ridx=rule_idx(rule); rule_tries[ridx]=rule_tries[ridx].saturating_add(1); rule_best[ridx]=rule_best[ridx].min(mk);
        let improved=commit_best(save_solution,&mut best_makespan,&mut best_solution,&sol,mk)?;

        if improved {
            stuck=0;
            let sig=ensure_solution_features(pre,challenge,&sol,&mut feature_cache)?;
            let cached=feature_cache.get(&sig).unwrap();
            maybe_add_elite(&mut elite,EliteParams{jb:cached.0.clone(),mp:cached.1.clone(),rp:cached.2.clone(),score:mk},elite_cap);
        } else {
            stuck=stuck.saturating_add(1);
            let add_p=(0.075+0.025*flex01).clamp(0.07,0.11);
            if mk<=best_makespan.saturating_add(target_margin/2)&&rng.gen::<f64>()<add_p {
                let sig=ensure_solution_features(pre,challenge,&sol,&mut feature_cache)?;
                let cached=feature_cache.get(&sig).unwrap();
                maybe_add_elite(&mut elite,EliteParams{jb:cached.0.clone(),mp:cached.1.clone(),rp:cached.2.clone(),score:mk},elite_cap);
            }
        }
        push_top_solutions(&mut top_solutions,&sol,mk,15);
    }

    let route_w_ls: f64=(route_w_base*1.40).clamp(route_w_base,0.40);
    let mut refine_results: Vec<(Solution,u32)>=Vec::new();
    let refine_top_len=top_solutions.len();
    let mut refine_phase_exact_sigs: Vec<u64>=Vec::with_capacity(refine_top_len);
    for (sol,_) in top_solutions.iter() {
        refine_phase_exact_sigs.push(exact_solution_sig64(sol));
    }
    for (idx,(base_sol,_)) in top_solutions.iter().enumerate() {
        let sig=refine_phase_exact_sigs[idx];
        match feature_cache.entry(sig) {
            std::collections::hash_map::Entry::Occupied(_) => {}
            std::collections::hash_map::Entry::Vacant(e) => {
                let jb=job_bias_from_solution(pre,base_sol)?;
                let mp_base=machine_penalty_from_solution(pre,base_sol,challenge.num_machines)?;
                let rp_base=route_pref_from_solution_lite(pre,base_sol,challenge)?;
                e.insert((jb,mp_base,rp_base));
            }
        }
        let target_ls=if best_makespan<(u32::MAX/2){Some(best_makespan.saturating_add(target_margin/2))}else{None};
        let mix_ref_p=(0.045+0.10*pre.high_flex+0.09*pre.jobshopness).clamp(0.04,0.22);
        for attempt in 0..10 {
            let rule=match attempt{0=>r0,1=>Rule::Adaptive,2=>Rule::BnHeavy,3=>Rule::EndTight,4=>Rule::Regret,5=>Rule::CriticalPath,6=>Rule::LeastFlex,7=>Rule::MostWork,8=>Rule::FlexBalance,_=>Rule::ShortestProc};
            let k=match attempt%6{0=>2,1=>3,2=>4,3=>5,4=>3,_=>2}.min(k_hi);
            let (mut sol,mut mk)={
                let cached=feature_cache.get(&sig).unwrap();
                let jb:&[f64]=&cached.0;
                let mut mp_ref: Option<&Vec<f64>>=Some(&cached.1);
                let mut rp_ref: Option<&RoutePrefLite>=Some(&cached.2);
                if !elite.is_empty()&&rng.gen::<f64>()<mix_ref_p {
                    let eidx=pick_elite_idx(&mut rng,&elite);
                    if rng.gen::<f64>()<0.62{mp_ref=Some(&elite[eidx].mp);}
                    if rng.gen::<f64>()<0.72{rp_ref=Some(&elite[eidx].rp);}
                    if rng.gen::<f64>()<0.055{rp_ref=None;}
                }
                let rw_j=if rp_ref.is_some(){(route_w_ls*(0.86+0.50*rng.gen::<f64>())).clamp(route_w_ls*0.70,0.45)}else{0.0};
                construct_solution_conflict(challenge,pre,rule,k,target_ls,&mut rng,Some(jb),mp_ref.map(|v|&**v),rp_ref,rw_j,horizon,time_scale)
            }?;
            if let Some((sol2,mk2))=maybe_intensify_ls(pre,challenge,&mut rng,&sol,mk,best_makespan,target_margin,attempt,true,&mut cache)?{sol=sol2;mk=mk2;}
            if commit_best(save_solution,&mut best_makespan,&mut best_solution,&sol,mk)? {
                let sig2=ensure_solution_features(pre,challenge,&sol,&mut feature_cache)?;
                let cached2=feature_cache.get(&sig2).unwrap();
                maybe_add_elite(&mut elite,EliteParams{jb:cached2.0.clone(),mp:cached2.1.clone(),rp:cached2.2.clone(),score:mk},elite_cap);
            }
            refine_results.push((sol,mk));
        }
    }
    for (sol,mk) in refine_results { push_top_solutions(&mut top_solutions,&sol,mk,15); }

    let det_lim=top_solutions.len().min(15);
    let mut det_phase_exact_sigs: Vec<u64>=Vec::with_capacity(det_lim);
    for i in 0..det_lim {
        det_phase_exact_sigs.push(exact_solution_sig64(&top_solutions[i].0));
    }
    let mut best_by_sig: std::collections::HashMap<u64, usize> =
        std::collections::HashMap::with_capacity(det_lim.saturating_mul(2).max(1));
    for i in 0..det_lim {
        let sig=det_phase_exact_sigs[i];
        match best_by_sig.get_mut(&sig) {
            Some(best_i) => {
                if top_solutions[i].1 < top_solutions[*best_i].1 {
                    *best_i = i;
                }
            }
            None => {
                best_by_sig.insert(sig, i);
            }
        }
    }
    let mut ls_base_indices: Vec<usize> = best_by_sig.into_values().collect();
    ls_base_indices.sort_by(|&a, &b| top_solutions[a].1.cmp(&top_solutions[b].1).then_with(|| a.cmp(&b)));
    for &i in &ls_base_indices {
        let base_sol=&top_solutions[i].0;
        if let Some((sol2,mk2))=cached_cbm(pre,challenge,base_sol,40,64,12,&mut cache)?{
            commit_best(save_solution,&mut best_makespan,&mut best_solution,&sol2,mk2)?;
            push_top_solutions(&mut top_solutions,&sol2,mk2,15);
        }
    }

    if let Some(ref sol)=best_solution.clone() {
        if pre.high_flex+pre.jobshopness > 0.55 {
            if let Some((sol2,mk2))=cached_cbm(pre,challenge,sol,50,80,14,&mut cache)?{
                commit_best(save_solution,&mut best_makespan,&mut best_solution,&sol2,mk2)?;
            }
        }
    }

    if let Some(ref sol)=best_solution.clone() {
        let sig_best_exact=exact_solution_sig64(sol);
        let sig_best_div=solution_sig64(sol);
        let greedy_lim=top_solutions.len().min(15);
        let mut greedy_phase_exact_sigs: Vec<u64>=Vec::with_capacity(greedy_lim);
        let mut greedy_phase_div_sigs: Vec<u64>=Vec::with_capacity(greedy_lim);
        for i in 0..greedy_lim {
            let cand=&top_solutions[i].0;
            greedy_phase_exact_sigs.push(exact_solution_sig64(cand));
            greedy_phase_div_sigs.push(solution_sig64(cand));
        }
        let mut best_by_sig: std::collections::HashMap<u64, usize> =
            std::collections::HashMap::with_capacity(greedy_lim.saturating_mul(2).max(1));
        for i in 0..greedy_lim {
            let sig=greedy_phase_exact_sigs[i];
            match best_by_sig.get_mut(&sig) {
                Some(best_i) => {
                    if top_solutions[i].1 < top_solutions[*best_i].1 {
                        *best_i = i;
                    }
                }
                None => {
                    best_by_sig.insert(sig, i);
                }
            }
        }
        let mut greedy_base_indices: Vec<usize> = best_by_sig.into_values().collect();
        greedy_base_indices.sort_by(|&a, &b| top_solutions[a].1.cmp(&top_solutions[b].1).then_with(|| a.cmp(&b)));

        let mut base2: Option<&Solution> = None;
        let mut base2_dist: u32 = 0;
        for &idx in &greedy_base_indices {
            let cand=&top_solutions[idx].0;
            if greedy_phase_exact_sigs[idx]==sig_best_exact { continue; }
            let dist=(sig_best_div ^ greedy_phase_div_sigs[idx]).count_ones();
            if base2.is_none() || dist > base2_dist {
                base2=Some(cand);
                base2_dist=dist;
            }
        }

        let mut best_improved: Option<(Solution, u32)> = None;

        if let Ok(Some((sol2, mk2))) = cached_gr(pre, challenge, sol, &mut cache) {
            if mk2 < best_makespan { best_improved = Some((sol2, mk2)); }
        }

        if let Some(b2) = base2 {
            if let Ok(Some((sol2, mk2))) = cached_gr(pre, challenge, b2, &mut cache) {
                if mk2 < best_makespan && best_improved.as_ref().map_or(true, |x| mk2 < x.1) {
                    best_improved = Some((sol2, mk2));
                }
            }
        }

        if let Some((sol2, _mk2)) = best_improved {
            best_solution = Some(sol2.clone());
            save_solution(&sol2)?;
        }
    }

    if let Some(sol)=best_solution { save_solution(&sol)?; }

    Ok(())
}