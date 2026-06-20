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
    let flex_regime=(pre.high_flex+pre.jobshopness).clamp(0.0,1.5);
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
    let mut idle_pos: Vec<usize> = vec![0usize; num_machines];
    for (i, p) in idle_pos.iter_mut().enumerate() { *p = i; }
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
            let exact_only=op_flex<2||((flex_regime<0.34)&&!pre.chaotic_like)||(k==0&&progress<0.10&&best_cnt_total>1);
            let near_band=if exact_only{
                0u32
            }else{
                let base=((pre.avg_op_min*(0.35+0.30*pre.high_flex+0.28*pre.jobshopness+0.16*progress)).round() as u32).max(1);
                let regret_cap=if second_end>=INF{base.max(op.min_pt/2)}else{second_end.saturating_sub(best_end).min(base.max(1))};
                regret_cap.max(1)
            };
            let allow_end=best_end.saturating_add(near_band);
            let detour_mult=if best_cnt_total<=1{0.72}else if best_cnt_total==2{0.86}else{1.0};

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
                let end=time.saturating_add(pt); if end>allow_end{continue;}
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
                let pop=pre.machine_best_pop[m].clamp(0.0,1.2);
                let pop_pen=if pre.chaotic_like&&op_flex>=2{
                    (0.07+0.15*(1.0-progress)).clamp(0.05,0.24)*pop*pre.flex_factor
                }else{
                    0.0
                };
                let load_u=if load_n<=0.0{0.0}else{load_n/(1.0+load_n)};
                let proc_u=if proc_n<=0.0{0.0}else{proc_n/(1.0+proc_n)};
                let mpen_u=if mpen<=0.0{0.0}else{mpen/(1.0+mpen)};
                let end_gap=end.saturating_sub(best_end);
                let end_gap_n=(end_gap as f64)/avg_op_min_scale;
                let end_gap_u=if end_gap_n<=0.0{0.0}else{end_gap_n/(1.0+end_gap_n)};
                let preserve_pen=if op_flex>=2&&flex_regime>0.35{
                    (0.06+0.12*flex_regime.min(1.0)+0.08*progress)*pop*(1.0-flex_inv)
                }else{
                    0.0
                };
                let detour_pen=if end_gap>0{
                    end_gap_u*(0.55+0.45*rigidity+0.18*sat_scarcity)*detour_mult
                }else{
                    0.0
                };
                let detour_credit=if end_gap>0{
                    (0.08+0.10*pre.jobshopness+0.08*pre.high_flex)*(1.0-load_u)*(1.0-pop.min(1.0))
                }else{
                    0.0
                };
                let scarce_machine_bonus=if op_flex>=2&&flex_regime>0.35{
                    (0.03+0.06*progress)*(1.0-pop.min(1.0))*sat_scarcity
                }else{
                    0.0
                };
                let base_bias=base_bias0+jitter;
                let base0=match rule {
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
                let base=base0+scarce_machine_bonus+detour_credit-detour_pen-preserve_pen;
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
            let load_factor = (machine_load[m] as f64) / avg_machine_load_scale;
            let cs = conflict_scale * (1.0 - 0.30 * load_factor).clamp(0.40, 1.60);
            for rc in &raw_by_machine[m] {
                let rig=rc.rigidity.clamp(0.0,2.5); let regc=rc.reg_n.clamp(0.0,4.5);
                let mut boost=conflict_w * cs * dem_n * (1.15*rig+0.85*regc);
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

#[inline]
fn seed_sig_distance(exact_a: u64, coarse_a: u64, exact_b: u64, coarse_b: u64) -> u32 {
    let d_exact = (exact_a ^ exact_b).count_ones();
    let d_coarse = (coarse_a ^ coarse_b).count_ones();
    d_exact.saturating_add(d_coarse.saturating_mul(2))
}

fn select_seed_portfolio_indices(
    best_sol: Option<&Solution>,
    best_mk: u32,
    top: &[(Solution, u32)],
    scan_limit: usize,
    seed_cap: usize,
    quality_band: u32,
) -> Vec<usize> {
    if top.is_empty() || seed_cap == 0 {
        return Vec::new();
    }
    let scan = scan_limit.min(top.len());
    let best_exact = best_sol.map(exact_solution_sig64);
    let best_coarse = best_sol.map(solution_sig64);
    let mut exact_sigs: Vec<u64> = Vec::with_capacity(scan);
    let mut coarse_sigs: Vec<u64> = Vec::with_capacity(scan);
    for i in 0..scan {
        exact_sigs.push(exact_solution_sig64(&top[i].0));
        coarse_sigs.push(solution_sig64(&top[i].0));
    }

    let build_pool = |limit_mk: Option<u32>| -> Vec<usize> {
        let mut best_by_sig: std::collections::HashMap<u64, usize> =
            std::collections::HashMap::with_capacity(scan.saturating_mul(2).max(1));
        for i in 0..scan {
            let keep = match limit_mk {
                Some(lim) => top[i].1 <= lim || best_exact.map_or(false, |sig| exact_sigs[i] == sig),
                None => true,
            };
            if !keep {
                continue;
            }
            match best_by_sig.get_mut(&exact_sigs[i]) {
                Some(best_i) => {
                    if top[i].1 < top[*best_i].1 {
                        *best_i = i;
                    }
                }
                None => {
                    best_by_sig.insert(exact_sigs[i], i);
                }
            }
        }
        let mut pool: Vec<usize> = best_by_sig.into_values().collect();
        pool.sort_by(|&a, &b| top[a].1.cmp(&top[b].1).then_with(|| a.cmp(&b)));
        pool
    };

    let mut pool = build_pool(Some(best_mk.saturating_add(quality_band.max(1))));
    if pool.len() < seed_cap.min(4) {
        pool = build_pool(None);
    }
    if pool.is_empty() {
        return Vec::new();
    }

    let mut out: Vec<usize> = Vec::with_capacity(seed_cap.min(pool.len()));
    let incumbent_idx = best_exact
        .and_then(|sig| pool.iter().copied().find(|&i| exact_sigs[i] == sig))
        .unwrap_or(pool[0]);
    out.push(incumbent_idx);

    if out.len() < seed_cap {
        if let Some(idx) = pool.iter().copied().find(|&i| i != incumbent_idx) {
            out.push(idx);
        }
    }

    if out.len() < seed_cap {
        let ref_exact = best_exact.unwrap_or(exact_sigs[incumbent_idx]);
        let ref_coarse = best_coarse.unwrap_or(coarse_sigs[incumbent_idx]);
        let mut best_pick: Option<(usize, u32, u32)> = None;
        for &i in &pool {
            if out.contains(&i) {
                continue;
            }
            let dist = seed_sig_distance(ref_exact, ref_coarse, exact_sigs[i], coarse_sigs[i]);
            match best_pick {
                None => best_pick = Some((i, dist, top[i].1)),
                Some((bi, bd, bmk)) => {
                    if dist > bd || (dist == bd && (top[i].1 < bmk || (top[i].1 == bmk && i < bi))) {
                        best_pick = Some((i, dist, top[i].1));
                    }
                }
            }
        }
        if let Some((idx, _, _)) = best_pick {
            out.push(idx);
        }
    }

    if out.len() < seed_cap {
        let ref_exact = best_exact.unwrap_or(exact_sigs[incumbent_idx]);
        let ref_coarse = best_coarse.unwrap_or(coarse_sigs[incumbent_idx]);
        let close_limit = best_mk.saturating_add((quality_band / 2).max(1));
        let mut best_pick: Option<(usize, u32, u32)> = None;
        for &i in &pool {
            if out.contains(&i) || top[i].1 > close_limit {
                continue;
            }
            let dist = seed_sig_distance(ref_exact, ref_coarse, exact_sigs[i], coarse_sigs[i]);
            match best_pick {
                None => best_pick = Some((i, dist, top[i].1)),
                Some((bi, bd, bmk)) => {
                    if dist > bd || (dist == bd && (top[i].1 < bmk || (top[i].1 == bmk && i < bi))) {
                        best_pick = Some((i, dist, top[i].1));
                    }
                }
            }
        }
        if let Some((idx, _, _)) = best_pick {
            out.push(idx);
        }
    }

    let target_len = seed_cap.min(pool.len());
    while out.len() < target_len {
        let mut best_pick: Option<(usize, u32, u32)> = None;
        for &i in &pool {
            if out.contains(&i) {
                continue;
            }
            let mut min_dist = u32::MAX;
            for &j in &out {
                min_dist = min_dist.min(seed_sig_distance(exact_sigs[i], coarse_sigs[i], exact_sigs[j], coarse_sigs[j]));
            }
            match best_pick {
                None => best_pick = Some((i, min_dist, top[i].1)),
                Some((bi, bd, bmk)) => {
                    if min_dist > bd || (min_dist == bd && (top[i].1 < bmk || (top[i].1 == bmk && i < bi))) {
                        best_pick = Some((i, min_dist, top[i].1));
                    }
                }
            }
        }
        if let Some((idx, _, _)) = best_pick {
            out.push(idx);
        } else {
            break;
        }
    }
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

    let chain_on = flex > 0.52 && (mk < best_mk || (very_near_best && (late || stuck > 110)));
    if chain_on {
        let mut best: Option<(Solution, u32)> = None;

        if let Some((s1, m1)) = cached_gr(pre, challenge, sol, cache)? {
            if m1 < mk && best.as_ref().map_or(true, |b| m1 < b.1) {
                best = Some((s1.clone(), m1));
            }
            if let Some((s2, m2)) = cached_cbm(pre, challenge, &s1, p1, p2, p3, cache)? {
                if m2 < mk && best.as_ref().map_or(true, |b| m2 < b.1) {
                    best = Some((s2, m2));
                }
            }
        }

        let need_cbm_branch = best.is_none() || flex < 0.85;
        if need_cbm_branch {
            if let Some((s1, m1)) = cached_cbm(pre, challenge, sol, p1, p2, p3, cache)? {
                if m1 < mk && best.as_ref().map_or(true, |b| m1 < b.1) {
                    best = Some((s1.clone(), m1));
                }
                if m1 <= mk.saturating_add((target_margin / 8).max(1)) {
                    if let Some((s2, m2)) = cached_gr(pre, challenge, &s1, cache)? {
                        if m2 < mk && best.as_ref().map_or(true, |b| m2 < b.1) {
                            best = Some((s2, m2));
                        }
                    }
                }
            }
        }
        return Ok(best);
    }

    if flex > 0.62 && near_best && late && rng.gen::<f64>() < 0.35 {
        if let Some(res) = cached_gr(pre, challenge, sol, cache)? {
            return Ok(Some(res));
        }
    }

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

fn greedy_reassign_try_node(
    pre: &Pre,
    ds: &mut DisjSchedule,
    buf: &mut EvalBuf,
    job_pred_node: &[usize],
    cur_starts: &mut Vec<u32>,
    current_mk: &mut u32,
    node: usize,
) -> Result<bool> {
    let job=ds.node_job[node]; let op_idx=ds.node_op[node]; let product=pre.job_products[job];
    let op_info=&pre.product_ops[product][op_idx]; if op_info.machines.len()<=1{return Ok(false);}
    let cur_machine=ds.node_machine[node]; let cur_pt=ds.node_pt[node];
    let old_pos=match ds.machine_seq[cur_machine].iter().position(|&x|x==node){Some(p)=>p,None=>return Ok(false)};
    let cur_start=cur_starts[node];
    let mut best_m=cur_machine; let mut best_pt=cur_pt; let mut best_mk=*current_mk; let mut best_ins_pos=0usize;

    {
        let seq=&mut ds.machine_seq[cur_machine];
        seq[old_pos..].rotate_left(1);
        seq.pop();
    }

    for &(new_m,new_pt) in &op_info.machines {
        if new_m==cur_machine{continue;}
        ds.node_machine[node]=new_m; ds.node_pt[node]=new_pt;
        let target_len=ds.machine_seq[new_m].len();
        let mut positions=find_insert_positions_hfs(ds,cur_starts.as_slice(),node,new_m,job_pred_node);
        let mut sorted_pos=target_len;
        for (k,&nd) in ds.machine_seq[new_m].iter().enumerate(){if cur_starts[nd]>=cur_start{sorted_pos=k;break;}}
        for p in [sorted_pos, sorted_pos.saturating_sub(1), target_len] {
            if p<=target_len&&!positions.contains(&p){positions.push(p);}
        }
        if target_len>2 {
            let mid=(sorted_pos+target_len)/2;
            if mid<=target_len&&!positions.contains(&mid){positions.push(mid);}
        }
        if positions.len()>7{positions.truncate(7);}

        for insert_pos in positions {
            let ins=insert_pos.min(ds.machine_seq[new_m].len());
            {
                let seq=&mut ds.machine_seq[new_m];
                seq.insert(ins,node);
            }
            if let Some((test_mk,_))=eval_disj(ds,buf){if test_mk<best_mk{best_mk=test_mk;best_m=new_m;best_pt=new_pt;best_ins_pos=ins;}}
            {
                let seq=&mut ds.machine_seq[new_m];
                seq.remove(ins);
            }
        }
    }

    if best_m!=cur_machine {
        let ins=best_ins_pos.min(ds.machine_seq[best_m].len());
        {
            let seq=&mut ds.machine_seq[best_m];
            seq.insert(ins,node);
        }
        ds.node_machine[node]=best_m; ds.node_pt[node]=best_pt;
        let Some((mk_now,_))=eval_disj(ds,buf) else{return Err(anyhow!("Stalled greedy reassign"))};
        *current_mk=mk_now; cur_starts.clone_from(&buf.start);
        Ok(true)
    } else {
        let seq=&mut ds.machine_seq[cur_machine];
        seq.push(node);
        seq[old_pos..].rotate_right(1);
        ds.node_machine[node]=cur_machine; ds.node_pt[node]=cur_pt;
        Ok(false)
    }
}

fn greedy_reassign_pass(pre: &Pre, challenge: &Challenge, base_sol: &Solution) -> Result<Option<(Solution, u32)>> {
    let mut ds=build_disj_from_solution(pre,challenge,base_sol)?; let mut buf=EvalBuf::new(ds.n); let n=ds.n;
    let Some((mut current_mk,_))=eval_disj(&ds,&mut buf) else{return Ok(None)};
    let initial_mk=current_mk;
    let mut job_pred_node=vec![NONE_USIZE;n];
    for j in 0..ds.num_jobs {
        let base=ds.job_offsets[j];
        let end=ds.job_offsets[j+1];
        for k in (base+1)..end { job_pred_node[k]=k-1; }
    }
    let mut cur_starts=buf.start.clone();
    let flex01=(pre.high_flex+pre.jobshopness).clamp(0.0,1.5);
    let focus_base=((pre.avg_op_min*(1.8+1.2*flex01)).max(1.0)) as u32;
    let max_rounds=if flex01>0.60{12}else{10};
    let mut rounds=0usize;
    let mut improved_any=false;

    while rounds<max_rounds {
        rounds+=1;
        let Some((mk_now,_))=eval_disj(&ds,&mut buf) else{return Ok(None)};
        current_mk=mk_now; cur_starts.clone_from(&buf.start);
        let tails=compute_tails_hfs(&ds,&buf);

        let build_order = |focused: bool, ds_ref: &DisjSchedule, cur_starts_ref: &[u32], current_mk_val: u32, tails_ref: &[u32]| -> Vec<usize> {
            let mut order: Vec<(usize,u32,u16,usize,usize)> = Vec::with_capacity(n);
            for node in 0..n {
                let job=ds_ref.node_job[node]; let op_idx=ds_ref.node_op[node]; let product=pre.job_products[job];
                let op_info=&pre.product_ops[product][op_idx];
                let flex=op_info.machines.len();
                if flex<=1{continue;}
                let cur_machine=ds_ref.node_machine[node];
                let path=cur_starts_ref[node].saturating_add(ds_ref.node_pt[node]).saturating_add(tails_ref[node]);
                let slack=current_mk_val.saturating_sub(path);
                if focused && flex01>0.55 {
                    let extra=(flex.saturating_sub(2).min(3) as u32).saturating_mul((pre.avg_op_min.max(1.0) as u32).max(1));
                    if slack>focus_base.saturating_add(extra){continue;}
                }
                let pop=(pre.machine_best_pop[cur_machine].clamp(0.0,1.0)*1000.0) as u16;
                let mach_len=ds_ref.machine_seq[cur_machine].len();
                order.push((node,slack,pop,flex,mach_len));
            }
            order.sort_by(|a,b|a.1.cmp(&b.1).then_with(||b.2.cmp(&a.2)).then_with(||b.3.cmp(&a.3)).then_with(||b.4.cmp(&a.4)).then_with(||a.0.cmp(&b.0)));
            order.into_iter().map(|x|x.0).collect()
        };

        let mut moved=false;

        if flex01>0.55 {
            let mut order=build_order(true, &ds, cur_starts.as_slice(), current_mk, &tails);
            if !order.is_empty() {
                let cap=((n/3).max(24)).min(order.len());
                order.truncate(cap);
                for node in order {
                    if greedy_reassign_try_node(pre,&mut ds,&mut buf,&job_pred_node,&mut cur_starts,&mut current_mk,node)? {
                        improved_any=true; moved=true; break;
                    }
                }
            }
        }

        if !moved {
            for node in build_order(false, &ds, cur_starts.as_slice(), current_mk, &tails) {
                if greedy_reassign_try_node(pre,&mut ds,&mut buf,&job_pred_node,&mut cur_starts,&mut current_mk,node)? {
                    improved_any=true; moved=true; break;
                }
            }
        }

        if !moved { break; }
    }

    if !improved_any||current_mk>=initial_mk{return Ok(None);}
    let Some((mk_now,_))=eval_disj(&ds,&mut buf) else{return Ok(None)};
    current_mk=mk_now;
    if current_mk>=initial_mk{return Ok(None);}
    let sol=disj_to_solution(pre,&ds,&buf.start)?; Ok(Some((sol,current_mk)))
}

fn compute_tails_hfs(ds: &DisjSchedule, buf: &EvalBuf) -> Vec<u32> {
    let mut order: Vec<usize> = (0..ds.n).collect();
    order.sort_unstable_by(|&a, &b| buf.start[b].cmp(&buf.start[a]));
    let mut tails = vec![0u32; ds.n];
    for &nd in &order {
        let mut after = 0u32;
        let js = ds.job_succ[nd];
        if js != NONE_USIZE {
            after = after.max(ds.node_pt[js].saturating_add(tails[js]));
        }
        let ms = buf.machine_succ[nd];
        if ms != NONE_USIZE {
            after = after.max(ds.node_pt[ms].saturating_add(tails[ms]));
        }
        tails[nd] = after;
    }
    tails
}

#[inline]
fn estimate_swap_hfs(
    u: usize,
    v: usize,
    heads: &[u32],
    tails: &[u32],
    pt: &[u32],
    job_pred: &[usize],
    job_succ: &[usize],
    machine_pred: &[usize],
    machine_succ: &[usize],
) -> u32 {
    let mp_u = machine_pred[u];
    let ms_v = machine_succ[v];
    let jp_v = job_pred[v];
    let jp_u = job_pred[u];
    let js_u = job_succ[u];
    let js_v = job_succ[v];
    let r_mp_u = if mp_u != NONE_USIZE { heads[mp_u].saturating_add(pt[mp_u]) } else { 0 };
    let r_jp_v = if jp_v != NONE_USIZE { heads[jp_v].saturating_add(pt[jp_v]) } else { 0 };
    let new_r_v = r_jp_v.max(r_mp_u);
    let r_jp_u = if jp_u != NONE_USIZE { heads[jp_u].saturating_add(pt[jp_u]) } else { 0 };
    let new_r_u = r_jp_u.max(new_r_v.saturating_add(pt[v]));
    let q_js_u = if js_u != NONE_USIZE { pt[js_u].saturating_add(tails[js_u]) } else { 0 };
    let q_ms_v = if ms_v != NONE_USIZE { pt[ms_v].saturating_add(tails[ms_v]) } else { 0 };
    let new_q_u = q_js_u.max(q_ms_v);
    let q_js_v = if js_v != NONE_USIZE { pt[js_v].saturating_add(tails[js_v]) } else { 0 };
    let new_q_v = q_js_v.max(pt[u].saturating_add(new_q_u));
    new_r_v.saturating_add(pt[v]).saturating_add(new_q_v).max(new_r_u.saturating_add(pt[u]).saturating_add(new_q_u))
}

#[inline]
fn estimate_reassign_hfs(
    ds: &DisjSchedule,
    heads: &[u32],
    tails: &[u32],
    node: usize,
    new_machine: usize,
    new_pt: u32,
    insert_pos: usize,
    job_pred: &[usize],
    machine_pred: &[usize],
    machine_succ: &[usize],
) -> u32 {
    let jp = job_pred[node];
    let js = ds.job_succ[node];
    let old_mp = machine_pred[node];
    let old_ms = machine_succ[node];
    let jp_end = if jp != NONE_USIZE { heads[jp].saturating_add(ds.node_pt[jp]) } else { 0 };
    let new_seq = &ds.machine_seq[new_machine];
    let new_mp_end = if insert_pos > 0 && !new_seq.is_empty() {
        let pred = new_seq[(insert_pos - 1).min(new_seq.len() - 1)];
        heads[pred].saturating_add(ds.node_pt[pred])
    } else {
        0
    };
    let new_end = jp_end.max(new_mp_end).saturating_add(new_pt);
    let js_tail = if js != NONE_USIZE { ds.node_pt[js].saturating_add(tails[js]) } else { 0 };
    let new_ms_tail = if insert_pos < new_seq.len() {
        let succ = new_seq[insert_pos];
        ds.node_pt[succ].saturating_add(tails[succ])
    } else {
        0
    };
    let node_path = new_end.saturating_add(js_tail.max(new_ms_tail));
    let old_reconnect = if old_mp != NONE_USIZE && old_ms != NONE_USIZE {
        let old_mp_end = heads[old_mp].saturating_add(ds.node_pt[old_mp]);
        old_mp_end.saturating_add(ds.node_pt[old_ms]).saturating_add(tails[old_ms])
    } else {
        0
    };
    node_path.max(old_reconnect)
}

fn find_insert_positions_hfs(ds: &DisjSchedule, starts: &[u32], node: usize, new_machine: usize, job_pred: &[usize]) -> Vec<usize> {
    let seq = &ds.machine_seq[new_machine];
    let len = seq.len();
    if len == 0 {
        return vec![0];
    }
    let jp = job_pred[node];
    let job_pred_end = if jp != NONE_USIZE { starts[jp].saturating_add(ds.node_pt[jp]) } else { 0 };
    let cur_start = starts[node];
    let mut pos_after_jp = len;
    for (i, &nd) in seq.iter().enumerate() {
        if starts[nd] > job_pred_end {
            pos_after_jp = i;
            break;
        }
    }
    let mut pos_by_cur = len;
    for (i, &nd) in seq.iter().enumerate() {
        if starts[nd] >= cur_start {
            pos_by_cur = i;
            break;
        }
    }
    let mut out: Vec<usize> = Vec::with_capacity(6);
    for p in [pos_after_jp, pos_after_jp.saturating_sub(1), pos_by_cur, pos_by_cur.saturating_sub(1), 0, len] {
        if p <= len && !out.contains(&p) {
            out.push(p);
        }
    }
    if out.is_empty() {
        out.push(len);
    }
    if out.len() > 6 {
        out.truncate(6);
    }
    out
}

enum MoveHfs {
    Swap { machine: usize, pos: usize },
    Reassign { node: usize, new_machine: usize, new_pt: u32, insert_pos: usize },
}

fn bottleneck_bridge_repair(
    pre: &Pre,
    challenge: &Challenge,
    base_sol: &Solution,
    _seed_mk: u32,
    max_machines: usize,
    max_rounds: usize,
) -> Result<Option<(Solution, u32)>> {
    let Ok(mut ds) = build_disj_from_solution(pre, challenge, base_sol) else {
        return Ok(None);
    };
    let mut buf = EvalBuf::new(ds.n);
    let Some((mut current_mk, _)) = eval_disj(&ds, &mut buf) else {
        return Ok(None);
    };
    let initial_mk = current_mk;
    let n = ds.n;
    if n == 0 || ds.num_machines == 0 {
        return Ok(None);
    }

    let mut job_pred_node = vec![NONE_USIZE; n];
    for j in 0..ds.num_jobs {
        let base = ds.job_offsets[j];
        let end = ds.job_offsets[j + 1];
        for k in (base + 1)..end {
            job_pred_node[k] = k - 1;
        }
    }

    let flex01 = (pre.high_flex + pre.jobshopness).clamp(0.0, 1.5);
    let near_slack = ((pre.avg_op_min * (0.80 + 0.75 * flex01)).round() as u32).max(1);
    let swap_limit_per_machine = if flex01 < 0.40 { 8usize } else { 5usize };
    let reassign_node_cap = if flex01 > 0.70 { 8usize } else if flex01 > 0.35 { 6usize } else { 3usize };
    let alt_machine_cap = if flex01 > 0.70 { 3usize } else { 2usize };
    let eval_cap = if flex01 > 0.60 { 84usize } else { 60usize };
    let rounds = max_rounds.max(1);
    let mut improved_any = false;
    let mut crit = vec![false; n];
    let mut near = vec![false; n];
    let mut selected_mask = vec![false; ds.num_machines];

    for _round in 0..rounds {
        let Some((mk_now, mk_node)) = eval_disj(&ds, &mut buf) else {
            return Ok(None);
        };
        current_mk = mk_now;
        let cur_starts = buf.start.clone();
        let tails = compute_tails_hfs(&ds, &buf);

        crit.fill(false);
        near.fill(false);
        let mut u = mk_node;
        while u != NONE_USIZE {
            crit[u] = true;
            u = buf.best_pred[u];
        }
        for node in 0..n {
            let path = cur_starts[node].saturating_add(ds.node_pt[node]).saturating_add(tails[node]);
            if current_mk.saturating_sub(path) <= near_slack {
                near[node] = true;
            }
        }

        let mut machine_scores: Vec<(u32, u32, u32, usize)> = Vec::new();
        for m in 0..ds.num_machines {
            let seq = &ds.machine_seq[m];
            if seq.is_empty() {
                continue;
            }
            let mut crit_cnt = 0u32;
            let mut near_cnt = 0u32;
            let mut adj_cnt = 0u32;
            for i in 0..seq.len() {
                let nd = seq[i];
                if crit[nd] { crit_cnt += 1; }
                if near[nd] { near_cnt += 1; }
                if i + 1 < seq.len() {
                    let nx = seq[i + 1];
                    if (crit[nd] || near[nd]) && (crit[nx] || near[nx]) {
                        adj_cnt += 1;
                    }
                }
            }
            let score = crit_cnt
                .saturating_mul(10)
                .saturating_add(near_cnt.saturating_mul(4))
                .saturating_add(adj_cnt.saturating_mul(5))
                .saturating_add(seq.len().min(6) as u32);
            if score > 0 {
                machine_scores.push((score, crit_cnt, near_cnt, m));
            }
        }
        machine_scores.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| b.1.cmp(&a.1)).then_with(|| a.3.cmp(&b.3)));
        if machine_scores.is_empty() {
            break;
        }

        selected_mask.fill(false);
        let top_score = machine_scores[0].0;
        let mut selected_machines: Vec<usize> = Vec::with_capacity(max_machines.min(ds.num_machines));
        for &(score, crit_cnt, near_cnt, m) in &machine_scores {
            if selected_machines.len() >= max_machines.max(1) {
                break;
            }
            if !selected_machines.is_empty() && score.saturating_mul(3).saturating_add(1) < top_score.saturating_mul(2) {
                break;
            }
            if crit_cnt == 0 && near_cnt < 2 {
                continue;
            }
            selected_mask[m] = true;
            selected_machines.push(m);
        }
        if selected_machines.is_empty() {
            break;
        }

        let mut best_move: Option<MoveHfs> = None;
        let mut best_move_mk = current_mk;
        let mut evals = 0usize;

        for &m in &selected_machines {
            if ds.machine_seq[m].len() <= 1 {
                continue;
            }
            let mut ranked_pos: Vec<(u32, usize)> = Vec::new();
            for pos in 0..(ds.machine_seq[m].len() - 1) {
                let a = ds.machine_seq[m][pos];
                let b = ds.machine_seq[m][pos + 1];
                let mut rank = 0u32;
                if crit[a] { rank += 6; }
                if crit[b] { rank += 6; }
                if near[a] { rank += 3; }
                if near[b] { rank += 3; }
                if rank == 0 {
                    continue;
                }
                if pos > 0 {
                    let p = ds.machine_seq[m][pos - 1];
                    if crit[p] || near[p] { rank += 1; }
                }
                if pos + 2 < ds.machine_seq[m].len() {
                    let s = ds.machine_seq[m][pos + 2];
                    if crit[s] || near[s] { rank += 1; }
                }
                ranked_pos.push((rank, pos));
            }
            ranked_pos.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
            let lim = ranked_pos.len().min(swap_limit_per_machine);
            for &(_, pos) in ranked_pos.iter().take(lim) {
                ds.machine_seq[m].swap(pos, pos + 1);
                evals += 1;
                if let Some((test_mk, _)) = eval_disj(&ds, &mut buf) {
                    if test_mk < best_move_mk {
                        best_move_mk = test_mk;
                        best_move = Some(MoveHfs::Swap { machine: m, pos });
                    }
                }
                ds.machine_seq[m].swap(pos, pos + 1);
                if evals >= eval_cap {
                    break;
                }
            }
            if evals >= eval_cap {
                break;
            }
        }

        if evals < eval_cap && reassign_node_cap > 0 {
            let mut ranked_nodes: Vec<(u32, usize, usize)> = Vec::new();
            for &m in &selected_machines {
                for &node in &ds.machine_seq[m] {
                    let job = ds.node_job[node];
                    let op_idx = ds.node_op[node];
                    let product = pre.job_products[job];
                    let op_info = &pre.product_ops[product][op_idx];
                    let flex = op_info.machines.len();
                    if flex <= 1 {
                        continue;
                    }
                    let path = cur_starts[node].saturating_add(ds.node_pt[node]).saturating_add(tails[node]);
                    let slack = current_mk.saturating_sub(path);
                    let pri = if crit[node] { 0 } else if near[node] { slack } else { slack.saturating_add(near_slack) };
                    ranked_nodes.push((pri, flex, node));
                }
            }
            ranked_nodes.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| b.1.cmp(&a.1)).then_with(|| a.2.cmp(&b.2)));

            let node_lim = ranked_nodes.len().min(reassign_node_cap);
            for &(_, _, node) in ranked_nodes.iter().take(node_lim) {
                let job = ds.node_job[node];
                let op_idx = ds.node_op[node];
                let product = pre.job_products[job];
                let op_info = &pre.product_ops[product][op_idx];
                let cur_machine = ds.node_machine[node];
                let cur_pt = ds.node_pt[node];
                let old_pos = match ds.machine_seq[cur_machine].iter().position(|&x| x == node) {
                    Some(p) => p,
                    None => continue,
                };

                {
                    let seq = &mut ds.machine_seq[cur_machine];
                    seq[old_pos..].rotate_left(1);
                    seq.pop();
                }

                let mut alt_machines: Vec<(usize, u32)> = op_info.machines.iter().copied().filter(|(mm, _)| *mm != cur_machine).collect();
                alt_machines.sort_by(|a, b| {
                    let ka = (if selected_mask[a.0] { 1u8 } else { 0u8 }, ds.machine_seq[a.0].len(), a.1, a.0);
                    let kb = (if selected_mask[b.0] { 1u8 } else { 0u8 }, ds.machine_seq[b.0].len(), b.1, b.0);
                    ka.cmp(&kb)
                });

                for &(new_m, new_pt) in alt_machines.iter().take(alt_machine_cap) {
                    ds.node_machine[node] = new_m;
                    ds.node_pt[node] = new_pt;
                    let target_len = ds.machine_seq[new_m].len();
                    let mut positions = find_insert_positions_hfs(&ds, &cur_starts, node, new_m, &job_pred_node);
                    for p in [0usize, target_len / 2, target_len] {
                        if p <= target_len && !positions.contains(&p) {
                            positions.push(p);
                        }
                    }
                    if positions.len() > 6 {
                        positions.truncate(6);
                    }

                    for insert_pos in positions {
                        let ins = insert_pos.min(ds.machine_seq[new_m].len());
                        {
                            let seq = &mut ds.machine_seq[new_m];
                            seq.insert(ins, node);
                        }
                        evals += 1;
                        if let Some((test_mk, _)) = eval_disj(&ds, &mut buf) {
                            if test_mk < best_move_mk {
                                best_move_mk = test_mk;
                                best_move = Some(MoveHfs::Reassign { node, new_machine: new_m, new_pt, insert_pos: ins });
                            }
                        }
                        {
                            let seq = &mut ds.machine_seq[new_m];
                            seq.remove(ins);
                        }
                        if evals >= eval_cap {
                            break;
                        }
                    }
                    if evals >= eval_cap {
                        break;
                    }
                }

                {
                    let seq = &mut ds.machine_seq[cur_machine];
                    seq.insert(old_pos.min(seq.len()), node);
                }
                ds.node_machine[node] = cur_machine;
                ds.node_pt[node] = cur_pt;

                if evals >= eval_cap {
                    break;
                }
            }
        }

        if best_move_mk >= current_mk {
            break;
        }

        match best_move {
            Some(MoveHfs::Swap { machine, pos }) => {
                ds.machine_seq[machine].swap(pos, pos + 1);
            }
            Some(MoveHfs::Reassign { node, new_machine, new_pt, insert_pos }) => {
                let old_machine = ds.node_machine[node];
                if let Some(op) = ds.machine_seq[old_machine].iter().position(|&x| x == node) {
                    ds.machine_seq[old_machine].remove(op);
                }
                let ins = insert_pos.min(ds.machine_seq[new_machine].len());
                ds.machine_seq[new_machine].insert(ins, node);
                ds.node_machine[node] = new_machine;
                ds.node_pt[node] = new_pt;
            }
            None => break,
        }

        let Some((mk_after, _)) = eval_disj(&ds, &mut buf) else {
            return Ok(None);
        };
        if mk_after >= current_mk {
            break;
        }
        current_mk = mk_after;
        improved_any = true;
    }

    if !improved_any || current_mk >= initial_mk {
        return Ok(None);
    }
    let Some((mk_now, _)) = eval_disj(&ds, &mut buf) else {
        return Ok(None);
    };
    current_mk = mk_now;
    if current_mk >= initial_mk {
        return Ok(None);
    }
    let sol = disj_to_solution(pre, &ds, &buf.start)?;
    Ok(Some((sol, current_mk)))
}

fn tabu_ils_hfs(
    pre: &Pre,
    challenge: &Challenge,
    seed_sol: &Solution,
    seed_mk: u32,
    max_iters: usize,
    tenure_base: usize,
    stagnation_limit: usize,
    reassign_every: usize,
) -> Result<(Solution, u32)> {
    let Ok(mut ds) = build_disj_from_solution(pre, challenge, seed_sol) else {
        return Ok((seed_sol.clone(), seed_mk));
    };
    let mut buf = EvalBuf::new(ds.n);
    let n = ds.n;
    let Some((initial_mk, _)) = eval_disj(&ds, &mut buf) else {
        return Ok((seed_sol.clone(), seed_mk));
    };
    let mut best_mk = initial_mk.min(seed_mk);
    let mut best_ds = ds.clone();
    let tenure = tenure_base.max(5);
    let tenure_delta = (tenure / 3).max(2);
    let kick_threshold = (stagnation_limit * 2 / 3).max(50);
    let aspiration_margin = ((pre.avg_op_min * (0.45 + 0.35 * pre.high_flex + 0.25 * pre.jobshopness)).round() as u32).max(1);
    let recent_window = (tenure * 2).clamp(8, 64);
    let freq_decay_every = (12usize + tenure).clamp(12, 28);
    let mut tabu_swap: std::collections::HashMap<(usize, usize), usize> = std::collections::HashMap::with_capacity(tenure * 8);
    let mut tabu_reassign: std::collections::HashMap<(usize, usize), usize> = std::collections::HashMap::with_capacity(tenure * 4);
    let mut job_pred_node = vec![NONE_USIZE; n];
    for j in 0..ds.num_jobs {
        let base = ds.job_offsets[j];
        let end = ds.job_offsets[j + 1];
        for k in (base + 1)..end {
            job_pred_node[k] = k - 1;
        }
    }
    let mut no_improve = 0usize;
    let mut kicks_left = 5usize;
    let mut pseed: u64 = (challenge.seed[0] as u64).wrapping_mul(0x9E3779B97F4A7C15)
        ^ (initial_mk as u64).wrapping_shl(16)
        ^ (n as u64).wrapping_mul(0x517CC1B727220A95);
    let mut machine_pred_node = vec![NONE_USIZE; n];
    let mut crit = vec![false; n];
    let mut job_touch_iter = vec![usize::MAX; ds.num_jobs];
    let mut machine_touch_iter = vec![usize::MAX; ds.num_machines];
    let mut job_freq = vec![0u16; ds.num_jobs];
    let mut machine_freq = vec![0u16; ds.num_machines];

    for iter in 0..max_iters {
        if no_improve >= stagnation_limit {
            if kicks_left == 0 {
                break;
            }
            ds = best_ds.clone();
            if eval_disj(&ds, &mut buf).is_none() {
                break;
            }
            no_improve = 0;
            kicks_left -= 1;
            tabu_swap.clear();
            tabu_reassign.clear();
            continue;
        }

        if no_improve > 0 && no_improve % kick_threshold == 0 && kicks_left > 0 {
            let Some((_, kick_mk_node)) = eval_disj(&ds, &mut buf) else { break };
            crit.fill(false);
            let mut u = kick_mk_node;
            while u != NONE_USIZE {
                crit[u] = true;
                u = buf.best_pred[u];
            }
            let mut kick_swaps: Vec<(usize, usize)> = Vec::new();
            for m in 0..ds.num_machines {
                if ds.machine_seq[m].len() <= 1 {
                    continue;
                }
                for i in 0..(ds.machine_seq[m].len() - 1) {
                    if crit[ds.machine_seq[m][i]] || crit[ds.machine_seq[m][i + 1]] {
                        kick_swaps.push((m, i));
                    }
                }
            }
            if !kick_swaps.is_empty() {
                for _ in 0..3 {
                    pseed ^= pseed.wrapping_shl(13);
                    pseed ^= pseed.wrapping_shr(7);
                    pseed ^= pseed.wrapping_shl(17);
                    let (m, pos) = kick_swaps[(pseed as usize) % kick_swaps.len()];
                    if pos + 1 < ds.machine_seq[m].len() {
                        ds.machine_seq[m].swap(pos, pos + 1);
                    }
                }
            }
            kicks_left -= 1;
            continue;
        }

        if iter > 0 && iter % freq_decay_every == 0 {
            for v in &mut job_freq { *v /= 2; }
            for v in &mut machine_freq { *v /= 2; }
        }

        let Some((cur_mk, mk_node)) = eval_disj(&ds, &mut buf) else { break };
        if cur_mk < best_mk {
            best_mk = cur_mk;
            best_ds = ds.clone();
            no_improve = 0;
        } else {
            no_improve += 1;
        }

        let tails = compute_tails_hfs(&ds, &buf);
        machine_pred_node.fill(NONE_USIZE);
        for seq in &ds.machine_seq {
            for i in 1..seq.len() {
                machine_pred_node[seq[i]] = seq[i - 1];
            }
        }

        crit.fill(false);
        let mut u = mk_node;
        while u != NONE_USIZE {
            crit[u] = true;
            u = buf.best_pred[u];
        }

        let age_since = |last: usize| -> usize {
            if last == usize::MAX { recent_window + 1 } else { iter.saturating_sub(last) }
        };

        let mut best_move: Option<MoveHfs> = None;
        let mut best_move_rank = u32::MAX;
        let mut fallback_move: Option<MoveHfs> = None;
        let mut fallback_rank = u32::MAX;

        for m in 0..ds.num_machines {
            if ds.machine_seq[m].len() <= 1 {
                continue;
            }
            let mut i = 0usize;
            while i < ds.machine_seq[m].len() {
                if !crit[ds.machine_seq[m][i]] {
                    i += 1;
                    continue;
                }
                let bstart = i;
                let mut bend = i;
                while bend + 1 < ds.machine_seq[m].len() {
                    let x = ds.machine_seq[m][bend];
                    let y = ds.machine_seq[m][bend + 1];
                    if !crit[y] {
                        break;
                    }
                    let end_x = buf.start[x].saturating_add(ds.node_pt[x]);
                    if buf.start[y] != end_x {
                        break;
                    }
                    bend += 1;
                }
                if bend > bstart {
                    let block_len = bend - bstart + 1;
                    let swap_positions = if block_len >= 3 { [bstart, bend - 1] } else { [bstart, NONE_USIZE] };
                    let num_swaps = if block_len >= 3 { 2 } else { 1 };
                    for si in 0..num_swaps {
                        let pos = swap_positions[si];
                        if pos == NONE_USIZE || pos + 1 >= ds.machine_seq[m].len() {
                            continue;
                        }
                        let node_u = ds.machine_seq[m][pos];
                        let node_v = ds.machine_seq[m][pos + 1];
                        let job_u = ds.node_job[node_u];
                        let job_v = ds.node_job[node_v];
                        let est_mk = estimate_swap_hfs(node_u, node_v, &buf.start, &tails, &ds.node_pt, &job_pred_node, &ds.job_succ, &machine_pred_node, &buf.machine_succ);
                        let key = (node_u.min(node_v), node_u.max(node_v));
                        let is_tabu = tabu_swap.get(&key).map_or(false, |&exp| iter < exp);
                        let age_ju = age_since(job_touch_iter[job_u]);
                        let age_jv = age_since(job_touch_iter[job_v]);
                        let age_m = age_since(machine_touch_iter[m]);
                        let recent_hits =
                            (if age_ju < recent_window { 1u32 } else { 0u32 }) +
                            (if age_jv < recent_window { 1u32 } else { 0u32 }) +
                            (if age_m < recent_window { 1u32 } else { 0u32 });
                        let freq_pen = ((machine_freq[m] as u32) + ((job_freq[job_u] as u32 + job_freq[job_v] as u32 + 1) / 2)).min(18);
                        let recent_pen = (2 * recent_hits).min(8);
                        let novelty = recent_hits <= 1 || age_ju + age_jv + age_m > recent_window.saturating_mul(2);
                        let aspiration = est_mk < best_mk || (est_mk <= best_mk.saturating_add(aspiration_margin) && novelty);
                        let tabu_pen = if is_tabu && !aspiration { aspiration_margin.min(12) } else { 0 };
                        let rank = est_mk.saturating_add(freq_pen).saturating_add(recent_pen).saturating_add(tabu_pen);
                        if (!is_tabu || aspiration) && rank < best_move_rank {
                            best_move_rank = rank;
                            best_move = Some(MoveHfs::Swap { machine: m, pos });
                        }
                        if rank < fallback_rank {
                            fallback_rank = rank;
                            fallback_move = Some(MoveHfs::Swap { machine: m, pos });
                        }
                    }
                }
                i = bend + 1;
            }
        }

        if iter % reassign_every == 0 {
            for node in 0..n {
                if !crit[node] {
                    continue;
                }
                let job = ds.node_job[node];
                let op_idx = ds.node_op[node];
                let product = pre.job_products[job];
                if op_idx >= pre.product_ops[product].len() {
                    continue;
                }
                let op_info = &pre.product_ops[product][op_idx];
                if op_info.machines.len() <= 1 {
                    continue;
                }
                let cur_machine = ds.node_machine[node];
                for &(new_m, new_pt) in &op_info.machines {
                    if new_m == cur_machine {
                        continue;
                    }
                    let key = (node, new_m);
                    let is_tabu = tabu_reassign.get(&key).map_or(false, |&exp| iter < exp);
                    let positions = find_insert_positions_hfs(&ds, &buf.start, node, new_m, &job_pred_node);
                    for insert_pos in positions {
                        let est_mk = estimate_reassign_hfs(&ds, &buf.start, &tails, node, new_m, new_pt, insert_pos, &job_pred_node, &machine_pred_node, &buf.machine_succ);
                        let age_job = age_since(job_touch_iter[job]);
                        let age_old = age_since(machine_touch_iter[cur_machine]);
                        let age_new = age_since(machine_touch_iter[new_m]);
                        let recent_hits =
                            (if age_job < recent_window { 1u32 } else { 0u32 }) +
                            (if age_old < recent_window { 1u32 } else { 0u32 }) +
                            (if age_new < recent_window { 1u32 } else { 0u32 });
                        let freq_pen = ((job_freq[job] as u32) + (machine_freq[new_m] as u32) + ((machine_freq[cur_machine] as u32 + 1) / 2)).min(20);
                        let recent_pen = (2 * recent_hits).min(8);
                        let novelty = recent_hits <= 1 || age_job + age_old + age_new > recent_window.saturating_mul(2);
                        let aspiration = est_mk < best_mk || (est_mk <= best_mk.saturating_add(aspiration_margin) && novelty);
                        let tabu_pen = if is_tabu && !aspiration { aspiration_margin.min(12) } else { 0 };
                        let rank = est_mk.saturating_add(freq_pen).saturating_add(recent_pen).saturating_add(tabu_pen);
                        if (!is_tabu || aspiration) && rank < best_move_rank {
                            best_move_rank = rank;
                            best_move = Some(MoveHfs::Reassign { node, new_machine: new_m, new_pt, insert_pos });
                        }
                        if rank < fallback_rank {
                            fallback_rank = rank;
                            fallback_move = Some(MoveHfs::Reassign { node, new_machine: new_m, new_pt, insert_pos });
                        }
                    }
                }
            }
        }

        match best_move.or(fallback_move) {
            Some(MoveHfs::Swap { machine, pos }) => {
                let node_a = ds.machine_seq[machine][pos];
                let node_b = ds.machine_seq[machine][pos + 1];
                let job_a = ds.node_job[node_a];
                let job_b = ds.node_job[node_b];
                let age_a = age_since(job_touch_iter[job_a]);
                let age_b = age_since(job_touch_iter[job_b]);
                let age_m = age_since(machine_touch_iter[machine]);
                ds.machine_seq[machine].swap(pos, pos + 1);
                pseed ^= pseed.wrapping_shl(13);
                pseed ^= pseed.wrapping_shr(7);
                pseed ^= pseed.wrapping_shl(17);
                let offset = (pseed % ((2 * tenure_delta + 1) as u64)) as usize;
                let base_tenure = (tenure + offset).saturating_sub(tenure_delta);
                let recent_bonus =
                    (if age_a < recent_window { 1usize } else { 0usize }) +
                    (if age_b < recent_window { 1usize } else { 0usize }) +
                    (if age_m < recent_window { 1usize } else { 0usize });
                let freq_bonus = ((job_freq[job_a] as usize + job_freq[job_b] as usize + machine_freq[machine] as usize) / 5).min(4);
                let this_tenure = base_tenure + 1 + recent_bonus + freq_bonus;
                tabu_swap.insert((node_a.min(node_b), node_a.max(node_b)), iter + this_tenure);
                job_touch_iter[job_a] = iter;
                job_touch_iter[job_b] = iter;
                machine_touch_iter[machine] = iter;
                job_freq[job_a] = job_freq[job_a].saturating_add(1);
                job_freq[job_b] = job_freq[job_b].saturating_add(1);
                machine_freq[machine] = machine_freq[machine].saturating_add(1);
            }
            Some(MoveHfs::Reassign { node, new_machine, new_pt, insert_pos }) => {
                let old_machine = ds.node_machine[node];
                let job = ds.node_job[node];
                let age_job = age_since(job_touch_iter[job]);
                let age_old = age_since(machine_touch_iter[old_machine]);
                let age_new = age_since(machine_touch_iter[new_machine]);
                if let Some(op) = ds.machine_seq[old_machine].iter().position(|&x| x == node) {
                    ds.machine_seq[old_machine].remove(op);
                }
                let ins = insert_pos.min(ds.machine_seq[new_machine].len());
                ds.machine_seq[new_machine].insert(ins, node);
                ds.node_machine[node] = new_machine;
                ds.node_pt[node] = new_pt;
                pseed ^= pseed.wrapping_shl(13);
                pseed ^= pseed.wrapping_shr(7);
                pseed ^= pseed.wrapping_shl(17);
                let offset = (pseed % ((2 * tenure_delta + 1) as u64)) as usize;
                let base_tenure = (tenure + offset).saturating_sub(tenure_delta / 2);
                let recent_bonus =
                    (if age_job < recent_window { 1usize } else { 0usize }) +
                    (if age_old < recent_window { 1usize } else { 0usize }) +
                    (if age_new < recent_window { 1usize } else { 0usize });
                let freq_bonus = ((job_freq[job] as usize + machine_freq[old_machine] as usize + machine_freq[new_machine] as usize) / 4).min(5);
                let this_tenure = base_tenure + 2 + recent_bonus + freq_bonus;
                tabu_reassign.insert((node, old_machine), iter + this_tenure);
                job_touch_iter[job] = iter;
                machine_touch_iter[old_machine] = iter;
                machine_touch_iter[new_machine] = iter;
                job_freq[job] = job_freq[job].saturating_add(1);
                machine_freq[old_machine] = machine_freq[old_machine].saturating_add(1);
                machine_freq[new_machine] = machine_freq[new_machine].saturating_add(1);
            }
            None => break,
        }
    }

    let Some((_, _)) = eval_disj(&best_ds, &mut buf) else {
        return Ok((seed_sol.clone(), seed_mk));
    };
    match disj_to_solution(pre, &best_ds, &buf.start) {
        Ok(s) => Ok((s, best_mk)),
        Err(_) => Ok((seed_sol.clone(), seed_mk)),
    }
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

    let post_band = target_margin.saturating_mul(3).max(((pre.avg_op_min * (1.0 + 0.80 * flex01)).round() as u32).max(2));
    let ls_base_indices = select_seed_portfolio_indices(best_solution.as_ref(), best_makespan, &top_solutions, 15, 15, post_band);
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
        let gr_base_indices = select_seed_portfolio_indices(Some(sol), best_makespan, &top_solutions, 15, 4, post_band);

        let mut best_improved: Option<(Solution, u32)> = None;

        if let Ok(Some((sol2, mk2))) = cached_gr(pre, challenge, sol, &mut cache) {
            if mk2 < best_makespan { best_improved = Some((sol2, mk2)); }
        }

        for &idx in &gr_base_indices {
            let base=&top_solutions[idx].0;
            if exact_solution_sig64(base)==sig_best_exact { continue; }
            if let Ok(Some((sol2, mk2))) = cached_gr(pre, challenge, base, &mut cache) {
                if mk2 < best_makespan && best_improved.as_ref().map_or(true, |x| mk2 < x.1) {
                    best_improved = Some((sol2, mk2));
                }
            }
        }

        if let Some((sol2, mk2)) = best_improved {
            best_makespan = mk2;
            best_solution = Some(sol2.clone());
            push_top_solutions(&mut top_solutions,&sol2,mk2,15);
            save_solution(&sol2)?;
        }
    }

    {
        let bridge_seed_cap = if flex01 > 0.55 { 4usize } else { 3usize };
        let bridge_rounds = if flex01 > 0.60 { 2usize } else { 1usize };
        let bridge_indices = select_seed_portfolio_indices(
            best_solution.as_ref(),
            best_makespan,
            &top_solutions,
            top_solutions.len().min(15),
            bridge_seed_cap,
            post_band,
        );
        let mut bridge_results: Vec<(Solution,u32)> = Vec::with_capacity(bridge_indices.len());
        let mut bridge_sigs: Vec<u64> = Vec::with_capacity(bridge_indices.len());
        for &idx in &bridge_indices {
            let base_sol=&top_solutions[idx].0;
            let base_mk=top_solutions[idx].1;
            let sig=exact_solution_sig64(base_sol);
            if bridge_sigs.contains(&sig) { continue; }
            bridge_sigs.push(sig);
            if let Some((sol2,mk2))=bottleneck_bridge_repair(pre,challenge,base_sol,base_mk,3,bridge_rounds)?{
                commit_best(save_solution,&mut best_makespan,&mut best_solution,&sol2,mk2)?;
                bridge_results.push((sol2,mk2));
            }
        }
        for (sol,mk) in bridge_results {
            push_top_solutions(&mut top_solutions,&sol,mk,15);
        }
    }

    {
        let tenure=((pre.total_ops as f64).sqrt() as usize).clamp(6,16);
        let tabu_seed_cap=effort.hybrid_flow_shop_tabu_seeds;
        let seed_indices = select_seed_portfolio_indices(best_solution.as_ref(), best_makespan, &top_solutions, top_solutions.len().min(15), tabu_seed_cap, post_band.saturating_mul(2));
        let mut seeds: Vec<(Solution,u32)> = Vec::with_capacity(tabu_seed_cap);
        if let Some(ref sol)=best_solution {
            seeds.push((sol.clone(),best_makespan));
        }
        for &idx in &seed_indices {
            if seeds.len()>=tabu_seed_cap{break;}
            let (sol,mk)=(&top_solutions[idx].0, top_solutions[idx].1);
            let sig=exact_solution_sig64(sol);
            if seeds.iter().any(|(s,_)|exact_solution_sig64(s)==sig){continue;}
            seeds.push((sol.clone(),mk));
        }
        for (sol,mk) in top_solutions.iter().take(8) {
            if seeds.len()>=tabu_seed_cap{break;}
            let sig=exact_solution_sig64(sol);
            if seeds.iter().any(|(s,_)|exact_solution_sig64(s)==sig){continue;}
            seeds.push((sol.clone(),*mk));
        }
        for (seed_sol,seed_mk) in seeds {
            let (sol2,mk2)=tabu_ils_hfs(pre,challenge,&seed_sol,seed_mk,effort.hybrid_flow_shop_tabu_iters,tenure,effort.hybrid_flow_shop_tabu_stagnation,effort.hybrid_flow_shop_tabu_reassign_every)?;
            commit_best(save_solution,&mut best_makespan,&mut best_solution,&sol2,mk2)?;
        }
    }

    if let Some(sol)=best_solution { save_solution(&sol)?; }

    Ok(())
}