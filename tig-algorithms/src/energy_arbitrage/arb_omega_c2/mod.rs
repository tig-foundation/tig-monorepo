pub mod refined;
pub mod classic;
pub mod linepulse_base;
pub mod donor_harimox;
use anyhow::Result;
use serde_json::{Map, Value};
use tig_challenges::energy_arbitrage::*;

// Fixed routing selected exclusively from Cycle 2 Bank S measurements.
pub fn solve_challenge(ch: &Challenge, save: &dyn Fn(&Solution)->Result<()>, _: &Option<Map<String,Value>>) -> Result<()> {
    let callback = |solution: &Solution| save(&strict_schedule(ch, solution));
    match ch.num_batteries {
        10 | 60 => refined::solve_challenge(ch, &callback, &None),
        20 => linepulse_base::solve_challenge(ch, &callback, &None),
        40 => classic::solve_challenge(ch, &callback, &None),
        _ => donor_harimox::solve_challenge(ch, &callback, &None),
    }
}

pub fn wasser_premium(ch: &Challenge, save: &dyn Fn(&Solution)->Result<()>, _: &Option<Map<String,Value>>) -> Result<()> {
    let mut hp=Map::new();
    hp.insert("cong_prem".into(), Value::from(1.0));
    donor_harimox::wasserwert::solve_challenge(ch, &|solution|save(&strict_schedule(ch,solution)), &Some(hp))
}

pub fn classic_arm(ch: &Challenge, save: &dyn Fn(&Solution)->Result<()>, _: &Option<Map<String,Value>>) -> Result<()> {
    classic::solve_challenge(ch, &|solution| save(&strict_schedule(ch,solution)), &None)
}

pub fn strict_donor(ch: &Challenge, save: &dyn Fn(&Solution)->Result<()>, hp: &Option<Map<String,Value>>) -> Result<()> {
    donor_harimox::solve_challenge(ch, &|solution|save(&strict_schedule(ch,solution)), hp)
}

// Feasible exogenous loading provides a convex anchor. Tiny donor tolerance
// overshoots are contracted toward it, with SOC bounds replayed afterward.
fn strict_schedule(ch: &Challenge, solution: &Solution) -> Solution {
    let mut schedule=Vec::with_capacity(ch.num_steps);
    let mut socs:Vec<f64>=ch.batteries.iter().map(|b|b.soc_initial_mwh).collect();
    for (t, original) in solution.schedule.iter().enumerate() {
        let mut actions=original.clone();
        for (i,b) in ch.batteries.iter().enumerate() {
            let lo=-b.power_charge_mw.min((b.soc_max_mwh-socs[i]).max(0.0)/(b.efficiency_charge*0.25));
            let hi=b.power_discharge_mw.min((socs[i]-b.soc_min_mwh).max(0.0)*b.efficiency_discharge/0.25);
            actions[i]=actions[i].clamp(lo,hi);
        }
        let mut base=ch.exogenous_injections[t].clone();
        base[ch.network.slack_bus]=-base.iter().enumerate().filter(|(i,_)|*i!=ch.network.slack_bus).map(|(_,x)|x).sum::<f64>();
        let base_flows=ch.network.compute_flows(&base);
        let mut total=base.clone();
        for (b,a) in ch.batteries.iter().zip(&actions) { total[b.node]+=a; }
        total[ch.network.slack_bus]=-total.iter().enumerate().filter(|(i,_)|*i!=ch.network.slack_bus).map(|(_,x)|x).sum::<f64>();
        let flows=ch.network.compute_flows(&total);
        let mut scale=1.0_f64;
        for ((f,zero),limit) in flows.iter().zip(&base_flows).zip(&ch.network.flow_limits) {
            if f.abs() > limit-1e-8 {
                let change=f-zero;
                if change != 0.0 {
                    let cap=if change>0.0 {(limit-1e-8-zero)/change} else {(-limit+1e-8-zero)/change};
                    scale=scale.min(cap.clamp(0.0,1.0));
                }
            }
        }
        for (i,b) in ch.batteries.iter().enumerate() { actions[i]*=scale; socs[i]=b.apply_action_to_soc(actions[i],socs[i]); }
        schedule.push(actions);
    }
    Solution { schedule }
}

pub fn help() {
    println!("arb_omega_c2 - fixed track-routed solver with strict flow schedule");
}
