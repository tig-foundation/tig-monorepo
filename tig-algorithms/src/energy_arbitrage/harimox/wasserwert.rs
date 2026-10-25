use anyhow::{anyhow, Result};
use serde_json::{Map, Value};
use tig_challenges::energy_arbitrage::constants as C;
use tig_challenges::energy_arbitrage::*;

const EPS: f64 = 1e-12;
const MAX_FLOW_ADJUST_ITERS: usize = 500;
const GLOBAL_SCALE_BSEARCH_ITERS: usize = 32;

pub fn help() {
    println!("Hyperparameter dp_cache (default 1) precomputes regular DP transitions; dp_cache=0 disables it.");
    println!(
        "wasserwert: Bellman-DP-Wasserwert je Batterie ueber DA-Preise, online \
         gegen RT-Preis; wertgewichtetes Netz-Weichklopfen. \
         hp (per-track defaults, see track_defaults): grid, samples, lp_iters, lp_step, \
         plan_rounds, nu_avg, refill, gate_k, cong_prem, ..."
    );
}

fn hp_usize(hp: &Option<Map<String, Value>>, key: &str, default: usize) -> usize {
    hp.as_ref()
        .and_then(|m| m.get(key))
        .and_then(|v| v.as_u64().or_else(|| v.as_f64().map(|f| f.round().max(0.0) as u64)))
        .map(|v| v as usize)
        .unwrap_or(default)
}

fn hp_f64(hp: &Option<Map<String, Value>>, key: &str, default: f64) -> f64 {
    hp.as_ref()
        .and_then(|m| m.get(key))
        .and_then(|v| v.as_f64())
        .unwrap_or(default)
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let td = track_defaults(challenge.num_batteries, challenge.num_steps);
    let grid = hp_usize(hyperparameters, "grid", 96).clamp(16, 512);
    let samples = hp_usize(hyperparameters, "samples", 24).clamp(4, 256);
    let lp_iters = hp_usize(hyperparameters, "lp_iters", td.lp_iters).min(1000);
    let lp_step = hp_f64(hyperparameters, "lp_step", td.lp_step).max(0.0);
    let plan_rounds = hp_usize(hyperparameters, "plan_rounds", td.plan_rounds).min(20);
    let cong_prem = hp_f64(hyperparameters, "cong_prem", 0.0).max(0.0);
    let opts = Opts {
        dp_exact: hp_usize(hyperparameters, "dp_exact", 0) != 0,
        dp_cache: hp_usize(hyperparameters, "dp_cache", 1) != 0,
        lp_repeat_cache: hp_usize(hyperparameters, "lp_repeat_cache", 0) != 0,
        lp_secant: hp_usize(hyperparameters, "lp_secant", 0) != 0,
        nu_avg: hp_usize(hyperparameters, "nu_avg", td.nu_avg) != 0,
        nu_warm: hp_usize(hyperparameters, "nu_warm", 0) != 0,
        lp_sched: hp_usize(hyperparameters, "lp_sched", 0),
        lp_norm: hp_usize(hyperparameters, "lp_norm", 0) != 0,
        plan_price_min: hp_f64(hyperparameters, "plan_price_min", C::LAMBDA_MIN),
        samples,
        refill: hp_usize(hyperparameters, "refill", td.refill) != 0,
        gate_k: hp_usize(hyperparameters, "gate_k", td.gate_k).min(64),
        nu_damp: hp_f64(hyperparameters, "nu_damp", 1.0).clamp(0.05, 1.0),
        sdp_k: hp_usize(hyperparameters, "sdp_k", td.sdp_k).min(256),
        cap_q: hp_f64(hyperparameters, "cap_q", 0.0).clamp(0.0, 1.0),
        lp_exact: hp_usize(hyperparameters, "lp_exact", td.lp_exact).min(3),
        lp_seg: hp_usize(hyperparameters, "lp_seg", td.lp_seg).min(32),
        nu_scale: hp_f64(hyperparameters, "nu_scale", td.nu_scale).max(0.0),
        grp: hp_usize(hyperparameters, "grp", 0).min(2),
        grp_theta: hp_f64(hyperparameters, "grp_theta", 0.15).clamp(0.01, 1.0),
        grp_relax: hp_f64(hyperparameters, "grp_relax", 1.0).clamp(0.1, 10.0),
        grp_max: hp_usize(hyperparameters, "grp_max", 8).max(2),
        mpc_n: hp_usize(hyperparameters, "mpc_n", 0),
        mpc_r: hp_usize(hyperparameters, "mpc_r", 1).clamp(1, 5),
        res_k: hp_usize(hyperparameters, "res_k", 0).min(64),
        res_w: hp_f64(hyperparameters, "res_w", 0.5).clamp(0.0, 1.0),
        qw: [0.0; 32],
        qn: 0,
    };
    let mut opts = opts;
    let sdp_q = hp_usize(hyperparameters, "sdp_q", td.sdp_q).min(5);
    let sdp_qj = hp_usize(hyperparameters, "sdp_qj", td.sdp_qj).min(3);

    let prem = expected_congestion_premium(challenge, cong_prem);
    let scen: Vec<Vec<Vec<f64>>> = if sdp_q > 0 {
        let (paths, w) = quadrature_paths(challenge, &prem, sdp_q, sdp_qj);
        opts.qn = w.len();
        opts.qw[..w.len()].copy_from_slice(&w);
        paths
    } else if opts.sdp_k > 0 {
        sample_rt_paths(challenge, opts.sdp_k, 0x51)
    } else {
        vec![expected_path(challenge, &prem)]
    };

    let (plans, nu_prof): (Vec<Plan>, Vec<Vec<f64>>) = if plan_rounds > 0 && lp_iters > 0 {
        build_plans_netaware(challenge, grid, plan_rounds, lp_iters, lp_step, &prem, &scen, &opts)
    } else {
        (
            (0..challenge.num_batteries).map(|b| build_plan(challenge, b, grid, &scen, &opts)).collect(),
            vec![vec![0.0f64; challenge.network.num_lines]; challenge.num_steps],
        )
    };
    let plans = if opts.grp > 0 && lp_iters > 0 {
        build_plans_grouped(challenge, grid, lp_iters, lp_step, &prem, &scen, &opts, plans, &nu_prof)
    } else {
        plans
    };
    let plans = if opts.res_k > 0 { blend_reserve(challenge, grid, &nu_prof, &opts, plans) } else { plans };

    let plans_cell = std::cell::RefCell::new(plans);
    let nu_carry = std::cell::RefCell::new(vec![0.0f64; challenge.network.num_lines]);
    let policy = |c: &Challenge, s: &State| -> Result<Vec<f64>> {
        if opts.mpc_n > 0 && lp_iters > 0 && s.time_step > 0 && s.time_step % opts.mpc_n == 0 && s.time_step + 2 < c.num_steps {
            let newp = {
                let cur = plans_cell.borrow();
                let nu_now = nu_carry.borrow().clone();
                replan_mpc(c, s, &cur, grid, lp_iters, lp_step, &prem, &scen, &opts, &nu_now)
            };
            if let Some(p) = newp {
                *plans_cell.borrow_mut() = p;
            }
        }
        let plans = plans_cell.borrow();
        let base = decide(c, s, &plans, samples, opts.refill)?;
        if lp_iters == 0 {
            Ok(base)
        } else {
            decide_lp(c, s, &plans, lp_iters, lp_step, &base, &opts, &nu_carry)
        }
    };
    let solution = challenge.grid_optimize(&policy)?;
    save_solution(&solution)?;
    Ok(())
}

struct TrackDefaults {
    lp_step: f64,
    lp_iters: usize,
    sdp_k: usize,
    sdp_q: usize,
    sdp_qj: usize,
    nu_avg: usize,
    refill: usize,
    gate_k: usize,
    plan_rounds: usize,
    lp_exact: usize,
    lp_seg: usize,
    nu_scale: f64,
}

fn track_defaults(num_batteries: usize, num_steps: usize) -> TrackDefaults {
    let _ = num_steps;
    match num_batteries {
        0..=14 => TrackDefaults { lp_step: 10.0, lp_iters: 100, sdp_k: 128, nu_avg: 1, refill: 1, gate_k: 20, plan_rounds: 8, lp_exact: 2, lp_seg: 4, nu_scale: 4.0, sdp_q: 0, sdp_qj: 0 },
        15..=29 => TrackDefaults { lp_step: 30.0, lp_iters: 40, sdp_k: 128, nu_avg: 1, refill: 1, gate_k: 20, plan_rounds: 3, lp_exact: 2, lp_seg: 2, nu_scale: 4.0, sdp_q: 0, sdp_qj: 0 },
        30..=49 => TrackDefaults { lp_step: 0.3, lp_iters: 200, sdp_k: 16, nu_avg: 1, refill: 0, gate_k: 0, plan_rounds: 3, lp_exact: 2, lp_seg: 4, nu_scale: 4.0, sdp_q: 3, sdp_qj: 2 },
        50..=79 => TrackDefaults { lp_step: 3.0, lp_iters: 200, sdp_k: 8, nu_avg: 1, refill: 0, gate_k: 0, plan_rounds: 2, lp_exact: 2, lp_seg: 2, nu_scale: 4.0, sdp_q: 5, sdp_qj: 2 },
        _ => TrackDefaults { lp_step: 3.0, lp_iters: 400, sdp_k: 0, nu_avg: 0, refill: 0, gate_k: 0, plan_rounds: 3, lp_exact: 2, lp_seg: 4, nu_scale: 4.0, sdp_q: 5, sdp_qj: 2 },
    }
}

#[derive(Clone, Copy)]
struct Opts {
    dp_exact: bool,
    dp_cache: bool,
    lp_repeat_cache: bool,
    lp_secant: bool,
    nu_avg: bool,
    nu_warm: bool,
    lp_sched: usize,
    lp_norm: bool,
    plan_price_min: f64,
    samples: usize,
    refill: bool,
    gate_k: usize,
    nu_damp: f64,
    sdp_k: usize,
    cap_q: f64,
    lp_exact: usize,
    lp_seg: usize,
    nu_scale: f64,
    grp: usize,
    grp_theta: f64,
    grp_relax: f64,
    grp_max: usize,
    mpc_n: usize,
    mpc_r: usize,
    res_k: usize,
    res_w: f64,
    qw: [f64; 32],
    qn: usize,
}

#[derive(Clone)]
struct Plan {
    node: usize,
    soc_min: f64,
    soc_max: f64,
    g: usize,
    dsoc: f64,
    v: Vec<f64>,
    h: usize,
}

#[inline]
fn step_profit(u: f64, price: f64, cap: f64) -> f64 {
    if u == 0.0 {
        return 0.0;
    }
    let dt = C::DELTA_T;
    let revenue = u * price * dt;
    let abs_u = u.abs();
    let tx = C::KAPPA_TX * abs_u * dt;
    let db = (abs_u * dt) / cap;
    let deg = C::KAPPA_DEG * db.powf(C::BETA_DEG);
    revenue - tx - deg
}

#[inline]
fn soc_to_action(delta_soc: f64, eta_c: f64, eta_d: f64) -> f64 {
    let dt = C::DELTA_T;
    if delta_soc > 0.0 {
        -(delta_soc / (eta_c * dt))
    } else if delta_soc < 0.0 {
        (-delta_soc) * eta_d / dt
    } else {
        0.0
    }
}

fn expected_congestion_premium(challenge: &Challenge, scale: f64) -> Vec<Vec<f64>> {
    let net = &challenge.network;
    let h = challenge.num_steps;
    let n = net.num_nodes;
    let mut prem = vec![vec![0.0f64; n]; h];
    if scale <= 0.0 {
        return prem;
    }
    let e_prem = scale * C::GAMMA_PRICE / (2.0 * std::f64::consts::PI).sqrt();
    for t in 1..h {
        let flows = net.compute_flows(&challenge.exogenous_injections[t - 1]);
        let mut keep = vec![1.0f64; n];
        for l in 0..net.num_lines {
            let r = flows[l].abs() / (net.congestion_threshold * net.flow_limits[l]).max(EPS);
            let p_l = r.powi(10).min(1.0);
            let (from, to) = net.lines[l];
            keep[from] *= 1.0 - p_l;
            keep[to] *= 1.0 - p_l;
        }
        for i in 0..n {
            prem[t][i] = e_prem * (1.0 - keep[i]);
        }
    }
    prem
}

fn plan_prices(challenge: &Challenge, i: usize, scen: &[Vec<Vec<f64>>], nu_prof: Option<&[Vec<f64>]>, price_min: f64, nu_scale: f64) -> Vec<Vec<f64>> {
    let node = challenge.batteries[i].node;
    let net = &challenge.network;
    let nl = net.num_lines;
    let col: Vec<f64> = if nu_prof.is_some() { (0..nl).map(|ln| net.ptdf[ln][node]).collect() } else { Vec::new() };
    let cong: Vec<f64> = (0..challenge.num_steps)
        .map(|t| match nu_prof {
            None => 0.0,
            Some(nu) => nu_scale * super::wide::dot_acc(-0.0, &col, &nu[t][..nl]),
        })
        .collect();
    scen.iter()
        .map(|path| (0..challenge.num_steps).map(|t| (path[t][node] - cong[t]).max(price_min)).collect())
        .collect()
}

fn build_plan(challenge: &Challenge, i: usize, g: usize, scen: &[Vec<Vec<f64>>], opts: &Opts) -> Plan {
    let b = &challenge.batteries[i];
    build_plan_priced(challenge, i, g, &plan_prices(challenge, i, scen, None, opts.plan_price_min, 1.0), (b.power_charge_mw, b.power_discharge_mw), opts)
}

#[inline]
fn interp_row(v: &[f64], base: usize, g: usize, soc_min: f64, dsoc: f64, soc: f64) -> f64 {
    let x = ((soc - soc_min) / dsoc).clamp(0.0, (g - 1) as f64);
    let g0 = x.floor() as usize;
    let g1 = (g0 + 1).min(g - 1);
    let frac = x - g0 as f64;
    v[base + g0] * (1.0 - frac) + v[base + g1] * frac
}

fn build_plan_priced(challenge: &Challenge, i: usize, g: usize, prices: &[Vec<f64>], pcap: (f64, f64), opts: &Opts) -> Plan {
    let b = &challenge.batteries[i];
    let p = BatParams {
        soc_min: b.soc_min_mwh,
        soc_max: b.soc_max_mwh,
        cap: b.capacity_mwh,
        deg_cap: b.capacity_mwh,
        eta_c: b.efficiency_charge,
        eta_d: b.efficiency_discharge,
        p_chg: pcap.0.min(b.power_charge_mw).max(0.0),
        p_dis: pcap.1.min(b.power_discharge_mw).max(0.0),
    };
    build_plan_core(b.node, &p, challenge.num_steps, g, prices, None, opts)
}

#[derive(Clone, Copy)]
struct BatParams {
    soc_min: f64,
    soc_max: f64,
    cap: f64,
    deg_cap: f64,
    eta_c: f64,
    eta_d: f64,
    p_chg: f64,
    p_dis: f64,
}

fn build_plan_core(node: usize, p: &BatParams, h: usize, g: usize, prices: &[Vec<f64>], caps_t: Option<&[(f64, f64)]>, opts: &Opts) -> Plan {
    if opts.dp_cache && h > 1 && caps_t.is_none() {
        build_plan_core_stationary(node, p, h, g, prices, opts)
    } else if opts.dp_cache && h > 1 {
        build_plan_core_cached(node, p, h, g, prices, caps_t, opts)
    } else {
        build_plan_core_uncached(node, p, h, g, prices, caps_t, opts)
    }
}

fn build_plan_core_uncached(node: usize, p: &BatParams, h: usize, g: usize, prices: &[Vec<f64>], caps_t: Option<&[(f64, f64)]>, opts: &Opts) -> Plan {
    let soc_min = p.soc_min;
    let soc_max = p.soc_max;
    let cap = p.deg_cap;
    let eta_c = p.eta_c;
    let eta_d = p.eta_d;
    let range = (soc_max - soc_min).max(EPS);
    let dsoc = range / (g as f64 - 1.0);
    let k = prices.len().max(1);
    let dt = C::DELTA_T;

    let mut v = vec![0.0f64; (h + 1) * g];
    let mut best = vec![0.0f64; k];
    let mut step_prices = vec![0.0f64; k];
    let mut cands: Vec<(f64, f64)> = Vec::with_capacity(64);
    for t in (0..h).rev() {
        for kk in 0..k { step_prices[kk] = prices[kk][t]; }
        let (p_chg, p_dis) = match caps_t {
            Some(c) => (p.p_chg.min(c[t].0).max(0.0), p.p_dis.min(c[t].1).max(0.0)),
            None => (p.p_chg, p.p_dis),
        };
        let max_charge_dsoc = eta_c * p_chg * dt;
        let max_dis_dsoc = p_dis * dt / eta_d;
        let up = ((max_charge_dsoc / dsoc).floor() as usize).max(1);
        let down = ((max_dis_dsoc / dsoc).floor() as usize).max(1);
        let next = (t + 1) * g;
        let cur = t * g;
        for gi in 0..g {
            let soc = soc_min + gi as f64 * dsoc;
            cands.clear();
            let hi = (gi + up).min(g - 1);
            for gt in (gi + 1)..=hi {
                let target = soc_min + gt as f64 * dsoc;
                let u = soc_to_action(target - soc, eta_c, eta_d);
                if -u > p_chg + EPS {
                    continue;
                }
                cands.push((u, v[next + gt]));
            }
            let lo = gi.saturating_sub(down);
            for gt in lo..gi {
                let target = soc_min + gt as f64 * dsoc;
                let u = soc_to_action(target - soc, eta_c, eta_d);
                if u > p_dis + EPS {
                    continue;
                }
                cands.push((u, v[next + gt]));
            }
            if opts.dp_exact {
                let tc = (soc + max_charge_dsoc).min(soc_max);
                if tc - soc > EPS {
                    cands.push((soc_to_action(tc - soc, eta_c, eta_d), interp_row(&v, next, g, soc_min, dsoc, tc)));
                }
                let td = (soc - max_dis_dsoc).max(soc_min);
                if soc - td > EPS {
                    cands.push((soc_to_action(td - soc, eta_c, eta_d), interp_row(&v, next, g, soc_min, dsoc, td)));
                }
            }
            for bk in best.iter_mut() {
                *bk = v[next + gi];
            }
            for &(u, cont) in &cands {
                let abs_u = u.abs();
                let cost = C::KAPPA_TX * abs_u * dt + C::KAPPA_DEG * ((abs_u * dt) / cap).powf(C::BETA_DEG);
                let rev = u * dt;
                relax(&mut best, &step_prices, rev, cost, cont);
            }
            v[cur + gi] = scen_mean(&best, opts);
        }
    }
    Plan { node, soc_min, soc_max, g, dsoc, v, h }
}

#[inline(always)]
fn relax(best: &mut [f64], prices: &[f64], rev: f64, cost: f64, cont: f64) {
    let n = best.len().min(prices.len());
    let m = n - n % 8;
    let (b8, br) = best[..n].split_at_mut(m);
    let (p8, pr) = prices[..n].split_at(m);
    for (bc, pc) in b8.chunks_exact_mut(8).zip(p8.chunks_exact(8)) {
        let bc: &mut [f64; 8] = bc.try_into().unwrap();
        let pc: &[f64; 8] = pc.try_into().unwrap();
        for j in 0..8 {
            let val = rev * pc[j] - cost + cont;
            bc[j] = if val > bc[j] { val } else { bc[j] };
        }
    }
    let m4 = br.len() - br.len() % 4;
    let (b4, br) = br.split_at_mut(m4);
    let (p4, pr) = pr.split_at(m4);
    for (bc, pc) in b4.chunks_exact_mut(4).zip(p4.chunks_exact(4)) {
        let bc: &mut [f64; 4] = bc.try_into().unwrap();
        let pc: &[f64; 4] = pc.try_into().unwrap();
        for j in 0..4 {
            let val = rev * pc[j] - cost + cont;
            bc[j] = if val > bc[j] { val } else { bc[j] };
        }
    }
    for (b, &pv) in br.iter_mut().zip(pr) {
        let val = rev * pv - cost + cont;
        *b = if val > *b { val } else { *b };
    }
}

fn build_plan_core_stationary(node: usize, p: &BatParams, h: usize, g: usize, prices: &[Vec<f64>], opts: &Opts) -> Plan {
    let soc_min = p.soc_min;
    let soc_max = p.soc_max;
    let dsoc = (soc_max - soc_min).max(EPS) / (g as f64 - 1.0);
    let k = prices.len().max(1);
    let dt = C::DELTA_T;
    let max_charge_dsoc = p.eta_c * p.p_chg * dt;
    let max_dis_dsoc = p.p_dis * dt / p.eta_d;
    let up = ((max_charge_dsoc / dsoc).floor() as usize).max(1);
    let down = ((max_dis_dsoc / dsoc).floor() as usize).max(1);
    let mut actions: Vec<(f64, f64, usize, Option<f64>)> = Vec::new();
    let mut starts = Vec::with_capacity(g + 1);
    let cost = |u: f64| {
        let abs_u = u.abs();
        C::KAPPA_TX * abs_u * dt + C::KAPPA_DEG * ((abs_u * dt) / p.deg_cap).powf(C::BETA_DEG)
    };
    for gi in 0..g {
        starts.push(actions.len());
        let soc = soc_min + gi as f64 * dsoc;
        for gt in (gi + 1)..=(gi + up).min(g - 1) {
            let target = soc_min + gt as f64 * dsoc;
            let u = soc_to_action(target - soc, p.eta_c, p.eta_d);
            if -u <= p.p_chg + EPS { actions.push((u * dt, cost(u), gt, None)); }
        }
        for gt in gi.saturating_sub(down)..gi {
            let target = soc_min + gt as f64 * dsoc;
            let u = soc_to_action(target - soc, p.eta_c, p.eta_d);
            if u <= p.p_dis + EPS { actions.push((u * dt, cost(u), gt, None)); }
        }
        if opts.dp_exact {
            for target in [(soc + max_charge_dsoc).min(soc_max), (soc - max_dis_dsoc).max(soc_min)] {
                if (target - soc).abs() > EPS {
                    let u = soc_to_action(target - soc, p.eta_c, p.eta_d);
                    let x = ((target - soc_min) / dsoc).clamp(0.0, (g - 1) as f64);
                    let g0 = x.floor() as usize;
                    actions.push((u * dt, cost(u), g0, Some(x - g0 as f64)));
                }
            }
        }
    }
    starts.push(actions.len());
    let mut v = vec![0.0; (h + 1) * g];
    let mut best = vec![0.0; k];
    let mut step_prices = vec![0.0; if k > 1 { k } else { 0 }];
    let mut acts: Vec<(f64, f64, f64)> = Vec::new();
    let mut cidx: Vec<(usize, usize, f64, bool)> = Vec::new();
    let mut sp: Vec<usize> = Vec::with_capacity(g + 1);
    if k > 1 {
        for gi in 0..g {
            sp.push(acts.len());
            for &(rev, cost, gt, frac) in &actions[starts[gi]..starts[gi + 1]] {
                acts.push((rev, cost, 0.0));
                cidx.push(match frac {
                    None => (gt, gt, 0.0, false),
                    Some(f) => (gt, (gt + 1).min(g - 1), f, true),
                });
            }
            while (acts.len() - sp[gi]) % 4 != 0 {
                acts.push((0.0, f64::INFINITY, 0.0));
                cidx.push((gi, gi, 0.0, false));
            }
        }
        sp.push(acts.len());
    }
    for t in (0..h).rev() {
        if k > 1 { for kk in 0..k { step_prices[kk] = prices[kk][t]; } }
        let next = (t + 1) * g;
        if k > 1 {
            let (vcur, vnext) = v.split_at_mut(next);
            let vn = &vnext[..g];
            let vrow = &mut vcur[t * g..t * g + g];
            let gl = g - 1;
            for gi in 0..g {
                best.fill(vn[gi]);
                let (s0, e0) = (sp[gi], sp[gi + 1]);
                for (a, ci) in acts[s0..e0].chunks_exact_mut(4).zip(cidx[s0..e0].chunks_exact(4)) {
                    for q in 0..4 {
                        let (gt, gt2, f, has) = ci[q];
                        let (x0, x1) = unsafe { (*vn.get_unchecked(gt.min(gl)), *vn.get_unchecked(gt2.min(gl))) };
                        let c1 = x0 * (1.0 - f) + x1 * f;
                        a[q].2 = if has { c1 } else { x0 };
                    }
                }
                super::wide::relax_multi(&mut best, &step_prices, &acts[s0..e0]);
                vrow[gi] = scen_mean(&best, opts);
            }
            continue;
        }
        for gi in 0..g {
            best.fill(v[next + gi]);
            for &(rev, cost, gt, frac) in &actions[starts[gi]..starts[gi + 1]] {
                let cont = match frac {
                    None => v[next + gt],
                    Some(f) => v[next + gt] * (1.0 - f) + v[next + (gt + 1).min(g - 1)] * f,
                };
                if k == 1 {
                    let val = rev * prices[0][t] - cost + cont;
                    if val > best[0] { best[0] = val; }
                } else {
                    relax(&mut best, &step_prices, rev, cost, cont);
                }
            }
            v[t * g + gi] = scen_mean(&best, opts);
        }
    }
    Plan { node, soc_min, soc_max, g, dsoc, v, h }
}

fn build_plan_core_cached(node: usize, p: &BatParams, h: usize, g: usize, prices: &[Vec<f64>], caps_t: Option<&[(f64, f64)]>, opts: &Opts) -> Plan {
    let soc_min = p.soc_min;
    let soc_max = p.soc_max;
    let cap = p.deg_cap;
    let eta_c = p.eta_c;
    let eta_d = p.eta_d;
    let range = (soc_max - soc_min).max(EPS);
    let dsoc = range / (g as f64 - 1.0);
    let k = prices.len().max(1);
    let dt = C::DELTA_T;

    let mut v = vec![0.0f64; (h + 1) * g];
    let mut best = vec![0.0f64; k];
    let mut step_prices = vec![0.0f64; k];
    let transition = |u: f64| {
        let abs_u = u.abs();
        C::KAPPA_TX * abs_u * dt + C::KAPPA_DEG * ((abs_u * dt) / cap).powf(C::BETA_DEG)
    };
    let (cache_charge, cache_discharge) = match caps_t {
        Some(c) => c.iter().take(h).fold((0.0f64, 0.0f64), |(a,b), &(x,y)|
            (a.max(p.p_chg.min(x).max(0.0)), b.max(p.p_dis.min(y).max(0.0)))),
        None => (p.p_chg, p.p_dis),
    };
    let cache_up = ((eta_c * cache_charge * dt / dsoc).floor() as usize).max(1);
    let cache_down = ((cache_discharge * dt / eta_d / dsoc).floor() as usize).max(1);
    let mut row_start = Vec::with_capacity(g + 1);
    let mut row_lo = Vec::with_capacity(g);
    let mut transitions: Vec<(f64, f64)> = Vec::new();
    for gi in 0..g {
        let lo = gi.saturating_sub(cache_down);
        let hi = (gi + cache_up).min(g - 1);
        row_start.push(transitions.len());
        row_lo.push(lo);
        let soc = soc_min + gi as f64 * dsoc;
        for gt in lo..=hi {
            let target = soc_min + gt as f64 * dsoc;
            let u = soc_to_action(target - soc, eta_c, eta_d);
            transitions.push((u, transition(u)));
        }
    }
    row_start.push(transitions.len());
    let mut cands: Vec<(f64, f64, f64)> = Vec::with_capacity(64);
    for t in (0..h).rev() {
        for kk in 0..k { step_prices[kk] = prices[kk][t]; }
        let (p_chg, p_dis) = match caps_t {
            Some(c) => (p.p_chg.min(c[t].0).max(0.0), p.p_dis.min(c[t].1).max(0.0)),
            None => (p.p_chg, p.p_dis),
        };
        let max_charge_dsoc = eta_c * p_chg * dt;
        let max_dis_dsoc = p_dis * dt / eta_d;
        let up = ((max_charge_dsoc / dsoc).floor() as usize).max(1);
        let down = ((max_dis_dsoc / dsoc).floor() as usize).max(1);
        let next = (t + 1) * g;
        let cur = t * g;
        for gi in 0..g {
            let soc = soc_min + gi as f64 * dsoc;
            cands.clear();
            let hi = (gi + up).min(g - 1);
            for gt in (gi + 1)..=hi {
                let (u, cost) = transitions[row_start[gi] + gt - row_lo[gi]];
                if -u > p_chg + EPS {
                    continue;
                }
                cands.push((u, v[next + gt], cost));
            }
            let lo = gi.saturating_sub(down);
            for gt in lo..gi {
                let (u, cost) = transitions[row_start[gi] + gt - row_lo[gi]];
                if u > p_dis + EPS {
                    continue;
                }
                cands.push((u, v[next + gt], cost));
            }
            if opts.dp_exact {
                let tc = (soc + max_charge_dsoc).min(soc_max);
                if tc - soc > EPS {
                    let u = soc_to_action(tc - soc, eta_c, eta_d);
                    cands.push((u, interp_row(&v, next, g, soc_min, dsoc, tc), transition(u)));
                }
                let td = (soc - max_dis_dsoc).max(soc_min);
                if soc - td > EPS {
                    let u = soc_to_action(td - soc, eta_c, eta_d);
                    cands.push((u, interp_row(&v, next, g, soc_min, dsoc, td), transition(u)));
                }
            }
            for bk in best.iter_mut() {
                *bk = v[next + gi];
            }
            for &(u, cont, cost) in &cands {
                let rev = u * dt;
                relax(&mut best, &step_prices, rev, cost, cont);
            }
            v[cur + gi] = scen_mean(&best, opts);
        }
    }
    Plan { node, soc_min, soc_max, g, dsoc, v, h }
}

#[inline]
fn interp_v(plan: &Plan, t: usize, soc: f64) -> f64 {
    if t >= plan.h {
        return 0.0;
    }
    let x = ((soc - plan.soc_min) / plan.dsoc).clamp(0.0, (plan.g - 1) as f64);
    let g0 = x.floor() as usize;
    let g1 = (g0 + 1).min(plan.g - 1);
    let frac = x - g0 as f64;
    let base = t * plan.g;
    plan.v[base + g0] * (1.0 - frac) + plan.v[base + g1] * frac
}

#[inline]
fn v_slope(plan: &Plan, t: usize, soc: f64) -> f64 {
    if t + 1 > plan.h {
        return 0.0;
    }
    let x = ((soc - plan.soc_min) / plan.dsoc).clamp(0.0, (plan.g - 1) as f64);
    let g0 = (x.floor() as usize).min(plan.g - 2);
    let base = (t + 1) * plan.g;
    ((plan.v[base + g0 + 1] - plan.v[base + g0]) / plan.dsoc).max(0.0)
}

#[inline]
fn action_value(challenge: &Challenge, state: &State, plans: &[Plan], action: &[f64]) -> f64 {
    let t = state.time_step;
    let mut tot = 0.0;
    for (i, b) in challenge.batteries.iter().enumerate() {
        let plan = &plans[i];
        let u = action[i];
        let dsoc_c = b.efficiency_charge * (-u).max(0.0) * C::DELTA_T;
        let dsoc_d = u.max(0.0) * C::DELTA_T / b.efficiency_discharge;
        let soc_next =
            (state.socs[i] + dsoc_c - dsoc_d).clamp(plan.soc_min, plan.soc_max);
        tot += step_profit(u, state.rt_prices[plan.node], b.capacity_mwh)
            + interp_v(plan, t + 1, soc_next);
    }
    tot
}

fn lp_dispatch(
    challenge: &Challenge,
    state: &State,
    plans: &[Plan],
    iters: usize,
    step0: f64,
    base_action: &[f64],
    opts: &Opts,
    nu0: Option<&[f64]>,
    planning: bool,
) -> (Vec<f64>, Vec<f64>) {
    let t = state.time_step;
    let m = challenge.num_batteries;
    let l = challenge.network.num_lines;
    let net = &challenge.network;
    let dt = C::DELTA_T;
    let tx = C::KAPPA_TX;

    let exact = match opts.lp_exact { 1 => !planning, 2 => true, 3 => planning, _ => false };
    if exact {
        if let Some((u, nu)) = lp_exact_dispatch(challenge, state, plans, opts.lp_seg) {
            let base_val = action_value(challenge, state, plans, base_action);
            let act = if is_flow_feasible(challenge, state, &u) {
                Some(u)
            } else {
                let av: Vec<f64> = (0..m).map(|i| u[i].abs()).collect();
                enforce_flow_feasibility(challenge, state, u, &av, opts.refill).ok()
            };
            return match act {
                Some(a) if action_value(challenge, state, plans, &a) > base_val => (a, nu),
                _ => (base_action.to_vec(), nu),
            };
        }
    }

    let exo_inj = {
        let mut inj = vec![0.0; net.num_nodes];
        for i in 0..net.num_nodes {
            if i != net.slack_bus { inj[i] = state.exogenous_injections[i]; }
        }
        let mut sum = 0.0;
        for i in 0..net.num_nodes {
            if i != net.slack_bus { sum += inj[i]; }
        }
        inj[net.slack_bus] = -sum;
        inj
    };
    let exo_flow: Vec<f64> = (0..l)
        .map(|ln| (0..net.num_nodes).map(|k| net.ptdf[ln][k] * exo_inj[k]).sum::<f64>())
        .collect();

    let mut a_dis = vec![0.0; m];
    let mut a_chg = vec![0.0; m];
    let mut node = vec![0usize; m];
    let mut lo = vec![0.0; m];
    let mut hi = vec![0.0; m];
    for (i, b) in challenge.batteries.iter().enumerate() {
        node[i] = b.node;
        let (min_b, max_b) = state.action_bounds[i];
        lo[i] = min_b;
        hi[i] = max_b;
        let rt = state.rt_prices[b.node];
        let (w_dis, w_chg) = if opts.lp_secant {
            let plan = &plans[i];
            let soc = state.socs[i];
            let d_dis = (max_b * dt / b.efficiency_discharge).min(soc - plan.soc_min);
            let d_chg = (b.efficiency_charge * (-min_b) * dt).min(plan.soc_max - soc);
            let v0 = interp_v(plan, t + 1, soc);
            let wd = if d_dis > EPS { (v0 - interp_v(plan, t + 1, soc - d_dis)) / d_dis } else { v_slope(plan, t, soc) };
            let wc = if d_chg > EPS { (interp_v(plan, t + 1, soc + d_chg) - v0) / d_chg } else { v_slope(plan, t, soc) };
            (wd.max(0.0), wc.max(0.0))
        } else {
            let w = v_slope(&plans[i], t, state.socs[i]);
            (w, w)
        };
        a_dis[i] = dt * (rt - tx - w_dis / b.efficiency_discharge);
        a_chg[i] = dt * (w_chg * b.efficiency_charge - rt - tx);
    }

    let denom: Vec<f64> = (0..l)
        .map(|ln| {
            if opts.lp_norm {
                (0..m).map(|i| net.ptdf[ln][node[i]].abs() * hi[i].max(-lo[i])).sum::<f64>()
            } else {
                net.flow_limits[ln].max(EPS)
            }
        })
        .collect();
    let mut nu = match nu0 {
        Some(n) if opts.nu_warm && n.len() == l => n.to_vec(),
        _ => vec![0.0f64; l],
    };
    let mut best_action = base_action.to_vec();
    let mut best_val = action_value(challenge, state, plans, base_action);
    let mut nu_sum = vec![0.0f64; l];
    let mut n_it = 0usize;

    let mut u = vec![0.0f64; m];
    let mut av = vec![0.0f64; m];
    let mut flow = exo_flow.clone();
    let ptdf_b: Vec<Vec<f64>> = (0..m).map(|i| (0..l).map(|ln| net.ptdf[ln][node[i]]).collect()).collect();
    for it in 0..iters {
        let mut changed = it == 0 || !opts.lp_repeat_cache;
        for i in 0..m {
            let sh: f64 = nu[..l].iter().zip(&ptdf_b[i][..l]).map(|(&n, &p)| n * p).sum();
            let ad = a_dis[i] - sh;
            let ac = a_chg[i] + sh;
            let (next_u, next_av) = if ad > 0.0 && ad >= ac {
                (hi[i], (a_dis[i] * hi[i]).max(0.0))
            } else if ac > 0.0 {
                (lo[i], (a_chg[i] * (-lo[i])).max(0.0))
            } else { (0.0, 0.0) };
            changed |= u[i].to_bits() != next_u.to_bits() || av[i].to_bits() != next_av.to_bits();
            u[i] = next_u;
            av[i] = next_av;
        }
        if changed {
        if let Ok(feas) = enforce_flow_feasibility(challenge, state, u.clone(), &av, opts.refill) {
            let val = action_value(challenge, state, plans, &feas);
            if val > best_val {
                best_val = val;
                best_action = feas;
            }
        }
        flow.clone_from(&exo_flow);
        for i in 0..m {
            if u[i] == 0.0 { continue; }
            let ui = u[i];
            for (f, &p) in flow[..l].iter_mut().zip(&ptdf_b[i][..l]) { *f += p * ui; }
        }
        }
        let step = match opts.lp_sched {
            1 => step0,
            2 => step0 / (1.0 + it as f64).sqrt(),
            _ => step0 / (1.0 + it as f64),
        };
        let mut maxviol = 0.0f64;
        for ln in 0..l {
            let lim = net.flow_limits[ln];
            let over = flow[ln] - flow[ln].clamp(-lim, lim);
            let rel = over.abs() / lim.max(EPS);
            if rel > maxviol { maxviol = rel; }
            if denom[ln] > EPS {
                nu[ln] += step * over / denom[ln];
            }
        }
        if opts.nu_avg { for ln in 0..l { nu_sum[ln] += nu[ln]; } }
        n_it += 1;
        if maxviol < 1e-6 { break; }
    }
    if opts.nu_avg && n_it > 0 {
        for ln in 0..l { nu[ln] = nu_sum[ln] / n_it as f64; }
    }
    (best_action, nu)
}

#[inline(always)]
fn row_sub(row: &mut [f64], prow: &[f64], f: f64) {
    let n = row.len().min(prow.len());
    let m = n - n % 8;
    let (r8, rr) = row[..n].split_at_mut(m);
    let (p8, pr) = prow[..n].split_at(m);
    for (rc, pc) in r8.chunks_exact_mut(8).zip(p8.chunks_exact(8)) {
        let rc: &mut [f64; 8] = rc.try_into().unwrap();
        let pc: &[f64; 8] = pc.try_into().unwrap();
        for j in 0..8 { rc[j] -= f * pc[j]; }
    }
    for (r, &pv) in rr.iter_mut().zip(pr) { *r -= f * pv; }
}

fn solve_bounded_lp(a: &[Vec<f64>], b: &[f64], sb: &[f64], c: &[f64], ub: &[f64]) -> Option<(Vec<f64>, Vec<f64>)> {
    const TOL: f64 = 1e-9;
    const PIV: f64 = 1e-10;
    let r = a.len();
    let n = c.len();
    if r == 0 {
        let x = (0..n).map(|j| if ub[j] > 0.0 && c[j] > TOL { ub[j] } else { 0.0 }).collect();
        return Some((x, Vec::new()));
    }
    let nc = n + r;
    let mut t = vec![0.0f64; r * nc];
    for i in 0..r {
        t[i * nc..i * nc + n].copy_from_slice(&a[i][..n]);
        t[i * nc + n + i] = 1.0;
    }
    let mut d: Vec<f64> = (0..nc).map(|j| if j < n { c[j] } else { 0.0 }).collect();
    let ubx: Vec<f64> = (0..nc).map(|j| if j < n { ub[j] } else { sb[j - n] }).collect();
    let mut xb: Vec<f64> = b.to_vec();
    for i in 0..r {
        if xb[i] < -TOL || xb[i] > sb[i] + TOL {
            return None;
        }
    }
    let mut basis: Vec<usize> = (0..r).map(|i| n + i).collect();
    let mut is_basic = vec![false; nc];
    for i in 0..r {
        is_basic[n + i] = true;
    }
    let mut at_upper = vec![false; nc];
    let max_iter = 60 * (nc + 1) + 200;
    let mut prow_buf: Vec<f64> = Vec::with_capacity(nc);
    let mut col_buf: Vec<f64> = Vec::with_capacity(r);
    let mut nz_i: Vec<usize> = Vec::new();
    let mut nz_f: Vec<f64> = Vec::new();
    let mut degenerate = 0usize;
    let mut bland = false;
    let mut converged = false;
    for _ in 0..max_iter {
        let mut enter: Option<usize> = None;
        if bland {
            for j in 0..nc {
                if is_basic[j] || ubx[j] <= 0.0 {
                    continue;
                }
                let gain = if at_upper[j] { -d[j] } else { d[j] };
                if gain > TOL {
                    enter = Some(j);
                    break;
                }
            }
        } else {
            let mut best = TOL;
            let mut ent = usize::MAX;
            let (ib, ub_, au, dd) = (&is_basic[..nc], &ubx[..nc], &at_upper[..nc], &d[..nc]);
            let m8 = nc - nc % 8;
            let mut j0 = 0;
            unsafe {
                while j0 < m8 {
                    for q in 0..8 {
                        let j = j0 + q;
                        let dj = *dd.get_unchecked(j);
                        let nd = -dj;
                        let gain = if *au.get_unchecked(j) { nd } else { dj };
                        let take = !(*ib.get_unchecked(j) | (*ub_.get_unchecked(j) <= 0.0)) & (gain > best);
                        ent = if take { j } else { ent };
                        best = if take { gain } else { best };
                    }
                    j0 += 8;
                }
            }
            for j in m8..nc {
                let nd = -dd[j];
                let gain = if au[j] { nd } else { dd[j] };
                let take = !(ib[j] | (ub_[j] <= 0.0)) & (gain > best);
                ent = if take { j } else { ent };
                best = if take { gain } else { best };
            }
            if ent != usize::MAX { enter = Some(ent); }
        }
        let Some(j) = enter else {
            converged = true;
            break;
        };
        let dir = if at_upper[j] { -1.0 } else { 1.0 };
        let mut theta = ubx[j];
        let mut leave: Option<(usize, bool)> = None;
        if !bland {
            let (xs, bs) = (&xb[..r], &basis[..r]);
            let mut li = usize::MAX;
            let mut lup = false;
            for i in 0..r {
                let (tij, ubb) = unsafe { (*t.get_unchecked(i * nc + j), *ubx.get_unchecked(bs[i])) };
                let alpha = tij * dir;
                let x = xs[i];
                let thp = (x / alpha).max(0.0);
                let thn = ((ubb - x) / (-alpha)).max(0.0);
                let pos = alpha > PIV;
                let neg = alpha < -PIV;
                let th = super::wide::sel(pos, thp, thn);
                let take = (pos | neg) & (th < theta);
                theta = super::wide::sel(take, th, theta);
                li = if take { i } else { li };
                lup = if take { neg } else { lup };
            }
            if li != usize::MAX {
                leave = Some((li, lup));
            }
        } else {
        for i in 0..r {
            let alpha = t[i * nc + j] * dir;
            if alpha > PIV {
                let th = (xb[i] / alpha).max(0.0);
                if th < theta || (bland && th == theta && leave.map_or(false, |(li, _)| basis[i] < basis[li])) {
                    theta = th;
                    leave = Some((i, false));
                }
            } else if alpha < -PIV {
                let th = ((ubx[basis[i]] - xb[i]) / (-alpha)).max(0.0);
                if th < theta || (bland && th == theta && leave.map_or(false, |(li, _)| basis[i] < basis[li])) {
                    theta = th;
                    leave = Some((i, true));
                }
            }
        }
        }
        if theta <= 1e-13 {
            degenerate += 1;
            if degenerate > 30 {
                bland = true;
            }
        } else {
            degenerate = 0;
        }
        for (i, x) in xb[..r].iter_mut().enumerate() {
            *x -= unsafe { *t.get_unchecked(i * nc + j) } * dir * theta;
        }
        match leave {
            None => {
                at_upper[j] = !at_upper[j];
            }
            Some((rr, up)) => {
                let p = t[rr * nc + j];
                let inv = 1.0 / p;
                super::wide::scale(&mut t[rr * nc..rr * nc + nc], inv);
                prow_buf.clear();
                prow_buf.extend_from_slice(&t[rr * nc..rr * nc + nc]);
                col_buf.clear();
                col_buf.resize(r, 0.0);
                for (i, cv) in col_buf.iter_mut().enumerate() {
                    let v = unsafe { *t.get_unchecked(i * nc + j) };
                    *cv = if i == rr { 0.0 } else { v };
                }
                super::wide::nonzeros(&col_buf, &mut nz_i, &mut nz_f);
                super::wide::rows_sub(&mut t, nc, &nz_i, &nz_f, &prow_buf);
                let f = d[j];
                if f != 0.0 { row_sub(&mut d, &prow_buf, f); }
                let old = basis[rr];
                xb[rr] = (if at_upper[j] { ubx[j] } else { 0.0 }) + dir * theta;
                is_basic[old] = false;
                at_upper[old] = up;
                is_basic[j] = true;
                at_upper[j] = false;
                basis[rr] = j;
            }
        }
    }
    if !converged {
        return None;
    }
    let mut x = vec![0.0f64; nc];
    for j in 0..nc {
        if !is_basic[j] && at_upper[j] {
            x[j] = ubx[j];
        }
    }
    for i in 0..r {
        x[basis[i]] = xb[i].clamp(0.0, ubx[basis[i]]);
    }
    let y: Vec<f64> = (0..r).map(|i| -d[n + i]).collect();
    x.truncate(n);
    Some((x, y))
}

fn lp_exact_dispatch(challenge: &Challenge, state: &State, plans: &[Plan], seg: usize) -> Option<(Vec<f64>, Vec<f64>)> {
    let t = state.time_step;
    let m = challenge.num_batteries;
    let net = &challenge.network;
    let l = net.num_lines;
    let dt = C::DELTA_T;
    let tx = C::KAPPA_TX;
    let s_per = seg.max(1);
    let mut col_b: Vec<usize> = Vec::with_capacity(2 * s_per * m);
    let mut col_sgn: Vec<f64> = Vec::with_capacity(2 * s_per * m);
    let mut ub: Vec<f64> = Vec::with_capacity(2 * s_per * m);
    let mut c: Vec<f64> = Vec::with_capacity(2 * s_per * m);
    for (i, b) in challenge.batteries.iter().enumerate() {
        let plan = &plans[i];
        let (lo, hi) = state.action_bounds[i];
        let rt = state.rt_prices[b.node];
        let soc = state.socs[i];
        let cap = b.capacity_mwh;
        if seg == 0 {
            let w = v_slope(plan, t, soc);
            if hi > EPS {
                col_b.push(i); col_sgn.push(1.0); ub.push(hi);
                c.push(dt * (rt - tx - w / b.efficiency_discharge));
            }
            if -lo > EPS {
                col_b.push(i); col_sgn.push(-1.0); ub.push(-lo);
                c.push(dt * (w * b.efficiency_charge - rt - tx));
            }
            continue;
        }
        let deg = |u: f64| C::KAPPA_DEG * ((u * dt) / cap).powf(C::BETA_DEG);
        for (sgn, bound) in [(1.0f64, hi), (-1.0f64, -lo)] {
            if bound <= EPS {
                continue;
            }
            let du = bound / s_per as f64;
            let mut slopes: Vec<f64> = Vec::with_capacity(s_per);
            let mut v_prev = interp_v(plan, t + 1, soc);
            for s in 0..s_per {
                let ua = s as f64 * du;
                let ubound = (s + 1) as f64 * du;
                let soc_b = if sgn > 0.0 { soc - ubound * dt / b.efficiency_discharge } else { soc + ubound * b.efficiency_charge * dt };
                let v_b = interp_v(plan, t + 1, soc_b.clamp(plan.soc_min, plan.soc_max));
                let rev = sgn * rt * dt * du - tx * dt * du - (deg(ubound) - deg(ua));
                slopes.push((rev + v_b - v_prev) / du);
                v_prev = v_b;
            }
            slopes.sort_unstable_by(|a, b2| b2.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            for sl in slopes {
                col_b.push(i); col_sgn.push(sgn); ub.push(du); c.push(sl);
            }
        }
    }
    let n = c.len();
    let exo_flow: Vec<f64> = {
        let mut inj = vec![0.0; net.num_nodes];
        let mut sum = 0.0;
        for k in 0..net.num_nodes {
            if k != net.slack_bus { inj[k] = state.exogenous_injections[k]; sum += inj[k]; }
        }
        inj[net.slack_bus] = -sum;
        (0..l).map(|ln| super::wide::dot_acc(-0.0, &net.ptdf[ln][..net.num_nodes], &inj[..net.num_nodes])).collect()
    };
    const SHRINK: f64 = 1e-9;
    let mut rows: Vec<usize> = Vec::new();
    let mut in_rows = vec![false; l];
    let mut u = vec![0.0f64; m];
    let mut nu = vec![0.0f64; l];
    let mut a: Vec<Vec<f64>> = Vec::new();
    let mut b: Vec<f64> = Vec::new();
    let mut sb: Vec<f64> = Vec::new();
    let mut pb = vec![0.0f64; l * m];
    let nodes: Vec<usize> = challenge.batteries.iter().map(|bt| bt.node).collect();
    for ln in 0..l {
        super::wide::gather::<false>(&mut pb[ln * m..ln * m + m], &net.ptdf[ln], &nodes, &[]);
    }
    for _round in 0..(l + 1) {
        let (x, y) = solve_bounded_lp(&a, &b, &sb, &c, &ub)?;
        for i in 0..m { u[i] = 0.0; }
        super::wide::scatter_add(&mut u[..m], &col_b[..n], &col_sgn[..n], &x[..n]);
        for i in 0..m {
            let (lo, hi) = state.action_bounds[i];
            u[i] = u[i].clamp(lo, hi);
        }
        for ln in 0..l { nu[ln] = 0.0; }
        for (k, &ln) in rows.iter().enumerate() { nu[ln] = y[k]; }
        let mut added = false;
        for ln in 0..l {
            if in_rows[ln] { continue; }
            let prow = &pb[ln * m..ln * m + m];
            let f = super::wide::dot_acc(exo_flow[ln], prow, &u[..m]);
            let lim = net.flow_limits[ln];
            if f.abs() > lim * (1.0 - 0.5 * SHRINK) {
                rows.push(ln);
                let mut arow = vec![0.0f64; n];
                super::wide::gather::<true>(&mut arow, prow, &col_b[..n], &col_sgn[..n]);
                a.push(arow);
                b.push(net.flow_limits[ln] * (1.0 - SHRINK) - exo_flow[ln]);
                sb.push(2.0 * net.flow_limits[ln] * (1.0 - SHRINK));
                in_rows[ln] = true;
                added = true;
            }
        }
        if !added {
            return Some((u, nu));
        }
    }
    None
}

fn decide_lp(
    challenge: &Challenge,
    state: &State,
    plans: &[Plan],
    iters: usize,
    step0: f64,
    base_action: &[f64],
    opts: &Opts,
    nu_carry: &std::cell::RefCell<Vec<f64>>,
) -> Result<Vec<f64>> {
    let (action, nu) = {
        let prev = nu_carry.borrow();
        lp_dispatch(challenge, state, plans, iters, step0, base_action, opts, Some(&prev), false)
    };
    *nu_carry.borrow_mut() = nu;
    Ok(action)
}

fn calc_bounds(b: &Battery, soc: f64) -> (f64, f64) {
    let dt = C::DELTA_T;
    let headroom = (b.soc_max_mwh - soc).max(0.0);
    let available = (soc - b.soc_min_mwh).max(0.0);
    let max_charge_soc = if b.efficiency_charge > 0.0 { headroom / (b.efficiency_charge * dt) } else { 0.0 };
    let max_dis_soc = if b.efficiency_discharge > 0.0 { available * b.efficiency_discharge / dt } else { 0.0 };
    let max_charge = max_charge_soc.min(b.power_charge_mw).max(0.0);
    let max_dis = max_dis_soc.min(b.power_discharge_mw).max(0.0);
    (-max_charge, max_dis)
}

fn apply_soc(b: &Battery, u: f64, soc: f64) -> f64 {
    let dt = C::DELTA_T;
    let c = (-u).max(0.0);
    let d = u.max(0.0);
    (soc + b.efficiency_charge * c * dt - d * dt / b.efficiency_discharge)
        .clamp(b.soc_min_mwh, b.soc_max_mwh)
}

fn build_plan_eff(challenge: &Challenge, i: usize, g: usize, scen: &[Vec<Vec<f64>>], nu_prof: &[Vec<f64>], pcap: (f64, f64), opts: &Opts) -> Plan {
    build_plan_priced(challenge, i, g, &plan_prices(challenge, i, scen, Some(nu_prof), opts.plan_price_min, opts.nu_scale), pcap, opts)
}

fn expected_path(challenge: &Challenge, prem: &[Vec<f64>]) -> Vec<Vec<f64>> {
    (0..challenge.num_steps)
        .map(|t| (0..challenge.network.num_nodes).map(|k| challenge.market.day_ahead_prices[t][k] + prem[t][k]).collect())
        .collect()
}

#[inline]
fn scen_mean(best: &[f64], opts: &Opts) -> f64 {
    if opts.qn > 0 && opts.qn == best.len() {
        super::wide::dot_acc(-0.0, best, &opts.qw[..opts.qn])
    } else {
        super::wide::sum_acc(-0.0, best) / best.len().max(1) as f64
    }
}

fn quadrature_paths(challenge: &Challenge, prem: &[Vec<f64>], nq: usize, nj: usize) -> (Vec<Vec<Vec<f64>>>, Vec<f64>) {
    let mp = &challenge.market.params;
    let (sigma, p, alpha) = (mp.volatility, mp.jump_probability, mp.tail_index);
    let (xs, xw): (Vec<f64>, Vec<f64>) = match nq {
        1 => (vec![0.0], vec![1.0]),
        2 | 3 => (vec![-3f64.sqrt(), 0.0, 3f64.sqrt()], vec![1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0]),
        _ => (
            vec![-2.856970013872806, -1.355626179974266, 0.0, 1.355626179974266, 2.856970013872806],
            vec![0.011257411327721, 0.222075922005613, 0.533333333333333, 0.222075922005613, 0.011257411327721],
        ),
    };
    let e = 1.0 - 1.0 / alpha.max(1.0001);
    let slice = |a: f64, b: f64| -> (f64, f64) { (((1.0 - a).powf(e) - (1.0 - b).powf(e)) / e / (b - a), b - a) };
    let jumps: Vec<(f64, f64)> = match nj {
        0 => vec![],
        1 => vec![slice(0.0, 1.0)],
        2 => vec![slice(0.0, 0.9), slice(0.9, 1.0)],
        _ => vec![slice(0.0, 0.9), slice(0.9, 0.99), slice(0.99, 1.0)],
    };
    let pj = if jumps.is_empty() { 0.0 } else { p };
    let (h, n) = (challenge.num_steps, challenge.network.num_nodes);
    let da = &challenge.market.day_ahead_prices;
    let mut paths = Vec::new();
    let mut w = Vec::new();
    for (&x, &wx) in xs.iter().zip(&xw) {
        let mut add = |jx: f64, wj: f64| {
            paths.push(
                (0..h)
                    .map(|t| (0..n).map(|k| (da[t][k] * (1.0 + sigma * x) + prem[t][k] + da[t][k] * jx).clamp(C::LAMBDA_MIN, C::LAMBDA_MAX)).collect())
                    .collect(),
            );
            w.push(wx * wj);
        };
        add(0.0, 1.0 - pj);
        for &(mx, mass) in &jumps {
            add(mx, pj * mass);
        }
    }
    (paths, w)
}

fn sample_rt_paths(challenge: &Challenge, k: usize, salt: u8) -> Vec<Vec<Vec<f64>>> {
    use rand::{rngs::SmallRng, SeedableRng};
    let h = challenge.num_steps;
    let n = challenge.network.num_nodes;
    (0..k)
        .map(|s| {
            let mut seed = challenge.seed;
            for (j, byte) in seed.iter_mut().enumerate() {
                *byte ^= (s as u8).wrapping_mul(0x9D).wrapping_add(j as u8).wrapping_mul(0x5B) ^ salt;
            }
            let mut rng = SmallRng::from_seed(seed);
            let mut path = Vec::with_capacity(h);
            path.push(challenge.market.generate_rt_prices(&mut rng, 0, &vec![false; n]));
            for t in 1..h {
                let cong = challenge.network.generate_congestion_indicators(&mut rng, &challenge.exogenous_injections[t - 1]);
                path.push(challenge.market.generate_rt_prices(&mut rng, t, &cong));
            }
            path
        })
        .collect()
}

fn forward_plan(challenge: &Challenge, plans: &[Plan], iters: usize, step0: f64, da: &[Vec<f64>], opts: &Opts) -> (Vec<Vec<f64>>, f64, Vec<Vec<f64>>) {
    let socs0: Vec<f64> = challenge.batteries.iter().map(|b| b.soc_initial_mwh).collect();
    forward_plan_from(challenge, plans, iters, step0, da, opts, 0, &socs0, None)
}

fn forward_plan_from(challenge: &Challenge, plans: &[Plan], iters: usize, step0: f64, da: &[Vec<f64>], opts: &Opts, t0: usize, socs0: &[f64], nu0: Option<&[f64]>) -> (Vec<Vec<f64>>, f64, Vec<Vec<f64>>) {
    let h = challenge.num_steps;
    let l = challenge.network.num_lines;
    let m = challenge.num_batteries;
    let mut socs: Vec<f64> = socs0.to_vec();
    let mut nu_prof = vec![vec![0.0f64; l]; h];
    let mut acts: Vec<Vec<f64>> = Vec::with_capacity(h);
    let mut profit = 0.0f64;
    for _ in 0..t0.min(h) {
        acts.push(vec![0.0; m]);
    }
    for t in t0..h {
        let action_bounds: Vec<(f64, f64)> =
            (0..m).map(|i| calc_bounds(&challenge.batteries[i], socs[i])).collect();
        let state = State {
            time_step: t,
            socs: socs.clone(),
            rt_prices: da[t].clone(),
            exogenous_injections: challenge.exogenous_injections[t].clone(),
            action_bounds,
            total_profit: 0.0,
        };
        let base = decide(challenge, &state, plans, opts.samples, opts.refill).unwrap_or_else(|_| vec![0.0; m]);
        let nu_prev = if t > t0 { Some(&nu_prof[t - 1][..]) } else { nu0 };
        let (action, nu) = lp_dispatch(challenge, &state, plans, iters, step0, &base, opts, nu_prev, true);
        nu_prof[t] = nu;
        for (i, b) in challenge.batteries.iter().enumerate() {
            profit += step_profit(action[i], da[t][b.node], b.capacity_mwh);
            socs[i] = apply_soc(b, action[i], socs[i]);
        }
        acts.push(action);
    }
    (nu_prof, profit, acts)
}

fn power_caps(challenge: &Challenge, acts: &[Vec<f64>], q: f64) -> Vec<(f64, f64)> {
    (0..challenge.num_batteries)
        .map(|i| {
            let b = &challenge.batteries[i];
            let quant = |mut xs: Vec<f64>, nominal: f64| -> f64 {
                if xs.len() < 4 {
                    return nominal;
                }
                xs.sort_unstable_by(|a, c| a.partial_cmp(c).unwrap_or(std::cmp::Ordering::Equal));
                let idx = ((xs.len() as f64 - 1.0) * q).round() as usize;
                xs[idx.min(xs.len() - 1)].min(nominal)
            };
            let dis: Vec<f64> = acts.iter().map(|a| a[i]).filter(|&u| u > EPS).collect();
            let chg: Vec<f64> = acts.iter().map(|a| -a[i]).filter(|&u| u > EPS).collect();
            (quant(chg, b.power_charge_mw), quant(dis, b.power_discharge_mw))
        })
        .collect()
}

fn build_plans_netaware(challenge: &Challenge, grid: usize, rounds: usize, iters: usize, step0: f64, prem: &[Vec<f64>], scen: &[Vec<Vec<f64>>], opts: &Opts) -> (Vec<Plan>, Vec<Vec<f64>>) {
    let m = challenge.num_batteries;
    let exp_path = expected_path(challenge, prem);
    let gate_scen = sample_rt_paths(challenge, opts.gate_k, 0);
    let gate = |plans: &[Plan], p_exp: f64| -> f64 {
        if gate_scen.is_empty() {
            p_exp
        } else {
            gate_scen.iter().map(|path| forward_plan(challenge, plans, iters, step0, path, opts).1).sum::<f64>() / gate_scen.len() as f64
        }
    };
    let nominal: Vec<(f64, f64)> = challenge.batteries.iter().map(|b| (b.power_charge_mw, b.power_discharge_mw)).collect();
    let mut cur: Vec<Plan> = (0..m).map(|b| build_plan(challenge, b, grid, scen, opts)).collect();
    let (mut nu_prof, p0, mut acts) = forward_plan(challenge, &cur, iters, step0, &exp_path, opts);
    let mut best_plans = cur.clone();
    let mut best_p = gate(&cur, p0);
    let mut best_nu = nu_prof.clone();
    for _ in 0..rounds {
        let pcap = if opts.cap_q > 0.0 { power_caps(challenge, &acts, opts.cap_q) } else { nominal.clone() };
        cur = (0..m).map(|b| build_plan_eff(challenge, b, grid, scen, &nu_prof, pcap[b], opts)).collect();
        let (nu_k, pk, acts_k) = forward_plan(challenge, &cur, iters, step0, &exp_path, opts);
        acts = acts_k;
        let a = opts.nu_damp;
        for t in 0..nu_prof.len() {
            for ln in 0..nu_prof[t].len() {
                nu_prof[t][ln] = (1.0 - a) * nu_prof[t][ln] + a * nu_k[t][ln];
            }
        }
        let gk = gate(&cur, pk);
        if gk > best_p {
            best_p = gk;
            best_plans = cur.clone();
            best_nu = nu_prof.clone();
        }
    }
    (best_plans, best_nu)
}

fn decide(
    challenge: &Challenge,
    state: &State,
    plans: &[Plan],
    n_samples: usize,
    refill: bool,
) -> Result<Vec<f64>> {
    let t = state.time_step;
    let mut action = vec![0.0; challenge.num_batteries];
    let mut av = vec![0.0; challenge.num_batteries];
    for (i, b) in challenge.batteries.iter().enumerate() {
        let plan = &plans[i];
        let node = plan.node;
        let price = state.rt_prices[node];
        let soc = state.socs[i];
        let (min_b, max_b) = state.action_bounds[i];
        let cap = b.capacity_mwh;
        let eta_c = b.efficiency_charge;
        let eta_d = b.efficiency_discharge;

        let hold_val = interp_v(plan, t + 1, soc);
        let mut best_u = 0.0;
        let mut best_val = hold_val;
        let t1 = t + 1;
        if (plan.g >= 1) & (t1 < plan.h) & (plan.v.len() >= (t1 + 1) * plan.g) & (plan.soc_min <= plan.soc_max) {
            use super::wide::sel;
            let (smin, smax, dsoc) = (plan.soc_min, plan.soc_max, plan.dsoc);
            let gm = (plan.g - 1) as f64;
            let row = &plan.v[t1 * plan.g..(t1 + 1) * plan.g];
            let dt = C::DELTA_T;
            for s in 0..=n_samples {
                let u = min_b + (max_b - min_b) * (s as f64 / n_samples as f64);
                let dsoc_c = eta_c * (-u).max(0.0) * dt;
                let dsoc_d = u.max(0.0) * dt / eta_d;
                let sn = soc + dsoc_c - dsoc_d;
                let sn = if sn < smin { smin } else { sn };
                let soc_next = if sn > smax { smax } else { sn };
                let revenue = u * price * dt;
                let abs_u = u.abs();
                let tx = C::KAPPA_TX * abs_u * dt;
                let db = (abs_u * dt) / cap;
                let deg = C::KAPPA_DEG * db.powf(C::BETA_DEG);
                let sp = sel(u == 0.0, 0.0, revenue - tx - deg);
                let x = (soc_next - smin) / dsoc;
                let x = if x < 0.0 { 0.0 } else { x };
                let x = if x > gm { gm } else { x };
                let g0 = x.floor() as usize;
                let g1 = (g0 + 1).min(plan.g - 1);
                let frac = x - g0 as f64;
                let (v0, v1) = unsafe { (*row.get_unchecked(g0), *row.get_unchecked(g1)) };
                let val = sp + (v0 * (1.0 - frac) + v1 * frac);
                let take = val > best_val;
                best_val = sel(take, val, best_val);
                best_u = sel(take, u, best_u);
            }
        } else {
        for s in 0..=n_samples {
            let u = min_b + (max_b - min_b) * (s as f64 / n_samples as f64);
            let dsoc_c = eta_c * (-u).max(0.0) * C::DELTA_T;
            let dsoc_d = u.max(0.0) * C::DELTA_T / eta_d;
            let soc_next = (soc + dsoc_c - dsoc_d).clamp(plan.soc_min, plan.soc_max);
            let val = step_profit(u, price, cap) + interp_v(plan, t + 1, soc_next);
            if val > best_val {
                best_val = val;
                best_u = u;
            }
        }
        }
        action[i] = best_u.clamp(min_b, max_b);
        av[i] = (best_val - hold_val).max(0.0);
    }
    enforce_flow_feasibility(challenge, state, action, &av, refill)
}

struct Violation {
    line: usize,
    flow: f64,
    amount: f64,
}

fn compute_flows(challenge: &Challenge, state: &State, action: &[f64]) -> Vec<f64> {
    let inj = challenge.compute_total_injections(state, action);
    let nn = challenge.network.num_nodes;
    (0..challenge.network.num_lines)
        .map(|l| super::wide::dot_acc(-0.0, &challenge.network.ptdf[l][..nn], &inj[..nn]))
        .collect()
}

fn most_violated_line(challenge: &Challenge, flows: &[f64]) -> Option<Violation> {
    let lims = &challenge.network.flow_limits;
    if lims.len() >= flows.len() {
        let mut bl = usize::MAX;
        let mut bf = 0.0;
        let mut ba = 0.0;
        for (l, (&flow, &limit)) in flows.iter().zip(&lims[..flows.len()]).enumerate() {
            let v = flow.abs() - limit;
            let take = (v > C::EPS_FLOW * limit) & ((bl == usize::MAX) | (v > ba));
            bl = if take { l } else { bl };
            bf = super::wide::sel(take, flow, bf);
            ba = super::wide::sel(take, v, ba);
        }
        return if bl == usize::MAX { None } else { Some(Violation { line: bl, flow: bf, amount: ba }) };
    }
    let mut best: Option<Violation> = None;
    for (l, &flow) in flows.iter().enumerate() {
        let limit = challenge.network.flow_limits[l];
        let v = flow.abs() - limit;
        if v > C::EPS_FLOW * limit {
            let c = Violation { line: l, flow, amount: v };
            match best {
                Some(ref cur) if c.amount <= cur.amount => {}
                _ => best = Some(c),
            }
        }
    }
    best
}

fn is_flow_feasible(challenge: &Challenge, state: &State, action: &[f64]) -> bool {
    most_violated_line(challenge, &compute_flows(challenge, state, action)).is_none()
}

fn soften(challenge: &Challenge, v: &Violation, action: &mut [f64], av: &[f64]) -> bool {
    let dir = v.flow.signum();
    if dir.abs() <= EPS {
        return false;
    }
    let mut worsening: Vec<(usize, f64, f64)> = Vec::new();
    let mut strength = 0.0;
    for (i, b) in challenge.batteries.iter().enumerate() {
        let ptdf = challenge.network.ptdf[v.line][b.node];
        let sc = dir * (ptdf * action[i]);
        if sc > EPS {
            strength += sc;
            let relief = (dir * ptdf).abs() * action[i].abs();
            let lpr = if relief > EPS {
                av[i].max(0.0) / relief
            } else {
                f64::INFINITY
            };
            worsening.push((i, sc, lpr));
        }
    }
    if worsening.is_empty() || strength <= EPS {
        return false;
    }
    worsening.sort_unstable_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
    let mut remaining = v.amount;
    let mut changed = false;
    for (i, contribution, _) in worsening {
        if remaining <= EPS {
            break;
        }
        let cut_frac = (remaining / contribution).clamp(0.0, 1.0);
        if cut_frac <= EPS {
            continue;
        }
        action[i] *= 1.0 - cut_frac;
        remaining -= cut_frac * contribution;
        changed = true;
    }
    changed
}

fn refill_pass(challenge: &Challenge, action: &mut [f64], target: &[f64], av: &[f64], flows: &mut [f64]) {
    let net = &challenge.network;
    let mut order: Vec<(usize, f64)> = (0..action.len())
        .filter(|&i| target[i].abs() > action[i].abs() + EPS)
        .map(|i| (i, av[i].max(0.0) / target[i].abs().max(EPS)))
        .collect();
    order.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    for (i, _) in order {
        let node = challenge.batteries[i].node;
        let dir = target[i].signum();
        let mut room = target[i].abs() - action[i].abs();
        for ln in 0..net.num_lines {
            let p = net.ptdf[ln][node] * dir;
            if p.abs() <= EPS {
                continue;
            }
            let lim = net.flow_limits[ln];
            let head = if p > 0.0 { lim - flows[ln] } else { lim + flows[ln] } - 1e-9 * lim;
            room = room.min(head.max(0.0) / p.abs());
            if room <= EPS {
                break;
            }
        }
        if room > EPS {
            let new_u = if dir > 0.0 { (action[i] + room).min(target[i]) } else { (action[i] - room).max(target[i]) };
            let du = new_u - action[i];
            action[i] = new_u;
            for ln in 0..net.num_lines {
                flows[ln] += net.ptdf[ln][node] * du;
            }
        }
    }
}

fn enforce_flow_feasibility(
    challenge: &Challenge,
    state: &State,
    mut action: Vec<f64>,
    av: &[f64],
    refill: bool,
) -> Result<Vec<f64>> {
    let net = &challenge.network;
    let mut flows = compute_flows(challenge, state, &action);
    if most_violated_line(challenge, &flows).is_none() { return Ok(action); }
    let target = if refill { action.clone() } else { Vec::new() };
    let mut prev = action.clone();
    let ptdf_min_len = net.ptdf[..net.num_lines].iter().map(|r| r.len()).min().unwrap_or(usize::MAX);
    let mut softened = false;
    for _ in 0..MAX_FLOW_ADJUST_ITERS {
        let Some(v) = most_violated_line(challenge, &flows) else {
            break;
        };
        if !soften(challenge, &v, &mut action, av) {
            break;
        }
        softened = true;
        for (i, b) in challenge.batteries.iter().enumerate() {
            let du = action[i] - prev[i];
            if du != 0.0 && ptdf_min_len > b.node {
                let nl = net.num_lines;
                let rows = &net.ptdf[..nl];
                let fl = &mut flows[..nl];
                unsafe {
                    let m8 = nl - nl % 8;
                    let mut ln = 0;
                    while ln < m8 {
                        for q in 0..8 {
                            *fl.get_unchecked_mut(ln + q) += *rows.get_unchecked(ln + q).get_unchecked(b.node) * du;
                        }
                        ln += 8;
                    }
                    while ln < nl {
                        *fl.get_unchecked_mut(ln) += *rows.get_unchecked(ln).get_unchecked(b.node) * du;
                        ln += 1;
                    }
                }
                prev[i] = action[i];
            } else if du != 0.0 {
                for ln in 0..net.num_lines {
                    flows[ln] += net.ptdf[ln][b.node] * du;
                }
                prev[i] = action[i];
            }
        }
    }
    if refill && softened && most_violated_line(challenge, &flows).is_none() {
        refill_pass(challenge, &mut action, &target, av, &mut flows);
    }
    if is_flow_feasible(challenge, state, &action) {
        return Ok(action);
    }
    let zero = vec![0.0; action.len()];
    if !is_flow_feasible(challenge, state, &zero) {
        return Err(anyhow!("grid infeasible even with zero battery actions"));
    }
    let base = action;
    let (mut low, mut high) = (0.0, 1.0);
    for _ in 0..GLOBAL_SCALE_BSEARCH_ITERS {
        let mid = 0.5 * (low + high);
        let scaled: Vec<f64> = base.iter().map(|u| mid * u).collect();
        if is_flow_feasible(challenge, state, &scaled) {
            low = mid;
        } else {
            high = mid;
        }
    }
    Ok(base.into_iter().map(|u| low * u).collect())
}

fn build_plans_grouped(challenge: &Challenge, grid: usize, iters: usize, step0: f64, prem: &[Vec<f64>], scen: &[Vec<Vec<f64>>], opts: &Opts, plans: Vec<Plan>, nu_prof: &[Vec<f64>]) -> Vec<Plan> {
    let net = &challenge.network;
    let m = challenge.num_batteries;
    let h = challenge.num_steps;
    let l = net.num_lines;
    let dt = C::DELTA_T;
    let exp_path = expected_path(challenge, prem);
    let gate_scen = sample_rt_paths(challenge, opts.gate_k, 0);
    let gate = |p: &[Plan], p_exp: f64| -> f64 {
        if gate_scen.is_empty() {
            p_exp
        } else {
            gate_scen.iter().map(|path| forward_plan(challenge, p, iters, step0, path, opts).1).sum::<f64>() / gate_scen.len() as f64
        }
    };
    let (_, p_exp, acts) = forward_plan(challenge, &plans, iters, step0, &exp_path, opts);
    let p_cur = gate(&plans, p_exp);
    let exo_flow: Vec<Vec<f64>> = (0..h)
        .map(|t| {
            (0..l)
                .map(|ln| (0..net.num_nodes).filter(|&k| k != net.slack_bus).map(|k| net.ptdf[ln][k] * challenge.exogenous_injections[t][k]).sum::<f64>())
                .collect()
        })
        .collect();
    let mut binding = vec![false; l];
    for t in 0..h {
        for ln in 0..l {
            let mut f = exo_flow[t][ln];
            for i in 0..m {
                f += net.ptdf[ln][challenge.batteries[i].node] * acts[t][i];
            }
            if f.abs() >= net.flow_limits[ln] * (1.0 - 1e-3) {
                binding[ln] = true;
            }
        }
    }
    fn find(p: &mut [usize], i: usize) -> usize {
        let mut r = i;
        while p[r] != r {
            r = p[r];
        }
        let mut x = i;
        while p[x] != r {
            let nx = p[x];
            p[x] = r;
            x = nx;
        }
        r
    }
    let theta = opts.grp_theta;
    let mut parent: Vec<usize> = (0..m).collect();
    for ln in 0..l {
        if !binding[ln] {
            continue;
        }
        for sgn in [1.0f64, -1.0] {
            let mut first: Option<usize> = None;
            for i in 0..m {
                if net.ptdf[ln][challenge.batteries[i].node] * sgn >= theta {
                    match first {
                        None => first = Some(i),
                        Some(f0) => {
                            let a = find(&mut parent, f0);
                            let b = find(&mut parent, i);
                            if a != b {
                                parent[b] = a;
                            }
                        }
                    }
                }
            }
        }
    }
    let mut clusters: Vec<Vec<usize>> = vec![Vec::new(); m];
    for i in 0..m {
        let r = find(&mut parent, i);
        clusters[r].push(i);
    }
    let orig = plans.clone();
    let mut out = plans;
    let mut changed = false;
    let nu_opt = if opts.grp == 2 { Some(nu_prof) } else { None };
    for members in clusters.iter().filter(|c| c.len() >= 2 && c.len() <= opts.grp_max) {
        let n = members.len() as f64;
        let cap: f64 = members.iter().map(|&i| challenge.batteries[i].capacity_mwh).sum();
        let share = |i: usize| challenge.batteries[i].capacity_mwh / cap.max(EPS);
        let p = BatParams {
            soc_min: members.iter().map(|&i| challenge.batteries[i].soc_min_mwh).sum(),
            soc_max: members.iter().map(|&i| challenge.batteries[i].soc_max_mwh).sum(),
            cap,
            deg_cap: cap * n.powf(-1.0 / C::BETA_DEG),
            eta_c: members.iter().map(|&i| share(i) * challenge.batteries[i].efficiency_charge).sum(),
            eta_d: members.iter().map(|&i| share(i) * challenge.batteries[i].efficiency_discharge).sum(),
            p_chg: members.iter().map(|&i| challenge.batteries[i].power_charge_mw).sum(),
            p_dis: members.iter().map(|&i| challenge.batteries[i].power_discharge_mw).sum(),
        };
        let mut prices = vec![vec![0.0f64; h]; scen.len().max(1)];
        for &i in members {
            let pi = plan_prices(challenge, i, scen, nu_opt, opts.plan_price_min, opts.nu_scale);
            for kk in 0..pi.len().min(prices.len()) {
                for t in 0..h {
                    prices[kk][t] += share(i) * pi[kk][t];
                }
            }
        }
        let mut caps = vec![(p.p_chg, p.p_dis); h];
        for ln in 0..l {
            if !binding[ln] {
                continue;
            }
            let pbar: f64 = members.iter().map(|&i| share(i) * net.ptdf[ln][challenge.batteries[i].node]).sum();
            if pbar.abs() < theta {
                continue;
            }
            let lim = net.flow_limits[ln];
            for t in 0..h {
                let e = exo_flow[t][ln];
                let (dis_room, chg_room) = if pbar > 0.0 { ((lim - e) / pbar, (lim + e) / pbar) } else { ((lim + e) / -pbar, (lim - e) / -pbar) };
                caps[t].0 = caps[t].0.min((chg_room * opts.grp_relax).max(0.0));
                caps[t].1 = caps[t].1.min((dis_room * opts.grp_relax).max(0.0));
            }
        }
        let vg = build_plan_core(challenge.batteries[members[0]].node, &p, h, grid, &prices, Some(&caps), opts);
        let kf = prices.len() as f64;
        let mut traj = vec![0.0f64; h + 1];
        let mut s: f64 = members.iter().map(|&i| challenge.batteries[i].soc_initial_mwh).sum();
        traj[0] = s;
        for t in 0..h {
            let price = prices.iter().map(|pk| pk[t]).sum::<f64>() / kf;
            let max_chg = ((p.soc_max - s).max(0.0) / (p.eta_c * dt)).min(caps[t].0).max(0.0);
            let max_dis = ((s - p.soc_min).max(0.0) * p.eta_d / dt).min(caps[t].1).max(0.0);
            let mut best_u = 0.0;
            let mut best_v = interp_v(&vg, t + 1, s);
            for j in 0..=opts.samples {
                let u = -max_chg + (max_chg + max_dis) * (j as f64 / opts.samples as f64);
                let sn = (s + p.eta_c * (-u).max(0.0) * dt - u.max(0.0) * dt / p.eta_d).clamp(p.soc_min, p.soc_max);
                let val = step_profit(u, price, p.deg_cap) + interp_v(&vg, t + 1, sn);
                if val > best_v {
                    best_v = val;
                    best_u = u;
                }
            }
            s = (s + p.eta_c * (-best_u).max(0.0) * dt - best_u.max(0.0) * dt / p.eta_d).clamp(p.soc_min, p.soc_max);
            traj[t + 1] = s;
        }
        for &i in members {
            let pl = &mut out[i];
            let others = 1.0 - share(i);
            for t in 0..h {
                let s_o = traj[t] * others;
                let base_v = interp_v(&vg, t, s_o);
                for gi in 0..pl.g {
                    let soc = pl.soc_min + gi as f64 * pl.dsoc;
                    pl.v[t * pl.g + gi] = interp_v(&vg, t, s_o + soc) - base_v;
                }
            }
        }
        changed = true;
    }
    if !changed {
        return out;
    }
    let (_, pe_new, _) = forward_plan(challenge, &out, iters, step0, &exp_path, opts);
    if gate(&out, pe_new) > p_cur { out } else { orig }
}

fn replan_mpc(challenge: &Challenge, state: &State, plans: &[Plan], grid: usize, iters: usize, step0: f64, prem: &[Vec<f64>], scen: &[Vec<Vec<f64>>], opts: &Opts, nu_now: &[f64]) -> Option<Vec<Plan>> {
    let t0 = state.time_step;
    let m = challenge.num_batteries;
    let exp_path = expected_path(challenge, prem);
    let gate_scen = sample_rt_paths(challenge, opts.gate_k, 0);
    let roll = |p: &[Plan]| -> (Vec<Vec<f64>>, f64) {
        let (nu, pe, _) = forward_plan_from(challenge, p, iters, step0, &exp_path, opts, t0, &state.socs, Some(nu_now));
        let val = if gate_scen.is_empty() {
            pe
        } else {
            gate_scen.iter().map(|path| forward_plan_from(challenge, p, iters, step0, path, opts, t0, &state.socs, Some(nu_now)).1).sum::<f64>() / gate_scen.len() as f64
        };
        (nu, val)
    };
    let (mut nu_prof, mut best_p) = roll(plans);
    let nominal: Vec<(f64, f64)> = challenge.batteries.iter().map(|b| (b.power_charge_mw, b.power_discharge_mw)).collect();
    let mut best: Option<Vec<Plan>> = None;
    for _ in 0..opts.mpc_r {
        let cur: Vec<Plan> = (0..m).map(|b| build_plan_eff(challenge, b, grid, scen, &nu_prof, nominal[b], opts)).collect();
        let (nu_k, pk) = roll(&cur);
        let a = opts.nu_damp;
        for t in t0..nu_prof.len() {
            for ln in 0..nu_prof[t].len() {
                nu_prof[t][ln] = (1.0 - a) * nu_prof[t][ln] + a * nu_k[t][ln];
            }
        }
        if pk > best_p {
            best_p = pk;
            best = Some(cur);
        }
    }
    best
}

fn blend_reserve(challenge: &Challenge, grid: usize, nu_prof: &[Vec<f64>], opts: &Opts, mut plans: Vec<Plan>) -> Vec<Plan> {
    let paths = sample_rt_paths(challenge, opts.res_k, 0x77);
    if paths.is_empty() {
        return plans;
    }
    let nu_opt = if nu_prof.iter().any(|r| r.iter().any(|&x| x != 0.0)) { Some(nu_prof) } else { None };
    let kf = paths.len() as f64;
    let w = opts.res_w;
    for i in 0..challenge.num_batteries {
        let b = &challenge.batteries[i];
        let mut acc = vec![0.0f64; plans[i].v.len()];
        for path in &paths {
            let prices = plan_prices(challenge, i, std::slice::from_ref(path), nu_opt, opts.plan_price_min, opts.nu_scale);
            let p = build_plan_priced(challenge, i, grid, &prices, (b.power_charge_mw, b.power_discharge_mw), opts);
            for (a, v) in acc.iter_mut().zip(p.v.iter()) {
                *a += v;
            }
        }
        for (v, a) in plans[i].v.iter_mut().zip(acc.iter()) {
            *v = (1.0 - w) * *v + w * a / kf;
        }
    }
    plans
}
