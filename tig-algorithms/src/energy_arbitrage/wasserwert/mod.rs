// wasserwert — clean-room energy_arbitrage (c008) optimizer.
//
// Kernidee: Statt fester Preisperzentil-Schwellen (spitzenernte) berechnen wir
// je Batterie den *Wasserwert* der gespeicherten Energie — den Grenznutzen einer
// MWh im Speicher als Funktion von Zeitschritt und Ladezustand — via rueckwaerts
// laufender dynamischer Programmierung (Bellman) ueber den Day-Ahead-Preisverlauf
// am Knoten. Das ist der klassische OR-Ansatz fuer Speicher-Arbitrage:
//   V_t(SoC) = max_u [ Stufengewinn(u, preis_t) + V_{t+1}(SoC') ],  V_H(.) = 0.
// Online handeln wir dann gegen den *tatsaechlich enthuellten* Echtzeitpreis:
// wir maximieren Sofortgewinn(u, rt_preis) + V_{t+1}(SoC'(u)). Damit sind drei
// Dinge zugleich richtig: (a) optimale Kapazitaetsreservierung fuer die groessten
// Spreads (statt fixer Perzentile), (b) automatische Ernte realisierter RT-Spitzen
// (rt >> Wasserwert ⇒ sofort entladen — spitzenerntes Kante, jetzt fundiert),
// (c) saubere Endabwicklung (V_H=0 erzwingt Liquidation).
//
// Netz-Kopplung (der entscheidende Hebel fuer enge Netze): statt jede Batterie
// einzeln zu entscheiden und dann per Weichklopfen zu kuerzen, loesen wir pro
// Zeitschritt ein kleines Netz-LP ueber ALLE Batterien+Leitungen gemeinsam —
//   max Σ_i [a_dis_i·p_i + a_chg_i·n_i],  u_i = p_i − n_i,
//   s.t. |exo_l + Σ_i PTDF[l][i]·u_i| ≤ limit_l,  Boxgrenzen,
// mit linearen Zielkoeffizienten a_* aus der Wasserwert-Steigung w_i=∂V/∂SoC.
// Geloest via Lagrange-Dual auf den (zweiseitigen) Leitungsgrenzen: Schatten-
// preise ν_l per Subgradient, je Iterat wird das Bang-Bang-Primal per wert-
// gewichtetem Weichklopfen zulaessig gemacht und das beste zulaessige Primal
// (nach echtem Wert) behalten. Ein Wert-Gate gegen das reine Weichklopfen sorgt
// dafuer, dass es auf lockeren Netzen nie schlechter wird. Wertgewichtetes
// Weichklopfen bleibt der Feasibility-Backstop.
//
// Alles aus publizierten OR-Methoden (Bellman-DP, Lagrange-/wertgewichtete
// Constraint-Relaxation); keine fremden Codezeilen.

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
    // P10: per-track defaults from the instance shape; explicit hyperparameters
    // always win.
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
    // Q1: deterministic quadrature of the public RT price law (0 = off).
    let sdp_q = hp_usize(hyperparameters, "sdp_q", td.sdp_q).min(5);
    let sdp_qj = hp_usize(hyperparameters, "sdp_qj", td.sdp_qj).min(3);

    // Expected congestion premium per (t, node) from the public price law —
    // the only predictable time structure in RT beyond DA (P1).
    let prem = expected_congestion_premium(challenge, cong_prem);
    // P3: planning price scenarios. K=0 -> single expected path (DA + premium).
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

    // Per-battery water-value tables. With plan_rounds>0 they are made
    // *network-aware* by an outer DP<->shadow-price iteration (gated so loose
    // networks never regress); otherwise per-battery/network-blind.
    let (plans, nu_prof): (Vec<Plan>, Vec<Vec<f64>>) = if plan_rounds > 0 && lp_iters > 0 {
        build_plans_netaware(challenge, grid, plan_rounds, lp_iters, lp_step, &prem, &scen, &opts)
    } else {
        (
            (0..challenge.num_batteries).map(|b| build_plan(challenge, b, grid, &scen, &opts)).collect(),
            vec![vec![0.0f64; challenge.network.num_lines]; challenge.num_steps],
        )
    };
    // S2: line-coupled cluster value function (gated on the planned profit).
    let plans = if opts.grp > 0 && lp_iters > 0 {
        build_plans_grouped(challenge, grid, lp_iters, lp_step, &prem, &scen, &opts, plans, &nu_prof)
    } else {
        plans
    };
    // S5: scenario-valued reserve.
    let plans = if opts.res_k > 0 { blend_reserve(challenge, grid, &nu_prof, &opts, plans) } else { plans };

    let plans_cell = std::cell::RefCell::new(plans);
    let nu_carry = std::cell::RefCell::new(vec![0.0f64; challenge.network.num_lines]);
    let policy = |c: &Challenge, s: &State| -> Result<Vec<f64>> {
        // S3: MPC re-planning from the realised SoC every mpc_n steps.
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

// P10: per-track defaults, keyed on the public instance shape
// (num_batteries, num_steps) of the five tracks; unknown shapes take the
// nearest fleet size. Values are the measured optima (see LIESMICH/BEFUND §8):
// the dual step optimum is track-specific and non-monotone in fleet size.
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
    // 2026-09-10: exact fleet LP per step (lp_exact=2, nu_scale=4) + lp_seg=4 + plan_rounds=3 on
    // every track; baseline/congested additionally sdp_k 128, gate_k 20, plan_rounds 8 (official
    // 100-nonce: 250.5k / 671.4k, +1.1 % / +0.45 %, fuel 47 / 113 billion on nonce 0). Official path (fuel .so, test_algorithm, 100 nonces/track): BASELINE 247.7k,
    // CONGESTED 668.4k, MULTIDAY 3.639M, DENSE 2.479M (vs. qualifier floors 258.7k / 676.5k /
    // 2.889M / 2.405M) -- +5..+43 % over the previous defaults; fuel < 1 % of budget.
    // 2026-09-19 (Q1): multiday / dense / capstone plan on a deterministic quadrature of the public
    // RT price law (Gauss-Hermite in the Gaussian shock + conditional Pareto jump means, weighted
    // DP expectation) instead of sampled paths: tuning seeds +0.7 % / +1.3 % / +4.8 %. On the
    // 96-step tracks the sampled SDP (sdp_k 128) stays better (-3.5 % / -0.3 % with quadrature).
    match num_batteries {
        // BASELINE (10 bat, 96 steps): loose grid, dual step 10, refill helps a little;
        // SDP over K=32 sampled RT paths (+5.7 %, K=128 gives nothing more).
        0..=14 => TrackDefaults { lp_step: 10.0, lp_iters: 100, sdp_k: 128, nu_avg: 1, refill: 1, gate_k: 20, plan_rounds: 8, lp_exact: 2, lp_seg: 4, nu_scale: 4.0, sdp_q: 0, sdp_qj: 0 },
        // CONGESTED (20, 96): tight grid, large dual step, refill + scenario gate (cheap here);
        // SDP K=32 (+4.4 %; K=128 only +0.6 % more at 3x the DP cost).
        15..=29 => TrackDefaults { lp_step: 30.0, lp_iters: 40, sdp_k: 128, nu_avg: 1, refill: 1, gate_k: 20, plan_rounds: 8, lp_exact: 2, lp_seg: 4, nu_scale: 4.0, sdp_q: 0, sdp_qj: 0 },
        // MULTIDAY (40, 192): tiny smooth dual step; refill/gate hurt or cost too much;
        // SDP K=16 (+2.2 %; K=8/32 slightly less).
        30..=49 => TrackDefaults { lp_step: 0.3, lp_iters: 200, sdp_k: 16, nu_avg: 1, refill: 0, gate_k: 0, plan_rounds: 3, lp_exact: 2, lp_seg: 4, nu_scale: 4.0, sdp_q: 3, sdp_qj: 2 },
        // DENSE (60, 192): default step, averaged dual; refill hurts; SDP K=8 (+1.8 %, K=32 less).
        50..=79 => TrackDefaults { lp_step: 3.0, lp_iters: 200, sdp_k: 8, nu_avg: 1, refill: 0, gate_k: 0, plan_rounds: 3, lp_exact: 2, lp_seg: 4, nu_scale: 4.0, sdp_q: 5, sdp_qj: 2 },
        // CAPSTONE (100, 192): planning rounds bring nothing (wall time only) -> 0;
        // more online dual iterations pay off instead. SDP is measured negative here
        // (K=8 -2k, K=32 -10k): CAPSTONE gains only from the online dual, not from planning.
        _ => TrackDefaults { lp_step: 3.0, lp_iters: 400, sdp_k: 0, nu_avg: 0, refill: 0, gate_k: 0, plan_rounds: 3, lp_exact: 2, lp_seg: 4, nu_scale: 4.0, sdp_q: 5, sdp_qj: 2 },
    }
}

// Algorithm switches (hyperparameters, measured one by one).
#[derive(Clone, Copy)]
struct Opts {
    // P8: add the exact power-limited SoC targets as DP candidates (the grid
    // window floor() otherwise caps discharge power at ~90 % on grid=96).
    dp_exact: bool,
    // Optional precomputation; short horizons can lose to the setup cost.
    dp_cache: bool,
    // P5: LP objective coefficients from the secant of V over the actual
    // bang-bang SoC move (per direction) instead of the one-sided tangent.
    lp_repeat_cache: bool,
    lp_secant: bool,
    // P2: return the ergodic (running-mean) dual instead of the last iterate.
    nu_avg: bool,
    // P2: warm-start the dual from the previous time step.
    nu_warm: bool,
    // P2: subgradient step rule: 0 = step0/(1+it), 1 = constant step0, 2 = step0/sqrt(1+it).
    lp_sched: usize,
    // P2: normalise the dual step by the line's maximal battery flow swing
    // sum_i |PTDF[l][i]|*max(|lo_i|,hi_i) instead of the line limit, so the
    // step is track-independent (flow response per nu-unit grows with fleet size).
    lp_norm: bool,
    // P9c: floor for the effective planning price (LAMBDA_MIN=-200 or LAMBDA_DA_MIN=0).
    plan_price_min: f64,
    // P9b: online sample count, also used in the planning rollout.
    samples: usize,
    // P6: ratio-test refill pass after softening (batteries by value density).
    refill: bool,
    // P4: plan gate on K self-sampled RT scenarios (public price law, own seeds)
    // instead of the expected path; 0 = expected-path gate.
    gate_k: usize,
    // P4: damping of the dual profile between outer rounds (1 = replace).
    nu_damp: f64,
    // P3: stochastic DP over K self-sampled RT scenarios (public price law,
    // own seeds): v = (1/K) sum_k max_u{profit(u, lambda_k) + v'} instead of
    // the certainty-equivalent max over the expected price. 0 = deterministic.
    sdp_k: usize,
    // P7: effective power bounds in network-aware planning from the realised
    // |u_i(t)| of the expected-path rollout (quantile cap_q); 0 = off.
    cap_q: f64,
    // S1: exact per-step fleet dispatch by bounded-variable simplex instead of
    // the subgradient dual: 0 = off, 1 = online only, 2 = online + planning
    // rollouts, 3 = planning rollouts only.
    lp_exact: usize,
    // S1: objective segments per direction for the exact LP: 0 = tangent
    // coefficients (as the dual uses), S >= 1 = S secant segments of
    // V_{t+1} over the actual SoC move incl. degradation (piecewise-linear
    // concave objective, concavified by sorting slopes).
    lp_seg: usize,
    // S1: scale of the line shadow prices in the planning price feedback
    // (exact duals are $/MW per step; /DELTA_T converts to $/MWh).
    nu_scale: f64,
    // S2: line-coupled value function. Batteries pushing the same binding
    // line in the same direction (|PTDF| >= grp_theta) form a cluster; the
    // cluster is planned as ONE aggregate battery whose per-step power is
    // capped by the binding lines' headroom, and each member's table is the
    // marginal value of its own energy along the cluster's planned
    // trajectory. 0 = off, 1 = node prices, 2 = nu-corrected prices.
    grp: usize,
    grp_theta: f64,
    // S2: multiplier on the line-headroom power caps (1 = physical headroom).
    grp_relax: f64,
    // S2: largest cluster that is aggregated (bigger unions stay per-battery).
    grp_max: usize,
    // S3: MPC re-planning every mpc_n steps from the realised SoC (0 = off),
    // mpc_r outer DP<->nu rounds per re-plan, gated like the initial plan.
    mpc_n: usize,
    mpc_r: usize,
    // S5: scenario-valued reserve: blend the (S)DP table with the mean of
    // res_k per-scenario deterministic tables (perfect-information relaxation
    // of the reserve value), weight res_w.
    res_k: usize,
    res_w: f64,
    // Q1: quadrature weights of the planning scenarios (qn > 0: the DP takes the weighted
    // expectation sum_k qw[k] * best_k instead of the plain mean over the scenarios).
    qw: [f64; 32],
    qn: usize,
}

// ---------------- per-battery DP water-value plan ----------------
#[derive(Clone)]
struct Plan {
    node: usize,
    soc_min: f64,
    soc_max: f64,
    g: usize,
    dsoc: f64,
    v: Vec<f64>, // flattened [(H+1) * g], row t*g + gi
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
    // delta_soc = soc_target - soc_now; signed action u (neg = charge, pos = discharge)
    let dt = C::DELTA_T;
    if delta_soc > 0.0 {
        -(delta_soc / (eta_c * dt))
    } else if delta_soc < 0.0 {
        (-delta_soc) * eta_d / dt
    } else {
        0.0
    }
}

// Expected congestion premium E[RT − DA] per (t, node) from the PUBLIC price law
// (market.rs / network.rs, no seeds involved): the RT price at step t carries
// GAMMA_PRICE·max(z,0) at node i iff a congestion indicator fired for i, and the
// indicators for step t are drawn from the exogenous injections of step t−1:
// line l fires with P_l = min(1, (|PTDF_l·exo|/(τ·limit_l))^10) independently,
// node i fires if any line incident to i fires. Step 0 has no indicators.
// E[GAMMA_PRICE·max(z,0)] = GAMMA_PRICE/√(2π) ≈ 7.98 $/MWh. `scale` = hp cong_prem.
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

// Planning prices per scenario and step for battery i: scenario node price,
// optionally minus the LMP-like congestion component from line shadow prices.
fn plan_prices(challenge: &Challenge, i: usize, scen: &[Vec<Vec<f64>>], nu_prof: Option<&[Vec<f64>]>, price_min: f64, nu_scale: f64) -> Vec<Vec<f64>> {
    let node = challenge.batteries[i].node;
    let net = &challenge.network;
    let cong: Vec<f64> = (0..challenge.num_steps)
        .map(|t| match nu_prof {
            None => 0.0,
            Some(nu) => nu_scale * (0..net.num_lines).map(|ln| net.ptdf[ln][node] * nu[t][ln]).sum::<f64>(),
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

// Linear interpolation of one DP row (row start `base`) at SoC `soc`.
#[inline]
fn interp_row(v: &[f64], base: usize, g: usize, soc_min: f64, dsoc: f64, soc: f64) -> f64 {
    let x = ((soc - soc_min) / dsoc).clamp(0.0, (g - 1) as f64);
    let g0 = x.floor() as usize;
    let g1 = (g0 + 1).min(g - 1);
    let frac = x - g0 as f64;
    v[base + g0] * (1.0 - frac) + v[base + g1] * frac
}

// Backward DP over per-scenario planning price vectors for battery i (P3:
// v_t = (1/K) sum_k max_u{profit(u, price_k) + v_{t+1}}; K=1 = deterministic).
// `pcap` = (charge, discharge) power caps used for the planning window (P7).
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

// Physical parameters of a (real or aggregate) battery for the DP.
#[derive(Clone, Copy)]
struct BatParams {
    soc_min: f64,
    soc_max: f64,
    cap: f64,
    // Capacity used in the degradation term (aggregate: cap * n^(-1/beta)).
    deg_cap: f64,
    eta_c: f64,
    eta_d: f64,
    p_chg: f64,
    p_dis: f64,
}

// The DP proper; `caps_t` = optional per-step (charge, discharge) power caps (S2).
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
    let mut cands: Vec<(f64, f64)> = Vec::with_capacity(64); // (u, continuation)
    for t in (0..h).rev() {
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
                // Exact power-limited targets (P8): full charge / full discharge
                // for one step, clamped to the SoC box; value by interpolation.
                let tc = (soc + max_charge_dsoc).min(soc_max);
                if tc - soc > EPS {
                    cands.push((soc_to_action(tc - soc, eta_c, eta_d), interp_row(&v, next, g, soc_min, dsoc, tc)));
                }
                let td = (soc - max_dis_dsoc).max(soc_min);
                if soc - td > EPS {
                    cands.push((soc_to_action(td - soc, eta_c, eta_d), interp_row(&v, next, g, soc_min, dsoc, td)));
                }
            }
            // hold
            for bk in best.iter_mut() {
                *bk = v[next + gi];
            }
            for &(u, cont) in &cands {
                // step_profit(u, price) = u*dt*price - (tx*|u|*dt + deg)
                let abs_u = u.abs();
                let cost = C::KAPPA_TX * abs_u * dt + C::KAPPA_DEG * ((abs_u * dt) / cap).powf(C::BETA_DEG);
                let rev = u * dt;
                for kk in 0..k {
                    let val = rev * prices[kk][t] - cost + cont;
                    if val > best[kk] {
                        best[kk] = val;
                    }
                }
            }
            v[cur + gi] = scen_mean(&best, opts);
        }
    }
    Plan { node, soc_min, soc_max, g, dsoc, v, h }
}

// With fixed power limits, feasible actions, costs and interpolation weights
// are identical at every time step. Store them once in reference action order.
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
    // (revenue coefficient, cost, next row index, optional interpolation weight)
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
    for t in (0..h).rev() {
        if k > 1 { for kk in 0..k { step_prices[kk] = prices[kk][t]; } }
        let next = (t + 1) * g;
        for gi in 0..g {
            best.fill(v[next + gi]);
            for &(rev, cost, gt, frac) in &actions[starts[gi]..starts[gi + 1]] {
                let cont = match frac {
                    None => v[next + gt],
                    Some(f) => v[next + gt] * (1.0 - f) + v[next + (gt + 1).min(g - 1)] * f,
                };
                if k == 1 {
                    // A single scenario needs no price gathering or vector traversal.
                    let val = rev * prices[0][t] - cost + cont;
                    if val > best[0] { best[0] = val; }
                } else {
                    for (bk, &price) in best.iter_mut().zip(&step_prices) {
                        let val = rev * price - cost + cont;
                        if val > *bk { *bk = val; }
                    }
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
    let transition = |u: f64| {
        let abs_u = u.abs();
        C::KAPPA_TX * abs_u * dt + C::KAPPA_DEG * ((abs_u * dt) / cap).powf(C::BETA_DEG)
    };
    // Cache only the union of transitions reachable under the largest power
    // caps in this horizon. Flat storage avoids one allocation per source row.
    // Each (source,target) subtraction remains exactly the reference operation.
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
    let mut cands: Vec<(f64, f64, f64)> = Vec::with_capacity(64); // action, continuation, cost
    for t in (0..h).rev() {
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
                // Exact power-limited targets (P8): full charge / full discharge
                // for one step, clamped to the SoC box; value by interpolation.
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
            // hold
            for bk in best.iter_mut() {
                *bk = v[next + gi];
            }
            for &(u, cont, cost) in &cands {
                // step_profit(u, price) = u*dt*price - (tx*|u|*dt + deg)
                let rev = u * dt;
                for kk in 0..k {
                    let val = rev * prices[kk][t] - cost + cont;
                    if val > best[kk] {
                        best[kk] = val;
                    }
                }
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


// ---------------- water-value slope + per-step network LP ----------------
#[inline]
fn v_slope(plan: &Plan, t: usize, soc: f64) -> f64 {
    // Local marginal value dV_{t+1}/dSoC at soc ($/MWh), from the DP grid.
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
    // True per-step value achieved: immediate profit + continuation V_{t+1}(SoC').
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

// Per-step network dispatch by Lagrangian dual on the two-sided flow limits.
// Returns (best feasible action, line shadow prices nu). The objective is
// linearized at the current SoC via the DP water-value slope; each dual iterate
// yields a bang-bang primal made feasible by value-weighted softening, and the
// best-valued feasible primal (gated against `base_action`) is kept.
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

    // S1: exact fleet dispatch (simplex), gated against the base action on true value.
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
            // Secant of V_{t+1} over the actual bang-bang move in each direction.
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

    // Step denominator per line (P2 normalisation).
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
    for it in 0..iters {
        let mut changed = it == 0 || !opts.lp_repeat_cache;
        for i in 0..m {
            let sh: f64 = (0..l).map(|ln| nu[ln] * net.ptdf[ln][node[i]]).sum();
            let ad = a_dis[i] - sh;
            let ac = a_chg[i] + sh;
            let (next_u, next_av) = if ad > 0.0 && ad >= ac {
                (hi[i], (a_dis[i] * hi[i]).max(0.0))
            } else if ac > 0.0 {
                (lo[i], (a_chg[i] * (-lo[i])).max(0.0))
            } else { (0.0, 0.0) };
            // Bit equality also distinguishes signed zero. The repair, refill,
            // value and raw flow depend only on these vectors and fixed inputs.
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
            for ln in 0..l { flow[ln] += net.ptdf[ln][node[i]] * u[i]; }
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


// ---------------- S1: exact per-step fleet dispatch (bounded-variable simplex) ----------------
// max c'x  s.t.  A x + s = b,  0 <= x <= ub,  0 <= s <= sb.  Dense-tableau primal
// simplex for bounded variables (textbook: Chvátal ch. 8, Dantzig pricing with
// Bland's rule as anti-cycling fallback). The slack basis is primal feasible
// whenever 0 <= b <= sb, i.e. the network is feasible at zero battery action.
// Returns (x, y) with y the row duals (y_r >= 0 iff the upper side binds).
fn solve_bounded_lp(a: &[Vec<f64>], b: &[f64], sb: &[f64], c: &[f64], ub: &[f64]) -> Option<(Vec<f64>, Vec<f64>)> {
    const TOL: f64 = 1e-9;
    const PIV: f64 = 1e-10;
    let r = a.len();
    let n = c.len();
    // The first lazy-row round has no network constraints. Independent bounded
    // variables need no simplex pricing/pivot loop; use the identical tolerance.
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
    let mut degenerate = 0usize;
    let mut bland = false;
    let mut converged = false;
    for _ in 0..max_iter {
        // Pricing.
        let mut enter: Option<usize> = None;
        let mut best = TOL;
        for j in 0..nc {
            if is_basic[j] || ubx[j] <= 0.0 {
                continue;
            }
            let gain = if at_upper[j] { -d[j] } else { d[j] };
            if gain > best {
                enter = Some(j);
                if bland {
                    break;
                }
                best = gain;
            }
        }
        let Some(j) = enter else {
            converged = true;
            break;
        };
        let dir = if at_upper[j] { -1.0 } else { 1.0 };
        // Ratio test (bound flip of the entering variable included).
        let mut theta = ubx[j];
        let mut leave: Option<(usize, bool)> = None;
        for i in 0..r {
            let alpha = t[i * nc + j] * dir; // basic i moves by -alpha*theta
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
        if theta <= 1e-13 {
            degenerate += 1;
            if degenerate > 30 {
                bland = true;
            }
        } else {
            degenerate = 0;
        }
        for i in 0..r {
            xb[i] -= t[i * nc + j] * dir * theta;
        }
        match leave {
            None => {
                at_upper[j] = !at_upper[j];
            }
            Some((rr, up)) => {
                let p = t[rr * nc + j];
                let inv = 1.0 / p;
                for k in 0..nc {
                    t[rr * nc + k] *= inv;
                }
                let (pre, rest) = t.split_at_mut(rr * nc);
                let (prow, post) = rest.split_at_mut(nc);
                for (i, row) in pre.chunks_mut(nc).enumerate() {
                    let f = row[j];
                    if f != 0.0 {
                        for k in 0..nc { row[k] -= f * prow[k]; }
                    }
                    let _ = i;
                }
                for row in post.chunks_mut(nc) {
                    let f = row[j];
                    if f != 0.0 {
                        for k in 0..nc { row[k] -= f * prow[k]; }
                    }
                }
                let f = d[j];
                if f != 0.0 {
                    for k in 0..nc { d[k] -= f * prow[k]; }
                }
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

// Column model: per battery up to `seg` discharge and `seg` charge segments.
// seg = 0: one segment per direction with the tangent coefficients of the
// dual (dt*(rt - tx - w/eta_d), dt*(w*eta_c - rt - tx)); seg >= 1: secants of
// V_{t+1} over equal SoC moves plus the exact transaction/degradation cost
// increments, slopes sorted non-increasing (concave envelope). Lines enter
// the LP lazily (row generation): only lines the current dispatch violates
// are added, and the LP is re-solved until all |flow| <= limit.
fn lp_exact_dispatch(challenge: &Challenge, state: &State, plans: &[Plan], seg: usize) -> Option<(Vec<f64>, Vec<f64>)> {
    let t = state.time_step;
    let m = challenge.num_batteries;
    let net = &challenge.network;
    let l = net.num_lines;
    let dt = C::DELTA_T;
    let tx = C::KAPPA_TX;
    let s_per = seg.max(1);
    // Columns: (battery, sign, ub, c).
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
    // Exogenous line flows (slack bus balances the exogenous injections).
    let exo_flow: Vec<f64> = {
        let mut inj = vec![0.0; net.num_nodes];
        let mut sum = 0.0;
        for k in 0..net.num_nodes {
            if k != net.slack_bus { inj[k] = state.exogenous_injections[k]; sum += inj[k]; }
        }
        inj[net.slack_bus] = -sum;
        (0..l).map(|ln| (0..net.num_nodes).map(|k| net.ptdf[ln][k] * inj[k]).sum::<f64>()).collect()
    };
    const SHRINK: f64 = 1e-9; // keep flows strictly inside the limit (EPS_FLOW = 1e-6)
    let mut rows: Vec<usize> = Vec::new();
    let mut in_rows = vec![false; l];
    let mut u = vec![0.0f64; m];
    let mut nu = vec![0.0f64; l];
    // Previously added constraint rows are invariant during this dispatch.
    let mut a: Vec<Vec<f64>> = Vec::new();
    let mut b: Vec<f64> = Vec::new();
    let mut sb: Vec<f64> = Vec::new();
    for _round in 0..(l + 1) {
        let (x, y) = solve_bounded_lp(&a, &b, &sb, &c, &ub)?;
        for i in 0..m { u[i] = 0.0; }
        for j in 0..n { u[col_b[j]] += col_sgn[j] * x[j]; }
        for i in 0..m {
            let (lo, hi) = state.action_bounds[i];
            u[i] = u[i].clamp(lo, hi);
        }
        for ln in 0..l { nu[ln] = 0.0; }
        for (k, &ln) in rows.iter().enumerate() { nu[ln] = y[k]; }
        let mut added = false;
        for ln in 0..l {
            if in_rows[ln] { continue; }
            let mut f = exo_flow[ln];
            for i in 0..m { f += net.ptdf[ln][challenge.batteries[i].node] * u[i]; }
            let lim = net.flow_limits[ln];
            if f.abs() > lim * (1.0 - 0.5 * SHRINK) {
                rows.push(ln);
                a.push((0..n).map(|j| col_sgn[j] * net.ptdf[ln][challenge.batteries[col_b[j]].node]).collect());
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

// ---------------- network-aware multi-period water-value planning ----------------
// Deterministic outer loop: DP (per battery, across time) <-> per-step network LP
// (all batteries, across the network). The line shadow prices nu_l(t) from the LP
// are fed back into the DP as an LMP-like effective node price
//   da_eff_i(t) = da_i(t) - sum_l PTDF[l][i]*nu_l(t),
// so planning already discounts energy behind congested lines. Gated on the
// deterministic DA-planned profit so loose networks never regress.

// Physical action bounds (challenge model spec, replicated — not an algorithm).
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

// DP over an effective per-step planning price (DA + premium minus congestion component).
fn build_plan_eff(challenge: &Challenge, i: usize, g: usize, scen: &[Vec<Vec<f64>>], nu_prof: &[Vec<f64>], pcap: (f64, f64), opts: &Opts) -> Plan {
    build_plan_priced(challenge, i, g, &plan_prices(challenge, i, scen, Some(nu_prof), opts.plan_price_min, opts.nu_scale), pcap, opts)
}

// Deterministic forward pass on the expected RT path: returns (nu_profile[t][l], planned_profit).
// Expected RT path = DA + expected congestion premium (deterministic plan).
fn expected_path(challenge: &Challenge, prem: &[Vec<f64>]) -> Vec<Vec<f64>> {
    (0..challenge.num_steps)
        .map(|t| (0..challenge.network.num_nodes).map(|k| challenge.market.day_ahead_prices[t][k] + prem[t][k]).collect())
        .collect()
}

// Q1: expectation over the planning scenarios for the DP: quadrature weights when the scenarios
// come from `quadrature_paths` (opts.qn == number of scenarios), otherwise the plain mean.
#[inline]
fn scen_mean(best: &[f64], opts: &Opts) -> f64 {
    if opts.qn > 0 && opts.qn == best.len() {
        best.iter().zip(&opts.qw[..opts.qn]).map(|(b, w)| b * w).sum()
    } else {
        best.iter().sum::<f64>() / best.len().max(1) as f64
    }
}

// Q1: deterministic quadrature of the public RT price law instead of sampled paths.
// Per node and step RT = DA * (1 + sigma * xi) + premium + [jump] DA * X with xi ~ N(0, 1),
// jump probability p and X Pareto(alpha) on [1, inf). The DP of one battery only sees the price
// at its own node, so the scenarios are "paths" of fixed quantile levels:
//   xi: probabilists' Gauss-Hermite nodes (nq = 1, 3 or 5 points);
//   jump: none (weight 1-p) or the conditional mean of X on nj equal-probability-mass slices of
//   the upper tail (0: ignore jumps, 1: E[X], 2: slices [0, 0.9) / [0.9, 1), 3: + [0.99, 1)).
// Congestion premium: the expected premium `prem` (as in the certainty-equivalent path).
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
    // E[X * 1(U in [a, b))] for X = (1-U)^(-1/alpha): ((1-a)^e - (1-b)^e) / e with e = 1 - 1/alpha.
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

// P4: K RT price paths sampled from the PUBLIC price law (Market::generate_rt_prices,
// Network::generate_congestion_indicators) with OUR OWN seeds derived from the
// public instance seed — the same structure the rollout uses (indicators for
// step t from exo[t-1], none at step 0), just our own randomness.
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

// S3: rollout from step t0 with SoCs socs0 (nu rows / actions before t0 stay
// zero, profit counts steps t0..H only); nu0 seeds the dual warm start at t0.
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

// P7: effective (charge, discharge) power caps per battery = quantile q of the
// realised non-zero |u_i(t)| in the planning rollout; nominal if too few samples.
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
    // Gate value: mean profit over K sampled RT scenarios (P4), or the
    // expected-path profit when gate_k = 0.
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
        // nu_prof / acts stem from the expected-path rollout of `cur` (the
        // rollout of the previous round doubles as this round's dual sample).
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

// ---------------- online decision ----------------
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

        let hold_val = interp_v(plan, t + 1, soc); // step_profit(0)=0
        let mut best_u = 0.0;
        let mut best_val = hold_val;
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
        action[i] = best_u.clamp(min_b, max_b);
        av[i] = (best_val - hold_val).max(0.0);
    }
    enforce_flow_feasibility(challenge, state, action, &av, refill)
}

// ---------------- value-weighted network feasibility ----------------
struct Violation {
    line: usize,
    flow: f64,
    amount: f64,
}

fn compute_flows(challenge: &Challenge, state: &State, action: &[f64]) -> Vec<f64> {
    let inj = challenge.compute_total_injections(state, action);
    (0..challenge.network.num_lines)
        .map(|l| {
            (0..challenge.network.num_nodes)
                .map(|k| challenge.network.ptdf[l][k] * inj[k])
                .sum::<f64>()
        })
        .collect()
}

fn most_violated_line(challenge: &Challenge, flows: &[f64]) -> Option<Violation> {
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

// Cut the least-profitable trades first (smallest value-loss per unit of flow
// relief) until the most-violated line is within limits.
fn soften(challenge: &Challenge, v: &Violation, action: &mut [f64], av: &[f64]) -> bool {
    let dir = v.flow.signum();
    if dir.abs() <= EPS {
        return false;
    }
    let mut worsening: Vec<(usize, f64, f64)> = Vec::new(); // (idx, contribution, loss_per_relief)
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

// P6: softening only shrinks trades and leaves the point strictly inside several
// limits. Ratio-test refill: batteries in order of value density av/|u_target|
// get their |u| raised back toward the target as far as every line's headroom
// permits (min over lines of headroom/|PTDF|), flows updated incrementally.
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
            let p = net.ptdf[ln][node] * dir; // flow change per unit |u| increase
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
            // Never overshoot the (in-bounds) target by rounding.
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
    // Nothing changed since this exact flow evaluation: return without copying
    // action buffers or repeating the same matrix-vector product.
    if most_violated_line(challenge, &flows).is_none() { return Ok(action); }
    let target = if refill { action.clone() } else { Vec::new() };
    let mut prev = action.clone();
    let mut softened = false;
    for _ in 0..MAX_FLOW_ADJUST_ITERS {
        let Some(v) = most_violated_line(challenge, &flows) else {
            break;
        };
        if !soften(challenge, &v, &mut action, av) {
            break;
        }
        softened = true;
        // Incremental flow update: flow_l += PTDF[l][node_i] * delta_u_i.
        for (i, b) in challenge.batteries.iter().enumerate() {
            let du = action[i] - prev[i];
            if du != 0.0 {
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
    // Final exact check (guards against drift of the incremental update).
    if is_flow_feasible(challenge, state, &action) {
        return Ok(action);
    }
    // Last resort: global bisection scale (feasible at zero is guaranteed).
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

// ---------------- S2: line-coupled (cluster) value function ----------------
// Batteries that push the same binding line in the same direction are
// substitutes: the value of their joint energy is not the sum of the
// per-battery values. Each such cluster is planned as ONE aggregate battery
// (capacities/powers summed, efficiencies capacity-weighted, degradation
// scaled for n equal shares) whose per-step power is capped by the binding
// lines' headroom (limit -/+ exogenous flow, divided by the mean PTDF). The
// aggregate's greedy expected-path trajectory S(t) fixes the operating point;
// member i then gets the marginal table V_i(t,s) = V_G(t, S_-i(t)+s) - V_G(t, S_-i(t)),
// S_-i = S*(1 - cap_i/cap_G). Gated like the network-aware plan.
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
    // Exogenous line flows per step (PTDF[l][slack] = 0, so the slack balance is moot).
    let exo_flow: Vec<Vec<f64>> = (0..h)
        .map(|t| {
            (0..l)
                .map(|ln| (0..net.num_nodes).filter(|&k| k != net.slack_bus).map(|k| net.ptdf[ln][k] * challenge.exogenous_injections[t][k]).sum::<f64>())
                .collect()
        })
        .collect();
    // Lines that bind somewhere in the expected-path rollout of the current plans.
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
    // Union-find: batteries with |PTDF| >= theta of the same sign on a binding line.
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
        // Cluster prices: capacity-weighted member planning prices per scenario and step.
        let mut prices = vec![vec![0.0f64; h]; scen.len().max(1)];
        for &i in members {
            let pi = plan_prices(challenge, i, scen, nu_opt, opts.plan_price_min, opts.nu_scale);
            for kk in 0..pi.len().min(prices.len()) {
                for t in 0..h {
                    prices[kk][t] += share(i) * pi[kk][t];
                }
            }
        }
        // Per-step power caps from the headroom of the cluster's binding lines.
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
        // Greedy cluster trajectory on the scenario-mean price.
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
        // Member tables: marginal value of own energy along the cluster path.
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

// ---------------- S3: MPC re-planning from the realised state ----------------
// Every mpc_n steps: roll the current plans out from (t0, realised SoCs) on
// the expected path to get a dual profile consistent with where the fleet
// actually is (the initial plan's profile assumed the planned trajectory),
// rebuild the tables with it (mpc_r rounds), and accept only if the planned
// profit from t0 on (expected path or the gate_k scenarios) improves.
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

// ---------------- S5: scenario-valued reserve ----------------
// The (S)DP table values held energy by the expected continuation. The
// perfect-information relaxation values it by the mean over res_k sampled RT
// paths of the path-optimal continuation (each path exploits its own spikes),
// an upper valuation of the reserve. Blend: V = (1-w)*V_sdp + w*V_pi.
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
