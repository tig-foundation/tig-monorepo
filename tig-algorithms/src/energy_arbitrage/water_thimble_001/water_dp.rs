// TIG's UI uses the pattern `tig_challenges::<challenge_name>` to automatically detect your algorithm's challenge
use crate::{seeded_hasher, HashMap, HashSet};
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::sync::{Arc, Mutex};
use tig_challenges::energy_arbitrage::*;

// Physical / economic constants of the challenge (kept local on purpose).
const DT: f64 = 0.25;
const KAPPA_TX: f64 = 0.25;
const KAPPA_DEG: f64 = 1.0;
// Congestion premium scale ($/MWh) added to an RT price at a node flagged congested.
const GAMMA_PRICE: f64 = 20.0;
// E[max(Z, 0)] for Z ~ N(0,1) = 1/sqrt(2*pi); the premium is GAMMA_PRICE * max(Z, 0).
const E_HALF_NORMAL: f64 = 0.398_942_280_401_432_7;
// E[Z | Z > 0] for Z ~ N(0,1) = sqrt(2/pi). Conditional on the premium being nonzero at all,
// this is its mean size, which is twice E_HALF_NORMAL because the premium is zero on the
// half of the draws with Z <= 0.
const E_HALF_NORMAL_POS: f64 = 0.797_884_560_802_865_4;
// Exponent in the congestion trigger probability (|flow| / (tau * limit))^CONG_EXP.
const CONG_EXP: i32 = 10;
// Most price atoms a single step can need: five diffusive points, each split across the two
// congestion-premium branches, plus the jump atom.
const MAX_PRICE_ATOMS: usize = 11;
// Below this probability of a nonzero congestion premium the branch is folded back into its
// mean, so a node the network never threatens costs the sweep nothing extra.
const MIN_CONG_PROB: f64 = 1e-3;

// Consecutive rejected pairs after which the counter-flow pairing walk gives up. The walk is
// ordered by marginal value from both ends, so a short run of rejects means the ends have met
// in the middle where no pair pays; three is enough to tolerate one odd battery.
const PAIR_MISS_LIMIT: usize = 3;

// Exact deliverable-MW evaluations the counter-flow pairing walk is allowed per discharge end
// before it settles for the best partner it has already priced exactly.
//
// The walk scores a candidate partner by `(m_dis + m_chg) * mw`, where `mw` is bounded by the two
// ends' remaining power and by the headroom on the single most-loaded line. That single-line bound
// is a relaxation of the feasible set, so the score it produces is an *upper bound* on the pair's
// real one-step gain, and it is loose by exactly the amount the other lines bind: a partner whose
// difference-of-PTDF-columns barely touches `lstar` but loads some other line heavily screens as
// the best pair in the fleet and then delivers nothing. On the tight tracks that is the common
// case -- 300 lines at a limit scaled by 0.40, most of them within 15% of their cap -- and the
// walk's response was to take the tiny step anyway, retire the partner and re-scan, which spends
// an O(lines) pass per rejected partner and hands the step's counter-flow capacity to whichever
// partner happens to survive rather than to the best one.
//
// Because the screened score is an upper bound, the exact argmax is recoverable by the standard
// lazy refinement: take the best remaining bound, compute that partner's deliverable MW over every
// line, and stop as soon as the exact value dominates every bound still on the list. The first
// such evaluation is the `fraction` pass the walk already made after selecting, so the common case
// -- one dominant line, where the bound is tight and the first candidate wins outright -- costs
// nothing beyond what was already paid and selects the same partner. Only where the bound is
// misleading does a second or third pass run, and those are the steps where the old walk was
// paying for a retire-and-re-scan cycle instead.
//
// Three is the cap because the list is ordered by a valid bound: the three best bounds contain the
// exact winner unless two partners' bounds are both loose and both above it, and a fourth pass
// costs an O(lines) sweep to re-rank candidates whose bounds are already below the best exact gain.
// A partner that turns out to deliver zero MW is dropped from this discharge end's list rather than
// retired fleet-wide, so a different discharge end -- whose difference-of-columns points elsewhere
// -- can still use it.
const PAIR_REFINE: usize = 3;

// Most action magnitudes evaluated per sign during the offline DP sweep. The grid is aligned to
// the SOC grid (see `aligned_actions`), so one evaluation per reachable SOC node plus one for the
// bound finds the inner max exactly; this is the buffer size, not the usual count.
// At MAX_BINS = 97 one full-power step spans about 7.9 grid nodes (the generator ties power to
// capacity, so `mag * rate / dk` is the same 7.9 for every battery on every track), which is 8
// interior breakpoints plus the bound. Ten is that plus slack: if the buffer were too small
// `aligned_actions` would silently fall back to a *uniform* sample of the reachable range, which
// contains no breakpoint of the piecewise-linear objective and so stops finding the inner max
// exactly -- the one property that makes a finer SOC grid worth having.
const DP_ACTION_HALF: usize = 10;
// Candidate actions per SOC bin in the DP sweep: both signs plus idle.
const DP_CANDIDATES: usize = 2 * DP_ACTION_HALF + 1;
// Candidate actions per SOC bin assumed when sizing the SOC grid. The aligned grid usually
// spends about three per sign at these grid sizes against the uniform grid's six; this is left at
// the uniform figure so the alignment does not also change the grid size.
const DP_EVALS_PER_BIN: usize = 12;
// Most action magnitudes evaluated per sign when choosing the action actually played. Same
// reasoning as `DP_ACTION_HALF`, and `GainCurve` carries this many entries per direction.
const ONLINE_ACTION_HALF: usize = 10;
// SOC grid sizing bounds.
const MIN_BINS: usize = 17;
// Nodes in the water-value table's SOC grid.
//
// This is the resolution of the only quantity the policy ever reads. Every action, online and in
// the sweep, is chosen by comparing `price * DT` against the slope of `interp(W, soc_after(u))`,
// and `interp` is piecewise linear, so that slope is a chord of a concave function over one bin.
// The band the comparison has to resolve is the round trip's two transaction fees plus its 9.75%
// efficiency loss -- about $5/MWh at a $50 reference -- and the chord error is set by the bin
// width.
//
// 49 was never chosen against that band; it was the largest grid the fuel budget could seat on the
// capstone track, and the previous iteration raised the budget purely to stop the cap binding
// below it. Halving the bin width halves the chord error in the marginal water value everywhere,
// and it does so on all five tracks at once rather than on the two the budget used to bind.
//
// The fuel comes from deleting the reference-price second opinion (see `build_plan`), which cost
// two full-horizon allocator replays at this same `bins` -- about 69M multiply-adds on capstone
// against the 33M that quadrupling the sweep's work costs there, so the solver is cheaper at 97
// bins than it was at 49.
const MAX_BINS: usize = 97;
// Rough cap on (steps x batteries x bins x action samples) for the DP sweep.
//
// At 6.5e6 this cap bound only the two largest tracks, and it bound the largest one hard: with
// 192 steps and 100 batteries it allowed 28 SOC bins against the 49 every smaller track got.
// That is the one approximation in this solver whose size is set by a fuel constant rather than
// by a modelling choice, and it is worst exactly where the score is worst.
//
// What 28 bins costs is resolution in the *marginal* water value, which is the only thing the
// policy ever reads: every action, online and in the sweep, is chosen by comparing
// `price * DT` against the slope of `interp(W, soc_after(u))`, and `interp` is piecewise linear,
// so that slope is a chord of a concave function over one bin. The generator ties power to
// capacity -- both scale with the same per-battery size factor -- so one full-power step always
// moves about 8.2% of the SOC span, which is 3.95 bins at 49 and only 2.22 at 28. At 28 the
// battery's whole reachable range for the step therefore straddles two or three chords, so the
// charge and discharge thresholds the table is supposed to encode are quantised to two or three
// levels per step of movement, and the band between them -- the round trip's two transaction
// fees plus its 9.75% efficiency loss, about $5/MWh against a capstone diffusive spread of $15 --
// is narrower than one chord's worth of slope error. The table is flattest precisely in the band
// where it has to discriminate.
//
// With `MAX_BINS` now 97 the cap has to clear `steps x batteries x 12 x 97` on the largest track
// for the grid size to be set by the modelling choice rather than by this constant again:
// 192 x 100 x 12 x 97 = 2.23e7. At 2.3e7 every track reaches 97 (capstone 99.8 -> 97,
// dense 166 -> 97, the rest far above), so the cap binds nowhere and the resolution is uniform
// across tracks for the first time.
//
// The sweep is not the dominant plan-time cost even so. It is O(bins x candidates) per battery and
// step, while each full-horizon allocator replay is O(batteries x lines) per step -- 30k
// multiply-adds per pass on capstone against the sweep's 2k -- which is why deleting the two
// replays the reference-price second opinion needed pays for this outright.
const DEFAULT_DP_BUDGET: f64 = 2.3e7;
// Fraction of each line limit the planner is willing to load.
const DEFAULT_FLOW_SAFETY: f64 = 0.999;
// Weight on the analytic jump-mean uplift applied to the price forecast.
const DEFAULT_JUMP_OPTIMISM: f64 = 1.0;
// Weight on the analytic congestion-premium uplift applied to the price forecast.
const DEFAULT_CONG_OPTIMISM: f64 = 1.0;
// Multiplier on the 1/sqrt(m) fleet share of line headroom each battery is assumed to get
// when the DP decides how fast it can cycle. 0 disables the correction.
const DEFAULT_POWER_SHARE: f64 = 1.0;
// Floor on the DP's planning power as a fraction of nameplate, so a battery wedged behind a
// saturated line still gets a value table with usable shape rather than an all-zero one.
const MIN_POWER_FRACTION: f64 = 0.10;
// Floor on the probability the DP assumes a battery is served at all. At 0.10 a battery still
// expects ~10 full-power steps per 96, which is enough to complete a cycle, so the table keeps
// its shape; below that it degenerates into "never trades".
const MIN_SERVICE: f64 = 0.10;
// The split of a measured delivery ratio into "how often" and "how deep" is no longer read off a
// threshold frequency; it is moment-matched. See `service_split` and the decomposition in
// `build_plan`.
// Steps per block in the measured service profile. The replay measures how much of the power a
// battery asks for actually clears the lines; accumulating that per block rather than over the
// whole horizon turns one scalar into a shape in time. Three hours is the right granularity for
// the same reason the earlier per-step headroom profile was the wrong one: what varies
// systematically is the fleet's *desired sign*, which follows the smooth diurnal day-ahead curve
// and so puts every battery in contention at the trough and at the peak and leaves the shoulders
// open, while what varies step to step is which line won the generator's U(-100, 100) noise
// lottery. A block of twelve steps averages the lottery away and keeps the curve.
const SERVICE_BLOCK: usize = 12;
// Half-width, in blocks, of the moving average applied to the profile. One block either side
// pools three hours of asked-for and cleared MW on each side of a block before the ratio is
// taken, which is a volume-weighted smoothing rather than an average of ratios.
const SERVICE_SMOOTH: usize = 1;
// Most action magnitudes evaluated per sign in the plan-time replay that measures delivered
// power. Same aligned grid as the online step, so the magnitude the replay records a battery
// asking for is the one the online step would really ask for.
const CALIB_ACTION_HALF: usize = 10;
// SOC bins used by the calibration sweep, whose only output is one delivery ratio per battery.
// The online tables are swept at the full `bins`; this grid only has to be fine enough that a
// battery's charge-at-night / discharge-at-peak timing comes out right, because the replay
// collapses the whole table into a scalar.
const CAL_BINS: usize = 13;
// Price atoms the DP takes its expectation over: five for the diffusive RT shock and
// one for the jump branch.
const PRICE_ATOMS: usize = 6;
// Abscissae of the symmetric 5-point rule the DP integrates the diffusive RT shock with:
// {-b, -a, 0, +a, +b} with weights {w_b, w_a, w_0, w_a, w_b}.
//
// What is being integrated is
//     f(lambda) = max_u (u * lambda * DT + base_u),
// a maximum of straight lines: convex and piecewise linear in the price, with kinks exactly at
// the battery's charge and discharge thresholds and no curvature anywhere. So the only thing a
// rule has to get right is the kink, and the quantity it has to get right at a kink `k` standard
// deviations out is the option value
//     T(k) = E[max(xi - k, 0)] = phi(k) - k * (1 - Phi(k)).
// An atom rule returns sum_i w_i * max(x_i - k, 0), which is piecewise linear in `k` with a kink
// at each atom, so the number of atoms per side is the number of segments available to track a
// strictly convex T.
//
// The 3-point rule this replaces -- {-r, 0, +r}, weights {1/pi, 1 - 2/pi, 1/pi},
// r = sqrt(pi/2) -- was pinned on mean 0, variance 1 and T(0) = 1/sqrt(2*pi), which is exactly
// right at a kink sitting on the center and nowhere else. With only one segment it is forced to
// leave T(0) with slope -1/pi = -0.318 where the truth leaves it at -(1 - Phi(0)) = -0.5, so it
// overstates the option value at every kink off the center, monotonically in how far off:
// +13.7% at k = 0.3, +23% at k = 0.57, +20% at k = 0.8, and zero past r where T is still
// positive.
//
// Those are the kinks this challenge actually has. A water value equilibrates until the battery
// is near indifference at a typical price, so the discharge threshold sits about
// lambda_da * (1/eta_d - 1) + kappa_tx = $2.9 above the day-ahead price and the charge threshold
// about as far below, against a diffusive spread sigma * lambda_da of $5 at sigma = 0.10 and $15
// at sigma = 0.30. In standard deviations that is k = 0.57 on the baseline track, 0.38 on
// congested, 0.29 on multiday, 0.23 on dense and 0.19 on capstone -- so the old rule overstated
// the option to wait by 23% on the loosest track and by 9% on the tightest, every step, every
// SOC, propagated back through all 96 or 192 rows of the sweep. Overstating it is the mirror of
// the Gauss-Hermite error that preceded it: the table is too patient, holding charge for a spread
// the price distribution will not actually deliver instead of taking the day-ahead cycle that is
// there, and it is worst exactly where the network is loosest and the day-ahead cycle is most
// deliverable.
//
// Two atoms per side give two segments and fix the slope problem. The weights below are pinned on
// the same three quantities -- total mass 1, variance 2*(w_a a^2 + w_b b^2) = 1, and
// T(0) = w_a a + w_b b = 1/sqrt(2*pi) -- which determines them exactly once the two abscissae are
// chosen, leaving the abscissae themselves as the free pair. They are placed at 0.7 and 1.7, which
// straddles the band of kinks above and holds the relative error on T within about 9% over
// k in [0, 0.8] against the old rule's one-sided +14% to +23%, and -- the part that matters over a
// 192-row backward sweep -- makes the error change sign inside that band instead of accumulating
// in one direction.
//
// Five diffusive atoms against three; the jump atom is unchanged. The inner envelope walk makes
// the per-atom cost O(1) rather than O(candidates) (see `sweep_battery`), so this is plan-time
// arithmetic on the sweep only, and nothing in the online step is touched.
const KINK_A: f64 = 0.7;
// Outer abscissa.
const KINK_B: f64 = 1.7;
// Weight on each of the two atoms at +/- KINK_A: (T(0) * b - 1/2) / (a * (b - a)).
const KINK_W_A: f64 = 0.254_574_109_546_336_6;
// Weight on each of the two atoms at +/- KINK_B: (1/2 - T(0) * a) / (b * (b - a)).
const KINK_W_B: f64 = 0.129_847_296_305_292_4;
// Weight on the central atom, 1 - 2 * (w_a + w_b).
const KINK_W_CENTRE: f64 = 0.231_157_188_296_742_0;

// Steps in the look-ahead window of the reference-price value function: three hours at 15
// minutes, which is the window the challenge's own greedy day-ahead baseline averages over.
//
// The three `REF_*` constants and `reference_values` are retained but unreferenced; the selector
// that consumed them is gone (see `build_plan`).
#[allow(dead_code)]
const REF_LOOKAHEAD: usize = 12;
// Floor on how much more the reference-price tables have to earn in the plan-time replay before
// they are played instead of the dynamic program's. The replay's price path is deterministic, so
// it cannot pay the DP for the whole of the option value the DP holds energy for, and the margin
// covers that one-sided bias. The path now carries the congestion-premium and Pareto-jump branches
// as realised events rather than as their means (see `forecast` in `build_plan`), which is most of
// the mass the bias was made of; what is left out is the diffusive dispersion, 70% of which is a
// factor shared by every node.
//
// 1.10 was chosen rather than derived, and it is the only hand-set number in this solver that
// stands in for a bias the solver can measure. The DP's own table says what the bias is worth:
// `W[0][e_0]` is, by construction of the sweep, the expected total profit of the DP policy from
// step zero at the initial SOC under the modelled price *distribution* -- the five diffusive
// atoms, the congestion-premium branch and the Pareto jump branch -- with the measured service
// probabilities already in it. The replay produces the same quantity, in the same dollars, for
// the same tables and the same initial SOCs, with the remaining dispersion collapsed onto its
// mean. So the ratio of the two is the part of that premium the replay path still refuses to
// credit, and `REF_MARGIN` is an estimate of that ratio frozen at 1.10 for every instance.
//
// It is not 1.10 on the tracks that matter. The dispersion still omitted is `sigma * lambda_da`
// wide -- $5/MWh on the baseline track and $15 on capstone against a round-trip band of about $5.
// A threshold policy's profit scales with the price deviation it can trade against, so a path with
// none of that understates what the DP expects to earn, and it understates it hardest on the
// heavy-tailed tracks where the fallback is most likely to be chosen by mistake.
//
// So the margin is measured per instance as `dp_self / dp_profit` and 1.10 is kept as its floor:
// where the DP prices no option value the smooth path does not already deliver, the ratio falls
// to one, the floor binds, and the selection is bit-identical to what it was.
#[allow(dead_code)]
const REF_MARGIN: f64 = 1.10;
// Ceiling on the measured margin. The ratio conflates the option value with the sweep's other
// optimism -- it models the shared network as an independent per-step service coin, where the
// replay runs the real allocator -- and its denominator can be near zero, so the clamp is a guard
// on the estimator rather than a tuned quantity.
#[allow(dead_code)]
const REF_MARGIN_MAX: f64 = 3.0;

#[derive(Serialize, Deserialize)]
pub struct Hyperparameters {
    pub dp_budget: Option<f64>,
    pub flow_safety: Option<f64>,
    pub jump_optimism: Option<f64>,
    pub cong_optimism: Option<f64>,
    pub power_share: Option<f64>,
}

/// Precomputed water-value table + flow sensitivities for one battery.
struct BatteryPlan {
    lo: f64,
    dk: f64,
    cap: f64,
    eta_c: f64,
    eta_d: f64,
    node: usize,
    /// (num_steps + 1) * bins, row-major by time step. Row `num_steps` is all zeros.
    values: Vec<f64>,
    /// d flow[l] / d u_b, i.e. the PTDF column of this battery's node.
    sens: Vec<f64>,
}

struct Plan {
    seed: [u8; 32],
    bins: usize,
    num_steps: usize,
    num_lines: usize,
    flow_safety: f64,
    batteries: Vec<BatteryPlan>,
    /// Line flows of the exogenous injections alone, one row per step. These are the base flows
    /// every allocation starts from and they do not depend on the actions, so they are computed
    /// once instead of once per step.
    exo_flows: Vec<Vec<f64>>,
}

static PLAN: Mutex<Option<Arc<Plan>>> = Mutex::new(None);

pub fn help() {
    println!("energy_arbitrage: per-battery stochastic backward dynamic program on day-ahead");
    println!("prices.");
    println!();
    println!("Offline (once per challenge): for every battery, sweep a backward DP over a");
    println!("discretised SOC grid using the known day-ahead price path at that battery's node,");
    println!("including charge/discharge efficiency, transaction cost and quadratic degradation.");
    println!("The result is a water-value table W[t][soc] = best profit achievable from t onward.");
    println!();
    println!("The DP takes an expectation over the RT price rather than planning against a");
    println!("single forecast path: W[t][e] = E_lambda[ max_u (reward(u, lambda) + W[t+1][e']) ].");
    println!("An RT price is lambda_da * (1 + sigma * xi) with xi standard normal, plus a Pareto");
    println!("jump with probability rho, and both parameters are published in the challenge. The");
    println!("expectation uses six atoms: a symmetric 5-point rule on xi carrying weight");
    println!("1 - rho, and one jump atom at lambda_da * (1 + alpha/(alpha-1)) carrying weight");
    println!("rho. Above the price at which a battery discharges at full power the inner max is");
    println!("linear in lambda, so collapsing the heavy tail onto its conditional mean is exact");
    println!("there. Because the max sits inside the expectation, the table prices the option to");
    println!("wait: energy is worth more than the day-ahead peak implies when a spike may arrive,");
    println!("and empty headroom is worth more when a price collapse may arrive. A deterministic");
    println!("table has neither effect and so sells into the day-ahead peak and misses the");
    println!("spikes, which is where the money is on the high-sigma, heavy-tailed tracks.");
    println!();
    println!("The quadrature rule is chosen for the integrand's kink rather than for its moments.");
    println!("The integrand is max_u of u * lambda * dt + base_u, a maximum of straight lines: it");
    println!("is convex and piecewise linear in the price, with kinks exactly at the battery's");
    println!("charge and discharge thresholds and no curvature anywhere. So the only quantity a");
    println!("rule has to reproduce is the option value at a kink k standard deviations out,");
    println!("T(k) = E[max(xi - k, 0)] = phi(k) - k * (1 - Phi(k)). An atom rule returns");
    println!("sum_i w_i * max(x_i - k, 0), which is piecewise linear in k with a kink at each");
    println!("atom, so the atoms per side are the segments available to track a strictly convex");
    println!("T -- and matching high moments of a function with no curvature buys nothing.");
    println!();
    println!("The rule was a symmetric 3-point one, plus/minus sqrt(pi/2) and zero with weights");
    println!("1/pi, 1 - 2/pi, 1/pi, pinned on mean zero, variance one and T(0) = 1/sqrt(2*pi).");
    println!("That is exact for a kink sitting on the center and nowhere else. With one segment");
    println!("per side it has to leave T(0) with slope -1/pi = -0.318 where the truth leaves it");
    println!("at -(1 - Phi(0)) = -0.5, so it overstates the option value at every kink off the");
    println!("center, and monotonically in how far off: +13.7% at k = 0.3, +23% at k = 0.57,");
    println!("+20% at k = 0.8, and zero past sqrt(pi/2) where T is still positive.");
    println!();
    println!("Those are the kinks this challenge has. A water value equilibrates until the");
    println!("battery is near indifference at a typical price, so the discharge threshold sits");
    println!("about lambda_da * (1/eta_d - 1) + kappa_tx = $2.9 above the day-ahead price and the");
    println!("charge threshold about as far below, against a diffusive spread sigma * lambda_da");
    println!("of $5 at sigma = 0.10 and $15 at sigma = 0.30. In standard deviations that is");
    println!("k = 0.57 on baseline, 0.38 on congested, 0.29 on multiday, 0.23 on dense and 0.19");
    println!("on capstone. So the old rule overstated the option to wait by 23% on the loosest");
    println!("track and by 9% on the tightest, at every step and every SOC, propagated back");
    println!("through all 96 or 192 rows of the sweep. Overstating it is the mirror of the");
    println!("Gauss-Hermite error that preceded it -- that rule understated T(0) by 28% and made");
    println!("the table too eager, buying on spreads that did not cover the round trip. This one");
    println!("makes it too patient: the battery holds charge for a spread the price distribution");
    println!("will not deliver instead of taking the day-ahead cycle that is there. That is worst");
    println!("exactly where the network is loosest and the day-ahead cycle is most deliverable.");
    println!();
    println!("So each side gets a second atom and a second segment. The weights are pinned on the");
    println!("same three quantities -- total mass one, variance 2 * (w_a a^2 + w_b b^2) = 1, and");
    println!("T(0) = w_a a + w_b b = 1/sqrt(2*pi) -- which fixes them exactly once the abscissae");
    println!("are chosen, leaving the abscissae as the free pair. They are placed at 0.7 and 1.7,");
    println!("which straddles the band of kinks above, holds the relative error on T within about");
    println!("9% over k in [0, 0.8] against the old rule's one-sided +14% to +23%, and -- the part");
    println!("that matters over a 192-row backward sweep -- makes the error change sign inside");
    println!("that band instead of accumulating in one direction. Five diffusive atoms against");
    println!("three, with the jump atom unchanged. The envelope walk below makes the per-atom cost");
    println!("O(1) rather than O(candidates), so this is plan-time arithmetic on the sweep alone");
    println!("and the online step is untouched.");
    println!();
    println!("The inner max over actions is solved exactly rather than sampled. Both the sweep");
    println!("and the online step used to score a uniform grid of magnitudes, edge * a/N. That");
    println!("grid misses the only points that can be optimal. A candidate's value is");
    println!("u * price * dt plus the frictions plus the interpolated continuation: revenue and");
    println!("transaction cost are linear in u, the degradation cost is quadratic but worth about");
    println!("1e-4 $ at full power against a revenue slope near 12 $/MW, and the continuation is");
    println!("piecewise linear in the next SOC with breakpoints at the grid nodes, the next SOC");
    println!("being affine in u. So the objective is piecewise linear in u with breakpoints");
    println!("exactly at the actions that land the next SOC on a node, and a piecewise-linear");
    println!("function on an interval is maximised at a breakpoint or an endpoint. Enumerating");
    println!("those actions plus the bound therefore maximises over the whole interval, while a");
    println!("uniform grid generally contains no breakpoint at all -- it understates the inner");
    println!("max, and that understatement propagates back through all 96 or 192 steps of the");
    println!("sweep, and it hands back an action that is not the optimum, which is the action the");
    println!("policy plays and the magnitude the plan-time replay records as asked for.");
    println!();
    println!("The aligned set is also smaller than the uniform grids it replaces. The generator");
    println!("ties power to capacity -- both scale with the same per-battery size factor -- so one");
    println!("step always moves about 8% of the SOC span and the reachable range spans at most");
    println!("four nodes at the largest grid built here. Three to five magnitudes per sign replace");
    println!("twenty online and six offline, so exactness costs less fuel than sampling did. The");
    println!("SOC grid is sized off the old uniform count so the alignment does not also change");
    println!("the number of bins.");
    println!();
    println!("A node incident to a near-loaded line also gets a congestion premium");
    println!("gamma_price * max(Z,0). Its trip probability depends only on the previous step's");
    println!("exogenous flows, which are known for the whole horizon, so P(flagged) is computed");
    println!("per node and step offline. That term is node-specific, so it is what tells the");
    println!("fleet which nodes are worth saving charge for.");
    println!();
    println!("The premium enters as its own branch, not as an uplift. It is zero unless the node");
    println!("is flagged and Z > 0, so it is exactly nothing on most steps and large on a few:");
    println!("at the 85% loading the generator leaves, a node on the worst line is flagged about");
    println!("half the time and then draws about $16/MWh with a tail past $40, against a mean of");
    println!("$4.5. Adding that mean to every atom -- which is what the sweep used to do -- hands");
    println!("the DP a price path that is uniformly slightly better rather than one that is");
    println!("usually ordinary and occasionally excellent, and with the max inside the");
    println!("expectation those are different problems: the smoothed version trades a little");
    println!("harder everywhere, the branched one holds charge into the windows where its node");
    println!("may spike and sells the cycle at once. On the low-volatility tracks the premium is");
    println!("the dominant source of RT upside -- sigma * lambda_da is only about $5/MWh there --");
    println!("so smoothing it away removed most of the reason to time a discharge. Each diffusive");
    println!("atom therefore splits into a no-premium and a premium copy carrying P(flagged)/2,");
    println!("which preserves the mean exactly and costs five extra atoms only on steps where");
    println!("the node is genuinely at risk. The jump atom keeps the mean: it already sits above");
    println!("the full-discharge threshold, where the inner max is linear in price.");
    println!();
    println!("Online (each step): each battery's candidate actions are scored as (realised");
    println!("RT-price profit + W[t+1][next soc]). If the unconstrained fleet plan overloads no");
    println!("line, it is played directly. Otherwise three feasible allocations compete. (1) A");
    println!("sequential greedy serves the most valuable batteries first. (2) The whole fleet");
    println!("plan is shrunk by one uniform factor, which keeps every battery trading instead of");
    println!("spending scarce headroom on the first few and preserves whatever cancellation the");
    println!("fleet's line footprints already have. (3) That shrunk plan is then topped up:");
    println!("batteries are revisited in descending value and each climbs back toward its own");
    println!("optimum as far as the remaining headroom allows. The uniform factor is set by the");
    println!("single tightest line, so without the top-up one bottleneck idles batteries that do");
    println!("not sit behind it and the slack on every other line is discarded; starting the");
    println!("climb from a feasible point rather than from zero also stops the first few");
    println!("batteries taking everything, as they do in the sequential greedy. Whichever scores");
    println!("best under the same one-step lookahead is played. All three are");
    println!("O(batteries x lines); per-step dual iteration on the line multipliers was measured");
    println!("to cost more fuel than the allocation it bought.");
    println!();
    println!("The two value-ordered passes walk the fleet in descending value *density* -- the");
    println!("dollars of one-step gain a battery earns per unit of the scarcest line headroom its");
    println!("action consumes -- not in descending dollar gain. A dollar gain is dominated by");
    println!("battery size: power is NOMINAL_POWER * 3^(h(2r-1)) with r uniform, a 9x spread at");
    println!("h = 1.0, so a battery nine times the size earns nine times the dollars at the same");
    println!("price signal and is served first for that reason alone. Size is not what the lines");
    println!("charge for. What a battery costs the network is |u * ptdf[l][node]| against the");
    println!("headroom line l has left, and both the PTDF column and the headroom are independent");
    println!("of how big the battery is, so two bidders at the same $/MWh can differ by an order");
    println!("of magnitude in how much of the bottleneck they eat per MW -- one sits across it,");
    println!("the other barely moves it -- and the gain order cannot tell them apart. That is the");
    println!("ordinary failure of a knapsack greedy ranked on value rather than on value per unit");
    println!("of the constrained resource, and the resource is scarcest exactly where this solver");
    println!("loses: the generator rescales every step until the worst line sits at 85% of a limit");
    println!("already scaled by 0.4, leaving a 100-battery fleet bidding for about 3 MW on it.");
    println!();
    println!("The key is gain / ||f||_2, where f_l = |u * ptdf[l][node]| / headroom_l is the");
    println!("fraction of line l's remaining headroom the action eats, with the headroom taken in");
    println!("the direction the action pushes the line. Numerator and denominator both scale with");
    println!("|u|, so the ranking is marginal value per MWh over the headroom one MW consumes, and");
    println!("size cancels out of it; that also makes it the right order for the top-up pass,");
    println!("whose batteries climb only part of the way. A battery at the slack bus has an");
    println!("all-zero PTDF column, costs the network nothing and sorts first.");
    println!();
    println!("The norm runs over the whole column rather than its single largest entry. A max says");
    println!("a battery costs the network whatever it does to the line it hurts most and nothing");
    println!("for what it does to every other line, so two bidders that each take half of their");
    println!("worst line's headroom rank identically even when one sits on one line and the other");
    println!("takes half of five at once and blocks five times as much of the fleet behind it. A");
    println!("node's PTDF column has a significant entry on roughly every line of its local mesh,");
    println!("so how many lines a battery really loads scales with the line-to-node ratio -- 1.5 on");
    println!("baseline, congested and multiday against 2.0 on dense and capstone -- and the");
    println!("under-pricing of a broad footprint is worst on exactly the two tracks this solver");
    println!("loses on.");
    println!();
    println!("L2 rather than a plain sum, because a PTDF column on a meshed network has a long");
    println!("tail of small entries and 300 lines at 1% of headroom each would outweigh one line");
    println!("taken at 100%, handing the order to the tail. The L2 norm is dominated by the large");
    println!("entries instead: one line at 0.5 against a tail of 300 at 0.01 gives 0.529 against");
    println!("the max's 0.5, while ten lines at 0.4 give 1.26 against the max's 0.4. It reduces to");
    println!("the max exactly when one line dominates the column, so on a loose track the order is");
    println!("the one the max produced and the change acts only where several lines are scarce at");
    println!("once. One multiply-add per line and one square root per battery on top of the same");
    println!("single O(batteries x lines) pass, only on steps where a line is actually overloaded,");
    println!("and nothing about the candidate actions, the value tables or the feasible set");
    println!("changes -- only the order in which the headroom is handed out. The plan-time replay");
    println!("orders the same way, so the service probability the sweep plans with still describes");
    println!("the policy that is played.");
    println!();
    println!("A fourth allocation trades across the fleet instead of across time. The three");
    println!("above are sub-selections of a single direction -- every battery takes the sign its");
    println!("own price-versus-water-value comparison prefers -- and those signs are strongly");
    println!("correlated, because nodal prices share a diurnal curve and a 0.7-correlated common");
    println!("shock. At night nearly every battery wants to charge and at the peak nearly every");
    println!("one wants to discharge, and correlated injections add in every PTDF column, so on");
    println!("the tight tracks the uniform factor collapses: the fleet asks the grid to move");
    println!("several hundred MW one way through lines whose exogenous flows already sit at 85%");
    println!("of a limit scaled by 0.4. Every battery gets a sliver and no cycle completes.");
    println!();
    println!("Discharging at the fleet's dearest node while charging the same MW at its cheapest");
    println!("escapes that. The pair's aggregate injection is zero and its line cost is");
    println!("|ptdf[l][node_d] - ptdf[l][node_c]| per MW rather than |ptdf[l][node]| per MW, which");
    println!("is small except on the lines separating the two nodes. It is spatial rather than");
    println!("temporal arbitrage, so it is available at every step, including the ones where the");
    println!("time direction is hopeless -- and the spread it trades is widest exactly where this");
    println!("solver loses. The idiosyncratic part of an RT shock is");
    println!("sigma * lambda_da * sqrt(1 - rho_spatial), about $8/MWh on capstone; a flagged node");
    println!("adds a premium averaging $16/MWh; a jump lands on 5% of (node, step) pairs at a");
    println!("Pareto(2.5) multiple of the day-ahead price. The dearest and cheapest of 150 nodes");
    println!("are tens of dollars apart on most steps, against a round trip costing the 9.75%");
    println!("efficiency loss plus two transaction fees, roughly $5/MWh.");
    println!();
    println!("Batteries are sorted once by the marginal value of discharging and the two ends of");
    println!("that order are paired inward. A pair is taken only when the two marginal values sum");
    println!("positive -- that sum carries both water values, so it already charges the transfer");
    println!("for the energy it destroys and credits it for the SOC it moves -- and when the");
    println!("realised price spread covers the round trip in cash, which stops the pairing from");
    println!("trading on a disagreement between two value tables at equal prices. Each pair is");
    println!("scaled as a unit against the remaining line headroom so the two ends stay equal and");
    println!("the cancellation survives; the headroom left over is then spent on the ordinary");
    println!("per-battery optimum, so this candidate degrades into the top-up one when no pair");
    println!("pays. One sort and O(batteries x lines) in total, and it is only built on steps");
    println!("where the unconstrained plan actually overloads a line.");
    println!();
    println!("A pair's size is a joint maximisation over its two ends, not either end's own");
    println!("optimum. It used to be min(u_dis, -u_chg), the two ends' best actions in the");
    println!("prescribed directions -- and on the steps where this allocation is the one the");
    println!("lookahead picks, that is the wrong number by construction. The lines bind there");
    println!("because the fleet's desired signs agree, so every pair has one end moving against");
    println!("its own price-versus-water-value comparison, and that end's best action in the");
    println!("prescribed direction is the smallest magnitude on its candidate list: one SOC grid");
    println!("node of movement, about a quarter of nameplate at 49 bins. The fleet's whole");
    println!("counter-flow volume was capped at a quantity that says nothing about the spread the");
    println!("pair trades. What the transfer is worth per MWh is m_dis + m_chg, which already");
    println!("carries both water values; how much clears is set by the lines.");
    println!();
    println!("So the magnitude maximises the sum of the two ends' gain curves over the interval");
    println!("the lines and the two ends' remaining power leave. Each curve is the one-step gain");
    println!("of moving m MW in one direction, read off the candidate magnitudes already scanned:");
    println!("piecewise linear with breakpoints exactly there, because soc_after is affine in the");
    println!("action and interp is piecewise linear on the SOC grid, and concave, because the");
    println!("recursion preserves concavity. A sum of two such curves is concave piecewise linear");
    println!("with breakpoints only at their union, so its maximum over the interval sits at a");
    println!("breakpoint or at the endpoint and enumerating them is exact -- about a dozen scalar");
    println!("evaluations per pair against the two O(lines) passes the walk already makes. Where a");
    println!("line clips the pair the magnitude is still the line's, unchanged, because the");
    println!("endpoint is the minimum of the two ends' MW and the largest magnitude every line");
    println!("allows; so this acts on the pairs the network leaves room for -- those joining");
    println!("electrically adjacent nodes, whose difference-of-PTDF-columns footprint is near");
    println!("zero, which are the only pairs that clear at all on the tight tracks.");
    println!();
    println!("A pair moves the same MW through both ends, because that is what makes its");
    println!("injections cancel, so whichever end runs out first is retired. The other end's");
    println!("remainder is carried rather than discarded: an end is");
    println!("retired only once the MW it had to give are actually spent, so the fleet's best");
    println!("discharge is matched against successive chargers inward until it is exhausted");
    println!("instead of being capped by whichever battery it happened to be paired with. The");
    println!("sort key is a per-MWh marginal value and carries no information about size, while");
    println!("size spans 3^(h(2r-1)) with r uniform -- 1.25x at h = 0.2 but 9x at h = 1.0 -- so");
    println!("the match is effectively random in size and retiring both ends transacted");
    println!("E[3^min(U1,U2)] = 0.80 of nominal per pair against the 1.21 each end averages");
    println!("alone. A third of the fleet's counter-flow volume was being dropped, and dropped");
    println!("hardest on the two widest fleets, which are the two tracks this solver loses on.");
    println!("The charge end is chosen rather than taken. The walk used to run two indices inward");
    println!("from the ends of one descending marginal-discharge order, which fixes the partner by");
    println!("rank: the charger is whichever battery's discharge was worth least, and nothing");
    println!("about the pair's line cost enters. But the line cost is what decides how much of the");
    println!("pair clears. A pair moves MW against |ptdf[l][node_d] - ptdf[l][node_c]|, near zero");
    println!("for two electrically adjacent nodes and large for two either side of the bottleneck,");
    println!("so the deliverable MW varies by an order of magnitude with the partner. On the tight");
    println!("tracks, where the binding line leaves a couple of MW against a fleet of a hundred,");
    println!("almost every rank-chosen pair is clipped to a sliver and then retired. That is the");
    println!("same failure value density already fixes for the two per-battery allocators, which");
    println!("rank on dollars per unit of scarce headroom rather than on dollars, applied to the");
    println!("one allocator that never got it.");
    println!();
    println!("So each discharge end scans the remaining chargers and keeps the one maximising");
    println!("(m_dis + m_chg) x MW, the pair's whole one-step gain, with the MW bounded by the two");
    println!("ends' remaining power and by the headroom on the tightest line. That is the exact");
    println!("objective under a one-constraint relaxation of the feasible set, and the winner's");
    println!("fraction is then recomputed over every line, so the selection is approximate and the");
    println!("feasibility is not. The tightest line is tracked inside the flow update rather than");
    println!("in a pass of its own.");
    println!();
    println!("That scan used to stop at the sixteen best-priced chargers, and sixteen is the wrong");
    println!("sixteen. The pool was ranked on m_chg, which carries no information about");
    println!("|ptdf[l][node_d] - ptdf[l][node_c]| -- the quantity that decides how much of the pair");
    println!("clears and the only reason a partner is chosen rather than taken. On a 150-node");
    println!("network that pool is sixteen topologically arbitrary nodes, the chance one of them");
    println!("sits near node_d on the line binding this step is small, and joint x MW then gets");
    println!("maximised over a pool whose MW values are uniformly small. The cap also tightened");
    println!("with fleet size -- the whole fleet on baseline's ten batteries, 39% on multiday, 27%");
    println!("on dense, 16% on capstone -- so the search width was inversely proportional to the");
    println!("number of bidders and narrowest on the two tracks where the binding line leaves a");
    println!("couple of MW against a hundred of them. The scan now runs to the economic cutoff");
    println!("instead: corder is descending in m_chg, so it still breaks at the first partner with");
    println!("m_dis + m_chg <= 0 and no later one can pay. That is O(batteries) scalar comparisons");
    println!("per pair against the two O(lines) passes the walk already makes per pair -- 100");
    println!("batteries against 300 lines on capstone -- and it is bit-identical on baseline, where");
    println!("sixteen already covered every reachable partner.");
    println!();
    println!("A clipped pair now retires the partner and keeps the discharge end. A pair is sized");
    println!("at the smaller of the two ends and then scaled by the tightest line over the whole");
    println!("set, so there are three ways it can finish: one end exhausts its MW, or a line clips");
    println!("it short of both. The first two already kept the surviving end and matched it inward");
    println!("against successive chargers. The third retired both, which made a partner that");
    println!("delivered something the only one treated worse than a partner that delivered");
    println!("nothing -- the fraction = 0 branch has always retired the charger alone and re-scanned");
    println!("for the same discharge end, and a clip is that same case with a nonzero first");
    println!("instalment. What binds a pair is |ptdf[l][node_d] - ptdf[l][node_c]| on one line,");
    println!("which is a property of which two nodes it joins: a different charger gives a");
    println!("different difference-of-columns direction, small or opposite in sign on the line that");
    println!("just clipped this one, and the MW it moves varies by an order of magnitude with the");
    println!("partner. That is the same fact that makes the partner worth choosing rather than");
    println!("taking by rank.");
    println!();
    println!("The end that was being dropped is order[lo], the fleet's most valuable remaining");
    println!("sale, still holding MW it is bounded to move, and dropping it hands the step's");
    println!("remaining counter-flow capacity to strictly less valuable dischargers or -- once");
    println!("misses breaks the walk -- to none at all. It did so on exactly the steps where the");
    println!("pairing is the candidate the lookahead picks: a clip means the lines are tight, and");
    println!("tight lines are what collapse the uniform factor the other three candidates rest on.");
    println!("Every iteration still retires a charger or advances lo, so the walk is O(batteries)");
    println!("with twice the constant the double retirement bought, and the economic break on");
    println!("m_dis + m_chg <= 0 holds it well under that. Where no pair is clipped -- a loose");
    println!("network, or ends that exhaust each other exactly -- the branch never fires and the");
    println!("allocation is bit-identical.");
    println!();
    println!("Each iteration still retires at least one end, so the walk stays O(batteries) and");
    println!("the per-step cost rises by at most a factor of two on the one candidate of four");
    println!("that is built only when a line is actually overloaded. Only the end the pair drove");
    println!("against its own");
    println!("optimum is held there; the end that merely got less of the direction it already");
    println!("wanted keeps its solo optimum as the top-up's target and climbs toward it on the");
    println!("lines the pair did not load. Holding both ends hid that shortfall from the top-up");
    println!("entirely -- its delta came out zero -- and the two ends are the fleet's best");
    println!("discharger and best charger, so the two most valuable actions in the step were the");
    println!("frozen ones. The shortfall is set by fleet heterogeneity: power is");
    println!("NOMINAL_POWER * 3^(h(2r-1)) with r uniform, a 1.25x spread at h = 0.2 and a 9x");
    println!("spread at h = 1.0, so on the widest fleets a 75 MW battery is routinely paired with");
    println!("an 8 MW one and then held at 8 MW while the lines it alone loads sit unused. Letting");
    println!("it climb cannot reverse the pair or overshoot: the pair's magnitude is a minimum");
    println!("taken over that direction's own best action, so the segment from the pair to the");
    println!("solo optimum points away from zero. The plan-time replay that measures each");
    println!("battery's service probability allocates by the same rule, so the DP still plans");
    println!("against the policy that is played.");
    println!();
    println!("The sweep plans with the power the network allows, not nameplate power. The");
    println!("instance generator rescales exogenous injections until the worst line sits at 85%");
    println!("of its limit, so a battery's usable headroom is a fraction of one line limit shared");
    println!("with the fleet. A table built on nameplate power prices energy as if the battery");
    println!("could idle all morning and still slam a full cycle into the evening peak, so it");
    println!("waits -- and the allocator cannot deliver the slam, so the cycle never happens. The");
    println!("online step still uses the true action bounds: this changes how energy is valued,");
    println!("not what may be played.");
    println!();
    println!("And the network's effect on a battery is intermittent denial, not a derate. The");
    println!("allocator walks the fleet in descending value and gives each battery as much as the");
    println!("remaining headroom allows, so on a tight track a battery is denied outright on most");
    println!("steps and served at or near full power on a few. A derated table instead plans a");
    println!("slow steady cycle -- a little charge every step through the night, a little");
    println!("discharge every step through the peak -- which is an action no allocator ever");
    println!("delivers. Because the max sits inside the expectation the two are different");
    println!("problems, for the same reason the congestion premium had to enter as a branch");
    println!("rather than as an uplift. So the measured delivery ratio r is spent as a service");
    println!("probability at true nameplate power:");
    println!("W[t][e] = r * E[max_u (reward + W[t+1][e'])] + (1 - r) * W[t+1][e]. Expected");
    println!("throughput per step is unchanged at r x nameplate, so this is a mean-preserving");
    println!("refinement of the same measurement, but the action it plans is the full-power one a");
    println!("value-ordered greedy actually hands its highest bidder, and the SOC it holds is");
    println!("positioned to use it. It also makes the DP's transition model the one the policy");
    println!("really plays: online actions are bounded by nameplate and SOC, never by a derate.");
    println!("Where the lines never bind r is one and the service branch vanishes.");
    println!();
    println!("That probability is measured per direction, because one scalar pools two ratios");
    println!("taken on disjoint sets of steps. A battery asks to charge overnight and at price");
    println!("collapses and asks to discharge at the diurnal peak and into spikes, and the rest");
    println!("of the fleet is asking for the same sign at the same time -- nodal prices share a");
    println!("diurnal curve and a 0.7-correlated common shock. So the grid is being pushed one");
    println!("way on a battery's charge steps and the other way on its discharge steps, while");
    println!("the battery's own PTDF column keeps a fixed sign: its access to the two directions");
    println!("is two different numbers. Averaging them over-values whichever direction the");
    println!("network actually grants and under-values the one it denies, inside a value function");
    println!("whose entire job is deciding which of the two to hold out for. A battery that can");
    println!("charge freely but can rarely clear a discharge should fill early and sell only into");
    println!("the windows it gets, and the pooled scalar cannot say that.");
    println!();
    println!("The two grants are taken as independent, so each step falls into one of four");
    println!("regimes -- both directions, discharge only, charge only, neither -- with idle always");
    println!("available, and W is the expectation over them of the max within each regime's");
    println!("action set. The best discharge and the best charge come out of the same single scan");
    println!("the sweep already makes over the candidates, so the split costs a few comparisons");
    println!("per price atom and no extra pass. Setting both ratios equal reproduces the");
    println!("single-scalar model exactly wherever the inner optimum is one-directional, which is");
    println!("every price but a measure-zero set: the regime that grants only the unwanted");
    println!("direction contributes the idle value, which is what a flat denial contributes. The");
    println!("whole effect of the change is therefore the asymmetry, not the regime bookkeeping.");
    println!();
    println!("The discharge grant is also resolved across the step's price atoms rather than");
    println!("spent flat on all of them. A flat probability makes the grant independent of the");
    println!("price, and the allocator it is a model of is the opposite of independent: it walks");
    println!("the fleet in descending marginal value and serves bidders in order until the");
    println!("headroom runs out, so whether this battery clears is a question about its rank, and");
    println!("rank is set by how its price compares with the rest of the fleet's at that instant");
    println!("rather than by its price in absolute terms. The five diffusive atoms carry almost");
    println!("no rank information: an RT shock is sqrt(rho_sp) * z_t + sqrt(1 - rho_sp) * eps_i");
    println!("with rho_sp = 0.70, on top of a day-ahead curve whose diurnal base term is shared by");
    println!("every node, so when this battery sits at the high diffusive atom so does most of the");
    println!("fleet and they are all bidding to discharge at once. The other two atoms are");
    println!("node-specific and large -- a congestion premium is gamma_price * max(Z, 0) fired by a");
    println!("lagged test on the node's own incident lines, averaging $16/MWh against an");
    println!("idiosyncratic diffusive spread sigma * lambda_da * sqrt(1 - rho_sp) of about $7/MWh");
    println!("on capstone, and a jump multiplies one node's price by a Pareto draw -- so either one");
    println!("puts this battery at the front of the queue.");
    println!();
    println!("So the grant is split across a priority class instead of averaged over it. If the");
    println!("allocator clears q_d * m bidders per step and pi * m of them have an idiosyncratic");
    println!("event, then conditional on having one this battery competes only inside that class");
    println!("and clears with probability min(1, q_d / pi); what the class leaves is what the");
    println!("ordinary steps get, (q_d - pi * q_hi) / (1 - pi). That is the same queue model the");
    println!("measurement came out of, read at the resolution of the price atom, and");
    println!("pi * q_hi + (1 - pi) * q_lo = q_d, so expected throughput is exactly what was");
    println!("measured. What moves is the *relative* worth of the two: a flat q_d scales the whole");
    println!("opportunity down uniformly and leaves the undeliverable fleet-wide diurnal cycle");
    println!("looking as good as the deliverable node-specific spike, while the per-atom split says");
    println!("the cycle is the part the lines take away and the spike is the part they do not. That");
    println!("is the difference between spending a cycle into the peak alongside the whole fleet");
    println!("and holding it for the window where this node is alone in wanting to sell.");
    println!();
    println!("Both ends of the gate are identities. An unbinding network gives q_d = 1, hence");
    println!("q_hi = q_lo = 1 and the unconstrained recursion bit for bit. A node with no jump");
    println!("branch and no premium risk gives pi = 0, hence q_hi = q_lo = q_d, the flat model.");
    println!("Only the charge direction stays flat: every idiosyncratic price event the challenge");
    println!("generates is upward -- the premium is gamma_price * max(Z, 0) >= 0 and a Pareto jump");
    println!("is positive -- so there is no atom on which a battery outbids the fleet to *buy*.");
    println!("The regime weights are folded into each atom's weight once per step, outside the SOC");
    println!("loop, so the innermost arithmetic costs exactly what it did: at most eleven atoms");
    println!("against up to 49 bins, and nothing in the online step is touched.");
    println!();
    println!("That ratio is measured, not assumed. A first sweep uses the analytic bound");
    println!("headroom_l / |ptdf[l][node]| over the tightest line, times a 1/sqrt(m) guess at the");
    println!("fleet's sharing of it. Nothing in that chain knows whether the lines ever actually");
    println!("bind, so on a track with nominal limits and a sparse fleet it derates the table for");
    println!("a constraint that never fires, and the battery sits out intraday swings it could");
    println!("have taken at full power. So the allocator is then replayed once at plan time");
    println!("against the exogenous injections and PTDFs -- known exactly for the whole horizon --");
    println!("with the day-ahead path plus the expected congestion premium standing in for price,");
    println!("and each battery's delivery ratio (power cleared / power asked for) is recorded.");
    println!("That ratio is the service probability the second sweep plans with. Where the");
    println!("network never binds it is one and the correction disappears; where it binds it");
    println!("reads the fleet's sharing out of the allocator's own behaviour, including the");
    println!("cancellation between batteries that a per-line bound cannot see.");
    println!();
    println!("The replay runs the whole allocator, all four candidates picked by the same");
    println!("lookahead. It used to replay only shrink-then-top-up, which is the weakest of the");
    println!("four exactly where the measurement matters: when the fleet's desired signs are");
    println!("correlated the uniform factor collapses and the real policy switches to the");
    println!("counter-flow pairing, which clears a pair at the difference of two PTDF columns");
    println!("rather than the sum. So the measured ratio was biased low wherever pairing wins,");
    println!("and that ratio is the service probability the final sweep plans with -- the table");
    println!("told batteries on the tight tracks that the network would hardly ever let them act,");
    println!("so they held a middling SOC instead of positioning for the windows the pairing");
    println!("opens. The DP's transition model now describes the policy that is actually played.");
    println!("Where no line is overloaded the replay plays the unconstrained optimum, the ratio is");
    println!("one, and nothing changes. The replay is");
    println!("funded by caching the exogenous line flows for the horizon: the premium sweep, the");
    println!("headroom sweep, the replay and every online step all used to recompute the same");
    println!("O(lines x nodes) product.");
    println!();
    println!("The table the replay reads is deliberately coarse, and that sets the plan-time cost");
    println!("structure. The sweep is a leading plan-time term --");
    println!("batteries x steps x bins x atoms x candidates -- and it used to run twice at full");
    println!("resolution: once with the analytic");
    println!("power bound, then again for every battery whose measured service probability");
    println!("disagreed with that bound, which on the tight tracks is nearly all of them. The");
    println!("first of those two sweeps is never the table the policy plays. Its only consumer is");
    println!("the replay, and the replay reduces a battery's entire value table to one number: the");
    println!("fraction of the power it asked for that cleared the lines, summed over the whole");
    println!("horizon. That number does not need the SOC grid resolved to 2% of span. The action");
    println!("set is exact at any grid size -- the aligned magnitudes are the breakpoints of a");
    println!("piecewise-linear objective, so the inner max is exact on 13 nodes as on 49 -- so a");
    println!("coarse grid costs resolution in the value function's shape, i.e. the timing of the");
    println!("cycle, which a 96- or 192-step sum averages over. So the calibration sweep runs on");
    println!("13 bins, with the price model, the planning power and the four-way allocator it");
    println!("replays all left exactly as they were, which keeps the measured ratio comparable to");
    println!("the one the full-resolution pass produced; then every battery gets exactly one");
    println!("full-resolution sweep with that ratio as its service probability, instead of between");
    println!("one and two. Plan-time arithmetic only -- the online step is untouched.");
    println!();
    println!("The SOC grid the policy plays on is sized by a fuel cap, and that cap used to bind");
    println!("hardest on the largest track: 192 steps x 100 batteries left 28 bins against the 49");
    println!("every smaller track got. Resolution there is resolution in the *marginal* water");
    println!("value, which is the only thing any action is ever chosen by: interp is piecewise");
    println!("linear, so the slope the optimiser compares price * DT against is a chord of a");
    println!("concave function over one bin. Power and capacity scale together in the generator,");
    println!("so one full-power step always moves about 8.2% of the SOC span -- 3.95 bins at 49,");
    println!("2.22 at 28 -- and at 28 the charge and discharge thresholds the table exists to");
    println!("encode are quantised to two or three chords per step of movement, which is coarser");
    println!("than the round-trip band between them. The cap is now set so every track reaches the");
    println!("same 49: the three already there are bit-identical and only dense (47) and capstone");
    println!("(28) move. The sweep is not the dominant cost on those two -- the three full-horizon");
    println!("allocator replays and the online rollout are each O(batteries x lines) per step");
    println!("against the sweep's O(bins x candidates) -- so the extra resolution is cheap there.");
    println!();
    println!("That service probability now carries a shape in time as well as a level. The replay");
    println!("accumulates asked-for and cleared MW per three-hour block, smoothed one block either");
    println!("side before the division so the pooling is weighted by the MW at stake rather than");
    println!("averaging ratios off very different volumes, and the resulting profile is rescaled to");
    println!("the horizon ratio so only its placement in time is new. The reason the shape exists");
    println!("is structural: nodal day-ahead prices share one diurnal base curve, so the whole");
    println!("fleet wants to charge in the same hours and discharge in the same hours, and");
    println!("correlated injections add in every PTDF column. Contention therefore peaks at exactly");
    println!("the trough and the peak -- the windows a flat profile aims every battery at -- and");
    println!("the shoulders either side are comparatively open. A flat table plans a cycle it");
    println!("cannot execute: hold through the morning, sell into the peak, get denied with most of");
    println!("the fleet, and watch the shoulder hours where the sale would have cleared go by still");
    println!("holding. The measured shape makes the same table start the discharge early and finish");
    println!("it late, which is what the allocator actually grants. Where no line binds every block");
    println!("measures one, the rescale is the identity and the table is unchanged.");
    println!();
    println!("Three hours is the granularity, not one step, and that is the whole difference from");
    println!("the attempt described next.");
    println!("An earlier version scaled the planning power step by step by the worst line's per-step headroom over");
    println!("the fleet's RMS loading of that line, on the theory that the exogenous injections");
    println!("are diurnal sinusoids and so leave the grid open overnight and saturated at the load");
    println!("peaks. The generator does not work that way: each nodal injection is");
    println!("200 * (U(0,1) - 0.5) + pattern + noise, so an i.i.d. uniform draw on [-100, 100] MW");
    println!("per node and step, standard deviation about 58 MW, dominates the diurnal pattern");
    println!("term's 17 MW. The generator also rescales every timestep until the worst line sits");
    println!("at exactly 85% of its limit, so the tightest line's headroom barely moves in time");
    println!("and what actually varies is which line is tightest -- a draw from the per-step");
    println!("noise, not a property of the hour. Normalised to mean one and clamped into a 16x");
    println!("band, that quantity multiplied the DP's planning power by an essentially arbitrary");
    println!("factor at every step and told the sweep to time its cycle around whichever line won");
    println!("a noise lottery. It lost score on all five tracks simultaneously, which is what an");
    println!("injected-noise regression looks like rather than a tradeoff, so the planning power");
    println!("is flat in time and only its per-battery level is calibrated. The service profile");
    println!("above survives that objection because a twelve-step block averages the lottery away");
    println!("while leaving the fleet's diurnal sign agreement, which is what it measures, intact.");
    println!();
    println!("The sweep's tables used to be checked against a second set built from a reference");
    println!("price -- W[t][e] = r_t * e, linear in stored energy, with r_t the mean forecast price");
    println!("over the next three hours at that battery's node, which is the challenge's own greedy");
    println!("day-ahead threshold rule written as a water value -- by replaying both through the");
    println!("whole allocator over the full horizon and playing whichever earned more. That");
    println!("comparison is gone: forcing the reference tables on every track instead of letting the");
    println!("replay choose moved all five scores a long way (mean -39.4% against +46.8%, loosest");
    println!("track -116.6%), and a track where the replay had already selected them could not have");
    println!("moved at all, so the sweep's tables were being chosen everywhere. The two full-horizon");
    println!("replays that reached that conclusion cost about 35M multiply-adds each on the capstone");
    println!("track against the sweep's own 19M, and that fuel now buys SOC-grid resolution instead:");
    println!("MAX_BINS is 97 rather than 49, so the chord error in the marginal water value -- the");
    println!("only quantity any action in this solver is chosen by -- is halved on every track.");
    println!();
    println!("The remaining replay's price path carries the two idiosyncratic branches as events");
    println!("rather than as their means. The path used to be the day-ahead curve plus the mean");
    println!("congestion premium and the mean jump multiplier, so it never spiked. The premium is");
    println!("zero");
    println!("on most steps and about $16/MWh on a few, a jump multiplies one node's price by a");
    println!("Pareto draw on rho_jump of its steps, and with a max inside the expectation a");
    println!("uniformly-slightly-better path is a different problem from a usually-ordinary,");
    println!("occasionally-excellent one -- the same argument the sweep's own atoms already refuse");
    println!("to smooth the premium on.");
    println!();
    println!("The same path is what the delivery ratio is measured on, and there the smoothing was");
    println!("biased in one direction. Contention is how much of the fleet wants the same sign at");
    println!("the same instant, and on a path whose only variation is the diurnal day-ahead curve");
    println!("shared by every node, the answer is all of it: every battery charges in the same hours");
    println!("and discharges in the same hours, so the measured ratio understated the access a real");
    println!("rollout grants, and that ratio is the service probability every final table is derated");
    println!("with. A premium or a jump is the one thing in this challenge that moves a single node");
    println!("against the fleet, so realising them is what puts cross-sectional spread -- and the");
    println!("counter-flow pairing that trades on it -- into the measurement.");
    println!();
    println!("Placement is Bresenham-stratified rather than sampled: one accumulator per node");
    println!("carries the branch probability and fires when it crosses one, so the horizon's event");
    println!("count is the expected count to within one event, events land where the probability");
    println!("actually is, and no RNG enters a plan that must be reproducible across calls. The");
    println!("offsets are golden-ratio and sqrt(2) fractional sequences in the node index, which is");
    println!("what decorrelates the firing across nodes -- a path that fires every node on the same");
    println!("step has no cross-section. It is mean-preserving per node by construction: the");
    println!("premium branch's probability times its conditional size is the mean premium exactly,");
    println!("and the jump atom's size times rho_jump is the jump multiplier, so the level of the");
    println!("path is unchanged and only the placement of that mass in time and across nodes is new.");
    println!("rho_jump = 0 with no premium risk leaves the path bit-identical.");
    println!();
    println!("What follows describes reference_values, which is retained but no longer invoked.");
    println!();
    println!("That reference table is tapered to zero over the final drain window, because a");
    println!("window mean has no terminal boundary and it is the one half of the menu that is");
    println!("missing one. The sweep gets its boundary for free -- row num_steps is zero, so its");
    println!("last rows price stored energy at what is left to sell it for and the battery is");
    println!("driven to its floor by the end. The window mean instead prices a full battery at");
    println!("$50/MWh at the very last step, where there is no step after it to sell into, so the");
    println!("threshold rule it encodes has no reason to empty: it idles whenever the realised");
    println!("price sits inside its $5/MWh band right up to the horizon, and it will still buy on");
    println!("the final steps at any price under r * eta_c - kappa_tx.");
    println!();
    println!("What that leaves behind is not a rounding error, because every battery is handed a");
    println!("free endowment: soc_initial is 0.50 * cap against a floor of 0.10 * cap, so");
    println!("0.40 * cap MWh per battery is energy the episode starts with and no one paid for.");
    println!("Selling it at any positive price is pure profit -- about $2.3k per battery at a $50");
    println!("mean on the wide fleets -- and on the two tracks with the largest fleets and the");
    println!("largest capacities that is of the order of the whole deficit carried here. Those are");
    println!("also the two tracks the reference table is most likely to be playing on: the replay");
    println!("that chooses between the two runs on a deterministic day-ahead path with no shocks");
    println!("and no spikes, so it cannot pay the sweep for the option value the sweep holds");
    println!("energy for, and the sweep's thresholds are widest relative to that path exactly");
    println!("where volatility is highest.");
    println!();
    println!("The taper is that boundary written in the only form a linear table can hold. The");
    println!("marginal MWh of store is realisable only if there is time left to push it out: with");
    println!("rem = num_steps - t steps left at most rem * p_d * DT / eta_d MWh of store can be");
    println!("delivered, so the span is fully deliverable only while rem is at least");
    println!("drain_steps = span / (p_d * DT / eta_d). Scaling the window mean by");
    println!("min(1, rem / drain_steps) keeps the table linear in stored energy -- hence interp");
    println!("exact on it and the threshold reading intact -- while sending both thresholds to");
    println!("zero as the horizon arrives: the discharge threshold slope / eta_d + kappa_tx falls");
    println!("below any plausible price so the battery dumps at full power, and the charge");
    println!("threshold slope * eta_c - kappa_tx falls below zero so it stops buying energy it can");
    println!("never sell. The generator ties power to capacity, so drain_steps is the same ~12");
    println!("steps for every battery whatever its size: the taper is the identity on all but the");
    println!("last three hours of a 96- or 192-step horizon, and the identity everywhere for a");
    println!("battery with no discharge power. Nothing in the sweep, the allocator or the online");
    println!("step is touched, and where the replay keeps choosing the sweep nothing changes.");
    println!();
    println!("The sweep's expectation is taken by walking an upper envelope once instead of");
    println!("rescanning every candidate at every price atom. The sweep is the dominant");
    println!("plan-time cost -- batteries x steps x bins x atoms x candidates, and the two tracks");
    println!("this solver loses on are precisely the two largest, 60 batteries x 200 lines and");
    println!("100 x 300, against 40 x 120 and 20 x 60 on the two it wins by over 100% -- and its");
    println!("innermost factor was redundant work. A candidate's value is");
    println!("u * price * dt + base, a straight line in the price with slope u * dt, so each");
    println!("direction's maximum over its candidates is the upper envelope of a set of lines");
    println!("whose slopes `aligned_actions` already emits in order. Rebuilding that maximum from");
    println!("scratch at each of the up to eleven atoms recomputes an envelope that did not");
    println!("change.");
    println!();
    println!("Two facts make a single forward walk exact. The maximising rank is non-decreasing");
    println!("in the price: if a larger action is at least as good as a smaller one at some");
    println!("price, then at any higher price its lead has grown by the price difference times");
    println!("dt times the action difference, so the optimum never moves back down. That is an");
    println!("exchange argument needing nothing but sorted slopes, which is why the atoms are now");
    println!("sorted into increasing price -- they are generated as (low, low + premium, mid,");
    println!("mid + premium, high, high + premium, jump) and a $16/MWh premium outruns the");
    println!("diffusive spacing sigma * lambda_da on the low-volatility tracks, so the order was");
    println!("not already monotone. And at a fixed price the value is unimodal in the rank, so");
    println!("climbing until the next candidate is worse lands on the maximum and not on a local");
    println!("one: the value is linear in u, plus a transaction cost linear on each side of zero,");
    println!("plus a degradation cost -kappa_deg (dt/cap)^2 u^2 that is concave, plus the");
    println!("continuation, a concave function of a quantity affine in u. The continuation is");
    println!("concave because the recursion preserves concavity -- jointly concave reward, convex");
    println!("feasible set, so each regime's inner maximum is concave in SOC and the regime");
    println!("probabilities are a positive combination -- and linear interpolation of a concave");
    println!("function sampled on the grid is concave piecewise linear.");
    println!();
    println!("So the two ranks advance at most `candidates` steps in total across all the atoms");
    println!("rather than `candidates` steps per atom, and the sweep loses its last factor. The");
    println!("values are identical; only the summation order over atoms differs. Nothing in the");
    println!("price model, the planning power, the service probability, the allocator or the");
    println!("online step is touched -- this is the same table computed for less fuel, which is");
    println!("the one resource every previous iteration spent more of rather than less.");
    println!();
    println!("The replay measures how often the network grants a battery as well as how much it");
    println!("grants, and the sweep is given both. The delivery ratio r -- cleared MW over");
    println!("asked-for MW across the horizon -- is a mean, and every version of this solver has");
    println!("spent it as one of two distributions with that mean. Read as a derate it says the");
    println!("battery always moves r x nameplate, so the sweep plans a slow steady cycle. Read as");
    println!("a service probability at nameplate power it says the battery moves everything it");
    println!("asked for on a fraction r of the steps and nothing on the rest. Both were shipped");
    println!("globally and both lost, because r alone cannot tell them apart -- and because the");
    println!("allocator is not uniformly one or the other: the value-ordered greedy denies");
    println!("outright, while the uniform shrink and the top-up shave every battery a little, and");
    println!("which of the four the lookahead picks varies by instance and by step.");
    println!();
    println!("So the replay accumulates the second moment of the per-step delivered fraction");
    println!("phi = cleared / asked alongside its mean, on the same ask-weighting. Under the");
    println!("two-point model where phi is either zero or `depth`, with probability q of the");
    println!("nonzero branch,");
    println!("E[phi] = q * depth and E[phi^2] = q * depth^2, so depth = E[phi^2] / E[phi] and");
    println!("q = E[phi] / depth. The product is exactly the measured ratio, so expected throughput");
    println!("per step is unchanged and only its shape moves. All three corners are exact: an");
    println!("unbinding network gives phi = 1 every step, hence q = depth = 1 and a table");
    println!("bit-identical to one swept with no service model; all-or-nothing at rate r gives");
    println!("E[phi^2] = E[phi] = r, hence depth = 1 and the intermittent-grant model; a uniform");
    println!("shave to r gives E[phi^2] = r^2, hence depth = r and q = 1, the derate. The threshold");
    println!("frequency this replaces had the last of those backwards -- a shave shallower than half");
    println!("read as intermittent -- which is the regime the tight tracks are actually in, because");
    println!("the allocation the lookahead picks there is usually the uniform shrink. Two sums per");
    println!("battery and direction in a replay that already runs; the sweep, the allocator and the");
    println!("online step are untouched.");
    println!();
    println!("The size of that priority class is measured rather than inferred from the node's own");
    println!("event rate. The two are the same number only if the idiosyncratic events are");
    println!("independent across nodes, and the congestion premium is not: the generator draws");
    println!("zeta = max(z', 0) once per step, outside the node loop, and adds gamma_price * zeta to");
    println!("every flagged node, while the flags themselves come from one exogenous flow vector");
    println!("tested line by line. So when a battery's premium fires, every other flagged battery's");
    println!("fires with it, and the class it joins is the whole flagged share of the fleet rather");
    println!("than its own share. Sizing the class from the node's own probability therefore");
    println!("overstated the clearing rate on exactly the atoms the table holds charge for -- a node");
    println!("at P(flagged) = 0.9 came out with pi = 0.47 and claimed to clear at q_d / 0.47, twice");
    println!("the rate the replay measured, at every premium window of the horizon. With the max");
    println!("inside the expectation that is the option value the sweep prices waiting for, so the");
    println!("battery held its cycle through deliverable shoulder hours for a window it will be");
    println!("rationed in alongside everyone else.");
    println!();
    println!("The flagged share of the fleet is already determined by the published exogenous path,");
    println!("so it is one O(steps x batteries) pass at plan time. Mean preservation moves with it:");
    println!("how often this node is in the class is still its own event probability, the clearing");
    println!("rate inside the class is the fleet's, and the ordinary atoms absorb the difference, so");
    println!("pi * q_hi + (1 - pi) * q_lo = q_d still holds and expected throughput per step is the");
    println!("measured number unchanged. Where the fleet is no more contended than the node itself");
    println!("-- a node uniquely exposed on an otherwise open grid -- the class is the node's own and");
    println!("the table is bit-identical; q_d = 1 still gives the unconstrained recursion exactly.");
    println!();
    println!("The two places that stand one number in for the real-time price now use the");
    println!("expected real-time price rather than the day-ahead one. The replay that measures");
    println!("delivery and compares the two value functions, and the reference table's window");
    println!("mean, both priced a step at day-ahead plus the mean congestion premium. That");
    println!("premium is in because it is node-specific and worth up to $4.5/MWh -- and the jump");
    println!("branch, which is the same size, was left out. An RT price is da * (1 + sigma * xi),");
    println!("plus gamma_price * max(Z, 0) when the node is flagged, plus da * Pareto(alpha) with");
    println!("probability rho_jump on top of both, and that last term has mean");
    println!("rho_jump * alpha/(alpha - 1) * da: $0.67/MWh at the baseline track's (0.01, 4.0) but");
    println!("$3.18 on dense and $4.17 on capstone, against a round-trip band of about $5/MWh.");
    println!("So the forecast understated the price a battery will see by most of the width of the");
    println!("band it trades on, and it understated it hardest where the tails are heaviest.");
    println!();
    println!("For the reference table the error is one-directional. That table is W[t][e] = r_t e,");
    println!("a threshold rule charging while price < r * eta_c - kappa_tx and discharging while");
    println!("price > r / eta_d + kappa_tx, so both thresholds are proportional to r. Understating");
    println!("r by 8% while the realised prices it is compared against are 8% higher in the mean");
    println!("puts the battery on the sell side of the band at every step: it empties the");
    println!("0.40 * cap MWh endowment early, stops buying, and sits at its floor with no cycle");
    println!("left to trade. The sweep already integrates the jump properly -- its sixth atom sits");
    println!("at da * (1 + jump_optimism * pareto_mean) with weight rho_jump and the five diffusive");
    println!("atoms are mean-preserving -- so the multiplier applied here is exactly the mean of");
    println!("its own atoms, and all three parts of the plan now agree on what the expected");
    println!("real-time price is. It is one scalar computed once; rho_jump = 0 leaves it at one and");
    println!("every table is bit-identical.");
    println!();
    println!("Hyperparameters: dp_budget, flow_safety, jump_optimism, cong_optimism, power_share.");
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let (dp_budget, flow_safety, jump_optimism, cong_optimism, power_share) =
        read_hyperparameters(hyperparameters);
    let plan = Arc::new(build_plan(
        challenge,
        dp_budget,
        flow_safety,
        jump_optimism,
        cong_optimism,
        power_share,
    ));
    store_plan(plan);
    let solution = challenge.grid_optimize(&policy)?;
    save_solution(&solution)?;
    Ok(())
}

pub fn policy(challenge: &Challenge, state: &State) -> Result<Vec<f64>> {
    let num_batteries = challenge.batteries.len();
    if num_batteries == 0 {
        return Ok(Vec::new());
    }
    if state.action_bounds.len() != num_batteries || state.socs.len() != num_batteries {
        return Err(anyhow!("state does not describe {} batteries", num_batteries));
    }

    let plan = get_plan(challenge);
    let bins = plan.bins;
    // Continuation values live one step ahead; the terminal row is all zeros.
    let next_row_start = (state.time_step + 1).min(plan.num_steps) * bins;

    // --- Step 1: per-battery candidate actions.
    // Every candidate's value splits as `u * price * DT + base`, where `base` collects the
    // frictions and the continuation value and so does not depend on the price. Caching `base`
    // lets the relaxation below re-solve each battery under a shifted price for a few
    // multiply-adds instead of a fresh value-table interpolation.
    let mut cand_u: Vec<Vec<f64>> = Vec::with_capacity(num_batteries);
    let mut cand_base: Vec<Vec<f64>> = Vec::with_capacity(num_batteries);
    let mut price_at = vec![0.0f64; num_batteries];
    for b in 0..num_batteries {
        let bp = &plan.batteries[b];
        let row = &bp.values[next_row_start..next_row_start + bins];
        let top = bp.lo + bp.dk * (bins - 1) as f64;
        let soc = state.socs[b];
        let (lower, upper) = state.action_bounds[b];
        price_at[b] = state.rt_prices[bp.node];

        let mut us: Vec<f64> = Vec::with_capacity(2 * ONLINE_ACTION_HALF + 1);
        let mut bs: Vec<f64> = Vec::with_capacity(2 * ONLINE_ACTION_HALF + 1);
        us.push(0.0);
        bs.push(interp(row, soc, bp.lo, bp.dk, bins));
        let mut buf = [0.0f64; ONLINE_ACTION_HALF];
        for side in 0..2usize {
            let edge = if side == 0 { lower } else { upper };
            if edge == 0.0 {
                continue;
            }
            let na = aligned_actions(
                &mut buf, edge, soc, bp.lo, bp.dk, bins, bp.eta_c, bp.eta_d,
            );
            for i in 0..na {
                let u = buf[i];
                let soc_next = soc_after(bp.eta_c, bp.eta_d, u, soc, bp.lo, top);
                let abs_u = u.abs();
                let deg_base = (abs_u * DT) / bp.cap;
                let base = -KAPPA_TX * abs_u * DT - KAPPA_DEG * deg_base * deg_base
                    + interp(row, soc_next, bp.lo, bp.dk, bins);
                us.push(u);
                bs.push(base);
            }
        }
        cand_u.push(us);
        cand_base.push(bs);
    }

    let mut desired = vec![0.0f64; num_batteries];
    let mut gain = vec![0.0f64; num_batteries];
    // Best action in each direction and its value per MWh of throughput. The per-MWh form is
    // what makes two batteries comparable: the pairing below moves the same MW in and out, so
    // the pair pays iff the two marginal values sum to something positive.
    let mut u_dis = vec![0.0f64; num_batteries];
    let mut u_chg = vec![0.0f64; num_batteries];
    let mut m_dis = vec![f64::NEG_INFINITY; num_batteries];
    let mut m_chg = vec![f64::NEG_INFINITY; num_batteries];
    // The whole gain curve in each direction, not just its best point. The counter-flow pairing
    // sizes a pair by maximising the two ends' joint gain over these, which is the only way an end
    // moving against its own temporal optimum can go deeper than one SOC grid node; see
    // `GainCurve` and `best_pair_mw`.
    let mut dis_curve = vec![GainCurve::empty(); num_batteries];
    let mut chg_curve = vec![GainCurve::empty(); num_batteries];
    for b in 0..num_batteries {
        let idle = cand_base[b][0];
        let (u, value, ud, vd, uc, vc) = pick_best_signed(&cand_u[b], &cand_base[b], price_at[b]);
        desired[b] = u;
        gain[b] = value - idle;
        if ud > 0.0 {
            u_dis[b] = ud;
            m_dis[b] = (vd - idle) / (ud * DT);
        }
        if uc < 0.0 {
            u_chg[b] = uc;
            m_chg[b] = (vc - idle) / (-uc * DT);
        }
        // Index 0 is the idle action, whose gain over idle is zero and which the curve carries
        // implicitly.
        let us = &cand_u[b];
        let bs = &cand_base[b];
        for i in 1..us.len() {
            let cu = us[i];
            let g = cu * price_at[b] * DT + bs[i] - idle;
            if cu > 0.0 {
                dis_curve[b].push(cu, g);
            } else if cu < 0.0 {
                chg_curve[b].push(-cu, g);
            }
        }
    }

    // --- Step 2: allocate against the line limits.
    // flow[l] = flow_exogenous[l] + sum_b u_b * ptdf[l][node_b].
    // The zero-action injection vector is exactly this step's exogenous row (the slack entry is
    // re-derived from the same summands), so its flows are the ones cached at plan time.
    let base_flows = if state.time_step < plan.exo_flows.len() {
        plan.exo_flows[state.time_step].clone()
    } else {
        let zeros = vec![0.0f64; num_batteries];
        let injections = challenge.compute_total_injections(state, &zeros);
        challenge.network.compute_flows(&injections)
    };
    let limits = &challenge.network.flow_limits;
    let num_lines = base_flows.len().min(limits.len()).min(plan.num_lines);
    let caps: Vec<f64> = (0..num_lines)
        .map(|l| limits[l] * plan.flow_safety)
        .collect();

    // Which lines does the unconstrained plan overload?
    let mut des_flows = base_flows[..num_lines].to_vec();
    for b in 0..num_batteries {
        let d = desired[b];
        if d == 0.0 {
            continue;
        }
        let sens = &plan.batteries[b].sens;
        for l in 0..num_lines {
            des_flows[l] += d * sens[l];
        }
    }
    let mut any_overloaded = false;
    for l in 0..num_lines {
        if des_flows[l].abs() > caps[l] {
            any_overloaded = true;
            break;
        }
    }

    let mut action = if !any_overloaded {
        // Nothing binds: take every battery's unconstrained optimum as-is.
        desired.clone()
    } else {
        // Order the two value-ordered allocators by value per unit of scarce line headroom rather
        // than by dollar gain. See `value_density`: a dollar gain is dominated by battery size,
        // which is not what the lines charge for, so the gain order spends the step's headroom on
        // the largest bidders instead of the most efficient ones.
        let density = value_density(
            &plan.batteries,
            &desired,
            &gain,
            &base_flows,
            &caps,
            num_lines,
            num_batteries,
        );

        // Baseline allocation: serve the densest batteries first, shrinking each to fit.
        let greedy = greedy_allocate(
            &plan.batteries,
            &desired,
            &density,
            &base_flows,
            &caps,
            num_lines,
            num_batteries,
        );

        // Proportional allocation: shrink the whole unconstrained plan by a single factor.
        // When many batteries compete for little headroom the sequential greedy spends it
        // all on the first few and leaves the rest idle, whereas one uniform factor keeps
        // every battery trading and preserves whatever cancellation the fleet's line
        // footprints already have. `des_flows` is `base_flows + delta`, so the factor costs
        // O(num_lines) to find and the candidate is feasible by construction.
        let mut lambda = 1.0f64;
        for l in 0..num_lines {
            let d = des_flows[l] - base_flows[l];
            if d.abs() < 1e-12 {
                continue;
            }
            let bound = if d > 0.0 {
                (caps[l] - base_flows[l]) / d
            } else {
                (-caps[l] - base_flows[l]) / d
            };
            if bound < lambda {
                lambda = bound;
            }
        }
        let lambda = lambda.clamp(0.0, 1.0);
        let scaled: Vec<f64> = desired.iter().map(|&u| u * lambda).collect();

        // Topped-up allocation: start from the scaled plan and give the headroom it left
        // unused back to the batteries that value it most.
        //
        // One global factor is set by the single tightest line, so a lone near-saturated line
        // shrinks every battery -- including the ones whose PTDF column barely touches that
        // line -- and the slack on the other lines is simply thrown away. On the tight tracks
        // that is most of the fleet idling because of one bottleneck it does not sit behind.
        // Walking the batteries in descending value and raising each as far toward its own
        // unconstrained optimum as the remaining headroom allows reclaims that slack. It
        // starts from a feasible point rather than from zero, so unlike the sequential greedy
        // the first few batteries cannot take everything: each one only climbs from the share
        // the uniform factor already guaranteed it.
        //
        // The scaled flows are affine in lambda, so the starting flow vector costs O(lines)
        // rather than a re-accumulation over the fleet.
        let scaled_flows: Vec<f64> = (0..num_lines)
            .map(|l| base_flows[l] + lambda * (des_flows[l] - base_flows[l]))
            .collect();
        let topped = topup_allocate(
            &plan.batteries,
            &desired,
            &scaled,
            &density,
            &scaled_flows,
            &caps,
            num_lines,
            num_batteries,
        );

        // Counter-flow allocation: pair the fleet's most valuable discharge with its most
        // valuable charge and move the *same* MW through both, so the pair's nodal injections
        // cancel and its only line footprint is the difference between two PTDF columns.
        //
        // All three allocations above are sub-selections of one direction: every battery takes
        // the sign its own price-versus-water-value comparison prefers, and the network can
        // only shrink that. The signs are therefore strongly correlated across the fleet --
        // prices share a diurnal curve and a 0.7-correlated common shock, so at night nearly
        // every battery wants to charge and at the peak nearly every one wants to discharge --
        // and correlated injections add up in every PTDF column. That is why the uniform factor
        // collapses on the tight tracks: the fleet is asking the grid to move several hundred MW
        // one way through lines whose exogenous flows already sit at 85% of a limit scaled by
        // 0.4. Each battery gets a sliver, no cycle ever completes, and the headroom spent buys
        // the least valuable MW of every battery instead of the most valuable MW of a few.
        //
        // A pair breaks that. Discharging at the fleet's dearest node while charging at its
        // cheapest moves energy sideways through the fleet for a line cost of
        // |ptdf[l][node_d] - ptdf[l][node_c]| per MW instead of |ptdf[l][node]| per MW, which is
        // small for all but the lines separating the two nodes, and exactly zero in aggregate
        // injection. It is pure spatial arbitrage and it is available at every step, including
        // the ones where the time-arbitrage direction is hopeless. The cross-sectional spread it
        // trades is largest precisely where this solver loses: the idiosyncratic part of an RT
        // shock is sigma * lambda_da * sqrt(1 - rho_spatial), about $8/MWh on capstone, on top of
        // which a flagged node draws a congestion premium averaging $16/MWh and a jump lands on
        // 5% of (node, step) pairs with a Pareto(2.5) multiple of the day-ahead price. The
        // dearest and cheapest of 100 nodes are therefore tens of dollars apart most steps, while
        // a round trip costs the 9.75% efficiency loss plus two transaction fees -- about
        // $5/MWh. The batteries' own value tables price both ends, so a pair is taken only when
        // the two marginal values sum positive and the realised price spread covers the round
        // trip in cash, not merely in water value.
        //
        // Whatever headroom the pairs leave is then spent on the ordinary per-battery optimum,
        // so this candidate degrades into the top-up one when no pair pays.
        let (pair_action, pair_flows) = paired_allocate(
            &plan.batteries,
            &m_dis,
            &m_chg,
            &dis_curve,
            &chg_curve,
            &price_at,
            &base_flows,
            &caps,
            num_lines,
            num_batteries,
        );
        // Pin a paired battery only when the pair drove it against its own solo optimum. A pair
        // is sized at the smaller end, because the two ends must move the same MW for their
        // injections to cancel, so an end can still come out short of the action it asked for --
        // the walk now carries the remainder to the next battery inward, but a line that clips a
        // pair short of both ends' capacity still leaves both short. Pinning both ends hid that
        // shortfall from the top-up pass -- `delta` came out zero and the battery could never
        // climb, however much headroom the pair had left on the lines it does not load.
        //
        // The two ends are `order[0]` and `order[m - 1]`, the fleet's best discharger and best
        // charger, so the pinned actions are the two most valuable in the step. The shortfall
        // scales with fleet heterogeneity: a battery's power is `NOMINAL_POWER * 3^(h(2r-1))`
        // with r uniform, which is a 1.25x spread at h = 0.2 but a 9x spread at h = 1.0, so a
        // 75 MW battery is routinely paired with an 8 MW one and then held at 8 MW.
        //
        // Where the pair's direction agrees with the battery's own optimum there is nothing to
        // protect: `desired[b]` is that direction's best action, `mag` is a minimum taken over it
        // and the other end, so `|pair_action[b]| <= |desired[b]|` and moving along the segment
        // from one to the other only climbs toward the battery's own optimum -- it cannot reverse
        // the pair or overshoot. Where the directions differ -- the counter-flow end bought at a
        // node it would rather have sold at, or a battery that wanted to idle -- `desired[b]`
        // would drag it back through zero and undo the transfer, so those stay pinned.
        //
        // An end the pair drove *past* its own optimum is pinned for the same reason it is pinned
        // when the pair drove it the other way: the top-up moves a battery along the segment from
        // its current action to its target, so a target smaller in magnitude than the pair's action
        // would walk the pair back down and leave the other end's injection uncancelled. That case
        // did not arise while a pair was sized at the two ends' own optima -- `|pair_action[b]|`
        // was then bounded by `|desired[b]|` whenever the two agreed in sign -- and it does now
        // that the magnitude is a joint maximisation.
        let mut pair_target = desired.clone();
        for b in 0..num_batteries {
            let pa = pair_action[b];
            if pa != 0.0 && !(pa * desired[b] > 0.0 && pa.abs() < desired[b].abs()) {
                pair_target[b] = pa;
            }
        }
        let paired = topup_allocate(
            &plan.batteries,
            &pair_target,
            &pair_action,
            &density,
            &pair_flows,
            &caps,
            num_lines,
            num_batteries,
        );

        // Relief allocation: buy headroom on the binding line by paying a few well-placed
        // batteries to push it the other way, then hand the headroom that frees to the fleet.
        //
        // Every candidate above is a *sub-selection* of one direction field. `greedy`, `scaled`
        // and `topped` each take `desired[b]` -- the sign the battery's own price-versus-water
        // comparison prefers -- and let the network shrink it; `paired` is the one that puts a
        // battery on the other side, and it only does so in matched MW, because the pair's whole
        // point is that its two nodal injections cancel. So the fleet's *net* injection is always
        // the net the direction field asked for, scaled down. There is no candidate in which the
        // fleet deliberately injects against its own preference in order to make room.
        //
        // That is the move a tight instance wants. The exogenous path is rescaled until the worst
        // line sits at 85% of a limit already scaled by 0.40 on the capstone track, so the binding
        // line has about 3 MW of headroom against a hundred batteries all wanting the same sign at
        // the same hour. One battery whose PTDF entry on that line is 0.3, moved 20 MW the other
        // way, puts 6 MW back -- it triples the headroom the rest of the fleet is bidding for --
        // and it pays for that with its own one-step value, not with anyone else's. A pair cannot
        // express it: a pair is sized at the smaller of its two ends and needs the realised price
        // spread to cover the round trip in cash, so it moves MW sideways through the fleet rather
        // than opening the one constraint that is stopping everybody.
        //
        // Which batteries to flip is a ranking, not a price. For each battery the flip costs
        // `gain[b] - g_rel` -- what it gives up by taking its best action in the relieving
        // direction instead of its own optimum -- and frees `-s * (u_rel - desired[b]) * ptdf`
        // MW on the binding line. Taking them in ascending cost per MW freed and stopping as soon
        // as the line is back inside its cap buys the headroom from whoever is cheapest, and the
        // count is self-limiting: usually one or two batteries, never the whole fleet, because the
        // loop stops at the excess rather than at an exhausted list.
        //
        // The relievers are placed first, from the exogenous flows, each at the largest fraction
        // of its action every line allows; the ordinary top-up pass then climbs the rest of the
        // fleet toward `desired` in the same value-density order the other candidates use. The
        // flipped batteries are pinned at their relieving action so the top-up cannot drag them
        // back through zero and undo the relief, exactly as the pairing's counter-directional ends
        // are pinned.
        //
        // Cost is one `max_feasible_fraction` per flip plus one top-up pass, against the eight
        // `O(batteries x lines)` passes the step already makes, and it is skipped entirely when no
        // battery can relieve the line -- then there is no fifth candidate and the selection below
        // is bit-identical.
        let mut lstar = usize::MAX;
        let mut worst_load = 1.0f64;
        for l in 0..num_lines {
            if caps[l] <= 0.0 {
                continue;
            }
            let load = des_flows[l].abs() / caps[l];
            if load > worst_load {
                worst_load = load;
                lstar = l;
            }
        }
        let mut relief_target = desired.clone();
        let mut relief_start = vec![0.0f64; num_batteries];
        let mut relief_flows = base_flows[..num_lines].to_vec();
        let mut flipped = false;
        if lstar < num_lines {
            // Sign of the overload: the flip has to move `s * flow[lstar]` down.
            let s = if des_flows[lstar] >= 0.0 { 1.0f64 } else { -1.0f64 };
            // (cost per MW freed, MW freed, battery, relieving action).
            let mut cands: Vec<(f64, f64, usize, f64)> = Vec::new();
            for b in 0..num_batteries {
                let sens_l = plan.batteries[b].sens[lstar];
                if sens_l.abs() < 1e-12 {
                    continue;
                }
                // `u * sens_l` is this battery's contribution to the flow, so the relieving sign
                // of `u` is whichever makes `s * u * sens_l` negative.
                let (u_rel, m_rel) = if s * sens_l < 0.0 {
                    (u_dis[b], m_dis[b])
                } else {
                    (u_chg[b], m_chg[b])
                };
                if u_rel == 0.0 || !m_rel.is_finite() {
                    continue;
                }
                // A battery already pointing that way is not a flip; the ordinary candidates
                // already give it its own optimum.
                if u_rel * desired[b] > 0.0 {
                    continue;
                }
                let freed = -s * (u_rel - desired[b]) * sens_l;
                if freed <= 1e-9 {
                    continue;
                }
                // `m_rel` is the relieving direction's value per MWh of throughput, so this is
                // that direction's one-step gain in the same dollars as `gain[b]`.
                let g_rel = m_rel * u_rel.abs() * DT;
                let cost = (gain[b] - g_rel).max(0.0);
                cands.push((cost / freed, freed, b, u_rel));
            }
            cands.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));
            let mut excess = des_flows[lstar].abs() - caps[lstar];
            for &(_, _freed, b, u_rel) in cands.iter() {
                if excess <= 0.0 {
                    break;
                }
                let sens = &plan.batteries[b].sens;
                let fraction = max_feasible_fraction(sens, u_rel, &relief_flows, &caps, num_lines)
                    .clamp(0.0, 1.0);
                if fraction <= 0.0 {
                    continue;
                }
                let u = u_rel * fraction;
                for l in 0..num_lines {
                    relief_flows[l] += u * sens[l];
                }
                relief_start[b] = u;
                relief_target[b] = u_rel;
                excess -= -s * (u - desired[b]) * sens[lstar];
                flipped = true;
            }
        }
        let relief = if flipped {
            topup_allocate(
                &plan.batteries,
                &relief_target,
                &relief_start,
                &density,
                &relief_flows,
                &caps,
                num_lines,
                num_batteries,
            )
        } else {
            Vec::new()
        };

        // Keep whichever allocation scores best under the one-step lookahead. All of them are
        // feasible and the score is exactly the objective each battery's `pick_best_signed`
        // maximises, so adding a candidate can only raise the chosen value.
        let bats = &plan.batteries;
        let v_greedy = action_value(bats, &state.socs, &price_at, &greedy, next_row_start, bins);
        let v_scaled = action_value(bats, &state.socs, &price_at, &scaled, next_row_start, bins);
        let v_topped = action_value(bats, &state.socs, &price_at, &topped, next_row_start, bins);
        let v_paired = action_value(bats, &state.socs, &price_at, &paired, next_row_start, bins);
        let v_relief = if flipped {
            action_value(bats, &state.socs, &price_at, &relief, next_row_start, bins)
        } else {
            f64::NEG_INFINITY
        };
        if flipped
            && v_relief >= v_paired
            && v_relief >= v_topped
            && v_relief >= v_scaled
            && v_relief >= v_greedy
        {
            relief
        } else if v_paired >= v_topped && v_paired >= v_scaled && v_paired >= v_greedy {
            paired
        } else if v_topped >= v_scaled && v_topped >= v_greedy {
            topped
        } else if v_scaled >= v_greedy {
            scaled
        } else {
            greedy
        }
    };

    // --- Step 3: hard clamp to the authoritative bounds, then verify against the real model.
    for b in 0..num_batteries {
        let (lower, upper) = state.action_bounds[b];
        if action[b] < lower {
            action[b] = lower;
        }
        if action[b] > upper {
            action[b] = upper;
        }
    }

    let mut feasible = flows_ok(challenge, state, &action);
    if !feasible {
        for _ in 0..16 {
            for v in action.iter_mut() {
                *v *= 0.5;
            }
            if flows_ok(challenge, state, &action) {
                feasible = true;
                break;
            }
        }
    }
    if !feasible {
        // Zeros are feasible by construction of the exogenous injections.
        action = vec![0.0f64; num_batteries];
    }

    Ok(action)
}

/// Best candidate action for one battery at a given nodal price, overall and within each
/// direction. Returns `(u, value, u_discharge, value_discharge, u_charge, value_charge)`;
/// index 0 of the candidate list is always the idle action, so the overall best is finite.
/// A direction with no candidate at all (a full or empty battery) comes back as
/// `(0.0, NEG_INFINITY)`, which the pairing below reads as "unusable in that direction".
///
/// The per-direction winners are what make the counter-flow pairing cheap: both ends of a pair
/// need the best action *in a prescribed direction*, which is usually not the battery's own
/// optimum, and recovering it here costs nothing on top of the scan already being done.
fn pick_best_signed(us: &[f64], bs: &[f64], price: f64) -> (f64, f64, f64, f64, f64, f64) {
    let mut best_u = 0.0f64;
    let mut best_v = f64::NEG_INFINITY;
    let mut u_dis = 0.0f64;
    let mut v_dis = f64::NEG_INFINITY;
    let mut u_chg = 0.0f64;
    let mut v_chg = f64::NEG_INFINITY;
    for j in 0..us.len() {
        let u = us[j];
        let v = u * price * DT + bs[j];
        if v > best_v {
            best_v = v;
            best_u = u;
        }
        if u > 0.0 {
            if v > v_dis {
                v_dis = v;
                u_dis = u;
            }
        } else if u < 0.0 {
            if v > v_chg {
                v_chg = v;
                u_chg = u;
            }
        }
    }
    if best_v == f64::NEG_INFINITY {
        return (0.0, 0.0, 0.0, f64::NEG_INFINITY, 0.0, f64::NEG_INFINITY);
    }
    (best_u, best_v, u_dis, v_dis, u_chg, v_chg)
}

/// Value density of each battery's desired action: the dollars of one-step gain it earns per unit
/// of the scarcest line headroom it consumes.
///
/// Both value-ordered allocators below used to walk the fleet in descending `gain`, which is a
/// gain in dollars and so is dominated by battery size: power is `NOMINAL_POWER * 3^(h(2r-1))`
/// with r uniform, a 9x spread at h = 1.0, and a battery nine times the size earns nine times the
/// dollars at the same price signal. Serving in that order spends the step's headroom on the
/// largest bidders rather than on the most efficient ones, and size is not what the lines charge
/// for: what a battery costs the network is `|u * ptdf[l][node]|` against the headroom line l has
/// left, and the PTDF column and the headroom are both independent of how big the battery is. Two
/// batteries at the same `$/MWh` can differ by an order of magnitude in how much of the binding
/// line they eat per MW -- one sits across the bottleneck, the other barely moves it -- and the
/// gain order cannot tell them apart. That is the ordinary failure of a knapsack greedy run on
/// value instead of on value per unit of the constrained resource, and here the resource is
/// scarcest exactly on the tracks this solver loses: the generator rescales every step until the
/// worst line sits at 85% of a limit already scaled by 0.4, so a 100-battery fleet is bidding for
/// about 3 MW of headroom on that line.
///
/// The key is `gain / ||f||_2` where `f_l = |u * ptdf[l][node]| / headroom_l` is the fraction of
/// line l's remaining headroom the action eats, with the headroom taken in the direction the
/// action pushes the line. Both numerator and denominator scale with `|u|`, so the density is
/// still the battery's marginal value per MWh divided by the headroom one MW of it consumes -- a
/// ranking in which size cancels. A battery whose PTDF column is all zeros (one at the slack bus)
/// costs the network nothing and sorts first, which is right.
///
/// The norm is over the whole column rather than the single worst entry, and that is the one
/// change here. `max_l f_l` says a battery costs the network whatever it does to the line it hurts
/// most and nothing at all for what it does to every other line. Two bidders that both take 50% of
/// their worst line's headroom then rank identically, even when one of them sits on a single line
/// and the other takes 50% of five lines at once and so blocks five times as much of the fleet
/// behind it. That is not a tie-break detail: a node's PTDF column has a significant entry on
/// roughly every line of its local mesh, so the number of lines a battery genuinely loads scales
/// with the network's line-to-node ratio -- 1.5 on the baseline, congested and multiday tracks
/// against 2.0 on dense and capstone. The under-pricing of a broad footprint is therefore worst
/// exactly on the two tracks this solver loses on, and it is the knapsack greedy spending its
/// scarcest resource on the bidders that consume the most of it.
///
/// L2 rather than a plain sum because the sum is not robust here. A PTDF column on a meshed
/// network has a long tail of small entries, and 300 lines each at 1% of their headroom would add
/// up to three times a single line taken at 100% -- the ranking would be decided by the tail
/// instead of by the lines that bind. The L2 norm is dominated by the large entries: one line at
/// 0.5 against a tail of 300 at 0.01 gives 0.529 against the max's 0.5, a 6% correction, while ten
/// lines at 0.4 give 1.26 against the max's 0.4, a 3.2x penalty. It reduces to `max_l` exactly when
/// one line dominates the column, so on a loose track -- where the headroom is large and every
/// fraction is small except on the one line that triggered the allocation at all -- the order is
/// the one the max produced, and the change is active only where several lines are simultaneously
/// scarce.
///
/// Same single `O(batteries x lines)` pass, one multiply-add per line and one square root per
/// battery more than before. Nothing about the candidate actions, the value tables or the feasible
/// set changes -- only the order in which the headroom is handed out -- and the plan-time replay in
/// `measure_delivery` calls this same function, so the service probability the sweep plans with
/// still describes the policy that is played.
///
/// One O(batteries x lines) pass per step, against the eight the step already makes, and nothing
/// about the candidate actions, the value tables or the feasible set changes -- only the order in
/// which the headroom is handed out.
fn value_density(
    batteries: &[BatteryPlan],
    desired: &[f64],
    gain: &[f64],
    base_flows: &[f64],
    caps: &[f64],
    num_lines: usize,
    num_batteries: usize,
) -> Vec<f64> {
    let mut out = vec![0.0f64; num_batteries];
    for b in 0..num_batteries {
        let d = desired[b];
        if d == 0.0 {
            continue;
        }
        let sens = &batteries[b].sens;
        // How much of the network's remaining headroom this action would take, aggregated over
        // every line it loads rather than read off the single worst one.
        let mut sq = 0.0f64;
        for l in 0..num_lines {
            let delta = d * sens[l];
            if delta.abs() < 1e-12 {
                continue;
            }
            let head = if delta > 0.0 {
                caps[l] - base_flows[l]
            } else {
                caps[l] + base_flows[l]
            };
            let frac = delta.abs() / head.max(1e-6);
            sq += frac * frac;
        }
        let cost = sq.sqrt();
        out[b] = if cost > 1e-12 {
            gain[b] / cost
        } else {
            f64::INFINITY
        };
    }
    out
}

/// Sequential allocation: walk batteries in descending `order_key`, giving each the largest
/// fraction of its desired action that keeps every line inside its cap.
///
/// `order_key` is the value density from `value_density`, not the raw dollar gain: the headroom is
/// a shared resource and the greedy that spends it well is the one that ranks bidders by value per
/// unit of it.
fn greedy_allocate(
    batteries: &[BatteryPlan],
    desired: &[f64],
    order_key: &[f64],
    base_flows: &[f64],
    caps: &[f64],
    num_lines: usize,
    num_batteries: usize,
) -> Vec<f64> {
    let mut flows = base_flows[..num_lines].to_vec();
    let mut order: Vec<usize> = (0..num_batteries).collect();
    order.sort_by(|&x, &y| {
        order_key[y]
            .partial_cmp(&order_key[x])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut action = vec![0.0f64; num_batteries];
    for &b in order.iter() {
        let d = desired[b];
        if d == 0.0 {
            continue;
        }
        let sens = &batteries[b].sens;
        let fraction = max_feasible_fraction(sens, d, &flows, caps, num_lines);
        if fraction <= 0.0 {
            continue;
        }
        let u = d * fraction;
        for l in 0..num_lines {
            flows[l] += u * sens[l];
        }
        action[b] = u;
    }
    action
}

/// Raise an already-feasible allocation toward each battery's unconstrained optimum, walking
/// batteries in descending `order_key` and spending only the headroom `start` left unused.
///
/// `flows` must be the line flows of `start` itself (exogenous plus the started allocation).
/// Each battery moves along the segment from its current action to its desired one, so the
/// result stays feasible and no battery ever moves away from the action it asked for.
///
/// `order_key` is the value density from `value_density`. This is the site where the order matters
/// most: the uniform shrink that `start` usually comes from is set by the single tightest line, so
/// what this pass hands out is the slack every other line still has, and which batteries get it
/// decides most of the step's value. The density is scale-free in `|u|`, so it ranks the partial
/// climbs here exactly as it ranks the whole actions it was computed from.
fn topup_allocate(
    batteries: &[BatteryPlan],
    desired: &[f64],
    start: &[f64],
    order_key: &[f64],
    flows: &[f64],
    caps: &[f64],
    num_lines: usize,
    num_batteries: usize,
) -> Vec<f64> {
    let mut action = start.to_vec();
    let mut flows = flows[..num_lines].to_vec();

    let mut order: Vec<usize> = (0..num_batteries).collect();
    order.sort_by(|&x, &y| {
        order_key[y]
            .partial_cmp(&order_key[x])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    for &b in order.iter() {
        let delta = desired[b] - action[b];
        if delta.abs() < 1e-12 {
            continue;
        }
        let sens = &batteries[b].sens;
        let fraction = max_feasible_fraction(sens, delta, &flows, caps, num_lines);
        if fraction <= 0.0 {
            continue;
        }
        let step = delta * fraction;
        for l in 0..num_lines {
            flows[l] += step * sens[l];
        }
        action[b] += step;
    }
    action
}

/// Counter-flow allocation: take the fleet's discharges in descending value, pair each with a
/// charge end and move the same MW through both, so the pair's nodal injections cancel and its
/// line footprint is only the difference of two PTDF columns.
///
/// `m_dis` / `m_chg` are the per-MWh value of the best action in each direction, so a pair pays
/// iff they sum positive -- and that sum already carries both batteries' water values, hence the
/// efficiency loss and the SOC the transfer creates and destroys. On top of it the realised price
/// spread must cover the round trip in cash, which is what keeps the pairing from trading on a
/// value-table disagreement between two batteries at the same price.
///
/// The charge end is chosen rather than taken. This walk used to run two indices inward from the
/// ends of one descending-`m_dis` order, which fixes the partner by rank: the charger is whichever
/// battery's *discharge* was worth least, and nothing about the pair's line cost enters. But the
/// line cost is what decides how much of the pair clears. A pair moves MW against
/// `|ptdf[l][node_d] - ptdf[l][node_c]|`, which is near zero for two electrically adjacent nodes
/// and large for two either side of the bottleneck, so the deliverable MW varies by an order of
/// magnitude with the partner -- and on the tight tracks, where the binding line leaves a couple of
/// MW against a fleet of a hundred, almost every rank-chosen pair is clipped to a sliver and then
/// retired. That is the same failure `value_density` already fixes for the two per-battery
/// allocators, which rank on dollars per unit of scarce headroom rather than on dollars.
///
/// So each discharge end scans every remaining charger that can pay and keeps the one maximising
/// `(m_dis + m_chg) x MW x DT`, the pair's whole one-step gain. The MW is first bounded by the two
/// ends' remaining power and by the headroom on the single most-loaded line -- a one-constraint
/// relaxation of the feasible set -- and the ranking that bound induces is then refined exactly:
/// the best bound is popped, its deliverable MW is recomputed over every line, and the walk stops
/// as soon as that exact gain dominates every bound still on the list. See `PAIR_REFINE`.
///
/// The winner's magnitude is a joint maximisation, not either end's own optimum. The MW both ends
/// move is whatever maximises `dis_curve[bd] + chg_curve[bc]` over the interval the lines and the
/// two ends' remaining power leave -- exactly, because both curves are concave and piecewise linear
/// with known breakpoints (see `GainCurve` and `best_pair_mw`). The quantity that used to size a
/// pair, `min(u_dis[bd], -u_chg[bc])`, is each end's best action *in the prescribed direction*,
/// which on the steps where this allocation wins is the smallest magnitude on the curve for the end
/// moving against its own price-versus-water-value comparison -- one SOC grid node of movement,
/// about a quarter of nameplate. A spatial transfer's size has nothing to do with either end's
/// temporal optimum; what it is worth is carried by the two water values inside `m_dis + m_chg`,
/// and what clears is set by the lines.
///
/// Returns the allocation and its line flows, so the caller can keep spending the leftover
/// headroom without re-accumulating them. `role` keeps the discharge set and the charge set
/// disjoint, which the two-pointer walk got from `lo < hi` for free. An end is only retired once
/// the MW it had to give are spent: the pair moves as a unit, and the end with MW left over keeps
/// the remainder and is matched again, charged its true diminishing marginal value for what it has
/// already moved. That holds for a pair a line clips short of both ends as well -- the charger is
/// retired and the discharge end re-scans, exactly as it does when a charger turns out to deliver
/// nothing at all.
fn paired_allocate(
    batteries: &[BatteryPlan],
    m_dis: &[f64],
    m_chg: &[f64],
    dis_curve: &[GainCurve],
    chg_curve: &[GainCurve],
    price_at: &[f64],
    base_flows: &[f64],
    caps: &[f64],
    num_lines: usize,
    num_batteries: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut action = vec![0.0f64; num_batteries];
    let mut flows = base_flows[..num_lines].to_vec();
    if num_batteries < 2 {
        return (action, flows);
    }

    // Descending marginal value of discharging: the order the discharge ends are tried in.
    let mut order: Vec<usize> = (0..num_batteries).collect();
    order.sort_by(|&x, &y| {
        m_dis[y]
            .partial_cmp(&m_dis[x])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    // Descending marginal value of charging: the order the charge ends are *offered* in. The walk
    // used to take its charge end from the far end of `order`, i.e. the battery whose discharge
    // was worth least, which is only a proxy for the battery whose charge is worth most.
    let mut corder: Vec<usize> = (0..num_batteries).collect();
    corder.sort_by(|&x, &y| {
        m_chg[y]
            .partial_cmp(&m_chg[x])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // MW still available at each end, and the role each battery has been committed to. A pair must
    // move the same MW through both ends for its injections to cancel, so it is sized as a unit and
    // whichever end runs out first is retired; the other keeps its remainder and is matched again,
    // so the fleet's best discharge is spent against successive chargers rather than capped by the
    // first one. `role` keeps a battery that has charged from later discharging and vice versa,
    // which the two-pointer walk got from `lo < hi` for free.
    //
    // The MW available at an end is its action bound in that direction -- the largest magnitude on
    // its gain curve -- not the magnitude its own *temporal* optimum prefers. Those are the same
    // number only when the end already wants to move that way, and the steps where this allocation
    // is the one the lookahead picks are exactly the steps where it does not: the lines bind
    // because the fleet's desired signs agree, so every pair has one end moving against its own
    // price-versus-water-value comparison, and that end's best action *in the prescribed
    // direction* is the smallest magnitude on its curve -- one SOC grid node of movement, about a
    // quarter of nameplate at 49 bins. Capping the pair there caps the fleet's whole counter-flow
    // volume at a quantity that has nothing to do with the spread the pair is trading. What a
    // spatial transfer is worth is `m_dis + m_chg` per MWh, which already carries both water
    // values, and how much of it clears is set by the lines; the magnitude is then a joint
    // maximisation over the two curves (see `best_pair_mw`), and the end that is paying for the
    // transfer is free to go as deep as the joint value keeps rising.
    //
    // Where a line clips the pair the magnitude is the line's, unchanged: `hi` below takes the
    // minimum of the two ends' remaining MW and the largest magnitude every line allows, and the
    // latter is independent of what the ends asked for. So this acts on the pairs the network
    // leaves room for -- the ones joining electrically adjacent nodes, whose
    // difference-of-PTDF-columns footprint is near zero, which are precisely the only pairs that
    // clear at all on the tight tracks.
    let mut rem_d: Vec<f64> = (0..num_batteries).map(|b| dis_curve[b].top()).collect();
    let mut rem_c: Vec<f64> = (0..num_batteries).map(|b| chg_curve[b].top()).collect();
    let mut role = vec![0i8; num_batteries];

    // The line carrying the largest fraction of its cap, which is the one with the least headroom
    // left for the next pair. It is recomputed inside the flow update each pair applies, so it
    // costs nothing beyond the pass that was already there.
    let mut lstar = usize::MAX;
    {
        let mut worst = -1.0f64;
        for l in 0..num_lines {
            if caps[l] <= 0.0 {
                continue;
            }
            let load = flows[l].abs() / caps[l];
            if load > worst {
                worst = load;
                lstar = l;
            }
        }
    }

    // Candidate partners for the current discharge end, as (upper bound on the pair's one-step
    // gain, battery). Reused across pairs so the walk allocates once.
    let mut cands: Vec<(f64, usize)> = Vec::with_capacity(num_batteries);

    let mut lo = 0usize;
    let mut misses = 0usize;
    while lo < num_batteries {
        let bd = order[lo];
        if role[bd] < 0 || rem_d[bd] <= 1e-9 {
            lo += 1;
            continue;
        }
        let sd = &batteries[bd].sens;
        let (cap_star, flow_star, sd_star) = if lstar < num_lines {
            (caps[lstar], flows[lstar], sd[lstar])
        } else {
            (0.0f64, 0.0f64, 0.0f64)
        };

        // --- Choose the charge end on the value it can actually deliver, not on its rank.
        //
        // A pair's line footprint is |ptdf[l][node_d] - ptdf[l][node_c]| per MW, so how much of it
        // clears is a property of *which two nodes* it joins, and on a meshed network that varies
        // by an order of magnitude across the fleet: two electrically adjacent nodes have nearly
        // equal PTDF columns and move as much as their batteries hold, two nodes either side of
        // the bottleneck move almost nothing. Taking the single highest-value charger therefore
        // picks the pair's partner by the one quantity that says nothing about its size on the
        // lines, and on the tight tracks -- where the binding line has a couple of MW of headroom
        // against a fleet of a hundred batteries -- nearly every pair is clipped to a sliver and
        // then retired.
        //
        // This is the same argument `value_density` already makes for the two value-ordered
        // allocators, which rank on dollars per unit of scarce headroom rather than on dollars,
        // applied to the one allocator that never got it. The quantity maximised here is the
        // pair's whole one-step gain, `(m_dis + m_chg) x MW`.
        //
        // The scan runs over every remaining charger that can pay, not over a fixed-size pool of
        // the best-priced ones. `corder` is descending in marginal charge value, so the loop
        // still stops at the first charger whose joint marginal value is non-positive and no
        // later one can pay either -- the economics bound the width, the fleet size bounds the
        // worst case, and nothing truncates the middle.
        //
        // What this pass produces is a *bound* per charger, not its gain: the MW is capped by the
        // two ends' remaining power and by the headroom on `lstar` alone, which is a relaxation of
        // the feasible set and so can only overstate what clears. The refinement below turns that
        // ranking into the exact maximiser; see `PAIR_REFINE`.
        cands.clear();
        for &bc in corder.iter() {
            if bc == bd || role[bc] > 0 || rem_c[bc] <= 1e-9 {
                continue;
            }
            let joint = m_dis[bd] + m_chg[bc];
            if !(joint > 0.0) {
                break;
            }
            // Cash test: the discharge end sells a MWh at its node, the charge end buys one at
            // its own, each pays the transaction fee, and the round trip loses
            // (1 - eta_c * eta_d) of the energy, valued at the price it would be rebought for.
            let loss = 1.0 - batteries[bc].eta_c * batteries[bd].eta_d;
            let spread = price_at[bd] - price_at[bc];
            if !(spread > 2.0 * KAPPA_TX + loss * price_at[bc].max(0.0)) {
                continue;
            }
            let mut mw = rem_d[bd].min(rem_c[bc]);
            if lstar < num_lines {
                let slope = sd_star - batteries[bc].sens[lstar];
                if slope.abs() >= 1e-12 {
                    let bound = if slope > 0.0 {
                        (cap_star - flow_star) / slope
                    } else {
                        (-cap_star - flow_star) / slope
                    };
                    let bound = bound.max(0.0);
                    if bound < mw {
                        mw = bound;
                    }
                }
            }
            // In dollars, so the bound is on the same scale as the exact gains the refinement
            // below produces and the early break between them is a comparison of like with like.
            let score = joint * mw * DT;
            if score > 0.0 {
                cands.push((score, bc));
            }
        }

        // --- Refine the screened ranking into the exact argmax.
        //
        // `score` above relaxes the feasible set to the single most-loaded line, so it is an upper
        // bound on `(m_dis + m_chg) x deliverable MW`. Popping the best bound, pricing that partner
        // over every line, and stopping once the exact gain dominates every remaining bound is
        // therefore the exact maximiser, and the first pop is the `fraction` pass the walk made
        // after selecting anyway. `PAIR_REFINE` caps how many partners get priced exactly; a
        // partner the lines leave with no MW at all is dropped from this end's list rather than
        // retired fleet-wide.
        let mut best_c = usize::MAX;
        let mut best_gain = 0.0f64;
        let mut best_mw = 0.0f64;
        let mut refines = 0usize;
        while !cands.is_empty() && refines < PAIR_REFINE {
            let mut top = 0usize;
            for i in 1..cands.len() {
                if cands[i].0 > cands[top].0 {
                    top = i;
                }
            }
            if cands[top].0 <= best_gain {
                break;
            }
            let (_, bc) = cands.swap_remove(top);
            refines += 1;

            // Largest magnitude every line allows this pair's difference-of-columns direction.
            // Flows are affine in the pair's size, so this interval bound is exact; the pair moves
            // as a unit so the two ends stay equal and the cancellation survives.
            let mut mw_line = f64::INFINITY;
            {
                let sc = &batteries[bc].sens;
                for l in 0..num_lines {
                    let slope = sd[l] - sc[l];
                    if slope.abs() < 1e-12 {
                        continue;
                    }
                    let bound = if slope > 0.0 {
                        (caps[l] - flows[l]) / slope
                    } else {
                        (-caps[l] - flows[l]) / slope
                    };
                    if bound < mw_line {
                        mw_line = bound;
                    }
                }
            }
            let hi = rem_d[bd].min(rem_c[bc]).min(mw_line.max(0.0));
            if hi <= 1e-9 {
                continue;
            }
            // Exact joint maximisation over the feasible interval, charging each end for what it
            // has already moved this step.
            let (mw, gain) = best_pair_mw(
                &dis_curve[bd],
                action[bd].max(0.0),
                &chg_curve[bc],
                (-action[bc]).max(0.0),
                hi,
            );
            if gain > best_gain {
                best_gain = gain;
                best_c = bc;
                best_mw = mw;
            }
        }

        if best_c == usize::MAX {
            // No remaining charger pays against this discharge end. `order` is descending in
            // marginal discharge value, so the ends only get worse inward; the miss counter is the
            // tolerance for one odd battery that cannot move in either direction.
            misses += 1;
            if misses >= PAIR_MISS_LIMIT {
                break;
            }
            lo += 1;
            continue;
        }
        // The winner's magnitude was already maximised over the feasible interval during the
        // refinement, so there is no second feasibility pass and no zero branch to fall out of: a
        // partner the lines leave with nothing scores zero and is never selected.
        let bc = best_c;
        misses = 0;

        let step = best_mw;
        {
            let sc = &batteries[bc].sens;
            let mut worst = -1.0f64;
            let mut next_star = usize::MAX;
            for l in 0..num_lines {
                flows[l] += step * (sd[l] - sc[l]);
                if caps[l] > 0.0 {
                    let load = flows[l].abs() / caps[l];
                    if load > worst {
                        worst = load;
                        next_star = l;
                    }
                }
            }
            lstar = next_star;
        }
        // Accumulated, not assigned: an end that survives one pair is matched again below, and
        // its running total is capped by its own remaining MW, so it stays inside the action
        // bound its best-in-direction magnitude came from.
        action[bd] += step;
        action[bc] -= step;
        rem_d[bd] -= step;
        rem_c[bc] -= step;
        role[bd] = 1;
        role[bc] = -1;

        let spent_d = rem_d[bd] <= 1e-9;
        let spent_c = rem_c[bc] <= 1e-9;
        if !spent_d && !spent_c {
            // The lines clipped the pair short of both ends' capacity. Repeating *these two*
            // would find `fraction = 0` -- the line that bound them now sits at its cap in the
            // direction they pushed it -- so the partner is retired. The discharge end is not.
            //
            // It used to be retired with it, and that is the one case in this walk where a
            // partner that delivered something was treated worse than one that delivered
            // nothing: the `fraction <= 0` branch above already retires the charger alone and
            // re-scans for the same discharge end, and a clip is that case with a nonzero first
            // instalment. What binds a pair is `|ptdf[l][node_d] - ptdf[l][node_c]|` on one
            // line, a property of *which two nodes* it joins, so a different charger gives a
            // different difference-of-columns direction -- small, or opposite in sign, on the
            // line that just clipped this one -- and the MW it can move varies by an order of
            // magnitude with the partner. That is the same fact that makes the partner worth
            // choosing at all rather than taking by rank.
            //
            // The end being dropped is `order[lo]`, the fleet's most valuable remaining sale,
            // still holding MW it is bounded to move. Dropping it hands the step's remaining
            // counter-flow capacity to strictly less valuable dischargers, or to none at all
            // once `misses` breaks the walk, and it does so on exactly the steps where the
            // pairing is the candidate the lookahead picks: a clip means the lines are tight,
            // which is when the uniform factor has already collapsed.
            //
            // Termination is unchanged in order. Every iteration still retires a charger or
            // advances `lo`, so the walk runs at most `2 * batteries` iterations rather than the
            // `batteries` the double retirement bought; the economic break on
            // `m_dis + m_chg <= 0` bounds it well under that in practice. Where no pair is
            // clipped -- a loose network, or ends that exhaust each other exactly -- this branch
            // never executes and the allocation is bit-identical.
            rem_c[bc] = 0.0;
        }
        if spent_d {
            lo += 1;
        }
    }

    (action, flows)
}

/// One direction's gain curve for a single battery: up to `ONLINE_ACTION_HALF` candidate
/// magnitudes in ascending order, each with the one-step gain over idle of moving that many MW.
///
/// The per-battery one-step objective is `u * price * DT - kappa_tx |u| DT - kappa_deg (|u| DT /
/// cap)^2 + W[t+1][soc_after(u)]`. `soc_after` is affine in `u` and `interp` is piecewise linear
/// with breakpoints at the SOC grid nodes, so the objective is piecewise linear in `u` with
/// breakpoints exactly at the magnitudes `aligned_actions` enumerates -- plus the quadratic
/// degradation term, which is about 1e-4 dollars at full power against a revenue slope near
/// 12 dollars per MW. Linear interpolation between neighbouring magnitudes is therefore exact to
/// that term, and the gain is zero at zero magnitude by construction.
///
/// The curve is also concave: revenue and the transaction fee are linear in `|u|` on one side of
/// zero, the degradation term is concave, and `W[t+1]` is a concave function of a quantity affine
/// in `u` (the sweep preserves concavity; see `sweep_battery`). That is what makes the pair sizing
/// below a maximisation over breakpoints rather than a search.
#[derive(Clone, Copy)]
struct GainCurve {
    n: usize,
    mag: [f64; ONLINE_ACTION_HALF],
    val: [f64; ONLINE_ACTION_HALF],
}

impl GainCurve {
    fn empty() -> Self {
        Self {
            n: 0,
            mag: [0.0; ONLINE_ACTION_HALF],
            val: [0.0; ONLINE_ACTION_HALF],
        }
    }

    /// Insert one candidate, keeping the magnitudes ascending. `aligned_actions` already emits
    /// them that way, so the insertion walk normally does nothing; it is here so the curve does
    /// not depend on that.
    fn push(&mut self, mag: f64, val: f64) {
        if !(mag > 0.0) || self.n >= ONLINE_ACTION_HALF {
            return;
        }
        let mut i = self.n;
        while i > 0 && self.mag[i - 1] > mag {
            self.mag[i] = self.mag[i - 1];
            self.val[i] = self.val[i - 1];
            i -= 1;
        }
        self.mag[i] = mag;
        self.val[i] = val;
        self.n += 1;
    }

    /// Largest magnitude on the curve, i.e. this direction's action bound. Zero when the
    /// direction is unusable (a full or empty battery).
    fn top(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            self.mag[self.n - 1]
        }
    }

    /// Gain of moving `m` MW in this direction, clamped at the bound.
    fn at(&self, m: f64) -> f64 {
        if self.n == 0 || !(m > 0.0) {
            return 0.0;
        }
        let mut pm = 0.0f64;
        let mut pv = 0.0f64;
        for i in 0..self.n {
            let cm = self.mag[i];
            let cv = self.val[i];
            if m <= cm {
                let d = cm - pm;
                if d <= 1e-12 {
                    return cv;
                }
                return pv + (m - pm) / d * (cv - pv);
            }
            pm = cm;
            pv = cv;
        }
        pv
    }
}

/// Size a counter-flow pair on its own joint economics. Returns `(mw, gain)`, the magnitude both
/// ends move and the dollars it earns, or `(0, 0)` when no magnitude in `(0, hi]` pays.
///
/// The two ends must move the same MW for their nodal injections to cancel, so the pair's gain is
/// `dc.at(used_d + mw) - dc.at(used_d) + cc.at(used_c + mw) - cc.at(used_c)`: the sum of two
/// concave piecewise-linear functions of `mw`, hence concave piecewise linear with breakpoints
/// only at the two curves' own magnitudes. Its maximum over `[0, hi]` is attained at one of those
/// breakpoints or at `hi`, so enumerating them is exact rather than a sample of the interval.
///
/// `used_d` / `used_c` are what each end has already moved in this step, so an end matched against
/// successive partners is charged its true diminishing marginal value rather than its first
/// partner's.
fn best_pair_mw(
    dc: &GainCurve,
    used_d: f64,
    cc: &GainCurve,
    used_c: f64,
    hi: f64,
) -> (f64, f64) {
    if hi <= 1e-9 {
        return (0.0, 0.0);
    }
    let base = dc.at(used_d) + cc.at(used_c);
    let mut best_mw = hi;
    let mut best = dc.at(used_d + hi) + cc.at(used_c + hi) - base;
    for i in 0..dc.n {
        let mw = dc.mag[i] - used_d;
        if mw > 1e-9 && mw < hi {
            let g = dc.at(used_d + mw) + cc.at(used_c + mw) - base;
            if g > best {
                best = g;
                best_mw = mw;
            }
        }
    }
    for i in 0..cc.n {
        let mw = cc.mag[i] - used_c;
        if mw > 1e-9 && mw < hi {
            let g = dc.at(used_d + mw) + cc.at(used_c + mw) - base;
            if g > best {
                best = g;
                best_mw = mw;
            }
        }
    }
    if best <= 0.0 {
        return (0.0, 0.0);
    }
    (best_mw, best)
}

/// Largest `f` in [0, 1] with `|flows[l] + f * delta * sens[l]| <= caps[l]` for every line.
/// Flows are affine in the action, so this interval bound is exact.
fn max_feasible_fraction(
    sens: &[f64],
    delta: f64,
    flows: &[f64],
    caps: &[f64],
    num_lines: usize,
) -> f64 {
    let mut fraction = 1.0f64;
    for l in 0..num_lines {
        let slope = sens[l] * delta;
        if slope.abs() < 1e-12 {
            continue;
        }
        let bound = if slope > 0.0 {
            (caps[l] - flows[l]) / slope
        } else {
            (-caps[l] - flows[l]) / slope
        };
        if bound < fraction {
            fraction = bound;
        }
    }
    fraction
}

/// One-step lookahead score of a whole action vector: realised profit at the given per-battery
/// prices plus the water value of the resulting SOCs.
///
/// Taking the prices per battery rather than per node lets the plan-time replay score candidate
/// allocations with exactly this function under its day-ahead forecast, so the delivery ratio it
/// measures describes the allocator the online step really runs.
fn action_value(
    batteries: &[BatteryPlan],
    socs: &[f64],
    price_at: &[f64],
    action: &[f64],
    next_row_start: usize,
    bins: usize,
) -> f64 {
    let mut total = 0.0f64;
    for (b, &u) in action.iter().enumerate() {
        let bp = &batteries[b];
        let row = &bp.values[next_row_start..next_row_start + bins];
        let top = bp.lo + bp.dk * (bins - 1) as f64;
        let soc_next = soc_after(bp.eta_c, bp.eta_d, u, socs[b], bp.lo, top);
        total += step_reward(u, price_at[b], bp.cap) + interp(row, soc_next, bp.lo, bp.dk, bins);
    }
    total
}

fn read_hyperparameters(hyperparameters: &Option<Map<String, Value>>) -> (f64, f64, f64, f64, f64) {
    let mut dp_budget = DEFAULT_DP_BUDGET;
    let mut flow_safety = DEFAULT_FLOW_SAFETY;
    let mut jump_optimism = DEFAULT_JUMP_OPTIMISM;
    let mut cong_optimism = DEFAULT_CONG_OPTIMISM;
    let mut power_share = DEFAULT_POWER_SHARE;
    if let Some(map) = hyperparameters {
        if let Some(v) = map.get("dp_budget").and_then(|v| v.as_f64()) {
            dp_budget = v;
        }
        if let Some(v) = map.get("flow_safety").and_then(|v| v.as_f64()) {
            flow_safety = v;
        }
        if let Some(v) = map.get("jump_optimism").and_then(|v| v.as_f64()) {
            jump_optimism = v;
        }
        if let Some(v) = map.get("cong_optimism").and_then(|v| v.as_f64()) {
            cong_optimism = v;
        }
        if let Some(v) = map.get("power_share").and_then(|v| v.as_f64()) {
            power_share = v;
        }
    }
    (
        dp_budget.max(1.0e5),
        flow_safety.clamp(0.5, 1.0),
        jump_optimism.clamp(0.0, 4.0),
        cong_optimism.clamp(0.0, 4.0),
        power_share.clamp(0.0, 16.0),
    )
}

fn store_plan(plan: Arc<Plan>) {
    let mut guard = match PLAN.lock() {
        Ok(g) => g,
        Err(e) => e.into_inner(),
    };
    *guard = Some(plan);
}

/// Return the plan for this challenge, rebuilding with defaults if the cached one
/// belongs to a different instance.
fn get_plan(challenge: &Challenge) -> Arc<Plan> {
    let mut guard = match PLAN.lock() {
        Ok(g) => g,
        Err(e) => e.into_inner(),
    };
    if let Some(existing) = guard.as_ref() {
        if existing.seed == challenge.seed
            && existing.num_steps == challenge.num_steps
            && existing.batteries.len() == challenge.batteries.len()
        {
            return existing.clone();
        }
    }
    let fresh = Arc::new(build_plan(
        challenge,
        DEFAULT_DP_BUDGET,
        DEFAULT_FLOW_SAFETY,
        DEFAULT_JUMP_OPTIMISM,
        DEFAULT_CONG_OPTIMISM,
        DEFAULT_POWER_SHARE,
    ));
    *guard = Some(fresh.clone());
    fresh
}

fn build_plan(
    challenge: &Challenge,
    dp_budget: f64,
    flow_safety: f64,
    jump_optimism: f64,
    cong_optimism: f64,
    power_share: f64,
) -> Plan {
    let num_steps = challenge.num_steps;
    let num_batteries = challenge.batteries.len();
    let denom = (num_steps.max(1) * num_batteries.max(1) * DP_EVALS_PER_BIN) as f64;
    let bins = ((dp_budget / denom) as usize).clamp(MIN_BINS, MAX_BINS);

    // --- Price atoms the DP integrates over.
    //
    // An RT price is lambda_da * (1 + sigma * xi) with xi standard normal, plus, with
    // probability rho, a jump of lambda_da * Pareto(alpha). Both parameters are published.
    // Conditional on no jump, xi is integrated by the five-point rule at `KINK_A` / `KINK_B`,
    // which pins the mean, the variance and the half-normal mean rather than the high
    // moments, and uses its second segment per side to track the option value at the kinks
    // this challenge's thresholds actually sit at -- see those constants. The jump branch gets
    // one atom at the Pareto conditional mean
    // alpha/(alpha-1):
    // any price that large has the battery discharging at full power, and above that
    // threshold the inner max is linear in the price, so collapsing the tail onto its mean
    // is exact there rather than merely first-order.
    let alpha_tail = challenge.market.params.tail_index;
    let rho_jump = challenge.market.params.jump_probability.clamp(0.0, 1.0);
    let sigma = challenge.market.params.volatility.max(0.0);
    let pareto_mean = if alpha_tail > 1.0 {
        alpha_tail / (alpha_tail - 1.0)
    } else {
        4.0
    };
    // Multipliers on the day-ahead price, one per atom.
    // Generated in increasing price so the envelope walk's insertion sort has little to do.
    let atom_mult: [f64; PRICE_ATOMS] = [
        1.0 - sigma * KINK_B,
        1.0 - sigma * KINK_A,
        1.0,
        1.0 + sigma * KINK_A,
        1.0 + sigma * KINK_B,
        1.0 + jump_optimism * pareto_mean,
    ];
    let atom_weight: [f64; PRICE_ATOMS] = [
        (1.0 - rho_jump) * KINK_W_B,
        (1.0 - rho_jump) * KINK_W_A,
        (1.0 - rho_jump) * KINK_W_CENTRE,
        (1.0 - rho_jump) * KINK_W_A,
        (1.0 - rho_jump) * KINK_W_B,
        rho_jump,
    ];

    let num_lines = challenge
        .network
        .ptdf
        .len()
        .min(challenge.network.flow_limits.len());
    let num_nodes = challenge.network.num_nodes;

    // --- Exogenous line flows for the whole horizon, computed once.
    //
    // The congestion-premium sweep, the headroom sweep, the delivery measurement below
    // and every online step all need this same vector, and each recomputation is O(lines x nodes)
    // -- 45k multiply-adds per step on the capstone track. Caching it pays for the measurement
    // pass outright.
    let exo_flows: Vec<Vec<f64>> = (0..num_steps)
        .map(|t| {
            challenge
                .network
                .compute_flows(&challenge.exogenous_injections[t])
        })
        .collect();

    // --- Expected congestion premium, per (step, node).
    //
    // An RT price gets `+ GAMMA_PRICE * max(Z, 0)` whenever its node is flagged congested, and a
    // node is flagged when any incident line independently trips with probability
    // (|flow_l| / (tau * limit_l))^CONG_EXP. The flows in that test come from the *exogenous*
    // injections one step earlier -- battery actions do not enter -- and the whole exogenous
    // path is known at plan time, so the trip probabilities are computable offline. The premium
    // draw is independent of the trip draws, so
    //     E[premium at node i, step t] = GAMMA_PRICE * E[max(Z,0)] * P(node i flagged).
    // At the 85% loading the instance generator leaves, a line trips with probability up to
    // ~0.56, which puts the uplift near $4.5/MWh -- larger than the sigma * lambda_da spread on
    // the low-volatility tracks, and unlike that spread it is node-specific. Omitting it leaves
    // the water-value table blind to exactly the nodes where discharging pays most.
    //
    // Step 0's prices are generated with an all-false indicator vector, so its premium is zero.
    let flow_limits = &challenge.network.flow_limits;
    let lines = &challenge.network.lines;
    let cong_lines = num_lines.min(lines.len());
    let tau = challenge.network.congestion_threshold.max(1e-9);
    let premium_scale = cong_optimism * GAMMA_PRICE * E_HALF_NORMAL;
    let mut premium = vec![0.0f64; num_steps * num_nodes];
    // Size of the premium conditional on it being nonzero, and the probability of that branch.
    // The premium is zero unless the node is flagged *and* Z > 0, and those two draws are
    // independent, so the branch has probability P(flagged) / 2 and mean size
    // GAMMA_PRICE * E[Z | Z > 0]. The product reproduces `premium` exactly, so this is a
    // mean-preserving refinement of the same quantity, not a second opinion about it.
    let prem_c = cong_optimism * GAMMA_PRICE * E_HALF_NORMAL_POS;
    let mut cong_prob = vec![0.0f64; num_steps * num_nodes];

    // --- Mean line headroom left over by the exogenous flows, per line.
    //
    // This is the ingredient the DP was missing. `soc_action_bounds` below used to hand the
    // sweep each battery's *nameplate* power, so the table was the value function of a battery
    // that can fill from E^min to E^max in about three hours and empty just as fast. The
    // instance generator rescales the exogenous injections until the worst line sits at 85% of
    // its limit, so the headroom a battery actually gets is a fraction of one line limit shared
    // with the rest of the fleet -- often a small fraction of nameplate on the tight tracks.
    // A table built on nameplate power therefore prices energy as if the battery could idle
    // through the morning and still slam a full cycle into the evening peak, so it tells the
    // battery to wait; the online allocator then cannot deliver the slam and the cycle never
    // happens. Planning with the power the network really allows makes the same table start
    // charging early and hold charge through a long discharge, which is executable.
    //
    // The online step is left on the true `state.action_bounds`, so nothing here caps what is
    // actually played -- only how energy is valued.
    //
    // The horizon mean sets the planning power, and it is deliberately one scalar per battery
    // rather than a per-step profile. An earlier version scaled it step by step by
    // `min_l headroom_l(t) / pressure_l`, on the theory that the exogenous injections are diurnal
    // sinusoids and so leave the grid open overnight and saturated at the load peaks. The
    // generator does not work that way: each nodal injection is
    // `200 * (U(0,1) - 0.5) + pattern + noise`, so an i.i.d. U(-100, 100) MW draw per node and
    // step (sd ~58 MW) dominates the diurnal `pattern` term (sd ~17 MW). On top of that the
    // generator rescales each timestep until the *worst* line sits at exactly 85% of its limit, so
    // the tightest line's headroom is near constant in t and what actually varies is which line is
    // tightest -- a draw from the per-step noise, not a property of the hour. Normalised to mean
    // one and clamped into a 16x band, that quantity multiplied the DP's planning power by an
    // essentially arbitrary factor at every step, telling the sweep to time its cycle around
    // whichever line won a noise lottery. It cost score on all five tracks at once, which is the
    // signature of injected noise rather than of a tradeoff.
    let mut headroom = vec![0.0f64; num_lines];

    {
        let mut survive = vec![1.0f64; num_nodes];
        for t in 0..num_steps {
            let flows = &exo_flows[t];
            for l in 0..num_lines.min(flows.len()) {
                headroom[l] += (flow_limits[l] * flow_safety - flows[l].abs()).max(0.0);
            }
            // The congestion indicator for step t + 1 is drawn from step t's exogenous flows,
            // so the same flow vector serves both sweeps. Step 0's prices are generated with an
            // all-false indicator vector, which is why row 0 stays zero.
            if premium_scale <= 0.0 || t + 1 >= num_steps {
                continue;
            }
            // survive[i] = prod over lines incident to i of (1 - p_l), i.e. P(i not flagged).
            for s in survive.iter_mut() {
                *s = 1.0;
            }
            for l in 0..cong_lines.min(flows.len()) {
                let limit = flow_limits[l];
                if limit <= 0.0 {
                    continue;
                }
                let ratio = flows[l].abs() / (tau * limit);
                let p = ratio.powi(CONG_EXP).min(1.0);
                if p <= 0.0 {
                    continue;
                }
                let (from, to) = lines[l];
                if from < num_nodes {
                    survive[from] *= 1.0 - p;
                }
                if to < num_nodes {
                    survive[to] *= 1.0 - p;
                }
            }
            let row = (t + 1) * num_nodes;
            for i in 0..num_nodes {
                let p_flag = 1.0 - survive[i];
                premium[row + i] = premium_scale * p_flag;
                cong_prob[row + i] = 0.5 * p_flag;
            }
        }
        let inv = 1.0 / num_steps.max(1) as f64;
        for h in headroom.iter_mut() {
            *h *= inv;
        }
    }

    // --- Fleet contention for the idiosyncratic price class, per step.
    //
    // The sweep splits each step's discharge grant across a priority class: a battery whose node
    // draws a congestion premium or a price jump outbids the fleet, so conditional on having one
    // it competes only inside that class and clears with probability `q_d / pi`, where `pi` is the
    // class's share of the fleet. The sweep took `pi` to be *this node's own* probability of having
    // such an event, which is the share of the fleet only if the events are independent across
    // nodes. The congestion premium is not:
    //
    //     let z_prime = normal.sample(rng);
    //     let zeta = z_prime.max(0.0);
    //     for i in 0..num_nodes { ... if congestion_indicators[i] { price += GAMMA_PRICE * zeta } }
    //
    // `zeta` is drawn once per step, outside the node loop, and added to every flagged node. So the
    // premium is a *common* shock over the flagged set, and the flags themselves come from one
    // exogenous flow vector tested line by line, so a step that flags this node flags everything
    // incident to every other loaded line at the same time. When a battery's premium fires, so does
    // every other flagged battery's, and the class it joins is the whole flagged share of the fleet
    // rather than its own.
    //
    // That is the difference between "my node spikes and I go to the front of a queue of one" and
    // "half the fleet spikes with me and the queue is unchanged". Sizing the class from the node's
    // own probability therefore overstates the clearing rate on precisely the atoms the table holds
    // charge for: a node at `p_flag = 0.9` came out with `pi ~ 0.47` and claimed to clear at
    // `q_d / 0.47`, twice the rate the replay measured, every premium window of the horizon. With
    // the max inside the expectation that is not a mild mis-level -- it is the option value the
    // sweep prices waiting for, so the battery holds a cycle through deliverable shoulder hours for
    // a premium window it will be rationed in along with everyone else.
    //
    // The class share is measured instead: the fraction of the *fleet's* nodes flagged at this
    // step, which the exogenous path already determines for the whole horizon. `cong_prob` holds
    // `P(flagged) / 2`, the premium branch's probability (flagged *and* `zeta > 0`), so the flagged
    // share alone is twice it -- and that is the right conditional share, because the `zeta > 0`
    // half of the event is shared by the whole fleet rather than drawn per node.
    //
    // One `O(steps x batteries)` pass at plan time, used only to raise `pi` where the fleet is more
    // contended than the node's own event rate. Where it is not -- a node uniquely exposed on an
    // otherwise open grid -- the class stays the node's own and the sweep is bit-identical.
    let mut contention = vec![0.0f64; num_steps];
    if num_batteries > 0 && premium_scale > 0.0 {
        let inv_m = 1.0 / num_batteries as f64;
        for t in 0..num_steps {
            let row = t * num_nodes;
            let mut flagged = 0.0f64;
            for bat in challenge.batteries.iter() {
                let n = if bat.node < num_nodes { bat.node } else { 0 };
                flagged += 2.0 * cong_prob[row + n];
            }
            contention[t] = (flagged * inv_m).clamp(0.0, 1.0);
        }
    }

    // --- The plan-time replay's price path: the idiosyncratic branches as events, not as means.
    //
    // `measure_delivery` used to price a step at
    // `day_ahead_prices[t][node] * (1 + jump_optimism * rho_jump * pareto_mean) + premium[t][node]`,
    // i.e. the day-ahead curve with the Pareto jump folded into its mean multiplier and the
    // congestion premium folded into its mean uplift. The sweep itself refuses to do that with the
    // premium, and the argument it refuses on applies here word for word: the premium is exactly
    // zero on most steps and ~$16/MWh on a few, a jump multiplies one node's price by a Pareto draw
    // on `rho_jump` of its steps, and with a max inside the expectation a path that is uniformly
    // slightly better is a *different problem* from one that is usually ordinary and occasionally
    // excellent.
    //
    // The same path is what the delivery ratio is measured on, and there the old one was biased in
    // a definite direction. Contention is set by how much of the fleet wants the same sign at the
    // same instant, and on a path whose only variation is the shared diurnal day-ahead curve the
    // answer is "all of it": every battery charges in the same hours and discharges in the same
    // hours, so the measured ratio understates the access a real rollout grants, and that ratio is
    // the service probability the final sweep derates every table with. A premium or a jump is the
    // one thing in this challenge that moves a single node against the fleet, so realising them is
    // what puts cross-sectional spread -- and with it the counter-flow pairing the tight tracks
    // actually trade on -- into the measurement.
    //
    // Placement is Bresenham-stratified, not sampled: one accumulator per node carries the branch
    // probability and fires whenever it crosses one. So the horizon's event count is the expected
    // count to within a single event, the events land where the probability is actually high, and
    // no RNG enters a plan that has to be reproducible across the policy's calls. The offsets are
    // the golden-ratio and sqrt(2) fractional sequences in the node index, which is what
    // decorrelates the firing *across* nodes -- a path that fires every node on the same step has
    // no cross-section, which is the whole point.
    //
    // Mean-preserving per node and over the horizon, by construction. `cong_prob` is the premium
    // branch's probability and `prem_c` its conditional size, and `prem_c * cong_prob = premium`
    // exactly, because `E_HALF_NORMAL_POS = 2 * E_HALF_NORMAL` against the branch's `P(flagged)/2`;
    // the jump atom's size times `rho_jump` is `jump_mult - 1`. So the level of the path is
    // unchanged and only the placement of that mass in time and across nodes is new. With
    // `rho_jump = 0` and no premium risk anywhere the path is bit-identical to the old one.
    let mut forecast = vec![0.0f64; num_steps * num_nodes];
    {
        let jump_size = jump_optimism * pareto_mean;
        let mut acc_p = vec![0.0f64; num_nodes];
        let mut acc_j = vec![0.0f64; num_nodes];
        for i in 0..num_nodes {
            let g = (i as f64 + 1.0) * 0.618_033_988_749_894_8;
            acc_p[i] = g - g.floor();
            let h = (i as f64 + 1.0) * 0.414_213_562_373_095_1;
            acc_j[i] = h - h.floor();
        }
        for t in 0..num_steps {
            let row = t * num_nodes;
            let da_row = &challenge.market.day_ahead_prices[t];
            for i in 0..num_nodes {
                let da = if i < da_row.len() { da_row[i] } else { 0.0 };
                let mut p = da;
                acc_p[i] += cong_prob[row + i];
                if acc_p[i] >= 1.0 {
                    acc_p[i] -= 1.0;
                    p += prem_c;
                }
                acc_j[i] += rho_jump;
                if acc_j[i] >= 1.0 {
                    acc_j[i] -= 1.0;
                    p += da * jump_size;
                }
                forecast[row + i] = p;
            }
        }
    }

    // First-pass estimate only: each battery is assumed to get 1/sqrt(m) of the headroom it could
    // consume on its own. The second pass below replaces this with the sharing the allocator is
    // measured to produce, so what matters here is only that the starting tables be plausible
    // enough to drive that measurement.
    // Worst-case sharing would divide by m (every battery pushing the same line the same way);
    // perfect cancellation would not divide at all. Signs across a heterogeneous fleet are
    // closer to independent, and the net footprint of m independent actions on one line grows
    // like sqrt(m), so sqrt(m) is the sharing rule that matches. It also keeps the bound
    // per-battery rather than fleet-wide: dividing line l's headroom by |ptdf[l][node_b]|
    // leaves a battery that barely moves the tight line effectively unconstrained.
    let fleet_share = power_share / (num_batteries.max(1) as f64).sqrt();

    // --- Calibration sweep: a coarse table per battery, built only to be replayed.
    //
    // The sweep below used to run at the full `bins`, and then the second pass re-ran a second
    // full sweep for every battery whose measured service probability differed from the analytic
    // guess -- which on the tight tracks is most of them. So the dominant plan-time cost was paid
    // twice, and `batteries x steps x bins x atoms x candidates` is 26M multiply-adds per sweep on
    // dense and on capstone.
    //
    // The first of those two sweeps is not what the policy plays. Its only consumer is
    // `measure_delivery`, which runs the allocator against it for the whole horizon and reduces
    // each battery's entire table to one number: the fraction of the power it asked for that
    // cleared the lines. A ratio of asked-for to delivered MW does not need the SOC grid resolved
    // to 2% of span -- it needs the battery to ask for roughly the right action in roughly the
    // right window, and the action set is exact at any grid size (`aligned_actions` enumerates the
    // breakpoints of the piecewise-linear objective, so the inner max is exact whether the grid has
    // 13 nodes or 49). What a coarse grid costs is resolution in the value *function*, i.e. the
    // timing of the cycle, and the replay averages over 96 or 192 steps of it.
    //
    // So the calibration runs at CAL_BINS and the price model, the planning power and the
    // allocator it replays are all left exactly as they were, which keeps the measured ratio
    // comparable to what the full-resolution pass produced. Then every battery gets exactly one
    // full-resolution sweep with that ratio spent as its service probability, instead of between
    // one and two. The arithmetic is plan-time only; nothing in the online step changes.
    let cal_bins = CAL_BINS.min(bins).max(2);

    // Full service at every step: the calibration tables exist only to drive the replay that
    // measures what service really is.
    let full_service = vec![1.0f64; num_steps];

    let mut cal_batteries: Vec<BatteryPlan> = Vec::with_capacity(num_batteries);
    // Nameplate and first-pass planning power per battery, kept for the calibration below.
    let mut nameplate: Vec<(f64, f64)> = Vec::with_capacity(num_batteries);
    let mut plan_power: Vec<(f64, f64)> = Vec::with_capacity(num_batteries);
    for bat in challenge.batteries.iter() {
        let lo = bat.soc_min_mwh;
        let hi = bat.soc_max_mwh;
        let span = (hi - lo).max(1e-9);
        let cal_dk = span / (cal_bins - 1) as f64;
        let cal_top = lo + cal_dk * (cal_bins - 1) as f64;
        let cap = bat.capacity_mwh.max(1e-9);
        let eta_c = bat.efficiency_charge.max(1e-6);
        let eta_d = bat.efficiency_discharge.max(1e-6);
        let p_c = bat.power_charge_mw.max(0.0);
        let p_d = bat.power_discharge_mw.max(0.0);
        let node = if bat.node < num_nodes { bat.node } else { 0 };

        // A battery at the slack bus has no effect on flows; ptdf's slack column is zero,
        // so this picks that up automatically.
        let mut sens = vec![0.0f64; num_lines];
        // `solo` = largest |u| this battery could sustain alone against the mean headroom.
        let mut solo = f64::INFINITY;
        for l in 0..num_lines {
            let ptdf_row = &challenge.network.ptdf[l];
            if node < ptdf_row.len() {
                sens[l] = ptdf_row[node];
            }
            let a = sens[l].abs();
            if a > 1e-9 {
                let bound = headroom[l] / a;
                if bound < solo {
                    solo = bound;
                }
            }
        }
        // Power the DP plans with: the shared network bound, floored so the table keeps its
        // shape, and never above nameplate.
        // A battery at the slack bus has an all-zero PTDF column and so leaves `solo` infinite;
        // `power_share = 0` turns the correction off. Both cases must fall through to nameplate
        // rather than multiply out to a NaN.
        let net_power = if solo.is_finite() && fleet_share > 0.0 {
            solo * fleet_share
        } else {
            f64::INFINITY
        };
        let p_c_dp = p_c.min(net_power.max(MIN_POWER_FRACTION * p_c));
        let p_d_dp = p_d.min(net_power.max(MIN_POWER_FRACTION * p_d));

        let da_path = price_path(challenge, node, num_steps);
        let prem_path = premium_path(&premium, node, num_nodes, num_steps);
        let cprob_path = premium_path(&cong_prob, node, num_nodes, num_steps);
        let cal_values = sweep_battery(
            num_steps,
            cal_bins,
            lo,
            cal_dk,
            cal_top,
            cap,
            eta_c,
            eta_d,
            p_c_dp,
            p_d_dp,
            &full_service,
            &full_service,
            &da_path,
            &prem_path,
            &cprob_path,
            &contention,
            prem_c,
            &atom_mult,
            &atom_weight,
        );

        nameplate.push((p_c, p_d));
        plan_power.push((p_c_dp, p_d_dp));
        cal_batteries.push(BatteryPlan {
            lo,
            dk: cal_dk,
            cap,
            eta_c,
            eta_d,
            node,
            values: cal_values,
            sens,
        });
    }

    // --- Final sweep: spend the measured delivery ratio as a service probability, at nameplate
    // power, instead of as a power derate.
    //
    // The level the calibration sweep planned with is an analytic worst case: the tightest line's mean headroom divided by this
    // battery's PTDF entry, times a 1/sqrt(m) guess at the fleet's sharing of it. Nothing in that
    // chain knows whether the network ever actually binds. The delivery ratio is the same quantity
    // measured instead of bounded: replay the allocator at plan time against the exogenous
    // injections (known for the whole horizon) and the day-ahead forecast, and record what
    // fraction of the power each battery asked for actually cleared the lines.
    //
    // What that ratio *means* is the change here. Read as a derate it says the battery can always
    // move `r x` nameplate, so the sweep plans a slow steady cycle: charge a little every step
    // through the night, discharge a little every step through the peak. The allocator does not
    // behave like that. It walks the fleet in descending value and gives each battery as much as
    // the remaining headroom allows, so on a tight track a battery is denied outright on most
    // steps and served at or near full power on a few. Those are different problems because the
    // max sits inside the expectation -- the same reason the congestion premium had to enter as
    // its own branch rather than as an uplift. A derated table asks for an action no allocator
    // ever delivers and values energy as if a cycle took all day; an intermittent-service table
    // asks for the full-power action that a value-ordered greedy actually hands to its highest
    // bidder, and holds SOC positioned to use it.
    //
    // So the final sweep plans at true nameplate power with the measured ratio spent as a
    // service probability, and that probability is measured per direction. The replay records
    // how much of the power a battery asked for cleared the lines on the steps where it wanted
    // to charge, and separately on the steps where it wanted to discharge, and those are
    // disjoint sets of steps chosen by the price: charging happens overnight and at price
    // collapses, discharging at the diurnal peak and into spikes. The fleet's desired sign is
    // strongly correlated -- nodal prices share a diurnal curve and a 0.7-correlated common
    // shock -- so the fleet is pushing the grid one way on a battery's charge steps and the
    // other way on its discharge steps, and a battery's own PTDF column has a fixed sign. Its
    // access to the two directions is therefore not the same number, and pooling them into one
    // scalar averages two ratios measured on different regimes. That over-values the direction
    // the network actually grants and under-values the one it denies, in a value function whose
    // whole job is to decide which of the two to hold out for.
    //
    //     W[t][e] = sum_regimes P(regime) * E[max_{u in regime} (reward + W[t+1][e'])],
    // over the four independent grants (both directions, discharge only, charge only, neither),
    // with idle always available. Where the inner optimum is one-directional -- all but a
    // measure-zero set of prices -- setting q_c = q_d = r reproduces the single-scalar model
    // `r * E[max_u ...] + (1 - r) * W[t+1][e]` exactly, so the entire effect of this change is
    // the asymmetry the scalar could not express. Expected throughput in each direction is that
    // direction's measured ratio times nameplate, so it is still a mean-preserving reading of
    // the measurement rather than a second opinion about it, and it still makes the DP's
    // transition model the one the policy really plays: online actions are bounded by
    // `state.action_bounds`, i.e. nameplate and SOC, never by a derate. Where the network never
    // binds both ratios are one and the service branches vanish.
    //
    // Every battery is swept exactly once at the full `bins`, with the measured ratio as its
    // service probability. The earlier arrangement -- a full-resolution first sweep, then a second
    // full-resolution sweep for every battery whose measurement disagreed with the analytic guess
    // -- paid the dominant plan-time cost twice on the tracks where that cost is largest, and the
    // first of the two was never the table the policy played.
    let (ratio, block_ratio, moment2) = if num_batteries > 0 && num_lines > 0 {
        let (r, b, s, _) = measure_delivery(
            challenge,
            &cal_batteries,
            &nameplate,
            &exo_flows,
            &forecast,
            num_nodes,
            num_lines,
            num_steps,
            cal_bins,
            flow_safety,
        );
        (r, b, s)
    } else {
        (
            vec![(-1.0f64, -1.0f64); num_batteries],
            vec![Vec::new(); num_batteries],
            vec![(-1.0f64, -1.0f64); num_batteries],
        )
    };

    let mut batteries: Vec<BatteryPlan> = Vec::with_capacity(num_batteries);
    for (b, cal) in cal_batteries.into_iter().enumerate() {
        // The calibration plan already carries everything about this battery except its table:
        // the grid floor, the capacity, the efficiencies, the node and the PTDF column. Its `dk`
        // is the coarse spacing, so the SOC span is recovered from it rather than re-derived from
        // the challenge.
        let BatteryPlan {
            lo,
            dk: cal_dk,
            cap,
            eta_c,
            eta_d,
            node,
            values: _,
            sens,
        } = cal;
        let span = cal_dk * (cal_bins - 1) as f64;
        let dk = span / (bins - 1) as f64;
        let top = lo + dk * (bins - 1) as f64;

        // A negative ratio on one side means the battery never asked to move in that direction,
        // so there is no ratio to read out of 0/0 and the other side's measurement -- the only
        // evidence there is about how the network treats this node -- stands in for it. A
        // negative ratio on both sides means the battery never asked to move at all; that one
        // falls back to the analytic planning power at full service, which is what the
        // first-pass table used to give it.
        let (r_c, r_d) = ratio[b];
        let (f_c0, f_d0) = if b < moment2.len() {
            moment2[b]
        } else {
            (-1.0f64, -1.0f64)
        };
        // A battery that never asked in one direction has no ratio and no second moment there, so
        // the other direction's measurement -- the only evidence about how the network treats this
        // node -- stands in for both. The two quantities are measured on the same steps with the
        // same convention, so substituting them together keeps the pair consistent.
        let (r_c, r_d, s_c, s_d) = if r_c < 0.0 && r_d < 0.0 {
            (-1.0f64, -1.0f64, -1.0f64, -1.0f64)
        } else if r_c < 0.0 {
            (r_d, r_d, f_d0, f_d0)
        } else if r_d < 0.0 {
            (r_c, r_c, f_c0, f_c0)
        } else {
            (r_c, r_d, f_c0, f_d0)
        };
        // --- Split the measured throughput into how often and how deep.
        //
        // The replay measures one number per direction, `r` = cleared MW / asked-for MW over the
        // horizon, and every version of this solver has spent it as one of two things. Read as a
        // *derate* it says the battery always moves `r x nameplate`, so the sweep plans a slow
        // steady cycle. Read as a *service probability* at nameplate power it says the battery
        // moves everything it asked for on a fraction `r` of the steps and nothing on the rest.
        // Both were shipped globally and both lost -- the derate on the tracks where the allocator
        // really does deny outright, the probability on the tracks where it really does shave
        // every step -- because `r` alone cannot distinguish them. It is a mean, and the two
        // readings are two different distributions with that mean.
        //
        // Which reading it is used to be decided by a threshold frequency -- the fraction of
        // asking steps on which the battery cleared at least half of what it asked for, with
        // `q = max(f, r)` -- and that statistic has the mapping backwards in the regime the tight
        // tracks live in. The allocation the lookahead picks when a large fleet competes for little
        // headroom is usually the uniform shrink or the top-up built on it, so the battery is
        // shaved by one fleet-wide factor on *every* step. Shaved to 0.1 it clears under half its
        // ask every time, so `f = 0`, `max(f, r) = r`, and the split came out `q = 0.1, depth = 1`
        // -- the intermittent reading, read off a measurement that is the definition of a uniform
        // derate. The sweep then planned a full-power slam the allocator never grants, which is
        // exactly the failure the service model exists to remove, and it did so on the two tracks
        // where the shrink factor is smallest.
        //
        // The second moment of the same per-step fractions separates the two readings exactly, and
        // it is the only extra statistic needed -- see `service_split`, where the derivation and
        // all three corners (uniform shave, all-or-nothing, unbinding network) are worked through.
        // The replay accumulates it on the same ask-weighting as the ratio itself, so it costs one
        // multiply-add per battery per step and nothing online.
        //
        // `MIN_SERVICE` is still floored, still on `q` alone and with the depth re-derived from the
        // measurement afterwards, so `q x depth = r` holds whatever the floor does and the floor
        // buys shape rather than level. The old block then floored the *power* at
        // `MIN_POWER_FRACTION` of nameplate after that division, which broke the product in the
        // one place it mattered: a battery measured at `r = 0.003` came out planning
        // `q = 0.1, depth = 0.1`, more than three times the energy the replay recorded it moving,
        // and the overstatement ran to tenfold as `r` fell. A moment-matched depth cannot collapse
        // without `r` collapsing with it -- `depth = E[phi^2] / E[phi] >= E[phi]` by
        // Cauchy-Schwarz -- so that floor is gone and planned throughput is the measurement
        // everywhere.
        let (pw_c, pw_d, q_c, q_d) = if r_c < 0.0 {
            let (old_c, old_d) = plan_power[b];
            (old_c, old_d, 1.0f64, 1.0f64)
        } else {
            let (p_c, p_d) = nameplate[b];
            let (qc, depth_c) = service_split(r_c, s_c);
            let (qd, depth_d) = service_split(r_d, s_d);
            (p_c * depth_c, p_d * depth_d, qc, qd)
        };

        // --- Service profile: when, not just how much.
        //
        // The horizon ratio tells a battery what fraction of the power it asks for the lines let
        // through, and the sweep spends that as a per-step probability of being granted. Flat in
        // time, that says the network is as likely to clear a sale into the evening peak as one
        // three hours before it. The replay says otherwise, and for a reason that is structural
        // rather than incidental: nodal day-ahead prices share one diurnal base curve, so the
        // whole fleet wants to charge in the same hours and discharge in the same hours, and
        // correlated injections add in every PTDF column. Contention is therefore highest exactly
        // at the trough and the peak -- the windows a flat table aims every battery at -- and the
        // shoulders on either side are comparatively open.
        //
        // A flat table consequently plans a cycle it cannot execute: it tells the battery to hold
        // through the morning and sell into the peak, the allocator denies most of the fleet
        // there, and the shoulder hours where the sale would have cleared pass with the battery
        // still holding. Giving the sweep the measured shape instead makes it start the discharge
        // early and finish it late, which is what the allocator will actually grant.
        //
        // The profile is used for shape only: it is rescaled to the horizon ratio, which is the
        // better-measured quantity, so expected throughput is unchanged and only its placement in
        // time moves. Where no line binds every block measures one, the rescale is the identity,
        // and the table is bit-identical to one swept with no service model at all.
        let mut qc_path = vec![q_c; num_steps];
        let mut qd_path = vec![q_d; num_steps];
        let blocks: &[(f64, f64)] = if b < block_ratio.len() {
            &block_ratio[b]
        } else {
            &[]
        };
        if r_c >= 0.0 && !blocks.is_empty() {
            let nblocks = blocks.len();
            for t in 0..num_steps {
                let (bc, bd) = blocks[(t / SERVICE_BLOCK).min(nblocks - 1)];
                if bc >= 0.0 {
                    qc_path[t] = bc;
                }
                if bd >= 0.0 {
                    qd_path[t] = bd;
                }
            }
            rescale_service(&mut qc_path, q_c);
            rescale_service(&mut qd_path, q_d);
        }

        let da_path = price_path(challenge, node, num_steps);
        let prem_path = premium_path(&premium, node, num_nodes, num_steps);
        let cprob_path = premium_path(&cong_prob, node, num_nodes, num_steps);
        let values = sweep_battery(
            num_steps,
            bins,
            lo,
            dk,
            top,
            cap,
            eta_c,
            eta_d,
            pw_c,
            pw_d,
            &qc_path,
            &qd_path,
            &da_path,
            &prem_path,
            &cprob_path,
            &contention,
            prem_c,
            &atom_mult,
            &atom_weight,
        );
        batteries.push(BatteryPlan {
            lo,
            dk,
            cap,
            eta_c,
            eta_d,
            node,
            values,
            sens,
        });
    }

    // --- Removed: the reference-price second opinion.
    //
    // This used to build a second table per battery -- `W[t][e] = r_t * e`, the three-hour
    // day-ahead threshold rule written as a water value -- replay both it and the sweep's tables
    // through the whole allocator over the full horizon, and play whichever earned more on that
    // forecast path (with a margin on the DP side for the dispersion the path leaves out).
    //
    // It is measured dead. Forcing the reference tables on every track instead of letting the
    // replay choose moved every one of the five scores a long way -- mean -39.4% against +46.8%,
    // with the loosest track at -116.6%, i.e. a policy that loses money -- and a track where the
    // replay already selected the reference side could not have moved at all. So the comparison
    // picks the sweep's tables on all five tracks, and what it costs to reach that conclusion is
    // two full-horizon allocator replays at the full `bins`: 192 steps x (100 batteries x 300
    // lines) x ~6 passes is about 35M multiply-adds each on the capstone track, against the 19M
    // the sweep itself spends there.
    //
    // That fuel is what `MAX_BINS` is now spending on resolution in the marginal water value --
    // the one quantity every action in this solver is chosen by -- which is a quantity the dead
    // comparison could only ever degrade, because raising `bins` made those two replays more
    // expensive too. `reference_values` is kept for a future iteration that wants a second
    // opinion built some other way; nothing calls it.
    //
    // Nothing between the sweep and the plan now reads the full-resolution tables, so the only
    // replay left is the one at `CAL_BINS` whose measurement the sweep actually consumes.

    Plan {
        seed: challenge.seed,
        bins,
        num_steps,
        num_lines,
        flow_safety,
        batteries,
        exo_flows,
    }
}

/// Day-ahead price path at one node.
fn price_path(challenge: &Challenge, node: usize, num_steps: usize) -> Vec<f64> {
    (0..num_steps)
        .map(|t| challenge.market.day_ahead_prices[t][node])
        .collect()
}

/// Expected congestion premium path at one node.
fn premium_path(premium: &[f64], node: usize, num_nodes: usize, num_steps: usize) -> Vec<f64> {
    (0..num_steps)
        .map(|t| premium[t * num_nodes + node])
        .collect()
}

/// Reference-price water values for one battery: `W[t][e] = r_t * e`, linear in stored energy,
/// with `r_t` the mean forecast price over the next `REF_LOOKAHEAD` steps at this battery's node,
/// tapered to zero over the final drain window so the table liquidates by the horizon.
/// Same layout as `sweep_battery`: `(num_steps + 1) * bins`, row-major by step, terminal row zero.
///
/// A linear table is a threshold rule. The per-step optimiser maximises
/// `u * price * DT - kappa_tx |u| DT - deg + r * soc_next`, and `soc_next` is affine in `u` with
/// slope `eta_c * DT` when charging and `-DT / eta_d` when discharging, so charging pays while
/// `price < r * eta_c - kappa_tx`, discharging pays while `price > r / eta_d + kappa_tx`, and the
/// battery idles between the two. The gap between those bounds is the round trip's two
/// transaction fees plus its 9.75% efficiency loss -- about $5/MWh at a $50 reference -- so the
/// threshold is the challenge's own greedy day-ahead rule with its margin read off the physics
/// rather than chosen. Energy is never worth more than the reference price, so the table prices
/// no option to wait at all, which is exactly the difference from the sweep.
///
/// Writing the rule as a table rather than as a second policy is the point: the candidate
/// generation, the four-way allocation, the counter-flow pairing and the lookahead that chooses
/// between them are all driven by a table of this shape and need no change to run on it.
///
/// But a window mean has no terminal boundary, and this table is the one half of the menu that is
/// missing one. `sweep_battery` gets its boundary for free -- row `num_steps` is zero, so the last
/// rows of the sweep price stored energy at what is left to sell it for and the battery is driven
/// to its floor by the end. The window mean instead prices a full battery at `r_{H-1} = $50/MWh`
/// at the very last step, where there is no step after it to sell into. The threshold rule that
/// comes out of that has no reason to empty: it idles whenever the realised price sits inside its
/// $5/MWh band, right up to the horizon, and it will still *buy* on the final steps at any price
/// under `r * eta_c - kappa_tx`.
///
/// What that leaves behind is not a rounding error, because every battery is handed a free
/// endowment: `soc_initial = 0.50 * cap` against a floor of `E_MIN_FRAC = 0.10 * cap`, so
/// `0.40 * cap` MWh per battery is energy the episode starts with and no one paid for. Selling it
/// at any positive price is pure profit -- `0.4 * cap * eta_d` MWh delivered, about $2.3k per
/// battery at a $50 mean on the wide fleets -- and on the two tracks with the largest fleets and
/// the largest capacities that is of the order of the whole deficit this solver carries.
///
/// The taper is the boundary written in the only form a linear table can hold. The marginal MWh of
/// store can only be realised if there is time left to push it out, and with `rem = num_steps - t`
/// steps left at most `rem * p_d * DT / eta_d` MWh of store can be delivered, so the span is fully
/// deliverable only while `rem >= drain_steps = span / (p_d * DT / eta_d)`. Scaling the window mean
/// by `min(1, rem / drain_steps)` keeps the table linear in `e` -- hence `interp` exact on it and
/// the threshold reading intact -- while sending both thresholds to zero as the horizon arrives:
/// the discharge threshold `slope / eta_d + kappa_tx` falls below any plausible price, so the
/// battery dumps at full power, and the charge threshold `slope * eta_c - kappa_tx` falls below
/// zero, so it stops buying energy it can never sell.
///
/// `drain_steps` is about 12 steps at this generator's parameters -- the generator ties power to
/// capacity, so `span / (p_d * DT / eta_d) = 0.8 * cap / (0.25 * cap / 4 / 0.95)` is the same
/// number for every battery regardless of size -- so the taper is the identity on all but the last
/// three hours of a 96- or 192-step horizon, and it is the identity everywhere for a battery with
/// no discharge power at all. Nothing about the sweep's tables, the allocator or the online step
/// is touched; where the plan-time replay keeps choosing the sweep, this changes nothing.
#[allow(dead_code)]
fn reference_values(
    num_steps: usize,
    bins: usize,
    lo: f64,
    dk: f64,
    da_path: &[f64],
    prem_path: &[f64],
    jump_mult: f64,
    p_d: f64,
    eta_d: f64,
) -> Vec<f64> {
    let mut values = vec![0.0f64; (num_steps + 1) * bins];
    // Steps of full-power discharge needed to run the usable span down to its floor. A battery
    // with no discharge power, or a degenerate grid, leaves this at zero and the taper off.
    let span = dk * (bins.max(1) - 1) as f64;
    let per_step = if eta_d > 0.0 { p_d * DT / eta_d } else { 0.0 };
    let drain_steps = if per_step > 1e-9 && span > 0.0 {
        span / per_step
    } else {
        0.0
    };
    for t in 0..num_steps {
        let last = (t + REF_LOOKAHEAD).min(num_steps);
        let mut sum = 0.0f64;
        let mut n = 0usize;
        for s in t..last {
            // Expected real-time price over the window, not the day-ahead one: the jump branch
            // contributes `rho_jump * alpha/(alpha - 1)` of the day-ahead price in the mean, which
            // is most of the round-trip band this rule's thresholds are separated by on the
            // heavy-tailed tracks. Leaving it out made both thresholds too low against the prices
            // the rule is actually compared with, so the battery sold its endowment early and
            // stopped buying.
            sum += da_path[s] * jump_mult + prem_path[s];
            n += 1;
        }
        let mean = if n > 0 { sum / n as f64 } else { 0.0 };
        // One outside the final drain window, falling linearly to `1 / drain_steps` at the last
        // step, so the slope the rule reads is the part of the span that is still deliverable.
        let taper = if drain_steps > 1e-9 {
            ((num_steps - t) as f64 / drain_steps).min(1.0)
        } else {
            1.0
        };
        let slope = mean * taper;
        for k in 0..bins {
            values[t * bins + k] = slope * (lo + k as f64 * dk);
        }
    }
    values
}

/// Backward stochastic DP for one battery: returns `(num_steps + 1) * bins` water values,
/// row-major by time step, with the terminal row left at zero.
///
/// `p_c_dp` / `p_d_dp` are the power the plan assumes is executable, one scalar each for the whole
/// horizon; the caller has already placed them inside the `[MIN_POWER_FRACTION x nameplate,
/// nameplate]` band.
///
/// `serve_c` / `serve_d` are the probabilities the network lets this battery charge and discharge
/// on a given step, measured separately. The battery either gets the power it asks for in that
/// direction or gets nothing, and the two grants are independent, so each step falls into one of
/// four regimes and
///     W[t][e] = sum_regimes P(regime) * E_lambda[ max_{u in regime} (reward + W[t+1][e']) ],
/// where the regime's action set is both directions plus idle, one direction plus idle, or idle
/// alone. Idle is always available and its value is the continuation `W[t+1][e]`, already
/// computed as candidate 0. `serve_c = serve_d = 1` recovers the unconstrained sweep exactly.
///
/// With `serve_c = serve_d = q` this reproduces the single-scalar model
/// `q * E[max_u ...] + (1 - q) * W[t+1][e]` whenever the inner optimum is one-directional, which
/// it is at all but a measure-zero set of prices: the regime that grants only the direction the
/// battery does not want contributes the idle value, exactly as a flat denial would. So the
/// refinement is isolated to the case the single scalar could not express, `serve_c != serve_d`.
///
/// `serve_d[t]` is then resolved across the step's price atoms rather than spent flat on all of
/// them, because the allocator it models serves bidders in descending marginal value and so
/// grants by *rank*, not by price level. The congestion-premium and jump atoms are node-specific
/// and large; the five diffusive atoms move the whole fleet together. See the derivation at the
/// split below. The split is mean-preserving in expected throughput and is the identity both
/// where the network does not bind and where the node has no idiosyncratic branch at all.
///
/// `contend_path[t]` is the share of the fleet whose node is flagged congested at step `t`, and it
/// sizes the priority class the premium atoms put this battery into. The premium's `zeta` is one
/// draw per step shared by every flagged node, so a premium window is common to the flagged set
/// rather than exclusive to one node, and the class clears at `q_d / class` rather than at
/// `q_d / P(this node flagged)`. Only the class size moves; how often this node is in the class is
/// still its own event probability, so expected throughput per step is the measured `q_d` exactly.
fn sweep_battery(
    num_steps: usize,
    bins: usize,
    lo: f64,
    dk: f64,
    top: f64,
    cap: f64,
    eta_c: f64,
    eta_d: f64,
    p_c_dp: f64,
    p_d_dp: f64,
    serve_c: &[f64],
    serve_d: &[f64],
    da_path: &[f64],
    prem_path: &[f64],
    cprob_path: &[f64],
    contend_path: &[f64],
    prem_c: f64,
    atom_mult: &[f64; PRICE_ATOMS],
    atom_weight: &[f64; PRICE_ATOMS],
) -> Vec<f64> {
    let mut values = vec![0.0f64; (num_steps + 1) * bins];
    let mut next = vec![0.0f64; bins];
    let mut cur = vec![0.0f64; bins];
    // Candidate actions for the bin being solved, split into the action itself and the
    // price-independent part of its value (frictions plus continuation). Each atom then
    // re-maximises for a multiply-add per candidate instead of a fresh value-table
    // interpolation, which is what makes the expectation affordable.
    let mut cand_u = [0.0f64; DP_CANDIDATES];
    let mut cand_base = [0.0f64; DP_CANDIDATES];
    // Scratch for one direction's aligned action magnitudes.
    let mut abuf = [0.0f64; DP_ACTION_HALF];
    for t in (0..num_steps).rev() {
        // Probability the network grants each direction at this step. The four service regimes
        // built from them sum to one, so a flat continuation passes through the expectation
        // unchanged, and a profile that is one everywhere reproduces the unconstrained Bellman
        // recursion exactly.
        let q_c = serve_c.get(t).copied().unwrap_or(1.0).clamp(0.0, 1.0);
        let q_d = serve_d.get(t).copied().unwrap_or(1.0).clamp(0.0, 1.0);
        let da = da_path[t];
        let prem_mean = prem_path[t];
        let cong_q = cprob_path[t];
        // Share of the fleet whose node is flagged at this step -- see `contention` in
        // `build_plan`. It is the size of the priority class a premium window really puts this
        // battery into, as against the size its own event rate implies.
        let contend = contend_path.get(t).copied().unwrap_or(0.0).clamp(0.0, 1.0);

        // --- Price atoms for this step.
        //
        // The congestion premium is not a deterministic uplift. A node collects
        // GAMMA_PRICE * max(Z, 0) only when it is flagged, so the premium is exactly zero on
        // most steps and large on a few: at the 85% loading the instance generator leaves, a
        // node incident to the worst line is flagged about half the time, and then draws a
        // premium averaging $16/MWh with a tail past $40. Folding that into its $4.5/MWh mean
        // -- which is what adding `prem_mean` to every atom does -- hands the sweep a price
        // path that is uniformly slightly better instead of one that is usually ordinary and
        // occasionally excellent. Because the max sits inside the expectation, the two are not
        // interchangeable: the mean version tells the battery to trade a little harder
        // everywhere, while the branched version tells it to hold charge into the windows where
        // its node may spike and then sell the whole cycle at once. On the low-volatility
        // tracks the premium is the dominant source of RT upside -- sigma * lambda_da is only
        // ~$5/MWh there against the premium's $16 -- so smoothing it away removes most of the
        // reason to time a discharge at all.
        //
        // Splitting the five diffusive atoms across the two premium branches keeps the same
        // mean (the branch probability times its size is `prem_mean` by construction) and costs
        // five extra atoms only on steps where the node is actually at risk.
        let mut price_atom = [0.0f64; MAX_PRICE_ATOMS];
        let mut weight_atom = [0.0f64; MAX_PRICE_ATOMS];
        // Which atoms are *idiosyncratic* upside: a price this node gets while the rest of the
        // fleet does not. The congestion premium fires per node on a lagged incident-line test
        // and the jump lands on one (node, step) pair at a time, so both are node-specific; the
        // five diffusive atoms are dominated by the common factor sqrt(rho_spatial) * z_t and
        // the shared diurnal day-ahead curve, so every node moves with them together. The
        // distinction is what decides who wins the allocation -- see below.
        let mut idio_atom = [false; MAX_PRICE_ATOMS];
        let mut idio_mass = 0.0f64;
        let mut total_mass = 0.0f64;
        let mut n_atoms = 0usize;
        let split = cong_q > MIN_CONG_PROB && prem_c > 0.0;
        for i in 0..PRICE_ATOMS - 1 {
            let w = atom_weight[i];
            if w <= 0.0 {
                continue;
            }
            let p = da * atom_mult[i];
            if split {
                price_atom[n_atoms] = p;
                weight_atom[n_atoms] = w * (1.0 - cong_q);
                total_mass += w * (1.0 - cong_q);
                n_atoms += 1;
                price_atom[n_atoms] = p + prem_c;
                weight_atom[n_atoms] = w * cong_q;
                idio_atom[n_atoms] = true;
                idio_mass += w * cong_q;
                total_mass += w * cong_q;
                n_atoms += 1;
            } else {
                price_atom[n_atoms] = p + prem_mean;
                weight_atom[n_atoms] = w;
                total_mass += w;
                n_atoms += 1;
            }
        }
        // The jump atom already sits far above the price at which the battery discharges at
        // full power, and above that threshold the inner max is linear in price, so the
        // premium's mean is exact on that branch and splitting it would buy nothing.
        if atom_weight[PRICE_ATOMS - 1] > 0.0 {
            price_atom[n_atoms] = da * atom_mult[PRICE_ATOMS - 1] + prem_mean;
            weight_atom[n_atoms] = atom_weight[PRICE_ATOMS - 1];
            idio_atom[n_atoms] = true;
            idio_mass += atom_weight[PRICE_ATOMS - 1];
            total_mass += atom_weight[PRICE_ATOMS - 1];
            n_atoms += 1;
        }

        // --- Discharge service, resolved per price atom instead of per step.
        //
        // The measured ratio `q_d` says what fraction of the discharge power this battery asks
        // for actually clears the lines, and the sweep spends it as a per-step probability of
        // being granted -- the same probability on every atom. That makes the grant independent
        // of the price, and the allocator it is a model of is the opposite of independent: it
        // walks the fleet in descending marginal value and serves bidders in order until the
        // headroom runs out. Whether this battery is served is therefore a question about its
        // *rank*, and rank is set by how its price compares with the rest of the fleet's at the
        // same instant, not by its price in absolute terms.
        //
        // The five diffusive atoms carry almost no rank information. An RT shock is
        // sqrt(rho_spatial) * z_t + sqrt(1 - rho_spatial) * eps_i with rho_spatial = 0.70, on top
        // of a day-ahead curve whose diurnal base term is common to every node, so when this
        // battery's price is at the high diffusive atom so is most of the fleet's and they are all
        // bidding to discharge at once. The other two atoms are node-specific and large. A
        // congestion premium is gamma_price * max(Z, 0), fired by a lagged test on the node's own
        // incident lines, averaging $16/MWh when it fires against an idiosyncratic diffusive
        // spread sigma * lambda_da * sqrt(1 - rho_spatial) of about $7/MWh on the tightest track.
        // A jump multiplies one node's day-ahead price by a Pareto draw. Either one puts this
        // battery at the front of the queue.
        //
        // So the grant probability is split across a priority class rather than averaged over it.
        // If the allocator clears q_d * m bidders per step and pi * m of them have an
        // idiosyncratic event, then conditional on having one this battery competes only inside
        // that class: it is served with probability min(1, q_d / pi), and the headroom left over
        // after the class is served is what the ordinary steps get,
        // (q_d - pi * q_hi) / (1 - pi). That is the same queue model the measurement came out of,
        // read at the resolution of the price atom.
        //
        // The split is mean-preserving by construction -- pi * q_hi + (1 - pi) * q_lo = q_d, so
        // expected throughput over the step is exactly what was measured -- and it is the
        // *relative* attractiveness of the two that moves. A flat q_d scales the whole
        // opportunity down uniformly and so leaves the undeliverable fleet-wide diurnal cycle
        // looking just as good as the deliverable node-specific spike; resolving it per atom says
        // the cycle is the part the lines take away and the spike is the part they do not, which
        // is the difference between spending a cycle into the peak with the rest of the fleet and
        // holding it for the window where this node is alone in wanting to sell.
        //
        // Both ends of the gate are identities. `q_d = 1` -- an unbinding network -- gives
        // q_hi = q_lo = 1 and the weights below reduce to the unconstrained recursion bit for
        // bit. `pi = 0` -- no jump branch and a node the premium never threatens -- gives
        // q_hi = q_lo = q_d, which is the flat model. Only the charge direction is left flat:
        // every idiosyncratic price event the challenge generates is upward (the premium is
        // gamma_price * max(Z, 0) >= 0 and a Pareto jump is positive), so there is no atom on
        // which a battery outbids the fleet for the right to *buy*.
        //
        // Two quantities that used to be one. `pi` is the probability *this* node has an
        // idiosyncratic event, which is what decides how often it is in the class; `class` is the
        // share of the fleet in the class when it is, which is what decides how fast the class
        // clears. They coincide only if the events are independent across nodes, and the premium's
        // `zeta` is drawn once per step for the whole fleet, so they do not: when this node's
        // premium fires, every flagged node's fires with it. `class` is therefore at least the
        // flagged share of the fleet measured in `contention`.
        //
        // Mean preservation is on `pi`, not on `class`: the throughput identity is
        // `pi * q_hi + (1 - pi) * q_lo = q_d`, so `q_lo` absorbs whatever `class` does to `q_hi`
        // and expected throughput over the step is still exactly the measured `q_d`. The `min`
        // keeps `q_hi <= q_d / pi`, which is what keeps `q_lo >= 0` without a floor, and
        // `class <= 1` keeps `q_hi >= q_d >= q_lo`.
        //
        // Where the fleet is no more contended than this node's own event rate, `class` is `pi` and
        // the two branches below reduce to the previous expressions term by term; `q_d = 1` gives
        // `q_hi = q_lo = 1` and the unconstrained recursion bit for bit.
        let pi = if total_mass > 0.0 {
            (idio_mass / total_mass).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let (qd_hi, qd_lo) = if pi <= 0.0 || pi >= 1.0 || q_d >= 1.0 {
            (q_d, q_d)
        } else {
            let class = pi.max(contend).min(1.0);
            let hi = (q_d / class).min(1.0);
            let lo = ((q_d - pi * hi) / (1.0 - pi)).clamp(0.0, 1.0);
            (hi, lo)
        };
        let mut qd_atom = [q_d; MAX_PRICE_ATOMS];
        for a in 0..n_atoms {
            qd_atom[a] = if idio_atom[a] { qd_hi } else { qd_lo };
        }
        // Sort the atoms into increasing price. The order of a sum does not matter, but the
        // envelope walk below needs it: the diffusive atoms are generated in increasing price and
        // each is immediately followed by its own premium copy, and a premium of
        // GAMMA_PRICE * E[Z|Z>0] = $16/MWh is larger than the diffusive spacing
        // sigma * (KINK_B - KINK_A) * lambda_da on the low-volatility tracks, so the premium copy
        // of one diffusive atom frequently overtakes the next bare one. Eleven entries at most,
        // already nearly sorted, so insertion sort is a handful of comparisons.
        for i in 1..n_atoms {
            let p = price_atom[i];
            let w = weight_atom[i];
            let qd = qd_atom[i];
            let mut j = i;
            while j > 0 && price_atom[j - 1] > p {
                price_atom[j] = price_atom[j - 1];
                weight_atom[j] = weight_atom[j - 1];
                qd_atom[j] = qd_atom[j - 1];
                j -= 1;
            }
            price_atom[j] = p;
            weight_atom[j] = w;
            qd_atom[j] = qd;
        }

        // The four service regimes, folded into the atom's own weight. Doing it here rather than
        // inside the bin loop keeps the innermost arithmetic at exactly the flop count it had
        // before: the regime probabilities now vary across atoms but not across SOC bins, and
        // there are at most eleven atoms against up to 49 bins.
        let mut w_both_atom = [0.0f64; MAX_PRICE_ATOMS];
        let mut w_dis_atom = [0.0f64; MAX_PRICE_ATOMS];
        let mut w_chg_atom = [0.0f64; MAX_PRICE_ATOMS];
        let mut w_none_atom = [0.0f64; MAX_PRICE_ATOMS];
        for a in 0..n_atoms {
            let w = weight_atom[a];
            let qd = qd_atom[a];
            w_both_atom[a] = w * q_c * qd;
            w_dis_atom[a] = w * qd * (1.0 - q_c);
            w_chg_atom[a] = w * q_c * (1.0 - qd);
            w_none_atom[a] = w * (1.0 - q_c) * (1.0 - qd);
        }

        for k in 0..bins {
            let e = lo + k as f64 * dk;
            let (lower, upper) = soc_action_bounds(eta_c, eta_d, p_c_dp, p_d_dp, lo, top, e);

            cand_u[0] = 0.0;
            cand_base[0] = interp(&next, e, lo, dk, bins);
            let mut nc = 1usize;
            // One past the last charge candidate. Side 0 is the charge side and
            // `aligned_actions` writes it in increasing magnitude, i.e. decreasing `u`; side 1 is
            // the discharge side and comes out increasing in `u`. The walk below needs each
            // block ordered by `u`, which is the discharge block as written and the charge block
            // read backwards, so the boundary has to be recorded here.
            let mut chg_end = 1usize;
            for side in 0..2usize {
                let edge = if side == 0 { lower } else { upper };
                if edge != 0.0 {
                    let na = aligned_actions(&mut abuf, edge, e, lo, dk, bins, eta_c, eta_d);
                    for i in 0..na {
                        let u = abuf[i];
                        let soc_next = soc_after(eta_c, eta_d, u, e, lo, top);
                        let abs_u = u.abs();
                        let deg_base = (abs_u * DT) / cap;
                        cand_u[nc] = u;
                        cand_base[nc] = -KAPPA_TX * abs_u * DT - KAPPA_DEG * deg_base * deg_base
                            + interp(&next, soc_next, lo, dk, bins);
                        nc += 1;
                    }
                }
                if side == 0 {
                    chg_end = nc;
                }
            }

            // W[t][e] = sum_atoms weight * sum_regimes P(regime) * max over that regime's
            // actions. Candidate 0 is the idle action, always available, and its value is
            // exactly the continuation value of not being served.
            //
            // The per-direction maxima are found by walking the upper envelope once rather than
            // rescanning every candidate at every atom. A candidate's value is
            // `u * price * DT + base`, a straight line in the price with slope `u * DT`, so each
            // direction's maximum over its candidates is the upper envelope of a set of lines
            // whose slopes are already sorted by construction. Two facts make one forward walk
            // exact.
            //
            // The maximising rank is non-decreasing in the price. For ranks i < j, if j is at
            // least as good as i at price p then at any p' > p its lead has grown by
            // `(p' - p) * DT * (u_j - u_i) > 0`, so the optimum never moves back toward the
            // smaller action as the price rises. That is an exchange argument and needs nothing
            // but the sorted slopes, which is why the atoms above are sorted into increasing
            // price.
            //
            // And at a fixed price the value is unimodal in the rank, so climbing forward until
            // the next candidate is worse lands on the maximum rather than on a local one. The
            // candidate value as a function of `u` is `u * price * DT`, linear, plus
            // `-KAPPA_TX * |u| * DT`, linear on each side of zero, plus
            // `-KAPPA_DEG * (|u| DT / cap)^2 = -KAPPA_DEG (DT/cap)^2 u^2`, concave, plus
            // `interp(W[t+1], soc_after(u))`, a concave function of a quantity affine in `u`.
            // So it is concave on each branch, and a concave function sampled at increasing `u`
            // gives a unimodal sequence. W[t+1] is concave in stored energy because the DP
            // preserves concavity: the reward is jointly concave in (action, SOC), the feasible
            // set `soc_action_bounds` describes is convex, each regime's inner maximum is
            // therefore concave in SOC, and the regime probabilities are a positive combination.
            // `interp` of a concave function sampled on the grid is concave piecewise linear.
            // The SOC clamp inside `soc_after` never fires here, because the bounds are derived
            // from the same grid floor and ceiling.
            //
            // So the two ranks advance a total of at most `nc` steps across all the atoms
            // instead of `nc` steps per atom, and the sweep -- the dominant plan-time cost at
            // batteries x steps x bins x atoms x candidates -- loses its last factor. Same
            // values, in a different summation order.
            let n_chg = chg_end - 1;
            let n_dis = nc - chg_end;
            let v_idle = cand_base[0];
            let mut r_chg = 0usize;
            let mut r_dis = 0usize;
            let mut expected = 0.0f64;
            for a in 0..n_atoms {
                let s = price_atom[a] * DT;
                let mut v_dis = f64::NEG_INFINITY;
                if n_dis > 0 {
                    let j0 = chg_end + r_dis;
                    let mut best = cand_u[j0] * s + cand_base[j0];
                    while r_dis + 1 < n_dis {
                        let j = chg_end + r_dis + 1;
                        let v = cand_u[j] * s + cand_base[j];
                        if v >= best {
                            best = v;
                            r_dis += 1;
                        } else {
                            break;
                        }
                    }
                    v_dis = best;
                }
                let mut v_chg = f64::NEG_INFINITY;
                if n_chg > 0 {
                    // Rank `r` of the charge block, ascending in `u`, is index
                    // `chg_end - 1 - r`. The block is stored in increasing magnitude from index
                    // 1 up to the bound at `chg_end - 1`, and charge actions are negative, so
                    // the bound is the smallest `u` and therefore rank 0.
                    let j0 = chg_end - 1 - r_chg;
                    let mut best = cand_u[j0] * s + cand_base[j0];
                    while r_chg + 1 < n_chg {
                        let j = chg_end - 2 - r_chg;
                        let v = cand_u[j] * s + cand_base[j];
                        if v >= best {
                            best = v;
                            r_chg += 1;
                        } else {
                            break;
                        }
                    }
                    v_chg = best;
                }
                // A direction with no candidate at all -- a full or empty battery -- stays at
                // NEG_INFINITY and so collapses onto idle, which is the right reading.
                let best_dis = if v_dis > v_idle { v_dis } else { v_idle };
                let best_chg = if v_chg > v_idle { v_chg } else { v_idle };
                let best_both = if best_dis > best_chg { best_dis } else { best_chg };
                expected += w_both_atom[a] * best_both
                    + w_dis_atom[a] * best_dis
                    + w_chg_atom[a] * best_chg
                    + w_none_atom[a] * v_idle;
            }
            cur[k] = expected;
        }
        values[t * bins..(t + 1) * bins].copy_from_slice(&cur);
        let swap_tmp = cur;
        cur = next;
        next = swap_tmp;
    }
    values
}

/// Fraction of the power each battery asks for that the network actually delivers, measured by
/// replaying the online allocator at plan time.
///
/// The replay is exact in everything the challenge publishes: the exogenous injections, the PTDF
/// columns and the line limits are all known for the whole horizon, so the headroom each step
/// leaves is the real one. Only the price is a forecast -- `forecast`, the day-ahead path with the
/// congestion-premium and Pareto-jump branches realised as events of their conditional size on a
/// stratified subset of steps rather than smoothed into their means, which carries the same mass
/// per node as the mean path did and the cross-sectional spread it had none of. Batteries act on
/// their true nameplate
/// bounds here, so the ratio answers exactly the question the DP needs: of the power this battery
/// would like to use, how much clears the lines?
///
/// `batteries` are the calibration tables, swept on the coarse `CAL_BINS` grid, and this function
/// is their only consumer: the whole table is collapsed into one scalar per battery. The grid only
/// has to place each battery's cycle in roughly the right window, because the ratio is a sum over
/// all 96 or 192 steps of asked-for and delivered MW. The action set is exact at any grid size --
/// `aligned_actions` enumerates the breakpoints of a piecewise-linear objective, so the inner max
/// is exact on 13 nodes as on 49 -- so what a coarse grid costs here is resolution in the value
/// function's shape, which the sum averages over, rather than accuracy in each step's action.
///
/// The replay runs the *whole* online allocator, not a part of it: all four candidates -- value
/// ordered greedy, uniform shrink, shrink-then-top-up, and the counter-flow pairing -- are built
/// and the one scoring best under `action_value` is the one measured, exactly as online. An earlier
/// version replayed only shrink-then-top-up. That is the weakest of the four on precisely the
/// tracks where the measurement matters: when the fleet's desired signs are correlated the uniform
/// factor collapses and the real policy switches to pairing, which clears a counter-flowing pair at
/// the difference of two PTDF columns instead of the sum. Measuring the weaker allocator therefore
/// understated `r` wherever pairing wins, and `r` feeds straight into the service probability the
/// final sweep plans with -- so the table told batteries on the tight tracks that the network would
/// almost never let them act, and they held a middling SOC instead of positioning for the windows
/// the pairing actually opens. It understated it in exactly one direction and only on the
/// constrained tracks. Where no line is overloaded the replay plays each battery's unconstrained
/// optimum, `r` is one, and the measurement is unchanged.
///
/// A negative entry means the battery never asked to move and the ratio is undefined. A battery the
/// pairing moves against its own solo optimum delivers more than it asked for; the ratio is clamped
/// at one, which is the right reading for a quantity that means "does the network let me act".
///
/// The fourth return value is the profit the replayed policy earns under the forecast, summed with
/// the challenge's own reward function. It was the fair way to compare two value functions on one
/// allocator, one price path and one grid; the only caller left is the calibration pass, which
/// needs the ratios and discards it.
fn measure_delivery(
    challenge: &Challenge,
    batteries: &[BatteryPlan],
    nameplate: &[(f64, f64)],
    exo_flows: &[Vec<f64>],
    forecast: &[f64],
    num_nodes: usize,
    num_lines: usize,
    num_steps: usize,
    bins: usize,
    flow_safety: f64,
) -> (
    Vec<(f64, f64)>,
    Vec<Vec<(f64, f64)>>,
    Vec<(f64, f64)>,
    f64,
) {
    let m = batteries.len();
    let nblocks = ((num_steps + SERVICE_BLOCK - 1) / SERVICE_BLOCK).max(1);
    let limits = &challenge.network.flow_limits;
    let caps: Vec<f64> = (0..num_lines).map(|l| limits[l] * flow_safety).collect();

    let mut socs: Vec<f64> = challenge
        .batteries
        .iter()
        .map(|b| b.soc_initial_mwh)
        .collect();
    // Profit the replayed policy earns under the forecast, accumulated with the same reward
    // function the challenge scores with. Two value functions replayed through this same
    // allocator on this same price path are directly comparable on it, which is what decides
    // which of the two the policy plays.
    let mut total_profit = 0.0f64;
    // Asked-for and cleared MW, accumulated separately over the steps where the battery wanted
    // to charge and the steps where it wanted to discharge. Pooling the two, as this used to,
    // averages two ratios measured on disjoint and systematically different sets of steps.
    let mut want_c = vec![0.0f64; m];
    let mut want_d = vec![0.0f64; m];
    let mut deliv_c = vec![0.0f64; m];
    let mut deliv_d = vec![0.0f64; m];
    // The same four sums, split by time block, row-major by battery. Collapsing the horizon into
    // one ratio per direction says the network grants a battery the same share of what it asks
    // for at the trough, at the shoulder and at the peak. It does not: contention is set by how
    // much of the fleet wants the same sign at that moment, and the fleet's sign follows the
    // shared diurnal day-ahead curve, so a battery is denied precisely in the windows it is
    // aiming at and left alone in the ones between.
    // Second moment of the per-step delivered *fraction*, on the same ask-weighting as the sums
    // above. With `phi_t = cleared_t / asked_t` in [0, 1] and weights `asked_t`, the sums above
    // produce `E[phi]` and these produce `E[phi^2] = sum_t cleared_t^2 / asked_t / sum_t asked_t`.
    // The pair pins the two-point distribution the sweep models the grant with -- see
    // `service_split` -- and it does so in both directions at once, which a threshold frequency
    // cannot: a battery shaved to the same small fraction on every step and a battery granted in
    // full on that fraction of the steps have the same mean and different second moments.
    let mut sq_c = vec![0.0f64; m];
    let mut sq_d = vec![0.0f64; m];
    let mut want_c_blk = vec![0.0f64; m * nblocks];
    let mut want_d_blk = vec![0.0f64; m * nblocks];
    let mut deliv_c_blk = vec![0.0f64; m * nblocks];
    let mut deliv_d_blk = vec![0.0f64; m * nblocks];
    let mut desired = vec![0.0f64; m];
    let mut gain = vec![0.0f64; m];
    // Per-direction per-MWh marginal values, the inputs the counter-flow pairing ranks on. They
    // come out of the same candidate scan that produces `desired`, so building them here costs
    // nothing beyond the comparisons.
    let mut m_dis = vec![f64::NEG_INFINITY; m];
    let mut m_chg = vec![f64::NEG_INFINITY; m];
    // The same per-direction gain curves the online step builds, so the pairing sizes its pairs
    // here exactly as it does online. `CALIB_ACTION_HALF` equals `ONLINE_ACTION_HALF`, which is the
    // width `GainCurve` carries.
    let mut dis_curve = vec![GainCurve::empty(); m];
    let mut chg_curve = vec![GainCurve::empty(); m];
    let mut price_at = vec![0.0f64; m];
    // Scratch for one direction's aligned action magnitudes.
    let mut cbuf = [0.0f64; CALIB_ACTION_HALF];

    for t in 0..num_steps {
        let base_flows = &exo_flows[t];
        let nl = num_lines.min(base_flows.len());
        let next_row_start = (t + 1).min(num_steps) * bins;

        // Each battery's preferred action under the forecast price and its own value table.
        for b in 0..m {
            let bp = &batteries[b];
            let row = &bp.values[next_row_start..next_row_start + bins];
            let top = bp.lo + bp.dk * (bins - 1) as f64;
            let soc = socs[b];
            let (p_c, p_d) = nameplate[b];
            let (lower, upper) =
                soc_action_bounds(bp.eta_c, bp.eta_d, p_c, p_d, bp.lo, top, soc);
            // The replay's price path: the day-ahead price with the congestion-premium and Pareto
            // jump branches realised as events of their conditional size rather than folded into
            // their means. Same mass per node over the horizon, placed where the probability is --
            // see `forecast` in `build_plan` for why a path with no spike cannot be used to choose
            // between two value functions that differ over how much they hold back for one.
            let price = forecast[t * num_nodes + bp.node];
            price_at[b] = price;
            let idle = interp(row, soc, bp.lo, bp.dk, bins);
            let mut best_u = 0.0f64;
            let mut best_v = idle;
            let mut bu_d = 0.0f64;
            let mut bv_d = f64::NEG_INFINITY;
            let mut bu_c = 0.0f64;
            let mut bv_c = f64::NEG_INFINITY;
            dis_curve[b] = GainCurve::empty();
            chg_curve[b] = GainCurve::empty();
            for side in 0..2usize {
                let edge = if side == 0 { lower } else { upper };
                if edge == 0.0 {
                    continue;
                }
                let na = aligned_actions(
                    &mut cbuf, edge, soc, bp.lo, bp.dk, bins, bp.eta_c, bp.eta_d,
                );
                for i in 0..na {
                    let u = cbuf[i];
                    let soc_next = soc_after(bp.eta_c, bp.eta_d, u, soc, bp.lo, top);
                    let v = step_reward(u, price, bp.cap)
                        + interp(row, soc_next, bp.lo, bp.dk, bins);
                    if v > best_v {
                        best_v = v;
                        best_u = u;
                    }
                    if u > 0.0 {
                        dis_curve[b].push(u, v - idle);
                        if v > bv_d {
                            bv_d = v;
                            bu_d = u;
                        }
                    } else if u < 0.0 {
                        chg_curve[b].push(-u, v - idle);
                        if v > bv_c {
                            bv_c = v;
                            bu_c = u;
                        }
                    }
                }
            }
            desired[b] = best_u;
            gain[b] = best_v - idle;
            m_dis[b] = if bu_d > 0.0 {
                (bv_d - idle) / (bu_d * DT)
            } else {
                f64::NEG_INFINITY
            };
            m_chg[b] = if bu_c < 0.0 {
                (bv_c - idle) / (-bu_c * DT)
            } else {
                f64::NEG_INFINITY
            };
        }

        // The same four-way allocation the online step runs, picked by the same lookahead.
        let mut des_flows = base_flows[..nl].to_vec();
        for b in 0..m {
            let d = desired[b];
            if d == 0.0 {
                continue;
            }
            let sens = &batteries[b].sens;
            for l in 0..nl {
                des_flows[l] += d * sens[l];
            }
        }
        let mut any_overloaded = false;
        for l in 0..nl {
            if des_flows[l].abs() > caps[l] {
                any_overloaded = true;
                break;
            }
        }

        let action = if !any_overloaded {
            desired.clone()
        } else {
            // Same value-density ordering the online step uses; the ratio this loop measures is
            // the service probability the final sweep plans with, so it has to allocate exactly
            // as the policy does.
            let density =
                value_density(batteries, &desired, &gain, base_flows, &caps, nl, m);
            let greedy =
                greedy_allocate(batteries, &desired, &density, base_flows, &caps, nl, m);

            let mut lambda = 1.0f64;
            for l in 0..nl {
                let d = des_flows[l] - base_flows[l];
                if d.abs() < 1e-12 {
                    continue;
                }
                let bound = if d > 0.0 {
                    (caps[l] - base_flows[l]) / d
                } else {
                    (-caps[l] - base_flows[l]) / d
                };
                if bound < lambda {
                    lambda = bound;
                }
            }
            let lambda = lambda.clamp(0.0, 1.0);
            let scaled: Vec<f64> = desired.iter().map(|&u| u * lambda).collect();
            let scaled_flows: Vec<f64> = (0..nl)
                .map(|l| base_flows[l] + lambda * (des_flows[l] - base_flows[l]))
                .collect();
            let topped = topup_allocate(
                batteries,
                &desired,
                &scaled,
                &density,
                &scaled_flows,
                &caps,
                nl,
                m,
            );

            let (pair_action, pair_flows) = paired_allocate(
                batteries,
                &m_dis,
                &m_chg,
                &dis_curve,
                &chg_curve,
                &price_at,
                base_flows,
                &caps,
                nl,
                m,
            );
            // Same pin rule as the online step: hold only the end the pair drove against its own
            // optimum, and let the end the pair left short of its asked-for action climb in the
            // top-up. The replay measures the service probability the final sweep plans with, so
            // it has to allocate exactly as the policy does.
            let mut pair_target = desired.clone();
            for b in 0..m {
                let pa = pair_action[b];
                if pa != 0.0 && !(pa * desired[b] > 0.0 && pa.abs() < desired[b].abs()) {
                    pair_target[b] = pa;
                }
            }
            let paired = topup_allocate(
                batteries,
                &pair_target,
                &pair_action,
                &density,
                &pair_flows,
                &caps,
                nl,
                m,
            );

            let v_greedy = action_value(batteries, &socs, &price_at, &greedy, next_row_start, bins);
            let v_scaled = action_value(batteries, &socs, &price_at, &scaled, next_row_start, bins);
            let v_topped = action_value(batteries, &socs, &price_at, &topped, next_row_start, bins);
            let v_paired = action_value(batteries, &socs, &price_at, &paired, next_row_start, bins);
            if v_paired >= v_topped && v_paired >= v_scaled && v_paired >= v_greedy {
                paired
            } else if v_topped >= v_scaled && v_topped >= v_greedy {
                topped
            } else if v_scaled >= v_greedy {
                scaled
            } else {
                greedy
            }
        };

        for b in 0..m {
            let d = desired[b];
            let a = action[b];
            // Credit only the part of the cleared action that went the way the battery asked.
            // A battery the pairing drives against its own solo optimum asked for nothing in
            // that direction, so it contributes to neither side's denominator.
            let blk = b * nblocks + (t / SERVICE_BLOCK).min(nblocks - 1);
            if d > 0.0 {
                let got = a.max(0.0);
                want_d[b] += d;
                deliv_d[b] += got;
                want_d_blk[blk] += d;
                deliv_d_blk[blk] += got;
                // asked * (cleared / asked)^2, i.e. the ask-weighted second moment's numerator.
                sq_d[b] += got * got / d;
            } else if d < 0.0 {
                let asked = -d;
                let got = (-a).max(0.0);
                want_c[b] += asked;
                deliv_c[b] += got;
                want_c_blk[blk] += asked;
                deliv_c_blk[blk] += got;
                sq_c[b] += got * got / asked;
            }
            let bp = &batteries[b];
            total_profit += step_reward(action[b], price_at[b], bp.cap);
            let top = bp.lo + bp.dk * (bins - 1) as f64;
            socs[b] = soc_after(bp.eta_c, bp.eta_d, action[b], socs[b], bp.lo, top);
        }
    }

    let overall: Vec<(f64, f64)> = (0..m)
        .map(|b| {
            let r_c = if want_c[b] > 1e-6 {
                (deliv_c[b] / want_c[b]).clamp(0.0, 1.0)
            } else {
                -1.0
            };
            let r_d = if want_d[b] > 1e-6 {
                (deliv_d[b] / want_d[b]).clamp(0.0, 1.0)
            } else {
                -1.0
            };
            (r_c, r_d)
        })
        .collect();

    // Ask-weighted second moment of the delivered fraction, on the same steps and with the same
    // "never asked" convention as the ratios above, so the caller can pair them entry by entry.
    // `phi <= 1` gives `E[phi^2] <= E[phi]`, which is what keeps the depth `service_split` reads
    // out of the pair at or below one.
    let moment2: Vec<(f64, f64)> = (0..m)
        .map(|b| {
            let s_c = if want_c[b] > 1e-6 {
                (sq_c[b] / want_c[b]).clamp(0.0, 1.0)
            } else {
                -1.0
            };
            let s_d = if want_d[b] > 1e-6 {
                (sq_d[b] / want_d[b]).clamp(0.0, 1.0)
            } else {
                -1.0
            };
            (s_c, s_d)
        })
        .collect();

    // Per-block ratios, each pooled with its neighbours before the division so the smoothing is
    // weighted by the MW actually at stake rather than averaging ratios measured on very
    // different volumes. A block in which the battery never asked to move in a direction comes
    // back negative and the caller falls back to the horizon figure there.
    let blocks: Vec<Vec<(f64, f64)>> = (0..m)
        .map(|b| {
            (0..nblocks)
                .map(|j| {
                    let first = j.saturating_sub(SERVICE_SMOOTH);
                    let last = (j + SERVICE_SMOOTH).min(nblocks - 1);
                    let mut wc = 0.0f64;
                    let mut dc = 0.0f64;
                    let mut wd = 0.0f64;
                    let mut dd = 0.0f64;
                    for jj in first..=last {
                        let idx = b * nblocks + jj;
                        wc += want_c_blk[idx];
                        dc += deliv_c_blk[idx];
                        wd += want_d_blk[idx];
                        dd += deliv_d_blk[idx];
                    }
                    let r_c = if wc > 1e-6 {
                        (dc / wc).clamp(0.0, 1.0)
                    } else {
                        -1.0
                    };
                    let r_d = if wd > 1e-6 {
                        (dd / wd).clamp(0.0, 1.0)
                    } else {
                        -1.0
                    };
                    (r_c, r_d)
                })
                .collect()
        })
        .collect();

    (overall, blocks, moment2, total_profit)
}

/// Split one measured delivery ratio into the two numbers the sweep's service model needs: the
/// probability `q` that the direction is granted on a step and the fraction `depth` of nameplate
/// the battery may move when it is. Expected throughput per step is `q * depth`, and that product
/// is the measured ratio exactly.
///
/// The replay measures `r = E[phi]`, the ask-weighted mean of the per-step delivered fraction
/// `phi_t = cleared_t / asked_t`. One mean does not pin the distribution, and the two readings it
/// is consistent with are the two opposite models of what the allocator does. Read as a *derate*
/// -- `q = 1, depth = r` -- it says the battery is shaved to the same small fraction on every
/// step, so the sweep plans a slow steady cycle. Read as an *intermittent grant* --
/// `q = r, depth = 1` -- it says the battery is denied outright on most steps and served in full
/// on a few, so the sweep holds SOC positioned for a full-power slam. Both have mean `r` and they
/// are different value functions.
///
/// The previous split chose between them on the frequency with which the battery cleared at least
/// half of what it asked for, `q = max(f, r)`, and that statistic has the mapping backwards in the
/// regime the tight tracks actually live in. The allocation the lookahead picks when many
/// batteries compete for little headroom is usually the uniform shrink `scaled` or the top-up
/// built on it, so a battery is shaved by one fleet-wide factor on *every* step. Shaved to 0.1,
/// it clears under half its ask every time, so `f = 0`, `max(f, r) = r`, and the split comes out
/// `q = 0.1, depth = 1` -- the intermittent reading, when the measurement it came from is the
/// definition of a uniform derate. The sweep then plans a full-power cycle the allocator never
/// grants, which is precisely the failure the service model was introduced to remove, and it
/// happens on the two tracks where the shrink factor is smallest.
///
/// The second moment separates them, and it is the only additional statistic needed. Under the
/// two-point model `phi in {0, depth}` with `P(phi = depth) = q`,
///     E[phi] = q * depth,    E[phi^2] = q * depth^2,
/// so `depth = E[phi^2] / E[phi]` and `q = E[phi] / depth`. Uniform shaving at 0.1 gives
/// `E[phi] = 0.1, E[phi^2] = 0.01`, hence `depth = 0.1, q = 1`: the derate. All-or-nothing at rate
/// 0.1 gives `E[phi] = E[phi^2] = 0.1`, hence `depth = 1, q = 0.1`: the intermittent grant. An
/// unbinding network gives `phi = 1` every step, hence `depth = q = 1` and a table bit-identical
/// to one swept with no service model at all. The estimator is exact on all three.
///
/// `MIN_SERVICE` is still floored, and it is floored on `q` alone with the depth re-derived from
/// the measurement afterwards, so the product stays `r` whatever the floor does -- the floor buys
/// shape (fewer, shallower grants) and never level. The old split also floored the *power* at
/// `MIN_POWER_FRACTION` of nameplate after the division, which broke the product and planned up to
/// ten times the throughput the replay recorded for the most heavily rationed batteries; the
/// moment-matched depth cannot collapse without `r` collapsing with it, so the floor is no longer
/// needed here.
fn service_split(r: f64, m2: f64) -> (f64, f64) {
    // depth = E[phi^2] / E[phi]. A battery with no measured throughput has no shape to read, so
    // it falls through to full depth and lets the frequency carry the whole ratio.
    let depth_raw = if r > 1e-12 && m2 > 0.0 {
        (m2 / r).clamp(1e-6, 1.0)
    } else {
        1.0
    };
    let q = (r / depth_raw).clamp(MIN_SERVICE.min(1.0), 1.0);
    let depth = (r / q).clamp(0.0, 1.0);
    (q, depth)
}

/// Rescale a measured service profile so its horizon mean is `target`, then clamp it into the
/// range the sweep can use.
///
/// The horizon figure is the quantity the replay measures most reliably -- it is a ratio of two
/// sums over every step -- so the profile is used only for its shape and the level is pinned to
/// that figure. The change is therefore mean-preserving in expected throughput: the same total
/// amount of service is planned for, redistributed to the windows the allocator is measured to
/// leave open. A profile that is one everywhere, which is what an unbinding network produces,
/// rescales to itself and the sweep is bit-identical to planning with no service model at all.
fn rescale_service(path: &mut [f64], target: f64) {
    let n = path.len();
    if n == 0 {
        return;
    }
    let mean = path.iter().sum::<f64>() / n as f64;
    let level = target.clamp(MIN_SERVICE, 1.0);
    if !(mean > 1e-9) {
        for v in path.iter_mut() {
            *v = level;
        }
        return;
    }
    // The level is pinned on the *clamped* profile, not on the profile before clamping.
    //
    // Scaling by `target / mean` and clamping afterwards -- which is what this used to do -- lands
    // the mean on the target only while no entry hits a bound. The blocks in which a battery never
    // cleared anything measure zero, so on a rationed fleet several of them sit on the floor after
    // the scaling and the mean of what the sweep actually receives comes out above `target`: the
    // floor silently adds the throughput back that `target` was chosen to remove. Half the blocks
    // at zero and half at twice the target is a 1.5x overstatement, and that multiplies whatever
    // overstatement the level itself carries.
    //
    // `clamped_mean` is nondecreasing in the factor, tends to `MIN_SERVICE` as the factor goes to
    // zero and to one as it grows, and `level >= MIN_SERVICE`, so the target mean is bracketed and
    // a bisection finds the factor that hits it. Where nothing is clamped the bisection returns
    // `target / mean` and the profile is the old one exactly; where the whole profile is already
    // flat at one -- an unbinding network -- `clamped_mean` is one for every factor and this is
    // still the identity.
    let mut hi = (level / mean).max(1.0);
    for _ in 0..40 {
        if clamped_mean(path, hi) >= level {
            break;
        }
        hi *= 2.0;
    }
    let mut lo = 0.0f64;
    for _ in 0..40 {
        let mid = 0.5 * (lo + hi);
        if clamped_mean(path, mid) < level {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let factor = 0.5 * (lo + hi);
    for v in path.iter_mut() {
        *v = (*v * factor).clamp(MIN_SERVICE, 1.0);
    }
}

/// Mean of a service profile after it is scaled by `factor` and clamped into the band the sweep
/// can use. Monotone nondecreasing in `factor`, which is what lets `rescale_service` bisect on it.
fn clamped_mean(path: &[f64], factor: f64) -> f64 {
    let n = path.len();
    if n == 0 {
        return 0.0;
    }
    path.iter()
        .map(|&v| (v * factor).clamp(MIN_SERVICE, 1.0))
        .sum::<f64>()
        / n as f64
}

/// Per-step profit for one battery: revenue minus transaction and degradation cost.
fn step_reward(u: f64, price: f64, cap: f64) -> f64 {
    if u == 0.0 {
        return 0.0;
    }
    let abs_u = u.abs();
    let deg_base = (abs_u * DT) / cap;
    u * price * DT - KAPPA_TX * abs_u * DT - KAPPA_DEG * deg_base * deg_base
}

/// Mirror of `Battery::apply_action_to_soc`.
fn soc_after(eta_c: f64, eta_d: f64, u: f64, soc: f64, lo: f64, hi: f64) -> f64 {
    let charge = (-u).max(0.0);
    let discharge = u.max(0.0);
    let next = soc + eta_c * charge * DT - discharge * DT / eta_d;
    next.clamp(lo, hi)
}

/// Mirror of `Battery::compute_action_bounds`, used for planning on the SOC grid.
fn soc_action_bounds(
    eta_c: f64,
    eta_d: f64,
    p_c: f64,
    p_d: f64,
    lo: f64,
    hi: f64,
    soc: f64,
) -> (f64, f64) {
    let headroom = (hi - soc).max(0.0);
    let available = (soc - lo).max(0.0);
    let max_charge = (headroom / (eta_c * DT)).min(p_c).max(0.0);
    let max_discharge = (available * eta_d / DT).min(p_d).max(0.0);
    (-max_charge, max_discharge)
}

/// The action magnitudes worth evaluating in one direction, aligned to the SOC grid.
///
/// Writes them into `out` in increasing `|u|` and returns how many were written; `edge` is the
/// signed bound for the direction (positive = discharge, negative = charge) and is always the
/// last entry. The entries before it are the actions that land `soc_after` exactly on an interior
/// node of the SOC grid.
///
/// This set contains the exact maximiser, which a uniform grid does not. A candidate's value is
/// `u * price * DT` plus the frictions plus `interp(W, soc_after(u))`. The revenue and the
/// transaction cost are linear in `u`; the degradation cost is quadratic but worth about 1e-4 $
/// at full power against a revenue slope near 12 $/MW, so it bends the objective by nothing. The
/// continuation is `interp`, piecewise linear in the next SOC with breakpoints at the grid nodes,
/// and the next SOC is affine in `u`. The candidate value is therefore piecewise linear in `u`
/// with its breakpoints exactly at these actions, and a piecewise-linear function on an interval
/// attains its maximum at a breakpoint or an endpoint -- so maximising over this set is
/// maximising over the whole interval.
///
/// A uniform grid of the same size generally contains no breakpoint at all. It then both
/// understates the inner max, which propagates back through every step of the sweep, and returns
/// an action that is not the optimum, which is the action the policy plays. The generator ties
/// power to capacity (both scale with the same per-battery size factor), so one step always moves
/// about 8% of the SOC span and the reachable range spans at most four grid nodes at the largest
/// grid this solver builds: the aligned set is smaller than the uniform grids it replaces as well
/// as exact.
///
/// If the reachable range spans more nodes than `out` can hold, the range is sampled uniformly
/// instead, which is the old behaviour.
fn aligned_actions(
    out: &mut [f64],
    edge: f64,
    soc: f64,
    lo: f64,
    dk: f64,
    bins: usize,
    eta_c: f64,
    eta_d: f64,
) -> usize {
    let room = out.len();
    if room == 0 || edge == 0.0 || dk <= 0.0 || bins == 0 {
        return 0;
    }
    // MWh of store moved per MW of action in this direction, i.e. |d soc_next / du|.
    let rate = if edge > 0.0 { DT / eta_d } else { eta_c * DT };
    if rate <= 0.0 {
        out[0] = edge;
        return 1;
    }
    let mag = edge.abs();
    let top_x = (bins - 1) as f64;
    // Grid coordinates of the current SOC and of the SOC reached at the bound.
    let x0 = (soc - lo) / dk;
    let span = mag * rate / dk;
    let dir_down = edge > 0.0;
    let x1 = if dir_down { x0 - span } else { x0 + span };
    // Integer grid nodes strictly between x0 and x1, walked outward from x0.
    let (first, stop) = if dir_down {
        ((x0 - 1e-9).floor(), (x1 + 1e-9).ceil())
    } else {
        ((x0 + 1e-9).ceil(), (x1 - 1e-9).floor())
    };
    let count = if dir_down {
        (first - stop) as i64 + 1
    } else {
        (stop - first) as i64 + 1
    };
    let count = count.max(0) as usize;
    if count + 1 > room {
        for i in 0..room {
            out[i] = edge * ((i + 1) as f64 / room as f64);
        }
        return room;
    }
    let mut n = 0usize;
    let mut k = first;
    for _ in 0..count {
        if k >= 0.0 && k <= top_x {
            let target = lo + k * dk;
            let u = if dir_down {
                (soc - target) / rate
            } else {
                -(target - soc) / rate
            };
            let abs_u = u.abs();
            // Round-off can push a node onto 0 or onto the bound; both are already candidates.
            if abs_u > 1e-12 && abs_u < mag {
                out[n] = u;
                n += 1;
            }
        }
        k += if dir_down { -1.0 } else { 1.0 };
    }
    out[n] = edge;
    n + 1
}

/// Linear interpolation of a value-function row over the SOC grid.
fn interp(row: &[f64], soc: f64, lo: f64, dk: f64, bins: usize) -> f64 {
    if bins == 0 || row.is_empty() {
        return 0.0;
    }
    let x = (soc - lo) / dk;
    if !(x > 0.0) {
        return row[0];
    }
    let floor = x.floor();
    let k0 = floor as usize;
    if k0 + 1 >= bins {
        return row[bins - 1];
    }
    let frac = x - floor;
    row[k0] * (1.0 - frac) + row[k0 + 1] * frac
}

fn flows_ok(challenge: &Challenge, state: &State, action: &[f64]) -> bool {
    let injections = challenge.compute_total_injections(state, action);
    let flows = challenge.network.compute_flows(&injections);
    challenge.network.verify_flows(&flows).is_ok()
}

// Important! Do not include any tests in this file, it will result in your submission being rejected
