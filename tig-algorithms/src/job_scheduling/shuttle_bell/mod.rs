// TIG's UI uses the pattern `tig_challenges::job_scheduling` to automatically detect your algorithm's challenge
//
// shuttle_bell -- c007 job_scheduling.
//
// Fuel-aware anytime search for the flexible job shop. Three ideas:
//
//   1. A guarded validity floor: the challenge's own effort-0 dispatching
//      baseline is replayed verbatim and saved before anything else, so the
//      hard gate at tig-challenges/src/job_scheduling/mod.rs:306 can never be
//      violated and quality can never go negative.
//   2. An optional stronger warm start (`warm_effort`): the same dispatcher
//      replayed at effort >= 1. Its RNG stream is a prefix-extension of the
//      effort-0 stream, so it is provably never worse than the floor. Measured
//      to cost more fuel than the head start is worth on fjsp_medium, so it is
//      off by default; it is kept for very small fuel budgets.
//   3. Anytime search driven by `__fuel_remaining` rather than by a fixed
//      iteration count, with a Mastrolilli-Gambardella machine-reassignment
//      neighbourhood on top of the Nowicki-Smutnicki critical-block moves.
//      Every improvement is saved immediately, so being killed for running out
//      of fuel is harmless.
//
// See `help()` for the measurement notes.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::cell::RefCell;
use tig_challenges::job_scheduling::*;

use rand::{rngs::SmallRng, SeedableRng};

mod graph;
mod model;
mod ref_greedy;
mod tabu;

use graph::{Graph, GraphState};
use model::Model;
use tabu::{Budget, TabuCfg};

// `__fuel_remaining` is initialised by the runtime to the fuel cap (see
// tig-runtime/src/main.rs) and decremented by the forked LLVM fuel
// instrumentation pass as the algorithm executes; it is exported from the built
// `.so` (tig-binary/scripts/build_so). Budgeting against it instead of
// wall-clock time keeps the anytime behaviour deterministic across grading
// machines.
extern "C" {
    #[allow(non_upper_case_globals)]
    static __fuel_remaining: u64;
}

#[inline(always)]
fn fuel_remaining() -> u64 {
    unsafe { core::ptr::read_volatile(core::ptr::addr_of!(__fuel_remaining)) }
}

#[derive(Serialize, Deserialize, Clone)]
#[serde(default)]
pub struct Hyperparameters {
    /// Accepted for compatibility with benchmarker configs written for other
    /// c007 algorithms. The track is auto-detected from the instance, so this
    /// value is not required and not used.
    pub track: String,
    /// Hard cap on the fuel the search may spend, in units of fuel.
    /// 0 = spend everything available minus the safety reserve.
    ///
    /// The measured quality/fuel curve is flat well before the 5e12 fuel budget
    /// benchmarkers hand out, so the default caps the spend: it buys almost all
    /// of the attainable quality at a small fraction of the wall time, which is
    /// what actually decides throughput (and therefore adoption).
    pub fuel_cap: u64,
    /// Fuel safety reserve = available / reserve_div.
    pub reserve_div: u64,
    /// Dispatching-rule effort used for the warm start (0 = reuse the floor).
    /// Effort e costs roughly (15 + 200 + 50e) dispatcher runs.
    pub warm_effort: usize,
    /// Minimum tabu tenure.
    pub tenure_min: usize,
    /// Tenure is drawn from [tenure_min, tenure_min + tenure_span].
    pub tenure_span: usize,
    /// Critical flexible operations sampled for machine-reassignment moves each
    /// iteration. 0 disables the FJSP layer.
    pub max_assign_moves: usize,
    /// Iterations without improvement before restarting from the incumbent best
    /// on the multi-route flexible tracks (fjsp_medium, fjsp_high).
    pub restart_after: u32,
    /// Same, for the single-route flexible track (hybrid_flow_shop).
    pub restart_after_hybrid: u32,
    /// Same, for the multi-route track with exactly one eligible machine per
    /// operation (job_shop), where the search is pure sequencing.
    ///
    /// The old value of 2000 was tuned on the `h2h_v1` seed and is far too
    /// eager on a real production seed: resetting to the incumbent every 2000
    /// iterations stops the tabu search before it has left the basin. Measured
    /// on job_shop over 16 production-seed nonces at 5e10 fuel, 20000 moves the
    /// median from 72,774 to 83,470 -- 12 paired wins, 1 tie, 3 losses -- and
    /// triples the share of nonces above the live qualifier bar. 50000 is
    /// indistinguishable from 20000; 200000 is worse.
    pub restart_after_rigid: u32,
    /// Same, for the single-route rigid track (flow_shop).
    ///
    /// Kept at the old value on purpose: the same 16-nonce sweep moves
    /// flow_shop's median the *other* way (5,096 -> 3,385 at 20000), so the two
    /// rigid tracks want opposite settings and the shape detector separates
    /// them -- flow_shop is the single-route one.
    pub restart_after_flow: u32,
    /// Random adjacent swaps applied when restarting.
    pub perturb_kicks: usize,
    /// Enable the block-insertion (N7-style) moves.
    pub use_n7: bool,
    /// Cap on candidate moves generated per tabu iteration (0 = unlimited).
    pub max_moves: usize,
    /// Best-estimated moves tried per iteration before giving up.
    pub verify_tries: usize,
    /// Also try reinserting a critical operation at its best position on its
    /// own machine.
    pub self_reinsert: bool,
}

impl Default for Hyperparameters {
    fn default() -> Self {
        Hyperparameters {
            track: String::new(),
            fuel_cap: DEFAULT_FUEL_CAP,
            reserve_div: 64,
            warm_effort: 0,
            tenure_min: 8,
            tenure_span: 8,
            max_assign_moves: 32,
            restart_after: 100000,
            restart_after_hybrid: 100000,
            restart_after_rigid: 20000,
            restart_after_flow: 2000,
            perturb_kicks: 8,
            use_n7: true,
            max_moves: 0,
            verify_tries: 4,
            self_reinsert: false,
        }
    }
}

const DEFAULT_FUEL_CAP: u64 = 500_000_000_000;

pub fn help() {
    println!(
        "shuttle_bell -- c007 job_scheduling

Design
------
1. Validity floor. Challenge::evaluate_solution rejects any makespan above the
   effort-0 dispatching baseline (tig-challenges/src/job_scheduling/mod.rs:306),
   and that rejection invalidates the whole benchmark, not just the nonce.
   ref_greedy.rs is a verbatim copy of that baseline, so replaying it on the
   same challenge.seed reproduces the floor exactly. It is saved before any
   search runs and every later save is checked against it, so quality is never
   negative and the gate is never tripped.

2. Warm start. The same dispatcher is replayed at effort 1 (200 randomised
   restarts + local search). Effort 0 draws its restarts from the same RNG
   stream, so effort 1's restart set is a superset of effort 0's and its best is
   provably <= the floor. That is also exactly how the SOTA reference is
   computed, so the search starts at quality 0 instead of ~-11k.

3. Anytime search. The schedule is imported into a disjunctive graph and
   re-timed by longest path (which only left-shifts). Tabu search then works the
   Nowicki-Smutnicki critical-block neighbourhood -- reverse the first/last
   machine arc of a critical block, block-insertion moves -- plus a
   Mastrolilli-Gambardella layer that moves a critical operation to another
   eligible machine at the *best* insertion position on that machine, found by a
   single head/tail pass over the target sequence. On fjsp_medium (~1776
   operations, ~2.5 eligible machines each, only ~40 operations pinned) the
   instance is assignment-dominated, and that layer is where the gain is.

   Candidate moves are ranked by an O(1) head/tail estimate; only the chosen one
   is evaluated, so an iteration costs ~1.3 longest-path evaluations rather than
   one per candidate. Cycles created by a move are caught by the evaluation and
   the move is discarded, so reentrant routes need no special case.

4. Fuel. The loop reads `__fuel_remaining` (via read_volatile) and stops at a
   reserve, saving on every improvement. Nothing depends on wall-clock time or
   on a fixed iteration count, so the same binary is anytime-correct at 1e10 and
   at 5e12 fuel.

Hyperparameters (all optional; unknown keys are ignored)
  fuel_cap          u64   hard cap on fuel spent (0 = all)      (default 5e11)
  reserve_div       u64   reserve = available / reserve_div     (default 64)
  warm_effort       usize dispatcher effort for the warm start  (default 0)
  tenure_min        usize minimum tabu tenure                   (default 8)
  tenure_span       usize tenure randomisation span             (default 8)
  max_assign_moves  usize reassignment moves per iteration      (default 32)
  restart_after     u32   restart threshold, fjsp_* tracks      (default 100000)
  restart_after_hybrid u32 restart threshold, hybrid_flow_shop   (default 100000)
  restart_after_rigid  u32 restart threshold, job_shop            (default 20000)
  restart_after_flow   u32 restart threshold, flow_shop           (default 2000)
  perturb_kicks     usize random swaps on restart               (default 8)
  verify_tries      usize moves evaluated per iteration at most (default 4)
  self_reinsert     bool  same-machine best reinsertion         (default false)
  max_moves         usize cap on candidate moves (0 = all)      (default 0)
  track             str   accepted and ignored (auto-detected)"
    );
}


/// The five tracks differ in two structural ways that are visible from the
/// instance itself, so the `track` hyperparameter never has to be trusted:
///
///   * `flow_structure = 0.0` (flow_shop, hybrid_flow_shop) makes the generator
///     build exactly one route and map every product onto it
///     (tig-challenges/src/job_scheduling/mod.rs:62-94), so every product has
///     the same route length and the same eligible-machine set at each
///     position. The other three tracks build 20-50 routes.
///   * `avg_op_flexibility` is 1.0 for flow_shop / job_shop, 3.0 for
///     hybrid_flow_shop, 3.0-but-measured-2.5 for fjsp_medium and 10.0 for
///     fjsp_high.
#[derive(Clone, Copy, PartialEq)]
enum Shape {
    Rigid,
    Flexible,
}

fn detect_shape(challenge: &Challenge, flex_ops: usize) -> (Shape, bool) {
    let pts = &challenge.product_processing_times;
    let mut single_route = true;
    if pts.len() > 1 {
        let len0 = pts[0].len();
        for p in pts.iter() {
            if p.len() != len0 {
                single_route = false;
                break;
            }
        }
        if single_route && len0 > 0 {
            let probes = [0usize, len0 / 3, (2 * len0) / 3];
            'outer: for &k in probes.iter() {
                let k = k.min(len0 - 1);
                let mut base: Vec<usize> = pts[0][k].keys().copied().collect();
                base.sort_unstable();
                for p in pts.iter().skip(1) {
                    let mut other: Vec<usize> = p[k].keys().copied().collect();
                    other.sort_unstable();
                    if other != base {
                        single_route = false;
                        break 'outer;
                    }
                }
            }
        }
    }
    let shape = if flex_ops == 0 {
        Shape::Rigid
    } else {
        Shape::Flexible
    };
    (shape, single_route)
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let hp = match hyperparameters {
        Some(map) => serde_json::from_value::<Hyperparameters>(Value::Object(map.clone()))
            .unwrap_or_default(),
        None => Hyperparameters::default(),
    };

    let fuel_at_entry = fuel_remaining();

    // ---------------------------------------------------------------------
    // Phase 1 -- validity floor.
    // ---------------------------------------------------------------------
    let captured: RefCell<Option<Solution>> = RefCell::new(None);
    {
        let sink = |s: &Solution| -> Result<()> {
            *captured.borrow_mut() = Some(s.clone());
            Ok(())
        };
        ref_greedy::solve_challenge_with_effort(challenge, &sink, 0)?;
    }
    let greedy = captured
        .into_inner()
        .ok_or_else(|| anyhow!("greedy baseline produced no solution"))?;
    let floor_ms = challenge.evaluate_makespan(&greedy)?;
    save_solution(&greedy)?;

    // ---------------------------------------------------------------------
    // Fuel plan. Everything below is optional: if we are killed at any point,
    // the last saved solution stands.
    // ---------------------------------------------------------------------
    let available = fuel_remaining();
    let reserve = available / hp.reserve_div.max(2);
    let mut max_spend = available.saturating_sub(reserve);
    if hp.fuel_cap > 0 && hp.fuel_cap < max_spend {
        max_spend = hp.fuel_cap;
    }
    let floor_fuel = available.saturating_sub(max_spend);

    // ---------------------------------------------------------------------
    // Phase 2 -- warm start: the same dispatcher at higher effort.
    //
    // `solve_challenge_with_effort(_, _, 0)` draws (seed, rule, top_k) triples
    // from `SmallRng::from_seed(challenge.seed)` for 10 restarts;
    // effort e >= 1 draws the same triples for the first 10 of its 200+
    // restarts. Its best is therefore <= the floor, always.
    // ---------------------------------------------------------------------
    let mut start_sol = greedy;
    let mut start_ms = floor_ms;
    if hp.warm_effort > 0 {
        // Only attempt it if there is clearly fuel for it: one effort-1 replay
        // costs ~14x an effort-0 replay, which we have just measured.
        let spent_on_floor = fuel_at_entry.saturating_sub(available);
        let warm_cost = spent_on_floor
            .saturating_mul(16)
            .saturating_mul(1 + hp.warm_effort as u64 / 4);
        if fuel_remaining() > floor_fuel.saturating_add(warm_cost) {
            let captured: RefCell<Option<Solution>> = RefCell::new(None);
            let sink = |s: &Solution| -> Result<()> {
                *captured.borrow_mut() = Some(s.clone());
                Ok(())
            };
            if ref_greedy::solve_challenge_with_effort(challenge, &sink, hp.warm_effort).is_ok() {
                if let Some(s) = captured.into_inner() {
                    if let Ok(ms) = challenge.evaluate_makespan(&s) {
                        if ms <= start_ms {
                            start_ms = ms;
                            start_sol = s;
                            save_solution(&start_sol)?;
                        }
                    }
                }
            }
        }
    }

    // ---------------------------------------------------------------------
    // Phase 3 -- import into the disjunctive graph and re-time.
    // ---------------------------------------------------------------------
    let md = Model::build(challenge);
    let mut g = Graph::new(&md);
    if !g.load_schedule(&md, &start_sol.job_schedule) {
        return Ok(());
    }
    let retimed = match g.evaluate() {
        Some(ms) => ms,
        None => return Ok(()),
    };
    // Re-timing by longest path only ever left-shifts, so this is always <= the
    // imported makespan and therefore always verifier-clean.
    let ceiling = retimed.min(start_ms);
    let mut best_state = GraphState::new(&md);
    g.snapshot(&mut best_state);
    if retimed < start_ms {
        save_solution(&Solution {
            job_schedule: g.to_schedule(&md),
        })?;
    }

    let budget = Budget {
        remaining: &fuel_remaining,
        floor: floor_fuel,
    };

    // Calibrate the cost of one longest-path evaluation.
    let before_cal = fuel_remaining();
    let _ = g.evaluate();
    let per_eval = before_cal.saturating_sub(fuel_remaining()).max(1);

    // ---------------------------------------------------------------------
    // Phase 4 -- tabu search. Saves are monotone and always <= the ceiling,
    // which is itself <= the effort-0 floor.
    // ---------------------------------------------------------------------
    let mut rng = SmallRng::from_seed(challenge.seed);
    let saved_ms = RefCell::new(ceiling);
    let mut on_improve = |gr: &Graph, ms: u32| {
        if ms >= *saved_ms.borrow() || ms > floor_ms {
            return;
        }
        if save_solution(&Solution {
            job_schedule: gr.to_schedule(&md),
        })
        .is_ok()
        {
            *saved_ms.borrow_mut() = ms;
        }
    };

    // Scale the reassignment layer with how flexible the instance actually is:
    // flow_shop / job_shop have one eligible machine per operation and generate
    // no moves at all, fjsp_high has ten and would otherwise dominate the
    // iteration cost.
    let mut flex_ops = 0usize;
    let mut flex_opts = 0usize;
    for o in 0..md.n_ops {
        let k = md.n_opts(o);
        if k > 1 {
            flex_ops += 1;
            flex_opts += k;
        }
    }
    let avg_flex = if flex_ops > 0 {
        flex_opts / flex_ops
    } else {
        1
    };
    let assign_moves = if flex_ops == 0 {
        0
    } else {
        (hp.max_assign_moves * 2 / avg_flex.max(1)).max(4)
    };

    // Restart policy is the one setting that is genuinely track-dependent, and
    // all four shapes the detector can tell apart want a different value. On
    // the flexible tracks the reassignment layer needs long uninterrupted runs
    // and periodic resets to the incumbent cost more than they buy. On the
    // rigid ones the search is pure sequencing and does need them, but
    // job_shop wants them an order of magnitude rarer than flow_shop does --
    // measured, not assumed; see `restart_after_rigid`.
    let (shape, single_route) = detect_shape(challenge, flex_ops);
    let restart_after = match (shape, single_route) {
        (Shape::Rigid, true) => hp.restart_after_flow,
        (Shape::Rigid, false) => hp.restart_after_rigid,
        (Shape::Flexible, true) => hp.restart_after_hybrid,
        (Shape::Flexible, false) => hp.restart_after,
    };

    let cfg = TabuCfg {
        tenure_min: hp.tenure_min.max(1),
        tenure_span: hp.tenure_span,
        max_assign_moves: assign_moves,
        restart_after: restart_after.max(1),
        perturb_kicks: hp.perturb_kicks,
        use_n7: hp.use_n7,
        max_moves: hp.max_moves,
        verify_tries: hp.verify_tries,
        self_reinsert: hp.self_reinsert,
    };

    tabu::tabu_search(
        &md,
        &mut g,
        &mut best_state,
        ceiling,
        &cfg,
        &mut rng,
        &budget,
        per_eval,
        &mut on_improve,
    );

    Ok(())
}

// Important! Do not include any tests in this file, it will result in your submission being rejected
