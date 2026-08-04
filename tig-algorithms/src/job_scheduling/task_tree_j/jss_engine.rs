// Self-contained scheduling engine: specialised flow-shop and job-shop solvers.
#![allow(dead_code, clippy::all)]
pub mod types {
pub const INF: u32 = u32::MAX / 4;
pub const NONE_USIZE: usize = usize::MAX;

#[derive(Clone)]
pub struct OpInfo {
    pub machines: Vec<(usize, u32)>,
    pub min_pt: u32,
    pub avg_pt: f64,
    pub flex: usize,
    pub bn_avg: f64,
}

#[derive(Clone, Copy, Default)]
pub struct OpRoute {
    pub best_m: u8,
    pub best_w: u8,
    pub second_m: u8,
    pub second_w: u8,
}

pub type RoutePrefLite = Vec<Vec<OpRoute>>;

#[derive(Clone)]
pub struct Pre {
    pub job_products: Vec<usize>,
    pub job_ops_len: Vec<usize>,
    pub product_ops: Vec<Vec<OpInfo>>,
    pub product_suf_min: Vec<Vec<u32>>,
    pub product_suf_avg: Vec<Vec<f64>>,
    pub product_suf_bn: Vec<Vec<f64>>,
    pub product_next_min: Vec<Vec<u32>>,
    pub product_next_flex_inv: Vec<Vec<f64>>,
    pub machine_load0: Vec<f64>,
    pub machine_scarcity: Vec<f64>,
    pub machine_weight: Vec<f64>,
    pub machine_best_pop: Vec<f64>,
    pub avg_machine_load: f64,
    pub avg_machine_scarcity: f64,
    pub avg_op_min: f64,
    pub horizon: f64,
    pub time_scale: f64,
    pub max_ops: usize,
    pub max_job_avg_work: f64,
    pub max_job_bn: f64,
    pub flex_avg: f64,
    pub flex_factor: f64,
    pub hi_flex: bool,
    pub high_flex: f64,
    pub flow_like: f64,
    pub flow_w: f64,
    pub job_flow_pref: Vec<f64>,
    pub jobshopness: f64,
    pub bn_focus: f64,
    pub load_cv: f64,
    pub slack_base: f64,
    pub total_ops: usize,
    pub chaotic_like: bool,
    pub flow_route: Option<Vec<usize>>,
    pub flow_pt_by_job: Option<Vec<Vec<u32>>>,
    pub strict_route: Option<Vec<usize>>,
}

#[derive(Clone, Copy)]
pub struct Cand {
    pub job: usize,
    pub machine: usize,
    pub pt: u32,
    pub score: f64,
}

#[derive(Clone, Copy)]
pub struct RawCand {
    pub job: usize,
    pub machine: usize,
    pub pt: u32,
    pub base_score: f64,
    pub rigidity: f64,
    pub reg_n: f64,
}

#[derive(Clone, Copy)]
pub enum GreedyRule {
    MostWork,
    MostOps,
    LeastFlex,
    ShortestProc,
    LongestProc,
}

#[derive(Clone)]
pub struct DisjSchedule {
    pub n: usize,
    pub num_jobs: usize,
    pub num_machines: usize,
    pub job_offsets: Vec<usize>,
    pub job_succ: Vec<usize>,
    pub indeg_job: Vec<u16>,
    pub node_machine: Vec<usize>,
    pub node_pt: Vec<u32>,
    pub node_job: Vec<usize>,
    pub node_op: Vec<usize>,
    pub machine_seq: Vec<Vec<usize>>,
}

pub struct EvalBuf {
    pub indeg: Vec<u16>,
    pub start: Vec<u32>,
    pub best_pred: Vec<usize>,
    pub machine_succ: Vec<usize>,
    pub stack: Vec<usize>,
}

impl EvalBuf {
    pub fn new(n: usize) -> Self {
        Self {
            indeg: vec![0u16; n],
            start: vec![0u32; n],
            best_pred: vec![NONE_USIZE; n],
            machine_succ: vec![NONE_USIZE; n],
            stack: Vec::with_capacity(n),
        }
    }
}

#[derive(Clone, Copy)]
pub struct MoveCand {
    pub kind: u8,
    pub m_from: usize,
    pub from: usize,
    pub m_to: usize,
    pub to: usize,
    pub new_pt: u32,
    pub score: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct EffortConfig {
    pub job_shop_iters: usize,
    pub hybrid_flow_shop_iters: usize,
    pub fjsp_medium_iters: usize,
    pub fjsp_high_iters: usize,
    /// t48 only (read exclusively by `mod job_shop`): how the `ts_starts` tabu-search seeds
    /// are picked out of `top_solutions`.
    /// 0 = legacy makespan prefix `[0..ts_starts]` (byte-identical control),
    /// 1 = farthest-point selection on the disjunctive-order signature (treatment),
    /// 2 = prefix with the LAST slot forced to pool rank `ts_starts` (liveness control).
    pub js_seed_select_mode: usize,
    /// t48 only (read exclusively by `mod job_shop::tabu_search_phase`): evaluation-kernel mode.
    /// 0 = legacy (byte-identical control): every tabu iteration runs TWO independent full O(n)
    ///     topological sorts — the backward `tail` pass and `eval_disj`'s forward pass — and
    ///     `eval_disj` rebuilds `machine_succ`/`indeg` from `machine_seq` from scratch.
    /// 1 = fast kernel (treatment): the forward pass records its topological order, the backward
    ///     `tail` pass replays it in reverse (no in-degree seeding, no worklist), and
    ///     `machine_succ` / the machine in-degree contribution are maintained incrementally
    ///     across adjacent swaps. **Semantics-preserving: `tail`, `start`, `best_pred`, `mk` and
    ///     `mk_node` are bit-for-bit identical to mode 0**, so Q must not move — only the wall.
    pub js_ts_fast_eval: usize,
    /// t48 only (read exclusively by `mod job_shop`): how many entries of `top_solutions` are
    /// actually consumed as tabu-search starting points. The site used to be the literal
    /// `top_solutions.len().min(10)`, while the pool feeding it is capped at **15**
    
    
    /// ADDITIVE at constant `job_shop_iters` (`ts_iters` is PER start): each extra start is one
    /// more full tabu run from another basin, retained only through `if mk2 < best_makespan`, so
    /// the TS phase itself can only improve or stay flat. Default 10 ⇒ inert unless raised
    /// through `hp_json`.
    pub js_ts_starts: usize,
    /// t48 only (read exclusively by `mod job_shop::tabu_search_phase`): the STAGNATION PATIENCE of
    /// the tabu search, in non-improving iterations, decoupled from the iteration budget.
    ///
    /// 0 = legacy law (byte-identical control). Every diversification trigger is a FRACTION OF
    ///     `max_iterations`: `max_no_improve = max_iterations/2`, `kick_threshold = 2/3` of that,
    ///     `diversify_threshold = 1/3` of that. At the injected `job_shop_iters = 200 000` this
    ///     means the search tolerates **100 000** consecutive non-improving iterations before it
    ///     jumps back to the incumbent and **66 666** before the first guided kick, so a start that
    ///     converges early spends the overwhelming majority of its budget on a dead plateau and the
    ///     whole run performs only ~2 intensification cycles.
    /// n > 0 = the patience becomes an ABSOLUTE work quantity of `n` non-improving iterations and
    ///     the two derived thresholds compress with it. The stall-out budget is PRESERVED: the kick
    ///     reservoir is refilled proportionally (`4 × ⌊max_iterations / n⌋`) so the run still gives
    ///     up after the same total fraction (~0,8) of `max_iterations` as the legacy law — only the
    ///     CADENCE changes, never the amount of work. See `tabu_search_phase`.
    ///
    
    /// 2026-06-10): raising `ts_iters` 85 000 → 87 000 *degraded* Q "by delaying kicks by 667
    /// iterations of patience, which delayed diversification and reduced exploration **due to
    /// kick_threshold scaling with max_iterations**". That coupling was diagnosed then and never
    /// repaired; it is the standing explanation for t48's STAGE 2 result (`job_shop_iters`
    /// 200 000 → 240 000 = 88 813 → 87 657, bench 25345), where +40 000 iterations of budget also
    /// bought +20 000 iterations of extra patience and cost 1 156 Q.
    pub js_ts_div_period: usize,
    
    /// the N5 move-selection argmin.
    ///
    /// The selection loop scores every candidate swap with `estimate_swap_mk` and keeps the best
    /// under a STRICT `adj_mk < best_move_key`, so on equality the FIRST candidate reached wins.
    /// "First reached" is an enumeration artefact, not a decision: machines in `crit_pos_machines`
    /// sorted order, blocks in increasing position, `run_start` before `run_end-1`. Ties are
    /// expected to be frequent because `estimate_swap_mk` is a LOCAL bound — it measures only the
    /// two longest paths rewritten through `u` and `v`, so every swap leaving those two paths at
    /// the same length collapses onto the same integer key, whatever it does to the rest of the
    /// graph. Whenever that happens the enumeration order — and nothing else — picks the move, and
    /// therefore picks the whole downstream trajectory.
    ///
    /// ISO-WORK BY CONSTRUCTION: one integer comparison and one counter per candidate, no extra
    /// move evaluated, and neither `no_improve`, nor `kicks_left`, nor `max_no_improve`, nor
    /// `max_iterations` is touched — it cannot move the wall the way `js_ts_div_period` did. It is
    /// also NOT a move-set change: the candidate set stays exactly the N5 set of the baseline,
    /// whose *enrichment* is dead on 6 witnesses (`brick:js_ts_neighborhood_n6` / `_n7`).
    ///
    /// 0 = legacy first-wins (byte-identical control).
    /// 1 = uniform reservoir sampling among the tied candidates, driven by a dedicated xorshift
    ///     `tseed` seeded from `challenge.seed[0]` (deterministic, no wall-clock, kept apart from
    ///     `pseed` so the tabu-tenure stream is untouched). TREATMENT.
    /// 2 = longest-critical-block wins. Structural and deterministic: a longer critical block is
    ///     more machine-critical work concentrated on one resource, so resolving it is the move
    ///     with the most makespan mass behind it.
    /// 3 = LAST-wins (`<=` instead of `<`). The exact logical negation of mode 0's arbitrary rule,
    ///     hence the POSITIVE CONTROL of the family: if mode 3 leaves Q byte-identical to the
    ///     baseline, ties never decide a move on these instances and the family closes in one
    ///     measurement; if it moves Q, the tie-break is a live zero-cost Q lever.
    pub js_ts_tie_mode: usize,
    
    
    /// on `max(path_u, path_v)` alone — the Taillard-1994 bound, which Nowicki & Smutnicki (2002)
    
    /// order statistic of the same arithmetic as a tie-level component, at zero extra work.
    /// 0 = legacy single-component key (byte-identical baseline).
    /// 1 = prefer the SMALLER second path among candidates tied on the bound.
    /// 2 = prefer the LARGER one — two-sided control, the exact negation of mode 1.
    pub js_ts_sel_key: usize,
    
    
    /// inside `estimate_swap_mk`, which is the Taillard-1994 lower bound over the two rewritten
    /// paths only: every path that avoids `u` and `v` is invisible to it, so candidates the bound
    
    /// are frequent AND decide moves (LAST-wins moved Q by −646), yet every BLIND rule landed under
    
    /// The untested quadrant is deterministic AND informed by the real objective.
    ///
    /// Mode `K >= 2` keeps the K best admissible candidates that TIE on the approximate key,
    /// recomputes their TRUE makespan with the very evaluator the loop already runs, and picks the
    /// true argmin (first-wins on an exact tie, so the rule stays deterministic and RNG-free).
    /// This is the "approximate screen over the whole neighbourhood → exact rerank of the top K"
    
    /// JOB_SHOP), never implemented on this track.
    ///
    /// 0 = off, legacy approximate argmin (byte-identical baseline). 1 collapses to 0 (a single
    /// candidate is its own argmin). Clamped to 4: the exact probes are the only super-O(1) work
    /// this iteration adds, and the reservoir in `tabu_search_phase` caps their total count anyway.
    pub js_ts_exact_topk: usize,

    
    /// multi-start. `0` keeps the legacy behaviour bit-for-bit.
    ///
    /// The multi-start at `:9142` runs `ts_starts` tabu searches and keeps the best. The starts
    /// differ ONLY by their seed solution — every one of them derives its tenure-jitter stream
    /// `pseed` from the SAME `(challenge.seed[0], initial_mk, n)` law, so ten starts are ten
    
    /// median instances (ranks 13-28 of the 32 nonces) the result is BYTE-IDENTICAL at
    /// `js_ts_starts` 4, 10 and 12 — adding starts adds no information, the effective sample size
    
    /// measured that a DIFFERENT trajectory over the same seeds lands on a genuinely different
    /// local optimum: the trajectory-to-trajectory spread of this search is ~1.6 % of Q, and it is
    /// currently thrown away because the ten starts all walk the same one.
    ///
    /// This knob salts `pseed` with the RANK of the start, and nothing else. The tenure law, its
    /// interval and its distribution are untouched — no candidate is ever judged differently, and
    /// no work is added (`hp:ts_iters_funding` is dead: every move on this track must be ISO-WORK,
    /// and this one is exactly free). Rank 0 is salted with 0, i.e. the incumbent champion
    /// trajectory is kept verbatim inside the portfolio.
    ///
    
    /// executed in parallel, each maintaining its OWN current solution, tabu list and constraint
    /// weights to explore INDEPENDENT search trajectories". Our ten instances own their tabu list
    /// but not their dynamic — this is what makes the trajectories independent.
    ///
    /// `1` and `2` are two different salt constants: same mechanism, different draw. They are a
    /// BILATERAL control — if only one of them gains, the gain is a lottery ticket, not a
    /// mechanism. Anything else degrades to `0`.
    pub js_ts_traj_decorr: usize,

    
    /// to the multi-start portfolio. `0` keeps the legacy behaviour bit-for-bit.
    ///
    
    
    /// without naming it: they REPLACED the ten baseline trajectories by ten other ones. Every one
    /// of the five landed strictly below 88 813 (88 364 / 88 167 / 87 356 / 87 283 / 86 187), a
    /// 5-of-5 one-sided signature which reads as: the baseline trajectory set is a good draw and
    
    /// never substituted" — the code falsifies that claim: salting ranks 1..9 DELETES nine baseline
    /// trajectories from the `min` at `:9220`. Nobody has ever ADDED a draw.
    ///
    /// An added draw cannot lose. The extra runs below are fenced so that the whole baseline
    /// computation stays bit-identical: their results are NOT pushed into `top_solutions` (which
    
    
    /// solution pool or shifted RNG"), and they do NOT touch `best_makespan` / `best_solution`
    /// (which the bottleneck, ILS, memetic and final-polish phases all read). They accumulate in a
    /// side variable consulted once, at the very last statement of `solve`. Consequence, per nonce
    /// and not merely in expectation: `mk_new = min(mk_baseline, mk_extra) <= mk_baseline`, so
    /// `Q_new >= Q_baseline` on EVERY nonce. The only thing that can move against us is the clock.
    ///
    /// The price is measured, not guessed: `js_ts_starts` 10 -> 12 cost 685.59 s -> 815.12 s
    /// (jobs 25494 / 25493), i.e. ~65 s per start, against a gate of 865 000 ms and a baseline of
    /// 689 090 ms (VC 25611). Two extra draws land at ~810 s, inside the gate; three would not.
    /// This is the track's 176 s of unspent headroom finally being converted into Q, which is what
    /// `roadmap/t48.md` has been asking for, and it is NOT `hp:ts_iters_funding` (dead): no existing
    /// start is shortened to pay for it.
    ///
    /// Each extra draw `e` re-runs the seed `ts_seed_idx[e % ts_starts]` — a pool ELITE in its
    /// pre-tabu state, exactly as the legacy starts consume it — with a distinct non-zero salt.
    /// That is NOT `engine:chained_second_ts` (dead): that family re-entered tabu on a solution the
    /// tabu had already driven to a local optimum, a provable no-op. Here the seed has never been
    /// walked by THIS trajectory.
    ///
    /// Clamped to 4 (the gate cannot afford more than 2, and a malformed `hp_json` must degrade to
    /// something benchable rather than to a time-out).
    pub js_ts_extra_draws: usize,

    
    /// guided kicks. `0` = the legacy predicate, byte-for-byte.
    ///
    /// The legacy rule fires on `no_improve % kick_threshold == 0` and then `continue`s WITHOUT
    /// advancing `no_improve` (the `no_improve += 1` at the bottom of the loop is skipped by that
    /// `continue`), so the condition is still true on the next iteration: the whole reservoir of
    /// four kicks drains in four CONSECUTIVE iterations, at ONE single stagnation depth
    /// (`kick_threshold` = `max_iterations/3` = 66 666 at the injected `job_shop_iters = 200 000`),
    /// and the branch is then dead for the remaining ~33 000 iterations. A 200 000-iteration start
    /// is therefore one descent perturbed ONCE, by twelve swaps applied back-to-back with no tabu
    /// search in between to exploit them.
    ///
    /// Under `v > 0` the k-th kick fires at `no_improve == k * max_no_improve / v` for `k = 1..4`
    /// and `no_improve` is advanced by one, so the four kicks are SPREAD across the same plateau
    /// and each is followed by real tabu iterations. `max_no_improve`, `cycles_left`, `div_period`,
    /// the break condition, the kick strength (`num_kicks`) and the kick direction
    /// (`use_base_first`) are all untouched: the run keeps the same iteration cap, the same four
    /// kicks and the same perturbation, and only their PLACEMENT changes.
    pub js_ts_kick_spread: usize,

    
    /// Metropolis acceptance test placed on the move of the EXTRA draws ONLY. `0` = no test at all,
    /// byte-for-byte legacy.
    ///
    /// WHY THE ACCEPTANCE CRITERION, AND WHY ONLY IN THE EXTRA DRAWS. `tabu_search_phase` has no
    /// acceptance rule: `chosen = best_move.or(relaxed_move)` is applied UNCONDITIONALLY, so the
    /// walk is "always move to the best admissible neighbour". Every one of the twelve dead
    /// families on this track changes the MOVES (n6/n7/tie_mode/sel_key/exact_topk), the SEEDS
    /// (select_mode/mandatory_k/dedup/path-relinking), the WORK (job_shop_iters/ts_starts/funding)
    /// or the PERTURBATION (div_period/ruin-recreate/kick_spread). None has ever changed WHAT THE
    
    ///
    
    /// three nonces (Σ +6 717) and did not move nonce 30 or nonce 5 — the two that FIX the median —
    /// by a single unit. The SAMPLING of that dynamic is closed; the DYNAMIC is not. So the extra
    /// draw keeps its seed, its budget and its neighbourhood, and changes only its acceptance law.
    ///
    
    /// never write `top_solutions`, `best_makespan`, `best_solution` nor call `save_solution`, and
    /// are consulted once, in strict `<`, at the last statement of `solve`. Q is therefore `>=` the
    /// baseline on EVERY nonce by construction, whatever the temperature does — this iteration
    /// cannot regress, it can only fail to gain. It is also the exact opposite of the geste that
    
    pub js_ts_sa_accept: usize,

    
    /// recombine. `0` = legacy, byte-for-byte (every machine flagged `is_bottleneck` is skipped and
    /// the child inherits the better parent's sequence for it verbatim).
    ///
    /// WHY THIS BRICK, AND WHY IT IS NOT ONE OF THE TWELVE DEAD FAMILIES. Every dead family on this
    /// track lives inside `tabu_search_phase` or in what feeds it: the MOVES (n6/n7/tie_mode/
    /// sel_key/exact_topk), the SEEDS (select_mode/mandatory_k/dedup/path-relinking), the WORK
    /// (job_shop_iters/ts_starts/funding), the PERTURBATION (div_period/ruin-recreate/kick_spread)
    
    /// champion has not moved off 88 813. The memetic phase — the only POPULATION engine in `solve`,
    /// and the phase that runs LAST before the final polish — has never been touched by any of them.
    ///
    /// THE STRUCTURAL MISMATCH THIS TESTS. The offline instance probe (`analyse/tools/jssp_lb_probe.rs`)
    /// established that `Scenario::JOB_SHOP` draws operation times as `base[op_type] * U[0.8,1.2]`
    /// with `base` in `[1,200]`, so machine loads are extremely unequal and the makespan is BOUNDED
    /// BY THE BOTTLENECK MACHINE. The crossover does the exact opposite of what that implies: it
    /// recombines only the NON-bottleneck machines and copies the better parent's bottleneck
    /// sequences unchanged. The population therefore never recombines the decision variables that
    /// set the objective — every child differs from its better parent only on machines that cannot,
    /// on their own, lower the makespan. That is a plausible mechanical reason why the median
    /// instances land on the same value no matter what the upstream tabu search is told to do.
    ///
    /// - `1` = recombine EVERY machine, bottleneck included — the hypothesis.
    /// - `2` = recombine the BOTTLENECK machines ONLY — the exact negation of the legacy scope, and
    
    ///   If `1` and `2` both leave Q byte-identical, the memetic phase is inert on this track and the
    ///   next iteration belongs somewhere else entirely; if they move Q in opposite directions, the
    ///   scope is a real lever and the sign says which way.
    ///
    /// ISOLATION. `inherit_machine_consensus_order` draws no randomness, and the mutation block below
    /// it draws exactly the same number of values from `rng` whatever the scope, so this iteration
    
    /// Feasibility stays arbitrated by the evaluator the solver already trusts: an infeasible child
    /// makes `critical_block_move_local_search_ex_disj` / `eval_disj` return `None` and the
    /// generation is skipped, exactly as in legacy. No tolerance is widened anywhere.
    pub js_mem_cross_scope: usize,
    
    ///
    
    /// concluded the crossover was "inert". The offline replay of the whole solver (harness
    /// `/tmp/jsscheck`, `jsi=60000`, nonces 30 and 5 = the two that set the median) shows the
    /// real cause, and it is a BUG, not a saturation: the generation loop takes the `continue`
    /// at the local-search site **12 times out of 12**, because `eval_disj` on the child returns
    /// `None`. `eval_disj` returns `None` on exactly one condition — a CYCLE in the disjunctive
    /// graph. Counters: `children=0`, `crossed_machines=288`, `identical=0`. So the crossover
    /// really does recombine 24 machines per generation, and every single resulting child is
    /// INFEASIBLE.
    ///
    /// That is the textbook failure of recombining machine sequences INDEPENDENTLY per machine:
    /// each `inherit_machine_consensus_order` call applies up to 2 disjunctive reversals, ~24
    /// machines are touched at once, and acyclicity of the union is never checked. Nothing
    /// downstream repairs it, so the block has been burning its whole budget to throw everything
    /// away, and every HP measured "on top of" it (`js_mem_cross_scope`) was measuring a no-op.
    ///
    /// `1` = per-machine admission gate: the child starts from the (feasible) better parent, each
    /// machine's inherited order is applied tentatively, `eval_disj` is run, and the machine is
    /// REVERTED if it made the graph cyclic. The child is feasible BY CONSTRUCTION and still
    /// inherits every machine that is mutually compatible — measured 169 accepted / 93 reverted
    /// on nonce 30, i.e. the recombination stays genuinely multi-machine, which is the property
    /// the track needs (the residual 4-6 % over the one-machine lower bound cannot be removed by
    
    ///
    /// `2` and `3` add the second half of the repair. With `1` alone the children are feasible
    /// but never beat the incumbent (`beats_best=0`): they are judged after a 4-round / 250-iter
    /// local search while the incumbent came out of a 200 000-iteration tabu search, so the
    /// comparison is rigged. `2` refines each child with a REAL `tabu_search_phase` at
    /// `ts_iters/24`, `3` at `ts_iters/8`. Measured at `jsi=60000` on nonce 30: `beats_best`
    /// 0 → 2, `pop_in` 0 → 15, and the wall time is 17.5 s → 17.6 s (`2`) / 18.4 s (`3`), so the
    /// refinement is paid out of budget that was previously being wasted on infeasible children.
    /// This is NOT `hp:ts_iters_funding` (dead-listed): the main multi-start budget is untouched.
    ///
    /// `0` keeps the legacy path expression-for-expression, so the default binary is
    /// byte-identical by construction rather than by inspection.
    pub js_mem_admit: usize,

    
    /// once every legacy phase is done. `0` = the whole block is skipped, byte-for-byte legacy.
    ///
    
    /// 1 returned the SAME 32 nonce qualities, to the unit (jobs 25661 / 25662). Recombining the six
    /// bottleneck machines instead of copying them verbatim changes every child of all 18 memetic
    /// generations, yet not one nonce moved. The only state the memetic block can export is
    /// `best_solution`, so the measurement says the block never lowers `best_makespan` on ANY of the
    
    
    /// therefore `engine:chained_second_ts` — a family already dead-listed as a no-op that hurts.
    ///
    /// WHAT IS LEFT AFTER THAT. The thirteen dead families all act on `tabu_search_phase` or on what
    /// feeds it: MOVES (n5/n6/n7/n8, tie_mode, sel_key, exact_topk), SEEDS (select_mode, mandatory_k,
    /// dedup, path-relinking), WORK (job_shop_iters, ts_starts, funding), PERTURBATION (div_period,
    /// ruin-recreate, kick_spread) and ACCEPTANCE (Metropolis). Every one of them is a variation of
    /// the SAME operator class: reverse one or two disjunctive arcs at the border of a critical
    
    /// `js_ts_starts` 4, 10 and 12, i.e. ten different trajectories land on the same makespan.
    ///
    /// THE MEASURED TARGET. The offline probe (`analyse/tools/jssp_lb_probe.rs`) inverts the nonce
    /// mapping and reads the absolute makespan per instance: nonce 30 sits **4.23 %** and nonce 5
    /// **5.92 %** above the ONE-MACHINE head+tail lower bound, and those two ranks (15 and 16) are
    /// exactly what the median metric reads. `Scenario::JOB_SHOP` has `avg_op_flexibility = 1.0`, so
    /// every operation has a single eligible machine and the makespan is set by the bottleneck
    /// machine. The residual gap is therefore measured in the very bound that the Adams-Balas-Zawack
    
    /// on that method — "suppose des machines fixes, nécessite des extensions FJSP" — is not a caveat
    /// here: t48 IS the fixed-machine case.
    ///
    /// THE OPERATOR, AND WHY IT IS NOT ONE OF THE THIRTEEN. A sweep takes each machine in decreasing
    /// load order, rebuilds the heads and tails of its operations **with that machine's own
    /// disjunctive chain removed** — the true SBP subproblem, not the current schedule's heads — and
    /// re-solves the resulting `1 | r_j, q_j | C_max` EXACTLY with Carlier's branch and bound under a
    /// fixed node budget. It replaces a 50-operation sequence in one shot. No dead family reverses
    /// more than two arcs at a time, and the current bottleneck block only relocates inside a span of
    /// 2..4 positions over the FIRST 18 of those 50 operations. This is a different class of move,
    /// not a wider setting of the old one.
    ///
    
    /// after the final polish, reads `best_solution`, writes nothing else, calls no `save_solution`,
    
    
    /// `eval_disj` itself certified. A re-sequencing that would create a cycle in the disjunctive
    /// graph makes `eval_disj` return `None` and is discarded; no feasibility tolerance exists here
    /// to widen. Q is therefore `>=` baseline on EVERY nonce by construction: `Q < 88 813` would mean
    /// a leak in the fence, i.e. a bug, not a verdict.
    ///
    /// Sweeps stop early as soon as a full pass improves no machine, so the dose-response between a
    /// small and a large budget is readable: identical Q at both means the champion is already
    /// one-machine-optimal on every machine, which closes the whole re-sequencing class at once.
    pub js_sbp_rounds: usize,

    
    /// the final polish, is allowed to survive an in-loop evaluation failure and return the best it
    /// had already certified, instead of `return Ok(None)`-ing its whole run away.
    ///
    /// MEASURED, not read. The full-budget offline attribution probe (`jsi=200000`, nonces 30 and 5
    /// — the two that FIX the median, `bundle_quality` being the mean of ranks 15/16) reports:
    ///   * `early_loop_None = 2` on BOTH nonces ⇒ two tabu runs per nonce abort mid-flight today;
    ///   * on nonce 30, `polish_try=1 / polish_None=1` ⇒ **the aborting run IS the final polish**;
    ///   * on nonce 5 the polish survives and is worth **10 895 → 10 831 = −64 makespan units**,
    ///     i.e. ≈ +4 900 Q on that nonce at the measured barème of ~76 Q per unit — more than the
    ///     whole 8-start portfolio's marginal value.
    /// So the phase with the largest measured per-iteration yield in the solver is being discarded
    /// on one of the two nonces that set Q.
    ///
    /// WHY IT CANNOT REGRESS. The flag is passed to exactly ONE call site (the final polish); every
    /// other call site receives a literal `0`, so no seed, no `top_solutions` entry, no
    
    
    /// "contaminated the solution pool or shifted RNG state"). The polish's own result is consumed
    /// by `if mk4 < best_makespan { .. }` alone, and the rescued solution is re-certified by the
    /// legacy `eval_disj` before it is returned. Q is therefore `>=` baseline on EVERY nonce by
    /// construction: `Q < 88 813` would mean a leak in the fence, i.e. a bug, not a verdict.
    ///
    /// Cost: strictly NEGATIVE or nil in work — a rescued run stops at the abort instead of
    /// restarting the whole polish, so the ladder of downstream phases is unchanged and the gate
    /// (865 000 ms against a 686-698 s champion) is not at risk.
    pub js_ts_polish_rescue: usize,

    
    /// Adams-Balas-Zawack **constructive** shifting bottleneck (machine by machine, FROM THE EMPTY
    
    /// draws. `2` = same, with a re-optimisation sweep over the already-sequenced machines after
    /// every insertion (the ABZ "partial reoptimisation" variant) — a dose-response on the exactness
    /// of the construction, not on the budget.
    ///
    /// WHY THIS FAMILY IS NOT ANY OF THE 17 DEAD ONES. Every dead family either lives inside
    /// `tabu_search_phase` (the TRAJECTORY: moves, tenure, work, perturbation, acceptance) or
    /// CHOOSES / RECOMBINES inside a pool that already exists (`js_seed_select_mode`,
    /// `js_seed_mandatory_k`, `js_ts_seed_pool_exact_dedup`, `engine:seed_path_relinking`). The two
    
    
    /// and re-intensify from it. **Nothing has ever changed how a solution of the pool is BUILT.**
    /// Today the pool is 9 dispatching rules + 450 randomised restarts + LS — all of them
    /// list-scheduling, i.e. greedy in TIME. ABZ is greedy in MACHINES and is the one classical JSSP
    /// constructor that is bottleneck-driven, which is what `bricks.md` fact #2 says this scenario
    /// is bounded by (`avg_op_flexibility = 1.0`, `base ∈ [1,200]` ⇒ makespan set by the heaviest
    /// machine).
    ///
    
    /// legacy departures finish at `12 056 · 12 170 · 12 184 · 12 185 · 12 193 · 12 193 · 12 248 ·
    /// 12 286 · 12 310` (nonce 30) and `10 895 … 10 961` (nonce 5): a very tight band, which is what
    /// a HOMOGENEOUS pool produces. The hypothesis is therefore NOT "a better seed gives a better
    
    /// STRUCTURALLY DIFFERENT seed lands in a different basin". The two claims are independent: ABZ
    /// may well construct a WORSE makespan than the 450-restart portfolio and still be the only
    /// point in the run whose disjunctive orientation was not produced by a priority rule.
    ///
    
    /// nor `best_makespan`, nor `best_solution`, calls no `save_solution`, and draws nothing from
    /// `rng` (the construction is fully deterministic). Its result is parked in a private variable
    
    /// tabu phase that "contaminated the solution pool or shifted RNG state") is structurally
    /// excluded. Q is `>=` baseline on EVERY nonce by construction.
    ///
    /// FEASIBILITY. Installing an exact one-machine sequence into a partial orientation can close a
    
    /// here, not hoped away: every installation is certified by `eval_disj`, and a rejected one
    /// falls back on the head-ordered sequence, which cannot cycle because `head[u] < head[v]`
    /// whenever `u` already precedes `v` in the partial graph. No tolerance is introduced or widened.
    pub js_seed_sbp_construct: usize,

    
    
    /// sentinel for "sweep until the fixed point", i.e. until a whole pass installs nothing;
    /// anything else is a hard cap on the number of passes. Every value is additionally clamped by
    /// `num_machines`, so the work is bounded whatever `hp_json` says.
    ///
    
    /// bottleneck by the ITERATED re-optimisation of the machines already sequenced under the arcs
    
    /// construction stops short of the ABZ optimum by construction.
    ///
    
    
    /// 8/8, anti-correlated, and is dead-listed as `brick:js_seed_sbp_randomized_construct`. Inside
    /// the construction class the measurement says EXACTNESS pays and DIVERSITY costs — this knob is
    /// pure exactness: deterministic, no `rng`, no extra departure, no new tolerance.
    ///
    
    /// `top_solutions`, nor `best_makespan`, nor `best_solution`, calls no `save_solution` and draws
    /// nothing from `rng`; its result is consulted by ONE strict `<` at the last statement of
    /// `solve`. The monotone guard `if val >= cur_val { continue }` is kept intact, so every extra
    /// pass can only install a sequence that is STRICTLY better in the subproblem's own metric, and
    
    /// the extra passes cannot introduce a cycle that the first pass could not.
    pub js_seed_sbp_reopt_cycles: usize,

    
    
    
    /// constructive seed with that cap and gives it its own fenced tabu departure; the two
    
    ///
    
    /// and classified the family `saturated`, because it judged each cap as a REPLACEMENT of the
    /// champion cap: the best single arm (`c=3`) moved the median by +399, under the +444 noise
    /// floor. But the median of t48 is the mean of exactly two nonces (n23 rank 15, n4 rank 16), and
    
    
    
    /// closure, `Q_n = max(Q_champion, Q_{c=5})`, gives a median of **91 028.5 = +671.5**, above the
    /// clean KEEP threshold of 90 801. The measurement was already in hand; only the way it was
    /// consumed was wrong. A single value has to be better EVERYWHERE to win; a portfolio only has
    /// to be better SOMEWHERE, and "somewhere" is a two-nonce target.
    ///
    
    /// extra departure to the SECOND-BEST construction, selected on the seed makespan `sbp_mk`.
    
    /// `c=5` — rigorously equal seeds — for `ts_mk` 11 914 vs 12 079. The seed-to-final map is
    /// chaotic, so ranking departures by seed quality ranks noise. Here nothing is selected in
    /// advance: both departures are RUN and the reduction is on the FINAL makespan.
    ///
    
    /// deterministic and draws nothing; `tabu_search_phase` is seeded by an explicit salt argument
    /// and owns its stream. Neither writes `top_solutions`, `best_makespan`, `best_solution`, nor
    /// calls `save_solution`. The second departure lands in the same private `sbp_seed_best` slot as
    /// the first, behind a strict `<`, and that slot is consulted by ONE strict `<` at the last
    /// statement of `solve`. Q is therefore `>=` the champion on every nonce by construction.
    ///
    
    
    /// A fresh salt would be a different, unmeasured trajectory.
    pub js_seed_sbp_dual_cycles: usize,
}

impl EffortConfig {
    pub fn default_effort() -> Self {
        Self { job_shop_iters: 25000, hybrid_flow_shop_iters: 2000, fjsp_medium_iters: 2000, fjsp_high_iters: 2000, js_seed_select_mode: 1, js_ts_fast_eval: 0, js_ts_starts: 10, js_ts_div_period: 0, js_ts_tie_mode: 0, js_ts_sel_key: 0, js_ts_exact_topk: 0, js_ts_traj_decorr: 0, js_ts_extra_draws: 0, js_ts_kick_spread: 0, js_ts_sa_accept: 0, js_mem_cross_scope: 0, js_mem_admit: 0, js_sbp_rounds: 0, js_ts_polish_rescue: 0, js_seed_sbp_construct: 0, js_seed_sbp_reopt_cycles: 1, js_seed_sbp_dual_cycles: 0 }
    }

    
    /// absurd value degrades to the champion run rather than to a VOID. The cap is clamped exactly
    /// like `js_seed_sbp_reopt_cycles`, and the sentinel `0` is NOT reachable as "sweep to the fixed
    /// point" here — that meaning belongs to the first departure only.
    pub fn with_js_seed_sbp_dual_cycles(mut self, v: usize) -> Self {
        self.js_seed_sbp_dual_cycles = if v <= 64 { v } else { 0 };
        self
    }

    
    /// the default; only absurd values are, and they degrade to `1` = the byte-identical champion.
    pub fn with_js_seed_sbp_reopt_cycles(mut self, v: usize) -> Self {
        self.js_seed_sbp_reopt_cycles = if v <= 64 { v } else { 1 };
        self
    }

    
    /// legacy path, so a malformed `hp_json` degrades to the champion run rather than to a VOID.
    pub fn with_js_seed_sbp_construct(mut self, v: usize) -> Self {
        self.js_seed_sbp_construct = if v == 1 || v == 2 { v } else { 0 };
        self
    }

    
    /// `hp_json` degrades to the champion run rather than to a VOID.
    pub fn with_js_ts_polish_rescue(mut self, v: usize) -> Self {
        self.js_ts_polish_rescue = if v == 1 || v == 2 { v } else { 0 };
        self
    }

    
    /// only a guard against a malformed `hp_json` turning the pass into an unbounded cost.
    pub fn with_js_sbp_rounds(mut self, v: usize) -> Self {
        self.js_sbp_rounds = v.min(64);
        self
    }

    
    /// legacy scope, so a malformed `hp_json` degrades to the champion run rather than to a VOID.
    pub fn with_js_mem_admit(mut self, v: usize) -> Self {
        self.js_mem_admit = if v <= 3 { v } else { 0 };
        self
    }
    pub fn with_js_mem_cross_scope(mut self, v: usize) -> Self {
        self.js_mem_cross_scope = if v == 1 || v == 2 { v } else { 0 };
        self
    }

    
    /// makespan is ~12 000, so a few hundred already accepts every candidate, and a malformed
    /// `hp_json` must degrade to a benchable random walk rather than to an overflow.
    pub fn with_js_ts_sa_accept(mut self, v: usize) -> Self {
        self.js_ts_sa_accept = v.min(4000);
        self
    }

    
    /// malformed `hp_json` must degrade to a benchable run rather than to a division by zero or to
    /// a kick on every iteration.
    pub fn with_js_ts_kick_spread(mut self, v: usize) -> Self {
        self.js_ts_kick_spread = v.min(64);
        self
    }

    
    /// gate blow-out.
    pub fn with_js_ts_extra_draws(mut self, v: usize) -> Self {
        self.js_ts_extra_draws = v.min(4);
        self
    }

    
    /// baseline rather than on an undefined salt.
    pub fn with_js_ts_traj_decorr(mut self, v: usize) -> Self {
        self.js_ts_traj_decorr = if v <= 2 { v } else { 0 };
        self
    }

    
    /// malformed `hp_json` degrades to the byte-identical baseline rather than to unbounded work.
    pub fn with_js_ts_exact_topk(mut self, v: usize) -> Self {
        self.js_ts_exact_topk = if v <= 1 { 0 } else { v.min(4) };
        self
    }

    
    /// byte-identical baseline rather than to an undefined branch.
    pub fn with_js_ts_sel_key(mut self, v: usize) -> Self {
        self.js_ts_sel_key = if v <= 2 { v } else { 0 };
        self
    }

    
    /// a malformed `hp_json` degrades to the byte-identical baseline, never to an undefined branch.
    pub fn with_js_ts_tie_mode(mut self, v: usize) -> Self {
        self.js_ts_tie_mode = if v <= 3 { v } else { 0 };
        self
    }

    pub fn with_js_seed_select_mode(mut self, v: usize) -> Self {
        self.js_seed_select_mode = v.min(2);
        self
    }

    pub fn with_js_ts_fast_eval(mut self, v: usize) -> Self {
        self.js_ts_fast_eval = v.min(1);
        self
    }

    
    /// `top_solutions` pool at the injection site, so asking for more is meaningless. The value is
    /// further capped by `top_solutions.len()` at the call site.
    pub fn with_js_ts_starts(mut self, v: usize) -> Self {
        self.js_ts_starts = v.clamp(1, 15);
        self
    }

    
    /// budget-proportional law (byte-identical control); any other value is clamped to
    /// `[60, 300_000]` — 60 is the floor the legacy law already applies (`.max(60)`), 300 000 the
    /// `job_shop_iters` clamp, above which the absolute law can no longer be tighter than the
    /// proportional one and the knob would be a no-op.
    pub fn with_js_ts_div_period(mut self, v: usize) -> Self {
        self.js_ts_div_period = if v == 0 { 0 } else { v.clamp(60, 300_000) };
        self
    }

    
    /// JobShop track, so the track ran AT the clamp and depth was unreachable by hp_json alone.
    
    /// −3,49 % of Q ⇒ the slope points up). The `js_ts_fast_eval` kernel now buys back ~21 % of the
    /// wall at identical work, so the bound is lifted to make that budget spendable. Inert unless
    /// `job_shop_iters` is explicitly raised: the default injection stays 200 000.
    pub fn with_job_shop_iters(mut self, v: usize) -> Self {
        self.job_shop_iters = v.clamp(100, 300000);
        self
    }

    pub fn with_hybrid_flow_shop_iters(mut self, v: usize) -> Self {
        self.hybrid_flow_shop_iters = v.clamp(100, 100000);
        self
    }

    pub fn with_fjsp_medium_iters(mut self, v: usize) -> Self {
        self.fjsp_medium_iters = v.clamp(100, 100000);
        self
    }

    pub fn with_fjsp_high_iters(mut self, v: usize) -> Self {
        self.fjsp_high_iters = v.clamp(100, 100000);
        self
    }
}
}
mod infra_shared {
use anyhow::{anyhow, Result};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use tig_challenges::job_scheduling::*;
use super::types::*;

pub fn run_simple_greedy_baseline(challenge: &Challenge) -> Result<(Solution, u32)> {
    let num_jobs = challenge.num_jobs;
    let mut job_products = Vec::with_capacity(num_jobs);
    for (p, &cnt) in challenge.jobs_per_product.iter().enumerate() {
        for _ in 0..cnt { job_products.push(p); }
    }
    let job_ops_len: Vec<usize> = job_products.iter()
        .map(|&p| challenge.product_processing_times[p].len()).collect();
    let job_total_work: Vec<f64> = job_products.iter().map(|&p| {
        challenge.product_processing_times[p].iter()
            .map(|op| op.values().sum::<u32>() as f64 / op.len().max(1) as f64).sum()
    }).collect();

    let rules = [GreedyRule::MostWork, GreedyRule::MostOps, GreedyRule::LeastFlex, GreedyRule::ShortestProc, GreedyRule::LongestProc];
    let mut best_mk = u32::MAX; let mut best_sol: Option<Solution> = None;
    for rule in rules {
        let (sol, mk) = run_greedy_rule(challenge, &job_products, &job_ops_len, &job_total_work, rule, None)?;
        if mk < best_mk { best_mk = mk; best_sol = Some(sol); }
    }
    let mut rng = SmallRng::from_seed(challenge.seed);
    for _ in 0..10 {
        let seed = rng.gen::<u64>(); let rule = rules[rng.gen_range(0..rules.len())];
        let random_top_k = rng.gen_range(2..=5); let mut local_rng = SmallRng::seed_from_u64(seed);
        let (sol, mk) = run_greedy_rule(challenge, &job_products, &job_ops_len, &job_total_work, rule, Some((random_top_k, &mut local_rng)))?;
        if mk < best_mk { best_mk = mk; best_sol = Some(sol); }
    }
    Ok((best_sol.ok_or_else(|| anyhow!("No greedy solution"))?, best_mk))
}

pub fn run_simple_greedy_baseline_weighted(challenge: &Challenge) -> Result<(Solution, u32)> {
    let num_jobs = challenge.num_jobs;
    let mut job_products = Vec::with_capacity(num_jobs);
    for (p, &cnt) in challenge.jobs_per_product.iter().enumerate() {
        for _ in 0..cnt { job_products.push(p); }
    }
    let job_ops_len: Vec<usize> = job_products.iter()
        .map(|&p| challenge.product_processing_times[p].len()).collect();
    let job_total_work: Vec<f64> = job_products.iter().map(|&p| {
        challenge.product_processing_times[p].iter()
            .map(|op| op.values().sum::<u32>() as f64 / op.len().max(1) as f64).sum()
    }).collect();

    let mut best_mk = u32::MAX; let mut best_sol: Option<Solution> = None;
    let base_weights = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, -1.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    for w in base_weights {
        let (sol, mk) = run_greedy_weighted(challenge, &job_products, &job_ops_len, &job_total_work, w, None)?;
        if mk < best_mk { best_mk = mk; best_sol = Some(sol); }
    }
    
    let mut rng = SmallRng::from_seed(challenge.seed);
    for _ in 0..10 {
        let seed = rng.gen::<u64>();
        let w = [
            rng.gen_range(0.0..1.0),
            rng.gen_range(0.0..1.0),
            rng.gen_range(-1.0..0.5),
            rng.gen_range(-1.0..1.0),
        ];
        let random_top_k = rng.gen_range(2..=5); let mut local_rng = SmallRng::seed_from_u64(seed);
        let (sol, mk) = run_greedy_weighted(challenge, &job_products, &job_ops_len, &job_total_work, w, Some((random_top_k, &mut local_rng)))?;
        if mk < best_mk { best_mk = mk; best_sol = Some(sol); }
    }
    Ok((best_sol.ok_or_else(|| anyhow!("No greedy solution"))?, best_mk))
}

pub fn run_greedy_weighted(
    challenge: &Challenge, job_products: &[usize], job_ops_len: &[usize], job_total_work: &[f64],
    weights: [f64; 4], mut random_top_k: Option<(usize, &mut SmallRng)>,
) -> Result<(Solution, u32)> {
    #[derive(Clone, Copy)]
    struct GCandidate { job: usize, priority: f64, end: u32, pt: u32, flex: usize }

    let num_jobs = challenge.num_jobs; let num_machines = challenge.num_machines;
    let mut job_next_op = vec![0usize; num_jobs]; let mut job_ready = vec![0u32; num_jobs]; let mut machine_avail = vec![0u32; num_machines];
    let mut job_schedule: Vec<Vec<(usize, u32)>> = job_ops_len.iter().map(|&len| Vec::with_capacity(len)).collect();
    let mut job_work_left = job_total_work.to_vec();
    let mut remaining = job_ops_len.iter().sum::<usize>(); let mut time = 0u32; let eps = 1e-9;
    let mut available_machines: Vec<usize> = Vec::with_capacity(num_machines);
    let mut candidates: Vec<GCandidate> = Vec::new();

    while remaining > 0 {
        available_machines.clear();
        for m in 0..num_machines {
            if machine_avail[m] <= time { available_machines.push(m); }
        }
        available_machines.sort_unstable();
        if let Some((_, ref mut rng)) = random_top_k { use rand::seq::SliceRandom; available_machines.shuffle(*rng); }

        if let Some((top_k, ref mut rng)) = random_top_k {
            for &m in &available_machines {
                candidates.clear();
                for j in 0..num_jobs {
                    if job_next_op[j] >= job_ops_len[j] || job_ready[j] > time { continue; }
                    let product = job_products[j]; let op_idx = job_next_op[j];
                    let op_times = &challenge.product_processing_times[product][op_idx];
                    let pt = match op_times.get(&m) { Some(&v) => v, None => continue };
                    let earliest = op_times.iter().map(|(&mm, &ppt)| time.max(machine_avail[mm]) + ppt).min().unwrap_or(u32::MAX);
                    let this_end = time.max(machine_avail[m]) + pt;
                    if this_end != earliest { continue; }
                    let flex = op_times.len(); let ops_left = job_ops_len[j] - job_next_op[j];
                    
                    let priority = weights[0] * job_work_left[j] 
                                 + weights[1] * (ops_left as f64) 
                                 + weights[2] * (flex as f64) 
                                 + weights[3] * (pt as f64);
                    
                    candidates.push(GCandidate { job: j, priority, end: this_end, pt, flex });
                }
                if candidates.is_empty() { continue; }

                candidates.sort_by(|a, b| { if (b.priority - a.priority).abs() > eps { b.priority.partial_cmp(&a.priority).unwrap() } else if a.end != b.end { a.end.cmp(&b.end) } else if a.pt != b.pt { a.pt.cmp(&b.pt) } else if a.flex != b.flex { a.flex.cmp(&b.flex) } else { a.job.cmp(&b.job) } });
                let top = candidates.len().min(top_k);
                let chosen = candidates[rng.gen_range(0..top)];
                let best_job = chosen.job;
                let product = job_products[best_job]; let op_idx = job_next_op[best_job];
                let op_times = &challenge.product_processing_times[product][op_idx];
                let avg_pt = op_times.values().sum::<u32>() as f64 / op_times.len().max(1) as f64;
                let st = time.max(machine_avail[m]); let end = st + chosen.pt;
                job_schedule[best_job].push((m, st)); job_next_op[best_job] += 1; job_ready[best_job] = end; machine_avail[m] = end;
                job_work_left[best_job] -= avg_pt; if job_work_left[best_job] < 0.0 { job_work_left[best_job] = 0.0; } remaining -= 1;
            }
        } else {
            for &m in &available_machines {
                let mut best: Option<GCandidate> = None;
                for j in 0..num_jobs {
                    if job_next_op[j] >= job_ops_len[j] || job_ready[j] > time { continue; }
                    let product = job_products[j]; let op_idx = job_next_op[j];
                    let op_times = &challenge.product_processing_times[product][op_idx];
                    let pt = match op_times.get(&m) { Some(&v) => v, None => continue };
                    let earliest = op_times.iter().map(|(&mm, &ppt)| time.max(machine_avail[mm]) + ppt).min().unwrap_or(u32::MAX);
                    let this_end = time.max(machine_avail[m]) + pt;
                    if this_end != earliest { continue; }
                    let flex = op_times.len(); let ops_left = job_ops_len[j] - job_next_op[j];
                    
                    let priority = weights[0] * job_work_left[j] 
                                 + weights[1] * (ops_left as f64) 
                                 + weights[2] * (flex as f64) 
                                 + weights[3] * (pt as f64);
                    
                    let cand = GCandidate {
                        job: j,
                        priority,
                        end: this_end,
                        pt,
                        flex,
                    };
                    let better = if let Some(b) = best {
                        if (cand.priority - b.priority).abs() > eps { cand.priority > b.priority } else if cand.end != b.end { cand.end < b.end } else if cand.pt != b.pt { cand.pt < b.pt } else if cand.flex != b.flex { cand.flex < b.flex } else { cand.job < b.job }
                    } else { true };
                    if better { best = Some(cand); }
                }
                let Some(best) = best else { continue };
                let best_job = best.job;
                let product = job_products[best_job]; let op_idx = job_next_op[best_job];
                let op_times = &challenge.product_processing_times[product][op_idx];
                let avg_pt = op_times.values().sum::<u32>() as f64 / op_times.len().max(1) as f64;
                let st = time.max(machine_avail[m]); let end = st + best.pt;
                job_schedule[best_job].push((m, st)); job_next_op[best_job] += 1; job_ready[best_job] = end; machine_avail[m] = end;
                job_work_left[best_job] -= avg_pt; if job_work_left[best_job] < 0.0 { job_work_left[best_job] = 0.0; } remaining -= 1;
            }
        }

        if remaining == 0 { break; }
        let mut next = u32::MAX;
        for &t in &machine_avail { if t > time && t < next { next = t; } }
        for j in 0..num_jobs { if job_next_op[j] < job_ops_len[j] && job_ready[j] > time && job_ready[j] < next { next = job_ready[j]; } }
        if next == u32::MAX { return Err(anyhow!("Greedy baseline stuck")); }
        time = next;
    }
    let mk = job_ready.iter().copied().max().unwrap_or(0);
    Ok((Solution { job_schedule }, mk))
}

pub fn run_greedy_rule(
    challenge: &Challenge, job_products: &[usize], job_ops_len: &[usize], job_total_work: &[f64],
    rule: GreedyRule, mut random_top_k: Option<(usize, &mut SmallRng)>,
) -> Result<(Solution, u32)> {
    #[derive(Clone, Copy)]
    struct GCandidate { job: usize, priority: f64, end: u32, pt: u32, flex: usize }

    let num_jobs = challenge.num_jobs; let num_machines = challenge.num_machines;
    let mut job_next_op = vec![0usize; num_jobs]; let mut job_ready = vec![0u32; num_jobs]; let mut machine_avail = vec![0u32; num_machines];
    let mut job_schedule: Vec<Vec<(usize, u32)>> = job_ops_len.iter().map(|&len| Vec::with_capacity(len)).collect();
    let mut job_work_left = job_total_work.to_vec();
    let mut remaining = job_ops_len.iter().sum::<usize>(); let mut time = 0u32; let eps = 1e-9;
    let mut available_machines: Vec<usize> = Vec::with_capacity(num_machines);
    let mut candidates: Vec<GCandidate> = Vec::new();

    while remaining > 0 {
        available_machines.clear();
        for m in 0..num_machines {
            if machine_avail[m] <= time { available_machines.push(m); }
        }
        available_machines.sort_unstable();
        if let Some((_, ref mut rng)) = random_top_k { use rand::seq::SliceRandom; available_machines.shuffle(*rng); }

        if let Some((top_k, ref mut rng)) = random_top_k {
            for &m in &available_machines {
                candidates.clear();
                for j in 0..num_jobs {
                    if job_next_op[j] >= job_ops_len[j] || job_ready[j] > time { continue; }
                    let product = job_products[j]; let op_idx = job_next_op[j];
                    let op_times = &challenge.product_processing_times[product][op_idx];
                    let pt = match op_times.get(&m) { Some(&v) => v, None => continue };
                    let earliest = op_times.iter().map(|(&mm, &ppt)| time.max(machine_avail[mm]) + ppt).min().unwrap_or(u32::MAX);
                    let this_end = time.max(machine_avail[m]) + pt;
                    if this_end != earliest { continue; }
                    let flex = op_times.len(); let ops_left = job_ops_len[j] - job_next_op[j];
                    let priority = match rule {
                        GreedyRule::MostWork => job_work_left[j], GreedyRule::MostOps => ops_left as f64,
                        GreedyRule::LeastFlex => -(flex as f64), GreedyRule::ShortestProc => -(pt as f64),
                        GreedyRule::LongestProc => pt as f64,
                    };
                    candidates.push(GCandidate { job: j, priority, end: this_end, pt, flex });
                }
                if candidates.is_empty() { continue; }

                candidates.sort_by(|a, b| { if (b.priority - a.priority).abs() > eps { b.priority.partial_cmp(&a.priority).unwrap() } else if a.end != b.end { a.end.cmp(&b.end) } else if a.pt != b.pt { a.pt.cmp(&b.pt) } else if a.flex != b.flex { a.flex.cmp(&b.flex) } else { a.job.cmp(&b.job) } });
                let top = candidates.len().min(top_k);
                let chosen = candidates[rng.gen_range(0..top)];
                let best_job = chosen.job;
                let product = job_products[best_job]; let op_idx = job_next_op[best_job];
                let op_times = &challenge.product_processing_times[product][op_idx];
                let avg_pt = op_times.values().sum::<u32>() as f64 / op_times.len().max(1) as f64;
                let st = time.max(machine_avail[m]); let end = st + chosen.pt;
                job_schedule[best_job].push((m, st)); job_next_op[best_job] += 1; job_ready[best_job] = end; machine_avail[m] = end;
                job_work_left[best_job] -= avg_pt; if job_work_left[best_job] < 0.0 { job_work_left[best_job] = 0.0; } remaining -= 1;
            }
        } else {
            for &m in &available_machines {
                let mut best: Option<GCandidate> = None;
                for j in 0..num_jobs {
                    if job_next_op[j] >= job_ops_len[j] || job_ready[j] > time { continue; }
                    let product = job_products[j]; let op_idx = job_next_op[j];
                    let op_times = &challenge.product_processing_times[product][op_idx];
                    let pt = match op_times.get(&m) { Some(&v) => v, None => continue };
                    let earliest = op_times.iter().map(|(&mm, &ppt)| time.max(machine_avail[mm]) + ppt).min().unwrap_or(u32::MAX);
                    let this_end = time.max(machine_avail[m]) + pt;
                    if this_end != earliest { continue; }
                    let flex = op_times.len(); let ops_left = job_ops_len[j] - job_next_op[j];
                    let cand = GCandidate {
                        job: j,
                        priority: match rule {
                            GreedyRule::MostWork => job_work_left[j], GreedyRule::MostOps => ops_left as f64,
                            GreedyRule::LeastFlex => -(flex as f64), GreedyRule::ShortestProc => -(pt as f64),
                            GreedyRule::LongestProc => pt as f64,
                        },
                        end: this_end,
                        pt,
                        flex,
                    };
                    let better = if let Some(b) = best {
                        if (cand.priority - b.priority).abs() > eps { cand.priority > b.priority } else if cand.end != b.end { cand.end < b.end } else if cand.pt != b.pt { cand.pt < b.pt } else if cand.flex != b.flex { cand.flex < b.flex } else { cand.job < b.job }
                    } else { true };
                    if better { best = Some(cand); }
                }
                let Some(best) = best else { continue };
                let best_job = best.job;
                let product = job_products[best_job]; let op_idx = job_next_op[best_job];
                let op_times = &challenge.product_processing_times[product][op_idx];
                let avg_pt = op_times.values().sum::<u32>() as f64 / op_times.len().max(1) as f64;
                let st = time.max(machine_avail[m]); let end = st + best.pt;
                job_schedule[best_job].push((m, st)); job_next_op[best_job] += 1; job_ready[best_job] = end; machine_avail[m] = end;
                job_work_left[best_job] -= avg_pt; if job_work_left[best_job] < 0.0 { job_work_left[best_job] = 0.0; } remaining -= 1;
            }
        }

        if remaining == 0 { break; }
        let mut next = u32::MAX;
        for &t in &machine_avail { if t > time && t < next { next = t; } }
        for j in 0..num_jobs { if job_next_op[j] < job_ops_len[j] && job_ready[j] > time && job_ready[j] < next { next = job_ready[j]; } }
        if next == u32::MAX { return Err(anyhow!("Greedy baseline stuck")); }
        time = next;
    }
    let mk = job_ready.iter().copied().max().unwrap_or(0);
    Ok((Solution { job_schedule }, mk))
}

pub fn build_disj_from_solution(pre: &Pre, challenge: &Challenge, sol: &Solution) -> Result<DisjSchedule> {
    let num_jobs = challenge.num_jobs; let num_machines = challenge.num_machines;
    let mut job_offsets = vec![0usize; num_jobs + 1];
    for j in 0..num_jobs { job_offsets[j + 1] = job_offsets[j] + pre.job_ops_len[j]; }
    let n = job_offsets[num_jobs];
    if n == 0 { return Err(anyhow!("No operations")); }
    let mut node_machine = vec![0usize; n]; let mut node_pt = vec![0u32; n]; let mut node_job = vec![0usize; n]; let mut node_op = vec![0usize; n];
    let mut per_machine: Vec<Vec<(u32, usize)>> = vec![Vec::new(); num_machines];
    for job in 0..num_jobs {
        let expected = pre.job_ops_len[job];
        if sol.job_schedule[job].len() != expected { return Err(anyhow!("Invalid solution: job {} ops len mismatch", job)); }
        let product = pre.job_products[job];
        for op_idx in 0..expected {
            let id = job_offsets[job] + op_idx; let (m, st) = sol.job_schedule[job][op_idx];
            let op = &pre.product_ops[product][op_idx];
            let pt = pt_from_op(op, m).ok_or_else(|| anyhow!("Invalid solution: pt missing"))?;
            if m >= num_machines { return Err(anyhow!("Invalid solution: machine out of range")); }
            node_machine[id] = m; node_pt[id] = pt; node_job[id] = job; node_op[id] = op_idx;
            per_machine[m].push((st, id));
        }
    }
    let mut machine_seq: Vec<Vec<usize>> = Vec::with_capacity(num_machines);
    for m in 0..num_machines {
        per_machine[m].sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        machine_seq.push(per_machine[m].iter().map(|&(_, id)| id).collect());
    }
    let mut job_succ = vec![NONE_USIZE; n]; let mut indeg_job = vec![0u16; n];
    for job in 0..num_jobs {
        let len = pre.job_ops_len[job]; let base = job_offsets[job];
        for k in 0..len { let id = base + k; if k + 1 < len { job_succ[id] = id + 1; indeg_job[id + 1] = indeg_job[id + 1].saturating_add(1); } }
    }
    Ok(DisjSchedule { n, num_jobs, num_machines, job_offsets, job_succ, indeg_job, node_machine, node_pt, node_job, node_op, machine_seq })
}

#[inline]
pub fn pt_from_op(op: &OpInfo, machine: usize) -> Option<u32> {
    for &(m, pt) in &op.machines { if m == machine { return Some(pt); } }
    None
}

pub fn eval_disj(ds: &DisjSchedule, buf: &mut EvalBuf) -> Option<(u32, usize)> {
    let n = ds.n;
    buf.indeg.clone_from_slice(&ds.indeg_job);
    buf.start.fill(0);
    buf.best_pred.fill(NONE_USIZE);
    buf.stack.clear();
    for seq in &ds.machine_seq {
        if seq.is_empty() { continue; }
        let mut prev = seq[0];
        for &v in &seq[1..] {
            buf.machine_succ[prev] = v;
            buf.indeg[v] = buf.indeg[v].saturating_add(1);
            prev = v;
        }
        buf.machine_succ[prev] = NONE_USIZE;
    }
    for i in 0..n {
        if buf.indeg[i] == 0 { buf.stack.push(i); }
    }
    let mut processed = 0usize;
    let mut mk = 0u32;
    let mut mk_node = 0usize;
    while let Some(u) = buf.stack.pop() {
        processed += 1;
        let end_u = buf.start[u].saturating_add(ds.node_pt[u]);
        if end_u > mk { mk = end_u; mk_node = u; }
        let js = ds.job_succ[u];
        if js != NONE_USIZE {
            if buf.start[js] < end_u {
                buf.start[js] = end_u;
                buf.best_pred[js] = u;
            }
            buf.indeg[js] = buf.indeg[js].saturating_sub(1);
            if buf.indeg[js] == 0 { buf.stack.push(js); }
        }
        let ms = buf.machine_succ[u];
        if ms != NONE_USIZE {
            if buf.start[ms] < end_u {
                buf.start[ms] = end_u;
                buf.best_pred[ms] = u;
            }
            buf.indeg[ms] = buf.indeg[ms].saturating_sub(1);
            if buf.indeg[ms] == 0 { buf.stack.push(ms); }
        }
    }
    if processed != n { return None; }
    Some((mk, mk_node))
}

pub fn disj_to_solution(pre: &Pre, ds: &DisjSchedule, start: &[u32]) -> Result<Solution> {
    let num_jobs = ds.num_jobs;
    let mut job_schedule: Vec<Vec<(usize, u32)>> = Vec::with_capacity(num_jobs);
    for j in 0..num_jobs {
        let len = pre.job_ops_len[j]; let mut v = Vec::with_capacity(len); let base = ds.job_offsets[j];
        for k in 0..len { let id = base + k; v.push((ds.node_machine[id], start[id])); }
        job_schedule.push(v);
    }
    Ok(Solution { job_schedule })
}

pub fn critical_block_move_local_search_ex(
    pre: &Pre, challenge: &Challenge, base_sol: &Solution,
    max_iters: usize, top_cands: usize, perturb_cycles: usize,
) -> Result<Option<(Solution, u32)>> {
    let mut ds = build_disj_from_solution(pre, challenge, base_sol)?;
    let mut buf = EvalBuf::new(ds.n); let mut crit = vec![false; ds.n];
    let mut cur_eval = match eval_disj(&ds, &mut buf) { Some(x) => x, None => return Ok(None) };
    let initial_mk = cur_eval.0;
    descent_phase(&mut ds, &mut buf, &mut crit, pre, &mut cur_eval, max_iters, top_cands);
    let Some((mk_after, _)) = eval_disj(&ds, &mut buf) else { return Ok(None) };
    let mut global_best_mk = mk_after; let mut global_best_ds = ds.clone();
    let mut sol_hash: u64 = 0;
    for m in 0..ds.num_machines.min(8) {
        if !ds.machine_seq[m].is_empty() {
            let first_node = ds.machine_seq[m][0];
            sol_hash ^= (first_node as u64).wrapping_mul(0xD2B54A6B68A5);
            sol_hash = sol_hash.rotate_left(7);
        }
    }
    let mut pseed: u64 = (challenge.seed[0] as u64).wrapping_mul(0x9E3779B97F4A7C15) ^ (initial_mk as u64).wrapping_shl(16) ^ (ds.n as u64) ^ sol_hash;
    for _cycle in 0..perturb_cycles {
        ds = global_best_ds.clone();
        let Some((_, mk_node)) = eval_disj(&ds, &mut buf) else { break };
        crit.fill(false); let mut u = mk_node; while u != NONE_USIZE { crit[u] = true; u = buf.best_pred[u]; }
        let mut blocks: Vec<(usize, usize, usize)> = Vec::new();
        for m in 0..ds.num_machines {
            let seq = &ds.machine_seq[m]; if seq.len() <= 1 { continue; }
            let mut i = 0usize;
            while i < seq.len() {
                if !crit[seq[i]] { i += 1; continue; }
                let bstart = i; let mut bend = i;
                while bend + 1 < seq.len() { let x = seq[bend]; let y = seq[bend+1]; if !crit[y] { break; } if buf.start[y] != buf.start[x].saturating_add(ds.node_pt[x]) { break; } bend += 1; }
                if bend > bstart { blocks.push((m, bstart, bend)); } i = bend + 1;
            }
        }
        if blocks.is_empty() { break; }
        for _ in 0..2 {
            pseed ^= pseed.wrapping_shl(13); pseed ^= pseed.wrapping_shr(7); pseed ^= pseed.wrapping_shl(17);
            let bidx = (pseed as usize) % blocks.len(); let (m, bstart, bend) = blocks[bidx];
            let block_len = bend - bstart; if block_len == 0 { continue; }
            pseed ^= pseed.wrapping_shl(13); pseed ^= pseed.wrapping_shr(7); pseed ^= pseed.wrapping_shl(17);
            let swap_pos = bstart + ((pseed as usize) % block_len);
            if swap_pos + 1 < ds.machine_seq[m].len() { ds.machine_seq[m].swap(swap_pos, swap_pos + 1); }
        }
        match eval_disj(&ds, &mut buf) { Some(x) => cur_eval = x, None => continue }
        descent_phase(&mut ds, &mut buf, &mut crit, pre, &mut cur_eval, max_iters, top_cands);
        if let Some((mk_now, _)) = eval_disj(&ds, &mut buf) { if mk_now < global_best_mk { global_best_mk = mk_now; global_best_ds = ds.clone(); } }
    }
    if global_best_mk >= initial_mk { return Ok(None); }
    ds = global_best_ds;
    let Some((mk_final, _)) = eval_disj(&ds, &mut buf) else { return Ok(None) };
    let sol = disj_to_solution(pre, &ds, &buf.start)?;
    Ok(Some((sol, mk_final)))
}

pub fn critical_block_move_local_search_ex_fjsp(
    pre: &Pre, challenge: &Challenge, base_sol: &Solution,
    max_iters: usize, top_cands: usize, perturb_cycles: usize,
) -> Result<Option<(Solution, u32)>> {
    let mut ds = build_disj_from_solution(pre, challenge, base_sol)?;
    let mut buf = EvalBuf::new(ds.n); let mut crit = vec![false; ds.n];
    let mut cur_eval = match eval_disj(&ds, &mut buf) { Some(x) => x, None => return Ok(None) };
    let initial_mk = cur_eval.0;
    descent_phase_fjsp(&mut ds, &mut buf, &mut crit, pre, &mut cur_eval, max_iters, top_cands);
    let Some((mk_after, _)) = eval_disj(&ds, &mut buf) else { return Ok(None) };
    let mut global_best_mk = mk_after; let mut global_best_ds = ds.clone();
    let mut sol_hash: u64 = 0;
    for m in 0..ds.num_machines.min(8) {
        if !ds.machine_seq[m].is_empty() {
            let first_node = ds.machine_seq[m][0];
            sol_hash ^= (first_node as u64).wrapping_mul(0xD2B54A6B68A5);
            sol_hash = sol_hash.rotate_left(7);
        }
    }
    let mut pseed: u64 = (challenge.seed[0] as u64).wrapping_mul(0x9E3779B97F4A7C15) ^ (initial_mk as u64).wrapping_shl(16) ^ (ds.n as u64) ^ sol_hash;
    for _cycle in 0..perturb_cycles {
        ds = global_best_ds.clone();
        let Some((_, mk_node)) = eval_disj(&ds, &mut buf) else { break };
        crit.fill(false); let mut u = mk_node; while u != NONE_USIZE { crit[u] = true; u = buf.best_pred[u]; }
        let mut blocks: Vec<(usize, usize, usize)> = Vec::new();
        for m in 0..ds.num_machines {
            let seq = &ds.machine_seq[m]; if seq.len() <= 1 { continue; }
            let mut i = 0usize;
            while i < seq.len() {
                if !crit[seq[i]] { i += 1; continue; }
                let bstart = i; let mut bend = i;
                while bend + 1 < seq.len() { let x = seq[bend]; let y = seq[bend+1]; if !crit[y] { break; } if buf.start[y] != buf.start[x].saturating_add(ds.node_pt[x]) { break; } bend += 1; }
                if bend > bstart { blocks.push((m, bstart, bend)); } i = bend + 1;
            }
        }
        if blocks.is_empty() { break; }
        let mut extracted = Vec::with_capacity(3);
        let num_extract = 3.min(blocks.len() * 2);
        for _ in 0..num_extract {
            pseed ^= pseed.wrapping_shl(13); pseed ^= pseed.wrapping_shr(7); pseed ^= pseed.wrapping_shl(17);
            let bidx = (pseed as usize) % blocks.len(); let (m, bstart, bend) = blocks[bidx];
            let block_len = bend - bstart; if block_len == 0 { continue; }
            pseed ^= pseed.wrapping_shl(13); pseed ^= pseed.wrapping_shr(7); pseed ^= pseed.wrapping_shl(17);
            let ext_pos = bstart + ((pseed as usize) % block_len);
            if ext_pos < ds.machine_seq[m].len() {
                extracted.push((m, ds.machine_seq[m].remove(ext_pos)));
            }
        }
        for (m, node) in extracted {
            let desired = buf.start[node];
            let mut pos = find_insert_pos_by_start(&ds.machine_seq[m], &buf.start, desired);
            pseed ^= pseed.wrapping_shl(13); pseed ^= pseed.wrapping_shr(7); pseed ^= pseed.wrapping_shl(17);
            let offset = (pseed as usize) % 3;
            if (pseed >> 3) & 1 == 0 { pos = pos.saturating_add(offset); } else { pos = pos.saturating_sub(offset); }
            let m_len = ds.machine_seq[m].len();
            ds.machine_seq[m].insert(pos.min(m_len), node);
        }
        match eval_disj(&ds, &mut buf) { Some(x) => cur_eval = x, None => continue }
        descent_phase_fjsp(&mut ds, &mut buf, &mut crit, pre, &mut cur_eval, max_iters, top_cands);
        if let Some((mk_now, _)) = eval_disj(&ds, &mut buf) { if mk_now < global_best_mk { global_best_mk = mk_now; global_best_ds = ds.clone(); } }
    }
    if global_best_mk >= initial_mk { return Ok(None); }
    ds = global_best_ds;
    let Some((mk_final, _)) = eval_disj(&ds, &mut buf) else { return Ok(None) };
    let sol = disj_to_solution(pre, &ds, &buf.start)?;
    Ok(Some((sol, mk_final)))
}

fn descent_phase(
    ds: &mut DisjSchedule, buf: &mut EvalBuf, crit: &mut Vec<bool>, pre: &Pre,
    cur_eval: &mut (u32, usize), max_iters: usize, top_cands: usize,
) -> bool {
    let mut cur_mk = cur_eval.0; let mut improved = false;
    for _iter in 0..max_iters {
        crit.fill(false); let mut u = cur_eval.1; while u != NONE_USIZE { crit[u] = true; u = buf.best_pred[u]; }
        let mut cands: Vec<MoveCand> = Vec::with_capacity(top_cands.min(64));
        for m in 0..ds.num_machines {
            let seq = &ds.machine_seq[m]; if seq.len() <= 1 { continue; }
            let mut i = 0usize;
            while i < seq.len() {
                let a = seq[i]; if !crit[a] { i += 1; continue; }
                let bstart = i; let mut bend = i;
                while bend + 1 < seq.len() { let x = seq[bend]; let y = seq[bend+1]; if !crit[y] { break; } if buf.start[y] != buf.start[x].saturating_add(ds.node_pt[x]) { break; } bend += 1; }
                if bend > bstart {
                    let max_shift = bend - bstart;
                    let mut shifts: [usize; 3] = [1, 2, max_shift];
                    for sh in shifts.iter_mut() { if *sh > max_shift { *sh = 0; } }
                    for &sh in &shifts {
                        if sh == 0 { continue; }
                        { let from = bstart; let to_after = bstart + sh; if from < seq.len() && to_after <= seq.len() { let tgt_idx = (bstart+sh).min(seq.len()-1); push_top_k_move(&mut cands, MoveCand { kind: 0, m_from: m, from, m_to: m, to: to_after, new_pt: 0, score: buf.start[seq[tgt_idx]] }, top_cands); } }
                        { let from = bend; let to_after = bend - sh; push_top_k_move(&mut cands, MoveCand { kind: 0, m_from: m, from, m_to: m, to: to_after, new_pt: 0, score: buf.start[seq[bend]] }, top_cands); }
                    }
                    if bstart > 0 { push_top_k_move(&mut cands, MoveCand { kind: 2, m_from: m, from: bstart-1, m_to: m, to: 0, new_pt: 0, score: buf.start[seq[bstart]] }, top_cands); }
                    if bend + 1 < seq.len() { push_top_k_move(&mut cands, MoveCand { kind: 2, m_from: m, from: bend, m_to: m, to: 0, new_pt: 0, score: buf.start[seq[bend]] }, top_cands); }
                    if bstart + 1 <= bend {
                        push_top_k_move(&mut cands, MoveCand { kind: 2, m_from: m, from: bstart, m_to: m, to: 0, new_pt: 0, score: buf.start[seq[bstart+1]] }, top_cands);
                        if bend >= 1 && bend - 1 >= bstart { push_top_k_move(&mut cands, MoveCand { kind: 2, m_from: m, from: bend-1, m_to: m, to: 0, new_pt: 0, score: buf.start[seq[bend]] }, top_cands); }
                    }
                    for &idx in &[bstart, bend] {
                        if idx >= seq.len() { continue; }
                        let node = seq[idx]; if !crit[node] { continue; }
                        let job = ds.node_job[node]; let op_idx = ds.node_op[node]; let product = pre.job_products[job];
                        let op = &pre.product_ops[product][op_idx];
                        if op.flex < 2 || op.machines.len() < 2 { continue; }
                        let old_m = ds.node_machine[node]; let old_pt = ds.node_pt[node];
                        let w_from = pre.machine_weight[old_m].max(1e-9);
                        let best2 = best_two_by_pt(op);
                        for &(m_to, new_pt) in &best2 {
                            if m_to == NONE_USIZE || m_to >= ds.num_machines || m_to == old_m || new_pt >= INF { continue; }
                            let w_to = pre.machine_weight[m_to].max(1e-9);
                            if !(new_pt + 1 < old_pt || w_to < w_from * 0.90) { continue; }
                            let desired = buf.start[node];
                            let pos0 = find_insert_pos_by_start(&ds.machine_seq[m_to][..], &buf.start, desired);
                            for pos in [pos0, pos0.saturating_add(1)] {
                                if pos > ds.machine_seq[m_to].len() { continue; }
                                let diffw = ((w_from - w_to).max(0.0) * pre.avg_op_min).max(0.0) as u32;
                                let difpt = old_pt.saturating_sub(new_pt);
                                let score = desired.saturating_add(old_pt).saturating_add(diffw).saturating_add(difpt.saturating_mul(2));
                                push_top_k_move(&mut cands, MoveCand { kind: 1, m_from: old_m, from: idx, m_to, to: pos, new_pt, score }, top_cands);
                            }
                        }
                    }
                }
                i = bend + 1;
            }
        }
        if cands.is_empty() { break; }
        let mut best_cand: Option<MoveCand> = None; let mut best_mk = cur_mk;
        for cand in &cands {
            if cand.kind == 0 {
                let m = cand.m_from; if m >= ds.num_machines || cand.from >= ds.machine_seq[m].len() { continue; }
                let new_idx = apply_insert(&mut ds.machine_seq[m], cand.from, cand.to);
                if let Some((mk2, _)) = eval_disj(ds, buf) { if mk2 < best_mk { best_mk = mk2; best_cand = Some(*cand); } }
                let _ = apply_insert(&mut ds.machine_seq[m], new_idx, cand.from);
            } else if cand.kind == 2 {
                let m = cand.m_from; if m >= ds.num_machines || cand.from + 1 >= ds.machine_seq[m].len() { continue; }
                if !apply_swap(&mut ds.machine_seq[m], cand.from) { continue; }
                if let Some((mk2, _)) = eval_disj(ds, buf) { if mk2 < best_mk { best_mk = mk2; best_cand = Some(*cand); } }
                let _ = apply_swap(&mut ds.machine_seq[m], cand.from);
            } else {
                let m_from = cand.m_from; let m_to = cand.m_to;
                if m_from >= ds.num_machines || m_to >= ds.num_machines || cand.from >= ds.machine_seq[m_from].len() { continue; }
                let node = ds.machine_seq[m_from][cand.from]; if ds.node_machine[node] != m_from { continue; }
                if let Some((node2, old_pt, ins_idx)) = apply_reroute(ds, m_from, cand.from, m_to, cand.to, cand.new_pt) {
                    if let Some((mk2, _)) = eval_disj(ds, buf) { if mk2 < best_mk { best_mk = mk2; best_cand = Some(*cand); } }
                    let _ = undo_reroute(ds, m_from, cand.from, m_to, ins_idx, node2, old_pt);
                }
            }
        }
        let Some(bc) = best_cand else { break };
        let mut accepted = false;
        if bc.kind == 0 {
            let m = bc.m_from; let new_idx = apply_insert(&mut ds.machine_seq[m], bc.from, bc.to);
            if let Some(ne) = eval_disj(ds, buf) { if ne.0 < cur_mk { *cur_eval = ne; cur_mk = ne.0; improved = true; accepted = true; } else { let _ = apply_insert(&mut ds.machine_seq[m], new_idx, bc.from); } }
            else { let _ = apply_insert(&mut ds.machine_seq[m], new_idx, bc.from); }
        } else if bc.kind == 2 {
            let m = bc.m_from;
            if m < ds.num_machines && bc.from + 1 < ds.machine_seq[m].len() {
                if apply_swap(&mut ds.machine_seq[m], bc.from) {
                    if let Some(ne) = eval_disj(ds, buf) { if ne.0 < cur_mk { *cur_eval = ne; cur_mk = ne.0; improved = true; accepted = true; } else { let _ = apply_swap(&mut ds.machine_seq[m], bc.from); } }
                    else { let _ = apply_swap(&mut ds.machine_seq[m], bc.from); }
                }
            }
        } else {
            if let Some((node2, old_pt, ins_idx)) = apply_reroute(ds, bc.m_from, bc.from, bc.m_to, bc.to, bc.new_pt) {
                if let Some(ne) = eval_disj(ds, buf) { if ne.0 < cur_mk { *cur_eval = ne; cur_mk = ne.0; improved = true; accepted = true; } else { let _ = undo_reroute(ds, bc.m_from, bc.from, bc.m_to, ins_idx, node2, old_pt); } }
                else { let _ = undo_reroute(ds, bc.m_from, bc.from, bc.m_to, ins_idx, node2, old_pt); }
            }
        }
        if !accepted { break; }
    }
    improved
}

fn descent_phase_fjsp(
    ds: &mut DisjSchedule, buf: &mut EvalBuf, crit: &mut Vec<bool>, pre: &Pre,
    cur_eval: &mut (u32, usize), max_iters: usize, top_cands: usize,
) -> bool {
    let mut cur_mk = cur_eval.0; let mut improved = false;
    for _iter in 0..max_iters {
        crit.fill(false); let mut u = cur_eval.1; while u != NONE_USIZE { crit[u] = true; u = buf.best_pred[u]; }
        let mut m_load = vec![0u32; ds.num_machines];
        for m in 0..ds.num_machines {
            if let Some(&last) = ds.machine_seq[m].last() {
                m_load[m] = buf.start[last].saturating_add(ds.node_pt[last]);
            }
        }
        let mut cands: Vec<MoveCand> = Vec::with_capacity(top_cands.min(64));
        for m in 0..ds.num_machines {
            let seq = &ds.machine_seq[m]; if seq.len() <= 1 { continue; }
            let mut i = 0usize;
            while i < seq.len() {
                let a = seq[i]; if !crit[a] { i += 1; continue; }
                let bstart = i; let mut bend = i;
                while bend + 1 < seq.len() { let x = seq[bend]; let y = seq[bend+1]; if !crit[y] { break; } if buf.start[y] != buf.start[x].saturating_add(ds.node_pt[x]) { break; } bend += 1; }
                if bend > bstart {
                    if bstart > 0 { push_top_k_move(&mut cands, MoveCand { kind: 2, m_from: m, from: bstart-1, m_to: m, to: 0, new_pt: 0, score: buf.start[seq[bstart]] }, top_cands); }
                    if bend + 1 < seq.len() { push_top_k_move(&mut cands, MoveCand { kind: 2, m_from: m, from: bend, m_to: m, to: 0, new_pt: 0, score: buf.start[seq[bend]] }, top_cands); }
                    if bstart + 1 <= bend {
                        push_top_k_move(&mut cands, MoveCand { kind: 2, m_from: m, from: bstart, m_to: m, to: 0, new_pt: 0, score: buf.start[seq[bstart+1]] }, top_cands);
                        if bend >= 1 && bend - 1 >= bstart { push_top_k_move(&mut cands, MoveCand { kind: 2, m_from: m, from: bend-1, m_to: m, to: 0, new_pt: 0, score: buf.start[seq[bend]] }, top_cands); }
                    }
                    let max_shift = bend - bstart;
                    if max_shift > 1 {
                        let to_after = bstart + max_shift;
                        if to_after <= seq.len() { let tgt_idx = (bstart+max_shift).min(seq.len()-1); push_top_k_move(&mut cands, MoveCand { kind: 0, m_from: m, from: bstart, m_to: m, to: to_after, new_pt: 0, score: buf.start[seq[tgt_idx]] }, top_cands); }
                        let to_after_bend = bend - max_shift;
                        push_top_k_move(&mut cands, MoveCand { kind: 0, m_from: m, from: bend, m_to: m, to: to_after_bend, new_pt: 0, score: buf.start[seq[bend]] }, top_cands);
                    }
                    for &idx in &[bstart, bend] {
                        if idx >= seq.len() { continue; }
                        let node = seq[idx]; if !crit[node] { continue; }
                        let job = ds.node_job[node]; let op_idx = ds.node_op[node]; let product = pre.job_products[job];
                        let op = &pre.product_ops[product][op_idx];
                        if op.flex < 2 || op.machines.len() < 2 { continue; }
                        let old_m = ds.node_machine[node]; let old_pt = ds.node_pt[node];
                        let w_from = pre.machine_weight[old_m].max(1e-9);
                        let desired = buf.start[node];
                        let mut best = (NONE_USIZE, INF, u64::MAX);
                        let mut second = (NONE_USIZE, INF, u64::MAX);
                        for &(m_to, pt) in &op.machines {
                            let end_time = m_load.get(m_to).copied().unwrap_or(INF).max(desired).saturating_add(pt);
                            let cost = (end_time as u64).saturating_mul(10000).saturating_add(pt as u64);
                            if cost < best.2 { second = best; best = (m_to, pt, cost); }
                            else if cost < second.2 { second = (m_to, pt, cost); }
                        }
                        let best2 = [(best.0, best.1), (second.0, second.1)];
                        for &(m_to, new_pt) in &best2 {
                            if m_to == NONE_USIZE || m_to >= ds.num_machines || m_to == old_m || new_pt >= INF { continue; }
                            let w_to = pre.machine_weight[m_to].max(1e-9);
                            if !(new_pt + 1 < old_pt || w_to < w_from * 0.90) { continue; }
                            let pos0 = find_insert_pos_by_start(&ds.machine_seq[m_to][..], &buf.start, desired);
                            for pos in [pos0, pos0.saturating_add(1)] {
                                if pos > ds.machine_seq[m_to].len() { continue; }
                                let diffw = ((w_from - w_to).max(0.0) * pre.avg_op_min).max(0.0) as u32;
                                let difpt = old_pt.saturating_sub(new_pt);
                                let score = desired.saturating_add(old_pt).saturating_add(diffw).saturating_add(difpt.saturating_mul(2));
                                push_top_k_move(&mut cands, MoveCand { kind: 1, m_from: old_m, from: idx, m_to, to: pos, new_pt, score }, top_cands);
                            }
                        }
                    }
                }
                i = bend + 1;
            }
        }
        if cands.is_empty() { break; }
        let mut best_cand: Option<MoveCand> = None; let mut best_mk = cur_mk;
        for cand in &cands {
            if cand.kind == 0 {
                let m = cand.m_from; if m >= ds.num_machines || cand.from >= ds.machine_seq[m].len() { continue; }
                let new_idx = apply_insert(&mut ds.machine_seq[m], cand.from, cand.to);
                if let Some((mk2, _)) = eval_disj(ds, buf) { if mk2 < best_mk { best_mk = mk2; best_cand = Some(*cand); } }
                let _ = apply_insert(&mut ds.machine_seq[m], new_idx, cand.from);
            } else if cand.kind == 2 {
                let m = cand.m_from; if m >= ds.num_machines || cand.from + 1 >= ds.machine_seq[m].len() { continue; }
                if !apply_swap(&mut ds.machine_seq[m], cand.from) { continue; }
                if let Some((mk2, _)) = eval_disj(ds, buf) { if mk2 < best_mk { best_mk = mk2; best_cand = Some(*cand); } }
                let _ = apply_swap(&mut ds.machine_seq[m], cand.from);
            } else {
                let m_from = cand.m_from; let m_to = cand.m_to;
                if m_from >= ds.num_machines || m_to >= ds.num_machines || cand.from >= ds.machine_seq[m_from].len() { continue; }
                let node = ds.machine_seq[m_from][cand.from]; if ds.node_machine[node] != m_from { continue; }
                if let Some((node2, old_pt, ins_idx)) = apply_reroute(ds, m_from, cand.from, m_to, cand.to, cand.new_pt) {
                    if let Some((mk2, _)) = eval_disj(ds, buf) { if mk2 < best_mk { best_mk = mk2; best_cand = Some(*cand); } }
                    let _ = undo_reroute(ds, m_from, cand.from, m_to, ins_idx, node2, old_pt);
                }
            }
        }
        let Some(bc) = best_cand else { break };
        let mut accepted = false;
        if bc.kind == 0 {
            let m = bc.m_from; let new_idx = apply_insert(&mut ds.machine_seq[m], bc.from, bc.to);
            if let Some(ne) = eval_disj(ds, buf) { if ne.0 < cur_mk { *cur_eval = ne; cur_mk = ne.0; improved = true; accepted = true; } else { let _ = apply_insert(&mut ds.machine_seq[m], new_idx, bc.from); } }
            else { let _ = apply_insert(&mut ds.machine_seq[m], new_idx, bc.from); }
        } else if bc.kind == 2 {
            let m = bc.m_from;
            if m < ds.num_machines && bc.from + 1 < ds.machine_seq[m].len() {
                if apply_swap(&mut ds.machine_seq[m], bc.from) {
                    if let Some(ne) = eval_disj(ds, buf) { if ne.0 < cur_mk { *cur_eval = ne; cur_mk = ne.0; improved = true; accepted = true; } else { let _ = apply_swap(&mut ds.machine_seq[m], bc.from); } }
                    else { let _ = apply_swap(&mut ds.machine_seq[m], bc.from); }
                }
            }
        } else {
            if let Some((node2, old_pt, ins_idx)) = apply_reroute(ds, bc.m_from, bc.from, bc.m_to, bc.to, bc.new_pt) {
                if let Some(ne) = eval_disj(ds, buf) { if ne.0 < cur_mk { *cur_eval = ne; cur_mk = ne.0; improved = true; accepted = true; } else { let _ = undo_reroute(ds, bc.m_from, bc.from, bc.m_to, ins_idx, node2, old_pt); } }
                else { let _ = undo_reroute(ds, bc.m_from, bc.from, bc.m_to, ins_idx, node2, old_pt); }
            }
        }
        if !accepted { break; }
    }
    improved
}

#[inline]
pub fn apply_insert(seq: &mut Vec<usize>, from: usize, to_after_removal: usize) -> usize {
    let len = seq.len();
    if len == 0 || from >= len { return from.min(len.saturating_sub(1)); }
    let t = to_after_removal.min(len - 1);
    if t > from {
        seq[from..=t].rotate_left(1);
    } else if t < from {
        seq[t..=from].rotate_right(1);
    }
    t
}

#[inline]
pub fn apply_swap(seq: &mut [usize], i: usize) -> bool {
    if i + 1 >= seq.len() { return false; } seq.swap(i, i + 1); true
}

#[inline]
pub fn find_insert_pos_by_start(seq: &[usize], start: &[u32], desired_start: u32) -> usize {
    let mut lo = 0usize;
    let mut hi = seq.len();
    while lo < hi {
        let mid = (lo + hi) >> 1;
        if start[seq[mid]] < desired_start {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

#[inline]
pub fn apply_reroute(ds: &mut DisjSchedule, m_from: usize, idx_from: usize, m_to: usize, idx_to: usize, new_pt: u32) -> Option<(usize, u32, usize)> {
    if m_from >= ds.num_machines || m_to >= ds.num_machines || idx_from >= ds.machine_seq[m_from].len() { return None; }
    let node = ds.machine_seq[m_from].remove(idx_from); let old_pt = ds.node_pt[node];
    ds.node_machine[node] = m_to; ds.node_pt[node] = new_pt;
    let ins = idx_to.min(ds.machine_seq[m_to].len()); ds.machine_seq[m_to].insert(ins, node);
    Some((node, old_pt, ins))
}

#[inline]
pub fn undo_reroute(ds: &mut DisjSchedule, m_from: usize, idx_from: usize, m_to: usize, ins_idx: usize, node: usize, old_pt: u32) -> bool {
    if m_from >= ds.num_machines || m_to >= ds.num_machines || ins_idx >= ds.machine_seq[m_to].len() { return false; }
    let x = ds.machine_seq[m_to].remove(ins_idx);
    if x != node { let len_now = ds.machine_seq[m_to].len(); ds.machine_seq[m_to].insert(ins_idx.min(len_now), x); return false; }
    let ins_from = idx_from.min(ds.machine_seq[m_from].len());
    ds.machine_seq[m_from].insert(ins_from, node);
    ds.node_machine[node] = m_from; ds.node_pt[node] = old_pt; true
}

#[inline]
pub fn push_top_k_move(top: &mut Vec<MoveCand>, c: MoveCand, k: usize) {
    if k == 0 { return; }
    let mut pos = top.len(); while pos > 0 && top[pos-1].score < c.score { pos -= 1; }
    if pos >= k { return; } top.insert(pos, c); if top.len() > k { top.pop(); }
}

pub fn best_two_by_pt(op: &OpInfo) -> [(usize, u32); 2] {
    let mut r = [(NONE_USIZE, INF); 2];
    for &(m, pt) in &op.machines {
        if pt < r[0].1 { r[1] = r[0]; r[0] = (m, pt); }
        else if pt < r[1].1 { r[1] = (m, pt); }
    }
    r
}

pub fn job_bias_from_solution(pre: &Pre, sol: &Solution) -> Result<Vec<f64>> {
    let num_jobs = pre.job_ops_len.len();
    let mut completion = vec![0u32; num_jobs];
    let mut makespan = 0u32;

    for job in 0..num_jobs {
        let product = pre.job_products[job];
        let mut end_j = 0u32;
        for (op_idx, &(m, st)) in sol.job_schedule[job].iter().enumerate() {
            let op = &pre.product_ops[product][op_idx];
            let pt = pt_from_op(op, m).ok_or_else(|| anyhow!("Missing pt in bias calc"))?;
            end_j = end_j.max(st.saturating_add(pt));
        }
        completion[job] = end_j;
        makespan = makespan.max(end_j);
    }

    let denom = (makespan as f64).max(1.0);
    let exp = 3.0 + 1.2 * pre.high_flex + 0.6 * pre.jobshopness;
    Ok(completion.into_iter().map(|c| ((c as f64) / denom).powf(exp).clamp(0.0, 1.0)).collect())
}

pub fn machine_penalty_from_solution(pre: &Pre, sol: &Solution, num_machines: usize) -> Result<Vec<f64>> {
    let num_jobs = pre.job_ops_len.len();
    let mut m_end = vec![0u32; num_machines];
    let mut m_sum = vec![0u64; num_machines];
    let mut makespan = 0u32;

    for job in 0..num_jobs {
        let product = pre.job_products[job];
        for (op_idx, &(m, st)) in sol.job_schedule[job].iter().enumerate() {
            let op = &pre.product_ops[product][op_idx];
            let pt = pt_from_op(op, m).ok_or_else(|| anyhow!("Missing pt in machine penalty"))?;
            let end = st.saturating_add(pt);
            if end > m_end[m] { m_end[m] = end; }
            m_sum[m] = m_sum[m].saturating_add(pt as u64);
            makespan = makespan.max(end);
        }
    }

    let mk = (makespan as f64).max(1.0);
    let total: u64 = m_sum.iter().copied().sum();
    let avg = ((total as f64) / (num_machines as f64).max(1.0)).max(1.0);

    let use_load = pre.high_flex > 0.35 || pre.jobshopness > 0.45;
    let w_load = if use_load {
        (0.20 + 0.30 * pre.high_flex + 0.12 * pre.jobshopness).clamp(0.18, 0.58)
    } else {
        0.0
    };
    let w_end = 1.0 - w_load;
    let exp = 2.0 + 1.2 * pre.high_flex + 0.55 * pre.jobshopness;

    let mut mp = vec![0.0f64; num_machines];
    for m in 0..num_machines {
        let endn = (m_end[m] as f64 / mk).clamp(0.0, 1.0);
        let loadr = ((m_sum[m] as f64) / avg).max(0.0);
        let loadn = (loadr / (loadr + 1.0)).clamp(0.0, 1.0);
        let mix = (w_end * endn + w_load * loadn).clamp(0.0, 1.0);
        mp[m] = mix.powf(exp).clamp(0.0, 1.0);
    }
    Ok(mp)
}

pub fn route_pref_from_solution_lite(pre: &Pre, sol: &Solution, challenge: &Challenge) -> Result<RoutePrefLite> {
    let nm = challenge.num_machines;
    let np = challenge.product_processing_times.len();

    let mut counts: Vec<Vec<u16>> = Vec::with_capacity(np);
    let mut ops_len: Vec<usize> = Vec::with_capacity(np);
    for p in 0..np {
        let ol = challenge.product_processing_times[p].len();
        ops_len.push(ol);
        counts.push(vec![0u16; ol.saturating_mul(nm)]);
    }

    for job in 0..challenge.num_jobs {
        let product = pre.job_products[job];
        let ol = ops_len[product];
        for (op_idx, &(m, _st)) in sol.job_schedule[job].iter().enumerate() {
            if op_idx >= ol || m >= nm { continue; }
            let idx = op_idx * nm + m;
            counts[product][idx] = counts[product][idx].saturating_add(1);
        }
    }

    let mut rp: RoutePrefLite = Vec::with_capacity(np);
    for p in 0..np {
        let ol = ops_len[p];
        let denom_u32 = (challenge.jobs_per_product[p].max(1) as u32).max(1);
        let mut v: Vec<OpRoute> = Vec::with_capacity(ol);

        for op_idx in 0..ol {
            let base = op_idx * nm;
            let mut best_m = 0usize;
            let mut best_c = 0u16;
            let mut second_m = 0usize;
            let mut second_c = 0u16;

            for m in 0..nm {
                let c = counts[p][base + m];
                if c > best_c {
                    second_c = best_c; second_m = best_m;
                    best_c = c; best_m = m;
                } else if c > second_c && m != best_m {
                    second_c = c; second_m = m;
                }
            }

            let best_w = (((best_c as u32).saturating_mul(255)).saturating_add(denom_u32 / 2) / denom_u32).min(255) as u8;
            let second_w = (((second_c as u32).saturating_mul(255)).saturating_add(denom_u32 / 2) / denom_u32).min(255) as u8;

            v.push(OpRoute { best_m: best_m.min(255) as u8, best_w, second_m: second_m.min(255) as u8, second_w });
        }

        rp.push(v);
    }

    Ok(rp)
}

pub fn push_top_solutions(top: &mut Vec<(Solution, u32)>, sol: &Solution, mk: u32, cap: usize) {
    let pos = top.binary_search_by_key(&mk, |(_, m)| *m).unwrap_or_else(|e| e);
    top.insert(pos, (sol.clone(), mk));
    if top.len() > cap { top.truncate(cap); }
}

pub fn neh_reentrant_flow_solution(pre: &Pre, num_jobs: usize, num_machines: usize) -> Result<(Solution, u32)> {
    let route = pre.flow_route.as_ref().ok_or_else(|| anyhow!("No flow route"))?;
    let pt = pre.flow_pt_by_job.as_ref().ok_or_else(|| anyhow!("No flow pt"))?;
    let ops = route.len(); if ops == 0 || pt.len() != num_jobs { return Err(anyhow!("Invalid flow data")); }
    let mut jobs: Vec<usize> = (0..num_jobs).collect();
    jobs.sort_unstable_by(|&a, &b| { let sa: u32 = pt[a].iter().copied().sum(); let sb: u32 = pt[b].iter().copied().sum(); sb.cmp(&sa).then_with(|| a.cmp(&b)) });
    let mut seq: Vec<usize> = Vec::with_capacity(num_jobs); let mut mready = vec![0u32; num_machines];
    for &j in &jobs {
        if seq.is_empty() { seq.push(j); continue; }
        let mut best_mk = u32::MAX; let mut best_pos = 0usize; let mut tmp = seq.clone();
        for pos in 0..=seq.len() {
            tmp.clear(); tmp.extend_from_slice(&seq[..pos]); tmp.push(j); tmp.extend_from_slice(&seq[pos..]);
            let mk = reentrant_makespan_local(&tmp, route, pt, &mut mready);
            if mk < best_mk { best_mk = mk; best_pos = pos; }
        }
        seq.insert(best_pos, j);
    }
    let mk = reentrant_makespan_local(&seq, route, pt, &mut mready);
    let sol = build_perm_solution_from_seq_local(&seq, route, pt, num_jobs, num_machines);
    Ok((sol, mk))
}

fn reentrant_makespan_local(seq: &[usize], route: &[usize], pt: &[Vec<u32>], mready: &mut [u32]) -> u32 {
    mready.fill(0); let mut mk = 0u32;
    for &j in seq { let row = &pt[j]; let mut prev = 0u32; for (op_idx, &m) in route.iter().enumerate() { let p = row[op_idx]; let st = prev.max(mready[m]); let end = st.saturating_add(p); mready[m] = end; prev = end; } if prev > mk { mk = prev; } }
    mk
}

fn build_perm_solution_from_seq_local(seq: &[usize], route: &[usize], pt: &[Vec<u32>], num_jobs: usize, num_machines: usize) -> Solution {
    let ops = route.len(); let mut job_schedule: Vec<Vec<(usize, u32)>> = vec![Vec::with_capacity(ops); num_jobs]; let mut machine_ready = vec![0u32; num_machines];
    for &j in seq {
        if j >= num_jobs { continue; } let row = &pt[j]; let mut prev_end = 0u32;
        for (op_idx, &m) in route.iter().enumerate() {
            if op_idx >= row.len() || m >= num_machines { break; }
            let p = row[op_idx]; let st = prev_end.max(machine_ready[m]); job_schedule[j].push((m, st)); let end = st.saturating_add(p); machine_ready[m] = end; prev_end = end;
        }
    }
    Solution { job_schedule }
}

#[inline]
pub fn best_second_and_counts(time: u32, machine_avail: &[u32], op: &OpInfo) -> (u32, u32, usize, usize) {
    let mut best = INF; let mut second = INF; let mut cnt_best = 0usize; let mut cnt_best_idle = 0usize;
    for &(m, pt) in &op.machines {
        let end = time.max(machine_avail[m]).saturating_add(pt);
        if end < best { second = best; best = end; cnt_best = 1; cnt_best_idle = if machine_avail[m] <= time { 1 } else { 0 }; }
        else if end == best { cnt_best += 1; if machine_avail[m] <= time { cnt_best_idle += 1; } }
        else if end < second { second = end; }
    }
    if cnt_best > 1 { second = best; }
    (best, second, cnt_best.max(1), cnt_best_idle)
}

#[inline]
pub fn push_top_k(top: &mut Vec<Cand>, c: Cand, k: usize) {
    if k == 0 { return; }
    let mut pos = top.len(); while pos > 0 && top[pos-1].score < c.score { pos -= 1; }
    if pos >= k { return; } top.insert(pos, c); if top.len() > k { top.pop(); }
}

#[inline]
pub fn push_top_k_raw(top: &mut Vec<RawCand>, c: RawCand, k: usize) {
    if k == 0 { return; }
    let mut pos = top.len(); while pos > 0 && top[pos-1].base_score < c.base_score { pos -= 1; }
    if pos >= k { return; } top.insert(pos, c); if top.len() > k { top.pop(); }
}

#[inline]
pub fn choose_from_top_weighted(rng: &mut SmallRng, top: &[Cand]) -> Cand {
    if top.len() <= 1 { return top[0]; }
    let min_s = top.last().unwrap().score; let n = top.len().min(8);
    let mut w = [0.0f64; 8]; let mut sum = 0.0f64;
    for i in 0..n { let d = (top[i].score - min_s) + 1e-9; let wi = d * d; w[i] = wi; sum += wi; }
    if !(sum > 0.0) { return top[rng.gen_range(0..top.len())]; }
    let mut r = rng.gen::<f64>() * sum;
    for i in 0..n { r -= w[i]; if r <= 0.0 { return top[i]; } }
    top[n - 1]
}}
pub mod preprocess {
use anyhow::{anyhow, Result};
use tig_challenges::job_scheduling::*;
use super::types::*;

#[inline]
fn flow_makespan(seq: &[usize], pt: &[Vec<u32>], comp: &mut [u32]) -> u32 {
    comp.fill(0);
    for &j in seq {
        let row = &pt[j];
        if row.is_empty() { continue; }
        comp[0] = comp[0].saturating_add(row[0]);
        for k in 1..row.len() {
            let v = comp[k].max(comp[k - 1]).saturating_add(row[k]);
            comp[k] = v;
        }
    }
    *comp.last().unwrap_or(&0)
}

pub fn build_pre(challenge: &Challenge) -> Result<Pre> {
    let num_jobs = challenge.num_jobs;
    let num_machines = challenge.num_machines;

    let mut job_products = Vec::with_capacity(num_jobs);
    for (p, &cnt) in challenge.jobs_per_product.iter().enumerate() {
        for _ in 0..cnt {
            job_products.push(p);
        }
    }
    if job_products.len() != num_jobs {
        return Err(anyhow!("jobs_per_product sum mismatch"));
    }

    let num_products = challenge.product_processing_times.len();

    let mut product_ops: Vec<Vec<OpInfo>> = Vec::with_capacity(num_products);
    let mut best_machine_by_product: Vec<Vec<usize>> = Vec::with_capacity(num_products);

    let mut machine_load0 = vec![0.0f64; num_machines];
    let mut machine_scarcity = vec![0.0f64; num_machines];
    let mut machine_best_cnt = vec![0.0f64; num_machines];

    let mut total_ops: usize = 0;
    let mut total_min_work: f64 = 0.0;
    let mut total_flex_weighted: f64 = 0.0;

    let mut max_ops: usize = 1;
    let mut max_job_avg_work: f64 = 1.0;

    for (p, ops) in challenge.product_processing_times.iter().enumerate() {
        max_ops = max_ops.max(ops.len());

        let mut ops_info: Vec<OpInfo> = Vec::with_capacity(ops.len());
        let mut bests: Vec<usize> = Vec::with_capacity(ops.len());

        let mut sum_min_u64: u64 = 0;
        let mut sum_avg_f: f64 = 0.0;

        for op in ops {
            if op.is_empty() {
                ops_info.push(OpInfo {
                    machines: vec![],
                    min_pt: INF,
                    avg_pt: 0.0,
                    flex: 0,
                    bn_avg: 0.0,
                });
                bests.push(0);
                continue;
            }

            let mut machines: Vec<(usize, u32)> = Vec::with_capacity(op.len());
            let mut min_pt = INF;
            let mut sum = 0u64;

            let mut best_m = 0usize;
            let mut best_pt = INF;

            for (&m, &pt) in op.iter() {
                if m >= num_machines {
                    return Err(anyhow!("machine id out of range"));
                }
                machines.push((m, pt));
                min_pt = min_pt.min(pt);
                sum += pt as u64;

                if pt < best_pt || (pt == best_pt && m < best_m) {
                    best_pt = pt;
                    best_m = m;
                }
            }

            let flex = machines.len().max(1);
            let avg_pt = (sum as f64) / (flex as f64);

            sum_min_u64 += min_pt.min(INF / 2) as u64;
            sum_avg_f += avg_pt;

            machines.sort_unstable_by_key(|x| x.0);

            ops_info.push(OpInfo {
                machines,
                min_pt,
                avg_pt,
                flex,
                bn_avg: 0.0,
            });
            bests.push(best_m);
        }

        max_job_avg_work = max_job_avg_work.max(sum_avg_f);

        let cnt_u = challenge.jobs_per_product[p] as usize;
        let cnt_f = challenge.jobs_per_product[p] as f64;

        total_ops += ops_info.len() * cnt_u;
        total_min_work += (sum_min_u64 as f64) * cnt_f;

        for (oi, &bm) in ops_info.iter().zip(bests.iter()) {
            total_flex_weighted += (oi.flex as f64) * cnt_f;
            if bm < num_machines {
                machine_best_cnt[bm] += cnt_f;
            }

            if oi.min_pt < INF && oi.flex > 0 && !oi.machines.is_empty() {
                let flex_f = (oi.flex as f64).max(1.0);
                let delta = (oi.min_pt as f64) * cnt_f / flex_f;
                let delta_s = (oi.min_pt as f64) * cnt_f / (flex_f * flex_f);
                for &(m, _) in &oi.machines {
                    machine_load0[m] += delta;
                    machine_scarcity[m] += delta_s;
                }
            }
        }

        product_ops.push(ops_info);
        best_machine_by_product.push(bests);
    }

    let job_ops_len: Vec<usize> = job_products.iter().map(|&p| product_ops[p].len()).collect();

    let avg_machine_load = (total_min_work / (num_machines as f64).max(1.0)).max(1.0);
    let horizon = avg_machine_load;

    let avg_op_min = (total_min_work / (total_ops as f64).max(1.0)).max(1.0);
    let flex_avg = (total_flex_weighted / (total_ops as f64).max(1.0)).max(1.0);
    let flex_factor = (3.0 / flex_avg).clamp(0.6, 2.2);
    let hi_flex = flex_avg >= 5.0;
    let high_flex = ((flex_avg - 3.0) / 7.0).clamp(0.0, 1.0);

    let avg_machine_scarcity = {
        let s: f64 = machine_scarcity.iter().sum();
        (s / (num_machines as f64).max(1.0)).max(1e-9)
    };

    let load_cv = {
        let mean = avg_machine_load.max(1e-9);
        let mut var = 0.0f64;
        for &x in &machine_load0 {
            let d = (x / mean) - 1.0;
            var += d * d;
        }
        (var / (num_machines as f64)).sqrt().clamp(0.0, 2.5)
    };

    let mut flow_sum = 0.0f64;
    let mut flow_cnt = 0usize;
    let mut counts = vec![0u32; num_machines];
    for op_idx in 0..max_ops {
        counts.fill(0);
        let mut tot = 0u32;

        for p in 0..num_products {
            if op_idx >= best_machine_by_product[p].len() {
                continue;
            }
            let bm = best_machine_by_product[p][op_idx];
            let w_u32 = challenge.jobs_per_product[p] as u32;
            if w_u32 == 0 {
                continue;
            }
            counts[bm] = counts[bm].saturating_add(w_u32);
            tot = tot.saturating_add(w_u32);
        }

        if tot > 0 {
            let mut mx = 0u32;
            for &c in &counts {
                mx = mx.max(c);
            }
            flow_sum += (mx as f64) / (tot as f64);
            flow_cnt += 1;
        }
    }
    let flow_like = if flow_cnt > 0 { (flow_sum / (flow_cnt as f64)).clamp(0.0, 1.0) } else { 0.5 };
    let jobshopness = (1.0 - flow_like).clamp(0.0, 1.0);

    let mut machine_weight = vec![1.0f64; num_machines];
    {
        let mean = avg_machine_load.max(1e-9);
        let exp = (1.10 + 0.35 * load_cv + 0.20 * jobshopness).clamp(1.05, 1.70);
        for m in 0..num_machines {
            let r = (machine_load0[m] / mean).max(0.05);
            machine_weight[m] = r.powf(exp).clamp(0.55, 3.75);
        }
    }

    let machine_best_pop = {
        let tot: f64 = machine_best_cnt.iter().sum();
        let mean = (tot / (num_machines as f64).max(1.0)).max(1e-9);
        let mut pop = vec![0.0f64; num_machines];
        for m in 0..num_machines {
            let r = (machine_best_cnt[m] / mean).clamp(0.0, 10.0);
            pop[m] = (r / (1.0 + r)).clamp(0.0, 1.0);
        }
        pop
    };

    let bn_focus = ((3.0 / flex_avg).clamp(0.7, 2.6) * (1.0 + 0.55 * load_cv) * (0.85 + 0.55 * jobshopness)).clamp(0.6, 3.4);

    let mut product_suf_min: Vec<Vec<u32>> = Vec::with_capacity(product_ops.len());
    let mut product_suf_avg: Vec<Vec<f64>> = Vec::with_capacity(product_ops.len());
    let mut product_suf_bn: Vec<Vec<f64>> = Vec::with_capacity(product_ops.len());
    let mut product_next_min: Vec<Vec<u32>> = Vec::with_capacity(product_ops.len());
    let mut product_next_flex_inv: Vec<Vec<f64>> = Vec::with_capacity(product_ops.len());

    let mut max_job_bn: f64 = 1e-9;

    for ops in product_ops.iter_mut() {
        let n = ops.len();
        let mut suf_m = vec![0u32; n + 1];
        let mut suf_a = vec![0.0f64; n + 1];
        let mut suf_bn = vec![0.0f64; n + 1];

        let mut nxt_m = vec![0u32; n + 1];
        let mut nxt_fi = vec![0.0f64; n + 1];

        for i in (0..n).rev() {
            let oi = &mut ops[i];

            if oi.flex == 0 || oi.machines.is_empty() || oi.min_pt >= INF {
                oi.bn_avg = 0.0;
            } else {
                let mut sum = 0.0f64;
                for &(m, pt) in &oi.machines {
                    sum += (pt as f64) * machine_weight[m];
                }
                oi.bn_avg = sum / (oi.flex as f64);
            }

            suf_m[i] = suf_m[i + 1].saturating_add(oi.min_pt.min(INF / 2));
            suf_a[i] = suf_a[i + 1] + oi.avg_pt;
            suf_bn[i] = suf_bn[i + 1] + oi.bn_avg;

            if i + 1 < n {
                let next = &ops[i + 1];
                nxt_m[i] = next.min_pt;
                nxt_fi[i] = if next.flex > 0 { 1.0 / (next.flex as f64) } else { 0.0 };
            }
        }

        max_job_bn = max_job_bn.max(suf_bn[0]);

        product_suf_min.push(suf_m);
        product_suf_avg.push(suf_a);
        product_suf_bn.push(suf_bn);
        product_next_min.push(nxt_m);
        product_next_flex_inv.push(nxt_fi);
    }

    let time_scale = (horizon * (2.65 + 0.15 * load_cv + 0.10 * jobshopness + 0.10 * high_flex)).max(1.0);

    let mut job_flow_pref = vec![0.0f64; num_jobs];
    let use_flow_pref = flow_like > 0.82 && jobshopness < 0.38 && max_ops >= 2;

    if use_flow_pref {
        let m = max_ops.max(1);
        let mut job_pt: Vec<Vec<u32>> = Vec::with_capacity(num_jobs);
        for j in 0..num_jobs {
            let p = job_products[j];
            let ops = &product_ops[p];
            let mut v = vec![0u32; m];
            for s in 0..m.min(ops.len()) {
                v[s] = ops[s].min_pt.min(INF / 2);
            }
            job_pt.push(v);
        }

        let mut jobs2: Vec<usize> = (0..num_jobs).collect();
        jobs2.sort_unstable_by(|&a, &b| {
            let sa: u32 = job_pt[a].iter().copied().sum();
            let sb: u32 = job_pt[b].iter().copied().sum();
            sb.cmp(&sa).then_with(|| a.cmp(&b))
        });

        let mut perm: Vec<usize> = Vec::with_capacity(num_jobs);
        let mut comp = vec![0u32; m];
        let mut tmp: Vec<usize> = Vec::with_capacity(num_jobs);

        for &j in &jobs2 {
            if perm.is_empty() {
                perm.push(j);
                continue;
            }
            let mut best_mk = u32::MAX;
            let mut best_pos = 0usize;
            for pos in 0..=perm.len() {
                tmp.clear();
                tmp.extend_from_slice(&perm[..pos]);
                tmp.push(j);
                tmp.extend_from_slice(&perm[pos..]);
                let mk = flow_makespan(&tmp, &job_pt, &mut comp);
                if mk < best_mk {
                    best_mk = mk;
                    best_pos = pos;
                }
            }
            perm.insert(best_pos, j);
        }

        let n1 = (num_jobs.saturating_sub(1)) as f64;
        for (pos, &j) in perm.iter().enumerate() {
            job_flow_pref[j] = if n1 > 0.0 { 1.0 - (pos as f64) / n1 } else { 1.0 };
        }
    }

    let flow_w = if use_flow_pref {
        let t = ((flow_like - 0.82) / 0.18).clamp(0.0, 1.0);
        let base = (0.10 + 0.26 * t).clamp(0.10, 0.36);
        let flex_adj = (1.0 - 0.45 * high_flex).clamp(0.55, 1.0);
        base * flex_adj
    } else {
        0.0
    };

    let slack_base = (0.04 + 0.14 * jobshopness + 0.11 * high_flex).clamp(0.03, 0.22);

    let mut flow_route: Option<Vec<usize>> = None;
    let mut flow_pt_by_job: Option<Vec<Vec<u32>>> = None;
    let mut strict_route: Option<Vec<usize>> = None;
    if !product_ops.is_empty() {
        let common_len = product_ops[0].len();
        let mut ok = common_len > 0;

        for ops in &product_ops {
            if ops.len() != common_len {
                ok = false;
                break;
            }
        }

        if ok && flex_avg <= 1.25 {
            let mut route: Vec<usize> = Vec::with_capacity(common_len);
            for i in 0..common_len {
                let mut m0: Option<usize> = None;
                for p in 0..num_products {
                    let op = &product_ops[p][i];
                    if op.flex != 1 || op.machines.len() != 1 {
                        ok = false;
                        break;
                    }
                    let mid = op.machines[0].0;
                    if let Some(mm) = m0 {
                        if mm != mid {
                            ok = false;
                            break;
                        }
                    } else {
                        m0 = Some(mid);
                    }
                }
                if !ok {
                    break;
                }
                route.push(m0.unwrap());
            }

            if ok {
                strict_route = Some(route.clone());

                let mut pt_by_job: Vec<Vec<u32>> = Vec::with_capacity(num_jobs);
                for j in 0..num_jobs {
                    let prod = job_products[j];
                    let mut row = Vec::with_capacity(common_len);
                    for i in 0..common_len {
                        row.push(product_ops[prod][i].machines[0].1);
                    }
                    pt_by_job.push(row);
                }
                flow_route = Some(route);
                flow_pt_by_job = Some(pt_by_job);
            }
        }
    }

    let chaotic_like = high_flex > 0.85 && jobshopness > 0.75;

    Ok(Pre {
        job_products,
        job_ops_len,
        product_ops,
        product_suf_min,
        product_suf_avg,
        product_suf_bn,
        product_next_min,
        product_next_flex_inv,
        machine_load0,
        machine_scarcity,
        machine_weight,
        machine_best_pop,
        avg_machine_load,
        avg_machine_scarcity,
        avg_op_min,
        horizon,
        time_scale,
        max_ops: max_ops.max(1),
        max_job_avg_work: max_job_avg_work.max(1.0),
        max_job_bn: max_job_bn.max(1e-9),
        flex_avg,
        flex_factor,
        hi_flex,
        high_flex,
        flow_like,
        flow_w,
        job_flow_pref,
        jobshopness,
        bn_focus,
        load_cv,
        slack_base,
        total_ops,
        chaotic_like,
        flow_route,
        flow_pt_by_job,
        strict_route,
    })
}
}


pub mod job_shop {
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
fn slack_urgency_js(pre: &Pre, target_mk: Option<u32>, time: u32, product: usize, op_idx: usize) -> f64 {
    let Some(tgt) = target_mk else { return 0.0 };
    let lb = (time as u64).saturating_add(pre.product_suf_min[product][op_idx] as u64);
    let slack = (tgt as i64) - (lb as i64);
    let scale = (0.70 * pre.avg_op_min).max(1.0);
    let pos = (slack.max(0) as f64) / scale; let neg = ((-slack).max(0) as f64) / scale;
    (1.0 / (1.0 + pos)).clamp(0.0, 1.0) + (0.35 * neg).min(3.0)
}

#[inline]
fn route_pref_bonus_js(rp: Option<&RoutePrefLite>, product: usize, op_idx: usize, machine: usize) -> f64 {
    let Some(rp) = rp else { return 0.0 };
    if product >= rp.len() || op_idx >= rp[product].len() { return 0.0; }
    let r = rp[product][op_idx]; let mu = js_cached_machine_u8(machine);
    if mu == r.best_m { (r.best_w as f64) / 255.0 } else if mu == r.second_m { (r.second_w as f64) / 255.0 } else { 0.0 }
}

#[inline(always)]
fn progress_mode_js(progress: f64) -> u8 {
    if progress < 0.34 { 0 } else if progress < 0.72 { 1 } else { 2 }
}

static mut JS_SCAR_N_PTR: *const f64 = std::ptr::null();
static mut JS_SCAR_N_LEN: usize = 0;
static mut JS_POP_PTR: *const f64 = std::ptr::null();
static mut JS_POP_LEN: usize = 0;
static mut JS_PEN_PTR: *const f64 = std::ptr::null();
static mut JS_PEN_LEN: usize = 0;
static mut JS_U8_PTR: *const u8 = std::ptr::null();
static mut JS_U8_LEN: usize = 0;

#[inline(always)]
unsafe fn set_js_machine_static_cache(scar_n: &[f64], pop: &[f64], pen: &[f64], mu8: &[u8]) {
    JS_SCAR_N_PTR = scar_n.as_ptr();
    JS_SCAR_N_LEN = scar_n.len();
    JS_POP_PTR = pop.as_ptr();
    JS_POP_LEN = pop.len();
    JS_PEN_PTR = pen.as_ptr();
    JS_PEN_LEN = pen.len();
    JS_U8_PTR = mu8.as_ptr();
    JS_U8_LEN = mu8.len();
}

#[inline(always)]
unsafe fn clear_js_machine_static_cache() {
    JS_SCAR_N_PTR = std::ptr::null();
    JS_SCAR_N_LEN = 0;
    JS_POP_PTR = std::ptr::null();
    JS_POP_LEN = 0;
    JS_PEN_PTR = std::ptr::null();
    JS_PEN_LEN = 0;
    JS_U8_PTR = std::ptr::null();
    JS_U8_LEN = 0;
}

struct JsMachineStaticCacheGuard;

impl Drop for JsMachineStaticCacheGuard {
    fn drop(&mut self) {
        unsafe { clear_js_machine_static_cache(); }
    }
}

#[inline(always)]
fn js_cached_scar_n(pre: &Pre, machine: usize) -> f64 {
    unsafe {
        if machine < JS_SCAR_N_LEN {
            *JS_SCAR_N_PTR.add(machine)
        } else {
            pre.machine_scarcity[machine] / pre.avg_machine_scarcity.max(1e-9)
        }
    }
}

#[inline(always)]
fn js_cached_pop(pre: &Pre, machine: usize) -> f64 {
    unsafe {
        if machine < JS_POP_LEN {
            *JS_POP_PTR.add(machine)
        } else {
            pre.machine_best_pop[machine]
        }
    }
}

#[inline(always)]
fn js_cached_penalty(machine: usize, machine_penalty: f64) -> f64 {
    unsafe {
        if machine < JS_PEN_LEN {
            *JS_PEN_PTR.add(machine)
        } else {
            machine_penalty.clamp(0.0, 1.0)
        }
    }
}

#[inline(always)]
fn js_cached_machine_u8(machine: usize) -> u8 {
    unsafe {
        if machine < JS_U8_LEN {
            *JS_U8_PTR.add(machine)
        } else {
            machine.min(255) as u8
        }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn score_candidate_critical_path(
    pre: &Pre, job: usize, product: usize, op_idx: usize,
    ops_rem: usize, op: &OpInfo, machine: usize, _pt: u32, time: u32,
    target_mk: Option<u32>, best_end: u32, second_end: u32, best_cnt_total: usize,
    progress: f64, job_bias: f64, _machine_penalty: f64, _dynamic_load: f64,
    route_pref: Option<&RoutePrefLite>, route_w: f64, jitter: f64,
) -> f64 {
    let rem_min = pre.product_suf_min[product][op_idx] as f64;
    let flex_f = (op.flex as f64).max(1.0); let flex_inv = 1.0 / flex_f;
    let rem_min_n = rem_min / pre.horizon.max(1.0); let ops_n = (ops_rem as f64) / (pre.max_ops as f64).max(1.0);
    let end_n = (best_end as f64) / pre.time_scale.max(1.0);
    let regret = if second_end >= INF { pre.avg_op_min * 2.6 } else { (second_end - best_end) as f64 };
    let reg_n = (regret / pre.avg_op_min.max(1.0)).clamp(0.0, 6.0);
    let scarcity_urg = 1.0 / (best_cnt_total as f64).max(1.0);
    let density_n = ((rem_min / (ops_rem as f64).max(1.0)) / pre.avg_op_min.max(1.0)).clamp(0.0, 4.0);
    let next_min = pre.product_next_min[product][op_idx] as f64; let next_min_n = next_min / pre.horizon.max(1.0);
    let next_flex_inv = pre.product_next_flex_inv[product][op_idx];
    let p2 = progress * progress; let next_w_base = 0.12 + p2 * 0.28;
    let next_term_raw = (0.55*next_min_n + 0.45*next_flex_inv) * (1.0 + 0.30*density_n*pre.high_flex);
    let flow_term = pre.flow_w * pre.job_flow_pref[job] * (0.65 + 0.70*(1.0-progress));
    let slack_u = slack_urgency_js(pre, target_mk, time, product, op_idx);
    let slack_w = pre.slack_base * (0.25 + 0.75*progress); let slack_reg_boost = 1.0 + 0.40*reg_n*progress;
    let pop_pen = if pre.chaotic_like && op.flex >= 2 { let pop = js_cached_pop(pre, machine); (0.07+0.15*(1.0-progress)).clamp(0.05,0.24)*pop*pre.flex_factor } else { 0.0 };
    let route_gain = (0.70 + 0.80*(1.0-progress)).clamp(0.70, 1.40);
    let route_term = if route_w > 0.0 && op.flex >= 2 { route_w*route_gain*route_pref_bonus_js(route_pref, product, op_idx, machine) } else { 0.0 };
    let next_term = next_w_base*0.30*next_term_raw; let slack_term = slack_w*slack_u*slack_reg_boost;
    (1.03*rem_min_n)+(0.10*ops_n)+(0.24*scarcity_urg)+(0.20*pre.flex_factor)*flex_inv+next_term+0.10*slack_term-(0.70*end_n)-pop_pen+(0.45*job_bias)+flow_term+route_term+jitter
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn score_candidate_most_work(
    pre: &Pre, job: usize, product: usize, op_idx: usize,
    ops_rem: usize, op: &OpInfo, machine: usize, _pt: u32, _time: u32,
    _target_mk: Option<u32>, best_end: u32, _second_end: u32, best_cnt_total: usize,
    progress: f64, job_bias: f64, _machine_penalty: f64, _dynamic_load: f64,
    route_pref: Option<&RoutePrefLite>, route_w: f64, jitter: f64,
) -> f64 {
    let rem_min = pre.product_suf_min[product][op_idx] as f64;
    let rem_avg = pre.product_suf_avg[product][op_idx];
    let flex_f = (op.flex as f64).max(1.0); let flex_inv = 1.0 / flex_f;
    let rem_avg_n = rem_avg / pre.max_job_avg_work.max(1e-9); let ops_n = (ops_rem as f64) / (pre.max_ops as f64).max(1.0);
    let end_n = (best_end as f64) / pre.time_scale.max(1.0);
    let scarcity_urg = 1.0 / (best_cnt_total as f64).max(1.0);
    let density_n = ((rem_min / (ops_rem as f64).max(1.0)) / pre.avg_op_min.max(1.0)).clamp(0.0, 4.0);
    let next_min = pre.product_next_min[product][op_idx] as f64; let next_min_n = next_min / pre.horizon.max(1.0);
    let next_flex_inv = pre.product_next_flex_inv[product][op_idx];
    let p2 = progress * progress; let next_w_base = 0.12 + p2 * 0.28;
    let next_term_raw = (0.55*next_min_n + 0.45*next_flex_inv) * (1.0 + 0.30*density_n*pre.high_flex);
    let flow_term = pre.flow_w * pre.job_flow_pref[job] * (0.65 + 0.70*(1.0-progress));
    let pop_pen = if pre.chaotic_like && op.flex >= 2 { let pop = js_cached_pop(pre, machine); (0.07+0.15*(1.0-progress)).clamp(0.05,0.24)*pop*pre.flex_factor } else { 0.0 };
    let route_gain = (0.70 + 0.80*(1.0-progress)).clamp(0.70, 1.40);
    let route_term = if route_w > 0.0 && op.flex >= 2 { route_w*route_gain*route_pref_bonus_js(route_pref, product, op_idx, machine) } else { 0.0 };
    let next_term = next_w_base*0.25*next_term_raw;
    (1.00*rem_avg_n)+(0.12*ops_n)+(0.18*scarcity_urg)+(0.15*pre.flex_factor)*flex_inv+next_term-(0.62*end_n)-pop_pen+(0.45*job_bias)+flow_term+route_term+jitter
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn score_candidate_least_flex(
    pre: &Pre, job: usize, product: usize, op_idx: usize,
    ops_rem: usize, op: &OpInfo, machine: usize, _pt: u32, _time: u32,
    _target_mk: Option<u32>, best_end: u32, _second_end: u32, best_cnt_total: usize,
    progress: f64, job_bias: f64, _machine_penalty: f64, _dynamic_load: f64,
    route_pref: Option<&RoutePrefLite>, route_w: f64, jitter: f64,
) -> f64 {
    let rem_min = pre.product_suf_min[product][op_idx] as f64;
    let flex_f = (op.flex as f64).max(1.0); let flex_inv = 1.0 / flex_f;
    let rem_min_n = rem_min / pre.horizon.max(1.0);
    let end_n = (best_end as f64) / pre.time_scale.max(1.0);
    let scarcity_urg = 1.0 / (best_cnt_total as f64).max(1.0);
    let density_n = ((rem_min / (ops_rem as f64).max(1.0)) / pre.avg_op_min.max(1.0)).clamp(0.0, 4.0);
    let next_min = pre.product_next_min[product][op_idx] as f64; let next_min_n = next_min / pre.horizon.max(1.0);
    let next_flex_inv = pre.product_next_flex_inv[product][op_idx];
    let p2 = progress * progress; let next_w_base = 0.12 + p2 * 0.28;
    let next_term_raw = (0.55*next_min_n + 0.45*next_flex_inv) * (1.0 + 0.30*density_n*pre.high_flex);
    let flow_term = pre.flow_w * pre.job_flow_pref[job] * (0.65 + 0.70*(1.0-progress));
    let pop_pen = if pre.chaotic_like && op.flex >= 2 { let pop = js_cached_pop(pre, machine); (0.07+0.15*(1.0-progress)).clamp(0.05,0.24)*pop*pre.flex_factor } else { 0.0 };
    let route_gain = (0.70 + 0.80*(1.0-progress)).clamp(0.70, 1.40);
    let route_term = if route_w > 0.0 && op.flex >= 2 { route_w*route_gain*route_pref_bonus_js(route_pref, product, op_idx, machine) } else { 0.0 };
    let next_term = next_w_base*0.20*next_term_raw;
    (1.00*flex_inv)+(0.28*rem_min_n)+(0.22*scarcity_urg)+next_term-(0.55*end_n)-pop_pen+(0.35*job_bias)+flow_term+route_term+jitter
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn score_candidate_shortest_proc(
    pre: &Pre, job: usize, product: usize, op_idx: usize,
    ops_rem: usize, op: &OpInfo, machine: usize, pt: u32, _time: u32,
    _target_mk: Option<u32>, best_end: u32, _second_end: u32, best_cnt_total: usize,
    progress: f64, job_bias: f64, _machine_penalty: f64, _dynamic_load: f64,
    route_pref: Option<&RoutePrefLite>, route_w: f64, jitter: f64,
) -> f64 {
    let rem_min = pre.product_suf_min[product][op_idx] as f64;
    let rem_min_n = rem_min / pre.horizon.max(1.0);
    let end_n = (best_end as f64) / pre.time_scale.max(1.0); let proc_n = (pt as f64) / pre.avg_op_min.max(1.0);
    let scarcity_urg = 1.0 / (best_cnt_total as f64).max(1.0);
    let density_n = ((rem_min / (ops_rem as f64).max(1.0)) / pre.avg_op_min.max(1.0)).clamp(0.0, 4.0);
    let next_min = pre.product_next_min[product][op_idx] as f64; let next_min_n = next_min / pre.horizon.max(1.0);
    let next_flex_inv = pre.product_next_flex_inv[product][op_idx];
    let p2 = progress * progress; let next_w_base = 0.12 + p2 * 0.28;
    let next_term_raw = (0.55*next_min_n + 0.45*next_flex_inv) * (1.0 + 0.30*density_n*pre.high_flex);
    let flow_term = pre.flow_w * pre.job_flow_pref[job] * (0.65 + 0.70*(1.0-progress));
    let pop_pen = if pre.chaotic_like && op.flex >= 2 { let pop = js_cached_pop(pre, machine); (0.07+0.15*(1.0-progress)).clamp(0.05,0.24)*pop*pre.flex_factor } else { 0.0 };
    let route_gain = (0.70 + 0.80*(1.0-progress)).clamp(0.70, 1.40);
    let route_term = if route_w > 0.0 && op.flex >= 2 { route_w*route_gain*route_pref_bonus_js(route_pref, product, op_idx, machine) } else { 0.0 };
    let next_term = next_w_base*0.20*next_term_raw;
    (-1.00*proc_n)+(0.25*rem_min_n)+(0.12*scarcity_urg)+next_term-(0.20*end_n)-pop_pen+(0.25*job_bias)+flow_term+route_term+jitter
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn score_candidate_regret(
    pre: &Pre, job: usize, product: usize, op_idx: usize,
    ops_rem: usize, op: &OpInfo, machine: usize, _pt: u32, _time: u32,
    _target_mk: Option<u32>, best_end: u32, second_end: u32, best_cnt_total: usize,
    progress: f64, job_bias: f64, _machine_penalty: f64, _dynamic_load: f64,
    route_pref: Option<&RoutePrefLite>, route_w: f64, jitter: f64,
) -> f64 {
    let rem_min = pre.product_suf_min[product][op_idx] as f64;
    let rem_min_n = rem_min / pre.horizon.max(1.0);
    let end_n = (best_end as f64) / pre.time_scale.max(1.0);
    let regret = if second_end >= INF { pre.avg_op_min * 2.6 } else { (second_end - best_end) as f64 };
    let reg_n = (regret / pre.avg_op_min.max(1.0)).clamp(0.0, 6.0);
    let scarcity_urg = 1.0 / (best_cnt_total as f64).max(1.0);
    let density_n = ((rem_min / (ops_rem as f64).max(1.0)) / pre.avg_op_min.max(1.0)).clamp(0.0, 4.0);
    let next_min = pre.product_next_min[product][op_idx] as f64; let next_min_n = next_min / pre.horizon.max(1.0);
    let next_flex_inv = pre.product_next_flex_inv[product][op_idx];
    let p2 = progress * progress; let next_w_base = 0.12 + p2 * 0.28;
    let next_term_raw = (0.55*next_min_n + 0.45*next_flex_inv) * (1.0 + 0.30*density_n*pre.high_flex);
    let flow_term = pre.flow_w * pre.job_flow_pref[job] * (0.65 + 0.70*(1.0-progress));
    let pop_pen = if pre.chaotic_like && op.flex >= 2 { let pop = js_cached_pop(pre, machine); (0.07+0.15*(1.0-progress)).clamp(0.05,0.24)*pop*pre.flex_factor } else { 0.0 };
    let route_gain = (0.70 + 0.80*(1.0-progress)).clamp(0.70, 1.40);
    let route_term = if route_w > 0.0 && op.flex >= 2 { route_w*route_gain*route_pref_bonus_js(route_pref, product, op_idx, machine) } else { 0.0 };
    let next_term = next_w_base*0.25*next_term_raw;
    (1.05*reg_n)+(0.55*rem_min_n)+(0.22*scarcity_urg)+next_term-(0.68*end_n)-pop_pen+(0.35*job_bias)+flow_term+route_term+jitter
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn score_candidate_end_tight(
    pre: &Pre, job: usize, product: usize, op_idx: usize,
    ops_rem: usize, op: &OpInfo, machine: usize, pt: u32, time: u32,
    target_mk: Option<u32>, best_end: u32, second_end: u32, best_cnt_total: usize,
    progress: f64, job_bias: f64, machine_penalty: f64, _dynamic_load: f64,
    route_pref: Option<&RoutePrefLite>, route_w: f64, jitter: f64,
) -> f64 {
    let rem_min = pre.product_suf_min[product][op_idx] as f64;
    let rem_avg = pre.product_suf_avg[product][op_idx];
    let flex_f = (op.flex as f64).max(1.0); let flex_inv = 1.0 / flex_f;
    let rem_min_n = rem_min / pre.horizon.max(1.0); let rem_avg_n = rem_avg / pre.max_job_avg_work.max(1e-9);
    let ops_n = (ops_rem as f64) / (pre.max_ops as f64).max(1.0);
    let scar_n = js_cached_scar_n(pre, machine);
    let end_n = (best_end as f64) / pre.time_scale.max(1.0); let proc_n = (pt as f64) / pre.avg_op_min.max(1.0);
    let regret = if second_end >= INF { pre.avg_op_min * 2.6 } else { (second_end - best_end) as f64 };
    let reg_n = (regret / pre.avg_op_min.max(1.0)).clamp(0.0, 6.0);
    let scarcity_urg = 1.0 / (best_cnt_total as f64).max(1.0);
    let density_n = ((rem_min / (ops_rem as f64).max(1.0)) / pre.avg_op_min.max(1.0)).clamp(0.0, 4.0);
    let next_min = pre.product_next_min[product][op_idx] as f64; let next_min_n = next_min / pre.horizon.max(1.0);
    let next_flex_inv = pre.product_next_flex_inv[product][op_idx];
    let next_term_raw = (0.55*next_min_n + 0.45*next_flex_inv) * (1.0 + 0.30*density_n*pre.high_flex);
    let js = pre.jobshopness;
    let avg_flex_inv = 1.0 / pre.flex_avg.max(1.0); let scarce_match = scar_n * (flex_inv - avg_flex_inv);
    let mpen = js_cached_penalty(machine, machine_penalty); let mpen_gain = 1.0 + 0.85*pre.high_flex;
    let phase = progress_mode_js(progress);
    let (cp_w, phase_reg_w, next_w, end_w, flow_gain, route_gain, slack_w, proc_w, flex_w, scarce_w, scarcity_w, mpen_w) = match phase {
        0 => (
            1.00 + 0.18*js,
            0.32 + 0.28*js,
            0.17*(0.80 + 0.40*js),
            0.92 + 0.08*pre.high_flex,
            1.24,
            1.24,
            0.0,
            0.18,
            0.24*pre.flex_factor,
            0.10*pre.flex_factor,
            0.14,
            (0.08 + 0.20*pre.high_flex)*pre.flex_factor,
        ),
        1 => (
            1.15 + 0.30*js,
            0.50 + 0.40*js,
            0.24*(0.90 + 0.50*js),
            1.18 + 0.15*pre.high_flex,
            0.96,
            0.98,
            pre.slack_base*(0.40 + 0.30*js),
            0.20,
            0.30*pre.flex_factor,
            0.18*pre.flex_factor,
            0.18,
            (0.10 + 0.34*pre.high_flex)*pre.flex_factor,
        ),
        _ => (
            1.30 + 0.32*js,
            0.72 + 0.42*js,
            0.18*(0.70 + 0.45*js),
            1.58 + 0.22*pre.high_flex,
            0.72,
            0.74,
            pre.slack_base*(0.92 + 0.34*js),
            0.24,
            0.34*pre.flex_factor,
            0.22*pre.flex_factor,
            0.20,
            (0.12 + 0.46*pre.high_flex)*pre.flex_factor,
        ),
    };
    let flow_term = pre.flow_w * pre.job_flow_pref[job] * flow_gain;
    let slack_term = if slack_w > 0.0 {
        let slack_u = slack_urgency_js(pre, target_mk, time, product, op_idx);
        let slack_reg_boost = 1.0 + if phase == 2 { 0.50 } else { 0.30 } * reg_n;
        slack_w * slack_u * slack_reg_boost
    } else { 0.0 };
    let pop_pen = if pre.chaotic_like && op.flex >= 2 {
        let pop = js_cached_pop(pre, machine);
        let ppw = match phase { 0 => 0.20, 1 => 0.14, _ => 0.08 };
        ppw * pop * pre.flex_factor
    } else { 0.0 };
    let route_term = if route_w > 0.0 && op.flex >= 2 { route_w*route_gain*route_pref_bonus_js(route_pref, product, op_idx, machine) } else { 0.0 };
    let next_term = next_w * next_term_raw;
    (cp_w*rem_min_n)+0.12*rem_avg_n+0.08*ops_n+scarcity_w*scarcity_urg+flex_w*flex_inv+scarce_w*scarce_match+(phase_reg_w*pre.flex_factor)*reg_n+next_term+slack_term-end_w*end_n-proc_w*proc_n-pop_pen-(mpen_gain*mpen_w)*mpen+0.55*job_bias+flow_term+route_term+jitter
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn score_candidate_bn_heavy(
    pre: &Pre, job: usize, product: usize, op_idx: usize,
    ops_rem: usize, op: &OpInfo, machine: usize, pt: u32, time: u32,
    target_mk: Option<u32>, best_end: u32, second_end: u32, best_cnt_total: usize,
    progress: f64, job_bias: f64, machine_penalty: f64, dynamic_load: f64,
    route_pref: Option<&RoutePrefLite>, route_w: f64, jitter: f64,
) -> f64 {
    let rem_min = pre.product_suf_min[product][op_idx] as f64;
    let rem_avg = pre.product_suf_avg[product][op_idx];
    let rem_bn = pre.product_suf_bn[product][op_idx];
    let flex_f = (op.flex as f64).max(1.0); let flex_inv = 1.0 / flex_f;
    let rem_min_n = rem_min / pre.horizon.max(1.0); let rem_avg_n = rem_avg / pre.max_job_avg_work.max(1e-9);
    let bn_n = rem_bn / pre.max_job_bn.max(1e-9); let ops_n = (ops_rem as f64) / (pre.max_ops as f64).max(1.0);
    let load_n = dynamic_load / pre.avg_machine_load.max(1e-9); let scar_n = js_cached_scar_n(pre, machine);
    let end_n = (best_end as f64) / pre.time_scale.max(1.0); let proc_n = (pt as f64) / pre.avg_op_min.max(1.0);
    let regret = if second_end >= INF { pre.avg_op_min * 2.6 } else { (second_end - best_end) as f64 };
    let reg_n = (regret / pre.avg_op_min.max(1.0)).clamp(0.0, 6.0);
    let scarcity_urg = 1.0 / (best_cnt_total as f64).max(1.0);
    let density_n = ((rem_min / (ops_rem as f64).max(1.0)) / pre.avg_op_min.max(1.0)).clamp(0.0, 4.0);
    let next_min = pre.product_next_min[product][op_idx] as f64; let next_min_n = next_min / pre.horizon.max(1.0);
    let next_flex_inv = pre.product_next_flex_inv[product][op_idx];
    let next_term_raw = (0.55*next_min_n + 0.45*next_flex_inv) * (1.0 + 0.30*density_n*pre.high_flex);
    let js = pre.jobshopness;
    let avg_flex_inv = 1.0 / pre.flex_avg.max(1.0); let scarce_match = scar_n * (flex_inv - avg_flex_inv);
    let mpen = js_cached_penalty(machine, machine_penalty); let mpen_gain = 1.0 + 0.85*pre.high_flex;
    let phase = progress_mode_js(progress);
    let (bn_w, end_w, phase_reg_w, load_w, density_w, next_w, flow_gain, route_gain, slack_w, proc_w, mpen_w, scarcity_w) = match phase {
        0 => (
            (0.44 + 0.26*js)*pre.bn_focus,
            0.58,
            0.28 + 0.18*js,
            if pre.hi_flex { -0.45 } else { 0.65 + 0.18*js } * pre.flex_factor,
            0.12,
            0.16*(0.70 + 0.50*js),
            1.26,
            1.26,
            0.0,
            0.16,
            (0.08 + 0.16*js)*pre.flex_factor*(0.95 + 0.45*pre.high_flex),
            0.16,
        ),
        1 => (
            (0.88 + 0.55*js)*pre.bn_focus,
            0.82,
            0.60 + 0.25*js,
            if pre.hi_flex { -0.28 } else { 0.55 + 0.24*js } * pre.flex_factor,
            0.22,
            0.23*(0.70 + 0.65*js),
            0.98,
            0.98,
            0.0,
            0.18,
            (0.12 + 0.30*js)*pre.flex_factor*(0.95 + 0.65*pre.high_flex),
            0.18,
        ),
        _ => (
            (0.60 + 0.30*js)*pre.bn_focus,
            1.10,
            0.70 + 0.30*js,
            if pre.hi_flex { -0.12 } else { 0.18 + 0.08*js } * pre.flex_factor,
            0.16,
            0.18*(0.55 + 0.55*js),
            0.70,
            0.72,
            pre.slack_base*(0.55 + 0.50*js),
            0.20,
            (0.14 + 0.34*js)*pre.flex_factor*(1.00 + 0.75*pre.high_flex),
            0.20,
        ),
    };
    let flow_term = pre.flow_w * pre.job_flow_pref[job] * flow_gain;
    let slack_term = if slack_w > 0.0 {
        let slack_u = slack_urgency_js(pre, target_mk, time, product, op_idx);
        slack_w * slack_u * (1.0 + 0.45*reg_n)
    } else { 0.0 };
    let pop_pen = if pre.chaotic_like && op.flex >= 2 {
        let pop = js_cached_pop(pre, machine);
        let ppw = match phase { 0 => 0.20, 1 => 0.14, _ => 0.09 };
        ppw * pop * pre.flex_factor
    } else { 0.0 };
    let route_term = if route_w > 0.0 && op.flex >= 2 { route_w*route_gain*route_pref_bonus_js(route_pref, product, op_idx, machine) } else { 0.0 };
    let next_term = next_w * next_term_raw;
    (0.94*rem_min_n)+(0.30*rem_avg_n)+(bn_w*bn_n)+(density_w*density_n)+(0.10*ops_n)+(0.64*pre.flex_factor)*flex_inv+(0.34*pre.flex_factor)*scarce_match+load_w*load_n+(phase_reg_w*pre.flex_factor)*reg_n+scarcity_w*scarcity_urg+next_term+slack_term-end_w*end_n-proc_w*proc_n-pop_pen-(mpen_gain*mpen_w)*mpen+0.60*job_bias+flow_term+route_term+jitter
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn score_candidate_adaptive(
    pre: &Pre, job: usize, product: usize, op_idx: usize,
    ops_rem: usize, op: &OpInfo, machine: usize, pt: u32, time: u32,
    target_mk: Option<u32>, best_end: u32, second_end: u32, best_cnt_total: usize,
    progress: f64, job_bias: f64, machine_penalty: f64, dynamic_load: f64,
    route_pref: Option<&RoutePrefLite>, route_w: f64, jitter: f64,
) -> f64 {
    let rem_min = pre.product_suf_min[product][op_idx] as f64;
    let rem_avg = pre.product_suf_avg[product][op_idx];
    let rem_bn = pre.product_suf_bn[product][op_idx];
    let flex_f = (op.flex as f64).max(1.0); let flex_inv = 1.0 / flex_f;
    let rem_min_n = rem_min / pre.horizon.max(1.0); let rem_avg_n = rem_avg / pre.max_job_avg_work.max(1e-9);
    let bn_n = rem_bn / pre.max_job_bn.max(1e-9); let ops_n = (ops_rem as f64) / (pre.max_ops as f64).max(1.0);
    let load_n = dynamic_load / pre.avg_machine_load.max(1e-9); let scar_n = js_cached_scar_n(pre, machine);
    let end_n = (best_end as f64) / pre.time_scale.max(1.0); let proc_n = (pt as f64) / pre.avg_op_min.max(1.0);
    let regret = if second_end >= INF { pre.avg_op_min * 2.6 } else { (second_end - best_end) as f64 };
    let reg_n = (regret / pre.avg_op_min.max(1.0)).clamp(0.0, 6.0);
    let scarcity_urg = 1.0 / (best_cnt_total as f64).max(1.0);
    let density_n = ((rem_min / (ops_rem as f64).max(1.0)) / pre.avg_op_min.max(1.0)).clamp(0.0, 4.0);
    let next_min = pre.product_next_min[product][op_idx] as f64; let next_min_n = next_min / pre.horizon.max(1.0);
    let next_flex_inv = pre.product_next_flex_inv[product][op_idx];
    let next_term_raw = (0.55*next_min_n + 0.45*next_flex_inv) * (1.0 + 0.30*density_n*pre.high_flex);
    let js = pre.jobshopness; let fl = 1.0 - js;
    let avg_flex_inv = 1.0 / pre.flex_avg.max(1.0); let scarce_match = scar_n * (flex_inv - avg_flex_inv);
    let mpen = js_cached_penalty(machine, machine_penalty); let mpen_gain = 1.0 + 0.85*pre.high_flex;
    let phase = progress_mode_js(progress);
    let load_sign = if pre.hi_flex { -1.0 } else { 1.0 };
    let (bn_w, end_w, phase_reg_w, load_w, density_w, next_w, flow_gain, route_gain, slack_w, proc_w, mpen_w, scarcity_w) = match phase {
        0 => (
            (0.20 + 0.18*js)*pre.bn_focus,
            0.86*fl + 0.72*js + 0.12*pre.high_flex,
            0.24*fl + 0.46*js,
            load_sign*(0.60*fl + 0.92*js)*pre.flex_factor,
            0.05*fl + 0.12*js,
            0.16*(0.65*fl + 1.20*js),
            1.28,
            1.28,
            0.0,
            0.18*fl + 0.14*js,
            (0.05*fl + 0.16*js)*pre.flex_factor*(1.0 + 0.70*pre.high_flex),
            0.12*pre.flex_factor,
        ),
        1 => (
            (0.48 + 0.42*js)*pre.bn_focus,
            1.00*fl + 0.92*js + 0.14*pre.high_flex,
            0.48*fl + 0.78*js,
            load_sign*(0.42*fl + 0.72*js)*pre.flex_factor,
            0.08*fl + 0.20*js,
            0.23*(0.55*fl + 1.50*js),
            0.98,
            0.98,
            0.0,
            0.16*fl + 0.12*js,
            (0.08*fl + 0.28*js)*pre.flex_factor*(1.0 + 0.85*pre.high_flex),
            0.18*pre.flex_factor,
        ),
        _ => (
            (0.34 + 0.30*js)*pre.bn_focus,
            1.22 + 0.28*js + 0.18*pre.high_flex,
            0.72*fl + 0.96*js,
            load_sign*(0.18*fl + 0.32*js)*pre.flex_factor,
            0.04*fl + 0.16*js,
            0.18*(0.40*fl + 1.25*js),
            0.72,
            0.72,
            pre.slack_base*(0.95 + 0.30*js),
            0.12*fl + 0.10*js,
            (0.10*fl + 0.34*js)*pre.flex_factor*(1.0 + 0.90*pre.high_flex),
            0.24*pre.flex_factor,
        ),
    };
    let flow_term = pre.flow_w * pre.job_flow_pref[job] * flow_gain;
    let slack_term = if slack_w > 0.0 {
        let slack_u = slack_urgency_js(pre, target_mk, time, product, op_idx);
        slack_w * slack_u * (1.0 + 0.55*reg_n)
    } else { 0.0 };
    let pop_pen = if pre.chaotic_like && op.flex >= 2 {
        let pop = js_cached_pop(pre, machine);
        let ppw = match phase { 0 => 0.20, 1 => 0.13, _ => 0.08 };
        ppw * pop * pre.flex_factor
    } else { 0.0 };
    let route_term = if route_w > 0.0 && op.flex >= 2 { route_w*route_gain*route_pref_bonus_js(route_pref, product, op_idx, machine) } else { 0.0 };
    let next_term = next_w * next_term_raw;
    (1.03*rem_min_n)+(0.48*rem_avg_n)+(bn_w*bn_n)+density_w*density_n+(0.08*ops_n)+(0.60*pre.flex_factor)*flex_inv+(0.50*pre.flex_factor)*scarce_match+load_w*load_n+(phase_reg_w*pre.flex_factor)*reg_n+scarcity_w*scarcity_urg+next_term+slack_term-end_w*end_n-proc_w*proc_n-pop_pen-(mpen_gain*mpen_w)*mpen+(0.60+0.06*js)*job_bias+flow_term+route_term+jitter
}

#[allow(clippy::too_many_arguments)]
#[inline]
fn score_candidate_flex_balance(
    pre: &Pre, job: usize, product: usize, op_idx: usize,
    ops_rem: usize, op: &OpInfo, machine: usize, pt: u32, time: u32,
    target_mk: Option<u32>, best_end: u32, second_end: u32, best_cnt_total: usize,
    progress: f64, job_bias: f64, machine_penalty: f64, dynamic_load: f64,
    route_pref: Option<&RoutePrefLite>, route_w: f64, jitter: f64,
) -> f64 {
    let rem_min = pre.product_suf_min[product][op_idx] as f64;
    let rem_avg = pre.product_suf_avg[product][op_idx];
    let rem_min_n = rem_min / pre.horizon.max(1.0); let rem_avg_n = rem_avg / pre.max_job_avg_work.max(1e-9);
    let ops_n = (ops_rem as f64) / (pre.max_ops as f64).max(1.0);
    let load_n = dynamic_load / pre.avg_machine_load.max(1e-9);
    let end_n = (best_end as f64) / pre.time_scale.max(1.0); let proc_n = (pt as f64) / pre.avg_op_min.max(1.0);
    let regret = if second_end >= INF { pre.avg_op_min * 2.6 } else { (second_end - best_end) as f64 };
    let reg_n = (regret / pre.avg_op_min.max(1.0)).clamp(0.0, 6.0);
    let scarcity_urg = 1.0 / (best_cnt_total as f64).max(1.0);
    let density_n = ((rem_min / (ops_rem as f64).max(1.0)) / pre.avg_op_min.max(1.0)).clamp(0.0, 4.0);
    let next_min = pre.product_next_min[product][op_idx] as f64; let next_min_n = next_min / pre.horizon.max(1.0);
    let next_flex_inv = pre.product_next_flex_inv[product][op_idx];
    let p2 = progress * progress; let next_w_base = 0.12 + p2 * 0.28;
    let next_term_raw = (0.55*next_min_n + 0.45*next_flex_inv) * (1.0 + 0.30*density_n*pre.high_flex);
    let js = pre.jobshopness;
    let mpen = machine_penalty.clamp(0.0, 1.0);
    let flow_term = pre.flow_w * pre.job_flow_pref[job] * (0.65 + 0.70*(1.0-progress));
    let slack_u = slack_urgency_js(pre, target_mk, time, product, op_idx);
    let slack_w = pre.slack_base * (0.25 + 0.75*progress);
    let pop_pen = if pre.chaotic_like && op.flex >= 2 { let pop = js_cached_pop(pre, machine); (0.07+0.15*(1.0-progress)).clamp(0.05,0.24)*pop*pre.flex_factor } else { 0.0 };
    let route_gain = (0.70 + 0.80*(1.0-progress)).clamp(0.70, 1.40);
    let route_term = if route_w > 0.0 && op.flex >= 2 { route_w*route_gain*route_pref_bonus_js(route_pref, product, op_idx, machine) } else { 0.0 };
    let end_w=(0.85+0.70*progress+0.15*js).clamp(0.85,1.75); let cp_w=(1.00+0.30*js+0.15*(1.0-progress)).clamp(0.95,1.45); let load_w=(0.55+0.35*pre.high_flex).clamp(0.55,0.95)*pre.flex_factor; let mpen_w=(0.55+0.65*pre.high_flex).clamp(0.55,1.15); let reg_w=(0.35+0.25*(1.0-progress)).clamp(0.35,0.70); let next_term=next_w_base*0.40*next_term_raw;
    (cp_w*rem_min_n)+0.55*rem_avg_n+0.08*ops_n+0.06*density_n+0.08*scarcity_urg+next_term+(0.70*slack_w)*slack_u-end_w*end_n-0.16*proc_n-pop_pen-load_w*load_n-(mpen_w*(1.0+0.85*pre.high_flex))*mpen+(reg_w*pre.flex_factor)*reg_n+(0.58+0.10*pre.high_flex)*job_bias+flow_term+route_term+jitter
}

#[inline]
fn rule_idx(r: Rule) -> usize {
    match r { Rule::Adaptive=>0, Rule::BnHeavy=>1, Rule::EndTight=>2, Rule::CriticalPath=>3, Rule::MostWork=>4, Rule::LeastFlex=>5, Rule::Regret=>6, Rule::ShortestProc=>7, Rule::FlexBalance=>8 }
}

fn choose_rule_bandit(rng: &mut SmallRng, rules: &[Rule], rule_best: &[u32], rule_tries: &[u32], global_best: u32, margin: u32, stuck: usize, chaos_like: bool, late_phase: bool) -> Rule {
    if rules.is_empty() { return Rule::Adaptive; }
    let mut best_seen = global_best; for &mk in rule_best { if mk < best_seen { best_seen = mk; } }
    let scale = (margin as f64).max(1.0); let s = ((stuck as f64)/140.0).clamp(0.0,1.0); let explore_mix = (0.10+0.55*s).clamp(0.10,0.65);
    let mut w = [0.0f64; 9];
    for (i, &r) in rules.iter().enumerate() {
        let idx = rule_idx(r);
        let mk = rule_best[idx]; let t = rule_tries[idx].max(1) as f64;
        let delta = mk.saturating_sub(best_seen) as f64; let exploit = (-delta/scale).exp(); let explore = (1.0/t).sqrt();
        let mut ww = (1.0-explore_mix)*exploit+explore_mix*explore; ww = ww.max(1e-6);
        if chaos_like { ww = ww.powf(0.70); } else if late_phase { ww = ww.powf(1.18); }
        w[i] = ww;
    }
    let mut sum = 0.0; for i in 0..rules.len() { sum += w[i].max(0.0); }
    if !(sum > 0.0) { return rules[rng.gen_range(0..rules.len())]; }
    let mut r = rng.gen::<f64>() * sum;
    for i in 0..rules.len() { r -= w[i].max(0.0); if r <= 0.0 { return rules[i]; } }
    rules[rules.len()-1]
}

#[inline]
fn choose_from_top_weighted_temp(rng: &mut SmallRng, top: &[Cand], temperature: f64) -> Cand {
    if top.len() <= 1 { return top[0]; }
    let flat = temperature.clamp(0.0, 1.0);
    let sharp = 1.0 - flat;
    let n = top.len();
    let mut sum = 0.0;
    for i in 0..n {
        sum += flat + sharp * ((n - i) as f64);
    }
    let mut r = rng.gen::<f64>() * sum;
    for i in 0..n {
        r -= flat + sharp * ((n - i) as f64);
        if r <= 0.0 { return top[i]; }
    }
    top[n - 1]
}

#[inline]
fn score_candidate_specialized<const RULE: u8>(
    pre: &Pre, job: usize, product: usize, op_idx: usize,
    ops_rem: usize, op: &OpInfo, machine: usize, pt: u32, time: u32,
    target_mk: Option<u32>, best_end: u32, second_end: u32, best_cnt_total: usize,
    progress: f64, job_bias: f64, machine_penalty: f64, dynamic_load: f64,
    route_pref: Option<&RoutePrefLite>, route_w: f64, jitter: f64,
) -> f64 {
    if RULE == 0 {
        score_candidate_adaptive(pre, job, product, op_idx, ops_rem, op, machine, pt, time, target_mk, best_end, second_end, best_cnt_total, progress, job_bias, machine_penalty, dynamic_load, route_pref, route_w, jitter)
    } else if RULE == 1 {
        score_candidate_bn_heavy(pre, job, product, op_idx, ops_rem, op, machine, pt, time, target_mk, best_end, second_end, best_cnt_total, progress, job_bias, machine_penalty, dynamic_load, route_pref, route_w, jitter)
    } else if RULE == 2 {
        score_candidate_end_tight(pre, job, product, op_idx, ops_rem, op, machine, pt, time, target_mk, best_end, second_end, best_cnt_total, progress, job_bias, machine_penalty, dynamic_load, route_pref, route_w, jitter)
    } else if RULE == 3 {
        score_candidate_critical_path(pre, job, product, op_idx, ops_rem, op, machine, pt, time, target_mk, best_end, second_end, best_cnt_total, progress, job_bias, machine_penalty, dynamic_load, route_pref, route_w, jitter)
    } else if RULE == 4 {
        score_candidate_most_work(pre, job, product, op_idx, ops_rem, op, machine, pt, time, target_mk, best_end, second_end, best_cnt_total, progress, job_bias, machine_penalty, dynamic_load, route_pref, route_w, jitter)
    } else if RULE == 5 {
        score_candidate_least_flex(pre, job, product, op_idx, ops_rem, op, machine, pt, time, target_mk, best_end, second_end, best_cnt_total, progress, job_bias, machine_penalty, dynamic_load, route_pref, route_w, jitter)
    } else if RULE == 6 {
        score_candidate_regret(pre, job, product, op_idx, ops_rem, op, machine, pt, time, target_mk, best_end, second_end, best_cnt_total, progress, job_bias, machine_penalty, dynamic_load, route_pref, route_w, jitter)
    } else if RULE == 7 {
        score_candidate_shortest_proc(pre, job, product, op_idx, ops_rem, op, machine, pt, time, target_mk, best_end, second_end, best_cnt_total, progress, job_bias, machine_penalty, dynamic_load, route_pref, route_w, jitter)
    } else {
        score_candidate_flex_balance(pre, job, product, op_idx, ops_rem, op, machine, pt, time, target_mk, best_end, second_end, best_cnt_total, progress, job_bias, machine_penalty, dynamic_load, route_pref, route_w, jitter)
    }
}

#[inline]
fn construct_solution_conflict_dispatch<const USE_JB: bool, const USE_MP: bool, const USE_ROUTE: bool, const CHAOTIC: bool>(
    challenge: &Challenge, pre: &Pre, rule: Rule, k: usize, target_mk: Option<u32>,
    rng: &mut SmallRng, job_bias: Option<&[f64]>, machine_penalty: Option<&[f64]>,
    route_pref: Option<&RoutePrefLite>, route_w: f64,
) -> Result<(Solution, u32)> {
    match rule {
        Rule::Adaptive => construct_solution_conflict_impl::<0, USE_JB, USE_MP, USE_ROUTE, CHAOTIC>(challenge, pre, k, target_mk, rng, job_bias, machine_penalty, route_pref, route_w),
        Rule::BnHeavy => construct_solution_conflict_impl::<1, USE_JB, USE_MP, USE_ROUTE, CHAOTIC>(challenge, pre, k, target_mk, rng, job_bias, machine_penalty, route_pref, route_w),
        Rule::EndTight => construct_solution_conflict_impl::<2, USE_JB, USE_MP, USE_ROUTE, CHAOTIC>(challenge, pre, k, target_mk, rng, job_bias, machine_penalty, route_pref, route_w),
        Rule::CriticalPath => construct_solution_conflict_impl::<3, USE_JB, USE_MP, USE_ROUTE, CHAOTIC>(challenge, pre, k, target_mk, rng, job_bias, machine_penalty, route_pref, route_w),
        Rule::MostWork => construct_solution_conflict_impl::<4, USE_JB, USE_MP, USE_ROUTE, CHAOTIC>(challenge, pre, k, target_mk, rng, job_bias, machine_penalty, route_pref, route_w),
        Rule::LeastFlex => construct_solution_conflict_impl::<5, USE_JB, USE_MP, USE_ROUTE, CHAOTIC>(challenge, pre, k, target_mk, rng, job_bias, machine_penalty, route_pref, route_w),
        Rule::Regret => construct_solution_conflict_impl::<6, USE_JB, USE_MP, USE_ROUTE, CHAOTIC>(challenge, pre, k, target_mk, rng, job_bias, machine_penalty, route_pref, route_w),
        Rule::ShortestProc => construct_solution_conflict_impl::<7, USE_JB, USE_MP, USE_ROUTE, CHAOTIC>(challenge, pre, k, target_mk, rng, job_bias, machine_penalty, route_pref, route_w),
        Rule::FlexBalance => construct_solution_conflict_impl::<8, USE_JB, USE_MP, USE_ROUTE, CHAOTIC>(challenge, pre, k, target_mk, rng, job_bias, machine_penalty, route_pref, route_w),
    }
}

fn construct_solution_conflict(
    challenge: &Challenge, pre: &Pre, rule: Rule, k: usize, target_mk: Option<u32>,
    rng: &mut SmallRng, job_bias: Option<&[f64]>, machine_penalty: Option<&[f64]>,
    route_pref: Option<&RoutePrefLite>, route_w: f64,
) -> Result<(Solution, u32)> {
    let use_jb = job_bias.is_some();
    let use_mp = machine_penalty.is_some();
    let use_route = route_pref.is_some() && route_w > 0.0;
    match (pre.chaotic_like, use_jb, use_mp, use_route) {
        (false, false, false, false) => construct_solution_conflict_dispatch::<false, false, false, false>(challenge, pre, rule, k, target_mk, rng, job_bias, machine_penalty, route_pref, route_w),
        (false, false, false, true) => construct_solution_conflict_dispatch::<false, false, true, false>(challenge, pre, rule, k, target_mk, rng, job_bias, machine_penalty, route_pref, route_w),
        (false, false, true, false) => construct_solution_conflict_dispatch::<false, true, false, false>(challenge, pre, rule, k, target_mk, rng, job_bias, machine_penalty, route_pref, route_w),
        (false, false, true, true) => construct_solution_conflict_dispatch::<false, true, true, false>(challenge, pre, rule, k, target_mk, rng, job_bias, machine_penalty, route_pref, route_w),
        (false, true, false, false) => construct_solution_conflict_dispatch::<true, false, false, false>(challenge, pre, rule, k, target_mk, rng, job_bias, machine_penalty, route_pref, route_w),
        (false, true, false, true) => construct_solution_conflict_dispatch::<true, false, true, false>(challenge, pre, rule, k, target_mk, rng, job_bias, machine_penalty, route_pref, route_w),
        (false, true, true, false) => construct_solution_conflict_dispatch::<true, true, false, false>(challenge, pre, rule, k, target_mk, rng, job_bias, machine_penalty, route_pref, route_w),
        (false, true, true, true) => construct_solution_conflict_dispatch::<true, true, true, false>(challenge, pre, rule, k, target_mk, rng, job_bias, machine_penalty, route_pref, route_w),
        (true, false, false, false) => construct_solution_conflict_dispatch::<false, false, false, true>(challenge, pre, rule, k, target_mk, rng, job_bias, machine_penalty, route_pref, route_w),
        (true, false, false, true) => construct_solution_conflict_dispatch::<false, false, true, true>(challenge, pre, rule, k, target_mk, rng, job_bias, machine_penalty, route_pref, route_w),
        (true, false, true, false) => construct_solution_conflict_dispatch::<false, true, false, true>(challenge, pre, rule, k, target_mk, rng, job_bias, machine_penalty, route_pref, route_w),
        (true, false, true, true) => construct_solution_conflict_dispatch::<false, true, true, true>(challenge, pre, rule, k, target_mk, rng, job_bias, machine_penalty, route_pref, route_w),
        (true, true, false, false) => construct_solution_conflict_dispatch::<true, false, false, true>(challenge, pre, rule, k, target_mk, rng, job_bias, machine_penalty, route_pref, route_w),
        (true, true, false, true) => construct_solution_conflict_dispatch::<true, false, true, true>(challenge, pre, rule, k, target_mk, rng, job_bias, machine_penalty, route_pref, route_w),
        (true, true, true, false) => construct_solution_conflict_dispatch::<true, true, false, true>(challenge, pre, rule, k, target_mk, rng, job_bias, machine_penalty, route_pref, route_w),
        (true, true, true, true) => construct_solution_conflict_dispatch::<true, true, true, true>(challenge, pre, rule, k, target_mk, rng, job_bias, machine_penalty, route_pref, route_w),
    }
}

#[inline]
fn construct_solution_conflict_impl<const RULE: u8, const USE_JB: bool, const USE_MP: bool, const USE_ROUTE: bool, const CHAOTIC: bool>(
    challenge: &Challenge, pre: &Pre, k: usize, target_mk: Option<u32>,
    rng: &mut SmallRng, job_bias: Option<&[f64]>, machine_penalty: Option<&[f64]>,
    route_pref: Option<&RoutePrefLite>, route_w: f64,
) -> Result<(Solution, u32)> {
    let num_jobs = challenge.num_jobs; let num_machines = challenge.num_machines;
    let mut job_next_op = vec![0usize; num_jobs]; let mut job_ready_time = vec![0u32; num_jobs];
    let mut machine_avail = vec![0u32; num_machines]; let mut machine_load = pre.machine_load0.clone();
    let mut job_schedule: Vec<Vec<(usize, u32)>> = pre.job_ops_len.iter().map(|&len| Vec::with_capacity(len)).collect();
    let mut remaining_ops = pre.total_ops; let mut time = 0u32;
    let mut demand: Vec<u16> = vec![0u16; num_machines];
    let mut raw_by_machine: Vec<Vec<RawCand>> = (0..num_machines).map(|_| Vec::with_capacity(12)).collect();
    let mut idle_machines: Vec<usize> = (0..num_machines).collect();
    let mut idle_pos: Vec<usize> = (0..num_machines).collect();
    let mut busy_machine_heap: std::collections::BinaryHeap<std::cmp::Reverse<(u32, usize)>> = std::collections::BinaryHeap::with_capacity(num_machines);
    let mut blocked_job_heap: std::collections::BinaryHeap<std::cmp::Reverse<(u32, usize)>> = std::collections::BinaryHeap::with_capacity(num_jobs);
    let chaotic_like = CHAOTIC;
    let mut machine_work: Vec<u64> = if chaotic_like { vec![0u64; num_machines] } else { vec![] };
    let mut sum_work: u64 = 0;
    let mut touched_machines: Vec<usize> = Vec::with_capacity(num_machines);
    let mut round_stamp: u32 = 1;
    let mut top: Vec<Cand> = if k > 0 { Vec::with_capacity(k) } else { Vec::new() };
    let mut ready_by_machine: Vec<Vec<(usize, u32, u32)>> = (0..num_machines).map(|_| Vec::with_capacity(32)).collect();
    let mut job_gen: Vec<u32> = vec![0u32; num_jobs];
    let mut job_eval_stamp: Vec<u32> = vec![0u32; num_jobs];
    let mut job_best_end: Vec<u32> = vec![INF; num_jobs];
    let mut job_second_end: Vec<u32> = vec![INF; num_jobs];
    let mut job_best_cnt_total: Vec<usize> = vec![0usize; num_jobs];
    let mut job_best_cnt_idle: Vec<usize> = vec![0usize; num_jobs];
    let mut job_rigidity: Vec<f64> = vec![0.0; num_jobs];
    let mut job_regn: Vec<f64> = vec![0.0; num_jobs];
    let mut job_product_cur: Vec<usize> = vec![0usize; num_jobs];
    let mut job_ops_rem: Vec<usize> = vec![0usize; num_jobs];
    let mut job_op_ptr: Vec<*const OpInfo> = vec![std::ptr::null(); num_jobs];
    let mut job_bias_cur: Vec<f64> = if USE_JB { vec![0.0; num_jobs] } else { Vec::new() };
    let job_bias = if USE_JB { unsafe { job_bias.unwrap_unchecked() } } else { &[][..] };
    let machine_penalty = if USE_MP { unsafe { machine_penalty.unwrap_unchecked() } } else { &[][..] };
    let route_pref = if USE_ROUTE { Some(unsafe { route_pref.unwrap_unchecked() }) } else { None };
    let route_w = if USE_ROUTE { route_w } else { 0.0 };
    let machine_scar_n: Vec<f64> = if RULE == 0 || RULE == 1 || RULE == 2 {
        let denom = pre.avg_machine_scarcity.max(1e-9);
        pre.machine_scarcity.iter().map(|&v| v / denom).collect()
    } else {
        Vec::new()
    };
    let machine_pop: &[f64] = if CHAOTIC { &pre.machine_best_pop[..] } else { &[][..] };
    let machine_penalty_clamped: Vec<f64> = if USE_MP {
        machine_penalty.iter().map(|&v| v.clamp(0.0, 1.0)).collect()
    } else {
        Vec::new()
    };
    let machine_u8: Vec<u8> = if USE_ROUTE {
        (0..num_machines).map(|m| m.min(255) as u8).collect()
    } else {
        Vec::new()
    };
    unsafe { set_js_machine_static_cache(&machine_scar_n, machine_pop, &machine_penalty_clamped, &machine_u8); }
    let _machine_static_cache_guard = JsMachineStaticCacheGuard;
    for job in 0..num_jobs {
        if pre.job_ops_len[job] == 0 { continue; }
        let product = pre.job_products[job];
        let op = &pre.product_ops[product][0];
        if op.flex == 0 || op.machines.is_empty() || op.min_pt >= INF { continue; }
        let gen = job_gen[job];
        for &(m, pt) in &op.machines {
            ready_by_machine[m].push((job, gen, pt));
        }
    }
    while remaining_ops > 0 {
        loop {
            while let Some(entry) = busy_machine_heap.peek() {
                let std::cmp::Reverse((t, m)) = *entry;
                if t > time { break; }
                busy_machine_heap.pop();
                if machine_avail[m] == t && idle_pos[m] == NONE_USIZE {
                    idle_pos[m] = idle_machines.len();
                    idle_machines.push(m);
                }
            }
            while let Some(entry) = blocked_job_heap.peek() {
                let std::cmp::Reverse((t, j)) = *entry;
                if t > time { break; }
                blocked_job_heap.pop();
                if job_next_op[j] >= pre.job_ops_len[j] || job_ready_time[j] != t { continue; }
                let product = pre.job_products[j];
                let op_idx = job_next_op[j];
                let op = &pre.product_ops[product][op_idx];
                if op.flex == 0 || op.machines.is_empty() || op.min_pt >= INF { continue; }
                let gen = job_gen[j];
                for &(m, pt) in &op.machines {
                    ready_by_machine[m].push((j, gen, pt));
                }
            }
            if idle_machines.is_empty() { break; }
            let cur_stamp = round_stamp;
            round_stamp = round_stamp.wrapping_add(1);
            if round_stamp == 0 { job_eval_stamp.fill(0); round_stamp = 1; }
            touched_machines.clear();
            let progress = 1.0 - (remaining_ops as f64) / (pre.total_ops as f64).max(1.0);
            let cap_per_machine = if k == 0 { 12usize } else { (k+6).min(12) };
            for &m in &idle_machines {
                demand[m] = 0;
                raw_by_machine[m].clear();
                let list = &mut ready_by_machine[m];
                let mut write = 0usize;
                for read in 0..list.len() {
                    let (job, gen, pt) = list[read];
                    if job_gen[job] != gen { continue; }
                    let op_idx = job_next_op[job];
                    if op_idx >= pre.job_ops_len[job] || job_ready_time[job] > time { continue; }
                    list[write] = (job, gen, pt);
                    write += 1;
                    if job_eval_stamp[job] != cur_stamp {
                        job_eval_stamp[job] = cur_stamp;
                        let product = pre.job_products[job];
                        job_product_cur[job] = product;
                        job_ops_rem[job] = pre.job_ops_len[job] - op_idx;
                        if USE_JB { job_bias_cur[job] = job_bias[job]; }
                        let op = &pre.product_ops[product][op_idx];
                        if op.flex == 0 || op.machines.is_empty() || op.min_pt >= INF {
                            job_op_ptr[job] = std::ptr::null();
                            job_best_end[job] = INF;
                            job_second_end[job] = INF;
                            job_best_cnt_total[job] = 0;
                            job_best_cnt_idle[job] = 0;
                        } else {
                            job_op_ptr[job] = op as *const OpInfo;
                            let (best_end, second_end, best_cnt_total, best_cnt_idle) = best_second_and_counts(time, &machine_avail, op);
                            job_best_end[job] = best_end;
                            job_second_end[job] = second_end;
                            job_best_cnt_total[job] = best_cnt_total;
                            job_best_cnt_idle[job] = best_cnt_idle;
                            if best_end < INF && best_cnt_idle > 0 {
                                let flex_inv = 1.0/(op.flex as f64).max(1.0);
                                let scarcity_urg = 1.0/(best_cnt_total as f64).max(1.0);
                                let regret = if second_end >= INF { pre.avg_op_min*2.6 } else { (second_end-best_end) as f64 };
                                job_regn[job] = (regret/pre.avg_op_min.max(1.0)).clamp(0.0,6.0);
                                job_rigidity[job] = (0.60*flex_inv+0.40*scarcity_urg).clamp(0.0,2.5);
                            }
                        }
                    }
                    if job_best_end[job] >= INF || job_best_cnt_idle[job] == 0 { continue; }
                    if time.saturating_add(pt) != job_best_end[job] { continue; }
                    if demand[m] == 0 { touched_machines.push(m); }
                    demand[m] = demand[m].saturating_add(1);
                    let product = job_product_cur[job];
                    let op = unsafe { &*job_op_ptr[job] };
                    let ops_rem = job_ops_rem[job];
                    let jb = if USE_JB { job_bias_cur[job] } else { 0.0 };
                    let tie_idle = op.flex >= 2 && job_best_cnt_idle[job] > 1;
                    let tie_frac = if tie_idle {
                        ((job_best_cnt_idle[job].saturating_sub(1) as f64) / (op.flex as f64).max(1.0)).clamp(0.0, 1.0)
                    } else { 0.0 };
                    let mp_eff = if USE_MP {
                        if tie_frac > 0.0 {
                            (machine_penalty[m] * (0.72 + 0.20*progress) + (0.10 + 0.16*(1.0-progress))*tie_frac).clamp(0.0, 1.0)
                        } else {
                            machine_penalty[m]
                        }
                    } else { 0.0 };
                    let route_w_eff = if USE_ROUTE {
                        if tie_frac > 0.0 {
                            route_w * (0.60 + 0.22*progress - 0.16*tie_frac).clamp(0.36, 0.82)
                        } else {
                            route_w
                        }
                    } else { 0.0 };
                    let jitter = if k > 0 { rng.gen::<f64>()*1e-9 } else { 0.0 };
                    let dynamic_load_m = machine_load[m];
                    let base = score_candidate_specialized::<RULE>(pre, job, product, op_idx, ops_rem, op, m, pt, time, target_mk, job_best_end[job], job_second_end[job], job_best_cnt_total[job], progress, jb, mp_eff, dynamic_load_m, route_pref, route_w_eff, jitter);
                    if raw_by_machine[m].len() < cap_per_machine || base >= raw_by_machine[m][cap_per_machine - 1].base_score {
                        push_top_k_raw(&mut raw_by_machine[m], RawCand { job, machine: m, pt, base_score: base, rigidity: job_rigidity[job], reg_n: job_regn[job] }, cap_per_machine);
                    }
                }
                list.truncate(write);
            }
            touched_machines.sort_unstable();
            let denom = (idle_machines.len() as f64).max(1.0);
            let (conflict_w, conflict_scale) = if chaotic_like { (-(0.05+0.08*(1.0-progress)).clamp(0.04,0.14), (0.95+0.20*pre.flex_factor).clamp(0.90,1.20)) } else { ((0.09+0.26*pre.jobshopness+0.11*pre.high_flex+0.16*(1.0-progress)).clamp(0.05,0.45), (0.90+0.40*pre.flex_factor).clamp(0.85,1.75)) };
            let (bal_w, avg_work) = if chaotic_like { ((0.030+0.070*(1.0-progress)).clamp(0.025,0.11), (sum_work as f64)/(num_machines as f64).max(1.0)) } else { (0.0, 0.0) };
            let mut best: Option<Cand> = None; top.clear();
            let mut demand_excess_sum = 0u32;
            let mut max_demand = 1u16;
            for &m in &touched_machines {
                let dem_u = demand[m];
                if dem_u > max_demand { max_demand = dem_u; }
                demand_excess_sum = demand_excess_sum.saturating_add(dem_u.saturating_sub(1) as u32);
                let dem = dem_u as f64; if dem <= 0.0 || raw_by_machine[m].is_empty() { continue; }
                let dem_n = ((dem-1.0)/denom).clamp(0.0,2.5);
                let bal_pen = if chaotic_like && bal_w > 0.0 { let denomw=(avg_work+(pre.avg_op_min*3.0).max(1.0)).max(1.0); let r=(machine_work[m] as f64)/denomw; let done_n=(r/(r+1.0)).clamp(0.0,1.0); -bal_w*done_n } else { 0.0 };
                for rc in &raw_by_machine[m] {
                    let rig=rc.rigidity.clamp(0.0,2.5); let regc=rc.reg_n.clamp(0.0,4.5);
                    let mut boost=conflict_w*conflict_scale*dem_n*(1.15*rig+0.85*regc);
                    if chaotic_like { boost=boost.max(-0.26); }
                    let c = Cand { job: rc.job, machine: rc.machine, pt: rc.pt, score: rc.base_score+boost+bal_pen };
                    if k == 0 {
                        if best.map_or(true, |bb| c.score > bb.score) { best = Some(c); }
                    } else if top.len() < k || c.score >= top[k - 1].score {
                        push_top_k(&mut top, c, k);
                    }
                }
            }
            let select_temp = if k == 0 || top.len() <= 1 { 0.0 } else {
                let conflict_avg = ((demand_excess_sum as f64)/denom).clamp(0.0,3.0) / 3.0;
                let conflict_peak = ((max_demand.saturating_sub(1)) as f64).clamp(0.0,4.0) / 4.0;
                ((1.0-progress) * (0.22 + 0.58*conflict_avg + 0.20*conflict_peak)).clamp(0.0,0.92)
            };
            let chosen = if k == 0 { match best { Some(c) => c, None => break } } else { if top.is_empty() { break; } choose_from_top_weighted_temp(rng, &top, select_temp) };
            let job = chosen.job; let machine = chosen.machine; let pt = chosen.pt;
            let _product = job_product_cur[job]; let _op_idx = job_next_op[job]; let op = unsafe { &*job_op_ptr[job] };
            let best_end_now = job_best_end[job];
            let end_check = time.max(machine_avail[machine]).saturating_add(pt);
            if machine_avail[machine] > time || end_check != best_end_now { break; }
            let end_time = time.saturating_add(pt);
            job_schedule[job].push((machine, time)); job_next_op[job]+=1; job_ready_time[job]=end_time; machine_avail[machine]=end_time; remaining_ops-=1;
            job_gen[job] = job_gen[job].wrapping_add(1);
            let pos = idle_pos[machine];
            if pos != NONE_USIZE {
                let last = idle_machines.pop().unwrap();
                if pos < idle_machines.len() {
                    idle_machines[pos] = last;
                    idle_pos[last] = pos;
                }
                idle_pos[machine] = NONE_USIZE;
            }
            busy_machine_heap.push(std::cmp::Reverse((end_time, machine)));
            if job_next_op[job] < pre.job_ops_len[job] {
                if end_time <= time {
                    let next_product = pre.job_products[job];
                    let next_op = &pre.product_ops[next_product][job_next_op[job]];
                    if next_op.flex > 0 && !next_op.machines.is_empty() && next_op.min_pt < INF {
                        let gen = job_gen[job];
                        for &(m, pt2) in &next_op.machines {
                            ready_by_machine[m].push((job, gen, pt2));
                        }
                    }
                } else {
                    blocked_job_heap.push(std::cmp::Reverse((end_time, job)));
                }
            }
            if chaotic_like { machine_work[machine]=machine_work[machine].saturating_add(pt as u64); sum_work=sum_work.saturating_add(pt as u64); }

            if op.min_pt < INF && op.flex > 0 && !op.machines.is_empty() { let delta=(op.min_pt as f64)/(op.flex as f64).max(1.0); if delta>0.0 { for &(mm,_) in &op.machines { let v=machine_load[mm]-delta; machine_load[mm]=if v>0.0{v}else{0.0}; } } }
            if remaining_ops == 0 { break; }
        }
        if remaining_ops == 0 { break; }
        let next_machine_time = loop {
            match busy_machine_heap.peek() {
                Some(entry) => {
                    let std::cmp::Reverse((t, m)) = *entry;
                    if machine_avail[m] != t || t <= time {
                        busy_machine_heap.pop();
                        continue;
                    }
                    break Some(t);
                }
                None => break None,
            }
        };
        let next_job_time = loop {
            match blocked_job_heap.peek() {
                Some(entry) => {
                    let std::cmp::Reverse((t, j)) = *entry;
                    if job_next_op[j] >= pre.job_ops_len[j] || job_ready_time[j] != t || t <= time {
                        blocked_job_heap.pop();
                        continue;
                    }
                    break Some(t);
                }
                None => break None,
            }
        };
        time = match (next_machine_time, next_job_time) {
            (Some(a), Some(b)) => a.min(b),
            (Some(a), None) => a,
            (None, Some(b)) => b,
            (None, None) => return Err(anyhow!("Stalled")),
        };
    }
    let mk = machine_avail.into_iter().max().unwrap_or(0);
    Ok((Solution { job_schedule }, mk))
}

#[inline]
fn rebuild_machine_pred_nodes(ds: &DisjSchedule, machine_pred_node: &mut [usize]) {
    machine_pred_node.fill(NONE_USIZE);
    for seq in &ds.machine_seq {
        for i in 1..seq.len() {
            machine_pred_node[seq[i]] = seq[i - 1];
        }
    }
}

/// `js_ts_fast_eval=1` only. Materialises the two quantities that stock `eval_disj` rebuilds from
/// `machine_seq` on EVERY call: the machine successor links, and the total in-degree
/// (`indeg_job` + one incoming machine arc for every node that is not first on its machine).
/// Both are invariant under an adjacent swap except in a 3-node window, so the tabu loop keeps
/// them up to date incrementally and only calls this on the rare structural resets (start, kick).
fn rebuild_machine_succ_indeg(ds: &DisjSchedule, machine_succ: &mut [usize], indeg_full: &mut [u16]) {
    indeg_full.copy_from_slice(&ds.indeg_job);
    for seq in &ds.machine_seq {
        if seq.is_empty() { continue; }
        let mut prev = seq[0];
        for &v in &seq[1..] {
            machine_succ[prev] = v;
            indeg_full[v] = indeg_full[v].saturating_add(1);
            prev = v;
        }
        machine_succ[prev] = NONE_USIZE;
    }
}

/// `js_ts_fast_eval=1` only. Bit-for-bit equivalent of `eval_disj`, minus the per-call rebuild of
/// `machine_succ`/`indeg` (taken from the incrementally-maintained `indeg_full`), plus the
/// recording of the topological pop order into `topo` so the backward `tail` pass can replay it in
/// reverse instead of running a second, independent topological sort.
///
/// The seeding scan (`0..n`) and the push order inside the relaxation body are IDENTICAL to
/// `eval_disj`, so the LIFO pop order — and therefore every `best_pred` tie-break — is identical.
fn eval_disj_topo(ds: &DisjSchedule, buf: &mut EvalBuf, indeg_full: &[u16], topo: &mut Vec<usize>) -> Option<(u32, usize)> {
    let n = ds.n;
    buf.indeg.copy_from_slice(indeg_full);
    buf.start.fill(0);
    buf.best_pred.fill(NONE_USIZE);
    buf.stack.clear();
    topo.clear();
    for i in 0..n {
        if buf.indeg[i] == 0 { buf.stack.push(i); }
    }
    let mut mk = 0u32;
    let mut mk_node = 0usize;
    while let Some(u) = buf.stack.pop() {
        topo.push(u);
        let end_u = buf.start[u].saturating_add(ds.node_pt[u]);
        if end_u > mk { mk = end_u; mk_node = u; }
        let js = ds.job_succ[u];
        if js != NONE_USIZE {
            if buf.start[js] < end_u {
                buf.start[js] = end_u;
                buf.best_pred[js] = u;
            }
            buf.indeg[js] = buf.indeg[js].saturating_sub(1);
            if buf.indeg[js] == 0 { buf.stack.push(js); }
        }
        let ms = buf.machine_succ[u];
        if ms != NONE_USIZE {
            if buf.start[ms] < end_u {
                buf.start[ms] = end_u;
                buf.best_pred[ms] = u;
            }
            buf.indeg[ms] = buf.indeg[ms].saturating_sub(1);
            if buf.indeg[ms] == 0 { buf.stack.push(ms); }
        }
    }
    if topo.len() != n { return None; }
    Some((mk, mk_node))
}

/// `js_ts_fast_eval=1` only. Backward longest-path (`tail`) pass, obtained by walking the forward
/// topological order backwards: every successor of a node is guaranteed to be already settled, so
/// the value is PULLED in one assignment instead of being PUSHED through a second in-degree
/// worklist. Drops the `tail.fill(0)`, the `back_deg` copy, the `machine_succ` presence scan, the
/// zero-degree seeding scan and the stack traffic of the legacy pass. Yields exactly
/// `tail[x] = max over successors y of (pt[y] + tail[y])`, i.e. the legacy values unchanged.
#[inline]
fn tails_from_topo(ds: &DisjSchedule, machine_succ: &[usize], topo: &[usize], tail: &mut [u32]) {
    for &nd in topo.iter().rev() {
        let js = ds.job_succ[nd];
        let mut t = if js != NONE_USIZE { ds.node_pt[js].saturating_add(tail[js]) } else { 0 };
        let ms = machine_succ[nd];
        if ms != NONE_USIZE {
            let c = ds.node_pt[ms].saturating_add(tail[ms]);
            if c > t { t = c; }
        }
        tail[nd] = t;
    }
}

#[inline]
fn rebuild_machine_pos_map(machine_seq: &[Vec<usize>], node_pos: &mut [usize]) {
    for seq in machine_seq {
        for (pos, &node) in seq.iter().enumerate() {
            node_pos[node] = pos;
        }
    }
}

#[inline]
fn collect_guided_kick_moves(
    ds: &DisjSchedule,
    best_pred: &[usize],
    mk_node: usize,
    current_pos: &[usize],
    node_machine: &[usize],
    ref_pos: &[usize],
    used_machine: &mut [u8],
    out: &mut Vec<(u32, usize, usize)>,
) {
    out.clear();
    used_machine.fill(0);
    let mut u = mk_node;
    while u != NONE_USIZE {
        let m = node_machine[u];
        if used_machine[m] == 0 {
            let pos = current_pos[u];
            let target = ref_pos[u];
            let seq = &ds.machine_seq[m];
            if pos > target && pos > 0 {
                let left = seq[pos - 1];
                if ref_pos[u] < ref_pos[left] {
                    out.push(((pos - target) as u32, m, pos - 1));
                    used_machine[m] = 1;
                }
            } else if pos < target && pos + 1 < seq.len() {
                let right = seq[pos + 1];
                if ref_pos[u] > ref_pos[right] {
                    out.push(((target - pos) as u32, m, pos));
                    used_machine[m] = 1;
                }
            }
        }
        u = best_pred[u];
    }
}


/// served the exact-duplicate test).
///
/// `tabu_search_phase` only ever reads the `DisjSchedule` returned by `build_disj_from_solution`,
/// i.e. the machine assignment plus the per-machine op order obtained by sorting on
/// `(start, node_id)`. The key reproduces that sort verbatim (op ids per machine, `u32::MAX` as
/// machine separator), so it captures the ONLY thing that distinguishes two job-shop seeds: the
/// disjunctive orientation. Returns an empty key for a malformed solution.
fn ts_seed_key(pre: &Pre, num_machines: usize, sol: &Solution) -> Vec<u32> {
    let num_jobs = sol.job_schedule.len();
    if num_jobs != pre.job_ops_len.len() { return Vec::new(); }
    let mut job_offsets = vec![0usize; num_jobs + 1];
    for j in 0..num_jobs { job_offsets[j + 1] = job_offsets[j] + pre.job_ops_len[j]; }
    let mut per_machine: Vec<Vec<(u32, usize)>> = vec![Vec::new(); num_machines];
    for j in 0..num_jobs {
        if sol.job_schedule[j].len() != pre.job_ops_len[j] { return Vec::new(); }
        for (k, &(m, st)) in sol.job_schedule[j].iter().enumerate() {
            if m >= num_machines { return Vec::new(); }
            per_machine[m].push((st, job_offsets[j] + k));
        }
    }
    let mut key: Vec<u32> = Vec::with_capacity(job_offsets[num_jobs] + num_machines);
    for m in 0..num_machines {
        per_machine[m].sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        for &(_, id) in &per_machine[m] { key.push(id as u32); }
        key.push(u32::MAX);
    }
    key
}

/// Positional Hamming distance between two `ts_seed_key`s.
///
/// In `job_shop` every operation has exactly ONE eligible machine (`avg_op_flexibility = 1.0`,
/// `tig-challenges/job_scheduling/scenarios.rs`), so every solution of a given nonce lays the
/// same op multiset on the same machines: the keys always have the same length and the same
/// separator positions, and they differ only where the machine ORDER differs. The distance is
/// therefore exactly the number of ops sitting at a different rank in their machine queue.
#[inline]
fn ts_seed_key_distance(a: &[u32], b: &[u32]) -> u32 {
    // A malformed / mismatched key scores 0 (= "no distance to offer") so the greedy below never
    // PREFERS it; if it still gets picked, `build_disj_from_solution` rejects it exactly as the
    // baseline would.
    if a.is_empty() || b.is_empty() || a.len() != b.len() { return 0; }
    let mut d = 0u32;
    for i in 0..a.len() {
        if a[i] != b[i] { d += 1; }
    }
    d
}


/// iterations, of this tabu run. `0` reproduces the legacy budget-proportional law bit-for-bit; any
/// other value decouples the cadence of the intensification/diversification machinery from
/// `max_iterations` while preserving the total amount of work it is allowed to burn.

/// legacy stream, bit-for-bit. See `EffortConfig::js_ts_traj_decorr`.

/// disables the acceptance test entirely and the loop below is the legacy one, bit-for-bit. Only
/// the fenced extra draws of `solve` ever pass a non-zero value.
///

///
/// This search has THREE in-loop points where the disjunctive graph it just modified fails to
/// evaluate (`:8372` restore-from-best after a stagnation plateau, `:8436` after a guided kick,
/// `:8693` after applying the chosen move). All three react the same way: `return Ok(None)`, which
/// to the CALLER means "this run found nothing" — so `best_global_mk` and `best_global_machine_seq`,
/// i.e. everything the run had already earned and had already certified with `eval_disj`, are
/// thrown on the floor.
///
/// That is not a hypothetical. The full-budget offline attribution probe (`jsi=200000`, harness
/// `/tmp/jssattr`) counted the aborts on the two nonces that FIX the median: **2 per nonce on BOTH
/// nonce 30 and nonce 5**, and on nonce 30 one of the two IS the final polish (`polish_try=1`,
/// `polish_None=1`, and that call never reaches the end of this function). The same probe measured
/// what the polish is worth when it does survive: on nonce 5 it takes the champion from **10 895 to
/// 10 831, −64 makespan units ≈ −4 900 Q**, more than the entire 8-start portfolio's marginal value
/// (the min over all 8 starts equals start #0 alone). So on nonce 30 the single most productive
/// phase of the solver is being silently discarded mid-flight.
///
/// `mid_abort_rescue > 0` replaces the three `return Ok(None)` with `break`, which falls through to
/// the normal exit path. That path is already the safe one: it re-evaluates
/// `best_global_machine_seq` — a clone stored SEPARATELY from the `ds.machine_seq` that just failed
/// — with the LEGACY `eval_disj` (`:8702`), and still returns `None` if that fails or if
/// `best_global_mk >= initial_mk`. No feasibility tolerance is introduced or widened anywhere; the
/// returned solution is certified by exactly the evaluator every other phase is judged by.
///
/// The stale-`cur_mk` write at `:8699` cannot fire on this path: all three abort sites sit AFTER
/// the `if iter > 0 { if cur_mk < best_global_mk { .. } }` fold of the current iteration (`:8440`),
/// or before it with a `cur_mk` already folded by the previous one, so `cur_mk >= best_global_mk`
/// holds at every abort and the broken `ds.machine_seq` is never cloned into the best.
///
/// `mid_abort_rescue == 0` is the legacy expression at all three sites, so the default binary is
/// byte-identical by construction rather than by inspection. Only the final polish of `solve` ever
/// passes a non-zero value — every other call site receives a literal `0` — which is what makes
/// this iteration monotone on Q per nonce (see the call site for the fence argument).
fn tabu_search_phase(pre: &Pre, challenge: &Challenge, base_sol: &Solution, max_iterations: usize, tenure_base: usize, fast_eval: bool, div_period: usize, tie_mode: usize, sel_key: usize, exact_topk: usize, traj_salt: u64, kick_spread: usize, sa_accept: usize, mid_abort_rescue: usize) -> Result<Option<(Solution, u32)>> {
    let mut ds = build_disj_from_solution(pre, challenge, base_sol)?;
    let mut buf = EvalBuf::new(ds.n); let n = ds.n;
    let Some((initial_mk, mut mk_node)) = eval_disj(&ds, &mut buf) else { return Ok(None) };
    let mut cur_mk = initial_mk; let mut best_global_mk = initial_mk; let mut best_global_machine_seq = ds.machine_seq.clone();
    let tenure = tenure_base.max(5); let tenure_delta = (tenure/3).max(2);
    
    // it is stuck used to be `max_iterations/2` — a FRACTION of the budget rather than a property
    // of the landscape. `div_period > 0` makes it an absolute count of non-improving iterations;
    // `div_period == 0` is the legacy expression, unchanged, so the default binary is byte-identical.
    let max_no_improve = if div_period > 0 { div_period.max(60) } else { (max_iterations/2).max(60) };
    let mut pair_offsets = vec![0usize; ds.num_machines + 1];
    let mut node_local = vec![0usize; n];
    let mut current_pos = vec![0usize; n];
    let mut node_machine = vec![0usize; n];
    for m in 0..ds.num_machines {
        let seq = &ds.machine_seq[m];
        for (i, &node) in seq.iter().enumerate() {
            node_local[node] = i;
            current_pos[node] = i;
            node_machine[node] = m;
        }
        let len = seq.len();
        pair_offsets[m + 1] = pair_offsets[m] + len.saturating_mul(len.saturating_sub(1)) / 2;
    }
    let mut tabu_expiry = vec![0usize; pair_offsets[ds.num_machines]];
    let mut no_improve = 0usize;
    
    // gives this run a different draw of the SAME tenure law (the interval `[tenure-tenure_delta,
    // tenure+tenure_delta]` at `:8289` is untouched, so the marginal distribution of every tabu
    // tenure is identical and no candidate is ever ranked differently). `0.wrapping_mul(_) == 0`
    // and `x ^ 0 == x`, so the default path is the legacy expression, byte-for-byte, by
    // construction rather than by inspection.
    let mut pseed: u64 = (challenge.seed[0] as u64).wrapping_mul(0x9E3779B97F4A7C15) ^ (initial_mk as u64).wrapping_shl(16) ^ (n as u64).wrapping_mul(0x517CC1B727220A95) ^ traj_salt.wrapping_mul(0x2545F4914F6CDD1D);
    
    // `pseed`: draining a shared stream would shift every subsequent tabu tenure and confound the
    // tie-break with a tenure change. Seeded from the challenge seed like `pseed`, so the run stays
    // fully deterministic (no wall-clock, no thread-local RNG). `| 1` guarantees a non-zero xorshift
    // state, whose orbit would otherwise be stuck at 0.
    let mut tseed: u64 = ((challenge.seed[0] as u64).wrapping_mul(0xD1B54A32D192ED03)
        ^ (initial_mk as u64).wrapping_mul(0xA24BAED4963EE407)
        ^ (n as u64).wrapping_shl(32)) | 1;
    
    
    // shift every subsequent tenure and confound the acceptance test with a tenure change. Seeded
    // from the challenge seed and the run's own `traj_salt`, so each extra draw gets its own
    // acceptance realisation while the run stays fully deterministic (no wall-clock, no
    // thread-local RNG). `| 1` guarantees a non-zero xorshift state, whose orbit is stuck at 0.
    // Under `sa_accept == 0` this value is computed and never read, and no other stream sees it,
    // so the legacy trajectory is untouched.
    let mut aseed: u64 = ((challenge.seed[0] as u64).wrapping_mul(0x2545F4914F6CDD1D)
        ^ traj_salt.wrapping_mul(0xBF58476D1CE4E5B9)
        ^ (initial_mk as u64).wrapping_shl(24)) | 1;
    let mut tail = vec![0u32; n]; let mut back_deg = vec![0u16; n]; let mut back_stack: Vec<usize> = Vec::with_capacity(n);
    let mut machine_pred_node = vec![NONE_USIZE; n]; let mut job_pred_node = vec![NONE_USIZE; n];
    let mut job_back_deg = vec![0u16; n];
    let mut crit_pos_by_machine: Vec<Vec<usize>> = (0..ds.num_machines).map(|_| Vec::new()).collect();
    let mut crit_pos_machines: Vec<usize> = Vec::with_capacity(ds.num_machines);
    let mut machine_last_pick = vec![0usize; ds.num_machines];
    let mut best_global_pos = node_local.clone();
    let mut guided_kicks: Vec<(u32, usize, usize)> = Vec::with_capacity(ds.num_machines);
    let mut guided_used_machine = vec![0u8; ds.num_machines];
    for j in 0..ds.num_jobs { let base = ds.job_offsets[j]; let end = ds.job_offsets[j+1]; for k in (base+1)..end { job_pred_node[k] = k-1; } }
    for i in 0..n { if ds.job_succ[i] != NONE_USIZE { job_back_deg[i] += 1; } }
    rebuild_machine_pred_nodes(&ds, &mut machine_pred_node);
    // js_ts_fast_eval=1 state: total in-degree and the forward topological order, both maintained
    // across the loop so that neither has to be recomputed from `machine_seq` every iteration.
    let mut indeg_full = vec![0u16; n];
    let mut topo: Vec<usize> = Vec::with_capacity(n);
    if fast_eval {
        rebuild_machine_succ_indeg(&ds, &mut buf.machine_succ, &mut indeg_full);
        
        // an order of its own — that is the whole point of the fast kernel. The pre-loop
        // evaluation above runs the LEGACY `eval_disj`, which leaves `topo` untouched, so on
        // `iter == 0` the backward pass walked an EMPTY order and silently left `tail` at
        // all-zeros, whereas the legacy branch recomputes `tail` from scratch every iteration and
        // is therefore correct from the very first one. `estimate_swap_mk` reads `tail`, so the
        // first move of EVERY `tabu_search_phase` call diverged and dragged the whole trajectory
        
        // Seeding `topo` here restores bit-for-bit equivalence with mode 0: `eval_disj_topo`
        // reproduces exactly the `start` / `best_pred` / `mk` that `eval_disj` just computed
        // (same in-degrees, same `0..n` seeding scan, same LIFO pop order).
        if eval_disj_topo(&ds, &mut buf, &indeg_full, &mut topo).is_none() { return Ok(None) }
    }
    let kick_threshold =(max_no_improve*2/3).max(40); let diversify_threshold = (max_no_improve/3).max(20); let diversify_unit = (pre.avg_op_min * (0.10 + 0.16*pre.jobshopness + 0.06*pre.high_flex)).max(1.0) as u32;
    let mut kicks_left = 4usize;
    
    // stops this loop early, and it does NOT drain one kick per cycle: the guided-kick branch
    // `continue`s WITHOUT resetting `no_improve`, so once `no_improve == kick_threshold` it re-fires
    // every iteration until the reservoir is empty, and the jump-back then `break`s on it. The
    // legacy run therefore gives up after ONE stagnation plateau of `max_no_improve`
    // (= `max_iterations/2`) non-improving iterations.
    //
    // Making the reservoir a global multiple would be WRONG (it would all drain into the very first
    // plateau, then `break` at ~`div_period` iterations — a catastrophic loss of WORK disguised as a
    // schedule change; the `js_ts_starts=4` probe prices that loss at −2 096 Q). The reservoir is
    // instead REFILLED per cycle, and the number of cycles is capped so that a fully stagnant run
    // still gives up after `max_iterations/2` non-improving iterations — EXACTLY the legacy budget:
    //     cycles × div_period = (max_iterations / div_period / 2) × div_period = max_iterations/2.
    // Only the CADENCE changes (1 long plateau → `max_iterations/(2·div_period)` short ones), never
    // the amount of work — the sole kind of move this track accepts (`hp:ts_iters_funding` is dead).
    // `div_period == 0` ⇒ `cycles_left == 0` ⇒ the jump-back `break`s on an empty reservoir exactly
    // as before ⇒ the default binary stays byte-identical.
    let mut cycles_left = if div_period > 0 { (max_iterations / max_no_improve.max(1) / 2).max(1) } else { 0 };
    
    // Every rerank costs at most `exact_topk` extra `eval_disj_topo` calls on top of the ONE this
    // loop already spends per iteration, so leaving the trigger to the collision rate alone would
    // make the wall a function of a quantity nobody has measured. Capping the number of FIRINGS at
    // `max_iterations/16` bounds the added work at `exact_topk/16` of the evaluation budget, i.e.
    // +12,5 % at K=2 and +25 % at K=4 in the WORST case — inside the 27 % headroom the champion
    // leaves (698,59 s measured against an 865 000 ms gate), whatever the instance does. The budget
    // is per `tabu_search_phase` call, like every other counter here, and is spent on the earliest
    // collisions: those happen while the search still has the most room left to be redirected.
    let mut exact_left = if exact_topk > 0 { (max_iterations / 16).max(1) } else { 0 };
    // The tied admissible candidates of the current iteration, as (machine, position) pairs.
    // Never written nor read under `exact_topk == 0`.
    let mut top_cand = [(0usize, 0usize); 4];
    let mut top_n = 0usize;
    for iter in 0..max_iterations {
        if no_improve >= max_no_improve {
            
            // (`cycles_left == 0`) stops the run right here, unchanged. Under a compressed patience
            // the reservoir is refilled and the next cycle starts, until the cycle allowance —
            // sized so the total stagnation budget matches the legacy one — is spent.
            if kicks_left == 0 {
                if cycles_left > 0 { cycles_left -= 1; kicks_left = 4; } else { break; }
            }
            ds.machine_seq.clone_from(&best_global_machine_seq); no_improve = 0; kicks_left -= 1; tabu_expiry.fill(0);
            rebuild_machine_pred_nodes(&ds, &mut machine_pred_node);
            for m in 0..ds.num_machines {
                for (i, &node) in ds.machine_seq[m].iter().enumerate() {
                    current_pos[node] = i;
                    node_machine[node] = m;
                }
            }
            // Structural reset: `machine_seq` was replaced wholesale, so the incremental state is
            // stale and has to be rebuilt before the fast kernel can be used again.
            if fast_eval { rebuild_machine_succ_indeg(&ds, &mut buf.machine_succ, &mut indeg_full); }
            let Some((mk, node)) = (if fast_eval { eval_disj_topo(&ds, &mut buf, &indeg_full, &mut topo) } else { eval_disj(&ds, &mut buf) }) else { if mid_abort_rescue > 0 { break } else { return Ok(None) } };
            cur_mk = mk; mk_node = node;
            continue;
        }
        
        // kicks and `kick_ord` the ordinal (1..=4) of the one about to fire. Legacy semantics are
        // reproduced exactly by `kick_spread == 0`: `kick_step = kick_threshold` and
        // `kick_ord = no_improve / kick_threshold`, which is 1 for all four firings because the
        // branch `continue`s without advancing `no_improve`.
        let (kick_step, kick_ord) = if kick_spread > 0 {
            let step = (max_no_improve / kick_spread).max(1);
            (step, if step > 0 { no_improve / step } else { 0 })
        } else {
            (kick_threshold, no_improve / kick_threshold)
        };
        // Under a spread schedule the reservoir is what caps the number of firings at four, exactly
        // as in legacy; `kick_ord <= 4` only guards against a divisor so small that the plateau
        // would offer more than four multiples before `max_no_improve` is reached.
        if no_improve > 0 && no_improve % kick_step == 0 && kicks_left > 0 && (kick_spread == 0 || kick_ord <= 4) {
            // Legacy fires all four kicks at `no_improve == kick_threshold`, i.e. always with
            // `no_improve / kick_threshold == 1`. The spread schedule pins the same value so that
            // the STRENGTH (`num_kicks`) and the DIRECTION (`use_base_first`) of every kick are
            // bit-identical to the legacy ones, and the only thing this iteration changes is WHEN
            // the four kicks land.
            let kick_phase = if kick_spread > 0 { 1usize } else { kick_ord };
            let use_base_first = (kick_phase & 1) == 1;
            let ref_first = if use_base_first { &node_local[..] } else { &best_global_pos[..] };
            let ref_second = if use_base_first { &best_global_pos[..] } else { &node_local[..] };
            collect_guided_kick_moves(&ds, &buf.best_pred, mk_node, &current_pos, &node_machine, ref_first, &mut guided_used_machine, &mut guided_kicks);
            let ref_pos = if guided_kicks.is_empty() {
                collect_guided_kick_moves(&ds, &buf.best_pred, mk_node, &current_pos, &node_machine, ref_second, &mut guided_used_machine, &mut guided_kicks);
                ref_second
            } else {
                ref_first
            };
            if !guided_kicks.is_empty() {
                guided_kicks.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)).then_with(|| a.2.cmp(&b.2)));
                let num_kicks = (2 + kick_phase).min(4);
                let mut applied = 0usize;
                for &(_, m, pos) in &guided_kicks {
                    if applied >= num_kicks { break; }
                    if pos + 1 >= ds.machine_seq[m].len() { continue; }
                    let node_a = ds.machine_seq[m][pos];
                    let node_b = ds.machine_seq[m][pos + 1];
                    if ref_pos[node_a] <= ref_pos[node_b] { continue; }
                    ds.machine_seq[m].swap(pos, pos + 1);
                    current_pos[node_a] = pos + 1;
                    current_pos[node_b] = pos;
                    machine_last_pick[m] = iter;
                    applied += 1;
                }
            }
            kicks_left -= 1;
            
            // of the loop, which is precisely why the legacy predicate re-fires at the same
            // stagnation depth until the reservoir is empty. Advancing the counter by hand under a
            // spread schedule leaves the plateau exactly as long as before (the run still gives up
            // at `no_improve >= max_no_improve`) and consumes the same four loop iterations, so the
            // total amount of work is the legacy one MINUS at most four iterations out of 200 000.
            if kick_spread > 0 { no_improve += 1; }
            rebuild_machine_pred_nodes(&ds, &mut machine_pred_node);
            // The guided kick applies up to 4 arbitrary swaps (including at position 0), so the
            // incremental machine-successor / in-degree state is rebuilt rather than patched.
            if fast_eval { rebuild_machine_succ_indeg(&ds, &mut buf.machine_succ, &mut indeg_full); }
            
            // swaps (`:8487`), unconstrained by the block structure that makes an N5 move
            // acyclicity-safe, so it can hand this evaluation a graph with a cycle. The offline
            // full-budget probe caught exactly that: on nonce 30 the FINAL POLISH dies here at
            // iteration 66 667 — `kick_threshold = (max_no_improve*2/3) = 66 666`, i.e. on its very
            // FIRST kick — and the legacy reaction throws the whole 200 000-iteration polish away.
            // On nonce 5, where the polish survives its kicks, that same phase is worth
            // `10 895 -> 10 831` = −64 makespan units.
            //
            // `mid_abort_rescue >= 2` does not accept the broken graph and does not give up either:
            // it UNDOES the kick by restoring `best_global_machine_seq` — the same restore the
            // stagnation branch performs 130 lines above, on the same known-good clone — rebuilds
            // the incremental state and carries on. `kicks_left` has already been decremented, so
            // the reservoir still caps the firings at four and the loop stays bounded; if even the
            // restored sequence fails to evaluate, the run falls back to the `>= 1` behaviour.
            let ev_kick = if fast_eval { eval_disj_topo(&ds, &mut buf, &indeg_full, &mut topo) } else { eval_disj(&ds, &mut buf) };
            let Some((mk, node)) = ev_kick else {
                if mid_abort_rescue >= 2 {
                    ds.machine_seq.clone_from(&best_global_machine_seq);
                    rebuild_machine_pred_nodes(&ds, &mut machine_pred_node);
                    for m in 0..ds.num_machines {
                        for (i, &node) in ds.machine_seq[m].iter().enumerate() {
                            current_pos[node] = i;
                            node_machine[node] = m;
                        }
                    }
                    if fast_eval { rebuild_machine_succ_indeg(&ds, &mut buf.machine_succ, &mut indeg_full); }
                    match (if fast_eval { eval_disj_topo(&ds, &mut buf, &indeg_full, &mut topo) } else { eval_disj(&ds, &mut buf) }) {
                        Some((mk_r, node_r)) => { cur_mk = mk_r; mk_node = node_r; continue; }
                        None => break,
                    }
                } else if mid_abort_rescue > 0 { break } else { return Ok(None) }
            };
            cur_mk = mk; mk_node = node;
            continue;
        }
        if iter > 0 { if cur_mk < best_global_mk { best_global_mk = cur_mk; best_global_machine_seq.clone_from(&ds.machine_seq); rebuild_machine_pos_map(&best_global_machine_seq, &mut best_global_pos); no_improve = 0; } else { no_improve += 1; } }

        if fast_eval {
            tails_from_topo(&ds, &buf.machine_succ, &topo, &mut tail);
        } else {
            tail.fill(0);
            back_deg.copy_from_slice(&job_back_deg);
            for i in 0..n { if buf.machine_succ[i] != NONE_USIZE { back_deg[i] += 1; } }
            back_stack.clear(); for i in 0..n { if back_deg[i] == 0 { back_stack.push(i); } }
            while let Some(nd) = back_stack.pop() {
                let contrib = ds.node_pt[nd].saturating_add(tail[nd]);
                let jp = job_pred_node[nd]; if jp != NONE_USIZE { if contrib > tail[jp] { tail[jp] = contrib; } back_deg[jp] = back_deg[jp].saturating_sub(1); if back_deg[jp] == 0 { back_stack.push(jp); } }
                let mp = machine_pred_node[nd]; if mp != NONE_USIZE { if contrib > tail[mp] { tail[mp] = contrib; } back_deg[mp] = back_deg[mp].saturating_sub(1); if back_deg[mp] == 0 { back_stack.push(mp); } }
            }
        }
        for &m in &crit_pos_machines { crit_pos_by_machine[m].clear(); }
        crit_pos_machines.clear();
        let mut u = mk_node;
        while u != NONE_USIZE {
            let m = node_machine[u];
            if crit_pos_by_machine[m].is_empty() { crit_pos_machines.push(m); }
            crit_pos_by_machine[m].push(current_pos[u]);
            u = buf.best_pred[u];
        }
        crit_pos_machines.sort_unstable();
        let diversify = ((no_improve.saturating_sub(diversify_threshold) as f64) / (max_no_improve.saturating_sub(diversify_threshold).max(1) as f64)).clamp(0.0, 1.0);
        let mut best_move: Option<(usize,usize,u32)> = None; let mut best_move_key = u32::MAX;
        
        top_n = 0;
        let mut relaxed_move: Option<(usize,usize,u32)> = None; let mut relaxed_key = u32::MAX;
        
        // matched the incumbent key (1 = the incumbent itself), `*_tie_block` remembers its
        // critical-block length for mode 2. Both are written on every strict improvement so a
        // fresh key always restarts the count; under `tie_mode == 0` they are never read.
        let mut best_tie_n: u64 = 0; let mut best_tie_block: usize = 0;
        let mut relaxed_tie_n: u64 = 0; let mut relaxed_tie_block: usize = 0;
        
        // i.e. the shorter of the two rewritten paths. Written on every strict improvement of the
        // primary key so it always describes the CURRENT incumbent; never read under `sel_key == 0`.
        let mut best_snd: u32 = u32::MAX; let mut relaxed_snd: u32 = u32::MAX;
        for &m in &crit_pos_machines {
            let positions = &mut crit_pos_by_machine[m];
            if positions.len() < 2 { continue; }
            positions.sort_unstable();
            let seq = &ds.machine_seq[m];
            let mut run_start = positions[0];
            let mut run_end = positions[0];
            let mut prev_pos = positions[0];
            let mut prev_node = seq[prev_pos];
            for idx in 1..positions.len() {
                let pos = positions[idx];
                let node = seq[pos];
                if pos == prev_pos + 1 && buf.start[node] == buf.start[prev_node].saturating_add(ds.node_pt[prev_node]) {
                    run_end = pos;
                } else {
                    if run_end > run_start {
                        let block_len = run_end-run_start+1;
                        let mut swap_positions = [run_start,NONE_USIZE]; let num_swaps = if block_len>=3 { swap_positions[1]=run_end-1; 2 } else { 1 };
                        for si in 0..num_swaps {
                            let pos=swap_positions[si]; if pos+1>=seq.len() { continue; }
                            let node_u=seq[pos]; let node_v=seq[pos+1];
                            let (est_mk, est_snd) = estimate_swap_mk2(node_u, node_v, &buf.start, &tail, &ds.node_pt, &job_pred_node, &ds.job_succ, &machine_pred_node, &buf.machine_succ);
                            let lu = node_local[node_u]; let lv = node_local[node_v];
                            let (a, b) = if lu < lv { (lu, lv) } else { (lv, lu) };
                            let tabu_idx = pair_offsets[m] + b * (b - 1) / 2 + a;
                            let is_tabu = tabu_expiry[tabu_idx] > iter; let aspiration=est_mk<best_global_mk;
                            let age = iter.saturating_sub(machine_last_pick[m]).min(9) as u32;
                            let block_bonus = block_len.saturating_sub(2).min(3) as u32;
                            let div_bonus = if diversify > 0.0 {
                                ((diversify_unit as f64) * diversify * (0.35*(age as f64) + 0.75*(block_bonus as f64)) / 3.0) as u32
                            } else { 0 };
                            let adj_mk = est_mk.saturating_sub(div_bonus);
                            
                            // exactly (first candidate reached wins); the `else if` is short-circuited dead.
                            
                            if !is_tabu||aspiration {
                                
                                // chain below settles the incumbent. Mirrors that chain's own comparisons, so the
                                // buffer holds exactly the set the Taillard bound declares indistinguishable, in
                                // enumeration order (`top_cand[0]` is the legacy first-wins pick).
                                if exact_topk != 0 {
                                    if adj_mk<best_move_key { top_cand[0] = (m, pos); top_n = 1; }
                                    else if adj_mk==best_move_key && top_n < exact_topk { top_cand[top_n] = (m, pos); top_n += 1; }
                                }
                                if adj_mk<best_move_key { best_move_key=adj_mk; best_move=Some((m,pos,est_mk)); best_tie_n=1; best_tie_block=block_len; best_snd=est_snd; }
                                else if sel_key!=0 && adj_mk==best_move_key { if sel_key_take(sel_key, est_snd, &mut best_snd) { best_move=Some((m,pos,est_mk)); } }
                                else if tie_mode!=0 && adj_mk==best_move_key && tie_break_take(tie_mode, block_len, &mut best_tie_n, &mut best_tie_block, &mut tseed) { best_move=Some((m,pos,est_mk)); }
                            }
                            if adj_mk<relaxed_key { relaxed_key=adj_mk; relaxed_move=Some((m,pos,est_mk)); relaxed_tie_n=1; relaxed_tie_block=block_len; relaxed_snd=est_snd; }
                            else if sel_key!=0 && adj_mk==relaxed_key { if sel_key_take(sel_key, est_snd, &mut relaxed_snd) { relaxed_move=Some((m,pos,est_mk)); } }
                            else if tie_mode!=0 && adj_mk==relaxed_key && tie_break_take(tie_mode, block_len, &mut relaxed_tie_n, &mut relaxed_tie_block, &mut tseed) { relaxed_move=Some((m,pos,est_mk)); }
                        }
                    }
                    run_start = pos;
                    run_end = pos;
                }
                prev_pos = pos;
                prev_node = node;
            }
            if run_end > run_start {
                let block_len = run_end-run_start+1;
                let mut swap_positions = [run_start,NONE_USIZE]; let num_swaps = if block_len>=3 { swap_positions[1]=run_end-1; 2 } else { 1 };
                for si in 0..num_swaps {
                    let pos=swap_positions[si]; if pos+1>=seq.len() { continue; }
                    let node_u=seq[pos]; let node_v=seq[pos+1];
                    let (est_mk, est_snd) = estimate_swap_mk2(node_u, node_v, &buf.start, &tail, &ds.node_pt, &job_pred_node, &ds.job_succ, &machine_pred_node, &buf.machine_succ);
                    let lu = node_local[node_u]; let lv = node_local[node_v];
                    let (a, b) = if lu < lv { (lu, lv) } else { (lv, lu) };
                    let tabu_idx = pair_offsets[m] + b * (b - 1) / 2 + a;
                    let is_tabu = tabu_expiry[tabu_idx] > iter; let aspiration=est_mk<best_global_mk;
                    let age = iter.saturating_sub(machine_last_pick[m]).min(9) as u32;
                    let block_bonus = block_len.saturating_sub(2).min(3) as u32;
                    let div_bonus = if diversify > 0.0 {
                        ((diversify_unit as f64) * diversify * (0.35*(age as f64) + 0.75*(block_bonus as f64)) / 3.0) as u32
                    } else { 0 };
                    let adj_mk = est_mk.saturating_sub(div_bonus);
                    
                    // exactly (first candidate reached wins); the `else if` is short-circuited dead.
                    
                    if !is_tabu||aspiration {
                        
                        // chain below settles the incumbent. Mirrors that chain's own comparisons, so the
                        // buffer holds exactly the set the Taillard bound declares indistinguishable, in
                        // enumeration order (`top_cand[0]` is the legacy first-wins pick).
                        if exact_topk != 0 {
                            if adj_mk<best_move_key { top_cand[0] = (m, pos); top_n = 1; }
                            else if adj_mk==best_move_key && top_n < exact_topk { top_cand[top_n] = (m, pos); top_n += 1; }
                        }
                        if adj_mk<best_move_key { best_move_key=adj_mk; best_move=Some((m,pos,est_mk)); best_tie_n=1; best_tie_block=block_len; best_snd=est_snd; }
                        else if sel_key!=0 && adj_mk==best_move_key { if sel_key_take(sel_key, est_snd, &mut best_snd) { best_move=Some((m,pos,est_mk)); } }
                        else if tie_mode!=0 && adj_mk==best_move_key && tie_break_take(tie_mode, block_len, &mut best_tie_n, &mut best_tie_block, &mut tseed) { best_move=Some((m,pos,est_mk)); }
                    }
                    if adj_mk<relaxed_key { relaxed_key=adj_mk; relaxed_move=Some((m,pos,est_mk)); relaxed_tie_n=1; relaxed_tie_block=block_len; relaxed_snd=est_snd; }
                    else if sel_key!=0 && adj_mk==relaxed_key { if sel_key_take(sel_key, est_snd, &mut relaxed_snd) { relaxed_move=Some((m,pos,est_mk)); } }
                    else if tie_mode!=0 && adj_mk==relaxed_key && tie_break_take(tie_mode, block_len, &mut relaxed_tie_n, &mut relaxed_tie_block, &mut tseed) { relaxed_move=Some((m,pos,est_mk)); }
                }
            }
        }
        
        // `estimate_swap_mk`, the Taillard-1994 bound over the two paths a swap rewrites: any path
        // that avoids `u` and `v` is invisible to it, so two candidates it cannot separate may have
        
        
        // a second PROXY component all land under the baseline. Here the collision is settled by the
        // objective itself: replay each tied candidate through the very evaluator the loop already
        // runs, and keep the true argmin.
        //
        // The probe is a TOGGLE (`probe_swap_toggle` is an involution), so applying it twice leaves
        // `machine_seq` and every incremental mirror bit-for-bit as they were — the search state is
        // read-only across this block. `buf.start` / `buf.best_pred` / `topo` ARE clobbered, which is
        // safe: nothing between here and the `eval_disj_topo` of the applied move reads them, and
        // that call rebuilds all three. A probe returning `None` (the evaluator's cycle guard) simply
        // withdraws that candidate instead of being applied — the feasibility of a move is decided by
        // the evaluator the solver already trusts, never by a re-implemented check.
        if exact_topk != 0 && top_n >= 2 && exact_left > 0 && best_move.is_some() {
            exact_left -= 1;
            let mut best_true = u32::MAX;
            let mut best_true_cand: Option<(usize, usize)> = None;
            for ci in 0..top_n {
                let (cm, cpos) = top_cand[ci];
                probe_swap_toggle(&mut ds, cm, cpos, &mut current_pos, &mut machine_pred_node, &mut buf.machine_succ, &mut indeg_full, fast_eval);
                let probe = if fast_eval { eval_disj_topo(&ds, &mut buf, &indeg_full, &mut topo) } else { eval_disj(&ds, &mut buf) };
                probe_swap_toggle(&mut ds, cm, cpos, &mut current_pos, &mut machine_pred_node, &mut buf.machine_succ, &mut indeg_full, fast_eval);
                // Strict `<` keeps the FIRST candidate on an exact tie, i.e. the legacy pick, so the
                // rerank only ever overrides the baseline when the true makespans genuinely differ.
                if let Some((true_mk, _)) = probe {
                    if true_mk < best_true { best_true = true_mk; best_true_cand = Some((cm, cpos)); }
                }
            }
            // The third field now carries the TRUE makespan. It is unread on this path (`chosen`
            // destructures it as `_est`), but leaving a stale estimate there would be a trap for the
            // next iteration on this brick.
            if let Some((cm, cpos)) = best_true_cand { best_move = Some((cm, cpos, best_true)); }
        }
        let chosen = best_move.or(relaxed_move);
        
        // nothing above or below ever ACCEPTS one. The legacy walk applies `chosen` unconditionally,
        // so at a local optimum it is forced to take the least-bad worsening swap every single
        
        // same byte-identical outcome from four, ten and twelve different starting points: the
        // dynamic, not the start, decides where the median instances land.
        //
        // Metropolis on the SAME candidate the loop already picked: a non-worsening move is always
        // taken (`cand_mk <= cur_mk` never enters the branch, so descent is never slowed), and a
        // worsening one is taken with probability `exp(-delta / T)`. `T` cools LINEARLY from
        // `sa_accept` to 0 over the run, so the tail of every draw is a pure descent and the run
        // still terminates on a local optimum of the very same neighbourhood — the last iterations
        // cannot leave the search parked on a worse solution than it had found.
        //
        // A REJECTED move applies nothing and `continue`s: `ds`, `machine_seq`, `current_pos`,
        // `machine_pred_node`, `tabu_expiry`, `machine_last_pick`, `pseed`, `tseed`, `cur_mk` and
        // `mk_node` are all left exactly as they were, so no incremental mirror can desynchronise.
        // The iteration is still consumed (`iter` advances, and the top of the loop credits one
        // more `no_improve` because `cur_mk` did not improve), which is what makes this ISO-BUDGET:
        // the run cannot buy extra work with its refusals. It is also not a stall — `tabu_expiry`
        // is compared against `iter`, so entries keep expiring and the admissible candidate set
        // genuinely changes underneath a repeated refusal.
        //
        // `cand_mk` is the same quantity the argmin above ranked on (`estimate_swap_mk`, or the
        // TRUE makespan when `exact_topk` rewrote it), so the acceptance test and the selection
        
        // paying `eval_disj_topo` to refine that scale costs −1 457 Q.
        if sa_accept > 0 {
            if let Some((_, _, cand_mk)) = chosen {
                if cand_mk > cur_mk {
                    let temp = (sa_accept as f64) * (1.0 - (iter as f64) / (max_iterations.max(1) as f64));
                    let accept = if temp <= 1e-9 {
                        false
                    } else {
                        aseed ^= aseed.wrapping_shl(13); aseed ^= aseed.wrapping_shr(7); aseed ^= aseed.wrapping_shl(17);
                        // 53 significant bits -> a uniform draw in [0, 1).
                        let u = ((aseed >> 11) as f64) * (1.0f64 / 9007199254740992.0);
                        u < (-((cand_mk - cur_mk) as f64) / temp).exp()
                    };
                    if !accept { continue; }
                }
            }
        }
        match chosen {
            Some((m,pos,_est)) => {
                let node_a=ds.machine_seq[m][pos]; let node_b=ds.machine_seq[m][pos+1];
                ds.machine_seq[m].swap(pos,pos+1);
                current_pos[node_a] = pos + 1;
                current_pos[node_b] = pos;
                machine_last_pick[m] = iter;
                let seq = &ds.machine_seq[m];
                let prev = if pos > 0 { seq[pos - 1] } else { NONE_USIZE };
                machine_pred_node[seq[pos]] = prev;
                machine_pred_node[seq[pos + 1]] = seq[pos];
                if pos + 2 < seq.len() { machine_pred_node[seq[pos + 2]] = seq[pos + 1]; }
                pseed^=pseed.wrapping_shl(13); pseed^=pseed.wrapping_shr(7); pseed^=pseed.wrapping_shl(17);
                let offset=(pseed%((2*tenure_delta+1) as u64)) as usize;
                let progress=(iter as f64)/(max_iterations as f64); let late_bonus=if progress>0.6{((progress-0.6)*10.0) as usize}else{0};
                let this_tenure=(tenure+offset+late_bonus).saturating_sub(tenure_delta);
                let la = node_local[node_a]; let lb = node_local[node_b];
                let (a, b) = if la < lb { (la, lb) } else { (lb, la) };
                let tabu_idx = pair_offsets[m] + b * (b - 1) / 2 + a;
                tabu_expiry[tabu_idx] = iter + this_tenure;
                if fast_eval {
                    // Mirror of the `machine_pred_node` patch just above, on the successor side.
                    // After `swap(pos, pos+1)` the machine chain reads ... -> seq[pos-1] ->
                    // seq[pos] -> seq[pos+1] -> seq[pos+2] ..., so exactly three links move.
                    if pos > 0 { buf.machine_succ[seq[pos - 1]] = seq[pos]; }
                    buf.machine_succ[seq[pos]] = seq[pos + 1];
                    buf.machine_succ[seq[pos + 1]] = if pos + 2 < seq.len() { seq[pos + 2] } else { NONE_USIZE };
                    // The machine in-degree contribution is "+1 for every node that is not first
                    // on its machine". A swap only changes WHICH node is first, and only when it
                    // happens at position 0 — everywhere else the contribution is untouched.
                    if pos == 0 {
                        indeg_full[node_a] = indeg_full[node_a].saturating_add(1);
                        indeg_full[node_b] = indeg_full[node_b].saturating_sub(1);
                    }
                }
                let Some((mk, node)) = (if fast_eval { eval_disj_topo(&ds, &mut buf, &indeg_full, &mut topo) } else { eval_disj(&ds, &mut buf) }) else { if mid_abort_rescue > 0 { break } else { return Ok(None) } };
                cur_mk = mk; mk_node = node;
            }
            None => break,
        }
    }
    if cur_mk < best_global_mk { best_global_mk = cur_mk; best_global_machine_seq.clone_from(&ds.machine_seq); rebuild_machine_pos_map(&best_global_machine_seq, &mut best_global_pos); }
    if best_global_mk >= initial_mk { return Ok(None); }
    ds.machine_seq = best_global_machine_seq;
    let Some((mk_final,_)) = eval_disj(&ds, &mut buf) else { return Ok(None) };
    let sol = disj_to_solution(pre, &ds, &buf.start)?;
    Ok(Some((sol, mk_final)))
}

#[inline]
fn estimate_swap_mk(u: usize, v: usize, heads: &[u32], tails: &[u32], pt: &[u32], job_pred: &[usize], job_succ: &[usize], machine_pred: &[usize], machine_succ: &[usize]) -> u32 {
    let mp_u=machine_pred[u]; let ms_v=machine_succ[v]; let jp_v=job_pred[v]; let jp_u=job_pred[u]; let js_u=job_succ[u]; let js_v=job_succ[v];
    let r_jp_v=if jp_v!=NONE_USIZE{heads[jp_v].saturating_add(pt[jp_v])}else{0}; let r_mp_u=if mp_u!=NONE_USIZE{heads[mp_u].saturating_add(pt[mp_u])}else{0};
    let new_r_v=r_jp_v.max(r_mp_u); let r_jp_u=if jp_u!=NONE_USIZE{heads[jp_u].saturating_add(pt[jp_u])}else{0}; let new_r_u=r_jp_u.max(new_r_v.saturating_add(pt[v]));
    let q_js_u=if js_u!=NONE_USIZE{pt[js_u].saturating_add(tails[js_u])}else{0}; let q_ms_v=if ms_v!=NONE_USIZE{pt[ms_v].saturating_add(tails[ms_v])}else{0};
    let new_q_u=q_js_u.max(q_ms_v); let q_js_v=if js_v!=NONE_USIZE{pt[js_v].saturating_add(tails[js_v])}else{0}; let new_q_v=q_js_v.max(pt[u].saturating_add(new_q_u));
    let path_v=new_r_v.saturating_add(pt[v]).saturating_add(new_q_v); let path_u=new_r_u.saturating_add(pt[u]).saturating_add(new_q_u);
    path_v.max(path_u)
}


/// returning BOTH order statistics of the two rewritten paths instead of only the larger one:
/// `.0 = max(path_u, path_v)` (the Taillard-1994 bound the baseline ranks on, unchanged) and
/// `.1 = min(path_u, path_v)` (the second path, computed and then discarded by the baseline).
///
/// The point is that `.0` alone is a LOWER BOUND on the new makespan restricted to the two paths
/// the swap actually rewrites — every path avoiding `u` and `v` is ignored — so it is coarse and

/// the byte-identical baseline), and that the baseline resolves them by enumeration order alone.
/// `.1` is free extra resolution from the very same head/tail arithmetic: among two swaps whose
/// bottleneck path is equal, it says which one leaves the second path shorter.
#[inline]
fn estimate_swap_mk2(u: usize, v: usize, heads: &[u32], tails: &[u32], pt: &[u32], job_pred: &[usize], job_succ: &[usize], machine_pred: &[usize], machine_succ: &[usize]) -> (u32, u32) {
    let mp_u=machine_pred[u]; let ms_v=machine_succ[v]; let jp_v=job_pred[v]; let jp_u=job_pred[u]; let js_u=job_succ[u]; let js_v=job_succ[v];
    let r_jp_v=if jp_v!=NONE_USIZE{heads[jp_v].saturating_add(pt[jp_v])}else{0}; let r_mp_u=if mp_u!=NONE_USIZE{heads[mp_u].saturating_add(pt[mp_u])}else{0};
    let new_r_v=r_jp_v.max(r_mp_u); let r_jp_u=if jp_u!=NONE_USIZE{heads[jp_u].saturating_add(pt[jp_u])}else{0}; let new_r_u=r_jp_u.max(new_r_v.saturating_add(pt[v]));
    let q_js_u=if js_u!=NONE_USIZE{pt[js_u].saturating_add(tails[js_u])}else{0}; let q_ms_v=if ms_v!=NONE_USIZE{pt[ms_v].saturating_add(tails[ms_v])}else{0};
    let new_q_u=q_js_u.max(q_ms_v); let q_js_v=if js_v!=NONE_USIZE{pt[js_v].saturating_add(tails[js_v])}else{0}; let new_q_v=q_js_v.max(pt[u].saturating_add(new_q_u));
    let path_v=new_r_v.saturating_add(pt[v]).saturating_add(new_q_v); let path_u=new_r_u.saturating_add(pt[u]).saturating_add(new_q_u);
    (path_v.max(path_u), path_v.min(path_u))
}


/// displaces the incumbent, on the strength of the SECOND path length. Never called under
/// `sel_key == 0`, so the baseline keeps its strict `<` on the primary key alone and stays
/// byte-identical.
///
/// Both comparisons are STRICT, so candidates equal on BOTH components fall back on the legacy
/// first-reached rule and the whole selection stays deterministic (no wall-clock, no RNG).
/// - mode 1: prefer the SMALLER second path — the hypothesis. Two swaps whose bottleneck estimate
///   is identical are not equivalent: the one leaving the other rewritten path shorter carries less
///   residual makespan pressure into the next iteration, so it should be the better descent.
/// - mode 2: prefer the LARGER second path — the exact negation of mode 1, carried as the two-sided
///   control. If both modes move Q away from 88 813 in OPPOSITE directions, the second component is
///   a real ordering signal and the sign tells us which way to amplify; if both collapse the same
///   way, the precision axis is refuted whatever the sign.
#[inline(always)]
fn sel_key_take(sel_key: usize, snd: u32, best_snd: &mut u32) -> bool {
    match sel_key {
        1 => { if snd < *best_snd { *best_snd = snd; true } else { false } }
        2 => { if snd > *best_snd { *best_snd = snd; true } else { false } }
        _ => false,
    }
}


/// selection key should displace it. Never called under `tie_mode == 0`, so the baseline keeps its
/// strict `<` and stays byte-identical.
///
/// `tie_n` counts the candidates seen at the current key, the incumbent included, so it is already
/// >= 1 on entry and becomes k for the k-th tied candidate.
/// - mode 1: uniform reservoir sampling — accept with probability exactly 1/k, which leaves every
///   tied candidate equally likely to end up selected, whatever the enumeration order.
/// - mode 2: keep the candidate sitting on the LONGEST critical block (strict, so equal lengths
///   fall back on the legacy first-reached and the rule stays deterministic).
/// - mode 3: always accept, i.e. LAST-wins — the exact negation of mode 0, used as the positive
///   control that ties exist and decide.
#[inline(always)]
fn tie_break_take(tie_mode: usize, block_len: usize, tie_n: &mut u64, tie_block: &mut usize, tseed: &mut u64) -> bool {
    *tie_n = tie_n.saturating_add(1);
    match tie_mode {
        1 => {
            *tseed ^= tseed.wrapping_shl(13);
            *tseed ^= tseed.wrapping_shr(7);
            *tseed ^= tseed.wrapping_shl(17);
            (*tseed % (*tie_n).max(1)) == 0
        }
        2 => { if block_len > *tie_block { *tie_block = block_len; true } else { false } }
        3 => true,
        _ => false,
    }
}


/// to every incremental mirror the fast evaluator depends on. It is a deliberate re-statement of the
/// apply block inside `tabu_search_phase` rather than a refactor of it: the default path of that
/// block has to stay byte-identical, and this track has already burned two iterations on VOIDs
/// caused by touching code the baseline runs.
///
/// The point of the function is that it is an INVOLUTION — calling it twice on the same `(m, pos)`
/// restores the exact prior state, which is what lets the exact rerank probe a candidate and put the
/// search back untouched:
///   * `swap(pos, pos+1)` is its own inverse;
///   * `current_pos` and the three `machine_pred_node` links are recomputed FROM the sequence, so
///     the second call recomputes them from the restored sequence;
///   * `machine_succ` likewise, over the same three-link window;
///   * `indeg_full` only moves when `pos == 0` (the "+1 for every node that is not first on its
///     machine" contribution changes owner), and the second call swaps the roles of `node_a` and
///     `node_b`, undoing the increment and the decrement.
#[inline]
fn probe_swap_toggle(ds: &mut DisjSchedule, m: usize, pos: usize, current_pos: &mut [usize], machine_pred_node: &mut [usize], machine_succ: &mut [usize], indeg_full: &mut [u16], fast_eval: bool) {
    let node_a = ds.machine_seq[m][pos];
    let node_b = ds.machine_seq[m][pos + 1];
    ds.machine_seq[m].swap(pos, pos + 1);
    current_pos[node_a] = pos + 1;
    current_pos[node_b] = pos;
    let seq = &ds.machine_seq[m];
    let prev = if pos > 0 { seq[pos - 1] } else { NONE_USIZE };
    machine_pred_node[seq[pos]] = prev;
    machine_pred_node[seq[pos + 1]] = seq[pos];
    if pos + 2 < seq.len() { machine_pred_node[seq[pos + 2]] = seq[pos + 1]; }
    if fast_eval {
        if pos > 0 { machine_succ[seq[pos - 1]] = seq[pos]; }
        machine_succ[seq[pos]] = seq[pos + 1];
        machine_succ[seq[pos + 1]] = if pos + 2 < seq.len() { seq[pos + 2] } else { NONE_USIZE };
        if pos == 0 {
            indeg_full[node_a] = indeg_full[node_a].saturating_add(1);
            indeg_full[node_b] = indeg_full[node_b].saturating_sub(1);
        }
    }
}

#[inline]
fn relocate_machine_seq(seq: &mut [usize], from: usize, to: usize) {
    if from < to {
        seq[from..=to].rotate_left(1);
    } else if to < from {
        seq[to..=from].rotate_right(1);
    }
}

#[inline]
fn inherit_machine_consensus_order(
    child_seq: &mut Vec<usize>,
    better_seq: &[usize],
    worse_seq: &[usize],
    elite_machine_seqs: &[&[usize]],
    pos_buf: &mut [usize],
    sum_buf: &mut [u32],
    cnt_buf: &mut [u16],
    pair_buf: &mut Vec<(usize, usize, i32)>,
) {
    if better_seq.len() <= 1 || better_seq.len() != worse_seq.len() {
        if child_seq.len() != better_seq.len() {
            child_seq.clear();
            child_seq.extend_from_slice(better_seq);
        }
        return;
    }
    if child_seq.len() != better_seq.len() {
        child_seq.clear();
        child_seq.extend_from_slice(better_seq);
    }

    for &node in better_seq {
        pos_buf[node] = 0;
        sum_buf[node] = 0;
        cnt_buf[node] = 0;
    }
    for (idx, &node) in better_seq.iter().enumerate() {
        pos_buf[node] = idx;
    }
    for &eseq in elite_machine_seqs {
        if eseq.len() != better_seq.len() { continue; }
        for (idx, &node) in eseq.iter().enumerate() {
            sum_buf[node] = sum_buf[node].saturating_add(idx as u32);
            cnt_buf[node] = cnt_buf[node].saturating_add(1);
        }
    }

    pair_buf.clear();
    for win in worse_seq.windows(2) {
        let u = win[0];
        let v = win[1];
        let pu = pos_buf[u];
        let pv = pos_buf[v];
        if pu < pv { continue; }
        let cu = cnt_buf[u];
        let cv = cnt_buf[v];
        if cu == 0 || cv == 0 { continue; }
        let avg_u = (sum_buf[u] as f64) / (cu as f64);
        let avg_v = (sum_buf[v] as f64) / (cv as f64);
        let gap = avg_v - avg_u;
        if gap <= 0.55 { continue; }
        let score = ((gap * 16.0) as i32) + ((pu - pv).min(10) as i32);
        pair_buf.push((u, v, score));
    }
    if pair_buf.is_empty() { return; }

    pair_buf.sort_unstable_by(|a, b| b.2.cmp(&a.2));
    let apply_cap = if better_seq.len() > 14 { 2usize } else { 1usize };
    let mut applied = 0usize;
    for &(u, v, _) in pair_buf.iter() {
        if applied >= apply_cap { break; }
        for (idx, &node) in child_seq.iter().enumerate() {
            pos_buf[node] = idx;
        }
        let from = pos_buf[u];
        let to = pos_buf[v];
        if from < to { continue; }
        child_seq[to..=from].rotate_right(1);
        applied += 1;
    }
}

#[inline]
fn invert_job_bias_guidance(jb: &[f64]) -> Vec<f64> {
    if jb.is_empty() { return Vec::new(); }
    let mean = jb.iter().copied().sum::<f64>() / (jb.len() as f64);
    let mut anti = Vec::with_capacity(jb.len());
    for &v in jb {
        anti.push((mean - v) * 0.75);
    }
    anti
}

#[inline]
fn invert_machine_penalty_guidance(mp: &[f64]) -> Vec<f64> {
    if mp.is_empty() { return Vec::new(); }
    let mean = mp.iter().copied().sum::<f64>() / (mp.len() as f64);
    let mut anti = Vec::with_capacity(mp.len());
    for &v in mp {
        anti.push((0.55*mean + 0.45*(1.0 - v.clamp(0.0, 1.0))).clamp(0.0, 1.0));
    }
    anti
}

fn critical_block_move_local_search_ex_disj(
    ds: &mut DisjSchedule,
    buf: &mut EvalBuf,
    max_rounds: usize,
    max_iters: usize,
    stall_limit: usize,
) -> Option<u32> {
    let n = ds.n;
    let Some((initial_mk, mut mk_node)) = eval_disj(ds, buf) else { return None };
    let mut test_buf_a = EvalBuf::new(n);
    let mut test_buf_b = EvalBuf::new(n);
    let mut cur_mk = initial_mk;
    let mut tail = vec![0u32; n];
    let mut back_deg = vec![0u16; n];
    let mut back_stack: Vec<usize> = Vec::with_capacity(n);
    let mut machine_pred_node = vec![NONE_USIZE; n];
    let mut job_pred_node = vec![NONE_USIZE; n];
    let mut moves: Vec<(u32,u16,usize,usize)> = Vec::with_capacity(64);
    let mut current_pos = vec![0usize; n];
    let mut node_machine = vec![0usize; n];
    let mut crit_positions: Vec<(usize,usize)> = Vec::with_capacity(n);

    for j in 0..ds.num_jobs {
        let base = ds.job_offsets[j];
        let end = ds.job_offsets[j + 1];
        for k in (base + 1)..end {
            job_pred_node[k] = k - 1;
        }
    }
    for m in 0..ds.num_machines {
        for (i, &node) in ds.machine_seq[m].iter().enumerate() {
            current_pos[node] = i;
            node_machine[node] = m;
        }
    }

    let iter_limit = max_iters.max(max_rounds).max(1);
    let stall_cap = stall_limit.max(max_rounds).max(1);
    let mut stalled = 0usize;

    for _ in 0..iter_limit {
        rebuild_machine_pred_nodes(ds, &mut machine_pred_node);
        tail.fill(0);
        back_deg.fill(0);
        for i in 0..n {
            if ds.job_succ[i] != NONE_USIZE { back_deg[i] += 1; }
            if buf.machine_succ[i] != NONE_USIZE { back_deg[i] += 1; }
        }
        back_stack.clear();
        for i in 0..n {
            if back_deg[i] == 0 { back_stack.push(i); }
        }
        while let Some(nd) = back_stack.pop() {
            let contrib = ds.node_pt[nd].saturating_add(tail[nd]);
            let jp = job_pred_node[nd];
            if jp != NONE_USIZE {
                if contrib > tail[jp] { tail[jp] = contrib; }
                back_deg[jp] = back_deg[jp].saturating_sub(1);
                if back_deg[jp] == 0 { back_stack.push(jp); }
            }
            let mp = machine_pred_node[nd];
            if mp != NONE_USIZE {
                if contrib > tail[mp] { tail[mp] = contrib; }
                back_deg[mp] = back_deg[mp].saturating_sub(1);
                if back_deg[mp] == 0 { back_stack.push(mp); }
            }
        }

        crit_positions.clear();
        let mut u = mk_node;
        while u != NONE_USIZE {
            crit_positions.push((node_machine[u], current_pos[u]));
            u = buf.best_pred[u];
        }
        if crit_positions.len() > 1 { crit_positions.sort_unstable(); }

        moves.clear();
        let mut cp_i = 0usize;
        while cp_i < crit_positions.len() {
            let m = crit_positions[cp_i].0;
            let seq = &ds.machine_seq[m];
            let mut run_start = crit_positions[cp_i].1;
            let mut run_end = run_start;
            let mut prev_pos = run_start;
            let mut prev_node = seq[prev_pos];
            cp_i += 1;
            while cp_i < crit_positions.len() && crit_positions[cp_i].0 == m {
                let pos = crit_positions[cp_i].1;
                let node = seq[pos];
                if pos == prev_pos + 1 && buf.start[node] == buf.start[prev_node].saturating_add(ds.node_pt[prev_node]) {
                    run_end = pos;
                } else {
                    if run_end > run_start {
                        let block_len = run_end - run_start + 1;
                        let mut swap_positions = [run_start, NONE_USIZE];
                        let num_swaps = if block_len >= 3 { swap_positions[1] = run_end - 1; 2 } else { 1 };
                        let block_len_u16 = block_len.min(u16::MAX as usize) as u16;
                        for si in 0..num_swaps {
                            let pos = swap_positions[si];
                            if pos + 1 >= seq.len() { continue; }
                            let node_u = seq[pos];
                            let node_v = seq[pos + 1];
                            let est_mk = estimate_swap_mk(node_u, node_v, &buf.start, &tail, &ds.node_pt, &job_pred_node, &ds.job_succ, &machine_pred_node, &buf.machine_succ);
                            if est_mk < cur_mk {
                                moves.push((est_mk, block_len_u16, m, pos));
                            }
                        }
                    }
                    run_start = pos;
                    run_end = pos;
                }
                prev_pos = pos;
                prev_node = node;
                cp_i += 1;
            }
            if run_end > run_start {
                let block_len = run_end - run_start + 1;
                let mut swap_positions = [run_start, NONE_USIZE];
                let num_swaps = if block_len >= 3 { swap_positions[1] = run_end - 1; 2 } else { 1 };
                let block_len_u16 = block_len.min(u16::MAX as usize) as u16;
                for si in 0..num_swaps {
                    let pos = swap_positions[si];
                    if pos + 1 >= seq.len() { continue; }
                    let node_u = seq[pos];
                    let node_v = seq[pos + 1];
                    let est_mk = estimate_swap_mk(node_u, node_v, &buf.start, &tail, &ds.node_pt, &job_pred_node, &ds.job_succ, &machine_pred_node, &buf.machine_succ);
                    if est_mk < cur_mk {
                        moves.push((est_mk, block_len_u16, m, pos));
                    }
                }
            }
        }

        if moves.is_empty() { break; }
        if moves.len() > 1 {
            moves.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| b.1.cmp(&a.1)).then_with(|| a.2.cmp(&b.2)).then_with(|| a.3.cmp(&b.3)));
        }

        let eval_cap = moves.len().min(2);
        let mut best_idx = NONE_USIZE;
        let mut best_actual_mk = cur_mk;
        let mut best_actual_node = NONE_USIZE;
        let mut tested = [(NONE_USIZE, NONE_USIZE); 2];

        for idx in 0..eval_cap {
            let (_, _, m, pos) = moves[idx];
            if pos + 1 >= ds.machine_seq[m].len() { continue; }
            tested[idx] = (m, pos);
            let node_a = ds.machine_seq[m][pos];
            let node_b = ds.machine_seq[m][pos + 1];
            ds.machine_seq[m].swap(pos, pos + 1);
            current_pos[node_a] = pos + 1;
            current_pos[node_b] = pos;
            let res = if idx == 0 {
                eval_disj(ds, &mut test_buf_a)
            } else {
                eval_disj(ds, &mut test_buf_b)
            };
            if let Some((new_mk, new_node)) = res {
                if new_mk < best_actual_mk {
                    best_actual_mk = new_mk;
                    best_actual_node = new_node;
                    best_idx = idx;
                }
            }
            ds.machine_seq[m].swap(pos, pos + 1);
            current_pos[node_a] = pos;
            current_pos[node_b] = pos + 1;
        }

        if best_idx != NONE_USIZE {
            let (m, pos) = tested[best_idx];
            let node_a = ds.machine_seq[m][pos];
            let node_b = ds.machine_seq[m][pos + 1];
            ds.machine_seq[m].swap(pos, pos + 1);
            current_pos[node_a] = pos + 1;
            current_pos[node_b] = pos;
            cur_mk = best_actual_mk;
            mk_node = best_actual_node;
            if best_idx == 0 {
                core::mem::swap(buf, &mut test_buf_a);
            } else {
                core::mem::swap(buf, &mut test_buf_b);
            }
            stalled = 0;
        } else {
            stalled += 1;
            if stalled >= stall_cap { break; }
        }
    }

    if cur_mk < initial_mk { Some(cur_mk) } else { None }
}

#[inline]
fn perturb_and_reoptimize_ils(
    pre: &Pre,
    challenge: &Challenge,
    base_sol: &Solution,
    num_perturb: usize,
    ls_rounds: usize,
    ls_iters: usize,
    ls_stall: usize,
) -> Result<Option<(Solution, u32)>> {
    let mut ds = build_disj_from_solution(pre, challenge, base_sol)?;
    let mut buf = EvalBuf::new(ds.n);
    let Some((start_mk, mut mk_node)) = eval_disj(&ds, &mut buf) else { return Ok(None) };
    let mut cur_mk = start_mk;

    let mut current_pos = vec![0usize; ds.n];
    let mut node_machine = vec![0usize; ds.n];
    let mut crit_pos_by_machine: Vec<Vec<usize>> = (0..ds.num_machines).map(|_| Vec::new()).collect();
    let mut crit_pos_machines: Vec<usize> = Vec::with_capacity(ds.num_machines);
    let mut moved = 0usize;

    for _ in 0..num_perturb {
        for m in 0..ds.num_machines {
            crit_pos_by_machine[m].clear();
            for (pos, &node) in ds.machine_seq[m].iter().enumerate() {
                current_pos[node] = pos;
                node_machine[node] = m;
            }
        }
        crit_pos_machines.clear();
        let mut u = mk_node;
        while u != NONE_USIZE {
            let m = node_machine[u];
            if crit_pos_by_machine[m].is_empty() { crit_pos_machines.push(m); }
            crit_pos_by_machine[m].push(current_pos[u]);
            u = buf.best_pred[u];
        }

        let mut best_move: Option<(usize, usize, usize, u8, usize, u32)> = None;
        for &m in &crit_pos_machines {
            let positions = &mut crit_pos_by_machine[m];
            if positions.len() < 2 { continue; }
            positions.sort_unstable();
            let seq = &ds.machine_seq[m];
            let mut run_start = positions[0];
            let mut run_end = positions[0];
            let mut prev_pos = positions[0];
            let mut prev_node = seq[prev_pos];

            for idx in 1..positions.len() {
                let pos = positions[idx];
                let node = seq[pos];
                if pos == prev_pos + 1 && buf.start[node] == buf.start[prev_node].saturating_add(ds.node_pt[prev_node]) {
                    run_end = pos;
                } else {
                    if run_end > run_start {
                        let block_len = run_end - run_start + 1;
                        let pt_sum = seq[run_start..=run_end].iter().fold(0u32, |acc, &nd| acc.saturating_add(ds.node_pt[nd]));
                        let cand = if block_len >= 5 {
                            let head_mass = ds.node_pt[seq[run_start]].saturating_add(ds.node_pt[seq[run_start + 1]]);
                            let tail_mass = ds.node_pt[seq[run_end - 1]].saturating_add(ds.node_pt[seq[run_end]]);
                            if head_mass <= tail_mass {
                                (m, run_start, run_end - 1, 0u8, block_len, pt_sum)
                            } else {
                                (m, run_end, run_start + 1, 0u8, block_len, pt_sum)
                            }
                        } else if block_len == 4 {
                            let head_mass = ds.node_pt[seq[run_start]].saturating_add(ds.node_pt[seq[run_start + 1]]);
                            let tail_mass = ds.node_pt[seq[run_end - 1]].saturating_add(ds.node_pt[seq[run_end]]);
                            if head_mass <= tail_mass {
                                (m, run_start, 0usize, 1u8, block_len, pt_sum)
                            } else {
                                (m, run_end - 2, 0usize, 2u8, block_len, pt_sum)
                            }
                        } else if block_len == 3 {
                            if ds.node_pt[seq[run_start]] <= ds.node_pt[seq[run_end]] {
                                (m, run_start, 0usize, 1u8, block_len, pt_sum)
                            } else {
                                (m, run_start, 0usize, 2u8, block_len, pt_sum)
                            }
                        } else {
                            (m, run_start, 0usize, 3u8, block_len, pt_sum)
                        };
                        if best_move.as_ref().map_or(true, |&(_, _, _, _, best_len, best_sum)| block_len > best_len || (block_len == best_len && pt_sum > best_sum)) {
                            best_move = Some(cand);
                        }
                    }
                    run_start = pos;
                    run_end = pos;
                }
                prev_pos = pos;
                prev_node = node;
            }

            if run_end > run_start {
                let block_len = run_end - run_start + 1;
                let pt_sum = seq[run_start..=run_end].iter().fold(0u32, |acc, &nd| acc.saturating_add(ds.node_pt[nd]));
                let cand = if block_len >= 5 {
                    let head_mass = ds.node_pt[seq[run_start]].saturating_add(ds.node_pt[seq[run_start + 1]]);
                    let tail_mass = ds.node_pt[seq[run_end - 1]].saturating_add(ds.node_pt[seq[run_end]]);
                    if head_mass <= tail_mass {
                        (m, run_start, run_end - 1, 0u8, block_len, pt_sum)
                    } else {
                        (m, run_end, run_start + 1, 0u8, block_len, pt_sum)
                    }
                } else if block_len == 4 {
                    let head_mass = ds.node_pt[seq[run_start]].saturating_add(ds.node_pt[seq[run_start + 1]]);
                    let tail_mass = ds.node_pt[seq[run_end - 1]].saturating_add(ds.node_pt[seq[run_end]]);
                    if head_mass <= tail_mass {
                        (m, run_start, 0usize, 1u8, block_len, pt_sum)
                    } else {
                        (m, run_end - 2, 0usize, 2u8, block_len, pt_sum)
                    }
                } else if block_len == 3 {
                    if ds.node_pt[seq[run_start]] <= ds.node_pt[seq[run_end]] {
                        (m, run_start, 0usize, 1u8, block_len, pt_sum)
                    } else {
                        (m, run_start, 0usize, 2u8, block_len, pt_sum)
                    }
                } else {
                    (m, run_start, 0usize, 3u8, block_len, pt_sum)
                };
                if best_move.as_ref().map_or(true, |&(_, _, _, _, best_len, best_sum)| block_len > best_len || (block_len == best_len && pt_sum > best_sum)) {
                    best_move = Some(cand);
                }
            }
        }

        let Some((m, a, b, kind, _, _)) = best_move else { break; };

        match kind {
            0 => {
                let seq = &mut ds.machine_seq[m];
                relocate_machine_seq(seq, a, b);
            }
            1 => {
                let seq = &mut ds.machine_seq[m];
                seq[a..=a + 2].rotate_left(1);
            }
            2 => {
                let seq = &mut ds.machine_seq[m];
                seq[a..=a + 2].rotate_right(1);
            }
            _ => {
                let seq = &mut ds.machine_seq[m];
                seq[a..=a + 1].swap(0, 1);
            }
        }

        match eval_disj(&ds, &mut buf) {
            Some((new_mk, new_node)) => {
                cur_mk = new_mk;
                mk_node = new_node;
                moved += 1;
            }
            None => {
                match kind {
                    0 => {
                        let seq = &mut ds.machine_seq[m];
                        relocate_machine_seq(seq, b, a);
                    }
                    1 => {
                        let seq = &mut ds.machine_seq[m];
                        seq[a..=a + 2].rotate_right(1);
                    }
                    2 => {
                        let seq = &mut ds.machine_seq[m];
                        seq[a..=a + 2].rotate_left(1);
                    }
                    _ => {
                        let seq = &mut ds.machine_seq[m];
                        seq[a..=a + 1].swap(0, 1);
                    }
                }
                let Some((restored_mk, restored_node)) = eval_disj(&ds, &mut buf) else { return Ok(None) };
                cur_mk = restored_mk;
                mk_node = restored_node;
                break;
            }
        }
    }

    if moved == 0 { return Ok(None); }

    let final_mk = match critical_block_move_local_search_ex_disj(&mut ds, &mut buf, ls_rounds, ls_iters, ls_stall) {
        Some(mk) => mk,
        None => match eval_disj(&ds, &mut buf) {
            Some((mk, _)) => mk,
            None => return Ok(None),
        },
    };
    if final_mk >= start_mk && cur_mk >= start_mk { return Ok(None); }

    let sol = disj_to_solution(pre, &ds, &buf.start)?;
    Ok(Some((sol, final_mk.min(cur_mk))))
}



// crossover scope 0 and 1 gave the SAME 32 nonce qualities) and the fence that makes it
// non-regressive. Everything below is dead code unless `js_sbp_rounds > 0`.

/// Deterministic node budget of ONE Carlier solve. Deliberately NOT an hyper-parameter: the axis of
/// this iteration is the number of sweeps, and a per-solve budget moving with it would confound
/// "more sweeps" with "more exact". 600 nodes is far above what a 50-operation subproblem needs, and
/// it bounds the cost of a pathological subproblem without reading a clock — `Instant`-driven
/// budgets are banned on this repo because they make Q machine-dependent.
const SBP_NODE_BUDGET: usize = 600;

/// Scratch buffers of the shifting-bottleneck pass. Allocated once per `sbp_reoptimize` call and
/// reused across every machine of every sweep, so the pass allocates O(1) times per nonce.
struct SbpBuf {
    msucc: Vec<usize>,
    mpred: Vec<usize>,
    jpred: Vec<usize>,
    deg: Vec<u16>,
    stack: Vec<usize>,
    head: Vec<u32>,
    tail: Vec<u32>,
    nodes: Vec<usize>,
    r: Vec<u32>,
    p: Vec<u32>,
    q: Vec<u32>,
    done: Vec<bool>,
    order: Vec<usize>,
    best_order: Vec<usize>,
    saved_seq: Vec<usize>,
}

impl SbpBuf {
    fn new(n: usize) -> Self {
        Self {
            msucc: vec![NONE_USIZE; n], mpred: vec![NONE_USIZE; n], jpred: vec![NONE_USIZE; n],
            deg: vec![0u16; n], stack: Vec::with_capacity(n),
            head: vec![0u32; n], tail: vec![0u32; n],
            nodes: Vec::new(), r: Vec::new(), p: Vec::new(), q: Vec::new(),
            done: Vec::new(), order: Vec::new(), best_order: Vec::new(), saved_seq: Vec::new(),
        }
    }
}

/// Longest-path heads and tails of every node on the disjunctive graph WITH MACHINE `m_excl`'s own

/// `head[u]` is the earliest start of `u` once the operations of `m_excl` no longer constrain each
/// other, `tail[u]` the longest chain that must still follow `u` once it completes. Taking the heads
/// off the CURRENT schedule instead — which is what `estimate_swap_mk` does for a single swap — would
/// bake the very sequence we are trying to replace into its own subproblem.
///
/// Returns `false` if either sweep fails to consume all `n` nodes. Deleting arcs from an acyclic
/// graph cannot create a cycle, so this is a guard, not an expected path.
fn sbp_heads_tails_excl(ds: &DisjSchedule, m_excl: usize, b: &mut SbpBuf) -> bool {
    let n = ds.n;
    for u in 0..n { b.msucc[u] = NONE_USIZE; b.mpred[u] = NONE_USIZE; b.jpred[u] = NONE_USIZE; }
    for (m, seq) in ds.machine_seq.iter().enumerate() {
        if m == m_excl || seq.len() < 2 { continue; }
        for w in seq.windows(2) { b.msucc[w[0]] = w[1]; b.mpred[w[1]] = w[0]; }
    }
    for u in 0..n {
        let js = ds.job_succ[u];
        if js != NONE_USIZE { b.jpred[js] = u; }
    }

    // Forward sweep: head[v] = max over predecessors u of (head[u] + p[u]).
    for u in 0..n {
        let mut d = ds.indeg_job[u];
        if b.mpred[u] != NONE_USIZE { d = d.saturating_add(1); }
        b.deg[u] = d;
        b.head[u] = 0;
    }
    b.stack.clear();
    for u in 0..n { if b.deg[u] == 0 { b.stack.push(u); } }
    let mut processed = 0usize;
    while let Some(u) = b.stack.pop() {
        processed += 1;
        let end_u = b.head[u].saturating_add(ds.node_pt[u]);
        let js = ds.job_succ[u];
        if js != NONE_USIZE {
            if b.head[js] < end_u { b.head[js] = end_u; }
            b.deg[js] = b.deg[js].saturating_sub(1);
            if b.deg[js] == 0 { b.stack.push(js); }
        }
        let ms = b.msucc[u];
        if ms != NONE_USIZE {
            if b.head[ms] < end_u { b.head[ms] = end_u; }
            b.deg[ms] = b.deg[ms].saturating_sub(1);
            if b.deg[ms] == 0 { b.stack.push(ms); }
        }
    }
    if processed != n { return false; }

    // Backward sweep: tail[u] = max over successors v of (p[v] + tail[v]).
    for u in 0..n {
        let mut d = 0u16;
        if ds.job_succ[u] != NONE_USIZE { d += 1; }
        if b.msucc[u] != NONE_USIZE { d += 1; }
        b.deg[u] = d;
        b.tail[u] = 0;
    }
    b.stack.clear();
    for u in 0..n { if b.deg[u] == 0 { b.stack.push(u); } }
    processed = 0;
    while let Some(u) = b.stack.pop() {
        processed += 1;
        let need = b.tail[u].saturating_add(ds.node_pt[u]);
        let jp = b.jpred[u];
        if jp != NONE_USIZE {
            if b.tail[jp] < need { b.tail[jp] = need; }
            b.deg[jp] = b.deg[jp].saturating_sub(1);
            if b.deg[jp] == 0 { b.stack.push(jp); }
        }
        let mp = b.mpred[u];
        if mp != NONE_USIZE {
            if b.tail[mp] < need { b.tail[mp] = need; }
            b.deg[mp] = b.deg[mp].saturating_sub(1);
            if b.deg[mp] == 0 { b.stack.push(mp); }
        }
    }
    processed == n
}

/// Value of an explicit sequence for `1 | r_j, q_j | C_max`.
#[inline]
fn sbp_eval_order(order: &[usize], r: &[u32], p: &[u32], q: &[u32]) -> u32 {
    let mut t = 0u32;
    let mut value = 0u32;
    for &i in order {
        let s = if t > r[i] { t } else { r[i] };
        let e = s.saturating_add(p[i]);
        let c = e.saturating_add(q[i]);
        if c > value { value = c; }
        t = e;
    }
    value
}

/// Schrage's heuristic for `1 | r_j, q_j | C_max`: at every decision point, among the operations
/// already released, start the one with the LARGEST tail; if none is released, jump the clock to the
/// next release date. Ties keep the lowest index, so the sequence is a deterministic function of
/// `(r, p, q)`.
fn sbp_schrage(r: &[u32], p: &[u32], q: &[u32], done: &mut Vec<bool>, order: &mut Vec<usize>) -> u32 {
    let k = r.len();
    done.clear(); done.resize(k, false);
    order.clear();
    let mut t = u32::MAX;
    for i in 0..k { if r[i] < t { t = r[i]; } }
    let mut value = 0u32;
    for _ in 0..k {
        let mut pick = usize::MAX;
        for i in 0..k {
            if done[i] || r[i] > t { continue; }
            if pick == usize::MAX || q[i] > q[pick] { pick = i; }
        }
        if pick == usize::MAX {
            let mut nt = u32::MAX;
            for i in 0..k { if !done[i] && r[i] < nt { nt = r[i]; } }
            if nt == u32::MAX { break; }
            t = nt;
            for i in 0..k {
                if done[i] || r[i] > t { continue; }
                if pick == usize::MAX || q[i] > q[pick] { pick = i; }
            }
            if pick == usize::MAX { break; }
        }
        done[pick] = true;
        order.push(pick);
        let e = t.saturating_add(p[pick]);
        let c = e.saturating_add(q[pick]);
        if c > value { value = c; }
        t = e;
    }
    value
}

/// Carlier's branch and bound for `1 | r_j, q_j | C_max`, capped at `SBP_NODE_BUDGET` nodes.
///
/// Each node is a pair of tightened release/tail vectors; tightening only REMOVES schedules, so the
/// Schrage sequence of any node stays feasible for the original problem and is scored back against
/// the original `(r0, q0)` before it can become the incumbent. Branching follows Carlier: locate the
/// critical block of the Schrage schedule, the critical operation `c` inside it whose tail is
/// smaller than the tail of the operation that realises the makespan, and force `c` either after or
/// before the block that follows it. `h(S) = min_S r + sum_S p + min_S q` is the node's lower bound.
///
/// Hitting the node cap simply returns the best sequence found so far — the caller re-evaluates any
/// candidate through `eval_disj` and only installs a STRICT improvement, so a truncated or even a
/// wrong search costs efficacy, never quality.
fn sbp_carlier(r0: &[u32], p: &[u32], q0: &[u32], b: &mut SbpBuf) -> Option<u32> {
    let k = r0.len();
    if k < 2 { return None; }
    b.best_order.clear();
    let mut ub = u32::MAX;
    let mut nodes = 0usize;
    let mut starts: Vec<u32> = Vec::with_capacity(k);
    let mut stack: Vec<(Vec<u32>, Vec<u32>)> = Vec::with_capacity(16);
    stack.push((r0.to_vec(), q0.to_vec()));

    while let Some((r, q)) = stack.pop() {
        if nodes >= SBP_NODE_BUDGET { break; }
        nodes += 1;

        let f = sbp_schrage(&r, p, &q, &mut b.done, &mut b.order);
        let true_val = sbp_eval_order(&b.order, r0, p, q0);
        if true_val < ub {
            ub = true_val;
            b.best_order.clear();
            b.best_order.extend_from_slice(&b.order);
        }
        if b.order.len() != k { continue; }

        // Start dates of the node's Schrage schedule, and the position that realises `f`.
        starts.clear();
        let mut t = 0u32;
        let mut pos_star = usize::MAX;
        for (pos, &i) in b.order.iter().enumerate() {
            let s = if t > r[i] { t } else { r[i] };
            starts.push(s);
            t = s.saturating_add(p[i]);
            if t.saturating_add(q[i]) == f { pos_star = pos; }
        }
        if pos_star == usize::MAX { continue; }

        // Critical block: walk back while the machine never idled.
        let mut a = pos_star;
        while a > 0 {
            let prev = b.order[a - 1];
            if starts[a] != starts[a - 1].saturating_add(p[prev]) { break; }
            a -= 1;
        }

        // Critical operation `c`: the LAST one in the block, strictly before `pos_star`, whose tail
        // is smaller than the tail of the operation realising the makespan. None => this node's
        // Schrage schedule is optimal for the node, nothing to branch on.
        let q_star = q[b.order[pos_star]];
        let mut pos_c = usize::MAX;
        let mut pos = pos_star;
        while pos > a {
            pos -= 1;
            if q[b.order[pos]] < q_star { pos_c = pos; break; }
        }
        if pos_c == usize::MAX { continue; }

        let c = b.order[pos_c];
        let mut r_min = u32::MAX;
        let mut q_min = u32::MAX;
        let mut p_sum = 0u32;
        for &i in &b.order[(pos_c + 1)..=pos_star] {
            if r[i] < r_min { r_min = r[i]; }
            if q[i] < q_min { q_min = q[i]; }
            p_sum = p_sum.saturating_add(p[i]);
        }
        if r_min == u32::MAX || q_min == u32::MAX { continue; }

        let h_j = r_min.saturating_add(p_sum).saturating_add(q_min);
        let r_min2 = if r[c] < r_min { r[c] } else { r_min };
        let q_min2 = if q[c] < q_min { q[c] } else { q_min };
        let h_jc = r_min2.saturating_add(p_sum).saturating_add(p[c]).saturating_add(q_min2);
        let lb = if h_j > h_jc { h_j } else { h_jc };
        if lb >= ub { continue; }
        if stack.len() + 2 > 2 * SBP_NODE_BUDGET { continue; }

        // Child 2 pushed first so that child 1 is expanded first (LIFO), keeping the traversal a
        // deterministic function of the subproblem.
        let forced_q = q_min.saturating_add(p_sum);
        if forced_q > q[c] {
            let mut q2 = q.clone();
            q2[c] = forced_q;
            stack.push((r.clone(), q2));
        }
        let forced_r = r_min.saturating_add(p_sum);
        if forced_r > r[c] {
            let mut r2 = r.clone();
            r2[c] = forced_r;
            stack.push((r2, q.clone()));
        }
    }

    if b.best_order.len() == k { Some(ub) } else { None }
}

/// One shifting-bottleneck pass over `sol`: sweep the machines by decreasing load, re-solve each
/// one's `1 | r_j, q_j | C_max` subproblem exactly, and install the new sequence only when
/// `eval_disj` certifies a strictly shorter makespan on the FULL graph. Returns `None` when the pass
/// improves nothing, so the caller has nothing to install.
fn sbp_reoptimize(pre: &Pre, challenge: &Challenge, sol: &Solution, rounds: usize) -> Option<(Solution, u32)> {
    if rounds == 0 { return None; }
    let mut ds = build_disj_from_solution(pre, challenge, sol).ok()?;
    let mut buf = EvalBuf::new(ds.n);
    let (start_mk, _) = eval_disj(&ds, &mut buf)?;
    let mut cur_mk = start_mk;
    let mut b = SbpBuf::new(ds.n);

    // Machine order: heaviest first. `Scenario::JOB_SHOP` draws operation times as
    // `base[op_type] * U[0.8,1.2]` with `base` in `[1,200]`, so the loads are very unequal and the
    // makespan is set by the heaviest machines — they are the ones worth re-sequencing first.
    let num_machines = challenge.num_machines.min(ds.machine_seq.len());
    let mut m_order: Vec<usize> = (0..num_machines).filter(|&m| ds.machine_seq[m].len() >= 3).collect();
    let mut load: Vec<u64> = vec![0u64; num_machines];
    for m in 0..num_machines {
        for &nd in &ds.machine_seq[m] { load[m] = load[m].saturating_add(ds.node_pt[nd] as u64); }
    }
    m_order.sort_by(|&x, &y| load[y].cmp(&load[x]).then(x.cmp(&y)));
    if m_order.is_empty() { return None; }

    for _ in 0..rounds {
        let mut improved = false;
        for mi in 0..m_order.len() {
            let m = m_order[mi];
            let k = ds.machine_seq[m].len();
            if k < 3 { continue; }
            if !sbp_heads_tails_excl(&ds, m, &mut b) { continue; }

            b.nodes.clear(); b.nodes.extend_from_slice(&ds.machine_seq[m]);
            b.r.clear(); b.p.clear(); b.q.clear();
            for &nd in &b.nodes {
                b.r.push(b.head[nd]);
                b.p.push(ds.node_pt[nd]);
                b.q.push(b.tail[nd]);
            }

            // Value of the sequence we already have, in the subproblem's own metric. Cheap filter:
            // Carlier can only be worth installing if it beats it.
            let mut cur_val = 0u32;
            {
                let mut t = 0u32;
                for i in 0..k {
                    let s = if t > b.r[i] { t } else { b.r[i] };
                    let e = s.saturating_add(b.p[i]);
                    let c = e.saturating_add(b.q[i]);
                    if c > cur_val { cur_val = c; }
                    t = e;
                }
            }

            let (r, p, q) = (b.r.clone(), b.p.clone(), b.q.clone());
            let Some(val) = sbp_carlier(&r, &p, &q, &mut b) else { continue };
            if val >= cur_val { continue; }

            b.saved_seq.clear(); b.saved_seq.extend_from_slice(&ds.machine_seq[m]);
            for pos in 0..k { ds.machine_seq[m][pos] = b.nodes[b.best_order[pos]]; }
            match eval_disj(&ds, &mut buf) {
                Some((new_mk, _)) if new_mk < cur_mk => { cur_mk = new_mk; improved = true; }
                _ => {
                    ds.machine_seq[m].clear();
                    ds.machine_seq[m].extend_from_slice(&b.saved_seq);
                }
            }
        }
        if !improved { break; }
    }

    if cur_mk >= start_mk { return None; }
    let (final_mk, _) = eval_disj(&ds, &mut buf)?;
    if final_mk >= start_mk { return None; }
    let out = disj_to_solution(pre, &ds, &buf.start).ok()?;
    Some((out, final_mk))
}

// ---------------------------------------------------------------------------------------------

// Dead code unless `js_seed_sbp_construct > 0`. See `EffortConfig::js_seed_sbp_construct` for the
// hypothesis and for why this family is none of the 17 dead ones.
// ---------------------------------------------------------------------------------------------

/// `true` iff the disjunctive graph is acyclic. Reuses the topological double sweep with an
/// out-of-range `m_excl`, so NO machine is excluded and the `processed != n` guard of
/// `sbp_heads_tails_excl` becomes exactly a cycle test — the same evaluator the rest of the solver
/// is judged by, not a re-implementation.
#[inline]
fn sbp_partial_acyclic(ds: &DisjSchedule, b: &mut SbpBuf) -> bool {
    sbp_heads_tails_excl(ds, usize::MAX, b)
}

/// Builds a schedule MACHINE BY MACHINE from the empty disjunctive graph (job precedence only),
/// the classical Adams-Balas-Zawack procedure. At each step every still-unsequenced machine is
/// posed as a `1 | r_j, q_j | C_max` subproblem — releases and tails read off the CURRENT partial

/// whose optimum is LARGEST is the bottleneck: it is the one that will set the makespan, so it is
/// fixed while it still has the most freedom, and the rest of the shop is then built around it.
///
/// `template` supplies only the instance topology (`node_machine`, `node_pt`, `job_succ`); every
/// disjunctive arc it carries is erased on entry. Fully deterministic: no `rng`, no clock.
fn sbp_construct_seed(
    pre: &Pre,
    challenge: &Challenge,
    template: &Solution,
    reopt: bool,
    reopt_cycles: usize,
) -> Option<(Solution, u32)> {
    let mut ds = build_disj_from_solution(pre, challenge, template).ok()?;
    let n = ds.n;
    let num_machines = challenge.num_machines.min(ds.machine_seq.len());
    if num_machines == 0 { return None; }

    // Machine membership is a property of the INSTANCE, not of the template: `Scenario::JOB_SHOP`
    // has `avg_op_flexibility = 1.0`, so each operation has exactly one eligible machine and the
    // template only ever supplied the ordering we are about to throw away.
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); num_machines];
    for u in 0..n {
        let m = ds.node_machine[u];
        if m < num_machines { members[m].push(u); }
    }

    // The empty disjunctive graph. Machines carrying fewer than two operations hold no disjunction
    // at all: place them immediately so the final graph is fully oriented whatever happens below.
    for m in 0..num_machines {
        ds.machine_seq[m].clear();
        if members[m].len() < 2 { ds.machine_seq[m].extend_from_slice(&members[m]); }
    }

    let mut b = SbpBuf::new(n);
    let mut buf = EvalBuf::new(n);
    let mut sequenced: Vec<bool> = (0..num_machines).map(|m| members[m].len() < 2).collect();
    let mut fixed_order: Vec<usize> = Vec::with_capacity(num_machines);
    let mut pending = sequenced.iter().filter(|&&s| !s).count();

    while pending > 0 {
        // --- ABZ bottleneck selection -------------------------------------------------------
        let mut best_m = NONE_USIZE;
        let mut best_val = 0u32;
        let mut best_seq: Vec<usize> = Vec::new();
        for m in 0..num_machines {
            if sequenced[m] { continue; }
            // `m` is still empty, so excluding it is a no-op — this is exactly the ABZ subproblem
            // data: heads and tails of the graph in which `m`'s operations do not yet constrain
            // each other.
            if !sbp_heads_tails_excl(&ds, m, &mut b) { continue; }
            b.nodes.clear(); b.nodes.extend_from_slice(&members[m]);
            b.r.clear(); b.p.clear(); b.q.clear();
            for &nd in &b.nodes {
                b.r.push(b.head[nd]);
                b.p.push(ds.node_pt[nd]);
                b.q.push(b.tail[nd]);
            }
            let (r, p, q) = (b.r.clone(), b.p.clone(), b.q.clone());
            let Some(val) = sbp_carlier(&r, &p, &q, &mut b) else { continue };
            if b.best_order.len() != b.nodes.len() { continue; }
            if best_m == NONE_USIZE || val > best_val {
                best_val = val;
                best_m = m;
                best_seq.clear();
                for &pos in b.best_order.iter() { best_seq.push(b.nodes[pos]); }
            }
        }
        if best_m == NONE_USIZE { return None; }

        // --- install, then CERTIFY ----------------------------------------------------------
        // Installing an exact one-machine sequence into a partial orientation can close a cycle.
        // This is the classical failure mode of shifting bottleneck, and it is precisely what
        
        ds.machine_seq[best_m].clear();
        ds.machine_seq[best_m].extend_from_slice(&best_seq);
        if !sbp_partial_acyclic(&ds, &mut b) {
            // Fall back on the head order. `head[u] < head[v]` whenever `u` already precedes `v` in
            // the partial graph (processing times are strictly positive), so ordering by head can
            // never contradict an existing arc: acyclic by construction.
            ds.machine_seq[best_m].clear();
            if !sbp_heads_tails_excl(&ds, best_m, &mut b) { return None; }
            let mut order: Vec<usize> = members[best_m].clone();
            order.sort_by(|&x, &y| b.head[x].cmp(&b.head[y]).then(x.cmp(&y)));
            ds.machine_seq[best_m].clear();
            ds.machine_seq[best_m].extend_from_slice(&order);
            if !sbp_partial_acyclic(&ds, &mut b) { return None; }
        }
        sequenced[best_m] = true;
        fixed_order.push(best_m);
        pending -= 1;

        // --- ABZ partial re-optimisation (mode 2 only) --------------------------------------
        // Re-solve the machines already fixed, in the light of the arcs added since. Bounded by
        // construction: one pass over `fixed_order` per insertion, i.e. O(num_machines²) Carlier
        // solves for the whole build, the same order of magnitude as the selection loop above.
        //
        
        // unlock a machine swept earlier in the same pass, which a single pass never revisits; ABZ
        // is defined by the fixed point, not by one pass. `cycles == 1` runs the loop body exactly
        
        //
        // TERMINATION, three independent bounds: (a) a pass that installs nothing breaks out;
        // (b) `cap <= num_machines`; (c) each accepted install strictly decreases `cur_val` of one
        // machine, an integer bounded below. No clock, no randomness — the whole block stays
        // deterministic.
        if reopt {
            let cap = if reopt_cycles == 0 { num_machines } else { reopt_cycles.min(num_machines) };
            for _cycle in 0..cap {
            let mut changed = false;
            for i in 0..fixed_order.len() {
                let m2 = fixed_order[i];
                let k2 = ds.machine_seq[m2].len();
                if k2 < 3 { continue; }
                if !sbp_heads_tails_excl(&ds, m2, &mut b) { continue; }
                b.nodes.clear(); b.nodes.extend_from_slice(&members[m2]);
                b.r.clear(); b.p.clear(); b.q.clear();
                for &nd in &b.nodes {
                    b.r.push(b.head[nd]);
                    b.p.push(ds.node_pt[nd]);
                    b.q.push(b.tail[nd]);
                }
                // Value of the sequence we already hold, in the subproblem's own metric: Carlier is
                // only worth installing if it beats it.
                let cur_val = {
                    let seq: Vec<usize> = ds.machine_seq[m2].clone();
                    let mut t = 0u32;
                    let mut v = 0u32;
                    for &nd in seq.iter() {
                        let pos = match b.nodes.iter().position(|&x| x == nd) { Some(p) => p, None => return None };
                        let s = if t > b.r[pos] { t } else { b.r[pos] };
                        let e = s.saturating_add(b.p[pos]);
                        let c = e.saturating_add(b.q[pos]);
                        if c > v { v = c; }
                        t = e;
                    }
                    v
                };
                let (r, p, q) = (b.r.clone(), b.p.clone(), b.q.clone());
                let Some(val) = sbp_carlier(&r, &p, &q, &mut b) else { continue };
                if val >= cur_val { continue; }
                if b.best_order.len() != b.nodes.len() { continue; }
                b.saved_seq.clear(); b.saved_seq.extend_from_slice(&ds.machine_seq[m2]);
                ds.machine_seq[m2].clear();
                for &pos in b.best_order.iter() { ds.machine_seq[m2].push(b.nodes[pos]); }
                if !sbp_partial_acyclic(&ds, &mut b) {
                    ds.machine_seq[m2].clear();
                    ds.machine_seq[m2].extend_from_slice(&b.saved_seq);
                } else {
                    // Only a CERTIFIED install counts as progress: a rolled-back one leaves the
                    // orientation exactly as it was, so re-sweeping on its account would loop.
                    changed = true;
                }
            }
            if !changed { break; }
            }
        }
    }

    // Every machine is oriented now, so this is a genuine schedule and `eval_disj` — the evaluator
    // every other phase is judged by — certifies it. No tolerance is introduced or widened.
    let (mk, _) = eval_disj(&ds, &mut buf)?;
    let sol = disj_to_solution(pre, &ds, &buf.start).ok()?;
    Some((sol, mk))
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
    let allow_flex_balance = pre.high_flex > 0.60 && pre.jobshopness > 0.38;
    let mut rules: Vec<Rule> = vec![Rule::Adaptive, Rule::BnHeavy, Rule::EndTight, Rule::CriticalPath, Rule::MostWork, Rule::LeastFlex, Rule::Regret, Rule::ShortestProc];
    if allow_flex_balance { rules.push(Rule::FlexBalance); }

    let mut best_makespan = greedy_mk; let mut best_solution: Option<Solution> = Some(greedy_sol.clone()); let mut top_solutions: Vec<(Solution, u32)> = Vec::new();
    push_top_solutions(&mut top_solutions, &greedy_sol, greedy_mk, 15);
    let target_margin: u32 = ((pre.avg_op_min * (0.9 + 0.9*pre.high_flex + 0.6*pre.jobshopness)).max(1.0)) as u32;
    let route_w_base: f64 = if pre.chaotic_like { 0.0 } else { (0.040 + 0.10*pre.high_flex + 0.08*pre.jobshopness).clamp(0.04, 0.22) };

    if pre.flow_route.is_some() && pre.flow_pt_by_job.is_some() {
        if let Ok((sol, mk)) = neh_reentrant_flow_solution(pre, challenge.num_jobs, challenge.num_machines) {
            if mk < best_makespan { best_makespan=mk; best_solution=Some(sol.clone()); save_solution(&sol)?; }
            push_top_solutions(&mut top_solutions, &sol, mk, 15);
        }
    }

    let mut ranked: Vec<(Rule,u32,Solution)> = Vec::with_capacity(rules.len());
    for &rule in &rules {
        let (sol, mk) = construct_solution_conflict(challenge, pre, rule, 0, None, &mut rng, None, None, None, 0.0)?;
        if mk < best_makespan { best_makespan=mk; best_solution=Some(sol.clone()); save_solution(&sol)?; }
        push_top_solutions(&mut top_solutions, &sol, mk, 15); ranked.push((rule, mk, sol));
    }
    ranked.sort_by_key(|x| x.1);
    let r0=ranked[0].0; let r1=ranked.get(1).map(|x|x.0).unwrap_or(r0); let r2=ranked.get(2).map(|x|x.0).unwrap_or(r1);

    let mut rule_best: Vec<u32> = vec![u32::MAX; 9]; let mut rule_tries: Vec<u32> = vec![0u32; 9];
    for (rr,mk,_) in &ranked { let idx=rule_idx(*rr); rule_best[idx]=rule_best[idx].min(*mk); rule_tries[idx]=rule_tries[idx].saturating_add(1); }

    let base = &ranked[0].2;
    let mut learned_jb = Some(job_bias_from_solution(pre, base)?);
    let mut learned_mp = Some(machine_penalty_from_solution(pre, base, challenge.num_machines)?);
    let mut learned_rp = if route_w_base > 0.0 { Some(route_pref_from_solution_lite(pre, base, challenge)?) } else { None };
    let mut learn_updates_left = 4usize;
    let num_restarts = 450usize;

    let mut k_hi = if pre.flex_avg > 8.0 { 6 } else if pre.flex_avg > 6.5 { 4 } else if pre.flex_avg > 4.0 { 5 } else { 6 };
    if pre.jobshopness > 0.60 && k_hi < 6 { k_hi += 1; }
    k_hi = k_hi.min(6).max(2);
    let mut stuck: usize = 0;

    for r in 0..num_restarts {
        let late = r >= (num_restarts*2)/3;
        let (k_min,k_max) = if stuck>170 { (4usize,6usize.min(k_hi)) } else if stuck>90 { (3usize,6usize.min(k_hi.max(4))) } else if stuck>35 { (2usize,k_hi) } else { (2usize,k_hi.min(4)) };
        let rule = if r < 35 { let u: f64=rng.gen(); if allow_flex_balance&&pre.high_flex>0.82&&u<0.10{Rule::FlexBalance}else if u<0.52{r0}else if u<0.80{r1}else if u<0.92{r2}else{rules[rng.gen_range(0..rules.len())]} }
            else { choose_rule_bandit(&mut rng, &rules, &rule_best, &rule_tries, best_makespan, target_margin, stuck, pre.chaotic_like, late) };
        let k = if k_max<=k_min { k_min } else { rng.gen_range(k_min..=k_max) };
        let learn_base = if pre.chaotic_like { 0.0 } else { (0.08+0.22*pre.jobshopness+0.18*pre.high_flex).clamp(0.05,0.42) };
        let learn_boost = (1.0+0.35*((stuck as f64)/120.0).clamp(0.0,1.0)).clamp(1.0,1.35);
        let learn_p = (learn_base*learn_boost).clamp(0.0,0.60);
        let use_learn = learned_jb.is_some() && learned_mp.is_some() && rng.gen::<f64>()<learn_p && (route_w_base==0.0||learned_rp.is_some());
        let target = if best_makespan < (u32::MAX/2) { Some(best_makespan.saturating_add(target_margin)) } else { None };
        let (sol, mk) = if use_learn {
            construct_solution_conflict(challenge, pre, rule, k, target, &mut rng, learned_jb.as_deref(), learned_mp.as_deref(), learned_rp.as_ref(), route_w_base)?
        } else {
            construct_solution_conflict(challenge, pre, rule, k, target, &mut rng, None, None, None, 0.0)?
        };
        let ridx=rule_idx(rule); rule_tries[ridx]=rule_tries[ridx].saturating_add(1); rule_best[ridx]=rule_best[ridx].min(mk);
        if mk < best_makespan {
            best_makespan=mk; best_solution=Some(sol.clone()); save_solution(&sol)?; stuck=0;
            if learn_updates_left > 0 && !pre.chaotic_like {
                learned_jb=Some(job_bias_from_solution(pre,&sol)?); learned_mp=Some(machine_penalty_from_solution(pre,&sol,challenge.num_machines)?);
                if route_w_base>0.0 { learned_rp=Some(route_pref_from_solution_lite(pre,&sol,challenge)?); }
                learn_updates_left-=1;
            }
        } else { stuck=stuck.saturating_add(1); }
        push_top_solutions(&mut top_solutions, &sol, mk, 15);
    }

    let route_w_ls: f64 = if route_w_base>0.0 { (route_w_base*1.40).clamp(route_w_base,0.40) } else { 0.0 };
    let mut refine_results: Vec<(Solution,u32)> = Vec::new();
    for (base_sol, _) in top_solutions.iter() {
        let jb = job_bias_from_solution(pre, base_sol)?;
        let mp = machine_penalty_from_solution(pre, base_sol, challenge.num_machines)?;
        let anti_jb = invert_job_bias_guidance(&jb);
        let anti_mp = invert_machine_penalty_guidance(&mp);
        let rp = if route_w_ls>0.0 { Some(route_pref_from_solution_lite(pre, base_sol, challenge)?) } else { None };
        let target_ls = if best_makespan < (u32::MAX/2) { Some(best_makespan.saturating_add(target_margin/2)) } else { None };
        for attempt in 0..10 {
            let use_anti = attempt % 2 == 1;
            let rule = if pre.chaotic_like {
                match attempt { 0=>Rule::Regret, 1=>Rule::MostWork, 2=>Rule::ShortestProc, 3=>Rule::Adaptive, 4=>Rule::ShortestProc, 5=>Rule::Regret, 6=>Rule::MostWork, 7=>Rule::Adaptive, 8=>Rule::Adaptive, _=>Rule::ShortestProc }
            } else {
                match attempt { 0=>r0, 1=>Rule::Adaptive, 2=>Rule::BnHeavy, 3=>Rule::EndTight, 4=>Rule::Regret, 5=>Rule::CriticalPath, 6=>Rule::LeastFlex, 7=>Rule::MostWork, 8=>if allow_flex_balance{Rule::FlexBalance}else{r1}, _=>r1 }
            };
            let k = match attempt%4 { 0=>2, 1=>3, 2=>4, _=>2 }.min(k_hi);
            let jb_ref: Option<&[f64]> = if use_anti { Some(anti_jb.as_slice()) } else { Some(jb.as_slice()) };
            let mp_ref: Option<&[f64]> = if use_anti { Some(anti_mp.as_slice()) } else { Some(mp.as_slice()) };
            let rp_ref = if use_anti { None } else { rp.as_ref() };
            let route_w = if use_anti { 0.0 } else if rp.is_some() { route_w_ls } else { 0.0 };
            let (sol, mk) = construct_solution_conflict(challenge, pre, rule, k, target_ls, &mut rng, jb_ref, mp_ref, rp_ref, route_w)?;
            if mk < best_makespan { best_makespan=mk; best_solution=Some(sol.clone()); save_solution(&sol)?; }
            refine_results.push((sol, mk));
        }
    }
    for (sol, mk) in refine_results { push_top_solutions(&mut top_solutions, &sol, mk, 15); }

    
    // is driven by `EffortConfig::js_ts_starts` (default 10 ⇒ byte-identical to the baseline when
    // the HP is absent). Only THIS `.min(..)` moves: the `.min(15)` pool caps, the escape-LS
    // `.min(..)` and the greedy/bottleneck start counts (`bn_starts` below) are other phases and
    // stay untouched.
    let ts_starts = top_solutions.len().min(effort.js_ts_starts.max(1));
    let ts_iters = effort.job_shop_iters;
    // t48 L6: evaluation-kernel mode, see EffortConfig::js_ts_fast_eval. Purely a cost knob —
    // mode 1 must leave every tabu trajectory bit-identical and only shorten the wall.
    let fast_eval = effort.js_ts_fast_eval == 1;
    
    // non-improving iterations. 0 = legacy budget-proportional law (byte-identical baseline).
    let div_period = effort.js_ts_div_period;
    
    // SAME `estimate_swap_mk`. 0 = legacy first-reached (byte-identical baseline).
    let tie_mode = effort.js_ts_tie_mode;
    
    // 0 = the single Taillard bound `max(path_u, path_v)` (byte-identical baseline).
    let sel_key = effort.js_ts_sel_key;
    
    // makespan recomputed before the argmin is settled. 0 = pure approximate screen (byte-identical
    // baseline).
    let exact_topk = effort.js_ts_exact_topk;
    
    let traj_decorr = effort.js_ts_traj_decorr;
    
    // and `ts_extra_best` stays `None`, so the final selection is a no-op).
    let extra_draws = effort.js_ts_extra_draws;
    
    let kick_spread = effort.js_ts_kick_spread;
    
    // anywhere, byte-identical. See `EffortConfig::js_ts_sa_accept`. It is inert unless
    // `extra_draws > 0`: with no extra draw there is no call site that receives it.
    let sa_accept = effort.js_ts_sa_accept;
    
    // 0 = legacy skip-the-bottleneck-machines, byte-identical. See `EffortConfig::js_mem_cross_scope`.
    let mem_cross_scope = effort.js_mem_cross_scope;
    
    // byte-identical. See `EffortConfig::js_mem_admit` for the measurement that motivates it.
    let mem_admit = effort.js_mem_admit;
    
    // 0 = the block never runs, byte-identical. See `EffortConfig::js_sbp_rounds`.
    let sbp_rounds = effort.js_sbp_rounds;
    
    // `return Ok(None)`, byte-identical. See `EffortConfig::js_ts_polish_rescue`.
    let polish_rescue = effort.js_ts_polish_rescue;
    
    // See `EffortConfig::js_seed_sbp_construct`.
    let sbp_construct = effort.js_seed_sbp_construct;
    
    // byte-identical; `0` = until the fixed point. Inert unless `sbp_construct == 2`: mode 0 and 1
    // never enter the `if reopt` block that reads it. See `EffortConfig::js_seed_sbp_reopt_cycles`.
    let sbp_reopt_cycles = effort.js_seed_sbp_reopt_cycles;
    
    // departure below runs. See `EffortConfig::js_seed_sbp_dual_cycles`.
    let sbp_dual_cycles = effort.js_seed_sbp_dual_cycles;
    // Best solution found by the constructive-SBP departure ONLY. Kept out of `best_makespan`,
    // `best_solution` and `top_solutions` for the same reason as `ts_extra_best`: every legacy
    // phase must consume exactly the state the baseline gave it.
    let mut sbp_seed_best: Option<(Solution, u32)> = None;
    // Best makespan found by the extra draws ONLY. Deliberately kept out of `best_makespan`,
    // `best_solution` and `top_solutions` so that every phase after the multi-start consumes
    // exactly the state the baseline gave it; see `EffortConfig::js_ts_extra_draws`.
    let mut ts_extra_best: Option<(Solution, u32)> = None;
    let ts_tenure =((pre.total_ops as f64).sqrt() as usize * (100 + (pre.load_cv * 60.0) as usize) / 100).clamp(8, 24);
    
    // `top_solutions` is ordered by makespan only, and the makespan PREFIX is a poor proxy for
    
    // (exact dedup was a no-op) yet the pool still collapses into one basin of attraction, so
    // several of the ten tabu runs re-explore ground already covered. Mode 1 replaces the prefix
    // by a farthest-point sub-selection over the WHOLE pool (cap 15): anchor on rank 1 — the
    // incumbent must always be refined — then greedily add the candidate maximising its minimum
    // distance to the seeds already picked. Ranks 11..15, today ignored, become free coverage.
    // Ties keep the lowest index, i.e. the better makespan, so the rule stays deterministic.
    //
    
    // `solution_machine_signature` (a verbatim port of the `fjsp_high` helper, which compares the
    // MACHINE ASSIGNED to each op). That signature is provably CONSTANT on this track:
    // `Scenario::JOB_SHOP` sets `avg_op_flexibility = 1.0`, and the generator only widens
    // eligibility `if avg_op_flexibility > 1.0`, so every op has exactly one eligible machine and
    // all pool entries share the same assignment vector. Every distance was 0, the greedy fell
    
    
    // mechanism. Here the distance runs on `ts_seed_key` — the per-machine op ORDER, i.e. the
    // disjunctive orientation, which is exactly what `tabu_search_phase` consumes and the only
    // thing that varies on a fixed-assignment instance.
    //
    // Iso-work by construction: still EXACTLY `ts_starts` runs of `ts_iters` tabu iterations.
    // The selection is one O(pool² · total_ops) pass (~15²×250 ≈ 6e4 ops) against the 2.4M tabu
    
    // regression (845 110 ms vs 862 120 ms baseline).
    let ts_seed_idx: Vec<usize> = if ts_starts == 0 || effort.js_seed_select_mode == 0 {
        (0..ts_starts).collect()
    } else if effort.js_seed_select_mode == 2 {
        // Liveness control: the legacy prefix with its LAST slot forced to pool rank `ts_starts`
        // (the first rank the baseline never uses). It MUST move Q whenever the pool holds more
        // than `ts_starts` entries — that is what makes a mode-1 null readable, and it is also
        
        let mut picked: Vec<usize> = (0..ts_starts).collect();
        if top_solutions.len() > ts_starts {
            picked[ts_starts - 1] = ts_starts;
        }
        picked
    } else {
        let pool_len = top_solutions.len();
        let keys: Vec<Vec<u32>> = top_solutions.iter()
            .map(|(s, _)| ts_seed_key(pre, challenge.num_machines, s))
            .collect();
        let mut picked: Vec<usize> = Vec::with_capacity(ts_starts);
        let mut taken = vec![false; pool_len];
        // `min_d[i]` = distance from candidate i to the closest already-picked seed.
        let mut min_d = vec![u32::MAX; pool_len];
        picked.push(0);
        taken[0] = true;
        while picked.len() < ts_starts {
            let last = *picked.last().unwrap();
            let mut best: Option<usize> = None;
            let mut best_d = 0u32;
            for i in 0..pool_len {
                if taken[i] { continue; }
                let d = ts_seed_key_distance(&keys[last], &keys[i]);
                if d < min_d[i] { min_d[i] = d; }
                if best.is_none() || min_d[i] > best_d {
                    best = Some(i);
                    best_d = min_d[i];
                }
            }
            let Some(b) = best else { break };
            picked.push(b);
            taken[b] = true;
        }
        picked
    };
    {
        let mut ts_results: Vec<(Solution, u32)> = Vec::new();
        for (rank, &idx) in ts_seed_idx.iter().enumerate() {
            
            // solution and its own tabu list, but they all derive `pseed` from the same
            
            // measured what that costs: on the median instances the outcome is byte-identical at
            // `js_ts_starts` 4, 10 and 12 — the `min` two lines below is a `min` over an effective
            
            // the same seeds is real and large (−646 and −1457 for a single rule change), i.e. this
            // search does not have one reachable optimum per instance, it has a distribution of
            // them, and we currently draw from it once. Salting the stream by RANK draws `ts_starts`
            // times instead, at exactly zero added work (the iteration count, the number of starts
            // and the tenure law are all untouched — `hp:ts_iters_funding` is dead, every move on
            // this track has to be ISO-WORK).
            //
            // Rank 0 gets salt 0, so the champion trajectory stays verbatim in the portfolio: the
            // draws are ADDED to it, never substituted for it.
            let traj_salt: u64 = match traj_decorr {
                0 => 0,
                1 => (rank as u64).wrapping_mul(0x9E3779B97F4A7C15),
                // Second draw of the SAME mechanism: a different multiplier, nothing else. This is
                // the bilateral control — a gain that shows up on 1 but not on 2 is a lottery
                // ticket, not a mechanism, and must not be promoted.
                _ => (rank as u64).wrapping_mul(0xD1B54A32D192ED03),
            };
            let res = {
                let base_sol = &top_solutions[idx].0;
                tabu_search_phase(pre, challenge, base_sol, ts_iters, ts_tenure, fast_eval, div_period, tie_mode, sel_key, exact_topk, traj_salt, kick_spread, 0, 0)?
            };
            if let Some((sol2, mk2)) = res {
                if mk2 < best_makespan { best_makespan=mk2; best_solution=Some(sol2.clone()); save_solution(&sol2)?; }
                ts_results.push((sol2, mk2));
            }
        }
        
        // purpose: `top_solutions` is still in the exact state the legacy starts read, so an extra
        
        // rank did. The only difference with that start is the salt of the tenure stream — the same
        
        // path self-controlled: if it executes at all, it produces a genuinely different walk.
        //
        // Nothing here writes `top_solutions`, `best_makespan`, `best_solution` or `ts_results`, and
        // nothing here calls `save_solution`. That is the whole design: every phase downstream of
        // this block — bottleneck, ILS, memetic, escape-LS, final polish — is bit-identical to the
        // baseline run, and the extra draws can only ever be consulted at the last statement of
        // `solve`. Adding a draw is therefore monotone on Q, per nonce, by construction.
        for e in 0..extra_draws {
            if ts_seed_idx.is_empty() { break; }
            let idx = ts_seed_idx[e % ts_seed_idx.len()];
            
            // mechanisms can never alias each other in a later combination.
            let extra_salt: u64 = ((e as u64) + 1)
                .wrapping_mul(0x9E3779B97F4A7C15)
                ^ 0x517CC1B727220A95;
            let res = {
                let base_sol = &top_solutions[idx].0;
                tabu_search_phase(pre, challenge, base_sol, ts_iters, ts_tenure, fast_eval, div_period, tie_mode, sel_key, exact_topk, extra_salt, kick_spread, sa_accept, 0)?
            };
            if let Some((sol2, mk2)) = res {
                let better = match &ts_extra_best { None => true, Some((_, bmk)) => mk2 < *bmk };
                if better { ts_extra_best = Some((sol2, mk2)); }
            }
        }
        for (sol2, mk2) in ts_results {
            push_top_solutions(&mut top_solutions, &sol2, mk2, 20);
        }
    }

    {
        let bn_starts = top_solutions.len().min(if pre.total_ops > 240 { 6 } else { 8 });
        let mut shared_bn_buf: Option<(usize, EvalBuf)> = None;
        let mut bn_results: Vec<(Solution, u32)> = Vec::new();
        let mut crit_pos: Vec<usize> = Vec::with_capacity(16);
        let mut move_pos: Vec<usize> = Vec::with_capacity(18);
        let mut machine_total_pt: Vec<u64> = vec![0u64; challenge.num_machines];
        let mut m_rank: Vec<usize> = Vec::with_capacity(challenge.num_machines);
        let relocate_span = if pre.total_ops > 260 { 2usize } else if pre.jobshopness > 0.60 { 4usize } else { 3usize };
        let max_rounds = if pre.total_ops > 260 { 2usize } else { 3usize };
        for idx in 0..bn_starts {
            let mut ds = {
                let base_sol = &top_solutions[idx].0;
                match build_disj_from_solution(pre, challenge, base_sol) { Ok(d) => d, Err(_) => continue }
            };
            if shared_bn_buf.as_ref().map_or(true, |(n, _)| *n != ds.n) {
                shared_bn_buf = Some((ds.n, EvalBuf::new(ds.n)));
            }
            let (_, buf) = shared_bn_buf.as_mut().unwrap();
            let Some((mut cur_mk, mut mk_node)) = eval_disj(&ds, buf) else { continue };

            machine_total_pt.fill(0);
            for m in 0..challenge.num_machines {
                if m < ds.machine_seq.len() {
                    for &nd in &ds.machine_seq[m] {
                        if nd < ds.node_pt.len() {
                            machine_total_pt[m] = machine_total_pt[m].saturating_add(ds.node_pt[nd] as u64);
                        }
                    }
                }
            }
            m_rank.clear();
            for m in 0..challenge.num_machines {
                if m < ds.machine_seq.len() && ds.machine_seq[m].len() > 1 {
                    m_rank.push(m);
                }
            }
            m_rank.sort_by(|&a, &b| machine_total_pt[b].cmp(&machine_total_pt[a]));

            let num_bn_ls = m_rank.len().min(3);
            let mut any_improved = false;

            for _ in 0..max_rounds {
                let mut round_improved = false;
                for bi in 0..num_bn_ls {
                    let m = m_rank[bi];
                    let seq_cap = ds.machine_seq[m].len().min(18);
                    if seq_cap < 2 { continue; }

                    crit_pos.clear();
                    {
                        let prefix = &ds.machine_seq[m][..seq_cap];
                        let mut u = mk_node;
                        while u != NONE_USIZE {
                            if let Some(pos) = prefix.iter().position(|&nd| nd == u) {
                                crit_pos.push(pos);
                            }
                            u = buf.best_pred[u];
                        }
                    }
                    if crit_pos.is_empty() { continue; }
                    crit_pos.sort_unstable();
                    crit_pos.dedup();

                    move_pos.clear();
                    for &pos in &crit_pos {
                        move_pos.push(pos);
                        if pos > 0 { move_pos.push(pos - 1); }
                        if pos + 1 < seq_cap { move_pos.push(pos + 1); }
                    }
                    move_pos.sort_unstable();
                    move_pos.dedup();
                    if move_pos.is_empty() { continue; }

                    let mut best_move: Option<(usize, usize)> = None;
                    let mut best_move_mk = cur_mk;

                    for &from in &move_pos {
                        let lo = from.saturating_sub(relocate_span);
                        let hi = (from + relocate_span).min(seq_cap - 1);
                        for to in lo..=hi {
                            if to == from { continue; }
                            {
                                let seq = &mut ds.machine_seq[m][..seq_cap];
                                relocate_machine_seq(seq, from, to);
                            }
                            if let Some((new_mk, _)) = eval_disj(&ds, buf) {
                                if new_mk < best_move_mk {
                                    best_move_mk = new_mk;
                                    best_move = Some((from, to));
                                }
                            }
                            {
                                let seq = &mut ds.machine_seq[m][..seq_cap];
                                relocate_machine_seq(seq, to, from);
                            }
                        }
                    }

                    let Some((from, to)) = best_move else {
                        let Some((restored_mk, restored_node)) = eval_disj(&ds, buf) else { continue };
                        cur_mk = restored_mk;
                        mk_node = restored_node;
                        continue;
                    };

                    {
                        let seq = &mut ds.machine_seq[m][..seq_cap];
                        relocate_machine_seq(seq, from, to);
                    }
                    match eval_disj(&ds, buf) {
                        Some((new_mk, new_node)) if new_mk < cur_mk => {
                            cur_mk = new_mk;
                            mk_node = new_node;
                            any_improved = true;
                            round_improved = true;
                        }
                        _ => {
                            {
                                let seq = &mut ds.machine_seq[m][..seq_cap];
                                relocate_machine_seq(seq, to, from);
                            }
                            let Some((restored_mk, restored_node)) = eval_disj(&ds, buf) else { continue };
                            cur_mk = restored_mk;
                            mk_node = restored_node;
                        }
                    }
                }
                if !round_improved { break; }
            }

            if any_improved {
                if let Ok(sol_bn) = disj_to_solution(pre, &ds, &buf.start) {
                    if cur_mk < best_makespan { best_makespan=cur_mk; best_solution=Some(sol_bn.clone()); save_solution(&sol_bn)?; }
                    bn_results.push((sol_bn, cur_mk));
                }
            }
        }
        for (sol_bn, mk_bn) in bn_results {
            push_top_solutions(&mut top_solutions, &sol_bn, mk_bn, 20);
        }
    }

    {
        let ils_starts = top_solutions.len().min(6);
        let mut ils_results: Vec<(Solution, u32)> = Vec::new();
        for idx in 0..ils_starts {
            let ls_res = {
                let base_sol = &top_solutions[idx].0;
                critical_block_move_local_search_ex(pre, challenge, base_sol, 5, 400, 120)
            };
            if let Ok(Some((ls_sol, ls_mk))) = ls_res {
                if ls_mk < best_makespan { best_makespan=ls_mk; best_solution=Some(ls_sol.clone()); save_solution(&ls_sol)?; }
                ils_results.push((ls_sol.clone(), ls_mk));
                let reopt_res = perturb_and_reoptimize_ils(pre, challenge, &ls_sol, 2, 3, 220, 80)?;
                if let Some((sol3, mk3)) = reopt_res {
                    if mk3 < best_makespan { best_makespan=mk3; best_solution=Some(sol3.clone()); save_solution(&sol3)?; }
                    ils_results.push((sol3, mk3));
                }
            }
        }
        for (sol, mk) in ils_results {
            push_top_solutions(&mut top_solutions, &sol, mk, 20);
        }
    }

    {
        let num_machines = challenge.num_machines;

        let mut machine_rank: Vec<(usize, f64)> = (0..num_machines).map(|m| {
            let scar = if m < pre.machine_scarcity.len() { pre.machine_scarcity[m] } else { 1.0 };
            (m, scar)
        }).collect();
        machine_rank.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let num_bn = (num_machines / 5).max(3).min(8);
        let mut is_bottleneck = vec![false; num_machines];
        for i in 0..num_bn { is_bottleneck[machine_rank[i].0] = true; }

        let pop_cap = 10usize;
        let mut mem_pop: Vec<(Solution, u32)> = top_solutions.iter().take(pop_cap).cloned().collect();
        let mut mem_ds: Vec<Option<DisjSchedule>> = vec![None; mem_pop.len()];

        let num_generations = 18usize;
        let mut gen_no_improve = 0usize;
        let max_gen_no_improve = 12usize;
        let mut shared_mem_buf: Option<(usize, EvalBuf)> = None;
        let mut cross_pos: Vec<usize> = Vec::new();
        let mut cross_sum: Vec<u32> = Vec::new();
        let mut cross_cnt: Vec<u16> = Vec::new();
        let mut cross_pairs: Vec<(usize, usize, i32)> = Vec::new();
        let elite_cap = 4usize;

        
        // generation loop; untouched when `mem_admit == 0`.
        let mut admit_buf: Option<(usize, EvalBuf)> = None;
        let mut admit_saved: Vec<usize> = Vec::new();
        for gen in 0..num_generations {
            if gen_no_improve >= max_gen_no_improve { break; }
            let cur_pop = mem_pop.len();
            if cur_pop < 2 { break; }

            let use_mutation = gen % 6 == 5;

            let ia = {
                let a = rng.gen_range(0..cur_pop);
                let b = rng.gen_range(0..cur_pop);
                if mem_pop[a].1 <= mem_pop[b].1 { a } else { b }
            };
            let ib = {
                let mut b = rng.gen_range(0..cur_pop);
                if b == ia { b = (b + 1) % cur_pop; }
                let c = rng.gen_range(0..cur_pop);
                let c = if c == ia { (c + 1) % cur_pop } else { c };
                if mem_pop[b].1 <= mem_pop[c].1 { b } else { c }
            };

            let mk_a = mem_pop[ia].1;
            let mk_b = mem_pop[ib].1;

            if mem_ds[ia].is_none() {
                let built = match build_disj_from_solution(pre, challenge, &mem_pop[ia].0) {
                    Ok(d) => d,
                    Err(_) => { gen_no_improve += 1; continue; }
                };
                mem_ds[ia] = Some(built);
            }
            if mem_ds[ib].is_none() {
                let built = match build_disj_from_solution(pre, challenge, &mem_pop[ib].0) {
                    Ok(d) => d,
                    Err(_) => { gen_no_improve += 1; continue; }
                };
                mem_ds[ib] = Some(built);
            }

            let mut elite_idx: Vec<usize> = (0..cur_pop).collect();
            elite_idx.sort_unstable_by_key(|&i| mem_pop[i].1);
            elite_idx.truncate(elite_cap.min(cur_pop));
            let mut elite_failed = false;
            for &ei in &elite_idx {
                if mem_ds[ei].is_none() {
                    match build_disj_from_solution(pre, challenge, &mem_pop[ei].0) {
                        Ok(d) => mem_ds[ei] = Some(d),
                        Err(_) => {
                            elite_failed = true;
                            break;
                        }
                    }
                }
            }
            if elite_failed {
                gen_no_improve += 1;
                continue;
            }

            let (better_idx, worse_idx) = if mk_a <= mk_b { (ia, ib) } else { (ib, ia) };
            let mut child_ds = {
                let better_ds = mem_ds[better_idx].as_ref().unwrap();
                let worse_ds = mem_ds[worse_idx].as_ref().unwrap();
                let mut child_ds = better_ds.clone();

                if cross_pos.len() != child_ds.n {
                    cross_pos.resize(child_ds.n, 0);
                    cross_sum.resize(child_ds.n, 0);
                    cross_cnt.resize(child_ds.n, 0);
                }

                for m in 0..num_machines {
                    
                    // { continue; }`, reproduced EXACTLY by `mem_cross_scope == 0`, so the default
                    // binary is byte-identical by construction rather than by inspection. See
                    // `EffortConfig::js_mem_cross_scope` for why skipping the bottleneck machines is
                    // anti-aligned with an instance family whose makespan the bottleneck machine sets.
                    let skip_machine = match mem_cross_scope {
                        1 => false,
                        2 => !is_bottleneck[m],
                        _ => is_bottleneck[m],
                    };
                    if skip_machine { continue; }
                    if m >= better_ds.machine_seq.len() || m >= worse_ds.machine_seq.len() || m >= child_ds.machine_seq.len() { continue; }
                    if better_ds.machine_seq[m].len() != worse_ds.machine_seq[m].len() { continue; }

                    let mut elite_machine_seqs: Vec<&[usize]> = Vec::with_capacity(elite_idx.len());
                    for &ei in &elite_idx {
                        if let Some(ds_e) = mem_ds[ei].as_ref() {
                            if m < ds_e.machine_seq.len() && ds_e.machine_seq[m].len() == child_ds.machine_seq[m].len() {
                                elite_machine_seqs.push(&ds_e.machine_seq[m]);
                            }
                        }
                    }

                    if mem_admit == 0 {
                        inherit_machine_consensus_order(
                            &mut child_ds.machine_seq[m],
                            &better_ds.machine_seq[m],
                            &worse_ds.machine_seq[m],
                            &elite_machine_seqs,
                            &mut cross_pos,
                            &mut cross_sum,
                            &mut cross_cnt,
                            &mut cross_pairs,
                        );
                        continue;
                    }

                    
                    // is entered feasible (it is a clone of the better parent) and every machine
                    // that would break that invariant is rolled back, so the loop exits with a
                    // child that `eval_disj` accepts, by construction.
                    admit_saved.clear();
                    admit_saved.extend_from_slice(&child_ds.machine_seq[m]);
                    inherit_machine_consensus_order(
                        &mut child_ds.machine_seq[m],
                        &better_ds.machine_seq[m],
                        &worse_ds.machine_seq[m],
                        &elite_machine_seqs,
                        &mut cross_pos,
                        &mut cross_sum,
                        &mut cross_cnt,
                        &mut cross_pairs,
                    );
                    if child_ds.machine_seq[m] == admit_saved { continue; }
                    if admit_buf.as_ref().map_or(true, |(n, _)| *n != child_ds.n) {
                        admit_buf = Some((child_ds.n, EvalBuf::new(child_ds.n)));
                    }
                    let (_, abuf) = admit_buf.as_mut().unwrap();
                    if eval_disj(&child_ds, abuf).is_none() {
                        child_ds.machine_seq[m].clear();
                        child_ds.machine_seq[m].extend_from_slice(&admit_saved);
                    }
                }

                child_ds
            };

            if use_mutation {
                let non_bn_machines: Vec<usize> = (0..num_machines).filter(|&m| !is_bottleneck[m] && child_ds.machine_seq[m].len() > 1).collect();
                if !non_bn_machines.is_empty() {
                    for _ in 0..2 {
                        let m = non_bn_machines[rng.gen_range(0..non_bn_machines.len())];
                        let seq_len = child_ds.machine_seq[m].len();
                        if seq_len > 1 {
                            let pos = rng.gen_range(0..seq_len - 1);
                            child_ds.machine_seq[m].swap(pos, pos + 1);
                        }
                    }
                }
                let bn_machines: Vec<usize> = (0..num_machines).filter(|&m| is_bottleneck[m] && child_ds.machine_seq[m].len() > 1).collect();
                if !bn_machines.is_empty() {
                    let m = bn_machines[rng.gen_range(0..bn_machines.len())];
                    let seq_len = child_ds.machine_seq[m].len();
                    if seq_len > 1 {
                        let pos = rng.gen_range(0..seq_len - 1);
                        child_ds.machine_seq[m].swap(pos, pos + 1);
                    }
                }
            }

            if shared_mem_buf.as_ref().map_or(true, |(n, _)| *n != child_ds.n) {
                shared_mem_buf = Some((child_ds.n, EvalBuf::new(child_ds.n)));
            }
            let (_, child_buf) = shared_mem_buf.as_mut().unwrap();
            let ls_mk = match critical_block_move_local_search_ex_disj(&mut child_ds, child_buf, 4, 250, 100) {
                Some(mk) => mk,
                None => match eval_disj(&child_ds, child_buf) {
                    Some((mk, _)) => mk,
                    None => { gen_no_improve += 1; continue; }
                },
            };
            let mut ls_sol = match disj_to_solution(pre, &child_ds, &child_buf.start) {
                Ok(s) => s,
                Err(_) => { gen_no_improve += 1; continue; }
            };
            
            // Under `mem_admit <= 1` this block is skipped and `ls_mk` keeps its legacy value.
            let mut ls_mk = ls_mk;
            if mem_admit >= 2 {
                let child_ts_iters = if mem_admit == 3 { ts_iters / 8 } else { ts_iters / 24 };
                if child_ts_iters > 0 {
                    if let Some((ts_sol, ts_mk)) = tabu_search_phase(pre, challenge, &ls_sol, child_ts_iters, ts_tenure, fast_eval, div_period, tie_mode, sel_key, exact_topk, 0, kick_spread, 0, 0)? {
                        if ts_mk < ls_mk {
                            ls_sol = ts_sol;
                            ls_mk = ts_mk;
                        }
                    }
                }
            }

            if ls_mk < best_makespan {
                best_makespan = ls_mk;
                best_solution = Some(ls_sol.clone());
                save_solution(&ls_sol)?;
                gen_no_improve = 0;
            } else {
                gen_no_improve += 1;
            }

            push_top_solutions(&mut top_solutions, &ls_sol, ls_mk, 20);

            if cur_pop >= pop_cap {
                let worst_idx = mem_pop.iter().enumerate().max_by_key(|(_, (_, mk))| *mk).map(|(i, _)| i).unwrap_or(cur_pop - 1);
                if ls_mk < mem_pop[worst_idx].1 {
                    mem_pop[worst_idx] = (ls_sol, ls_mk);
                    mem_ds[worst_idx] = Some(child_ds.clone());
                }
            } else {
                mem_pop.push((ls_sol, ls_mk));
                mem_ds.push(Some(child_ds.clone()));
            }
        }

        let mem_best: Vec<Solution> = {
            let mut sorted = mem_pop.clone();
            sorted.sort_by_key(|(_, mk)| *mk);
            sorted.into_iter().take(3).map(|(s, _)| s).collect()
        };
        for base_sol in &mem_best {
            
            // run, so there is no sibling trajectory to decorrelate it from.
            if let Some((ts_sol, ts_mk)) = tabu_search_phase(pre, challenge, base_sol, ts_iters / 3, ts_tenure, fast_eval, div_period, tie_mode, sel_key, exact_topk, 0, kick_spread, 0, 0)? {
                if ts_mk < best_makespan {
                    best_makespan = ts_mk;
                    best_solution = Some(ts_sol.clone());
                    save_solution(&ts_sol)?;
                }
                push_top_solutions(&mut top_solutions, &ts_sol, ts_mk, 20);
            }
        }
    }

    
    // legacy code never wrote `best_makespan` back after the final polish (it did not need to, the
    // value was dead); the extra-draw comparison below does need it, so it is mirrored here. Under
    // `extra_draws == 0` this variable is written and never read, and every branch below is the
    // legacy one, so the run stays byte-identical.
    
    // it ONE tabu departure. Placed here so that every legacy phase above has already run against
    // the exact state the baseline gave it; the block writes neither `top_solutions`, nor
    // `best_makespan`, nor `best_solution`, calls no `save_solution` and touches no RNG stream
    // (`sbp_construct_seed` is deterministic and draws nothing), so the whole run above is
    // bit-identical to the champion whatever it produces. Its result is consulted by ONE strict `<`
    
    // monotone on Q, per nonce, by construction.
    //
    
    // (`(e+1) * K ^ C`), so the three mechanisms can never alias each other in a later combination.
    if sbp_construct > 0 {
        if let Some((seed_sol, _seed_mk)) = sbp_construct_seed(pre, challenge, &greedy_sol, sbp_construct == 2, sbp_reopt_cycles) {
            let res = tabu_search_phase(pre, challenge, &seed_sol, ts_iters, ts_tenure, fast_eval, div_period, tie_mode, sel_key, exact_topk, 0x3C79AC492BA7B653, kick_spread, 0, 0)?;
            if let Some((sol5, mk5)) = res {
                sbp_seed_best = Some((sol5, mk5));
            }
        }

        
        // in every respect but the re-optimisation cap it is built with. It is not a diversifier:
        // both seeds are produced by the same deterministic ABZ, drawing nothing, and both are run
        // to completion — the reduction below is on the FINAL makespan, never on the seed makespan
        
        //
        // The fence is the one every added block on this track uses: `sbp_construct_seed` and
        // `tabu_search_phase` write neither `top_solutions`, nor `best_makespan`, nor
        // `best_solution`, call no `save_solution`, and take their randomness from the explicit salt
        // argument rather than a shared stream. The result lands in the same private slot as the
        // first departure behind a STRICT `<`, and that slot is consulted by ONE strict `<` at the
        // last statement of `solve`. Under `sbp_dual_cycles == 0` the block is skipped entirely and
        
        //
        // The salt is the SAME constant as the first departure on purpose — the two calls differ
        
        // `ts_mk = 11382` on n23. A fresh salt would be an unmeasured trajectory.
        if sbp_dual_cycles > 0 && sbp_dual_cycles != sbp_reopt_cycles {
            if let Some((seed_sol2, _seed_mk2)) = sbp_construct_seed(pre, challenge, &greedy_sol, sbp_construct == 2, sbp_dual_cycles) {
                let res2 = tabu_search_phase(pre, challenge, &seed_sol2, ts_iters, ts_tenure, fast_eval, div_period, tie_mode, sel_key, exact_topk, 0x3C79AC492BA7B653, kick_spread, 0, 0)?;
                if let Some((sol6, mk6)) = res2 {
                    let better = match sbp_seed_best.as_ref() {
                        Some((_, cur_mk)) => mk6 < *cur_mk,
                        None => true,
                    };
                    if better { sbp_seed_best = Some((sol6, mk6)); }
                }
            }
        }
    }

    let mut final_mk = best_makespan;
    if let Some(final_best) = best_solution.as_ref() {
        
        //
        
        // `mid_abort_rescue`. This is what makes it monotone: this polish is the LAST legacy phase,
        // its result is consumed by the single `if mk4 < best_makespan` below, and it writes no
        // seed, no `top_solutions` entry, no `ts_results` and no RNG stream. Rescuing it therefore
        
        // ("contaminated the solution pool or shifted RNG state") is structurally excluded, and the
        // `<` is STRICT so a rescued-but-worse run is discarded exactly as a `None` would be.
        if let Some((sol4, mk4)) = tabu_search_phase(pre, challenge, final_best, ts_iters, ts_tenure, fast_eval, div_period, tie_mode, sel_key, exact_topk, 0, kick_spread, 0, polish_rescue)? {
            if mk4 < best_makespan { best_solution=Some(sol4.clone()); save_solution(&sol4)?; final_mk = mk4; }
        }
    }

    
    // after every legacy phase, so nothing downstream of it exists to contaminate; it writes neither
    // `top_solutions` nor `best_makespan`, calls no `save_solution`, and draws no randomness, so the
    // whole run above is bit-identical to the baseline whatever it does. `final_mk` is the makespan
    // of the solution currently held, and the swap below is a STRICT decrease certified by
    // `eval_disj` on the full disjunctive graph — the same evaluator every other phase is judged by,
    // with no tolerance introduced anywhere. Q is therefore `>=` baseline on every nonce by
    // construction. Under `sbp_rounds == 0` the block is skipped entirely.
    if sbp_rounds > 0 {
        let sbp_res = best_solution.as_ref().and_then(|cur| sbp_reoptimize(pre, challenge, cur, sbp_rounds));
        if let Some((sbp_sol, sbp_mk)) = sbp_res {
            if sbp_mk < final_mk { final_mk = sbp_mk; best_solution = Some(sbp_sol); }
        }
    }

    
    // absent, so this is a no-op on the baseline. When present it is a strict `<`: an extra draw
    // replaces the incumbent only if it is genuinely shorter, hence "best, never current".
    if let Some((extra_sol, extra_mk)) = ts_extra_best {
        if extra_mk < final_mk { best_solution = Some(extra_sol); final_mk = extra_mk; }
    }

    
    // is absent, so this is a no-op on the baseline. Strict `<`: the ABZ basin replaces the
    // incumbent only if it is genuinely shorter — "best, never current".
    if let Some((sbp_sol, sbp_mk)) = sbp_seed_best {
        if sbp_mk < final_mk { best_solution = Some(sbp_sol); }
    }

    if let Some(sol) = best_solution { save_solution(&sol)?; }
    Ok(())
}}


pub mod solver {
use anyhow::Result;
use serde_json::{Map, Value};
use tig_challenges::job_scheduling::*;

use super::types::EffortConfig;
use super::preprocess::build_pre;
use super::job_shop;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Track {
    FlowShop,
    HybridFlowShop,
    JobShop,
    FjspMedium,
    FjspHigh,
}

fn parse_track(hyperparameters: &Option<Map<String, Value>>) -> Track {
    if let Some(map) = hyperparameters {
        if let Some(Value::String(s)) = map.get("track") {
            return match s.to_lowercase().as_str() {
                "flow_shop" | "flow" => Track::FlowShop,
                "hybrid_flow_shop" | "hybrid" => Track::HybridFlowShop,
                "job_shop" | "job" => Track::JobShop,
                "fjsp_medium" | "medium" => Track::FjspMedium,
                "fjsp_high" | "high" | "fjsp" => Track::FjspHigh,
                _ => Track::FjspHigh,
            };
        }
    }
    Track::FjspHigh
}

fn parse_effort(hyperparameters: &Option<Map<String, Value>>) -> EffortConfig {
    let mut cfg = EffortConfig::default_effort();
    if let Some(map) = hyperparameters {
        if let Some(Value::Number(n)) = map.get("job_shop_iters") {
            if let Some(v) = n.as_u64() {
                cfg = cfg.with_job_shop_iters(v as usize);
            }
        }
        if let Some(Value::Number(n)) = map.get("hybrid_flow_shop_iters") {
            if let Some(v) = n.as_u64() {
                cfg = cfg.with_hybrid_flow_shop_iters(v as usize);
            }
        }
        if let Some(Value::Number(n)) = map.get("fjsp_medium_iters") {
            if let Some(v) = n.as_u64() {
                cfg = cfg.with_fjsp_medium_iters(v as usize);
            }
        }
        if let Some(Value::Number(n)) = map.get("js_seed_select_mode") {
            if let Some(v) = n.as_u64() {
                cfg = cfg.with_js_seed_select_mode(v as usize);
            }
        }
        if let Some(Value::Number(n)) = map.get("js_ts_fast_eval") {
            if let Some(v) = n.as_u64() {
                cfg = cfg.with_js_ts_fast_eval(v as usize);
            }
        }
        if let Some(Value::Number(n)) = map.get("js_ts_starts") {
            if let Some(v) = n.as_u64() {
                cfg = cfg.with_js_ts_starts(v as usize);
            }
        }
        if let Some(Value::Number(n)) = map.get("js_ts_div_period") {
            if let Some(v) = n.as_u64() {
                cfg = cfg.with_js_ts_div_period(v as usize);
            }
        }
        if let Some(Value::Number(n)) = map.get("js_ts_tie_mode") {
            if let Some(v) = n.as_u64() {
                cfg = cfg.with_js_ts_tie_mode(v as usize);
            }
        }
        if let Some(Value::Number(n)) = map.get("js_ts_sel_key") {
            if let Some(v) = n.as_u64() {
                cfg = cfg.with_js_ts_sel_key(v as usize);
            }
        }
        if let Some(Value::Number(n)) = map.get("js_ts_exact_topk") {
            if let Some(v) = n.as_u64() {
                cfg = cfg.with_js_ts_exact_topk(v as usize);
            }
        }
        if let Some(Value::Number(n)) = map.get("js_ts_traj_decorr") {
            if let Some(v) = n.as_u64() {
                cfg = cfg.with_js_ts_traj_decorr(v as usize);
            }
        }
        if let Some(Value::Number(n)) = map.get("js_ts_extra_draws") {
            if let Some(v) = n.as_u64() {
                cfg = cfg.with_js_ts_extra_draws(v as usize);
            }
        }
        if let Some(Value::Number(n)) = map.get("js_ts_sa_accept") {
            if let Some(v) = n.as_u64() {
                cfg = cfg.with_js_ts_sa_accept(v as usize);
            }
        }
        if let Some(Value::Number(n)) = map.get("js_mem_admit") {
            if let Some(v) = n.as_u64() {
                cfg = cfg.with_js_mem_admit(v as usize);
            }
        }
        if let Some(Value::Number(n)) = map.get("js_mem_cross_scope") {
            if let Some(v) = n.as_u64() {
                cfg = cfg.with_js_mem_cross_scope(v as usize);
            }
        }
        if let Some(Value::Number(n)) = map.get("js_ts_kick_spread") {
            if let Some(v) = n.as_u64() {
                cfg = cfg.with_js_ts_kick_spread(v as usize);
            }
        }
        if let Some(Value::Number(n)) = map.get("js_sbp_rounds") {
            if let Some(v) = n.as_u64() {
                cfg = cfg.with_js_sbp_rounds(v as usize);
            }
        }
        if let Some(Value::Number(n)) = map.get("js_ts_polish_rescue") {
            if let Some(v) = n.as_u64() {
                cfg = cfg.with_js_ts_polish_rescue(v as usize);
            }
        }
        if let Some(Value::Number(n)) = map.get("js_seed_sbp_construct") {
            if let Some(v) = n.as_u64() {
                cfg = cfg.with_js_seed_sbp_construct(v as usize);
            }
        }
        if let Some(Value::Number(n)) = map.get("js_seed_sbp_dual_cycles") {
            if let Some(v) = n.as_u64() {
                cfg = cfg.with_js_seed_sbp_dual_cycles(v as usize);
            }
        }
        if let Some(Value::Number(n)) = map.get("js_seed_sbp_reopt_cycles") {
            if let Some(v) = n.as_u64() {
                cfg = cfg.with_js_seed_sbp_reopt_cycles(v as usize);
            }
        }
    }
    cfg
}

pub fn solve_challenge(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
) -> Result<()> {
    let pre = build_pre(challenge)?;
    let track = parse_track(hyperparameters);
    let effort = parse_effort(hyperparameters);

    {
            job_shop::solve(challenge, save_solution, &pre, &effort)
        }
}

pub fn help() {
    println!("scheduling engine hyperparameters");
    println!();
    println!("track (string):");
    println!("  selects which solver runs; each track is independent");
    println!("  accepted values:");
    println!("    \"flow_shop\" | \"flow\"");
    println!("    \"hybrid_flow_shop\" | \"hybrid\"");
    println!("    \"job_shop\" | \"job\"");
    println!("    \"fjsp_medium\" | \"medium\"");
    println!("    \"fjsp_high\" | \"high\" | \"fjsp\"");
    println!("  default if omitted or invalid: \"fjsp_high\"");
    println!();
    println!("job_shop_iters (integer):");
    println!("  affects track: job_shop (tabu search iteration budget)");
    println!("  range after clamp: 100..200000");
    println!("  default: 25000");
    println!();
    println!("hybrid_flow_shop_iters (integer):");
    println!("  affects track: hybrid_flow_shop (restart budget)");
    println!("  range after clamp: 100..100000");
    println!("  default: 2000");
    println!();
    println!("fjsp_medium_iters (integer):");
    println!("  affects track: fjsp_medium (restart budget; also scales tabu/cb/alns/ils budgets)");
    println!("  range after clamp: 100..100000");
    println!("  default: 2000");
    println!();
    println!("notes:");
    println!("  flow_shop: no tunable hyperparameter; iteration budget is fixed internally");
    println!("  fjsp_high: uses a fixed internal restart budget of 2000; not tunable via hyperparameters");
    println!("  all other hyperparameter keys are ignored");
}
}
