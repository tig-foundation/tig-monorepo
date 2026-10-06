// ============================================================================
// round_engine.rs -- wave13_parity
//
// THE SHARED SOLVER.  `track_20k.rs`, `track_50k.rs`, `track_100k.rs` and
// `track_200k.rs` are now ~70-line files that supply constants and call
// `solve_with`; every line of the actual algorithm -- construction, the
// refinement round (one fused launch, the device filter + node-order
// compaction, the packed-key host selection, the tabu marking, the three
// H2Ds), the swap phases, the ILS, the polish, the post-balance and the
// best-of-K wrapper -- lives HERE, exactly once.
//
// PROVENANCE.  This file is not hand-written.  `make_engine.py` takes the four
// shipped `solve()` bodies (challengers/submission002_fastbok for the 20k, and
// the same 3-way merge wave12_port used for the three ports, this time keeping
// the `probe_phases` instrument), replaces every per-track constant with a
// `tp.<field>` read and every kernel-name literal with a `kn.<field>` read, and
// ASSERTS that the four results are then character-for-character identical.
// That assertion is the proof that the engine loses nothing: the text below IS
// each track's own text.
//
// The only behavioural additions over submission002 are:
//   * `fused_mode` -- the filter geometry (see the comment at its definition);
//     mode 0 is byte-for-byte the shipped kernel and is the 20k DEFAULT, so the
//     20k build is behaviourally identical to `challengers/budget2`.
//   * `probe_phases = 3` -- three `Instant::now()` reads per round, off at the
//     default 0.
//
// NO TRACK CONSTANT MAY APPEAR IN THIS FILE.  sanity.py enforces it.
// ============================================================================
use cudarc::{
    driver::{safe::LaunchConfig, CudaModule, CudaStream, PushKernelArg},
    runtime::sys::cudaDeviceProp,
};
use serde_json::{Map, Value};
use std::sync::Arc;
use tig_challenges::hypergraph::*;

// ============================================================================
// BEST-OF-K (`runs`) WRAPPER  --  the ONLY change vs the base <track> solver.
// ----------------------------------------------------------------------------
// The construction phase (hyperedge clustering -> node preferences -> greedy
// priority assignment) is deterministic and seed-free, so it runs ONCE.
// Everything after it is re-run `runs` times from that same construction
// result, with a per-run offset
//     run_off = bok_run * 0x9E3779B97F4A7C15   (wrapping)
// folded with `wrapping_add` into EVERY seed that can influence the trajectory
// after construction (the device lottery seed `lcg_x0`, the stagnation
// perturbation seed, the ILS perturbation seed, and the three host-side
// zero-gain acceptance RNG states).  Run 0 has run_off == 0, so it reproduces
// the base solver bit for bit.  The run with the smallest TRUE km1 is saved.
// ============================================================================

/// Upper clamp of the `runs` hyperparameter.  One run costs ~1e10 fuel against
/// a 5e12 budget, so this is a sanity bound, not a fuel bound.
const K_MAX: i64 = 256;

/// Odd golden-ratio constant used to mix the run index into every seed.
const RUN_OFF_MUL: u64 = 0x9E3779B97F4A7C15;

/// Odd constant used to derive the per-run RUIN seed (kept distinct from
/// `RUN_OFF_MUL` so the ruin shuffle and the trajectory seeds do not move in
/// lockstep).
const RUIN_SEED_MUL: u64 = 0xD1B54A32D192ED03;
/// Additive salt for the ruin seed; makes run 1's seed differ from the value
/// `improvement007_ruin` used, purely so the two are not accidentally coupled.
const RUIN_SEED_ADD: u64 = 0xCA5A826395121157;

// ============================================================================
// wave9: PER-RUN DIVERSIFICATION FLAVORS
// ----------------------------------------------------------------------------
// wave8 gave every run k >= 1 the SAME operator: rank the hyperedges of the
// SAME construction partition by lambda, free the nodes of the top ones up to
// the same cap, and reinsert them greedily in a per-run shuffled order.  The
// ruined NODE SET is therefore identical for every k; only the reinsertion
// order and the trajectory seeds move.  Measured consequence (10 nonces, raw):
//
//     K=1 267,201 | K=2 274,655 (+7,454) | K=3 274,934 (+279) | K=4 276,327
//
// i.e. the second basin is worth a fortune and the third is worth nothing --
// runs 2,3 are not complementary to runs 0,1.  wave9 gives run k its own
// FLAVOR: a (source, criterion, strength) triple chosen by run index, so that
// successive runs sample structurally different regions instead of re-shuffling
// one region.  Run 0 and run 1 are pinned to the wave8 behaviour in EVERY mode,
// so `runs = 1` and `runs = 2` are bit-identical to the champion.
// ============================================================================

/// Which partition run k ruins.
const SRC_CONSTRUCTION: u32 = 0; // the greedy construction (wave8 behaviour)
const SRC_INCUMBENT: u32 = 1; // the best partition found so far (exploit / LNS)
const SRC_PREV: u32 = 2; // the previous run's final partition (keeps a lineage)

/// Which hyperedges the ruin walks, i.e. WHERE the freed nodes come from.
///
/// * `TOP_LAMBDA`  -- descending #parts spanned, ties by ascending id.  This is
///   the wave8 / `improvement007_ruin` operator, reproduced bit for bit (same
///   candidate filter, same key, same comparator).  It frees HUB nodes: the
///   giant (1,000-2,000 pin) hyperedges all span ~64 parts, so ~a handful of
///   them already fill the cap.
/// * `RANDOM_EDGE` -- every hyperedge with >= 2 pins, in an LCG-shuffled order.
///   The freed set is scattered over the whole hypergraph instead of being
///   concentrated on the hubs, and it is DIFFERENT FOR EVERY RUN.  This is the
///   direct antidote to "runs 2,3 free the same nodes".
/// * `LOW_LAMBDA`  -- ascending lambda among the CUT edges, so the lambda == 2
///   fringe goes first.  Structurally the complement of `TOP_LAMBDA`: it
///   rearranges the weakly-cut boundary and leaves the hubs alone.
/// * `BIG_EDGE`    -- descending pin count.  Correlated with `TOP_LAMBDA`
///   (lambda <= size), kept only as a grid arm.
/// * `PART_BLOCK`  -- region ruin: empty a contiguous window of PARTS.  July
///   measured region/BFS ruin as near-null on the sibling solver, so this is
///   never in the default ladder; it exists so the claim can be re-checked
///   here for free.
/// * `COUPLED_BLOCK` -- wave34 decomposition ruin: grow a block of parts by
///   inter-part coupling (the part pair sharing the most cut hyperedges, then
///   the part most coupled to the block so far, as the block backbone run does),
///   then free ONLY the pins that sit in block parts of hyperedges cut INSIDE
///   the block (>= 2 block parts touched; edges confined to the block first),
///   and recreate them into block parts only.  The freed set is a closed k-way
///   sub-problem on the most coupled region instead of the hub pins scattered
///   over all 64 parts, so the warm anneal that follows re-decides one coherent
///   boundary rather than re-settling the hubs the cold run already settled.
///   Differs from `PART_BLOCK` (an arbitrary window of consecutive part ids,
///   emptied wholesale) in both the block choice and the freed set.
const CRIT_TOP_LAMBDA: u32 = 0;
const CRIT_RANDOM_EDGE: u32 = 1;
const CRIT_LOW_LAMBDA: u32 = 2;
const CRIT_BIG_EDGE: u32 = 3;
const CRIT_PART_BLOCK: u32 = 4;
const CRIT_COUPLED_BLOCK: u32 = 5;
/// Span cap of the coupling count (a hyperedge spanning more parts than this
/// is not a "pair" signal; same value as the block backbone's `bb_group_span`
/// default).
const CBLK_SPAN: usize = 8;

/// Odd constants for the per-run CRITERION seed.  Kept distinct from both
/// `RUN_OFF_MUL` (trajectory) and `RUIN_SEED_MUL` (recreate shuffle) so that
/// which edges get ruined and in which order they get reinserted do not move in
/// lockstep across runs.
const CRIT_SEED_MUL: u64 = 0x9E6C63D0676A9A99;
const CRIT_SEED_ADD: u64 = 0x2545F4914F6CDD1D;

/// One run's diversification recipe.
///
/// * `src`     : which partition to ruin.
/// * `crit`    : which hyperedges (or parts) the ruin walks.
/// * `cap_pct` : freed-node cap as a percentage of `run_ruin_cap * num_nodes`.
/// * `k_all`   : walk the WHOLE ordered candidate list instead of only the
///   first `run_ruin_frac * m` entries.  `TOP_LAMBDA` needs the truncation
///   (that is what makes it a HUB ruin); every scattered criterion needs the
///   full list, because ~400 random 5-pin edges hold far fewer than
///   `free_cap` distinct nodes and the ruin would otherwise stop early and be
///   much weaker than run 1's.  July already showed the truncation point is
///   inert once the cap binds ("ruin_k is NOT a useful lever").
#[derive(Clone, Copy)]
struct RunFlavor {
    src: u32,
    crit: u32,
    cap_pct: u32,
    k_all: bool,
}

/// The wave8 recipe.  Runs 0 and 1 always use this, in every mode.
const FLAVOR_BASE: RunFlavor = RunFlavor {
    src: SRC_CONSTRUCTION,
    crit: CRIT_TOP_LAMBDA,
    cap_pct: 100,
    k_all: false,
};

/// Tables are indexed by `(k - 2) % 6`, so K <= 8 uses each entry at most once
/// and larger K cycles with a fresh seed per run.
const FLAVOR_CYCLE: usize = 6;

/// mode 1 -- LADDER (default).  Alternates EXPLOIT (ruin the incumbent: high
/// mean, the draw most likely to beat an already-good best-of-K) with EXPLORE
/// (a fresh construction basin reached by a criterion run 1 never uses: low
/// mean, fat tail, the draw that rescues a catastrophic basin).  Alternating is
/// the point -- two draws of the same kind is exactly the +279 that wave8
/// measured.
const F_LADDER: [RunFlavor; FLAVOR_CYCLE] = [
    // k=2  exploit: 25 % of the best-of-2 incumbent, hub ruin, full pipeline.
    RunFlavor { src: SRC_INCUMBENT, crit: CRIT_TOP_LAMBDA, cap_pct: 100, k_all: false },
    // k=3  explore: scattered random ruin of the construction (new node set).
    RunFlavor { src: SRC_CONSTRUCTION, crit: CRIT_RANDOM_EDGE, cap_pct: 100, k_all: true },
    // k=4  exploit, gentler and scattered.
    RunFlavor { src: SRC_INCUMBENT, crit: CRIT_RANDOM_EDGE, cap_pct: 60, k_all: true },
    // k=5  explore the fringe instead of the hubs.
    RunFlavor { src: SRC_CONSTRUCTION, crit: CRIT_LOW_LAMBDA, cap_pct: 100, k_all: true },
    // k=6  explore, same criterion as run 1 but double strength.
    RunFlavor { src: SRC_CONSTRUCTION, crit: CRIT_TOP_LAMBDA, cap_pct: 200, k_all: false },
    // k=7  exploit, small kick = intensification around the incumbent.
    RunFlavor { src: SRC_INCUMBENT, crit: CRIT_TOP_LAMBDA, cap_pct: 50, k_all: false },
];

/// mode 2 -- EXPLORE only.  Every run still starts from the construction, so
/// this isolates the value of CRITERION diversity from the value of exploiting
/// the incumbent.  Fully reproducible run-by-run with `run_probe_index`.
const F_EXPLORE: [RunFlavor; FLAVOR_CYCLE] = [
    RunFlavor { src: SRC_CONSTRUCTION, crit: CRIT_RANDOM_EDGE, cap_pct: 100, k_all: true },
    RunFlavor { src: SRC_CONSTRUCTION, crit: CRIT_LOW_LAMBDA, cap_pct: 100, k_all: true },
    RunFlavor { src: SRC_CONSTRUCTION, crit: CRIT_BIG_EDGE, cap_pct: 100, k_all: true },
    RunFlavor { src: SRC_CONSTRUCTION, crit: CRIT_RANDOM_EDGE, cap_pct: 50, k_all: true },
    RunFlavor { src: SRC_CONSTRUCTION, crit: CRIT_TOP_LAMBDA, cap_pct: 200, k_all: false },
    RunFlavor { src: SRC_CONSTRUCTION, crit: CRIT_RANDOM_EDGE, cap_pct: 200, k_all: true },
];

/// mode 3 -- EXPLOIT only.  Every run k >= 2 is a large-neighbourhood step from
/// the incumbent.  Isolates the other half of mode 1.
const F_EXPLOIT: [RunFlavor; FLAVOR_CYCLE] = [
    RunFlavor { src: SRC_INCUMBENT, crit: CRIT_TOP_LAMBDA, cap_pct: 100, k_all: false },
    RunFlavor { src: SRC_INCUMBENT, crit: CRIT_RANDOM_EDGE, cap_pct: 100, k_all: true },
    RunFlavor { src: SRC_INCUMBENT, crit: CRIT_TOP_LAMBDA, cap_pct: 60, k_all: false },
    RunFlavor { src: SRC_INCUMBENT, crit: CRIT_LOW_LAMBDA, cap_pct: 100, k_all: true },
    RunFlavor { src: SRC_INCUMBENT, crit: CRIT_RANDOM_EDGE, cap_pct: 150, k_all: true },
    RunFlavor { src: SRC_INCUMBENT, crit: CRIT_TOP_LAMBDA, cap_pct: 140, k_all: false },
];

/// mode 4 -- ALL-RANDOM.  The single-change hypothesis: the only thing wrong
/// with wave8 at K >= 3 is that every run frees the SAME nodes.  Nothing else
/// changes (same source, same strength), so a win here attributes the whole
/// gain to the freed set.
const F_RANDOM: [RunFlavor; FLAVOR_CYCLE] = [
    RunFlavor { src: SRC_CONSTRUCTION, crit: CRIT_RANDOM_EDGE, cap_pct: 100, k_all: true },
    RunFlavor { src: SRC_CONSTRUCTION, crit: CRIT_RANDOM_EDGE, cap_pct: 100, k_all: true },
    RunFlavor { src: SRC_CONSTRUCTION, crit: CRIT_RANDOM_EDGE, cap_pct: 100, k_all: true },
    RunFlavor { src: SRC_CONSTRUCTION, crit: CRIT_RANDOM_EDGE, cap_pct: 100, k_all: true },
    RunFlavor { src: SRC_CONSTRUCTION, crit: CRIT_RANDOM_EDGE, cap_pct: 100, k_all: true },
    RunFlavor { src: SRC_CONSTRUCTION, crit: CRIT_RANDOM_EDGE, cap_pct: 100, k_all: true },
];

/// mode 5 -- REGION.  Re-tests the July "region ruin is near-null" finding on
/// this solver, at the cost of one grid point.  Not recommended.
const F_REGION: [RunFlavor; FLAVOR_CYCLE] = [
    RunFlavor { src: SRC_CONSTRUCTION, crit: CRIT_PART_BLOCK, cap_pct: 100, k_all: true },
    RunFlavor { src: SRC_CONSTRUCTION, crit: CRIT_RANDOM_EDGE, cap_pct: 100, k_all: true },
    RunFlavor { src: SRC_INCUMBENT, crit: CRIT_PART_BLOCK, cap_pct: 100, k_all: true },
    RunFlavor { src: SRC_CONSTRUCTION, crit: CRIT_LOW_LAMBDA, cap_pct: 100, k_all: true },
    RunFlavor { src: SRC_INCUMBENT, crit: CRIT_RANDOM_EDGE, cap_pct: 100, k_all: true },
    RunFlavor { src: SRC_CONSTRUCTION, crit: CRIT_PART_BLOCK, cap_pct: 150, k_all: true },
];

/// mode 6 -- LINEAGE.  Half the runs ruin the PREVIOUS run's result rather than
/// the global best, so the search keeps two lineages instead of collapsing every
/// exploit step onto one attractor.
const F_LINEAGE: [RunFlavor; FLAVOR_CYCLE] = [
    RunFlavor { src: SRC_PREV, crit: CRIT_RANDOM_EDGE, cap_pct: 100, k_all: true },
    RunFlavor { src: SRC_INCUMBENT, crit: CRIT_TOP_LAMBDA, cap_pct: 100, k_all: false },
    RunFlavor { src: SRC_PREV, crit: CRIT_LOW_LAMBDA, cap_pct: 100, k_all: true },
    RunFlavor { src: SRC_INCUMBENT, crit: CRIT_RANDOM_EDGE, cap_pct: 100, k_all: true },
    RunFlavor { src: SRC_PREV, crit: CRIT_TOP_LAMBDA, cap_pct: 100, k_all: false },
    RunFlavor { src: SRC_INCUMBENT, crit: CRIT_TOP_LAMBDA, cap_pct: 60, k_all: false },
];

/// Flavor of run `k` under `mode`.  ONLY ever called with `k >= 2` and
/// `mode >= 1`; runs 0/1 and mode 0 use `FLAVOR_BASE` at the call site, which is
/// what makes `runs = 1` and `runs = 2` mode-independent.
fn flavor_of(mode: i64, k: usize) -> RunFlavor {
    // `saturating_sub`: the call site never passes k < 2, and this makes that
    // a no-op rather than an underflow if a future edit ever does.
    let idx = k.saturating_sub(2) % FLAVOR_CYCLE;
    match mode {
        2 => F_EXPLORE[idx],
        3 => F_EXPLOIT[idx],
        4 => F_RANDOM[idx],
        5 => F_REGION[idx],
        6 => F_LINEAGE[idx],
        _ => F_LADDER[idx],
    }
}

/// Exact connectivity (km1) of a partition, computed on the host:
///
///     sum over hyperedges of (number of DISTINCT parts the hyperedge touches - 1)
///
/// This is exactly what the GPU `compute_connectivity_<track>` kernel accumulates
/// and what the challenge scores, so best-of-K selects on the real objective
/// rather than on any internal proxy.  `seen` is caller-owned scratch of length
/// `np`; it is all-false on entry and is left all-false on exit (each hyperedge
/// clears precisely the flags it set), so the routine allocates nothing.
fn eval_km1(
    part: &[i32],
    hedge_offsets: &[i32],
    hedge_nodes: &[i32],
    nh: usize,
    np: usize,
    seen: &mut [bool],
) -> i64 {
    // `nh` is passed in rather than derived from `hedge_offsets.len() - 1` so
    // the count never depends on how the offsets buffer happens to be sized.
    let mut total = 0i64;
    for h in 0..nh {
        let s = hedge_offsets[h] as usize;
        let e = hedge_offsets[h + 1] as usize;
        let mut d = 0i64;
        for k in s..e {
            let p = part[hedge_nodes[k] as usize];
            if p >= 0 && (p as usize) < np && !seen[p as usize] {
                seen[p as usize] = true;
                d += 1;
            }
        }
        if d > 1 {
            total += d - 1;
        }
        for k in s..e {
            let p = part[hedge_nodes[k] as usize];
            if p >= 0 && (p as usize) < np {
                seen[p as usize] = false;
            }
        }
    }
    total
}

/// ------------------------------------------------------------------------
/// wave9: PERCENTAGE OF AN INTEGER BUDGET, exactly and deterministically.
/// ------------------------------------------------------------------------
/// `pct_of(x, 100, lo) == x` for every `x` (integer division of `100*x + 50`
/// by 100 is `x`, because `100*x + 50 < 100*(x+1)`), which is what makes every
/// wave9 budget knob a NO-OP at its default of 100 and keeps run 0 -- and
/// therefore `runs == 1` -- byte-identical to the base solver.
///
/// Rounding is half-UP so a budget can never be silently rounded to zero by a
/// truncation, and the result is clamped into `[lo, x]`: a percentage <= 100
/// must never INCREASE a budget (that would be a quality change dressed up as
/// a cost knob), and `lo` keeps the loop bodies inside the same domain their
/// outer clamps already guaranteed.  If `lo > x` the `min(x)` wins, so the
/// function never returns more than the budget it was handed.
///
/// u64 arithmetic: x <= 50_000 and pct <= 100 here, so `x * pct + 50` is at
/// most 5_000_050 -- no overflow is reachable, on any host.
#[inline]
fn pct_of(x: usize, pct: i64, lo: usize) -> usize {
    let p = if pct < 0 { 0u64 } else { pct as u64 };
    let v = ((x as u64) * p + 50) / 100;
    let v = v as usize;
    if v < lo { lo.min(x) } else if v > x { x } else { v }
}

/// ------------------------------------------------------------------------
/// wave8: RUIN-AND-RECREATE of a partition (host side, fully deterministic).
/// ------------------------------------------------------------------------
/// This is the diversification operator that `improvement007_ruin` /
/// `improvement008_best3ruin` / `improvement009_best5` measured as the single
/// largest quality lever on this track (+2,081 as a straight replacement of the
/// greedy construction, +3,033 as best-of-5 diversified starts, both at 90
/// nonces).  It is reproduced here VERBATIM in behaviour; the only changes are
/// that `free_cap` is now a parameter instead of `max(64, num_nodes/4)`, and
/// that the caller supplies `k_ruin`.
///
/// RUIN   : rank hyperedges by lambda = #distinct parts they touch (descending,
///          ties by ascending hyperedge id), walk the top `k_ruin` of them and
///          unassign (`part = -1`) every node they contain until `free_cap`
///          nodes have been freed.  A node is never freed out of a part of size
///          1, so no part can be emptied.
/// RECREATE: shuffle the freed list with a 64-bit LCG seeded by `seed`, then
///          insert the nodes back ONE AT A TIME, each into the part that
///          minimises the ADDED connectivity, i.e. maximises the number of
///          incident hyperedges that already touch that part (ties broken by
///          smaller part, then by lower part index), skipping parts that are
///          already at `max_part_size`.
///
/// !! The recreate ORDER is the fragile part.  The July campaign measured an
/// !! "improved" recreate (exact min-lambda scoring with a high-degree-first
/// !! order) at -20,620 quality.  Random-shuffle order is the proven recipe.
/// !! Do not reorder, do not sort, do not "improve" the scoring.
///
/// wave9 adds ONE degree of freedom: `crit` selects WHICH hyperedges (or, for
/// `CRIT_PART_BLOCK`, which parts) the RUIN walks.  The RECREATE core is
/// untouched -- same greedy min-added-connectivity scoring, same shuffled
/// order, same tie-breaks -- precisely because it is the fragile part.
/// `crit == CRIT_TOP_LAMBDA` runs the wave8 statements verbatim, so run 1 and
/// therefore `runs = 2` are bit-identical to the champion.
///
/// wave34: `CRIT_COUPLED_BLOCK` is the one criterion that also narrows the
/// RECREATE's target set (to the block's parts); the scoring, the shuffled
/// order and the tie-breaks are still the same statements.
///
/// wave35: `preset_blk` (per-part membership, `len() == num_parts`) hands
/// `CRIT_COUPLED_BLOCK` a block chosen by the CALLER (the vote-grown part
/// group of the block backbone run) instead of growing one by coupling.  An
/// empty slice (every other call) keeps the wave34 growth, so nothing else
/// changes.
///
/// Returns `false` and leaves `partition` / `part_sizes` COMPLETELY UNTOUCHED
/// if anything about the result would be invalid (a part outside
/// `[1, max_part_size]`, an unassigned node, nothing to ruin).  All of the work
/// happens on private clones and is only committed on the last line, so a
/// `false` return is a guaranteed no-op.
fn ruin_recreate(
    partition: &mut Vec<i32>,
    part_sizes: &mut Vec<i32>,
    hedge_offsets: &[i32],
    hedge_nodes: &[i32],
    node_offsets: &[i32],
    node_hedges: &[i32],
    num_hyperedges: usize,
    num_parts: usize,
    max_part_size: i32,
    k_ruin: usize,
    free_cap: usize,
    seed: u64,
    // wave9: which hyperedges (or parts) the ruin walks, and the seed that
    // drives the randomised criteria.  `crit == CRIT_TOP_LAMBDA` reproduces the
    // wave8 operator exactly and ignores `crit_seed`.
    crit: u32,
    crit_seed: u64,
    preset_blk: &[bool],
) -> bool {
    let num_nodes = partition.len();
    if num_nodes == 0 || num_parts == 0 || k_ruin == 0 || free_cap == 0 {
        return false;
    }
    let free_cap = std::cmp::min(free_cap, num_nodes);

    let mut part_new: Vec<i32> = partition.clone();
    let mut sizes_new: Vec<i32> = part_sizes.clone();

    // Stamp array: `stamp_arr[p] == stamp` means part p was already counted for
    // the hyperedge currently being scanned.  Cheaper than clearing 64 bools.
    let mut stamp_arr: Vec<u32> = vec![0u32; num_parts];
    let mut stamp: u32 = 0;

    // ---- rank / order the hyperedges to walk (wave9: by `crit`) ------------
    // `CRIT_TOP_LAMBDA` executes exactly the wave8 statements, in the same
    // order, with the same candidate filter, key and comparator, and advances
    // `stamp` the same number of times (it is shared with the recreate below),
    // so run 1 is bit-identical no matter which flavors the other runs use.
    let mut ranked: Vec<(i32, i32)> = Vec::with_capacity(num_hyperedges);
    let mut freed_flag: Vec<bool> = vec![false; num_nodes];
    let mut freed: Vec<usize> = Vec::new();
    // wave34: parts the RECREATE may place a freed node into.  All true for
    // every criterion except `CRIT_COUPLED_BLOCK`, which closes the recreate
    // onto its block (the global smallest-part fallback below stays, so the
    // all-or-nothing validation is the only thing that can reject the result).
    let mut allowed: Vec<bool> = vec![true; num_parts];
    if crit == CRIT_COUPLED_BLOCK {
        // ---- COUPLED-BLOCK RUIN: a closed k-way sub-problem -----------------
        let np: usize = num_parts;
        // wave35: a caller-chosen block (at least two parts) skips the growth
        // below; the ruin / recreate on it are the unchanged statements.
        let preset_on: bool =
            preset_blk.len() == np && preset_blk.iter().filter(|&&b| b).count() >= 2;
        let mut blk: Vec<bool> = vec![false; np];
        if preset_on {
            for p in 0..np {
                blk[p] = preset_blk[p];
            }
        }
        // (1) coupling matrix: aff[a*np+b] = cut hyperedges spanning both a
        //     and b among those spanning at most CBLK_SPAN parts.
        let mut aff: Vec<i64> = if preset_on { Vec::new() } else { vec![0i64; np * np] };
        let mut span: Vec<usize> = Vec::with_capacity(CBLK_SPAN + 1);
        for h in 0..num_hyperedges {
            if preset_on {
                break;
            }
            let s = hedge_offsets[h] as usize;
            let e = hedge_offsets[h + 1] as usize;
            if e <= s + 1 {
                continue;
            }
            span.clear();
            let mut too_wide: bool = false;
            for k in s..e {
                let p = part_new[hedge_nodes[k] as usize];
                if p < 0 || (p as usize) >= np {
                    continue;
                }
                let pu: usize = p as usize;
                if !span.contains(&pu) {
                    if span.len() >= CBLK_SPAN {
                        too_wide = true;
                        break;
                    }
                    span.push(pu);
                }
            }
            if too_wide || span.len() < 2 {
                continue;
            }
            for a in 0..span.len() {
                for b in (a + 1)..span.len() {
                    let pa: usize = span[a];
                    let pb: usize = span[b];
                    aff[pa * np + pb] += 1;
                    aff[pb * np + pa] += 1;
                }
            }
        }
        // (2) block width: twice the number of parts whose nodes could supply
        //     `free_cap` outright (only the in-block cut pins are freed, a
        //     fraction of each part), at least 2 and at most half the parts
        //     so the sub-problem stays a sub-problem.
        let per_part: usize = std::cmp::max(1, num_nodes / np);
        let want: usize = (free_cap + per_part - 1) / per_part;
        let g_cap: usize = std::cmp::max(1, np / 2);
        let g: usize = want.saturating_mul(2).clamp(2, std::cmp::max(2, g_cap)).min(np);
        // (3) seed = the part with the largest total coupling (ties: lowest
        //     id), then grow by coupling to the block.  wave35: skipped for a
        //     preset block (`blk` already holds it, `aff` is empty).
        let mut row_sum: Vec<i64> = vec![0i64; np];
        if !preset_on {
            for a in 0..np {
                let mut sum: i64 = 0;
                for b in 0..np {
                    sum = sum.saturating_add(aff[a * np + b]);
                }
                row_sum[a] = sum;
            }
        }
        let mut seed_p: usize = 0;
        for a in 1..np {
            if row_sum[a] > row_sum[seed_p] {
                seed_p = a;
            }
        }
        if !preset_on {
            blk[seed_p] = true;
        }
        let mut to_blk: Vec<i64> = vec![0i64; np];
        if !preset_on {
            for q in 0..np {
                to_blk[q] = aff[q * np + seed_p];
            }
        }
        let mut blk_cnt: usize = if preset_on { g } else { 1 };
        while blk_cnt < g {
            let mut best: Option<usize> = None;
            let mut best_score: i64 = -1;
            for q in 0..np {
                if blk[q] {
                    continue;
                }
                let score: i64 = to_blk[q];
                let better: bool = match best {
                    None => true,
                    Some(_) => score > best_score,
                };
                if better {
                    best = Some(q);
                    best_score = score;
                }
            }
            match best {
                Some(bq) => {
                    blk[bq] = true;
                    blk_cnt += 1;
                    for q in 0..np {
                        to_blk[q] = to_blk[q].saturating_add(aff[q * np + bq]);
                    }
                }
                None => break,
            }
        }
        for p in 0..np {
            allowed[p] = blk[p];
        }
        // (4) rank the hyperedges cut INSIDE the block: key = block parts
        //     touched (desc), then total parts touched (asc: an edge confined
        //     to the block can be resolved entirely by the sub-problem), then
        //     id (asc).  `lam <= np <= 256`, so the packed key cannot collide.
        let mut blk_stamp: Vec<u32> = vec![0u32; np];
        for h in 0..num_hyperedges {
            let s = hedge_offsets[h] as usize;
            let e = hedge_offsets[h + 1] as usize;
            if e <= s + 1 {
                continue;
            }
            stamp = stamp.wrapping_add(1);
            if stamp == 0 {
                for v in stamp_arr.iter_mut() {
                    *v = 0;
                }
                for v in blk_stamp.iter_mut() {
                    *v = 0;
                }
                stamp = 1;
            }
            let mut lam: i32 = 0;
            let mut lam_blk: i32 = 0;
            for k in s..e {
                let p = part_new[hedge_nodes[k] as usize];
                if p >= 0 && (p as usize) < np && stamp_arr[p as usize] != stamp {
                    stamp_arr[p as usize] = stamp;
                    lam += 1;
                    if blk[p as usize] && blk_stamp[p as usize] != stamp {
                        blk_stamp[p as usize] = stamp;
                        lam_blk += 1;
                    }
                }
            }
            if lam_blk >= 2 {
                ranked.push((lam_blk * 512 - lam, h as i32));
            }
        }
        if ranked.is_empty() {
            return false;
        }
        ranked.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        // (5) RUIN: free the block pins of the ranked edges, up to the cap.
        'cruin: for &(_, h32) in ranked.iter().take(k_ruin) {
            let h = h32 as usize;
            let s = hedge_offsets[h] as usize;
            let e = hedge_offsets[h + 1] as usize;
            for k in s..e {
                if freed.len() >= free_cap {
                    break 'cruin;
                }
                let n = hedge_nodes[k] as usize;
                if freed_flag[n] {
                    continue;
                }
                let cur = part_new[n];
                if cur < 0 || (cur as usize) >= np || !blk[cur as usize] {
                    continue;
                }
                if sizes_new[cur as usize] <= 1 {
                    continue;
                }
                sizes_new[cur as usize] -= 1;
                part_new[n] = -1;
                freed_flag[n] = true;
                freed.push(n);
            }
        }
    } else if crit == CRIT_PART_BLOCK {
        // ---- REGION RUIN: empty a contiguous window of parts ---------------
        // `width` is the smallest number of parts whose nodes can supply
        // `free_cap` (num_nodes / num_parts nodes per part on average), so the
        // window is exactly as wide as the strength knob asks for.  Node ids
        // ascend within a part and the parts are taken in window order, so the
        // freed list is a pure function of (partition, crit_seed, free_cap).
        let width = std::cmp::max(
            1,
            std::cmp::min(
                num_parts,
                (free_cap.saturating_mul(num_parts) + num_nodes - 1) / num_nodes,
            ),
        );
        let p0 = ((crit_seed >> 8) % (num_parts as u64)) as usize;
        'pb: for t in 0..width {
            let p = ((p0 + t) % num_parts) as i32;
            for n in 0..num_nodes {
                if freed.len() >= free_cap {
                    break 'pb;
                }
                if part_new[n] != p {
                    continue;
                }
                // Never empty a part: one node always stays behind.
                if sizes_new[p as usize] <= 1 {
                    break;
                }
                sizes_new[p as usize] -= 1;
                part_new[n] = -1;
                freed_flag[n] = true;
                freed.push(n);
            }
        }
    } else {
        for h in 0..num_hyperedges {
            let s = hedge_offsets[h] as usize;
            let e = hedge_offsets[h + 1] as usize;
            if e <= s + 1 {
                continue;
            }
            stamp = stamp.wrapping_add(1);
            let mut lambda = 0i32;
            for k in s..e {
                let p = part_new[hedge_nodes[k] as usize];
                if p >= 0 && (p as usize) < num_parts && stamp_arr[p as usize] != stamp {
                    stamp_arr[p as usize] = stamp;
                    lambda += 1;
                }
            }
            if crit == CRIT_BIG_EDGE {
                // key = pin count; every edge with >= 2 pins is a candidate.
                ranked.push(((e - s) as i32, h as i32));
            } else if crit == CRIT_RANDOM_EDGE {
                // Candidate set is deliberately ALL edges with >= 2 pins, not
                // just the cut ones: ~400 random 5-pin edges hold nowhere near
                // `free_cap` distinct nodes, and restricting to the cut set
                // would re-introduce the same boundary bias the criterion
                // exists to avoid.  The key is unused.
                ranked.push((lambda, h as i32));
            } else if lambda >= 2 {
                ranked.push((lambda, h as i32));
            }
        }
        if ranked.is_empty() {
            return false;
        }
        if crit == CRIT_RANDOM_EDGE {
            // Deterministic Fisher-Yates over the id-ascending candidate list.
            // Same LCG as the recreate shuffle, a different (per-run) seed.
            let mut st: u64 = crit_seed | 1;
            let mut i = ranked.len();
            while i > 1 {
                st = st
                    .wrapping_mul(6364136223846793005u64)
                    .wrapping_add(1442695040888963407u64);
                let j = (st % i as u64) as usize;
                i -= 1;
                ranked.swap(i, j);
            }
        } else if crit == CRIT_LOW_LAMBDA {
            // Total order (lambda ASCENDING, then id ascending).
            ranked.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        } else {
            // CRIT_TOP_LAMBDA (and CRIT_BIG_EDGE, on its own key):
            // total order (key descending, then id ascending) =>
            // `sort_unstable_by` has a unique answer and is deterministic.
            ranked.sort_unstable_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        }

        // ---- RUIN ----------------------------------------------------------
        'ruin: for &(_, h32) in ranked.iter().take(k_ruin) {
            let h = h32 as usize;
            let s = hedge_offsets[h] as usize;
            let e = hedge_offsets[h + 1] as usize;
            for k in s..e {
                if freed.len() >= free_cap {
                    break 'ruin;
                }
                let n = hedge_nodes[k] as usize;
                if freed_flag[n] {
                    continue;
                }
                let cur = part_new[n];
                if cur < 0 || (cur as usize) >= num_parts {
                    continue;
                }
                // Never empty a part: the balance validation below requires
                // every part to keep at least one node.
                if sizes_new[cur as usize] <= 1 {
                    continue;
                }
                sizes_new[cur as usize] -= 1;
                part_new[n] = -1;
                freed_flag[n] = true;
                freed.push(n);
            }
        }
    }
    if freed.is_empty() {
        return false;
    }

    // ---- deterministic Fisher-Yates over the freed list ---------------------
    let mut state: u64 = seed | 1;
    let mut i = freed.len();
    while i > 1 {
        state = state
            .wrapping_mul(6364136223846793005u64)
            .wrapping_add(1442695040888963407u64);
        let j = (state % i as u64) as usize;
        i -= 1;
        freed.swap(i, j);
    }

    // ---- RECREATE: greedy min-added-connectivity, in shuffled order ---------
    let mut score: Vec<i32> = vec![0i32; num_parts];
    for &n in freed.iter() {
        for v in score.iter_mut() {
            *v = 0;
        }
        let ns = node_offsets[n] as usize;
        let ne = node_offsets[n + 1] as usize;
        for idx in ns..ne {
            let h = node_hedges[idx] as usize;
            let s = hedge_offsets[h] as usize;
            let e = hedge_offsets[h + 1] as usize;
            stamp = stamp.wrapping_add(1);
            if stamp == 0 {
                for v in stamp_arr.iter_mut() {
                    *v = 0;
                }
                stamp = 1;
            }
            for k in s..e {
                let p = part_new[hedge_nodes[k] as usize];
                if p >= 0 && (p as usize) < num_parts && stamp_arr[p as usize] != stamp {
                    stamp_arr[p as usize] = stamp;
                    score[p as usize] += 1;
                }
            }
        }
        let mut best_p: i32 = -1;
        let mut best_score = -1i32;
        let mut best_size = i32::MAX;
        for p in 0..num_parts {
            if sizes_new[p] >= max_part_size || !allowed[p] {
                continue;
            }
            if score[p] > best_score || (score[p] == best_score && sizes_new[p] < best_size) {
                best_score = score[p];
                best_size = sizes_new[p];
                best_p = p as i32;
            }
        }
        if best_p < 0 {
            // Every part is full: fall back to the smallest part.  The final
            // validation below rejects the whole operation if that overflows.
            let mut min_sz = i32::MAX;
            for p in 0..num_parts {
                if sizes_new[p] < min_sz {
                    min_sz = sizes_new[p];
                    best_p = p as i32;
                }
            }
        }
        if best_p < 0 {
            return false;
        }
        part_new[n] = best_p;
        sizes_new[best_p as usize] += 1;
    }

    // ---- validate, then commit (all-or-nothing) ----------------------------
    for p in 0..num_parts {
        if sizes_new[p] < 1 || sizes_new[p] > max_part_size {
            return false;
        }
    }
    for &pv in part_new.iter() {
        if pv < 0 || (pv as usize) >= num_parts {
            return false;
        }
    }
    partition.copy_from_slice(&part_new);
    part_sizes.copy_from_slice(&sizes_new);
    true
}

/// ------------------------------------------------------------------------
/// wave14: PAIRWISE FM POLISH (host side, deterministic).
/// ------------------------------------------------------------------------
/// Decomposition + local search on the FINAL incumbent.  The 64-way problem
/// is decomposed into 2-way subproblems: for every unordered part pair (a, b)
/// a weight is accumulated over the cut hyperedges that touch both parts
/// (`num_parts / (lambda - 1)` per hyperedge, so a pair that could resolve a
/// hyperedge entirely counts far more than a pair inside a 64-way hyperedge).
/// The `pairs_per_round` heaviest pairs are then refined one after the other
/// with classic Fiduccia-Mattheyses restricted to the nodes of a and b:
///
///   gain(v, from -> to) = #(incident e with pins_from(e) == 1)
///                       - #(incident e with pins_to(e) == 0)
///
/// which is the exact km1 delta of the move (pins in third parts do not
/// change lambda).  A pass repeatedly executes the best-gain unlocked move,
/// INCLUDING zero- and negative-gain moves, locks the node, updates the gains
/// of the affected pins incrementally, and stops after `max_nonimp`
/// consecutive non-improving moves; it then rolls back to the best prefix, so
/// a pass can never make the partition worse.  Balance: a move needs
/// `sizes[to] < max_part_size` and `sizes[from] > 1`.
///
/// Rounds repeat the pair selection on the updated partition and stop early
/// when a whole round gains nothing.  Returns the total km1 gain (>= 0).
/// `partition` / `part_sizes` only ever hold accepted move prefixes, so they
/// are valid at every exit.
pub fn pairwise_fm_polish(
    partition: &mut Vec<i32>,
    part_sizes: &mut Vec<i32>,
    hedge_offsets: &[i32],
    hedge_nodes: &[i32],
    node_offsets: &[i32],
    node_hedges: &[i32],
    num_hyperedges: usize,
    num_parts: usize,
    max_part_size: i32,
    rounds: usize,
    pairs_per_round: usize,
    passes_per_pair: usize,
    max_nonimp: usize,
) -> i64 {
    let num_nodes = partition.len();
    if num_nodes == 0
        || num_parts < 2
        || rounds == 0
        || pairs_per_round == 0
        || passes_per_pair == 0
        || node_offsets.len() < num_nodes + 1
        || hedge_offsets.len() < num_hyperedges + 1
    {
        return 0;
    }
    let mut total_gain: i64 = 0;

    // Scratch shared by every pair: node -> local index, hyperedge -> local
    // index (both -1 when not in the current subproblem).
    let mut loc: Vec<i32> = vec![-1i32; num_nodes];
    let mut edge_loc: Vec<i32> = vec![-1i32; num_hyperedges];
    let mut stamp_arr: Vec<u32> = vec![0u32; num_parts];
    let mut stamp: u32 = 0;
    let mut parts_buf: Vec<usize> = Vec::with_capacity(num_parts);
    let mut pair_w: Vec<i64> = vec![0i64; num_parts * num_parts];

    for _round in 0..rounds {
        // ---- shared-cut weight of every part pair ---------------------------
        for w in pair_w.iter_mut() {
            *w = 0;
        }
        for h in 0..num_hyperedges {
            let s = hedge_offsets[h] as usize;
            let e = hedge_offsets[h + 1] as usize;
            if e <= s + 1 {
                continue;
            }
            stamp = stamp.wrapping_add(1);
            if stamp == 0 {
                for v in stamp_arr.iter_mut() {
                    *v = 0;
                }
                stamp = 1;
            }
            parts_buf.clear();
            for k in s..e {
                let p = partition[hedge_nodes[k] as usize];
                if p >= 0 && (p as usize) < num_parts && stamp_arr[p as usize] != stamp {
                    stamp_arr[p as usize] = stamp;
                    parts_buf.push(p as usize);
                }
            }
            let lambda = parts_buf.len();
            if lambda < 2 {
                continue;
            }
            let w: i64 = (num_parts as i64) / ((lambda as i64) - 1);
            for i in 0..lambda {
                for j in (i + 1)..lambda {
                    let a = parts_buf[i].min(parts_buf[j]);
                    let b = parts_buf[i].max(parts_buf[j]);
                    pair_w[a * num_parts + b] += w;
                }
            }
        }
        let mut pairs: Vec<(i64, usize, usize)> = Vec::new();
        for a in 0..num_parts {
            for b in (a + 1)..num_parts {
                let w = pair_w[a * num_parts + b];
                if w > 0 {
                    pairs.push((w, a, b));
                }
            }
        }
        if pairs.is_empty() {
            break;
        }
        // Total order (weight descending, then (a, b) ascending) => the
        // unstable sort has a unique answer.
        pairs.sort_unstable_by(|x, y| {
            y.0.cmp(&x.0)
                .then_with(|| x.1.cmp(&y.1))
                .then_with(|| x.2.cmp(&y.2))
        });

        // Per-part node lists (ascending node id), kept current between pairs.
        let mut part_nodes: Vec<Vec<usize>> = vec![Vec::new(); num_parts];
        for n in 0..num_nodes {
            let p = partition[n];
            if p >= 0 && (p as usize) < num_parts {
                part_nodes[p as usize].push(n);
            }
        }

        let mut round_gain: i64 = 0;
        for &(_, a, b) in pairs.iter().take(pairs_per_round) {
            let mut nodes: Vec<usize> = Vec::with_capacity(part_nodes[a].len() + part_nodes[b].len());
            nodes.extend_from_slice(&part_nodes[a]);
            nodes.extend_from_slice(&part_nodes[b]);
            if nodes.is_empty() {
                continue;
            }
            let mut pair_gain: i64 = 0;
            for _pass in 0..passes_per_pair {
                let g = fm_pass_pair(
                    partition,
                    part_sizes,
                    hedge_offsets,
                    hedge_nodes,
                    node_offsets,
                    node_hedges,
                    max_part_size,
                    a,
                    b,
                    max_nonimp,
                    &nodes,
                    &mut loc,
                    &mut edge_loc,
                );
                if g <= 0 {
                    break;
                }
                pair_gain += g;
            }
            if pair_gain > 0 {
                // Nodes only ever move between a and b, so the union is
                // unchanged; re-split it by the current assignment.
                part_nodes[a].clear();
                part_nodes[b].clear();
                for &n in nodes.iter() {
                    if partition[n] == a as i32 {
                        part_nodes[a].push(n);
                    } else {
                        part_nodes[b].push(n);
                    }
                }
            }
            round_gain += pair_gain;
        }
        total_gain += round_gain;
        if round_gain == 0 {
            break;
        }
    }
    total_gain
}

/// One FM pass on the 2-way subproblem (a, b).  `nodes` lists every node of
/// a and b.  `loc` / `edge_loc` are caller scratch, all -1 on entry and
/// restored to all -1 on exit.  Returns the gain of the accepted prefix (>= 0)
/// and leaves `partition` / `part_sizes` at exactly that prefix.
/// wave31: `pub` so the 10k track's pair-LNS kick can run it as the exact
/// repair of its closed pair sub-problem (same operator, no copy).
pub fn fm_pass_pair(
    partition: &mut Vec<i32>,
    part_sizes: &mut Vec<i32>,
    hedge_offsets: &[i32],
    hedge_nodes: &[i32],
    node_offsets: &[i32],
    node_hedges: &[i32],
    max_part_size: i32,
    a: usize,
    b: usize,
    max_nonimp: usize,
    nodes: &[usize],
    loc: &mut Vec<i32>,
    edge_loc: &mut Vec<i32>,
) -> i64 {
    use std::collections::BinaryHeap;
    let ai = a as i32;
    let bi = b as i32;
    let ln = nodes.len();
    if ln == 0 {
        return 0;
    }
    for (i, &n) in nodes.iter().enumerate() {
        loc[n] = i as i32;
    }

    // ---- local hyperedges with their pin counts in a and b ------------------
    let mut edges: Vec<usize> = Vec::new();
    let mut pins_a: Vec<i32> = Vec::new();
    let mut pins_b: Vec<i32> = Vec::new();
    for &n in nodes.iter() {
        let ns = node_offsets[n] as usize;
        let ne = node_offsets[n + 1] as usize;
        for idx in ns..ne {
            let h = node_hedges[idx] as usize;
            if edge_loc[h] >= 0 {
                continue;
            }
            let s = hedge_offsets[h] as usize;
            let e = hedge_offsets[h + 1] as usize;
            if e <= s + 1 {
                continue;
            }
            let mut ca = 0i32;
            let mut cb = 0i32;
            for k in s..e {
                let p = partition[hedge_nodes[k] as usize];
                if p == ai {
                    ca += 1;
                } else if p == bi {
                    cb += 1;
                }
            }
            edge_loc[h] = edges.len() as i32;
            edges.push(h);
            pins_a.push(ca);
            pins_b.push(cb);
        }
    }

    // ---- initial gains -------------------------------------------------------
    let mut gain: Vec<i32> = vec![0i32; ln];
    for (i, &n) in nodes.iter().enumerate() {
        let in_a = partition[n] == ai;
        let mut g = 0i32;
        let ns = node_offsets[n] as usize;
        let ne = node_offsets[n + 1] as usize;
        for idx in ns..ne {
            let el = edge_loc[node_hedges[idx] as usize];
            if el < 0 {
                continue;
            }
            let el = el as usize;
            let (pf, pt) = if in_a {
                (pins_a[el], pins_b[el])
            } else {
                (pins_b[el], pins_a[el])
            };
            if pf == 1 {
                g += 1;
            }
            if pt == 0 {
                g -= 1;
            }
        }
        gain[i] = g;
    }

    // Lazy max-heap: (gain, -local index, version).  Stale entries (version
    // or gain mismatch, or locked) are skipped on pop.  Ties: lower local index
    // first, which is a fixed function of the caller's node order.
    let mut ver: Vec<u32> = vec![0u32; ln];
    let mut locked: Vec<bool> = vec![false; ln];
    let mut heap: BinaryHeap<(i32, i32, u32)> = BinaryHeap::with_capacity(ln * 2);
    for i in 0..ln {
        heap.push((gain[i], -(i as i32), 0u32));
    }

    let mut moves: Vec<usize> = Vec::new();
    let mut cum: i64 = 0;
    let mut best_cum: i64 = 0;
    let mut best_len: usize = 0;
    let mut nonimp: usize = 0;
    let mut size_a = part_sizes[a];
    let mut size_b = part_sizes[b];
    // Nodes popped while their move was infeasible (target full or source
    // down to one node).  A move in the OPPOSITE direction fixes both causes,
    // so they are re-queued after one.
    let mut def_from_a: Vec<usize> = Vec::new();
    let mut def_from_b: Vec<usize> = Vec::new();

    while let Some((g, negi, v)) = heap.pop() {
        let i = (-negi) as usize;
        if locked[i] || v != ver[i] || g != gain[i] {
            continue;
        }
        let n = nodes[i];
        let from_a = partition[n] == ai;
        let feasible = if from_a {
            size_b < max_part_size && size_a > 1
        } else {
            size_a < max_part_size && size_b > 1
        };
        if !feasible {
            if from_a {
                def_from_a.push(i);
            } else {
                def_from_b.push(i);
            }
            continue;
        }
        locked[i] = true;
        let (from, to) = if from_a { (ai, bi) } else { (bi, ai) };
        moves.push(i);
        cum += g as i64;
        partition[n] = to;
        if from_a {
            size_a -= 1;
            size_b += 1;
            for &j in def_from_b.iter() {
                if !locked[j] {
                    heap.push((gain[j], -(j as i32), ver[j]));
                }
            }
            def_from_b.clear();
        } else {
            size_b -= 1;
            size_a += 1;
            for &j in def_from_a.iter() {
                if !locked[j] {
                    heap.push((gain[j], -(j as i32), ver[j]));
                }
            }
            def_from_a.clear();
        }
        let ns = node_offsets[n] as usize;
        let ne = node_offsets[n + 1] as usize;
        for idx in ns..ne {
            let h = node_hedges[idx] as usize;
            let el = edge_loc[h];
            if el < 0 {
                continue;
            }
            let el = el as usize;
            let (pf, pt) = if from_a {
                (pins_a[el], pins_b[el])
            } else {
                (pins_b[el], pins_a[el])
            };
            // Gain deltas of the OTHER unlocked pins of e after n leaves
            // `from` (pf -> pf-1) and enters `to` (pt -> pt+1):
            //   u in from: [pf-1==1] - [pf==1] + [pt==0]
            //   u in to  : -[pt==1] - [pf==1]
            let d_from: i32 = (pf == 2) as i32 - (pf == 1) as i32 + (pt == 0) as i32;
            let d_to: i32 = -((pt == 1) as i32) - ((pf == 1) as i32);
            if d_from != 0 || d_to != 0 {
                let s = hedge_offsets[h] as usize;
                let e = hedge_offsets[h + 1] as usize;
                for k in s..e {
                    let u = hedge_nodes[k] as usize;
                    if u == n {
                        continue;
                    }
                    let lu = loc[u];
                    if lu < 0 {
                        continue;
                    }
                    let lu = lu as usize;
                    if locked[lu] {
                        continue;
                    }
                    let pu = partition[u];
                    let d = if pu == from {
                        d_from
                    } else if pu == to {
                        d_to
                    } else {
                        0
                    };
                    if d != 0 {
                        gain[lu] += d;
                        ver[lu] = ver[lu].wrapping_add(1);
                        heap.push((gain[lu], -(lu as i32), ver[lu]));
                    }
                }
            }
            if from_a {
                pins_a[el] -= 1;
                pins_b[el] += 1;
            } else {
                pins_b[el] -= 1;
                pins_a[el] += 1;
            }
        }
        if cum > best_cum {
            best_cum = cum;
            best_len = moves.len();
            nonimp = 0;
        } else {
            nonimp += 1;
            if nonimp >= max_nonimp {
                break;
            }
        }
    }

    // ---- roll back to the best prefix ---------------------------------------
    for &i in moves[best_len..].iter() {
        let n = nodes[i];
        if partition[n] == ai {
            partition[n] = bi;
            size_a -= 1;
            size_b += 1;
        } else {
            partition[n] = ai;
            size_b -= 1;
            size_a += 1;
        }
    }
    part_sizes[a] = size_a;
    part_sizes[b] = size_b;

    for &n in nodes.iter() {
        loc[n] = -1;
    }
    for &h in edges.iter() {
        edge_loc[h] = -1;
    }
    best_cum
}

/// ------------------------------------------------------------------------
/// wave15: K-WAY BOUNDARY FM POLISH (host side, deterministic).
/// ------------------------------------------------------------------------
/// Local search on the whole 64-way partition, run after the pairwise
/// decomposition.  Every boundary node v (in part p) carries its single best
/// move p -> q over ALL parts q != p, with the exact km1 delta
///
///   gain(v, p -> q) = #(e ∋ v: pins_p(e) == 1) - #(e ∋ v: pins_q(e) == 0)
///
/// evaluated from a sparse per-hyperedge (part, pin-count) table.  A pass
/// pops the best-gain unlocked node, executes its move (negative and zero
/// gains included), locks it, and recomputes the best move of every unlocked
/// pin of the incident hyperedges whose pin counts crossed a gain threshold
/// (pins_p in {1, 2} or pins_q in {0, 1} before the move -- the only cases
/// in which another pin's gain can change).  The pass stops after
/// `max_nonimp` consecutive non-improving moves and rolls back to the best
/// prefix, so it can never make the partition worse.  Unlike the pairwise
/// operator, a node is free to leave for a third part mid-sequence, which the
/// 2-way subproblems cannot express.  Balance: `sizes[q] < max_part_size`
/// and `sizes[p] > 1`.  Returns the total km1 gain (>= 0); `partition` /
/// `part_sizes` only ever hold accepted prefixes.
///
/// wave18: `node_weight` is the node count behind each (super)node for the
/// multilevel operator; an EMPTY slice means unit weights, which reproduces
/// the wave15 behaviour bit for bit (`sizes[q] + 1 > max` is `sizes[q] >= max`
/// and `sizes[p] - 1 < 1` is `sizes[p] <= 1`).
pub fn kway_fm_polish(
    partition: &mut Vec<i32>,
    part_sizes: &mut Vec<i32>,
    hedge_offsets: &[i32],
    hedge_nodes: &[i32],
    node_offsets: &[i32],
    node_hedges: &[i32],
    node_weight: &[i32],
    num_hyperedges: usize,
    num_parts: usize,
    max_part_size: i32,
    passes: usize,
    max_nonimp: usize,
) -> i64 {
    use std::collections::BinaryHeap;
    let num_nodes = partition.len();
    if num_nodes == 0
        || num_parts < 2
        || passes == 0
        || max_nonimp == 0
        || node_offsets.len() < num_nodes + 1
        || hedge_offsets.len() < num_hyperedges + 1
        || part_sizes.len() < num_parts
        || (!node_weight.is_empty() && node_weight.len() < num_nodes)
    {
        return 0;
    }
    for &p in partition.iter() {
        if p < 0 || (p as usize) >= num_parts {
            return 0;
        }
    }
    let total_pins = hedge_offsets[num_hyperedges] as usize;
    if hedge_nodes.len() < total_pins {
        return 0;
    }

    // ---- sparse (part, count) table per hyperedge, laid out in the pin
    // slots of that hyperedge (lambda <= size, so it always fits) ------------
    let mut ep_part: Vec<i32> = vec![-1i32; total_pins];
    let mut ep_cnt: Vec<i32> = vec![0i32; total_pins];
    let mut ep_len: Vec<i32> = vec![0i32; num_hyperedges];
    for h in 0..num_hyperedges {
        let s = hedge_offsets[h] as usize;
        let e = hedge_offsets[h + 1] as usize;
        let mut len = 0usize;
        for k in s..e {
            let p = partition[hedge_nodes[k] as usize];
            let mut found = false;
            for j in 0..len {
                if ep_part[s + j] == p {
                    ep_cnt[s + j] += 1;
                    found = true;
                    break;
                }
            }
            if !found {
                ep_part[s + len] = p;
                ep_cnt[s + len] = 1;
                len += 1;
            }
        }
        ep_len[h] = len as i32;
    }

    // Best move of node v (in part p): gain = R - deg + C(q) with R = #edges
    // where v is the sole pin of p, deg = #non-trivial incident edges and
    // C(q) = #incident edges that already contain q.  Only feasible targets
    // (sizes[q] < max_part_size) are considered; ties go to the lowest q.
    let mut score: Vec<i32> = vec![0i32; num_parts];
    let mut score_stamp: Vec<u32> = vec![0u32; num_parts];
    let mut stamp: u32 = 0;
    let mut cand: Vec<usize> = Vec::with_capacity(num_parts);

    let mut cur_gain: Vec<i32> = vec![i32::MIN; num_nodes];
    let mut cur_tgt: Vec<i32> = vec![-1i32; num_nodes];
    let mut ver: Vec<u32> = vec![0u32; num_nodes];
    let mut locked: Vec<bool> = vec![false; num_nodes];
    let mut touched_stamp: Vec<u32> = vec![0u32; num_nodes];
    let mut touch: u32 = 0;
    let mut affected: Vec<usize> = Vec::new();

    let mut total_gain: i64 = 0;
    for _pass in 0..passes {
        for l in locked.iter_mut() {
            *l = false;
        }
        let mut heap: BinaryHeap<(i32, i32, u32)> = BinaryHeap::with_capacity(num_nodes);
        for v in 0..num_nodes {
            let p = partition[v];
            let wv: i32 = if node_weight.is_empty() { 1 } else { node_weight[v] };
            let (g, q) = kfm_best_move(
                v,
                p,
                wv,
                node_offsets,
                node_hedges,
                hedge_offsets,
                &ep_part,
                &ep_cnt,
                &ep_len,
                part_sizes,
                max_part_size,
                &mut score,
                &mut score_stamp,
                &mut stamp,
                &mut cand,
            );
            cur_gain[v] = g;
            cur_tgt[v] = q;
            if q >= 0 {
                heap.push((g, -(v as i32), ver[v]));
            }
        }

        let mut moves: Vec<(usize, i32, i32)> = Vec::new();
        let mut cum: i64 = 0;
        let mut best_cum: i64 = 0;
        let mut best_len: usize = 0;
        let mut nonimp: usize = 0;

        while let Some((g, negv, vv)) = heap.pop() {
            let v = (-negv) as usize;
            if locked[v] || vv != ver[v] || g != cur_gain[v] || cur_tgt[v] < 0 {
                continue;
            }
            let p = partition[v];
            let q = cur_tgt[v];
            let wv: i32 = if node_weight.is_empty() { 1 } else { node_weight[v] };
            if part_sizes[p as usize] - wv < 1 {
                // Source down to one node: drop for now; a neighbour's move
                // re-evaluates it (the version bump makes that re-queue).
                cur_tgt[v] = -1;
                ver[v] = ver[v].wrapping_add(1);
                continue;
            }
            if part_sizes[q as usize] + wv > max_part_size {
                // Target filled up since this entry was queued: re-evaluate
                // against the current sizes and re-queue (or drop).
                let (g2, q2) = kfm_best_move(
                    v,
                    p,
                    wv,
                    node_offsets,
                    node_hedges,
                    hedge_offsets,
                    &ep_part,
                    &ep_cnt,
                    &ep_len,
                    part_sizes,
                    max_part_size,
                    &mut score,
                    &mut score_stamp,
                    &mut stamp,
                    &mut cand,
                );
                cur_gain[v] = g2;
                cur_tgt[v] = q2;
                ver[v] = ver[v].wrapping_add(1);
                if q2 >= 0 {
                    heap.push((g2, -(v as i32), ver[v]));
                }
                continue;
            }

            locked[v] = true;
            moves.push((v, p, q));
            cum += g as i64;
            partition[v] = q;
            part_sizes[p as usize] -= wv;
            part_sizes[q as usize] += wv;

            // Update the pin table of every incident hyperedge first, noting
            // the ones whose counts crossed a threshold; then recompute the
            // unlocked pins of those edges once each.
            affected.clear();
            let ns = node_offsets[v] as usize;
            let ne = node_offsets[v + 1] as usize;
            for idx in ns..ne {
                let h = node_hedges[idx] as usize;
                let s = hedge_offsets[h] as usize;
                let e = hedge_offsets[h + 1] as usize;
                if e <= s + 1 {
                    continue;
                }
                let cp = kfm_count(&ep_part, &ep_cnt, s, ep_len[h] as usize, p);
                let cq = kfm_count(&ep_part, &ep_cnt, s, ep_len[h] as usize, q);
                if cp <= 2 || cq <= 1 {
                    affected.push(h);
                }
                kfm_apply(&mut ep_part, &mut ep_cnt, &mut ep_len, s, h, p, q);
            }
            touch = touch.wrapping_add(1);
            if touch == 0 {
                for t in touched_stamp.iter_mut() {
                    *t = 0;
                }
                touch = 1;
            }
            touched_stamp[v] = touch;
            for ai in 0..affected.len() {
                let h = affected[ai];
                let s = hedge_offsets[h] as usize;
                let e = hedge_offsets[h + 1] as usize;
                for k in s..e {
                    let u = hedge_nodes[k] as usize;
                    if locked[u] || touched_stamp[u] == touch {
                        continue;
                    }
                    touched_stamp[u] = touch;
                    let pu = partition[u];
                    let wu: i32 = if node_weight.is_empty() { 1 } else { node_weight[u] };
                    let (g2, q2) = kfm_best_move(
                        u,
                        pu,
                        wu,
                        node_offsets,
                        node_hedges,
                        hedge_offsets,
                        &ep_part,
                        &ep_cnt,
                        &ep_len,
                        part_sizes,
                        max_part_size,
                        &mut score,
                        &mut score_stamp,
                        &mut stamp,
                        &mut cand,
                    );
                    if g2 != cur_gain[u] || q2 != cur_tgt[u] {
                        cur_gain[u] = g2;
                        cur_tgt[u] = q2;
                        ver[u] = ver[u].wrapping_add(1);
                        if q2 >= 0 {
                            heap.push((g2, -(u as i32), ver[u]));
                        }
                    }
                }
            }

            if cum > best_cum {
                best_cum = cum;
                best_len = moves.len();
                nonimp = 0;
            } else {
                nonimp += 1;
                if nonimp >= max_nonimp {
                    break;
                }
            }
        }

        // ---- roll back to the best prefix (reverse order) ------------------
        let mut mi = moves.len();
        while mi > best_len {
            mi -= 1;
            let (v, p, q) = moves[mi];
            let wv: i32 = if node_weight.is_empty() { 1 } else { node_weight[v] };
            partition[v] = p;
            part_sizes[q as usize] -= wv;
            part_sizes[p as usize] += wv;
            let ns = node_offsets[v] as usize;
            let ne = node_offsets[v + 1] as usize;
            for idx in ns..ne {
                let h = node_hedges[idx] as usize;
                let s = hedge_offsets[h] as usize;
                let e = hedge_offsets[h + 1] as usize;
                if e <= s + 1 {
                    continue;
                }
                kfm_apply(&mut ep_part, &mut ep_cnt, &mut ep_len, s, h, q, p);
            }
        }
        total_gain += best_cum;
        if best_cum <= 0 {
            break;
        }
    }
    total_gain
}

/// Pin count of part `p` in the hyperedge whose table starts at `s` with
/// `len` entries (0 when absent).
#[inline]
fn kfm_count(ep_part: &[i32], ep_cnt: &[i32], s: usize, len: usize, p: i32) -> i32 {
    for j in 0..len {
        if ep_part[s + j] == p {
            return ep_cnt[s + j];
        }
    }
    0
}

/// Move one pin of hyperedge `h` (table at `s`) from part `from` to part
/// `to`: decrement `from` (dropping the entry at zero by swapping in the last
/// one) and increment `to` (appending it when absent).
#[inline]
fn kfm_apply(
    ep_part: &mut Vec<i32>,
    ep_cnt: &mut Vec<i32>,
    ep_len: &mut Vec<i32>,
    s: usize,
    h: usize,
    from: i32,
    to: i32,
) {
    let mut len = ep_len[h] as usize;
    let mut j = 0usize;
    while j < len {
        if ep_part[s + j] == from {
            ep_cnt[s + j] -= 1;
            if ep_cnt[s + j] <= 0 {
                let last = len - 1;
                ep_part[s + j] = ep_part[s + last];
                ep_cnt[s + j] = ep_cnt[s + last];
                ep_part[s + last] = -1;
                ep_cnt[s + last] = 0;
                len = last;
            }
            break;
        }
        j += 1;
    }
    let mut found = false;
    for j in 0..len {
        if ep_part[s + j] == to {
            ep_cnt[s + j] += 1;
            found = true;
            break;
        }
    }
    if !found {
        ep_part[s + len] = to;
        ep_cnt[s + len] = 1;
        len += 1;
    }
    ep_len[h] = len as i32;
}

/// Best feasible move of node `v` (currently in `p`, weight `wv` nodes):
/// returns `(gain, target)` with `target == -1` when no part of any incident
/// hyperedge is a feasible destination (interior node, or all full).
#[inline]
fn kfm_best_move(
    v: usize,
    p: i32,
    wv: i32,
    node_offsets: &[i32],
    node_hedges: &[i32],
    hedge_offsets: &[i32],
    ep_part: &[i32],
    ep_cnt: &[i32],
    ep_len: &[i32],
    part_sizes: &[i32],
    max_part_size: i32,
    score: &mut Vec<i32>,
    score_stamp: &mut Vec<u32>,
    stamp: &mut u32,
    cand: &mut Vec<usize>,
) -> (i32, i32) {
    *stamp = stamp.wrapping_add(1);
    if *stamp == 0 {
        for t in score_stamp.iter_mut() {
            *t = 0;
        }
        *stamp = 1;
    }
    let st = *stamp;
    cand.clear();
    let mut r = 0i32;
    let mut deg = 0i32;
    let ns = node_offsets[v] as usize;
    let ne = node_offsets[v + 1] as usize;
    for idx in ns..ne {
        let h = node_hedges[idx] as usize;
        let s = hedge_offsets[h] as usize;
        let e = hedge_offsets[h + 1] as usize;
        if e <= s + 1 {
            continue;
        }
        deg += 1;
        let len = ep_len[h] as usize;
        for j in 0..len {
            let q = ep_part[s + j];
            if q == p {
                if ep_cnt[s + j] == 1 {
                    r += 1;
                }
            } else if q >= 0 {
                let qu = q as usize;
                if score_stamp[qu] != st {
                    score_stamp[qu] = st;
                    score[qu] = 0;
                    cand.push(qu);
                }
                score[qu] += 1;
            }
        }
    }
    let mut best_g = i32::MIN;
    let mut best_q: i32 = -1;
    for &qu in cand.iter() {
        if part_sizes[qu] + wv > max_part_size {
            continue;
        }
        let g = r - deg + score[qu];
        if g > best_g || (g == best_g && (qu as i32) < best_q) {
            best_g = g;
            best_q = qu as i32;
        }
    }
    if best_q < 0 {
        return (i32::MIN, -1);
    }
    (best_g, best_q)
}

/// ------------------------------------------------------------------------
/// wave16: HYPEREDGE COLLAPSE POLISH (host side, deterministic).
/// ------------------------------------------------------------------------
/// A second decomposition, by HYPEREDGE instead of by part pair, paired with
/// a compound-move local search.  Every cut hyperedge e with at most
/// `max_size` pins and at most `max_lambda` parts is its own subproblem: for
/// each part t that e already touches, ALL pins of e outside t are moved into
/// t in one block move and the exact km1 delta of the block is accumulated
/// move by move from the sparse per-hyperedge (part, pin-count) table (the
/// same table as `kway_fm_polish`):
///
///   delta(v, p -> t) = #(g ∋ v: pins_t(g) == 0) - #(g ∋ v: pins_p(g) == 1)
///
/// applied sequentially, so interactions between the pins of e (and between
/// e's pins through their other hyperedges) are counted exactly.  The best
/// target is kept when the block strictly lowers km1; otherwise the block is
/// undone.  Single-node FM (pairwise or k-way) cannot find these moves: the
/// first pin to leave usually has NEGATIVE gain (it raises lambda of its own
/// hyperedges in t) and the global heap spends its non-improvement budget on
/// unrelated nodes before the remaining pins of e follow.  Balance: a block
/// needs `sizes[t] + #movers <= max_part_size` and every source part to keep
/// at least one node.  Passes repeat until one gains nothing.  Returns the
/// total km1 gain (>= 0); `partition` / `part_sizes` only ever hold accepted
/// blocks.
pub fn hyperedge_collapse_polish(
    partition: &mut Vec<i32>,
    part_sizes: &mut Vec<i32>,
    hedge_offsets: &[i32],
    hedge_nodes: &[i32],
    node_offsets: &[i32],
    node_hedges: &[i32],
    num_hyperedges: usize,
    num_parts: usize,
    max_part_size: i32,
    passes: usize,
    max_size: usize,
    max_lambda: usize,
) -> i64 {
    let num_nodes = partition.len();
    if num_nodes == 0
        || num_parts < 2
        || passes == 0
        || max_size < 2
        || max_lambda < 2
        || node_offsets.len() < num_nodes + 1
        || hedge_offsets.len() < num_hyperedges + 1
        || part_sizes.len() < num_parts
    {
        return 0;
    }
    for &p in partition.iter() {
        if p < 0 || (p as usize) >= num_parts {
            return 0;
        }
    }
    let total_pins = hedge_offsets[num_hyperedges] as usize;
    if hedge_nodes.len() < total_pins {
        return 0;
    }

    // ---- sparse (part, count) table per hyperedge (see kway_fm_polish) -----
    let mut ep_part: Vec<i32> = vec![-1i32; total_pins];
    let mut ep_cnt: Vec<i32> = vec![0i32; total_pins];
    let mut ep_len: Vec<i32> = vec![0i32; num_hyperedges];
    for h in 0..num_hyperedges {
        let s = hedge_offsets[h] as usize;
        let e = hedge_offsets[h + 1] as usize;
        let mut len = 0usize;
        for k in s..e {
            let p = partition[hedge_nodes[k] as usize];
            let mut found = false;
            for j in 0..len {
                if ep_part[s + j] == p {
                    ep_cnt[s + j] += 1;
                    found = true;
                    break;
                }
            }
            if !found {
                ep_part[s + len] = p;
                ep_cnt[s + len] = 1;
                len += 1;
            }
        }
        ep_len[h] = len as i32;
    }

    let mut tgts: Vec<i32> = Vec::with_capacity(max_lambda);
    let mut applied: Vec<(usize, i32)> = Vec::with_capacity(max_size);
    let mut total_gain: i64 = 0;

    for _pass in 0..passes {
        let mut pass_gain: i64 = 0;
        for h in 0..num_hyperedges {
            let s = hedge_offsets[h] as usize;
            let e = hedge_offsets[h + 1] as usize;
            let size = e - s;
            if size < 2 || size > max_size {
                continue;
            }
            let lam = ep_len[h] as usize;
            if lam < 2 || lam > max_lambda {
                continue;
            }
            // Snapshot the touched parts: the table mutates during the trials.
            tgts.clear();
            for j in 0..lam {
                tgts.push(ep_part[s + j]);
            }
            // Ascending target order => the strict `<` below picks the lowest
            // part id among equal deltas, independent of table layout.
            tgts.sort_unstable();

            let mut best_delta: i64 = 0;
            let mut best_t: i32 = -1;
            for &t in tgts.iter() {
                if t < 0 || (t as usize) >= num_parts {
                    continue;
                }
                let delta = collapse_apply(
                    partition,
                    part_sizes,
                    hedge_offsets,
                    hedge_nodes,
                    node_offsets,
                    node_hedges,
                    &mut ep_part,
                    &mut ep_cnt,
                    &mut ep_len,
                    s,
                    e,
                    t,
                    max_part_size,
                    &mut applied,
                );
                if let Some(d) = delta {
                    if d < best_delta {
                        best_delta = d;
                        best_t = t;
                    }
                }
                collapse_undo(
                    partition,
                    part_sizes,
                    hedge_offsets,
                    node_offsets,
                    node_hedges,
                    &mut ep_part,
                    &mut ep_cnt,
                    &mut ep_len,
                    t,
                    &mut applied,
                );
            }
            if best_t >= 0 && best_delta < 0 {
                let delta = collapse_apply(
                    partition,
                    part_sizes,
                    hedge_offsets,
                    hedge_nodes,
                    node_offsets,
                    node_hedges,
                    &mut ep_part,
                    &mut ep_cnt,
                    &mut ep_len,
                    s,
                    e,
                    best_t,
                    max_part_size,
                    &mut applied,
                );
                match delta {
                    Some(d) if d == best_delta => {
                        applied.clear();
                        pass_gain += -d;
                    }
                    _ => {
                        // Cannot happen (same state as the trial); keep the
                        // partition consistent regardless.
                        collapse_undo(
                            partition,
                            part_sizes,
                            hedge_offsets,
                            node_offsets,
                            node_hedges,
                            &mut ep_part,
                            &mut ep_cnt,
                            &mut ep_len,
                            best_t,
                            &mut applied,
                        );
                    }
                }
            }
        }
        total_gain += pass_gain;
        if pass_gain <= 0 {
            break;
        }
    }
    total_gain
}

/// Move every pin of the hyperedge whose pins are `hedge_nodes[s..e]` into
/// part `t`, sequentially, recording each (node, old part) in `applied` and
/// returning the exact km1 delta (`Some(delta)`), or `None` when the block is
/// infeasible (target would overflow, or a source would be emptied); in the
/// `None` case the moves made so far are still in `applied` and the caller
/// must undo them.
#[inline]
fn collapse_apply(
    partition: &mut Vec<i32>,
    part_sizes: &mut Vec<i32>,
    hedge_offsets: &[i32],
    hedge_nodes: &[i32],
    node_offsets: &[i32],
    node_hedges: &[i32],
    ep_part: &mut Vec<i32>,
    ep_cnt: &mut Vec<i32>,
    ep_len: &mut Vec<i32>,
    s: usize,
    e: usize,
    t: i32,
    max_part_size: i32,
    applied: &mut Vec<(usize, i32)>,
) -> Option<i64> {
    applied.clear();
    let tu = t as usize;
    let mut movers = 0i32;
    for k in s..e {
        if partition[hedge_nodes[k] as usize] != t {
            movers += 1;
        }
    }
    if movers == 0 || part_sizes[tu] + movers > max_part_size {
        return None;
    }
    let mut delta: i64 = 0;
    for k in s..e {
        let v = hedge_nodes[k] as usize;
        let p = partition[v];
        if p == t {
            continue;
        }
        let pu = p as usize;
        if part_sizes[pu] <= 1 {
            return None;
        }
        let ns = node_offsets[v] as usize;
        let ne = node_offsets[v + 1] as usize;
        for idx in ns..ne {
            let g = node_hedges[idx] as usize;
            let gs = hedge_offsets[g] as usize;
            let ge = hedge_offsets[g + 1] as usize;
            if ge <= gs + 1 {
                continue;
            }
            let len = ep_len[g] as usize;
            let cp = kfm_count(ep_part, ep_cnt, gs, len, p);
            let ct = kfm_count(ep_part, ep_cnt, gs, len, t);
            if ct == 0 {
                delta += 1;
            }
            if cp == 1 {
                delta -= 1;
            }
            kfm_apply(ep_part, ep_cnt, ep_len, gs, g, p, t);
        }
        partition[v] = t;
        part_sizes[pu] -= 1;
        part_sizes[tu] += 1;
        applied.push((v, p));
    }
    Some(delta)
}

/// Undo the moves recorded by `collapse_apply` (reverse order) and clear
/// `applied`.
#[inline]
fn collapse_undo(
    partition: &mut Vec<i32>,
    part_sizes: &mut Vec<i32>,
    hedge_offsets: &[i32],
    node_offsets: &[i32],
    node_hedges: &[i32],
    ep_part: &mut Vec<i32>,
    ep_cnt: &mut Vec<i32>,
    ep_len: &mut Vec<i32>,
    t: i32,
    applied: &mut Vec<(usize, i32)>,
) {
    let tu = t as usize;
    let mut mi = applied.len();
    while mi > 0 {
        mi -= 1;
        let (v, p) = applied[mi];
        let ns = node_offsets[v] as usize;
        let ne = node_offsets[v + 1] as usize;
        for idx in ns..ne {
            let g = node_hedges[idx] as usize;
            let gs = hedge_offsets[g] as usize;
            let ge = hedge_offsets[g + 1] as usize;
            if ge <= gs + 1 {
                continue;
            }
            kfm_apply(ep_part, ep_cnt, ep_len, gs, g, t, p);
        }
        partition[v] = p;
        part_sizes[tu] -= 1;
        part_sizes[p as usize] += 1;
    }
    applied.clear();
}

/// ------------------------------------------------------------------------
/// wave17: PART-GROUP FM POLISH (host side, deterministic).
/// ------------------------------------------------------------------------
/// A third decomposition, between the 2-way pairs of `pairwise_fm_polish`
/// and the global heap of `kway_fm_polish`.  The pairwise operator couples
/// exactly two parts and reuses the same hub parts in most of its 96 pairs;
/// the global k-way operator lets any node go anywhere but spends its single
/// `max_nonimp` budget on the whole hypergraph, so a short negative-gain
/// sequence that pays off inside one region is drowned by zero-gain moves
/// elsewhere.  This operator sits in between: the shared-cut weight matrix
/// (same `num_parts / (lambda - 1)` weighting as the pairwise operator) is
/// greedily cut into DISJOINT groups of `group_size` parts -- take the
/// heaviest unused pair, then repeatedly add the unused part with the largest
/// total weight to the group so far -- so every part is coupled at most once
/// per round and the groups together cover the partition.  Each group is then
/// refined by a k-way FM restricted to its own nodes and its own parts (exact
/// km1 gains from the sparse per-hyperedge (part, pin-count) table, lazy
/// heap, rollback to the best prefix, own non-improvement budget), so nodes
/// may hop to a THIRD part of the group mid-sequence, which no 2-way
/// subproblem can express, while each region gets its full budget.  Rounds
/// rebuild the groups on the updated partition and stop when one gains
/// nothing.  Balance: `sizes[q] < max_part_size` and `sizes[p] > 1`.
/// Returns the total km1 gain (>= 0); `partition` / `part_sizes` only ever
/// hold accepted prefixes.
pub fn group_fm_polish(
    partition: &mut Vec<i32>,
    part_sizes: &mut Vec<i32>,
    hedge_offsets: &[i32],
    hedge_nodes: &[i32],
    node_offsets: &[i32],
    node_hedges: &[i32],
    num_hyperedges: usize,
    num_parts: usize,
    max_part_size: i32,
    rounds: usize,
    group_size: usize,
    passes_per_group: usize,
    max_nonimp: usize,
) -> i64 {
    use std::collections::BinaryHeap;
    let num_nodes = partition.len();
    if num_nodes == 0
        || num_parts < 3
        || rounds == 0
        || group_size < 3
        || passes_per_group == 0
        || max_nonimp == 0
        || node_offsets.len() < num_nodes + 1
        || hedge_offsets.len() < num_hyperedges + 1
        || part_sizes.len() < num_parts
    {
        return 0;
    }
    for &p in partition.iter() {
        if p < 0 || (p as usize) >= num_parts {
            return 0;
        }
    }
    let total_pins = hedge_offsets[num_hyperedges] as usize;
    if hedge_nodes.len() < total_pins {
        return 0;
    }
    let group_size = group_size.min(num_parts);

    // ---- sparse (part, count) table per hyperedge (see kway_fm_polish) -----
    let mut ep_part: Vec<i32> = vec![-1i32; total_pins];
    let mut ep_cnt: Vec<i32> = vec![0i32; total_pins];
    let mut ep_len: Vec<i32> = vec![0i32; num_hyperedges];
    for h in 0..num_hyperedges {
        let s = hedge_offsets[h] as usize;
        let e = hedge_offsets[h + 1] as usize;
        let mut len = 0usize;
        for k in s..e {
            let p = partition[hedge_nodes[k] as usize];
            let mut found = false;
            for j in 0..len {
                if ep_part[s + j] == p {
                    ep_cnt[s + j] += 1;
                    found = true;
                    break;
                }
            }
            if !found {
                ep_part[s + len] = p;
                ep_cnt[s + len] = 1;
                len += 1;
            }
        }
        ep_len[h] = len as i32;
    }

    let mut pair_w: Vec<i64> = vec![0i64; num_parts * num_parts];
    let mut used: Vec<bool> = vec![false; num_parts];
    let mut in_group: Vec<bool> = vec![false; num_parts];

    let mut score: Vec<i32> = vec![0i32; num_parts];
    let mut score_stamp: Vec<u32> = vec![0u32; num_parts];
    let mut stamp: u32 = 0;
    let mut cand: Vec<usize> = Vec::with_capacity(num_parts);

    let mut cur_gain: Vec<i32> = vec![i32::MIN; num_nodes];
    let mut cur_tgt: Vec<i32> = vec![-1i32; num_nodes];
    let mut ver: Vec<u32> = vec![0u32; num_nodes];
    let mut locked: Vec<bool> = vec![false; num_nodes];
    let mut touched_stamp: Vec<u32> = vec![0u32; num_nodes];
    let mut touch: u32 = 0;
    let mut affected: Vec<usize> = Vec::new();

    let mut total_gain: i64 = 0;
    for _round in 0..rounds {
        // ---- shared-cut weight of every part pair, from the pin table ------
        for w in pair_w.iter_mut() {
            *w = 0;
        }
        for h in 0..num_hyperedges {
            let s = hedge_offsets[h] as usize;
            let lam = ep_len[h] as usize;
            if lam < 2 {
                continue;
            }
            let w: i64 = (num_parts as i64) / ((lam as i64) - 1);
            for i in 0..lam {
                for j in (i + 1)..lam {
                    let pi = ep_part[s + i];
                    let pj = ep_part[s + j];
                    if pi < 0 || pj < 0 {
                        continue;
                    }
                    let a = pi.min(pj) as usize;
                    let b = pi.max(pj) as usize;
                    pair_w[a * num_parts + b] += w;
                }
            }
        }
        let mut pairs: Vec<(i64, usize, usize)> = Vec::new();
        for a in 0..num_parts {
            for b in (a + 1)..num_parts {
                let w = pair_w[a * num_parts + b];
                if w > 0 {
                    pairs.push((w, a, b));
                }
            }
        }
        if pairs.is_empty() {
            break;
        }
        // Total order (weight descending, then (a, b) ascending).
        pairs.sort_unstable_by(|x, y| {
            y.0.cmp(&x.0)
                .then_with(|| x.1.cmp(&y.1))
                .then_with(|| x.2.cmp(&y.2))
        });

        // ---- greedy DISJOINT grouping: heaviest unused pair, then grow -----
        for u in used.iter_mut() {
            *u = false;
        }
        let mut groups: Vec<Vec<usize>> = Vec::new();
        for &(_, a, b) in pairs.iter() {
            if used[a] || used[b] {
                continue;
            }
            let mut g: Vec<usize> = vec![a, b];
            used[a] = true;
            used[b] = true;
            while g.len() < group_size {
                let mut best_c: i32 = -1;
                let mut best_w: i64 = 0;
                for c in 0..num_parts {
                    if used[c] {
                        continue;
                    }
                    let mut wsum: i64 = 0;
                    for &m in g.iter() {
                        let (x, y) = if m < c { (m, c) } else { (c, m) };
                        wsum += pair_w[x * num_parts + y];
                    }
                    // Strict `>` with best_w starting at 0: the lowest part id
                    // wins ties and a part sharing no cut with the group is
                    // never added.
                    if wsum > best_w {
                        best_w = wsum;
                        best_c = c as i32;
                    }
                }
                if best_c < 0 {
                    break;
                }
                used[best_c as usize] = true;
                g.push(best_c as usize);
            }
            // A 2-part group is exactly a pairwise subproblem, which the
            // pairwise operator has already refined; skip it.
            if g.len() >= 3 {
                groups.push(g);
            }
        }
        if groups.is_empty() {
            break;
        }

        let mut round_gain: i64 = 0;
        for g in groups.iter() {
            for &p in g.iter() {
                in_group[p] = true;
            }
            // Nodes of the group (ascending id).  Moves stay inside the group,
            // so the list is valid for every pass on this group.
            let mut nodes: Vec<usize> = Vec::new();
            for n in 0..num_nodes {
                if in_group[partition[n] as usize] {
                    nodes.push(n);
                }
            }
            for _pass in 0..passes_per_group {
                for &n in nodes.iter() {
                    locked[n] = false;
                }
                let mut heap: BinaryHeap<(i32, i32, u32)> =
                    BinaryHeap::with_capacity(nodes.len());
                for &v in nodes.iter() {
                    let p = partition[v];
                    let (gg, q) = gfm_best_move(
                        v,
                        p,
                        node_offsets,
                        node_hedges,
                        hedge_offsets,
                        &ep_part,
                        &ep_cnt,
                        &ep_len,
                        part_sizes,
                        max_part_size,
                        &in_group,
                        &mut score,
                        &mut score_stamp,
                        &mut stamp,
                        &mut cand,
                    );
                    cur_gain[v] = gg;
                    cur_tgt[v] = q;
                    if q >= 0 {
                        heap.push((gg, -(v as i32), ver[v]));
                    }
                }

                let mut moves: Vec<(usize, i32, i32)> = Vec::new();
                let mut cum: i64 = 0;
                let mut best_cum: i64 = 0;
                let mut best_len: usize = 0;
                let mut nonimp: usize = 0;

                while let Some((gg, negv, vv)) = heap.pop() {
                    let v = (-negv) as usize;
                    if locked[v] || vv != ver[v] || gg != cur_gain[v] || cur_tgt[v] < 0 {
                        continue;
                    }
                    let p = partition[v];
                    let q = cur_tgt[v];
                    if part_sizes[p as usize] <= 1 {
                        cur_tgt[v] = -1;
                        ver[v] = ver[v].wrapping_add(1);
                        continue;
                    }
                    if part_sizes[q as usize] >= max_part_size {
                        let (g2, q2) = gfm_best_move(
                            v,
                            p,
                            node_offsets,
                            node_hedges,
                            hedge_offsets,
                            &ep_part,
                            &ep_cnt,
                            &ep_len,
                            part_sizes,
                            max_part_size,
                            &in_group,
                            &mut score,
                            &mut score_stamp,
                            &mut stamp,
                            &mut cand,
                        );
                        cur_gain[v] = g2;
                        cur_tgt[v] = q2;
                        ver[v] = ver[v].wrapping_add(1);
                        if q2 >= 0 {
                            heap.push((g2, -(v as i32), ver[v]));
                        }
                        continue;
                    }

                    locked[v] = true;
                    moves.push((v, p, q));
                    cum += gg as i64;
                    partition[v] = q;
                    part_sizes[p as usize] -= 1;
                    part_sizes[q as usize] += 1;

                    affected.clear();
                    let ns = node_offsets[v] as usize;
                    let ne = node_offsets[v + 1] as usize;
                    for idx in ns..ne {
                        let h = node_hedges[idx] as usize;
                        let s = hedge_offsets[h] as usize;
                        let e = hedge_offsets[h + 1] as usize;
                        if e <= s + 1 {
                            continue;
                        }
                        let cp = kfm_count(&ep_part, &ep_cnt, s, ep_len[h] as usize, p);
                        let cq = kfm_count(&ep_part, &ep_cnt, s, ep_len[h] as usize, q);
                        if cp <= 2 || cq <= 1 {
                            affected.push(h);
                        }
                        kfm_apply(&mut ep_part, &mut ep_cnt, &mut ep_len, s, h, p, q);
                    }
                    touch = touch.wrapping_add(1);
                    if touch == 0 {
                        for t in touched_stamp.iter_mut() {
                            *t = 0;
                        }
                        touch = 1;
                    }
                    touched_stamp[v] = touch;
                    for ai in 0..affected.len() {
                        let h = affected[ai];
                        let s = hedge_offsets[h] as usize;
                        let e = hedge_offsets[h + 1] as usize;
                        for k in s..e {
                            let u = hedge_nodes[k] as usize;
                            let pu = partition[u];
                            // Pins outside the group are not candidates.
                            if !in_group[pu as usize] {
                                continue;
                            }
                            if locked[u] || touched_stamp[u] == touch {
                                continue;
                            }
                            touched_stamp[u] = touch;
                            let (g2, q2) = gfm_best_move(
                                u,
                                pu,
                                node_offsets,
                                node_hedges,
                                hedge_offsets,
                                &ep_part,
                                &ep_cnt,
                                &ep_len,
                                part_sizes,
                                max_part_size,
                                &in_group,
                                &mut score,
                                &mut score_stamp,
                                &mut stamp,
                                &mut cand,
                            );
                            if g2 != cur_gain[u] || q2 != cur_tgt[u] {
                                cur_gain[u] = g2;
                                cur_tgt[u] = q2;
                                ver[u] = ver[u].wrapping_add(1);
                                if q2 >= 0 {
                                    heap.push((g2, -(u as i32), ver[u]));
                                }
                            }
                        }
                    }

                    if cum > best_cum {
                        best_cum = cum;
                        best_len = moves.len();
                        nonimp = 0;
                    } else {
                        nonimp += 1;
                        if nonimp >= max_nonimp {
                            break;
                        }
                    }
                }

                // ---- roll back to the best prefix (reverse order) ----------
                let mut mi = moves.len();
                while mi > best_len {
                    mi -= 1;
                    let (v, p, q) = moves[mi];
                    partition[v] = p;
                    part_sizes[q as usize] -= 1;
                    part_sizes[p as usize] += 1;
                    let ns = node_offsets[v] as usize;
                    let ne = node_offsets[v + 1] as usize;
                    for idx in ns..ne {
                        let h = node_hedges[idx] as usize;
                        let s = hedge_offsets[h] as usize;
                        let e = hedge_offsets[h + 1] as usize;
                        if e <= s + 1 {
                            continue;
                        }
                        kfm_apply(&mut ep_part, &mut ep_cnt, &mut ep_len, s, h, q, p);
                    }
                }
                round_gain += best_cum;
                if best_cum <= 0 {
                    break;
                }
            }
            for &p in g.iter() {
                in_group[p] = false;
            }
        }
        total_gain += round_gain;
        if round_gain <= 0 {
            break;
        }
    }
    total_gain
}

/// `kfm_best_move` restricted to targets with `allowed[q]`: best feasible
/// move of node `v` (in `p`) inside its part group, `(gain, target)` with
/// `target == -1` when no allowed part of any incident hyperedge is feasible.
#[inline]
fn gfm_best_move(
    v: usize,
    p: i32,
    node_offsets: &[i32],
    node_hedges: &[i32],
    hedge_offsets: &[i32],
    ep_part: &[i32],
    ep_cnt: &[i32],
    ep_len: &[i32],
    part_sizes: &[i32],
    max_part_size: i32,
    allowed: &[bool],
    score: &mut Vec<i32>,
    score_stamp: &mut Vec<u32>,
    stamp: &mut u32,
    cand: &mut Vec<usize>,
) -> (i32, i32) {
    *stamp = stamp.wrapping_add(1);
    if *stamp == 0 {
        for t in score_stamp.iter_mut() {
            *t = 0;
        }
        *stamp = 1;
    }
    let st = *stamp;
    cand.clear();
    let mut r = 0i32;
    let mut deg = 0i32;
    let ns = node_offsets[v] as usize;
    let ne = node_offsets[v + 1] as usize;
    for idx in ns..ne {
        let h = node_hedges[idx] as usize;
        let s = hedge_offsets[h] as usize;
        let e = hedge_offsets[h + 1] as usize;
        if e <= s + 1 {
            continue;
        }
        deg += 1;
        let len = ep_len[h] as usize;
        for j in 0..len {
            let q = ep_part[s + j];
            if q == p {
                if ep_cnt[s + j] == 1 {
                    r += 1;
                }
            } else if q >= 0 {
                let qu = q as usize;
                if !allowed[qu] {
                    continue;
                }
                if score_stamp[qu] != st {
                    score_stamp[qu] = st;
                    score[qu] = 0;
                    cand.push(qu);
                }
                score[qu] += 1;
            }
        }
    }
    let mut best_g = i32::MIN;
    let mut best_q: i32 = -1;
    for &qu in cand.iter() {
        if part_sizes[qu] >= max_part_size {
            continue;
        }
        let g = r - deg + score[qu];
        if g > best_g || (g == best_g && (qu as i32) < best_q) {
            best_g = g;
            best_q = qu as i32;
        }
    }
    if best_q < 0 {
        return (i32::MIN, -1);
    }
    (best_g, best_q)
}

/// ------------------------------------------------------------------------
/// wave19: REGION FM POLISH (host side, deterministic).
/// ------------------------------------------------------------------------
/// A decomposition by HYPERGRAPH LOCALITY instead of by part: every other
/// operator couples parts (pairs, groups) or the whole partition (k-way),
/// so its nodes are scattered over the hypergraph and the non-improvement
/// budget of one FM pass is spent on unrelated zero-gain moves far away from
/// the sequence that would pay off.  Here a region is the closed 1-hop
/// neighbourhood of a seed hyperedge: its pins plus every pin of every
/// hyperedge (of at most `max_edge_size` pins) incident to them, grown in
/// ascending id order and capped at `max_nodes` nodes.  Seeds are the cut
/// hyperedges in descending lambda (lowest id on ties); round r takes the
/// seeds `r * regions .. (r + 1) * regions` of that order, so successive
/// rounds sweep DIFFERENT areas, and nodes are claimed by at most one region
/// per round (a seed whose pins are all claimed is skipped), so a round
/// touches each node at most once.  Each region is refined by k-way FM
/// restricted to its own nodes with ALL parts as targets (exact km1 gains
/// from the sparse per-hyperedge (part, pin-count) table, lazy heap,
/// rollback to the best prefix, own non-improvement budget), so a node may
/// leave for any part, which the group operator forbids, while the budget
/// stays local, which the global k-way operator cannot do.  Balance:
/// `sizes[q] < max_part_size` and `sizes[p] > 1`.  Returns the total km1
/// gain (>= 0); `partition` / `part_sizes` only ever hold accepted prefixes.
pub fn region_fm_polish(
    partition: &mut Vec<i32>,
    part_sizes: &mut Vec<i32>,
    hedge_offsets: &[i32],
    hedge_nodes: &[i32],
    node_offsets: &[i32],
    node_hedges: &[i32],
    num_hyperedges: usize,
    num_parts: usize,
    max_part_size: i32,
    rounds: usize,
    regions_per_round: usize,
    max_nodes: usize,
    max_edge_size: usize,
    passes_per_region: usize,
    max_nonimp: usize,
) -> i64 {
    use std::collections::BinaryHeap;
    let num_nodes = partition.len();
    if num_nodes == 0
        || num_parts < 2
        || rounds == 0
        || regions_per_round == 0
        || max_nodes < 2
        || max_edge_size < 2
        || passes_per_region == 0
        || max_nonimp == 0
        || node_offsets.len() < num_nodes + 1
        || hedge_offsets.len() < num_hyperedges + 1
        || part_sizes.len() < num_parts
    {
        return 0;
    }
    for &p in partition.iter() {
        if p < 0 || (p as usize) >= num_parts {
            return 0;
        }
    }
    let total_pins = hedge_offsets[num_hyperedges] as usize;
    if hedge_nodes.len() < total_pins || node_hedges.len() < node_offsets[num_nodes] as usize {
        return 0;
    }
    for &v in hedge_nodes[..total_pins].iter() {
        if v < 0 || (v as usize) >= num_nodes {
            return 0;
        }
    }
    for &h in node_hedges[..node_offsets[num_nodes] as usize].iter() {
        if h < 0 || (h as usize) >= num_hyperedges {
            return 0;
        }
    }

    // ---- sparse (part, count) table per hyperedge (see kway_fm_polish) -----
    let mut ep_part: Vec<i32> = vec![-1i32; total_pins];
    let mut ep_cnt: Vec<i32> = vec![0i32; total_pins];
    let mut ep_len: Vec<i32> = vec![0i32; num_hyperedges];
    for h in 0..num_hyperedges {
        let s = hedge_offsets[h] as usize;
        let e = hedge_offsets[h + 1] as usize;
        let mut len = 0usize;
        for k in s..e {
            let p = partition[hedge_nodes[k] as usize];
            let mut found = false;
            for j in 0..len {
                if ep_part[s + j] == p {
                    ep_cnt[s + j] += 1;
                    found = true;
                    break;
                }
            }
            if !found {
                ep_part[s + len] = p;
                ep_cnt[s + len] = 1;
                len += 1;
            }
        }
        ep_len[h] = len as i32;
    }

    let mut score: Vec<i32> = vec![0i32; num_parts];
    let mut score_stamp: Vec<u32> = vec![0u32; num_parts];
    let mut stamp: u32 = 0;
    let mut cand: Vec<usize> = Vec::with_capacity(num_parts);

    let mut cur_gain: Vec<i32> = vec![i32::MIN; num_nodes];
    let mut cur_tgt: Vec<i32> = vec![-1i32; num_nodes];
    let mut ver: Vec<u32> = vec![0u32; num_nodes];
    let mut locked: Vec<bool> = vec![false; num_nodes];
    let mut touched_stamp: Vec<u32> = vec![0u32; num_nodes];
    let mut touch: u32 = 0;
    let mut affected: Vec<usize> = Vec::new();

    // Region membership: `in_region[v]` is the region's own stamp while the
    // region is being refined; `claimed[v]` is the round stamp once a region
    // of this round owns v.
    let mut in_region: Vec<u32> = vec![0u32; num_nodes];
    let mut region_stamp: u32 = 0;
    let mut claimed: Vec<u32> = vec![0u32; num_nodes];

    let mut total_gain: i64 = 0;
    for round in 0..rounds {
        // ---- seed order: cut hyperedges by lambda descending, id ascending --
        let mut seeds: Vec<(i32, usize)> = Vec::new();
        for h in 0..num_hyperedges {
            let s = hedge_offsets[h] as usize;
            let e = hedge_offsets[h + 1] as usize;
            if e <= s + 1 || e - s > max_edge_size {
                continue;
            }
            if ep_len[h] >= 2 {
                seeds.push((ep_len[h], h));
            }
        }
        if seeds.is_empty() {
            break;
        }
        seeds.sort_unstable_by(|x, y| y.0.cmp(&x.0).then_with(|| x.1.cmp(&y.1)));
        let seed_lo = round * regions_per_round;
        if seed_lo >= seeds.len() {
            break;
        }
        let seed_hi = (seed_lo + regions_per_round).min(seeds.len());
        let round_tag: u32 = (round as u32).wrapping_add(1);

        let mut round_gain: i64 = 0;
        for si in seed_lo..seed_hi {
            let seed = seeds[si].1;
            let ss = hedge_offsets[seed] as usize;
            let se = hedge_offsets[seed + 1] as usize;
            // ---- grow the closed 1-hop neighbourhood of the seed -----------
            region_stamp = region_stamp.wrapping_add(1);
            if region_stamp == 0 {
                for t in in_region.iter_mut() {
                    *t = 0;
                }
                region_stamp = 1;
            }
            let mut nodes: Vec<usize> = Vec::new();
            for k in ss..se {
                let v = hedge_nodes[k] as usize;
                if claimed[v] == round_tag || in_region[v] == region_stamp {
                    continue;
                }
                in_region[v] = region_stamp;
                nodes.push(v);
                if nodes.len() >= max_nodes {
                    break;
                }
            }
            if nodes.is_empty() {
                continue;
            }
            let core_len = nodes.len();
            let mut ci = 0usize;
            while ci < core_len && nodes.len() < max_nodes {
                let v = nodes[ci];
                ci += 1;
                let ns = node_offsets[v] as usize;
                let ne = node_offsets[v + 1] as usize;
                for idx in ns..ne {
                    let h = node_hedges[idx] as usize;
                    let s = hedge_offsets[h] as usize;
                    let e = hedge_offsets[h + 1] as usize;
                    if e <= s + 1 || e - s > max_edge_size {
                        continue;
                    }
                    for k in s..e {
                        let u = hedge_nodes[k] as usize;
                        if claimed[u] == round_tag || in_region[u] == region_stamp {
                            continue;
                        }
                        in_region[u] = region_stamp;
                        nodes.push(u);
                        if nodes.len() >= max_nodes {
                            break;
                        }
                    }
                    if nodes.len() >= max_nodes {
                        break;
                    }
                }
            }
            for &v in nodes.iter() {
                claimed[v] = round_tag;
            }
            if nodes.len() < 2 {
                continue;
            }

            // ---- k-way FM restricted to the region's nodes -----------------
            for _pass in 0..passes_per_region {
                for &n in nodes.iter() {
                    locked[n] = false;
                }
                let mut heap: BinaryHeap<(i32, i32, u32)> =
                    BinaryHeap::with_capacity(nodes.len());
                for &v in nodes.iter() {
                    let p = partition[v];
                    let (gg, q) = kfm_best_move(
                        v,
                        p,
                        1,
                        node_offsets,
                        node_hedges,
                        hedge_offsets,
                        &ep_part,
                        &ep_cnt,
                        &ep_len,
                        part_sizes,
                        max_part_size,
                        &mut score,
                        &mut score_stamp,
                        &mut stamp,
                        &mut cand,
                    );
                    cur_gain[v] = gg;
                    cur_tgt[v] = q;
                    if q >= 0 {
                        heap.push((gg, -(v as i32), ver[v]));
                    }
                }

                let mut moves: Vec<(usize, i32, i32)> = Vec::new();
                let mut cum: i64 = 0;
                let mut best_cum: i64 = 0;
                let mut best_len: usize = 0;
                let mut nonimp: usize = 0;

                while let Some((gg, negv, vv)) = heap.pop() {
                    let v = (-negv) as usize;
                    if locked[v] || vv != ver[v] || gg != cur_gain[v] || cur_tgt[v] < 0 {
                        continue;
                    }
                    let p = partition[v];
                    let q = cur_tgt[v];
                    if part_sizes[p as usize] <= 1 {
                        cur_tgt[v] = -1;
                        ver[v] = ver[v].wrapping_add(1);
                        continue;
                    }
                    if part_sizes[q as usize] >= max_part_size {
                        let (g2, q2) = kfm_best_move(
                            v,
                            p,
                            1,
                            node_offsets,
                            node_hedges,
                            hedge_offsets,
                            &ep_part,
                            &ep_cnt,
                            &ep_len,
                            part_sizes,
                            max_part_size,
                            &mut score,
                            &mut score_stamp,
                            &mut stamp,
                            &mut cand,
                        );
                        cur_gain[v] = g2;
                        cur_tgt[v] = q2;
                        ver[v] = ver[v].wrapping_add(1);
                        if q2 >= 0 {
                            heap.push((g2, -(v as i32), ver[v]));
                        }
                        continue;
                    }

                    locked[v] = true;
                    moves.push((v, p, q));
                    cum += gg as i64;
                    partition[v] = q;
                    part_sizes[p as usize] -= 1;
                    part_sizes[q as usize] += 1;

                    affected.clear();
                    let ns = node_offsets[v] as usize;
                    let ne = node_offsets[v + 1] as usize;
                    for idx in ns..ne {
                        let h = node_hedges[idx] as usize;
                        let s = hedge_offsets[h] as usize;
                        let e = hedge_offsets[h + 1] as usize;
                        if e <= s + 1 {
                            continue;
                        }
                        let cp = kfm_count(&ep_part, &ep_cnt, s, ep_len[h] as usize, p);
                        let cq = kfm_count(&ep_part, &ep_cnt, s, ep_len[h] as usize, q);
                        if cp <= 2 || cq <= 1 {
                            affected.push(h);
                        }
                        kfm_apply(&mut ep_part, &mut ep_cnt, &mut ep_len, s, h, p, q);
                    }
                    touch = touch.wrapping_add(1);
                    if touch == 0 {
                        for t in touched_stamp.iter_mut() {
                            *t = 0;
                        }
                        touch = 1;
                    }
                    touched_stamp[v] = touch;
                    for ai in 0..affected.len() {
                        let h = affected[ai];
                        let s = hedge_offsets[h] as usize;
                        let e = hedge_offsets[h + 1] as usize;
                        for k in s..e {
                            let u = hedge_nodes[k] as usize;
                            // Pins outside the region are not candidates.
                            if in_region[u] != region_stamp {
                                continue;
                            }
                            if locked[u] || touched_stamp[u] == touch {
                                continue;
                            }
                            touched_stamp[u] = touch;
                            let pu = partition[u];
                            let (g2, q2) = kfm_best_move(
                                u,
                                pu,
                                1,
                                node_offsets,
                                node_hedges,
                                hedge_offsets,
                                &ep_part,
                                &ep_cnt,
                                &ep_len,
                                part_sizes,
                                max_part_size,
                                &mut score,
                                &mut score_stamp,
                                &mut stamp,
                                &mut cand,
                            );
                            if g2 != cur_gain[u] || q2 != cur_tgt[u] {
                                cur_gain[u] = g2;
                                cur_tgt[u] = q2;
                                ver[u] = ver[u].wrapping_add(1);
                                if q2 >= 0 {
                                    heap.push((g2, -(u as i32), ver[u]));
                                }
                            }
                        }
                    }

                    if cum > best_cum {
                        best_cum = cum;
                        best_len = moves.len();
                        nonimp = 0;
                    } else {
                        nonimp += 1;
                        if nonimp >= max_nonimp {
                            break;
                        }
                    }
                }

                // ---- roll back to the best prefix (reverse order) ----------
                let mut mi = moves.len();
                while mi > best_len {
                    mi -= 1;
                    let (v, p, q) = moves[mi];
                    partition[v] = p;
                    part_sizes[q as usize] -= 1;
                    part_sizes[p as usize] += 1;
                    let ns = node_offsets[v] as usize;
                    let ne = node_offsets[v + 1] as usize;
                    for idx in ns..ne {
                        let h = node_hedges[idx] as usize;
                        let s = hedge_offsets[h] as usize;
                        let e = hedge_offsets[h + 1] as usize;
                        if e <= s + 1 {
                            continue;
                        }
                        kfm_apply(&mut ep_part, &mut ep_cnt, &mut ep_len, s, h, q, p);
                    }
                }
                round_gain += best_cum;
                if best_cum <= 0 {
                    break;
                }
            }
        }
        total_gain += round_gain;
    }
    total_gain
}

/// ------------------------------------------------------------------------
/// wave18: MULTILEVEL V-CYCLE POLISH (host side, deterministic).
/// ------------------------------------------------------------------------
/// A fourth decomposition, by NODE CONTRACTION instead of by part pair, part
/// group or hyperedge, paired with the k-way FM of `kway_fm_polish`.  The
/// incumbent is coarsened `levels` times by heavy-edge matching restricted to
/// nodes of the SAME part (rating of a candidate = sum over the shared
/// hyperedges of at most `max_edge_size` pins of 1 / (|e| - 1); the best
/// unmatched same-part neighbour wins, lowest id on ties; nodes are visited
/// in ascending id order), so every supernode lies inside one part and the
/// coarse partition has exactly the km1 of the fine one: a hyperedge whose
/// pins all fall into one supernode has lambda 1 forever and is dropped, every
/// other one keeps its distinct supernodes as coarse pins, and the coarse
/// (part, pin-count) table of `kway_fm_polish` then yields exact km1 deltas.
/// The coarse levels are refined from the coarsest down with that k-way FM,
/// where one move relocates a whole block of up to 2^levels tightly coupled
/// nodes at once and the balance test uses the block weight; each level's
/// result is projected onto the next finer level before that one is refined.
/// Single-node FM cannot find these moves: the first node of a tight block to
/// leave has negative gain and the non-improvement budget is spent on
/// unrelated zero-gain moves before the rest of the block follows.  Returns
/// the total km1 gain (>= 0); `partition` / `part_sizes` only ever hold
/// accepted prefixes and are rebuilt from the final projection.
pub fn multilevel_fm_polish(
    partition: &mut Vec<i32>,
    part_sizes: &mut Vec<i32>,
    hedge_offsets: &[i32],
    hedge_nodes: &[i32],
    node_offsets: &[i32],
    node_hedges: &[i32],
    num_hyperedges: usize,
    num_parts: usize,
    max_part_size: i32,
    levels: usize,
    passes: usize,
    max_nonimp: usize,
    max_edge_size: usize,
) -> i64 {
    let num_nodes = partition.len();
    if num_nodes == 0
        || num_parts < 2
        || levels == 0
        || passes == 0
        || max_nonimp == 0
        || max_edge_size < 2
        || node_offsets.len() < num_nodes + 1
        || hedge_offsets.len() < num_hyperedges + 1
        || part_sizes.len() < num_parts
    {
        return 0;
    }
    for &p in partition.iter() {
        if p < 0 || (p as usize) >= num_parts {
            return 0;
        }
    }
    let total_pins = hedge_offsets[num_hyperedges] as usize;
    if hedge_nodes.len() < total_pins || node_hedges.len() < node_offsets[num_nodes] as usize {
        return 0;
    }
    for &v in hedge_nodes[..total_pins].iter() {
        if v < 0 || (v as usize) >= num_nodes {
            return 0;
        }
    }

    // Level 0 is the input; level l >= 1 is stored at index l - 1 of these.
    // `lv_cmap[l]` maps the nodes of level l to the supernodes of level l + 1.
    let mut lv_hoff: Vec<Vec<i32>> = Vec::new();
    let mut lv_hnodes: Vec<Vec<i32>> = Vec::new();
    let mut lv_noff: Vec<Vec<i32>> = Vec::new();
    let mut lv_nhedges: Vec<Vec<i32>> = Vec::new();
    let mut lv_weight: Vec<Vec<i32>> = Vec::new();
    let mut lv_part: Vec<Vec<i32>> = Vec::new();
    let mut lv_cmap: Vec<Vec<i32>> = Vec::new();

    const RATE_SCALE: i64 = 1 << 20;
    let mut rate: Vec<i64> = vec![0i64; num_nodes];
    let mut rate_stamp: Vec<u32> = vec![0u32; num_nodes];
    let mut stamp: u32 = 0;
    let mut cand: Vec<usize> = Vec::with_capacity(64);
    let mut seen_stamp: Vec<u32> = vec![0u32; num_nodes];
    let mut seen: u32 = 0;

    for _l in 0..levels {
        let cur = lv_hoff.len();
        let built: Option<(Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>)> = {
            let c_hoff: &[i32] = if cur == 0 { hedge_offsets } else { &lv_hoff[cur - 1] };
            let c_hnodes: &[i32] = if cur == 0 { hedge_nodes } else { &lv_hnodes[cur - 1] };
            let c_noff: &[i32] = if cur == 0 { node_offsets } else { &lv_noff[cur - 1] };
            let c_nhedges: &[i32] = if cur == 0 { node_hedges } else { &lv_nhedges[cur - 1] };
            let c_weight: &[i32] = if cur == 0 { &[] } else { &lv_weight[cur - 1] };
            let c_part: &[i32] = if cur == 0 { &partition[..] } else { &lv_part[cur - 1] };
            let n = c_part.len();
            let m = if cur == 0 { num_hyperedges } else { c_hoff.len() - 1 };

            // ---- heavy-edge matching inside each part --------------------
            let mut cmap: Vec<i32> = vec![-1i32; n];
            let mut coarse_n: usize = 0;
            let mut any_match = false;
            for v in 0..n {
                if cmap[v] >= 0 {
                    continue;
                }
                stamp = stamp.wrapping_add(1);
                if stamp == 0 {
                    for t in rate_stamp.iter_mut() {
                        *t = 0;
                    }
                    stamp = 1;
                }
                cand.clear();
                let pv = c_part[v];
                let ns = c_noff[v] as usize;
                let ne = c_noff[v + 1] as usize;
                for idx in ns..ne {
                    let h = c_nhedges[idx] as usize;
                    let s = c_hoff[h] as usize;
                    let e = c_hoff[h + 1] as usize;
                    let sz = e - s;
                    if sz < 2 || sz > max_edge_size {
                        continue;
                    }
                    let w: i64 = RATE_SCALE / ((sz as i64) - 1);
                    for k in s..e {
                        let u = c_hnodes[k] as usize;
                        if u == v || cmap[u] >= 0 || c_part[u] != pv {
                            continue;
                        }
                        if rate_stamp[u] != stamp {
                            rate_stamp[u] = stamp;
                            rate[u] = 0;
                            cand.push(u);
                        }
                        rate[u] += w;
                    }
                }
                let mut best_u: i32 = -1;
                let mut best_r: i64 = 0;
                for &u in cand.iter() {
                    let r = rate[u];
                    if r > best_r || (r == best_r && best_u >= 0 && (u as i32) < best_u) {
                        best_r = r;
                        best_u = u as i32;
                    }
                }
                cmap[v] = coarse_n as i32;
                if best_u >= 0 {
                    cmap[best_u as usize] = coarse_n as i32;
                    any_match = true;
                }
                coarse_n += 1;
            }
            if !any_match || coarse_n == 0 {
                None
            } else {
                // ---- coarse hyperedges: distinct supernodes of each pin set
                let mut n_hoff: Vec<i32> = Vec::with_capacity(m + 1);
                let mut n_hnodes: Vec<i32> = Vec::new();
                n_hoff.push(0i32);
                for h in 0..m {
                    let s = c_hoff[h] as usize;
                    let e = c_hoff[h + 1] as usize;
                    if e <= s + 1 {
                        continue;
                    }
                    seen = seen.wrapping_add(1);
                    if seen == 0 {
                        for t in seen_stamp.iter_mut() {
                            *t = 0;
                        }
                        seen = 1;
                    }
                    let start = n_hnodes.len();
                    for k in s..e {
                        let c = cmap[c_hnodes[k] as usize] as usize;
                        if seen_stamp[c] != seen {
                            seen_stamp[c] = seen;
                            n_hnodes.push(c as i32);
                        }
                    }
                    if n_hnodes.len() - start < 2 {
                        n_hnodes.truncate(start);
                        continue;
                    }
                    n_hoff.push(n_hnodes.len() as i32);
                }
                let n_m = n_hoff.len() - 1;
                // ---- coarse node -> hyperedge CSR -----------------------------
                let mut n_noff: Vec<i32> = vec![0i32; coarse_n + 1];
                for &c in n_hnodes.iter() {
                    n_noff[c as usize + 1] += 1;
                }
                for i in 0..coarse_n {
                    n_noff[i + 1] += n_noff[i];
                }
                let mut fill: Vec<i32> = n_noff.clone();
                let mut n_nhedges: Vec<i32> = vec![0i32; n_hnodes.len()];
                for h in 0..n_m {
                    let s = n_hoff[h] as usize;
                    let e = n_hoff[h + 1] as usize;
                    for k in s..e {
                        let c = n_hnodes[k] as usize;
                        n_nhedges[fill[c] as usize] = h as i32;
                        fill[c] += 1;
                    }
                }
                let mut n_weight: Vec<i32> = vec![0i32; coarse_n];
                let mut n_part: Vec<i32> = vec![-1i32; coarse_n];
                for v in 0..n {
                    let c = cmap[v] as usize;
                    n_weight[c] += if c_weight.is_empty() { 1 } else { c_weight[v] };
                    n_part[c] = c_part[v];
                }
                Some((cmap, n_hoff, n_hnodes, n_noff, n_nhedges, n_weight, n_part))
            }
        };
        match built {
            None => break,
            Some((cmap, n_hoff, n_hnodes, n_noff, n_nhedges, n_weight, n_part)) => {
                lv_cmap.push(cmap);
                lv_hoff.push(n_hoff);
                lv_hnodes.push(n_hnodes);
                lv_noff.push(n_noff);
                lv_nhedges.push(n_nhedges);
                lv_weight.push(n_weight);
                lv_part.push(n_part);
            }
        }
    }

    let nl = lv_hoff.len();
    if nl == 0 {
        return 0;
    }
    let mut total_gain: i64 = 0;
    for l in (0..nl).rev() {
        // Refine level l + 1 (stored at index l), then project it down.
        let m_l = lv_hoff[l].len() - 1;
        let g = kway_fm_polish(
            &mut lv_part[l],
            part_sizes,
            &lv_hoff[l],
            &lv_hnodes[l],
            &lv_noff[l],
            &lv_nhedges[l],
            &lv_weight[l],
            m_l,
            num_parts,
            max_part_size,
            passes,
            max_nonimp,
        );
        total_gain += g;
        if l == 0 {
            let cm = &lv_cmap[0];
            let coarse = &lv_part[0];
            for v in 0..num_nodes {
                partition[v] = coarse[cm[v] as usize];
            }
        } else {
            let cm = &lv_cmap[l];
            let (lo, hi) = lv_part.split_at_mut(l);
            let fine = &mut lo[l - 1];
            let coarse = &hi[0];
            for v in 0..fine.len() {
                fine[v] = coarse[cm[v] as usize];
            }
        }
    }
    // Sizes are maintained in fine-node units by the weighted FM; rebuild
    // them from the projection anyway so the caller's balance gate sees the
    // truth whatever happened above.
    for s in part_sizes.iter_mut() {
        *s = 0;
    }
    for &p in partition.iter() {
        part_sizes[p as usize] += 1;
    }
    total_gain
}


// ============================================================================
// PER-TRACK PARAMETERS
// ----------------------------------------------------------------------------
// Everything that differs between the four tracks, and nothing else.  The list
// is short because the instance family is SELF-SIMILAR: `tig-challenges`
// generates hyperedge sizes from one power law (min 2, max min(1954, n), alpha
// 1-2.5608) and node weights from another (min 1, max 4966, alpha 1-2.2864),
// both INDEPENDENT of the track, so E[hyperedge size] = 5.01 and the mean node
// degree = 5.45 on every track and the degree/size DISTRIBUTIONS are identical
// in shape.  Every threshold in the device code that is derived from those
// distributions (the 32-pin warp threshold, the 256-pin giant threshold, the
// degree-class table, the bit-sliced plane counts) is therefore the same on all
// four tracks by derivation, not by accident.
//
// `num_parts` is likewise NOT a per-track quantity: `Challenge::generate_instance`
// sets `num_parts = 1 << depth` with `depth = 6`, i.e. **64 on every track**.
// The `clusters` hyperparameter (64 / 72 in the benchmarker HPs) is the
// hyperedge CLUSTER count of the construction phase, not the partition count;
// it never reaches the bit-sliced counters, whose width is fixed by the 64-bit
// part mask and the used_degree <= 256 cap.
// ============================================================================

pub struct TrackParams {
    /// Track name, for the probe lines only.
    pub name: &'static str,
    /// Block size of every ORIGINAL (non-fused) kernel of this track.
    pub block_size: u32,
    /// `(refinement, ils_iterations, ils_quick_refine, post_ils_polish,
    /// post_balance)` for effort 0..=5, then index 6 = the catch-all arm.
    pub effort: [(usize, usize, usize, usize, usize); 7],
    pub tabu_tenure: usize,
    pub tabu_fail_tenure: usize,
    pub tabu_mark_base: usize,
    pub tabu_fail_mark_len: usize,
    pub extra_window: usize,
    /// Default of the `fused_mode` hyperparameter (filter geometry).  0 is the
    /// shipped wave6 tile loop and is the 20k default so that track stays
    /// byte-identical to `challengers/budget2`; the three larger tracks default
    /// to 2 (see the `fused_mode` comment inside `solve_with`).
    pub fused_mode: i64,
    // ---- mirrors of the DEVICE-side instantiation (documentation + sanity.py);
    //      the host never branches on these.
    pub dev_num_parts: u32,
    pub dev_warp_pin_threshold: u32,
    pub dev_giant_threshold: u32,
    pub dev_prefix_sampling: bool,
    pub dev_all_parts_scan: bool,
}

impl TrackParams {
    /// Exactly the original `match effort { 5 => .., .., 0 => .., _ => .. }`.
    pub fn effort_preset(&self, effort: i64) -> (usize, usize, usize, usize, usize) {
        match effort {
            0 | 1 | 2 | 3 | 4 | 5 => self.effort[effort as usize],
            _ => self.effort[6],
        }
    }
}

/// The PTX entry points of this track.  All five tracks share one PTX, so every
/// kernel carries the track suffix and the engine can never launch another
/// track's kernel by accident.
pub struct TrackKernels {
    pub hyperedge_clustering: &'static str,
    pub compute_node_preferences: &'static str,
    pub execute_node_assignments: &'static str,
    pub precompute_edge_flags: &'static str,
    pub compute_refinement_moves_optimized: &'static str,
    pub fused_flags_moves: &'static str,
    /// `fused_mode = 0`: the shipped wave6 fused tile loop.
    pub fused_flags_moves_filter: &'static str,
    /// `fused_mode = 1`: the FLAT filter, 128-thread blocks.
    pub fused_flags_moves_filter_flat: &'static str,
    /// `fused_mode = 2`: the FLAT filter, 256-thread blocks.
    pub fused_flags_moves_filter_flatw: &'static str,
    pub balance_final: &'static str,
    pub compute_connectivity: &'static str,
    pub reduce_connectivity_sum: &'static str,
    pub compute_swap_gains_extended: &'static str,
    pub choose_elite_per_hyperedge: &'static str,
    pub assign_from_elite_votes: &'static str,
}

pub fn solve_with(
    challenge: &Challenge,
    save_solution: &dyn Fn(&Solution) -> anyhow::Result<()>,
    hyperparameters: &Option<Map<String, Value>>,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
    tp: &TrackParams,
    kn: &TrackKernels,
) -> anyhow::Result<()> {
    let block_size = std::cmp::min(tp.block_size, prop.maxThreadsPerBlock as u32);

    let hyperedge_cluster_kernel = module.load_function(kn.hyperedge_clustering)?;
    let compute_preferences_kernel = module.load_function(kn.compute_node_preferences)?;
    let execute_assignments_kernel = module.load_function(kn.execute_node_assignments)?;
    let precompute_edge_flags_kernel = module.load_function(kn.precompute_edge_flags)?;

    let compute_moves_kernel = module.load_function(kn.compute_refinement_moves_optimized)?;
    let fused_moves_kernel = module.load_function(kn.fused_flags_moves)?;
    // ---- wave13_parity: FILTER GEOMETRY (`fused_mode`) -------------------
    //   0  the shipped kernel VERBATIM: wave6's fused tile loop, whose launch
    //      issues 2 + 2*ceil(ntiles/gridDim) grid barriers, 128-thread blocks.
    //      On the 20k that is 4 barriers (ntiles 144 <= grid 157) and the
    //      fusion is free; on the larger tracks ntiles is 2.5-10x the grid, so
    //      the SAME code pays 8 / 12 / 20 barriers per round and its per-tile
    //      prefix loops cost O(ntiles^2) L2 loads per round.
    //   1  the FLAT filter: three passes over block-CONTIGUOUS tile ranges,
    //      EXACTLY 4 barriers per launch and O(gridDim) prefixes, 128-thread
    //      blocks.
    //   2  the FLAT filter on 256-thread blocks: 2 x 256 resident threads per
    //      SM instead of 2 x 128.  FINDINGS closed occupancy on the 20k because
    //      its grid is WORK-limited (ceil(20000/128) = 157 < 2*SM = 164); on the
    //      larger tracks fused_blocks_needed is 391 / 781 / 1563 >> 164, so the
    //      kernel runs on 17% of the machine's resident threads and occupancy
    //      is NOT closed there.
    //
    // All three are BIT-IDENTICAL.  The filter emits exactly two derived
    // quantities: the LCG jump-ahead index of an eligible node (= how many
    // eligible nodes have a smaller node id) and the candidate slot of a valid
    // node (= how many valid nodes have a smaller node id).  Both are prefixes
    // over CONSECUTIVE node ranges, so neither depends on the block size nor on
    // which block owns which range.  `tp.fused_mode` is 0 on the 20k, so that
    // track keeps the shipped kernel byte for byte.
    let fused_mode: i64 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("fused_mode").and_then(|v| v.as_i64()))
        .unwrap_or(tp.fused_mode)
        .clamp(0, 2);
    let fused_filter_kernel = module.load_function(match fused_mode {
        0 => kn.fused_flags_moves_filter,
        1 => kn.fused_flags_moves_filter_flat,
        _ => kn.fused_flags_moves_filter_flatw,
    })?;
    let balance_kernel = module.load_function(kn.balance_final)?;
    let compute_connectivity_kernel = module.load_function(kn.compute_connectivity)?;
    let reduce_connectivity_sum_kernel = module.load_function(kn.reduce_connectivity_sum)?;
    let compute_swap_gains_kernel = module.load_function(kn.compute_swap_gains_extended)?;
    let choose_elite_per_hyperedge_kernel = module.load_function(kn.choose_elite_per_hyperedge)?;
    let assign_from_elite_votes_kernel = module.load_function(kn.assign_from_elite_votes)?;

    let cfg = LaunchConfig {
        grid_dim: (
            (challenge.num_nodes as u32 + block_size - 1) / block_size,
            1,
            1,
        ),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    let one_thread_cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };

    let hedge_cfg = LaunchConfig {
        grid_dim: (
            (challenge.num_hyperedges as u32 + block_size - 1) / block_size,
            1,
            1,
        ),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    // exp6: one fused launch per refinement round; grid capped at 2 blocks/SM so all blocks are co-resident
    // (software grid barrier), grid-stride loops cover the rest.
    // wave12_port: the fused kernels are declared __launch_bounds__(<bs>, 2), so
    // they get their OWN block size instead of the track's `block_size` (which
    // is 256 on the <track> and <track> tracks and would be an illegal launch).  The
    // fused kernel's output is independent of the block size -- the candidate
    // array is compacted in global NODE order and the lottery index is the
    // global eligible rank, both computed from tile prefixes -- so this is
    // purely a launch-geometry choice.
    // wave13_parity: mode 2 doubles the block, and therefore the resident warps
    // per SM.  `..._flatw` is declared __launch_bounds__(256, 2), so 2 blocks/SM
    // stays a compiler-enforced co-residency guarantee for the grid barrier.
    let fused_block: u32 = if fused_mode >= 2 {
        std::cmp::min(256, prop.maxThreadsPerBlock as u32)
    } else {
        std::cmp::min(128, prop.maxThreadsPerBlock as u32)
    };
    // The TAIL-phase kernel `fused_flags_moves_*` carries no __launch_bounds__,
    // so its co-residency is whatever the compiler chose for 128 threads.  It is
    // NOT part of this change and keeps 128-thread blocks, on the SAME grid, so
    // the shared monotone `grid_barrier` counter (gridDim.x ticks per barrier)
    // is unaffected.
    let fused_tail_block: u32 = std::cmp::min(128, prop.maxThreadsPerBlock as u32);
    let fused_blocks_needed = std::cmp::max(
        (challenge.num_hyperedges as u32 + fused_block - 1) / fused_block,
        (challenge.num_nodes as u32 + fused_block - 1) / fused_block,
    );
    let fused_grid: u32 = std::cmp::max(1, std::cmp::min(fused_blocks_needed, 2 * std::cmp::max(1, prop.multiProcessorCount as u32)));
    let fused_cfg = LaunchConfig {
        grid_dim: (fused_grid, 1, 1),
        block_dim: (fused_block, 1, 1),
        shared_mem_bytes: 0,
    };
    // Same grid, 128-thread blocks: the tail-phase fused kernel.
    let fused_tail_cfg = LaunchConfig {
        grid_dim: (fused_grid, 1, 1),
        block_dim: (fused_tail_block, 1, 1),
        shared_mem_bytes: 0,
    };

    let connectivity_reduce_cfg = LaunchConfig {
        grid_dim: (
            (challenge.num_hyperedges as u32 + block_size * 2 - 1) / (block_size * 2),
            1,
            1,
        ),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: (block_size as usize * std::mem::size_of::<i32>()) as u32,
    };

    let mut num_hedge_clusters = if let Some(params) = hyperparameters {
        params
            .get("clusters")
            .and_then(|v| v.as_i64())
            .map(|v| v.clamp(4, 256) as i32)
            .unwrap_or(64)
    } else {
        64
    };
    if num_hedge_clusters % 4 != 0 {
        num_hedge_clusters += 4 - (num_hedge_clusters % 4);
    }

    let mut d_hyperedge_clusters = stream.alloc_zeros::<i32>(challenge.num_hyperedges as usize)?;
    let mut d_partition = stream.alloc_zeros::<i32>(challenge.num_nodes as usize)?;
    let mut d_nodes_in_part = stream.alloc_zeros::<i32>(challenge.num_parts as usize)?;
    let mut d_pref_parts = stream.alloc_zeros::<i32>(challenge.num_nodes as usize)?;
    let mut d_pref_priorities = stream.alloc_zeros::<i32>(challenge.num_nodes as usize)?;

    let mut d_move_priorities = stream.alloc_zeros::<i32>(challenge.num_nodes as usize)?;
    let mut d_edge_flags_all = stream.alloc_zeros::<u64>(challenge.num_hyperedges as usize)?;
    let mut d_edge_flags_double = stream.alloc_zeros::<u64>(challenge.num_hyperedges as usize)?;
    // wave4_kmicro2: the refinement loop's fused kernel keeps (flags_all,
    // flags_double) as ONE 16-byte-aligned ulonglong2 array, so each of the
    // ~96k pin gathers of the moves phase is a single 16-byte .cg load instead
    // of two 8-byte loads in two different 32-byte sectors.  The two separate
    // arrays above stay for every OTHER kernel; nothing reads them between
    // refinement rounds (every later consumer is preceded by
    // precompute_edge_flags_<track> or by a fused kernel that writes them itself).
    let mut d_edge_flags_pair = stream.alloc_zeros::<u64>(2 * challenge.num_hyperedges as usize)?;
    let mut d_grid_barrier = stream.alloc_zeros::<u32>(1)?;
    // `fused_calls` counts BARRIERS issued so far (the old fused kernel issues 1 per
    // launch, the new filter kernel issues 3); the absolute barrier target is always
    // (barriers issued) * fused_grid, so the device counter stays monotone.
    let mut fused_calls: u32 = 0;

    // ---- exp12 (wave2 gpufilter) + wave3_kfilter: the main loop's host filter
    // runs on the GPU and its output is compacted GLOBALLY, in node order.
    // Layout of `d_cand`: [0] = number of candidates, then that many
    // (node, key) i32 pairs at [1 + 2*i], [2 + 2*i], strictly ascending in node.
    // The kernel works in TILEs of `fused_block` consecutive nodes (tile t owns
    // nodes [t*bs, (t+1)*bs)); `blk_elig[t]` / `tile_valid[t]` carry the
    // per-tile eligible / valid counts that give the two global prefixes.
    let nblk: usize = fused_grid as usize;
    let ntiles: usize = {
        let bs = fused_block as usize;
        (challenge.num_nodes as usize + bs - 1) / bs
    };
    let ntiles_i32: i32 = ntiles as i32;
    // wave6_kmicro4 (W5): the fused kernel's filter passes now carry TWO grid
    // barriers per tile round instead of two fixed ones, so a launch issues
    // 2 + 2*nrounds barriers.  The kernel derives `nrounds` from exactly these
    // two numbers (ntiles and gridDim.x), so both sides always agree.
    let fused_nrounds: u32 =
        ((ntiles as u32) + fused_grid - 1) / fused_grid;
    // wave13_parity: the FLAT filter issues EXACTLY 4 barriers per launch on
    // every track and every GPU (two before the filter, two inside it); mode 0
    // keeps wave6's 2 + 2*nrounds.  Host and device agree because both derive
    // the count from `fused_mode` and `nrounds` alone.
    let fused_barriers_per_call: u32 =
        if fused_mode >= 1 { 4 } else { 2 + 2 * fused_nrounds };
    let cand_len: usize = 1 + 2 * challenge.num_nodes as usize;
    let mut d_tabu_until = stream.alloc_zeros::<i32>(challenge.num_nodes as usize)?;
    let mut d_mark_nodes = stream.alloc_zeros::<i32>(2 * challenge.num_nodes as usize + 8)?;
    let mut d_max_gain = stream.alloc_zeros::<i32>(1)?;
    let mut d_blk_elig = stream.alloc_zeros::<i32>(std::cmp::max(1, std::cmp::max(nblk, ntiles)))?;
    // wave13_parity: modes 1/2 index this by BLOCK (0..fused_grid), not by tile,
    // so it must hold max(nblk, ntiles) entries.  ntiles > nblk on every track
    // and GPU we run on, so this is a no-op today; it is the bound that keeps
    // the FLAT filter correct on a GPU with more than ntiles/2 SMs.
    let mut d_tile_valid =
        stream.alloc_zeros::<i32>(std::cmp::max(1, std::cmp::max(nblk, ntiles)))?;
    let mut d_cand = stream.alloc_zeros::<i32>(cand_len)?;
    let mut cand_host: Vec<i32> = vec![0i32; cand_len];
    // Adaptive prefix size for the per-round D2H: we copy 1 + 2*cand_cap i32 and
    // only re-copy (rare) if the true count turned out larger. Purely a transfer
    // size; it cannot change any computed value.
    let mut cand_cap: usize = challenge.num_nodes as usize;
    let mut mark_buf: Vec<i32> = Vec::with_capacity(2 * challenge.num_nodes as usize + 8);
    let mut d_connectivity = stream.alloc_zeros::<i32>(challenge.num_hyperedges as usize)?;
    let mut d_total_connectivity = stream.alloc_zeros::<i32>(4096)?;

    let mut d_hedge_choice = stream.alloc_zeros::<i32>(challenge.num_hyperedges as usize)?;

    let swap_buf_size = 3 * challenge.num_nodes as usize;
    let mut d_swap_gains = stream.alloc_zeros::<i32>(swap_buf_size)?;

    let num_parts_usize = challenge.num_parts as usize;
    let is_sparse = (challenge.num_nodes as usize) > 4 * (challenge.num_hyperedges as usize + 1);

    let effort = hyperparameters
        .as_ref()
        .and_then(|p| p.get("effort").and_then(|v| v.as_i64()))
        .unwrap_or(3);

    // The per-track effort ladder lives in `TrackParams`; index 6 is the
    // catch-all arm of the original `match effort`.
    let (base_refine, base_ils, base_ils_quick, base_polish, base_post_balance) =
        tp.effort_preset(effort);

    let refinement_rounds = hyperparameters
        .as_ref()
        .and_then(|p| p.get("refinement").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(50, 50_000) as usize)
        .unwrap_or(base_refine);

    let ils_iterations = hyperparameters
        .as_ref()
        .and_then(|p| p.get("ils_iterations").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(1, 500) as usize)
        .unwrap_or(base_ils);

    let ils_quick_refine = hyperparameters
        .as_ref()
        .and_then(|p| p.get("ils_quick_refine").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(10, 500) as usize)
        .unwrap_or(base_ils_quick);

    let post_ils_polish = hyperparameters
        .as_ref()
        .and_then(|p| p.get("post_ils_polish").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(20, 500) as usize)
        .unwrap_or(base_polish);

    let tabu_tenure: usize = hyperparameters
        .as_ref()
        .and_then(|p| p.get("tabu_tenure").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(1, 30) as usize)
        .unwrap_or(tp.tabu_tenure);

    let tabu_fail_tenure = tp.tabu_fail_tenure;
    let tabu_mark_base = tp.tabu_mark_base;
    let tabu_mark_mult = 16usize;
    let tabu_fail_mark_len = tp.tabu_fail_mark_len;

    let extra_window = tp.extra_window;
    let slack_early = 8usize;
    let slack_mid = 4usize;
    let slack_late = 2usize;

    let move_limit: usize = hyperparameters
        .as_ref()
        .and_then(|p| p.get("move_limit").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(256, 1_000_000) as usize)
        .unwrap_or(if is_sparse {
            262_144
        } else if challenge.num_hyperedges as usize >= 150_000
            || challenge.num_nodes as usize >= 250_000
        {
            131_072
        } else {
            200_000
        });

    let neg_gain_thresh: i32 = 5;
    let scan_limit_swap = 32usize;
    let scan_limit_cycle = 8usize;

    let mut part_to_part: Vec<Vec<(usize, i32)>> = vec![vec![]; num_parts_usize * num_parts_usize];
    let mut swap_gains_host: Vec<i32> = vec![0i32; swap_buf_size];
    let mut partition_host_swap: Vec<i32> = vec![0i32; challenge.num_nodes as usize];
    let mut partition_mut_swap: Vec<i32> = vec![0i32; challenge.num_nodes as usize];
    let mut partition_host_refine: Vec<i32> = vec![0i32; challenge.num_nodes as usize];
    let mut used_ba_buf: Vec<bool> = Vec::with_capacity(1024);
    let mut nodes_in_part_host: Vec<i32> = vec![0i32; num_parts_usize];
    let mut move_keys_host: Vec<i32> = vec![0i32; challenge.num_nodes as usize];

    let hedge_offsets_host = stream.memcpy_dtov(&challenge.d_hyperedge_offsets)?;
    let hyperedge_nodes_host = stream.memcpy_dtov(&challenge.d_hyperedge_nodes)?;
    let mut hedge_sizes_host: Vec<i32> = Vec::with_capacity(challenge.num_hyperedges as usize);
    for h in 0..(challenge.num_hyperedges as usize) {
        let sz = hedge_offsets_host[h + 1] - hedge_offsets_host[h];
        hedge_sizes_host.push(sz);
    }

    // ---- wave4_kmicro2: degree/size-class WARP TASK lists ------------------
    // Built ONCE: both CSRs are fixed for the whole solve, so an item's class
    // is round-invariant.  See ff_build_slots below for the layout.
    let node_offsets_host = stream.memcpy_dtov(&challenge.d_node_offsets)?;
    let mv_units: Vec<(i32, i32, i32)> = (0..challenge.num_nodes as usize)
        .map(|n| {
            let s = node_offsets_host[n];
            (n as i32, s, node_offsets_host[n + 1] - s)
        })
        .collect();
    // wave6_kmicro4 (W1/W7): SLOT-MAJOR task lists.  `ff_warps` is the number of
    // resident warps the fused kernel grid-strides its task list with; the task
    // ORDER is chosen from it so that the warps which have to run a second task
    // run two CHEAP (class-0) ones.
    let ff_warps: usize =
        (fused_grid as usize) * std::cmp::max(1, (fused_block as usize) / 32);
    let (mut mv_slots_host, n_mv_tasks) = ff_build_slots(&mv_units, ff_warps);

    // Hyperedges up to 256 pins go through the warp-task lists; larger ones get
    // a whole 128-thread block each (phase 1a), which turns the 1,954-pin tail
    // from ceil(1954/32) = 61 warp iterations into ceil(1954/128) = 16.
    let mut fe_units: Vec<(i32, i32, i32)> =
        Vec::with_capacity(challenge.num_hyperedges as usize);
    let mut fe_giant_host: Vec<i32> = Vec::new();
    for h in 0..(challenge.num_hyperedges as usize) {
        let st = hedge_offsets_host[h];
        let sz = hedge_sizes_host[h];
        if sz > 256 {
            fe_giant_host.push(h as i32);
            fe_giant_host.push(st);
            fe_giant_host.push(sz);
            fe_giant_host.push(0);
        } else {
            fe_units.push((h as i32, st, sz));
        }
    }
    let (mut fe_slots_host, n_fe_tasks) = ff_build_slots(&fe_units, ff_warps);

    let n_fe_giant: i32 = (fe_giant_host.len() / 4) as i32;
    // A zero-length H2D is not worth risking; the kernel never touches a list
    // whose count is 0, so a one-record dummy is enough.
    if mv_slots_host.is_empty() { mv_slots_host.resize(4, 0); }
    if fe_slots_host.is_empty() { fe_slots_host.resize(4, 0); }
    if fe_giant_host.is_empty() { fe_giant_host.resize(4, 0); }

    let d_mv_slots = stream.memcpy_stod(&mv_slots_host)?;
    let d_fe_slots = stream.memcpy_stod(&fe_slots_host)?;
    let d_fe_giant = stream.memcpy_stod(&fe_giant_host)?;

    let build_high_hedge_ids =
        |connectivity: &[i32], num_high_hedges: usize| -> Vec<i32> {
            let mut impact: Vec<(i32, i32, i32)> = connectivity
                .iter()
                .enumerate()
                .map(|(i, &c)| (c, hedge_sizes_host[i], i as i32))
                .collect();
            impact.sort_unstable_by(|a, b| {
                b.0.cmp(&a.0)
                    .then_with(|| b.1.cmp(&a.1))
                    .then_with(|| a.2.cmp(&b.2))
            });
            impact
                .into_iter()
                .take(num_high_hedges)
                .map(|t| t.2)
                .collect()
        };

    unsafe {
        stream
            .launch_builder(&hyperedge_cluster_kernel)
            .arg(&(challenge.num_hyperedges as i32))
            .arg(&(num_hedge_clusters as i32))
            .arg(&challenge.d_hyperedge_offsets)
            .arg(&challenge.d_hyperedge_nodes)
            .arg(&mut d_hyperedge_clusters)
            .launch(LaunchConfig {
                grid_dim: (
                    (challenge.num_hyperedges as u32 + block_size - 1) / block_size,
                    1,
                    1,
                ),
                block_dim: (block_size, 1, 1),
                shared_mem_bytes: 0,
            })?;
    }

    unsafe {
        stream
            .launch_builder(&compute_preferences_kernel)
            .arg(&(challenge.num_nodes as i32))
            .arg(&(challenge.num_parts as i32))
            .arg(&(num_hedge_clusters as i32))
            .arg(&challenge.d_node_hyperedges)
            .arg(&challenge.d_node_offsets)
            .arg(&d_hyperedge_clusters)
            .arg(&challenge.d_hyperedge_offsets)
            .arg(&mut d_pref_parts)
            .arg(&mut d_pref_priorities)
            .launch(cfg.clone())?;
    }

    let pref_parts = stream.memcpy_dtov(&d_pref_parts)?;
    let pref_priorities = stream.memcpy_dtov(&d_pref_priorities)?;

    let mut indices: Vec<usize> = (0..challenge.num_nodes as usize).collect();
    indices.sort_unstable_by(|&a, &b| {
        pref_priorities[b].cmp(&pref_priorities[a]).then_with(|| a.cmp(&b))
    });

    let sorted_nodes: Vec<i32> = indices.iter().map(|&i| i as i32).collect();
    let sorted_parts: Vec<i32> = indices.iter().map(|&i| pref_parts[i]).collect();

    let d_sorted_nodes = stream.memcpy_stod(&sorted_nodes)?;
    let d_sorted_parts = stream.memcpy_stod(&sorted_parts)?;

    unsafe {
        stream
            .launch_builder(&execute_assignments_kernel)
            .arg(&(challenge.num_nodes as i32))
            .arg(&(challenge.num_parts as i32))
            .arg(&(challenge.max_part_size as i32))
            .arg(&d_sorted_nodes)
            .arg(&d_sorted_parts)
            .arg(&mut d_partition)
            .arg(&mut d_nodes_in_part)
            .launch(one_thread_cfg.clone())?;
    }

    stream.memcpy_dtoh(&d_partition, &mut partition_host_refine)?;
    stream.memcpy_dtoh(&d_nodes_in_part, &mut nodes_in_part_host)?;

    let simulate_execute_moves =
        |partition_mirror: &mut [i32],
         nodes_in_part_mirror: &mut [i32],
         move_nodes: &[i32],
         move_parts: &[i32]|
         -> i32 {
            let mut moves_executed = 0i32;
            for i in 0..move_nodes.len() {
                let node = move_nodes[i];
                let target_part = move_parts[i];
                if node < 0 || target_part < 0 {
                    continue;
                }
                let node_usize = node as usize;
                if node_usize >= partition_mirror.len() {
                    continue;
                }
                let current_part = partition_mirror[node_usize];
                if current_part >= 0
                    && (current_part as usize) < nodes_in_part_mirror.len()
                    && (target_part as usize) < nodes_in_part_mirror.len()
                    && nodes_in_part_mirror[target_part as usize] < challenge.max_part_size as i32
                    && nodes_in_part_mirror[current_part as usize] > 1
                {
                    partition_mirror[node_usize] = target_part;
                    nodes_in_part_mirror[current_part as usize] -= 1;
                    nodes_in_part_mirror[target_part as usize] += 1;
                    moves_executed += 1;
                }
            }
            moves_executed
        };

    let perturb_on_host =
        |partition_mirror: &mut [i32],
         nodes_in_part_mirror: &mut [i32],
         perturb_strength: i32,
         seed: u64| {
            let mut state = seed;
            let mut moves_made = 0i32;
            let target_moves = (challenge.num_nodes as i32 * perturb_strength) / 100;
            for _attempt in 0..(challenge.num_nodes as usize) {
                if moves_made >= target_moves {
                    break;
                }
                state = state
                    .wrapping_mul(6364136223846793005u64)
                    .wrapping_add(1442695040888963407u64);
                let node = (state % challenge.num_nodes as u64) as usize;
                let current_part = partition_mirror[node];
                if current_part < 0 || current_part >= challenge.num_parts as i32 {
                    continue;
                }
                if nodes_in_part_mirror[current_part as usize] <= 1 {
                    continue;
                }
                state = state
                    .wrapping_mul(6364136223846793005u64)
                    .wrapping_add(1442695040888963407u64);
                let target_part = (state % challenge.num_parts as u64) as i32;
                if target_part != current_part
                    && nodes_in_part_mirror[target_part as usize] < challenge.max_part_size as i32
                {
                    partition_mirror[node] = target_part;
                    nodes_in_part_mirror[current_part as usize] -= 1;
                    nodes_in_part_mirror[target_part as usize] += 1;
                    moves_made += 1;
                }
            }
        };

    // wave20 (backbone run): the stagnation mini-perturbation restricted to the
    // FREE node list.  Same LCG, same target-part draw and the same balance
    // guards as `perturb_on_host`; the strength is a percentage of the free
    // set (not of all nodes), so a small sub-problem is not swamped.
    let perturb_free_on_host =
        |partition_mirror: &mut [i32],
         nodes_in_part_mirror: &mut [i32],
         free_nodes: &[usize],
         perturb_strength: i32,
         seed: u64| {
            if free_nodes.is_empty() {
                return;
            }
            let mut state = seed;
            let mut moves_made = 0i32;
            let target_moves = ((free_nodes.len() as i64) * (perturb_strength as i64) / 100) as i32;
            let attempts = free_nodes.len().saturating_mul(4);
            for _attempt in 0..attempts {
                if moves_made >= target_moves {
                    break;
                }
                state = state
                    .wrapping_mul(6364136223846793005u64)
                    .wrapping_add(1442695040888963407u64);
                let node = free_nodes[(state % free_nodes.len() as u64) as usize];
                let current_part = partition_mirror[node];
                if current_part < 0 || current_part >= challenge.num_parts as i32 {
                    continue;
                }
                if nodes_in_part_mirror[current_part as usize] <= 1 {
                    continue;
                }
                state = state
                    .wrapping_mul(6364136223846793005u64)
                    .wrapping_add(1442695040888963407u64);
                let target_part = (state % challenge.num_parts as u64) as i32;
                if target_part != current_part
                    && nodes_in_part_mirror[target_part as usize] < challenge.max_part_size as i32
                {
                    partition_mirror[node] = target_part;
                    nodes_in_part_mirror[current_part as usize] -= 1;
                    nodes_in_part_mirror[target_part as usize] += 1;
                    moves_made += 1;
                }
            }
        };

    // wave29 (closed block): the stagnation kick of a CLOSED part-group block.
    // Same LCG, same node draw, same balance guards as `perturb_free_on_host`,
    // but a node that is free only because its part is in the block (`stable`
    // true and its part in `blk_mask`) draws its kick target from the block's
    // own parts (`blk_parts`), so the kick never leaks a block node into a
    // frozen part.  Disputed-core nodes (`stable` false) draw from every part,
    // exactly as before.
    let perturb_block_on_host =
        |partition_mirror: &mut [i32],
         nodes_in_part_mirror: &mut [i32],
         free_nodes: &[usize],
         stable: &[bool],
         blk_mask: &[bool],
         blk_parts: &[i32],
         perturb_strength: i32,
         seed: u64| {
            if free_nodes.is_empty() || blk_parts.is_empty() {
                return;
            }
            let mut state = seed;
            let mut moves_made = 0i32;
            let target_moves = ((free_nodes.len() as i64) * (perturb_strength as i64) / 100) as i32;
            let attempts = free_nodes.len().saturating_mul(4);
            for _attempt in 0..attempts {
                if moves_made >= target_moves {
                    break;
                }
                state = state
                    .wrapping_mul(6364136223846793005u64)
                    .wrapping_add(1442695040888963407u64);
                let node = free_nodes[(state % free_nodes.len() as u64) as usize];
                let current_part = partition_mirror[node];
                if current_part < 0 || current_part >= challenge.num_parts as i32 {
                    continue;
                }
                if nodes_in_part_mirror[current_part as usize] <= 1 {
                    continue;
                }
                state = state
                    .wrapping_mul(6364136223846793005u64)
                    .wrapping_add(1442695040888963407u64);
                let in_blk: bool = blk_mask.get(current_part as usize).copied().unwrap_or(false);
                let restricted: bool = in_blk && stable.get(node).copied().unwrap_or(false);
                let target_part: i32 = if restricted {
                    blk_parts[(state % blk_parts.len() as u64) as usize]
                } else {
                    (state % challenge.num_parts as u64) as i32
                };
                if target_part != current_part
                    && target_part >= 0
                    && (target_part as usize) < nodes_in_part_mirror.len()
                    && nodes_in_part_mirror[target_part as usize] < challenge.max_part_size as i32
                {
                    partition_mirror[node] = target_part;
                    nodes_in_part_mirror[current_part as usize] -= 1;
                    nodes_in_part_mirror[target_part as usize] += 1;
                    moves_made += 1;
                }
            }
        };

    let perturb_guided_on_host =
        |partition_mirror: &mut [i32],
         nodes_in_part_mirror: &mut [i32],
         high_hedge_ids_host: &[i32]| {
            let np = std::cmp::min(num_parts_usize, 64usize);
            for &hedge_i32 in high_hedge_ids_host.iter() {
                if hedge_i32 < 0 {
                    continue;
                }
                let hedge = hedge_i32 as usize;
                let start = hedge_offsets_host[hedge] as usize;
                let end = hedge_offsets_host[hedge + 1] as usize;
                let hedge_size = end - start;
                if hedge_size <= 1 {
                    continue;
                }

                let mut part_count = [0i32; 64];
                for k in start..end {
                    let node = hyperedge_nodes_host[k] as usize;
                    let part = partition_mirror[node];
                    if part >= 0 && (part as usize) < np {
                        part_count[part as usize] += 1;
                    }
                }

                let mut majority_part = 0usize;
                for p in 1..np {
                    if part_count[p] > part_count[majority_part] {
                        majority_part = p;
                    }
                }

                let mut parts_present = 0i32;
                for p in 0..np {
                    if part_count[p] > 0 {
                        parts_present += 1;
                    }
                }
                if parts_present <= 1 {
                    continue;
                }

                for _iter in 0..np {
                    if parts_present <= 1 {
                        break;
                    }
                    if nodes_in_part_mirror[majority_part] >= challenge.max_part_size as i32 {
                        break;
                    }

                    let mut min_part = usize::MAX;
                    let mut min_cnt = 0i32;
                    for p in 0..np {
                        if p == majority_part {
                            continue;
                        }
                        let cnt = part_count[p];
                        if cnt <= 0 {
                            continue;
                        }
                        if min_part == usize::MAX || cnt < min_cnt || (cnt == min_cnt && p < min_part)
                        {
                            min_part = p;
                            min_cnt = cnt;
                        }
                    }
                    if min_part == usize::MAX {
                        break;
                    }

                    let mut moved_any = false;
                    for k in start..end {
                        if part_count[min_part] <= 0 {
                            break;
                        }
                        if nodes_in_part_mirror[majority_part] >= challenge.max_part_size as i32 {
                            break;
                        }

                        let node = hyperedge_nodes_host[k] as usize;
                        if partition_mirror[node] != min_part as i32 {
                            continue;
                        }
                        if nodes_in_part_mirror[min_part] <= 1 {
                            continue;
                        }

                        partition_mirror[node] = majority_part as i32;
                        nodes_in_part_mirror[min_part] -= 1;
                        nodes_in_part_mirror[majority_part] += 1;
                        part_count[min_part] -= 1;
                        part_count[majority_part] += 1;
                        moved_any = true;
                        if part_count[min_part] == 0 {
                            parts_present -= 1;
                            break;
                        }
                    }

                    if !moved_any {
                        break;
                    }
                }
            }
        };

    let mut sorted_move_nodes: Vec<i32> = Vec::with_capacity(challenge.num_nodes as usize);
    let mut sorted_move_parts: Vec<i32> = Vec::with_capacity(challenge.num_nodes as usize);
    let mut valid_moves: Vec<(usize, i32)> = Vec::with_capacity(challenge.num_nodes as usize);
    // ---- wave4 kwhostsel (= wave3 hostsel, ported onto kwarptask's DENSE cand
    // layout) -----------------------------------------------------------------
    // The MAIN refinement loop (only) works on packed u64 sort keys instead of
    // (usize, i32) tuples:
    //     packed = (((key as u32) ^ 0x7FFF_FFFF) as u64) << 32 | (node as u32 as u64)
    // `x ^ 0x7FFF_FFFF` on the u32 bit pattern of an i32 is the order-REVERSING
    // bijection (it is !((x as u32) ^ 0x8000_0000), i.e. u32::MAX minus the
    // standard order-preserving map), so ascending u64 order on `packed` is
    // exactly the closure `cmp` used everywhere else here:
    //     key DESCENDING as i32, then node ASCENDING.
    // Nodes are unique across candidates, so packed keys are all distinct and
    // every sort/selection below has a single well-defined answer.
    // The other four refine sites keep `valid_moves` and `cmp` untouched.
    let mut vm_keys: Vec<u64> = Vec::with_capacity(challenge.num_nodes as usize);
    // Flat scratch for the counting sort by target part and for the survivors;
    // both are allocated once and never grow (at most one candidate per node,
    // and n_cand is clamped to num_nodes right after the D2H).
    let mut by_buf: Vec<u64> = vec![0u64; challenge.num_nodes as usize + 1];
    let mut sel_keys: Vec<u64> = vec![0u64; challenge.num_nodes as usize + 1];

    // ==== best-of-K: `runs` restarts from the ONE construction result ====
    let bok_runs = hyperparameters
        .as_ref()
        .and_then(|p| p.get("runs").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(1, K_MAX) as usize)
        .unwrap_or(2);

    // ---- wave8: START DIVERSIFICATION knobs (runs k >= 1 only) -------------
    // Every one of these is read ONLY inside `if bok_run >= 1 { .. }`, so run 0
    // -- and therefore `runs == 1` -- is byte-identical to the base solver no
    // matter how they are set.
    //
    // `run_ruin_frac`  : fraction of hyperedges considered for the ruin.  The
    //                    walk stops at `run_ruin_cap` freed nodes, so above
    //                    ~0.02 this knob is nearly inert (the cap binds first);
    //                    that is exactly the July "ruin_k is not a lever"
    //                    observation.  0 disables the ruin entirely.
    // `run_ruin_cap`   : fraction of NODES freed = the real strength knob.
    // `run_ruin_growth_pct`: cap multiplier per extra run, in percent, so a
    //                    large K sweeps a ladder of ruin strengths.  100 = flat.
    // `run_clusters_delta` : run k re-runs CONSTRUCTION with
    //                    `clusters + delta*k` (rounded to a multiple of 4,
    //                    clamped 4..256) before the ruin.  0 = off.
    // `run_tenure_delta`   : run k uses `tabu_tenure + delta*k` (clamped 1..30).
    //                    0 = off.
    let run_ruin_frac: f64 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("run_ruin_frac").and_then(|v| v.as_f64()))
        .unwrap_or(0.02)
        .max(0.0)
        .min(1.0);
    let run_ruin_cap: f64 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("run_ruin_cap").and_then(|v| v.as_f64()))
        .unwrap_or(0.25)
        .max(0.0)
        .min(0.90);
    let run_ruin_growth_pct: i64 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("run_ruin_growth_pct").and_then(|v| v.as_i64()))
        .unwrap_or(100)
        .clamp(25, 400);
    let run_clusters_delta: i64 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("run_clusters_delta").and_then(|v| v.as_i64()))
        .unwrap_or(0)
        .clamp(-252, 252);
    let run_tenure_delta: i64 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("run_tenure_delta").and_then(|v| v.as_i64()))
        .unwrap_or(0)
        .clamp(-29, 29);
    // OPT-IN, default 0.  Set to 1 to ruin run 0's start as well.  This
    // DELIBERATELY breaks the run-0 / K=1 bit-identity, so it is never on by
    // default -- it exists so that ONE build can also measure the ruin's pure
    // REPLACEMENT effect (`{"runs":1,"run_ruin_run0":1}` is exactly the
    // improvement007_ruin configuration on this solver) without a rebuild.
    let run_ruin_run0: i64 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("run_ruin_run0").and_then(|v| v.as_i64()))
        .unwrap_or(0);

    // ---- wave9: PER-RUN FLAVOR selection ----------------------------------
    // `run_flavor_mode` picks the flavor TABLE consulted for runs k >= 2 only.
    // Runs 0 and 1 use `FLAVOR_BASE` in every mode, so `runs = 1` (bit-identical
    // to the single-run solver) and `runs = 2` (the wave8 champion) do not
    // depend on it at all -- this HP is only observable at K >= 3.
    //   0 = legacy   : every k >= 1 = wave8 (top-lambda ruin of the
    //                  construction, `run_ruin_growth_pct` strength ladder).
    //   1 = ladder   : alternating exploit(incumbent) / explore(construction,
    //                  new criterion).  DEFAULT.
    //   2 = explore  : construction only, a different criterion each run.
    //   3 = exploit  : incumbent only, LNS steps of varying strength.
    //   4 = random   : construction + random-edge ruin every run (the
    //                  minimal-change hypothesis: only the freed SET differs).
    //   5 = region   : part-window ruin (July: near-null; kept as a grid arm).
    //   6 = lineage  : alternates previous-run and incumbent as the source.
    let run_flavor_mode: i64 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("run_flavor_mode").and_then(|v| v.as_i64()))
        .unwrap_or(1)
        .clamp(0, 6);
    // MEASUREMENT ONLY, default 0 (= off).  With `runs == 1` and
    // `run_probe_index == j >= 1`, the single run impersonates run index j of a
    // best-of-K: same `run_off`, same flavor, same seeds.  Because per-nonce
    // quality is deterministic, probing j = 0,1,2,3 at 1x cost each yields the
    // EXACT best-of-K curve for K = 1..4 (take the per-nonce min km1 offline),
    // plus the per-nonce complementarity matrix, for 4x instead of 1+2+3+4 = 10x
    // of solver time.  It cannot reproduce a run whose flavor source is the
    // incumbent or the previous run (no earlier run exists): such a run falls
    // back to the construction, which the probe REPORTS by behaving like the
    // corresponding SRC_CONSTRUCTION flavor.  Ignored unless `runs == 1`.
    let run_probe_index: i64 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("run_probe_index").and_then(|v| v.as_i64()))
        .unwrap_or(0)
        .clamp(0, K_MAX - 1);
    let probe_active: bool = run_probe_index >= 1 && bok_runs == 1;
    // ---- wave9: PER-RUN BUDGET SHAPING ------------------------------------
    // Best-of-K pays K x the full pipeline.  The measured value of the second
    // (ruined-start) run is dominated by RESCUES of a bad greedy basin -- on
    // the wave8 10-nonce grid one nonce moved 222,816 -> 274,595, i.e. ~69% of
    // the whole +7,454 K=2 gain came from ONE nonce.  A rescue only needs the
    // second run to reach a NORMAL basin (a ~52,000-point gap), which is one to
    // two orders of magnitude larger than anything a shorter main loop can
    // cost.  So the second run's budget should be the first thing we cut.
    //
    //   run_refinement_pct   main-loop rounds for runs k >= 1, in percent
    //   run0_refinement_pct  main-loop rounds for run 0, in percent
    //   run_ils_pct          ILS / polish / post-balance budgets, runs k >= 1
    //   run0_ils_pct         ILS / polish / post-balance budgets, run 0
    //   run_refine_mode      0 = SCALE the anneal onto the shorter budget
    //                        1 = TRUNCATE: keep the original 45,000-round
    //                            schedule and just stop early
    //   run_screen_pct       0 = off.  When > 0, run k >= 1 falls back to this
    //                        percentage ONLY IF its starting partition scores
    //                        worse (higher km1) than run 0's starting
    //                        partition; if it looks better it keeps the full
    //                        `run_refinement_pct`.  A free host-side signal
    //                        (one `eval_km1` per run, < 1 ms, no GPU work).
    //
    // ALL of them default to a value that reproduces the current behaviour
    // exactly, so this whole block is inert until an HP is set.
    let run_refinement_pct: i64 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("run_refinement_pct").and_then(|v| v.as_i64()))
        .unwrap_or(100)
        .clamp(5, 100);
    let run0_refinement_pct: i64 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("run0_refinement_pct").and_then(|v| v.as_i64()))
        .unwrap_or(100)
        .clamp(5, 100);
    let run_ils_pct: i64 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("run_ils_pct").and_then(|v| v.as_i64()))
        .unwrap_or(100)
        .clamp(5, 100);
    let run0_ils_pct: i64 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("run0_ils_pct").and_then(|v| v.as_i64()))
        .unwrap_or(100)
        .clamp(5, 100);
    // 0 = SCALE (default), 1 = TRUNCATE.  At 100% the two are identical, so
    // this knob cannot change the default behaviour either.
    let run_refine_mode: i64 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("run_refine_mode").and_then(|v| v.as_i64()))
        .unwrap_or(0)
        .clamp(0, 1);
    let run_screen_pct: i64 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("run_screen_pct").and_then(|v| v.as_i64()))
        .unwrap_or(0)
        .clamp(0, 100);
    // The screen needs a run 0 to compare against AND a run 1 to shorten, so
    // at `runs == 1` it is switched off and its `eval_km1` is never issued --
    // `runs == 1` stays identical in COST as well as in output.
    let run_screen_on: bool = run_screen_pct > 0 && bok_runs > 1;

    // ---- wave10: SCHEDULE SHAPE (host-side only; the kernel is untouched) --
    // The device accepts a zero/negative-gain candidate with probability
    //     prob_num / (prob_den_base * (1 + 5*penalty))
    // and the host feeds it `prob_num = rounds_left^2`, `prob_den_base = R^2`,
    // i.e. p(t) = (1 - t)^2 with t = round/R.  Both numbers are computed on the
    // HOST, so the whole acceptance schedule is a host-side design choice and
    // `kernels_<track>.cu` stays byte-identical.
    //
    // wave9 measured that the SHAPE dominates the round count:
    //   run0 at 50% rounds, schedule RESCALED  -> 265,605  (-1,596 vs K=1)
    //   run0 at 50% rounds, schedule TRUNCATED -> 261,142  (-6,059)
    // Halving the budget costs 1,596; ending the loop hot costs another 4,463.
    // Ending COLD is worth ~3x the entire second half of the loop.  wave10
    // therefore parameterises "how the run reaches cold":
    //
    //   run_sched_anneal_pct  the anneal is compressed into the FIRST this-many
    //                         percent of the run's rounds; the remaining rounds
    //                         run with prob_num == 0, i.e. a pure zero-
    //                         temperature QUENCH (only strictly-improving moves
    //                         are ever applied).  100 = today's schedule.
    //   run_sched_exp         the exponent e of p(t) = (1 - t)^e.  2 = today.
    //                         e = 3 removes a quarter of the accepted-worsening
    //                         mass (integral 1/(e+1)) while still ending cold;
    //                         e = 1 is the two-sided control (hotter).
    //   run_sched_cold_mode   0 = today: the stagnation mini-perturbation stays
    //                             armed until `sched_rounds - 50`.
    //                         1 = the mini-perturbation is disarmed once the
    //                             quench starts.  Without it a quench that runs
    //                             out of improving moves would be kicked by a
    //                             3%-of-nodes random perturbation every 3
    //                             rounds and could END on a perturbed state
    //                             (the main loop keeps no incumbent).  With it
    //                             the quench is monotone and the solver's OWN
    //                             `stagnant_rounds > 30` break becomes
    //                             reachable, so the quench costs only as many
    //                             rounds as descent actually needs.
    //
    // All three default to the current behaviour and are selected per run, with
    // a `run0_*` twin so a shape can be measured at `runs = 1` (~6-12 s/nonce)
    // instead of at K=2/K=3 prices.
    let run_sched_anneal_pct: i64 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("run_sched_anneal_pct").and_then(|v| v.as_i64()))
        .unwrap_or(100)
        .clamp(5, 100);
    let run0_sched_anneal_pct: i64 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("run0_sched_anneal_pct").and_then(|v| v.as_i64()))
        .unwrap_or(100)
        .clamp(5, 100);
    // Clamped to 3: the kernel evaluates `prob_den_base * (1 + 5*penalty)` with
    // penalty <= 3, so den <= 16 * A^e.  A <= 50,000 => 16 * A^3 = 2.0e15, far
    // below u64::MAX; A^4 would overflow.
    let run_sched_exp: i64 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("run_sched_exp").and_then(|v| v.as_i64()))
        .unwrap_or(2)
        .clamp(1, 3);
    let run0_sched_exp: i64 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("run0_sched_exp").and_then(|v| v.as_i64()))
        .unwrap_or(2)
        .clamp(1, 3);
    let run_sched_cold_mode: i64 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("run_sched_cold_mode").and_then(|v| v.as_i64()))
        .unwrap_or(0)
        .clamp(0, 1);
    let run0_sched_cold_mode: i64 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("run0_sched_cold_mode").and_then(|v| v.as_i64()))
        .unwrap_or(0)
        .clamp(0, 1);

    // ---- wave10: TAIL SPLIT ------------------------------------------------
    // wave9's `run_ils_pct` scales ils_iterations, ils_quick_refine,
    // post_ils_polish and post_refinement TOGETHER.  Those four are not the
    // same lever: `ils_iterations` buys extra RESTARTS (each an elite/consensus
    // restart + perturbation + quick refine + swaps -- i.e. in-run
    // diversification, the same currency as an extra best-of-K run), while
    // `ils_quick_refine` and the polish buy DEPTH inside one restart and both
    // already self-terminate (`moves_executed == 0 => break`).  These knobs
    // split them, and multiply on top of `run_ils_pct` (so at the defaults, or
    // with only one of the two set, nothing changes twice).
    //   run_ils_iters_pct / run0_ils_iters_pct   % of `ils_iterations`
    //   run_ils_quick_pct / run0_ils_quick_pct   % of `ils_quick_refine`
    //   run_polish_pct    / run0_polish_pct      % of polish + post-balance
    // A percentage can only ever SHRINK a budget (`pct_of` clamps to <= x), so
    // to test MORE ILS restarts there is an absolute override:
    //   run_ils_iters / run0_ils_iters   0 = off, else the exact count.
    let run_ils_iters_pct: i64 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("run_ils_iters_pct").and_then(|v| v.as_i64()))
        .unwrap_or(100)
        .clamp(5, 100);
    let run0_ils_iters_pct: i64 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("run0_ils_iters_pct").and_then(|v| v.as_i64()))
        .unwrap_or(100)
        .clamp(5, 100);
    let run_ils_quick_pct: i64 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("run_ils_quick_pct").and_then(|v| v.as_i64()))
        .unwrap_or(100)
        .clamp(5, 100);
    let run0_ils_quick_pct: i64 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("run0_ils_quick_pct").and_then(|v| v.as_i64()))
        .unwrap_or(100)
        .clamp(5, 100);
    let run_polish_pct: i64 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("run_polish_pct").and_then(|v| v.as_i64()))
        .unwrap_or(100)
        .clamp(5, 100);
    let run0_polish_pct: i64 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("run0_polish_pct").and_then(|v| v.as_i64()))
        .unwrap_or(100)
        .clamp(5, 100);
    // 0 = off (use the percentage path).  Clamped to the same 1..500 window the
    // `ils_iterations` HP itself uses, then to 64 because it also sizes the
    // elite pool allocation (`pool_size * num_nodes` i32s).
    let run_ils_iters: i64 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("run_ils_iters").and_then(|v| v.as_i64()))
        .unwrap_or(0)
        .clamp(0, 64);
    let run0_ils_iters: i64 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("run0_ils_iters").and_then(|v| v.as_i64()))
        .unwrap_or(0)
        .clamp(0, 64);

    // ---- wave10: MAIN-LOOP EARLY STOP -------------------------------------
    // The solver ALREADY has a stagnation break (`stagnant_rounds > 30`), but
    // it is unreachable in practice: `stagnant_rounds` is reset to 0 by the
    // mini-perturbation, which fires every 3 stagnant rounds for every round
    // below `sched_rounds - 50`.  So the break can only trigger inside the last
    // 50 rounds of a 45,000-round loop.
    //
    // `run_stag_stop` (0 = off) adds a SECOND counter that the perturbation
    // does NOT reset: the number of CONSECUTIVE rounds that executed zero
    // moves, counted straight through any perturbation.  The loop breaks once
    // it reaches `run_stag_stop`.
    //
    // This is a HEURISTIC stop, not an exact one, and the notes say so: with
    // the mini-perturbation armed the remaining rounds are not a fixed point,
    // so stopping can change the result.  It is exact only in the quench
    // (`prob_num == 0`) with the perturbation disarmed
    // (`run_sched_cold_mode = 1`) and every tabu stamp expired -- which is
    // exactly the regime `run_sched_cold_mode = 1` creates, and in that regime
    // the solver's own 30-round break already covers it.  MEASURE THE FIRE RATE
    // FIRST (`probe_phases` reports zero-move rounds and the longest streak per
    // run); do not grid this knob until the probe shows the streak can reach
    // the value being tested.
    let run_stag_stop: i64 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("run_stag_stop").and_then(|v| v.as_i64()))
        .unwrap_or(0)
        .clamp(0, 100_000);
    let run0_stag_stop: i64 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("run0_stag_stop").and_then(|v| v.as_i64()))
        .unwrap_or(0)
        .clamp(0, 100_000);

    // ---- wave20: BACKBONE-FROZEN FINAL RUN (metaheuristic) -----------------
    // Fixed-set / backbone decomposition paired with the GPU anneal as the
    // local search.  Every run starts from the SAME construction, so part
    // labels are comparable across runs.  A node that every earlier run's
    // FINAL partition agrees on is the BACKBONE; the nodes the runs disagree
    // on are the DISPUTED set.  The LAST run (k = runs-1, needs runs >= 3 so
    // at least two finals vote) no longer draws a fifth independent explore
    // start: it starts from the incumbent, freezes the backbone (device tabu
    // stamp i32::MAX + a host filter that only lets a frozen node move on a
    // strictly improving candidate) and replays the full hot -> cold anneal
    // on the disputed sub-problem alone.  The per-node search intensity on the
    // disputed nodes is therefore num_nodes / |disputed| times a normal run's,
    // at exactly a normal run's cost (same kernel, same round budget).
    //   bb_run          1 = on (default), 0 = every run is the explore draw.
    //   bb_runs         wave22: how many TRAILING runs are backbone runs
    //                   (default 2).  wave8 measured the third and later
    //                   explore basins at ~nothing (+279 of 274,655), so the
    //                   last explore draw is re-spent as a second backbone
    //                   intensification: ITERATED fixed-set search.  Every
    //                   backbone run starts from the CURRENT incumbent and
    //                   freezes the backbone voted by EVERY finished run so
    //                   far, earlier backbone runs included (their finals can
    //                   only demote a frozen node that a strictly improving
    //                   candidate moved), so the disputed set is re-derived
    //                   between the two intensifications instead of being a
    //                   second draw of the same sub-problem.  A backbone run
    //                   still needs at least two explore finals (k >= 2), so
    //                   runs = 3 keeps exactly one backbone run (200k track)
    //                   and runs = 5 spends runs 3 and 4 on the backbone.
    //   bb_min_pct      the disputed set must hold at least this percent of
    //                   the nodes, else the run falls back to the explore draw.
    //   bb_perturb_pct  stagnation kick strength, percent of the FREE set.
    let bb_run: i64 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("bb_run").and_then(|v| v.as_i64()))
        .unwrap_or(1)
        .clamp(0, 1);
    let bb_runs: usize = hyperparameters
        .as_ref()
        .and_then(|p| p.get("bb_runs").and_then(|v| v.as_i64()))
        .unwrap_or(2)
        .clamp(1, K_MAX) as usize;
    let bb_min_pct: i64 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("bb_min_pct").and_then(|v| v.as_i64()))
        .unwrap_or(1)
        .clamp(0, 100);
    let bb_perturb_pct: i64 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("bb_perturb_pct").and_then(|v| v.as_i64()))
        .unwrap_or(3)
        .clamp(0, 50);
    // wave21: WARM backbone anneal.  The backbone run starts from the
    // incumbent and may only move the disputed nodes, so its local search is
    // an INTENSIFICATION of an already-converged partition, not a restart.
    // Replaying the full hot schedule (every zero-gain move accepted in round
    // 0, 1/6 of the -1 moves) scrambles the disputed regions of the incumbent
    // before the cold tail re-converges them, which makes the run just another
    // draw.  `bb_heat_div` divides the lottery's acceptance rate for the whole
    // backbone run: with the default 4 the run opens at the acceptance a
    // normal run has at its MIDPOINT (zero-gain 1/4, -1 moves 1/24, -3 moves
    // 1/64) and decays quadratically to zero over its full round budget, i.e.
    // it stretches the cold half of a normal anneal over the whole run, on the
    // disputed sub-problem alone.  Device side this is one multiply on the
    // lottery denominator (`prob_den_base`), nothing else changes.  1 = the
    // wave20 hot replay.  Only the backbone run reads it.
    // wave30 (parameter tuning): default 4 -> 2.  wave21 chose 4 when the
    // backbone run freed scattered single nodes and any heat leaked them into
    // frozen parts (the run became "just another draw").  Since wave29 the
    // block run is CLOSED (block nodes drift only among the block's own parts,
    // part sizes conserved) and best-of-K keeps the incumbent whenever the
    // run ends worse, so extra heat inside the block is now a free variance
    // increase: each segment opens at zero-gain acceptance 1/2 (-1 moves 1/12,
    // -3 moves 1/32) instead of 1/4 (1/24, 1/64) and still decays
    // quadratically to cold over the segment, so the block anneal can leave
    // the incumbent's block arrangement instead of re-converging to it.
    let bb_heat_div: u64 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("bb_heat_div").and_then(|v| v.as_i64()))
        .unwrap_or(2)
        .clamp(1, 64) as u64;
    // wave33: RUN 1 AS A WARM REHEAT OF THE INCUMBENT (iterated annealing).
    // Run 1 has always been the wave8 rescue draw: the construction with its
    // hub pins ruined, annealed HOT (every zero-gain move accepted in round 0)
    // over half of run 0's base budget.  The hot opening scrambles the start,
    // so run 1 is an independent half-budget sample of the same construction
    // basin, and with 2-3x fewer rounds than run 0 it is the arm best-of-K
    // almost never keeps; it earns its budget mostly as the second VOTER of
    // the backbone decomposition.  The "exploit the incumbent" flavors of the
    // ladder (runs k >= 2) never execute any more, because wave24/26 skip
    // those runs and re-spend their rounds, so the solver currently has no
    // whole-graph intensification of run 0's final at all: the block backbone
    // run intensifies one part-group at a time with everything else frozen.
    // Run 1 now REHEATS run 0's final instead of re-drawing the construction:
    // it starts from the incumbent with a hub ruin of `run1_ruin_cap_pct`
    // percent of the usual strength (the ILS kick), and its lottery
    // acceptance is divided by `run1_heat_div` for the whole run, so it opens
    // at the acceptance a normal anneal has at its midpoint (zero-gain 1/4,
    // -1 moves 1/24) and decays quadratically to cold over its own budget --
    // the cold half of an anneal, replayed on the whole graph from a
    // converged state.  Same round budget, same kernel, same candidate volume
    // per round (the lottery denominator is the only number that changes), so
    // fuel is unchanged.  The backbone vote still compares run 1's final with
    // run 0's; the disputed set is now the set of nodes the reheat moved and
    // did not move back, which is a sharper "unsettled" signal than the
    // disagreement of two independent draws.  Best-of-K keeps run 0 whenever
    // the reheat ends worse.
    //   run1_reheat        1 = on (default), 0 = the wave8 rescue draw.
    //   run1_heat_div      acceptance divisor of run 1 (default 4).
    //   run1_ruin_cap_pct  hub-ruin strength of run 1 in percent of
    //                      `run_ruin_cap` (default 50; 0 = no ruin, pure reheat).
    let run1_reheat: i64 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("run1_reheat").and_then(|v| v.as_i64()))
        .unwrap_or(1)
        .clamp(0, 1);
    let run1_heat_div: u64 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("run1_heat_div").and_then(|v| v.as_i64()))
        .unwrap_or(4)
        .clamp(1, 64) as u64;
    let run1_ruin_cap_pct: u32 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("run1_ruin_cap_pct").and_then(|v| v.as_i64()))
        .unwrap_or(50)
        .clamp(0, 400) as u32;
    // wave34: DECOMPOSITION + LOCAL SEARCH on the reheat.  The hub ruin frees
    // the pins of the widest-spanning hyperedges, i.e. the giant edges that
    // touch ~every part: the recreate puts most of them straight back and the
    // warm anneal re-settles a boundary the cold run had already settled.
    // With `run1_block_ruin = 1` the reheat's ruin is `CRIT_COUPLED_BLOCK`
    // instead: a block of the most coupled parts is grown (as the block
    // backbone run grows its blocks), the pins of the hyperedges cut INSIDE
    // that block are freed up to the same cap, and they are re-placed into
    // block parts only, so the perturbation is one coherent, closed k-way
    // sub-problem that the warm anneal (unchanged: all nodes free, same
    // rounds, same kernel, same acceptance schedule) then re-decides.
    // Round-neutral; best-of-K still protects run 0.
    //   run1_block_ruin    1 = coupled-block ruin (default), 0 = the hub ruin.
    let run1_block_ruin: i64 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("run1_block_ruin").and_then(|v| v.as_i64()))
        .unwrap_or(1)
        .clamp(0, 1);
    // wave23: HYPEREDGE HALO around the disputed set (decomposition change).
    // The disputed set D is the set of nodes the finished runs DISAGREE on, and
    // the backbone run anneals D with every other node frozen.  But a disputed
    // node's best placement is coupled to its neighbours: the improving move is
    // usually a COORDINATED one (u and a same-hyperedge neighbour v move
    // together, each move alone being zero- or negative-gain), and v, being
    // agreed on by every run, is frozen and only passes on a strictly
    // improving candidate -- which the coordinated move is not, so the
    // sub-problem is over-constrained and re-converges to the incumbent.
    // wave22 measured the second backbone run (a re-draw of the SAME
    // sub-problem) at noise, so the LAST backbone run now solves a DIFFERENT
    // sub-problem: D plus its 1-hop halo through SMALL hyperedges (size <=
    // `bb_halo_size`; the small edges are the tight couplings, the giant ones
    // couple everything to everything and would free the whole graph).  The
    // halo is capped at `bb_halo_pct` percent of |D| so the per-node
    // intensification of the backbone idea is kept (at 100 the free set at
    // most doubles).  Core-then-halo: run k-1 intensifies the disputed core,
    // run k relaxes its boundary.  Deterministic: the core is walked in
    // ascending node id, pins in CSR order.
    //   bb_halo_size    0 = off (wave22 behaviour); default 8 pins.
    //   bb_halo_pct     halo cap as a percent of the disputed-set size.
    //   bb_halo_all     1 = every backbone run gets the halo (default 0 = only
    //                   the last run, so the first backbone run is unchanged).
    let bb_halo_size: usize = hyperparameters
        .as_ref()
        .and_then(|p| p.get("bb_halo_size").and_then(|v| v.as_i64()))
        .unwrap_or(8)
        .clamp(0, 4096) as usize;
    let bb_halo_pct: usize = hyperparameters
        .as_ref()
        .and_then(|p| p.get("bb_halo_pct").and_then(|v| v.as_i64()))
        .unwrap_or(100)
        .clamp(0, 10_000) as usize;
    let bb_halo_all: i64 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("bb_halo_all").and_then(|v| v.as_i64()))
        .unwrap_or(0)
        .clamp(0, 1);
    // wave25: PART-GROUP block decomposition on the LAST backbone run.
    // The disputed set D is scattered over the whole graph, so the backbone
    // anneal on D (and on D plus its 1-hop halo) is a set of ISOLATED node
    // moves with every neighbour frozen: the coordinated re-arrangements that
    // beat a converged incumbent (several pins of one hyperedge changing part
    // together, a part trading a cluster of nodes with another part) stay
    // out of reach, which is why the second core draw measured at noise.
    // Instead of relaxing D node by node, the last backbone run couples whole
    // PARTS: the `bb_group_parts` parts that hold the most disputed nodes of
    // the incumbent are the parts whose internal arrangement the explore runs
    // could not agree on, so EVERY node of those parts is freed (plus D
    // itself), everything else stays frozen.  That is the k-way block
    // sub-problem the host pairwise/group FM operators decompose into, but
    // solved by the GPU warm anneal (coordinated moves, lottery, tabu) on a
    // block that is coupled by construction instead of by a few-ms host FM
    // pass.  With 64 parts the default 8 frees 1/8 of the nodes, the same
    // order as the halo it replaces (`bb_group_parts = 0` restores the halo).
    //   bb_group_parts  number of parts in the block (0 = off, wave24 halo).
    //   bb_group_all    1 = every backbone run gets the block (default 0 =
    //                   only the last run, so the core run is unchanged).
    let bb_group_parts: usize = hyperparameters
        .as_ref()
        .and_then(|p| p.get("bb_group_parts").and_then(|v| v.as_i64()))
        .unwrap_or(8)
        .clamp(0, 64) as usize;
    let bb_group_all: i64 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("bb_group_all").and_then(|v| v.as_i64()))
        .unwrap_or(0)
        .clamp(0, 1);
    // wave27: COUPLED part-group growth (decomposition change).  wave25 picks
    // the block as the `bb_group_parts` parts with the most disputed nodes,
    // ranked independently, so the block is a set of parts that are each
    // internally unsettled but need not be coupled to EACH OTHER: a part
    // trades nodes through the hyperedges it shares with other parts of the
    // block, and two uncoupled parts in the block are two isolated
    // sub-problems the explore finals already converged.  The block is now
    // GROWN: the seed is still the most-disputed part, and every further part
    // is the one with the largest coupling to the block so far (the number of
    // incumbent-cut hyperedges that span it and a block part, counted over
    // hyperedges spanning at most `bb_group_span` parts -- wider edges couple
    // everything to everything), weighted by (1 + its disputed count) so the
    // block still prefers unsettled parts among equally coupled ones.  Ties
    // fall back to the disputed count, then the part id, so with no coupling
    // at all the wave25 ranking is reproduced.  Same block size, same free-set
    // size order, same warm anneal; only WHICH parts are coupled changes.
    //   bb_group_mode   1 = coupled growth (default), 0 = wave25 ranking.
    //   bb_group_span   hyperedges spanning more parts than this are ignored
    //                   in the coupling count (default 8).
    let bb_group_mode: i64 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("bb_group_mode").and_then(|v| v.as_i64()))
        .unwrap_or(1)
        .clamp(0, 1);
    let bb_group_span: usize = hyperparameters
        .as_ref()
        .and_then(|p| p.get("bb_group_span").and_then(|v| v.as_i64()))
        .unwrap_or(8)
        .clamp(2, 64) as usize;
    // wave24: TWO-VOTER BACKBONE, the spare explore draws fund run 0.
    // The backbone decomposition only needs two explore finals to vote (run 0
    // and the rescue run 1); wave8 measured the third and later explore basins
    // at ~nothing as best-of-K arms (+279 of 274,655), and as voters they only
    // widen the disputed set (every extra final can demote nodes, never
    // confirm them), which dilutes the per-node intensification the backbone
    // runs exist for.  So the explore runs strictly between run 1 and the
    // trailing backbone window (k in [2, runs - bb_runs)) are SKIPPED, and the
    // main-loop rounds they would have spent are re-spent as EXTRA rounds of
    // run 0's whole-graph anneal (`run_refinement_pct` per skipped run, on
    // top of run 0's own budget; the schedule is SCALED onto the longer
    // budget, so run 0 still ends cold).  Round-neutral by construction (same
    // number of full-graph kernel rounds, one explore tail fewer), so the
    // fuel does not grow.  Nothing changes with `runs <= 4 - bb_runs + 2`
    // (200k: runs = 3 has no spare draw) nor with `bb_run = 0`.
    //   bb_skip_explore   1 = on (default), 0 = wave23 behaviour.
    let bb_skip_explore: i64 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("bb_skip_explore").and_then(|v| v.as_i64()))
        .unwrap_or(1)
        .clamp(0, 1);
    let bb_skip_lo: usize = 2;
    let bb_skip_hi: usize = if bb_skip_explore != 0 && bb_run != 0 && bok_runs >= 3 {
        bok_runs.saturating_sub(bb_runs)
    } else {
        0
    };
    let bb_skipped: usize = bb_skip_hi.saturating_sub(bb_skip_lo);
    // Extra main-loop budget for run 0, in percent of `refinement`.  Capped so
    // run 0 is at most 5x its own schedule (the lottery arithmetic below is
    // u64 and `round` is an i32, both far from any limit at that cap).
    let run0_bonus_pct: i64 = (bb_skipped as i64)
        .saturating_mul(run_refinement_pct)
        .clamp(0, 400);
    // wave26: BLOCK-ONLY backbone window.  With `bb_runs = 2` the trailing
    // window holds a plain disputed-set run (k = runs-2) and the part-group
    // block run (k = runs-1).  The plain run re-anneals the same sub-problem
    // the explore finals already converged (wave22 measured that re-draw at
    // noise), while the block run is the only one that solves a DIFFERENT,
    // larger sub-problem (disputed set + whole parts).  So the non-last
    // backbone runs are SKIPPED outright (no start, no rounds, no tail, no
    // vote) and their main-loop rounds are re-spent as EXTRA rounds of the
    // LAST backbone run: the block decomposition gets a warm anneal twice as
    // long, scaled onto the longer budget so it still ends cold.  Round-
    // neutral by construction (same number of kernel rounds, one backbone
    // tail fewer).  Nothing changes with `bb_runs = 1`, `runs = 3` (200k: the
    // window is the last run alone) or `bb_run = 0`.
    //   bb_block_only   1 = on (default), 0 = wave25 behaviour.
    let bb_block_only: i64 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("bb_block_only").and_then(|v| v.as_i64()))
        .unwrap_or(1)
        .clamp(0, 1);
    // Skipped backbone runs are k in [bb_blk_lo, bb_blk_hi): the trailing
    // window (k >= 2 and k + bb_runs >= runs) minus its last member.
    let (bb_blk_lo, bb_blk_hi): (usize, usize) =
        if bb_block_only != 0 && bb_run != 0 && bok_runs >= 3 && bb_group_parts > 0 {
            let lo: usize = std::cmp::max(2usize, bok_runs.saturating_sub(bb_runs));
            let hi: usize = bok_runs.saturating_sub(1);
            (lo, hi)
        } else {
            (0usize, 0usize)
        };
    let bb_blk_skipped: usize = bb_blk_hi.saturating_sub(bb_blk_lo);
    // Extra main-loop budget for the LAST run, in percent of `refinement`
    // (same cap and arithmetic as `run0_bonus_pct`).
    let bb_last_bonus_pct: i64 = (bb_blk_skipped as i64)
        .saturating_mul(run_refinement_pct)
        .clamp(0, 400);
    // wave28: SEQUENTIAL DISJOINT BLOCKS (decomposition change, round-neutral).
    // wave26 hands the skipped plain backbone run's rounds to the LAST run,
    // so the ONE part-group block gets a warm anneal twice as long.  But the
    // block is a sub-problem the warm anneal converges well inside a single
    // run's budget (every re-draw of a converged sub-problem has measured at
    // noise), so the second half of the doubled budget re-converges a block
    // that is already at its local optimum while the disputed nodes in the
    // other 56 parts still only get isolated single-node moves.  The last run
    // now spends its budget on a SEQUENCE of DISJOINT part-group blocks: block
    // 0 is the wave27 coupled block, block s >= 1 is grown the same way from
    // the most-disputed part NOT in any earlier block (growth restricted to
    // the parts still untaken), and each block gets its own full warm -> cold
    // anneal over an equal share of the run's rounds (block-coordinate
    // descent over part groups, the GPU anneal as the block solver).  At a
    // block switch the frozen set is re-derived from the CURRENT partition
    // (disputed core stays free, the new block's parts are freed, everything
    // else is frozen) and the device tabu stamps are re-uploaded, exactly as
    // at a run start.  The disputed core is free in every segment, so no
    // segment is a weaker version of wave26's run on the core.
    //   bb_group_seq    number of sequential blocks.  0 (default) = AUTO =
    //                   1 + the number of skipped backbone runs (one block per
    //                   absorbed run's budget: runs = 5 gives 2 blocks, runs =
    //                   3 gives 1 block = wave27 unchanged).  1 = wave27.
    let bb_group_seq: i64 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("bb_group_seq").and_then(|v| v.as_i64()))
        .unwrap_or(0)
        .clamp(0, 8);
    let bbs_segments: usize = if bb_group_seq > 0 {
        bb_group_seq as usize
    } else {
        1usize.saturating_add(bb_blk_skipped)
    };
    // wave29: CLOSED part-group blocks (decomposition change, round-neutral).
    // wave25..28 free every node of the block's parts but let them move to ANY
    // part, so the block is not a closed sub-problem: the anneal's drift
    // (zero-gain plateau moves and lottery-accepted worsening moves) carries
    // block nodes into FROZEN parts, where nothing can respond (a frozen part's
    // own nodes only move on a strictly improving candidate), the receiving
    // part fills towards `max_part_size` and the block's parts thin out.  Every
    // such leak is a one-way street: the block anneal spends its budget
    // re-deciding the block/rest boundary instead of the block's INTERNAL
    // arrangement, which is the k-way sub-problem the vote identified.  With
    // the closed block, a node that is free ONLY because its part is in the
    // block (vote-stable, part in the block) may drift only to the block's own
    // parts; a strictly improving candidate still passes to any part (the
    // same leak rule the frozen filter uses), and the disputed core keeps its
    // unrestricted moves.  The stagnation kick obeys the same rule.  Block
    // part sizes are then conserved under drift, so the sub-problem is exactly
    // "re-arrange the block's nodes among the block's parts, rest fixed".
    // Host-only: one part lookup per candidate on the block run, no kernel
    // change, no extra rounds.  Only block backbone runs read it.
    //   bb_block_closed   1 = closed blocks (default), 0 = wave28 behaviour.
    let bb_block_closed: i64 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("bb_block_closed").and_then(|v| v.as_i64()))
        .unwrap_or(1)
        .clamp(0, 1);
    // wave35: VOTE-GUIDED BLOCK REHEAT instead of the frozen block backbone
    // (decomposition + local search, round-neutral, no kernel change).
    // wave25..29 solve the vote-grown part-group block with everything else
    // FROZEN: the kernel still flags every hyperedge and scores every node
    // each round, but the host filter drops every candidate outside the block
    // (a frozen node passes only on a strictly improving move, a block node's
    // drift is closed onto the block), so a block backbone round executes a
    // small fraction of an explore round's moves for the same fuel, and the
    // block is annealed FROM the incumbent's own arrangement, which the
    // closed-block anneal has measured to re-converge to.  With `bb_reheat =
    // 1` the last run keeps the decomposition (the same disputed-count-seeded,
    // coupling-grown, disjoint part blocks, one per segment) but changes what
    // is done with each block: at the start of segment s the block s is RUINED
    // AND RECREATED as a closed k-way sub-problem (`CRIT_COUPLED_BLOCK` on the
    // preset block: the pins sitting in block parts of the hyperedges cut
    // inside the block are freed, most block parts touched first, up to
    // `bb_reheat_cap_pct` percent of `run_ruin_cap`, and re-placed into block
    // parts only by the shuffled min-added-connectivity recreate), and the
    // segment's warm -> cold anneal then runs on the WHOLE graph with every
    // node free (no frozen filter, no closed-block filter, the plain kick), so
    // every round executes the explore anneal's move volume and the rest of
    // the graph can answer the block's new arrangement.  The perturbation is
    // the structured ruin, not heat, so the segment opens at the reheat
    // acceptance (`bb_reheat_heat_div`, the run-1 value) instead of the
    // closed block's 1/2.  Best-of-K still protects the incumbent.  Same run,
    // same rounds, same kernel, same candidate volume per round.
    //   bb_reheat           1 = on (default), 0 = the wave29 frozen block run.
    //   bb_reheat_heat_div  acceptance divisor of the reheat segments (default 4).
    //   bb_reheat_cap_pct   ruin strength in percent of `run_ruin_cap` (default 50).
    let bb_reheat: i64 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("bb_reheat").and_then(|v| v.as_i64()))
        .unwrap_or(1)
        .clamp(0, 1);
    let bb_reheat_heat_div: u64 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("bb_reheat_heat_div").and_then(|v| v.as_i64()))
        .unwrap_or(4)
        .clamp(1, 64) as u64;
    let bb_reheat_cap_pct: usize = hyperparameters
        .as_ref()
        .and_then(|p| p.get("bb_reheat_cap_pct").and_then(|v| v.as_i64()))
        .unwrap_or(50)
        .clamp(0, 400) as usize;

    // ---- wave10: PHASE PROBE (print-only; 0 = OFF and completely inert) ----
    //   probe_phases 0  off.  No `eval_km1`, no `Instant::now`, no formatting,
    //                    no file handle -- the binary behaves and costs exactly
    //                    like the champion.
    //                1  one line per PHASE per run (start / main_end / swap1 /
    //                    postswap30 / ils_in / one line per ILS iteration /
    //                    ils_end / polish / postbal / final), each with the
    //                    phase's km1 and its wall time in microseconds.
    //                    ~10 host `eval_km1` calls per run (~0.5 ms each) plus
    //                    the ILS lines, whose km1 is the value the solver
    //                    already computes -- well under 1% of a run.
    //                2  1 + a km1 trace of the MAIN LOOP every `probe_every`
    //                    rounds (0 = auto, 100 samples per run).  ~0.5% wall.
    //                    Level 2 is NOT cost-neutral: never quote solve_ms from
    //                    a level-2 run.
    // Output goes to stderr AND is appended to /tmp/probe_phases.txt, one
    // `write_all` per line so the 4 concurrent workers interleave whole lines;
    // every line carries the nonce fingerprint `n=` so they can be separated.
    let probe_phases: i64 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("probe_phases").and_then(|v| v.as_i64()))
        .unwrap_or(0)
        .clamp(0, 3);
    let probe_every: i64 = hyperparameters
        .as_ref()
        .and_then(|p| p.get("probe_every").and_then(|v| v.as_i64()))
        .unwrap_or(0)
        .clamp(0, 1_000_000);
    let probe_on: bool = probe_phases >= 1;
    let probe_trace: bool = probe_phases >= 2;
    // wave13_parity level 3: the PER-ROUND SPLIT of the main loop --
    // (launch + kernel + the runtime's per-launch fuel machinery) / (candidate
    // D2H) / (host selection + apply + the three async H2Ds).  Three
    // `Instant::now()` reads per round, ~60 ns out of a 250-1600 us round
    // (< 0.03%), accumulated in NANOSECONDS and printed once per run on the
    // `phase=main_end` line.  Caveat when reading it: the runtime ends every
    // launch with a pageable 8-byte D2H that drains the stream, so `launch_ns`
    // also contains any H2D still queued from the PREVIOUS round.
    let probe_split: bool = probe_phases >= 3;
    // Read-only fingerprint of the instance so interleaved worker output can be
    // demultiplexed offline.  Only ever read when the probe is on.
    // Only the first FOUR bytes are read: that is the exact slice `track_<track>.rs`
    // already reads (`init_random_seed`), so no assumption is made about the
    // length of `challenge.seed`.
    let probe_nonce: u32 = if probe_on {
        u32::from_le_bytes([
            challenge.seed[0],
            challenge.seed[1],
            challenge.seed[2],
            challenge.seed[3],
        ])
    } else {
        0
    };
    // Scratch for the probe's own `eval_km1` calls, kept separate from
    // `bok_seen` so the probe can never interact with the selection path.
    // Empty (no allocation) when the probe is off.
    let mut probe_seen: Vec<bool> = if probe_on {
        vec![false; num_parts_usize]
    } else {
        Vec::new()
    };
    // Per-phase stopwatch.  `Instant::now()` is called ONLY inside
    // `probe_lap!`, which is only ever expanded inside `if probe_on` blocks, so
    // with the probe off no clock is read anywhere in this file.  The clock
    // never feeds back into any decision -- it is formatted and thrown away.
    let mut probe_mark: Option<std::time::Instant> = None;

    macro_rules! probe_out {
        ($($arg:tt)*) => {{
            let line = format!($($arg)*);
            eprint!("{}", line);
            use std::io::Write;
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open("/tmp/probe_phases.txt")
            {
                let _ = f.write_all(line.as_bytes());
            }
        }};
    }
    macro_rules! probe_lap {
        () => {{
            let d: u64 = match probe_mark {
                Some(t) => t.elapsed().as_micros() as u64,
                None => 0,
            };
            probe_mark = Some(std::time::Instant::now());
            d
        }};
    }
    macro_rules! probe_km1 {
        ($part:expr) => {
            eval_km1(
                $part,
                &hedge_offsets_host,
                &hyperedge_nodes_host,
                challenge.num_hyperedges as usize,
                num_parts_usize,
                &mut probe_seen,
            )
        };
    }

    // Integer derivations, done ONCE.  The two f64 -> usize conversions are a
    // single IEEE multiply plus a truncation of a value that is far below
    // 2^53, so they are exact and identical on every host; everything
    // downstream of this point is integer-only.
    // `+ 0.5` then truncate = round-half-up, so a fraction whose binary double
    // lands a hair below the intended product (0.02 * 20000 = 399.999...) can
    // never silently drop an item.
    let ruin_k_base: usize =
        ((challenge.num_hyperedges as f64) * run_ruin_frac + 0.5) as usize;
    let ruin_cap_base: usize = ((challenge.num_nodes as f64) * run_ruin_cap + 0.5) as usize;
    let ruin_enabled: bool = ruin_k_base >= 1
        && ruin_cap_base >= 1
        && (bok_runs > 1 || run_ruin_run0 != 0);
    // wave9: a probe run (runs == 1, run_probe_index >= 1) also needs the ruin.
    // Purely additive: with the default `run_probe_index == 0` this is a no-op.
    let ruin_enabled: bool =
        ruin_enabled || (probe_active && ruin_k_base >= 1 && ruin_cap_base >= 1);

    // The node->hyperedge CSR is only needed by the recreate step; fetch it
    // once, and only if the ruin can actually fire.
    let node_hedges_host: Vec<i32> = if ruin_enabled {
        stream.memcpy_dtov(&challenge.d_node_hyperedges)?
    } else {
        Vec::new()
    };

    // The construction result, kept verbatim as every run's starting point.
    let bok_base_part = partition_host_refine.clone();
    let bok_base_nip = nodes_in_part_host.clone();
    let mut bok_seen: Vec<bool> = vec![false; num_parts_usize];
    let mut bok_best_part: Vec<u32> = Vec::new();
    let mut bok_best_km1: i64 = i64::MAX;
    // wave9: i32 mirrors of the incumbent and of the previous run's result, so
    // an exploit flavor can ruin them.  Both stay empty until run 0 finishes,
    // and an exploit flavor with an empty source silently falls back to the
    // construction (so nothing can read uninitialised state).
    let mut bok_best_raw: Vec<i32> = Vec::new();
    let mut bok_prev_raw: Vec<i32> = Vec::new();
    // wave9: km1 of run 0's STARTING partition, filled in on run 0 and read by
    // the optional start screen on runs k >= 1.  Only ever written when
    // `run_screen_on`, and only ever read after run 0 has written it (runs are
    // executed in ascending index order), so the i64::MAX sentinel is
    // unreachable at every read.
    let mut bok_start_km1_run0: i64 = i64::MAX;

    // wave20: backbone bookkeeping.  `bb_ref` is the first finished run's
    // final partition; `bb_stable[n]` stays true while every later final
    // agrees with it on node n.  `bb_frozen` / `bb_free_nodes` are the mask
    // and the free list of the run currently executing (only ever non-trivial
    // on the backbone run); `bb_tabu_host` is the staged device stamp vector.
    let bb_n: usize = challenge.num_nodes as usize;
    let mut bb_ref: Vec<i32> = Vec::new();
    let mut bb_stable: Vec<bool> = Vec::new();
    let mut bb_frozen: Vec<bool> = vec![false; bb_n];
    let mut bb_free_nodes: Vec<usize> = Vec::new();
    let mut bb_tabu_host: Vec<i32> = Vec::new();
    // wave28: the per-part membership of every sequential block of the run
    // currently executing (`bbs_blocks[s][p]` = part p is in block s).  Empty
    // on every run that is not a block backbone run.
    let mut bbs_blocks: Vec<Vec<bool>> = Vec::new();
    // wave29: per-part mask and part list of the CURRENT closed block (the
    // block of the segment executing now); `bb_closed_on` is true only on a
    // block backbone run with `bb_block_closed = 1`.  All-false / empty on
    // every other run.
    let mut bb_blk_mask: Vec<bool> = vec![false; num_parts_usize];
    let mut bb_blk_parts: Vec<i32> = Vec::new();
    let mut bb_closed_on: bool = false;

    for bok_run in 0..bok_runs {
        // wave9 (measurement only): with `runs == 1` and `run_probe_index = j`
        // the single iteration impersonates run j.  `probe_active` is false by
        // default, so this shadow is the identity and everything below --
        // `run_off`, the `bok_run >= 1` gate, the flavor, the seeds -- is
        // unchanged.
        let bok_run: usize = if probe_active {
            run_probe_index as usize
        } else {
            bok_run
        };
        // run 0 => run_off == 0 => bit-identical to the base solver.
        let run_off: u64 = (bok_run as u64).wrapping_mul(RUN_OFF_MUL);

        // wave24: a spare explore draw is skipped outright (no start, no
        // rounds, no tail, no vote); its rounds were handed to run 0 below.
        if bok_run >= bb_skip_lo && bok_run < bb_skip_hi {
            continue;
        }
        // wave26: the non-last backbone runs are skipped too; their rounds
        // were handed to the last (block) backbone run below.
        if bok_run >= bb_blk_lo && bok_run < bb_blk_hi {
            continue;
        }

        // ---- per-run state reset ------------------------------------------
        // Host mirrors restart from the construction result.
        partition_host_refine.copy_from_slice(&bok_base_part);
        nodes_in_part_host.copy_from_slice(&bok_base_nip);

        // ---- wave8: START DIVERSIFICATION, runs k >= 1 ONLY ---------------
        // Run 0 never enters this block, so its starting partition is exactly
        // the construction result the base solver saw.  Everything here runs
        // BEFORE the two H2Ds below, so the diversified start is what gets
        // uploaded.
        let mut run_tenure: usize = tabu_tenure;

        // ---- wave20/22: is this a BACKBONE run? ---------------------------
        // The last `bb_runs` runs, each needing at least two earlier EXPLORE
        // finals (k >= 2, i.e. runs >= 3), a valid incumbent and a disputed
        // set of at least `bb_min_pct` percent of the nodes.  On every other
        // run the mask stays all-false and the run body below is the
        // unchanged solver.  `bok_run + bb_runs >= bok_runs` selects exactly
        // the trailing window; with `bb_runs == 1` it is wave20's last-run
        // test (`bok_run + 1 == bok_runs`).
        let mut bb_active: bool = false;
        // wave35: true only on the block backbone run when `bb_reheat = 1`:
        // nothing is frozen, each segment opens with a block ruin instead.
        let mut bbr_on: bool = false;
        bbs_blocks.clear();
        if bb_run != 0
            && bok_runs >= 3
            && bok_run >= 2
            && bok_run.saturating_add(bb_runs) >= bok_runs
            && bok_best_raw.len() == bb_n
            && bb_stable.len() == bb_n
        {
            bb_free_nodes.clear();
            for i in 0..bb_n {
                if !bb_stable[i] {
                    bb_free_nodes.push(i);
                }
            }
            let free_cnt: usize = bb_free_nodes.len();
            if free_cnt > 0 && free_cnt.saturating_mul(100) >= bb_n.saturating_mul(bb_min_pct as usize) {
                let mut sizes_bb: Vec<i32> = vec![0i32; num_parts_usize];
                let mut src_valid = true;
                for &p in bok_best_raw.iter() {
                    if p >= 0 && (p as usize) < num_parts_usize {
                        sizes_bb[p as usize] += 1;
                    } else {
                        src_valid = false;
                    }
                }
                for p in 0..num_parts_usize {
                    if sizes_bb[p] < 1 || sizes_bb[p] > challenge.max_part_size as i32 {
                        src_valid = false;
                    }
                }
                if src_valid {
                    partition_host_refine.copy_from_slice(&bok_best_raw);
                    nodes_in_part_host.copy_from_slice(&sizes_bb);
                    for i in 0..bb_n {
                        bb_frozen[i] = bb_stable[i];
                    }
                    // wave23: hyperedge halo on the LAST backbone run (or on
                    // every backbone run with `bb_halo_all`).  Walks the
                    // disputed core (ascending node id) and unfreezes every pin
                    // of its incident hyperedges of size <= `bb_halo_size`,
                    // up to `bb_halo_pct` percent of the core size.  The core
                    // bound is fixed BEFORE the walk, so this is a 1-hop halo
                    // only: halo nodes do not recruit their own neighbours.
                    // wave25: the part-group block REPLACES the halo on the
                    // runs it fires on; with `bb_group_parts = 0` the halo
                    // gate below is exactly wave24's.
                    let group_on: bool = bb_group_parts > 0
                        && (bb_group_all != 0 || bok_run.saturating_add(1) == bok_runs);
                    let halo_on: bool = bb_halo_size >= 2
                        && bb_halo_pct > 0
                        && !group_on
                        && (bb_halo_all != 0 || bok_run.saturating_add(1) == bok_runs)
                        && node_offsets_host.len() >= bb_n + 1
                        && !node_hedges_host.is_empty();
                    if group_on {
                        // Rank the parts by how many DISPUTED nodes the
                        // incumbent holds in them (ties: ascending part id,
                        // so the ranking is deterministic), take the top
                        // `bb_group_parts` (at most half the parts, so the
                        // sub-problem stays a sub-problem), and free every
                        // node of those parts.  The disputed core walked
                        // here is the vote's own list (before any growth).
                        let core_cnt: usize = bb_free_nodes.len();
                        let mut disp_cnt: Vec<usize> = vec![0usize; num_parts_usize];
                        for idx in 0..core_cnt {
                            let u: usize = bb_free_nodes[idx];
                            let p = partition_host_refine[u];
                            if p >= 0 && (p as usize) < num_parts_usize {
                                disp_cnt[p as usize] += 1;
                            }
                        }
                        let mut order: Vec<usize> = (0..num_parts_usize).collect();
                        order.sort_by(|&a, &b| disp_cnt[b].cmp(&disp_cnt[a]).then(a.cmp(&b)));
                        let g_cap: usize = std::cmp::max(1, num_parts_usize / 2);
                        let g: usize = std::cmp::min(bb_group_parts, g_cap);
                        let mut in_group: Vec<bool> = vec![false; num_parts_usize];
                        // wave28: the coupling matrix outlives the block-0
                        // growth so the later disjoint blocks reuse it.
                        let mut aff: Vec<i64> = Vec::new();
                        if bb_group_mode != 0 && g >= 2 && !order.is_empty() {
                            // wave27: grow the block by inter-part coupling.
                            // aff[a*np+b] = number of incumbent-cut hyperedges
                            // spanning both part a and part b (span capped at
                            // `bb_group_span` parts).  One host pass over the
                            // pins, once per block run: a few ms.
                            let np: usize = num_parts_usize;
                            let n_hedges: usize = challenge.num_hyperedges as usize;
                            aff.resize(np * np, 0i64);
                            let mut span: Vec<usize> = Vec::with_capacity(bb_group_span + 1);
                            for h in 0..n_hedges {
                                if h + 1 >= hedge_offsets_host.len() {
                                    break;
                                }
                                let hs = hedge_offsets_host[h] as usize;
                                let he = hedge_offsets_host[h + 1] as usize;
                                if he < hs + 2 || he > hyperedge_nodes_host.len() {
                                    continue;
                                }
                                span.clear();
                                let mut too_wide: bool = false;
                                for j in hs..he {
                                    let v_i32 = hyperedge_nodes_host[j];
                                    if v_i32 < 0 || (v_i32 as usize) >= bb_n {
                                        continue;
                                    }
                                    let p = partition_host_refine[v_i32 as usize];
                                    if p < 0 || (p as usize) >= np {
                                        continue;
                                    }
                                    let pu: usize = p as usize;
                                    if !span.contains(&pu) {
                                        if span.len() >= bb_group_span {
                                            too_wide = true;
                                            break;
                                        }
                                        span.push(pu);
                                    }
                                }
                                if too_wide || span.len() < 2 {
                                    continue;
                                }
                                for a in 0..span.len() {
                                    for b in (a + 1)..span.len() {
                                        let pa: usize = span[a];
                                        let pb: usize = span[b];
                                        aff[pa * np + pb] += 1;
                                        aff[pb * np + pa] += 1;
                                    }
                                }
                            }
                            // Seed with the most-disputed part, then add the
                            // part most coupled to the block (disputed-
                            // weighted), g-1 times.
                            let seed: usize = order[0];
                            in_group[seed] = true;
                            let mut to_group: Vec<i64> = vec![0i64; np];
                            for q in 0..np {
                                to_group[q] = aff[q * np + seed];
                            }
                            for _step in 1..g {
                                let mut best: Option<usize> = None;
                                let mut best_score: i64 = -1;
                                for q in 0..np {
                                    if in_group[q] {
                                        continue;
                                    }
                                    let score: i64 =
                                        to_group[q].saturating_mul(disp_cnt[q] as i64 + 1);
                                    let better: bool = match best {
                                        None => true,
                                        Some(bq) => {
                                            score > best_score
                                                || (score == best_score
                                                    && (disp_cnt[q] > disp_cnt[bq]
                                                        || (disp_cnt[q] == disp_cnt[bq] && q < bq)))
                                        }
                                    };
                                    if better {
                                        best = Some(q);
                                        best_score = score;
                                    }
                                }
                                match best {
                                    Some(bq) => {
                                        in_group[bq] = true;
                                        for q in 0..np {
                                            to_group[q] = to_group[q].saturating_add(aff[q * np + bq]);
                                        }
                                    }
                                    None => break,
                                }
                            }
                        } else {
                            for &p in order.iter().take(g) {
                                in_group[p] = true;
                            }
                        }
                        // wave28: grow the further DISJOINT blocks now, from
                        // the same disputed counts and coupling matrix, so the
                        // whole sequence is fixed by the incumbent at run
                        // start (deterministic, one host pass).  Block s is
                        // seeded with the most-disputed part not yet taken and
                        // grown over the untaken parts only; a block that
                        // finds no seed ends the sequence early.
                        bbs_blocks.clear();
                        bbs_blocks.push(in_group.clone());
                        if bbs_segments >= 2 && bok_run.saturating_add(1) == bok_runs {
                            let np: usize = num_parts_usize;
                            let mut taken: Vec<bool> = in_group.clone();
                            for _s in 1..bbs_segments {
                                let mut seed_opt: Option<usize> = None;
                                for &p in order.iter() {
                                    if !taken[p] {
                                        seed_opt = Some(p);
                                        break;
                                    }
                                }
                                let seed2: usize = match seed_opt {
                                    Some(s) => s,
                                    None => break,
                                };
                                let mut grp: Vec<bool> = vec![false; np];
                                grp[seed2] = true;
                                taken[seed2] = true;
                                if bb_group_mode != 0 && g >= 2 && aff.len() == np * np {
                                    let mut to_grp: Vec<i64> = vec![0i64; np];
                                    for q in 0..np {
                                        to_grp[q] = aff[q * np + seed2];
                                    }
                                    for _step in 1..g {
                                        let mut best: Option<usize> = None;
                                        let mut best_score: i64 = -1;
                                        for q in 0..np {
                                            if taken[q] {
                                                continue;
                                            }
                                            let score: i64 =
                                                to_grp[q].saturating_mul(disp_cnt[q] as i64 + 1);
                                            let better: bool = match best {
                                                None => true,
                                                Some(bq) => {
                                                    score > best_score
                                                        || (score == best_score
                                                            && (disp_cnt[q] > disp_cnt[bq]
                                                                || (disp_cnt[q] == disp_cnt[bq]
                                                                    && q < bq)))
                                                }
                                            };
                                            if better {
                                                best = Some(q);
                                                best_score = score;
                                            }
                                        }
                                        match best {
                                            Some(bq) => {
                                                grp[bq] = true;
                                                taken[bq] = true;
                                                for q in 0..np {
                                                    to_grp[q] =
                                                        to_grp[q].saturating_add(aff[q * np + bq]);
                                                }
                                            }
                                            None => break,
                                        }
                                    }
                                } else {
                                    let mut cnt: usize = 1;
                                    for &p in order.iter() {
                                        if cnt >= g {
                                            break;
                                        }
                                        if !taken[p] {
                                            grp[p] = true;
                                            taken[p] = true;
                                            cnt += 1;
                                        }
                                    }
                                }
                                bbs_blocks.push(grp);
                            }
                        }
                        for v in 0..bb_n {
                            if !bb_frozen[v] {
                                continue;
                            }
                            let p = partition_host_refine[v];
                            if p >= 0 && (p as usize) < num_parts_usize && in_group[p as usize] {
                                bb_frozen[v] = false;
                                bb_free_nodes.push(v);
                            }
                        }
                        // wave35: vote-guided block REHEAT.  Every node is
                        // free (the frozen filter sees an all-false mask and
                        // passes everything, exactly as on an explore run),
                        // and block 0 is ruined and recreated as a closed
                        // k-way sub-problem on the preset vote-grown block
                        // before the segment's whole-graph warm anneal.  The
                        // later blocks are ruined at their segment switch.
                        if bb_reheat != 0
                            && bok_run.saturating_add(1) == bok_runs
                            && !bbs_blocks.is_empty()
                            && !node_hedges_host.is_empty()
                            && node_offsets_host.len() >= bb_n + 1
                        {
                            bbr_on = true;
                            bb_free_nodes.clear();
                            for v in 0..bb_n {
                                bb_frozen[v] = false;
                                bb_free_nodes.push(v);
                            }
                            let cap_r: usize = (ruin_cap_base
                                .saturating_mul(bb_reheat_cap_pct)
                                / 100)
                                .min(challenge.num_nodes as usize);
                            let seed_r: u64 = (bok_run as u64)
                                .wrapping_mul(RUIN_SEED_MUL)
                                .wrapping_add(RUIN_SEED_ADD)
                                | 1;
                            let blk0: Vec<bool> = bbs_blocks[0].clone();
                            let _ = ruin_recreate(
                                &mut partition_host_refine,
                                &mut nodes_in_part_host,
                                &hedge_offsets_host,
                                &hyperedge_nodes_host,
                                &node_offsets_host,
                                &node_hedges_host,
                                challenge.num_hyperedges as usize,
                                num_parts_usize,
                                challenge.max_part_size as i32,
                                usize::MAX,
                                cap_r,
                                seed_r,
                                CRIT_COUPLED_BLOCK,
                                seed_r,
                                &blk0,
                            );
                        }
                    }
                    if halo_on {
                        let core_cnt: usize = bb_free_nodes.len();
                        let halo_cap: usize = core_cnt.saturating_mul(bb_halo_pct) / 100;
                        let mut added: usize = 0;
                        let n_hedges: usize = challenge.num_hyperedges as usize;
                        'halo: for idx in 0..core_cnt {
                            let u: usize = bb_free_nodes[idx];
                            let ns = node_offsets_host[u] as usize;
                            let ne = node_offsets_host[u + 1] as usize;
                            if ne > node_hedges_host.len() || ns > ne {
                                continue;
                            }
                            for k in ns..ne {
                                let h_i32 = node_hedges_host[k];
                                if h_i32 < 0 || (h_i32 as usize) >= n_hedges {
                                    continue;
                                }
                                let h = h_i32 as usize;
                                let hs = hedge_offsets_host[h] as usize;
                                let he = hedge_offsets_host[h + 1] as usize;
                                if he < hs + 2 || he - hs > bb_halo_size || he > hyperedge_nodes_host.len() {
                                    continue;
                                }
                                for j in hs..he {
                                    let v_i32 = hyperedge_nodes_host[j];
                                    if v_i32 < 0 {
                                        continue;
                                    }
                                    let v = v_i32 as usize;
                                    if v < bb_n && bb_frozen[v] {
                                        if added >= halo_cap {
                                            break 'halo;
                                        }
                                        bb_frozen[v] = false;
                                        bb_free_nodes.push(v);
                                        added += 1;
                                    }
                                }
                            }
                        }
                    }
                    bb_active = true;
                }
            }
        }
        if !bb_active {
            for v in bb_frozen.iter_mut() {
                *v = false;
            }
            bb_free_nodes.clear();
        }
        // wave29: stage the closed-block mask of block 0 (segment 0).  Empty
        // `bbs_blocks` (plain backbone / halo run, or an explore run) leaves
        // the mask all-false and the closure off.
        // wave35: the reheat run is open (no closed-block drift filter).
        bb_closed_on = bb_active && bb_block_closed != 0 && !bbs_blocks.is_empty() && !bbr_on;
        for p in 0..num_parts_usize {
            bb_blk_mask[p] = if bb_closed_on {
                bbs_blocks[0].get(p).copied().unwrap_or(false)
            } else {
                false
            };
        }
        bb_blk_parts.clear();
        for p in 0..num_parts_usize {
            if bb_blk_mask[p] {
                bb_blk_parts.push(p as i32);
            }
        }

        // Run 0 enters this block ONLY if `run_ruin_run0` was explicitly set,
        // so by default run 0 -- and therefore `runs == 1` -- is byte-identical
        // to the base solver.  Inside, every lever is scaled by `bok_run`, so
        // on run 0 the cluster and tenure levers are no-ops by construction
        // (delta * 0 == 0) and only the ruin fires.
        // wave20: the backbone run starts from the incumbent as-is (no ruin,
        // no re-construction); the anneal on the disputed set is its ruin.
        // wave33: run 1 reheats the incumbent.  Needs run 0's final (always
        // present at k = 1: runs execute in ascending order and run 0 is never
        // skipped); on a backbone run the gate is off by construction.
        let run1_reheat_on: bool = run1_reheat != 0
            && bok_run == 1
            && !bb_active
            && !probe_active
            && bok_best_raw.len() == bb_n;
        if (bok_run >= 1 || run_ruin_run0 != 0) && !bb_active {
            // (B) optional RE-CONSTRUCTION with a different cluster count.
            //     `hyperedge_clustering_<track>` is a pure function of the CSR and
            //     `num_clusters`, and `compute_node_preferences_<track>` folds the
            //     cluster id into a part with `(best_cluster*num_parts)/
            //     min(num_clusters,64)`, so changing the cluster count moves
            //     the whole greedy assignment.  Off by default.
            if run_clusters_delta != 0 {
                let mut c_run: i32 = ((num_hedge_clusters as i64)
                    + run_clusters_delta * (bok_run as i64))
                    .clamp(4, 256) as i32;
                if c_run % 4 != 0 {
                    c_run += 4 - (c_run % 4);
                }
                if c_run != num_hedge_clusters {
                    // `execute_node_assignments_<track>` ACCUMULATES into
                    // nodes_in_part and only writes partition[node] for nodes
                    // it can place, so both must be zeroed first.
                    stream.memset_zeros(&mut d_partition)?;
                    stream.memset_zeros(&mut d_nodes_in_part)?;
                    unsafe {
                        stream
                            .launch_builder(&hyperedge_cluster_kernel)
                            .arg(&(challenge.num_hyperedges as i32))
                            .arg(&c_run)
                            .arg(&challenge.d_hyperedge_offsets)
                            .arg(&challenge.d_hyperedge_nodes)
                            .arg(&mut d_hyperedge_clusters)
                            .launch(LaunchConfig {
                                grid_dim: (
                                    (challenge.num_hyperedges as u32 + block_size - 1)
                                        / block_size,
                                    1,
                                    1,
                                ),
                                block_dim: (block_size, 1, 1),
                                shared_mem_bytes: 0,
                            })?;
                    }
                    unsafe {
                        stream
                            .launch_builder(&compute_preferences_kernel)
                            .arg(&(challenge.num_nodes as i32))
                            .arg(&(challenge.num_parts as i32))
                            .arg(&c_run)
                            .arg(&challenge.d_node_hyperedges)
                            .arg(&challenge.d_node_offsets)
                            .arg(&d_hyperedge_clusters)
                            .arg(&challenge.d_hyperedge_offsets)
                            .arg(&mut d_pref_parts)
                            .arg(&mut d_pref_priorities)
                            .launch(cfg.clone())?;
                    }
                    let pref_parts_run = stream.memcpy_dtov(&d_pref_parts)?;
                    let pref_priorities_run = stream.memcpy_dtov(&d_pref_priorities)?;
                    let mut idx_run: Vec<usize> = (0..challenge.num_nodes as usize).collect();
                    idx_run.sort_unstable_by(|&a, &b| {
                        pref_priorities_run[b]
                            .cmp(&pref_priorities_run[a])
                            .then_with(|| a.cmp(&b))
                    });
                    let sorted_nodes_run: Vec<i32> =
                        idx_run.iter().map(|&i| i as i32).collect();
                    let sorted_parts_run: Vec<i32> =
                        idx_run.iter().map(|&i| pref_parts_run[i]).collect();
                    let d_sorted_nodes_run = stream.memcpy_stod(&sorted_nodes_run)?;
                    let d_sorted_parts_run = stream.memcpy_stod(&sorted_parts_run)?;
                    unsafe {
                        stream
                            .launch_builder(&execute_assignments_kernel)
                            .arg(&(challenge.num_nodes as i32))
                            .arg(&(challenge.num_parts as i32))
                            .arg(&(challenge.max_part_size as i32))
                            .arg(&d_sorted_nodes_run)
                            .arg(&d_sorted_parts_run)
                            .arg(&mut d_partition)
                            .arg(&mut d_nodes_in_part)
                            .launch(one_thread_cfg.clone())?;
                    }
                    stream.memcpy_dtoh(&d_partition, &mut partition_host_refine)?;
                    // Re-derive the part sizes on the host rather than trusting
                    // the kernel's accumulator: every balance guard downstream
                    // (simulate_execute_moves, the quota select, the ruin) is
                    // only as correct as this vector, and a silent drift here
                    // would be invisible until the solution failed validation.
                    for v in nodes_in_part_host.iter_mut() {
                        *v = 0;
                    }
                    for &p in partition_host_refine.iter() {
                        if p >= 0 && (p as usize) < num_parts_usize {
                            nodes_in_part_host[p as usize] += 1;
                        }
                    }
                }
            }

            // (A) RUIN-AND-RECREATE of the starting partition.  On by default.
            if ruin_enabled {
                // ---- wave9: pick this run's FLAVOR -------------------------
                // Runs 0 and 1 -- and every run in legacy mode 0 -- use the
                // wave8 recipe, so `runs = 1` and `runs = 2` are unaffected by
                // `run_flavor_mode`.
                let flavor: RunFlavor = if run1_reheat_on {
                    // wave33: hub ruin of the INCUMBENT at the reheat strength.
                    // wave34: coupled-block ruin instead (walks the whole
                    // in-block cut list; the cap is what stops it).
                    if run1_block_ruin != 0 {
                        RunFlavor {
                            src: SRC_INCUMBENT,
                            crit: CRIT_COUPLED_BLOCK,
                            cap_pct: run1_ruin_cap_pct,
                            k_all: true,
                        }
                    } else {
                        RunFlavor {
                            src: SRC_INCUMBENT,
                            crit: CRIT_TOP_LAMBDA,
                            cap_pct: run1_ruin_cap_pct,
                            k_all: false,
                        }
                    }
                } else if run_flavor_mode == 0 || bok_run <= 1 {
                    FLAVOR_BASE
                } else {
                    flavor_of(run_flavor_mode, bok_run)
                };

                // ---- wave9: the SOURCE partition --------------------------
                // Default (`SRC_CONSTRUCTION`) leaves the host mirrors exactly
                // as the reset above left them = the construction result.  An
                // exploit flavor instead restarts from an already-refined
                // partition and re-derives the part sizes from it rather than
                // trusting any accumulator; if that partition is missing or
                // malformed it falls back to the construction, so the operator
                // can never read uninitialised or invalid state.
                if flavor.src != SRC_CONSTRUCTION {
                    let n_nodes = partition_host_refine.len();
                    let use_best = flavor.src == SRC_INCUMBENT;
                    let src_ok = if use_best {
                        bok_best_raw.len() == n_nodes
                    } else {
                        bok_prev_raw.len() == n_nodes
                    };
                    if src_ok {
                        if use_best {
                            partition_host_refine.copy_from_slice(&bok_best_raw);
                        } else {
                            partition_host_refine.copy_from_slice(&bok_prev_raw);
                        }
                        for v in nodes_in_part_host.iter_mut() {
                            *v = 0;
                        }
                        let mut src_valid = true;
                        for &p in partition_host_refine.iter() {
                            if p >= 0 && (p as usize) < num_parts_usize {
                                nodes_in_part_host[p as usize] += 1;
                            } else {
                                src_valid = false;
                            }
                        }
                        if !src_valid {
                            partition_host_refine.copy_from_slice(&bok_base_part);
                            nodes_in_part_host.copy_from_slice(&bok_base_nip);
                        }
                    }
                }

                // ---- strength ---------------------------------------------
                // Legacy path (mode 0, or runs 0/1 in any mode): run 1 gets
                // `ruin_cap_base`, run k gets it multiplied by
                // (growth_pct/100)^(k-1), integer arithmetic.  On runs 0 and 1
                // the loop body never executes, so `cap == ruin_cap_base`
                // exactly, as in wave8.
                // Flavor path: cap = ruin_cap_base * cap_pct / 100, and
                // `run_ruin_growth_pct` is ignored for k >= 2 (the table owns
                // the ladder).
                let cap: usize = if run1_reheat_on {
                    // wave33: the reheat's own strength; 0 = no ruin at all
                    // (`ruin_recreate` declines on `free_cap == 0`).
                    (ruin_cap_base.saturating_mul(flavor.cap_pct as usize) / 100)
                        .min(challenge.num_nodes as usize)
                } else if run_flavor_mode == 0 || bok_run <= 1 {
                    let mut cap: usize = ruin_cap_base;
                    for _ in 1..bok_run {
                        cap = cap.saturating_mul(run_ruin_growth_pct as usize) / 100;
                        if cap >= challenge.num_nodes as usize {
                            cap = challenge.num_nodes as usize;
                            break;
                        }
                    }
                    cap.clamp(1, challenge.num_nodes as usize)
                } else {
                    (ruin_cap_base.saturating_mul(flavor.cap_pct as usize) / 100)
                        .clamp(1, challenge.num_nodes as usize)
                };
                // `k_all` walks the whole ordered candidate list; the cap is
                // then the only thing that stops the walk (July: "ruin_k is NOT
                // a useful lever" once the cap binds).
                let k_ruin_run: usize = if flavor.k_all {
                    usize::MAX
                } else {
                    ruin_k_base
                };
                let ruin_seed: u64 = (bok_run as u64)
                    .wrapping_mul(RUIN_SEED_MUL)
                    .wrapping_add(RUIN_SEED_ADD)
                    | 1;
                // Distinct from `ruin_seed`: which edges are ruined and in what
                // order they are put back must not move in lockstep.
                let crit_seed: u64 = (bok_run as u64)
                    .wrapping_mul(CRIT_SEED_MUL)
                    .wrapping_add(CRIT_SEED_ADD)
                    | 1;
                // Return value deliberately ignored: `false` means the operator
                // declined and left both host mirrors untouched, i.e. run k
                // simply starts from its (possibly incumbent) source partition,
                // still diversified by `run_off`.
                let _ = ruin_recreate(
                    &mut partition_host_refine,
                    &mut nodes_in_part_host,
                    &hedge_offsets_host,
                    &hyperedge_nodes_host,
                    &node_offsets_host,
                    &node_hedges_host,
                    challenge.num_hyperedges as usize,
                    num_parts_usize,
                    challenge.max_part_size as i32,
                    k_ruin_run,
                    cap,
                    ruin_seed,
                    flavor.crit,
                    crit_seed,
                    &[],
                );
            }

            // (C) optional STRUCTURAL knob: tabu tenure per run.  Off by
            //     default.  `run_tenure` shadows `tabu_tenure` inside the run
            //     body below.
            if run_tenure_delta != 0 {
                run_tenure = ((tabu_tenure as i64) + run_tenure_delta * (bok_run as i64))
                    .clamp(1, 30) as usize;
            }
        }

        // ---- wave9: this run's BUDGET -------------------------------------
        // Decided here, AFTER the start has been diversified (so the screen can
        // look at the real starting partition) and BEFORE the run body, which
        // only ever reads the two percentages.  With the defaults
        // (100 / 100 / off) `run_main_pct == 100` and `run_tail_pct == 100` for
        // every run, and `pct_of(x, 100, _) == x`, so every budget below is the
        // base solver's.
        let mut run_main_pct: i64 = if bok_run == 0 {
            run0_refinement_pct
        } else {
            run_refinement_pct
        };
        let run_tail_pct: i64 = if bok_run == 0 {
            run0_ils_pct
        } else {
            run_ils_pct
        };
        if run_screen_on {
            // `partition_host_refine` currently holds this run's STARTING
            // partition (the construction result for run 0, the ruined /
            // re-constructed one for k >= 1).  Scoring it is host-only: one
            // pass over the pins, ~1e5 pin visits, < 1 ms, no launch, no
            // transfer.  `bok_seen` is left all-false by `eval_km1`.
            let start_km1 = eval_km1(
                &partition_host_refine,
                &hedge_offsets_host,
                &hyperedge_nodes_host,
                challenge.num_hyperedges as usize,
                num_parts_usize,
                &mut bok_seen,
            );
            if bok_run == 0 {
                bok_start_km1_run0 = start_km1;
            } else if start_km1 > bok_start_km1_run0 {
                // This run starts from a WORSE partition than run 0 did, so it
                // is the arm more likely to be discarded: give it the screen's
                // reduced budget.  Strictly better-or-equal starts keep the
                // full `run_refinement_pct`.  NOTE the degenerate case is safe:
                // if the sign of this comparison never flips, the screen is
                // exactly the blind `run_screen_pct` schedule.
                run_main_pct = run_screen_pct;
            }
        }

        // ---- wave10: this run's SHAPE / TAIL-SPLIT / STOP selections -------
        // Same `bok_run == 0 ? run0_* : run_*` shape as wave9's two budgets, so
        // every one of them is a no-op at its default and none of them is even
        // reachable from run 0 unless its `run0_*` twin was set explicitly.
        let run_anneal_pct: i64 = if bok_run == 0 {
            run0_sched_anneal_pct
        } else {
            run_sched_anneal_pct
        };
        let run_exp: i64 = if bok_run == 0 {
            run0_sched_exp
        } else {
            run_sched_exp
        };
        let run_cold_mode: i64 = if bok_run == 0 {
            run0_sched_cold_mode
        } else {
            run_sched_cold_mode
        };
        let run_iters_pct: i64 = if bok_run == 0 {
            run0_ils_iters_pct
        } else {
            run_ils_iters_pct
        };
        let run_quick_pct: i64 = if bok_run == 0 {
            run0_ils_quick_pct
        } else {
            run_ils_quick_pct
        };
        let run_pol_pct: i64 = if bok_run == 0 {
            run0_polish_pct
        } else {
            run_polish_pct
        };
        let run_iters_abs: i64 = if bok_run == 0 {
            run0_ils_iters
        } else {
            run_ils_iters
        };
        let run_stop: usize = (if bok_run == 0 {
            run0_stag_stop
        } else {
            run_stag_stop
        }) as usize;

        // Device buffers that carry state ACROSS rounds are put back into the
        // exact state run 0 saw at this point (all of these were alloc_zeros
        // and untouched before the first round), so every run starts from the
        // same device state and run 0 is unaffected.
        stream.memcpy_htod(&partition_host_refine, &mut d_partition)?;
        stream.memcpy_htod(&nodes_in_part_host, &mut d_nodes_in_part)?;
        // Critical: stale tabu stamps (up to refinement_rounds + tenure) would
        // make every node tabu for the whole of the next run.
        stream.memset_zeros(&mut d_tabu_until)?;
        // wave20: freeze the backbone on the device for the whole run.  The
        // kernel's mark scatter only ever rewrites stamps of executed or
        // failed CANDIDATES, and a frozen node can only become one through
        // the aspiration leak (gain >= 3/4 of the round's max gain), which
        // the host filter below narrows to strictly improving moves.
        if bb_active {
            bb_tabu_host.clear();
            bb_tabu_host.resize(bb_n, 0i32);
            for i in 0..bb_n {
                if bb_frozen[i] {
                    bb_tabu_host[i] = i32::MAX;
                }
            }
            stream.memcpy_htod(&bb_tabu_host, &mut d_tabu_until)?;
        }
        // Barrier counter + its host-side mirror. Resetting BOTH keeps the
        // absolute barrier targets of run k identical to run 0's and removes
        // any risk of the u32 counter wrapping over many runs.
        stream.memset_zeros(&mut d_grid_barrier)?;
        fused_calls = 0;
        // Belt-and-braces: none of these is read before it is written in a
        // round (the fused kernel rewrites every entry of each), but zeroing
        // them makes a run's device state independent of the previous run.
        stream.memset_zeros(&mut d_mark_nodes)?;
        stream.memset_zeros(&mut d_blk_elig)?;
        stream.memset_zeros(&mut d_tile_valid)?;
        stream.memset_zeros(&mut d_cand)?;
        stream.memset_zeros(&mut d_edge_flags_all)?;
        stream.memset_zeros(&mut d_edge_flags_double)?;
        stream.memset_zeros(&mut d_edge_flags_pair)?;
        stream.memset_zeros(&mut d_move_priorities)?;
        // Host-side loop state that outlives a round.
        cand_cap = challenge.num_nodes as usize;
        mark_buf.clear();

    // ==== BEGIN per-run body: the base solver, verbatim except that every
    // ==== trajectory seed below carries `.wrapping_add(run_off)`.
    // wave8: shadow the outer `tabu_tenure` with the per-run value.  With
    // `run_tenure_delta == 0` (the default) and on run 0 always,
    // `run_tenure == tabu_tenure`, so this rebinding is a pure no-op.
    let tabu_tenure: usize = run_tenure;
    // wave9: per-run budgets, as SHADOWS of the outer bindings.  At 100% every
    // one of these is the identity (`pct_of(x, 100, _) == x`), so run 0 with
    // the default HPs executes the base solver's numbers exactly.
    //
    // `run_rounds`  = how many main-loop rounds this run actually executes.
    // `sched_rounds` = the R that the ANNEAL is written against:
    //     mode 0 (SCALE, default)  sched_rounds == run_rounds, so the whole
    //         hot -> cold trajectory is replayed on the shorter budget and the
    //         run still ends COLD (acceptance -> 0), which is what the ILS
    //         needs: the ILS takes the main loop's final partition as its
    //         incumbent and re-perturbs THAT on every iteration.
    //     mode 1 (TRUNCATE)  sched_rounds == refinement_rounds, so the run
    //         executes the first `run_rounds` rounds of the ORIGINAL schedule
    //         and stops while the lottery is still accepting worsening moves
    //         at (1 - run_rounds/R)^2 of its initial rate.
    // Both are pure integer functions of the HPs => both are deterministic.
    let run_rounds: usize = pct_of(refinement_rounds, run_main_pct, 50);
    // wave24: run 0 absorbs the rounds of the skipped explore draws.  `pct_of`
    // can only shrink a budget, so the bonus is applied as a plain percentage
    // on top of run 0's own `run_main_pct`; every other run keeps `pct_of`.
    // wave26: the last run absorbs the rounds of the skipped plain backbone
    // runs the same way (only when it IS the block backbone run, so a
    // fallen-back explore draw never inherits them).
    let run_bonus_pct: i64 = if bok_run == 0 {
        run0_bonus_pct
    } else if bb_active && bok_run.saturating_add(1) == bok_runs {
        bb_last_bonus_pct
    } else {
        0
    };
    let run_rounds: usize = if run_bonus_pct > 0 {
        let p: u64 = (run_main_pct.max(0) as u64).saturating_add(run_bonus_pct as u64);
        let v: u64 = (refinement_rounds as u64)
            .saturating_mul(p)
            .saturating_add(50)
            / 100;
        std::cmp::max(run_rounds, v as usize)
    } else {
        run_rounds
    };
    let sched_rounds: usize = if run_refine_mode == 1 {
        refinement_rounds
    } else {
        run_rounds
    };
    // The ILS / polish / post-balance budgets.  `ils_iterations` also sizes the
    // elite pool (`pool_size`), so shrinking it shrinks the pool coherently.
    // wave10 splits the single wave9 knob into three, applied ON TOP of
    // `run_tail_pct`; `pct_of(x, 100, lo) == x`, so with any subset of them at
    // their defaults the remaining ones behave exactly as in wave9.
    let ils_iterations: usize = pct_of(pct_of(ils_iterations, run_tail_pct, 1), run_iters_pct, 1);
    // wave10: absolute override, so the ILS restart count can be raised as well
    // as lowered (a percentage may only shrink a budget).  0 = off = wave9.
    let ils_iterations: usize = if run_iters_abs > 0 {
        run_iters_abs as usize
    } else {
        ils_iterations
    };
    let ils_quick_refine: usize = pct_of(pct_of(ils_quick_refine, run_tail_pct, 1), run_quick_pct, 1);
    let post_ils_polish: usize = pct_of(pct_of(post_ils_polish, run_tail_pct, 1), run_pol_pct, 1);
    // wave10: the anneal window.  `pct_of(x, 100, 1) == x`, so at the default
    // `anneal_rounds == sched_rounds` and `sched_custom` is false, which selects
    // the champion's verbatim schedule expressions below.
    let anneal_rounds: usize = pct_of(sched_rounds, run_anneal_pct, 1);
    let sched_custom: bool = run_anneal_pct != 100 || run_exp != 2;
    // The mini-perturbation's "don't perturb near the end" guard.  Default
    // (`run_cold_mode == 0`) reads `sched_rounds`, i.e. the champion's line.
    let perturb_guard_end: usize = if run_cold_mode == 1 {
        anneal_rounds
    } else {
        sched_rounds
    };
    // wave10 probe/stop bookkeeping.  `zero_streak` is the ONLY unconditional
    // per-round addition in this wave (one compare + one add per round out of a
    // ~218 us round); the rest is written once per run or gated on `probe_on`.
    let mut zero_streak: usize = 0;
    let mut split_launch_ns: u64 = 0;
    let mut split_d2h_ns: u64 = 0;
    let mut split_rest_ns: u64 = 0;
    let mut split_rounds: u64 = 0;
    let mut probe_zero_rounds: usize = 0;
    let mut probe_max_zero: usize = 0;
    let mut probe_break_round: usize = usize::MAX;
    let probe_step: usize = if probe_every > 0 {
        probe_every as usize
    } else {
        std::cmp::max(1, run_rounds / 100)
    };
    if probe_on {
        let _ = probe_lap!(); // start this run's stopwatch
        let k = probe_km1!(&partition_host_refine);
        probe_out!(
            "P11 n={:08x} run={} phase=start km1={} rounds={} sched={} anneal={} exp={} cold={} ils={} quick={} polish={} main_pct={} tail_pct={} stop={}\n",
            probe_nonce, bok_run, k, run_rounds, sched_rounds, anneal_rounds, run_exp,
            run_cold_mode, ils_iterations, ils_quick_refine, post_ils_polish,
            run_main_pct, run_tail_pct, run_stop
        );
    }
    let mut stagnant_rounds = 0usize;
    let max_stagnant_rounds = 30usize;

    let mut tgt_used: Vec<usize> = vec![0; num_parts_usize];
    let mut tgt_quota: Vec<usize> = vec![0; num_parts_usize];

    let mut total_moves_executed = 0usize;
    let mut total_perturbations = 0usize;

    // Pending device tabu marks produced at the END of the previous round; the next
    // round's kernel scatters them into d_tabu_until before it reads them.
    let mut pend_n_a: i32 = 0;
    let mut pend_until_a: i32 = 0;
    let mut pend_n_b: i32 = 0;
    let mut pend_until_b: i32 = 0;
    // refinement_rounds <= 50_000 => R*R <= 2.5e9 and R*R*(1+5*3) <= 4e10, so the host's
    // saturating_mul in the lottery denominator can never saturate and the device may
    // use a plain multiply.
    // wave9: `sched_rounds <= refinement_rounds <= 50_000`, so the overflow
    // argument above is unchanged.
    // wave10: with `sched_custom == false` (every HP at its default) this is
    // LITERALLY the champion's expression; the general branch is only ever
    // taken when a shape HP was set.  Overflow: A <= 50,000 and e <= 3, so
    // A^e <= 1.25e14 and the kernel's `den * (1 + 5*penalty)` <= 2.0e15.
    let prob_den_base: u64 = if sched_custom {
        let a = anneal_rounds as u64;
        let mut d = 1u64;
        for _ in 0..run_exp {
            d = d.saturating_mul(a);
        }
        d
    } else {
        (sched_rounds as u64) * (sched_rounds as u64)
    };
    // wave21: the backbone run anneals WARM (see `bb_heat_div`).  `prob_num`
    // below is unchanged, so the acceptance rate of every round is the hot
    // schedule's divided by `bb_heat_div`.  Overflow: base <= 1.25e14 and
    // bb_heat_div <= 64, so the kernel's `den * (1 + 5*3)` <= 1.3e17 < 2^63.
    // Every non-backbone run multiplies by 1 (identity), so run 0 and the
    // explore runs are untouched.
    // wave35: the reheat segments open at the run-1 reheat acceptance (the
    // ruin is the perturbation), the frozen block run keeps `bb_heat_div`.
    let bb_heat_eff: u64 = if bbr_on { bb_reheat_heat_div } else { bb_heat_div };
    let prob_den_base: u64 = if bb_active {
        prob_den_base.saturating_mul(bb_heat_eff)
    } else if run1_reheat_on {
        // wave33: run 1 anneals WARM from the incumbent.  Same arithmetic
        // bound as the backbone divisor (base <= 1.25e14, divisor <= 64).
        prob_den_base.saturating_mul(run1_heat_div)
    } else {
        prob_den_base
    };
    // wave28: sequential disjoint blocks on the block backbone run.  Only the
    // champion schedule shape is segmented (`sched_custom` false and SCALE
    // mode, so `sched_rounds == run_rounds`); segment s covers rounds
    // [s*len, (s+1)*len), the last one runs to `run_rounds`, and each segment
    // replays the warm (1/bb_heat_div) -> cold quadratic schedule over its own
    // length.  Every other run has `bbs_nseg == 1` and reads the unchanged
    // numbers (`prob_den_round == prob_den_base`, `rounds_left` as before).
    let bbs_on: bool = bb_active
        && bbs_blocks.len() >= 2
        && !sched_custom
        && sched_rounds == run_rounds
        && run_rounds >= bbs_blocks.len().saturating_mul(2);
    let bbs_nseg: usize = if bbs_on { bbs_blocks.len() } else { 1 };
    let bbs_seg_len: usize = std::cmp::max(1, run_rounds / bbs_nseg);
    let mut bbs_seg: usize = 0;
    for round in 0..run_rounds {
        let barrier_base: u32 = fused_calls.wrapping_mul(fused_grid);
        fused_calls = fused_calls.wrapping_add(fused_barriers_per_call); // wave6: 2 + 2*nrounds
        let round_i32: i32 = round as i32;
        // wave28: block switch.  Re-derive the frozen set from the CURRENT
        // partition (core free, new block's parts free, rest frozen), restage
        // the device stamps (free nodes start untabu, like a run start) and
        // drop the pending marks so a finite stamp cannot overwrite a fresh
        // freeze.  Both partition copies are already in sync here (the host
        // applies every move and uploads it).
        if bbs_on && bbs_seg.saturating_add(1) < bbs_nseg && round == bbs_seg.saturating_add(1) * bbs_seg_len {
            bbs_seg += 1;
            let grp: &Vec<bool> = &bbs_blocks[bbs_seg];
            // wave29: the closed-block mask follows the segment's block.
            if bb_closed_on {
                for p in 0..num_parts_usize {
                    bb_blk_mask[p] = grp.get(p).copied().unwrap_or(false);
                }
                bb_blk_parts.clear();
                for p in 0..num_parts_usize {
                    if bb_blk_mask[p] {
                        bb_blk_parts.push(p as i32);
                    }
                }
            }
            if bbr_on {
                // wave35: no refreeze -- ruin and recreate block s of the
                // CURRENT partition (closed onto its parts) and upload it;
                // the segment's warm anneal then runs on the whole graph.
                let cap_r: usize = (ruin_cap_base
                    .saturating_mul(bb_reheat_cap_pct)
                    / 100)
                    .min(challenge.num_nodes as usize);
                let seed_r: u64 = (bok_run as u64)
                    .wrapping_mul(RUIN_SEED_MUL)
                    .wrapping_add(RUIN_SEED_ADD)
                    .wrapping_add((bbs_seg as u64).wrapping_mul(RUN_OFF_MUL))
                    | 1;
                let blk_s: Vec<bool> = grp.clone();
                let changed: bool = ruin_recreate(
                    &mut partition_host_refine,
                    &mut nodes_in_part_host,
                    &hedge_offsets_host,
                    &hyperedge_nodes_host,
                    &node_offsets_host,
                    &node_hedges_host,
                    challenge.num_hyperedges as usize,
                    num_parts_usize,
                    challenge.max_part_size as i32,
                    usize::MAX,
                    cap_r,
                    seed_r,
                    CRIT_COUPLED_BLOCK,
                    seed_r,
                    &blk_s,
                );
                if changed {
                    stream.memcpy_htod(&partition_host_refine, &mut d_partition)?;
                    stream.memcpy_htod(&nodes_in_part_host, &mut d_nodes_in_part)?;
                }
                bb_free_nodes.clear();
                for v in 0..bb_n {
                    bb_frozen[v] = false;
                    bb_free_nodes.push(v);
                }
            } else {
                bb_free_nodes.clear();
                for v in 0..bb_n {
                    let p = partition_host_refine[v];
                    let in_blk: bool = p >= 0 && (p as usize) < num_parts_usize && grp[p as usize];
                    let frozen: bool = bb_stable[v] && !in_blk;
                    bb_frozen[v] = frozen;
                    if !frozen {
                        bb_free_nodes.push(v);
                    }
                }
            }
            bb_tabu_host.clear();
            bb_tabu_host.resize(bb_n, 0i32);
            for i in 0..bb_n {
                if bb_frozen[i] {
                    bb_tabu_host[i] = i32::MAX;
                }
            }
            stream.memcpy_htod(&bb_tabu_host, &mut d_tabu_until)?;
            mark_buf.clear();
            pend_n_a = 0;
            pend_until_a = 0;
            pend_n_b = 0;
            pend_until_b = 0;
            stagnant_rounds = 0;
        }
        let bbs_seg_end: usize = if bbs_seg.saturating_add(1) >= bbs_nseg {
            run_rounds
        } else {
            bbs_seg.saturating_add(1) * bbs_seg_len
        };
        let bbs_seg_start: usize = bbs_seg * bbs_seg_len;
        let bbs_cur_len: u64 = bbs_seg_end.saturating_sub(bbs_seg_start).max(1) as u64;
        let prob_den_round: u64 = if bbs_on {
            bbs_cur_len
                .saturating_mul(bbs_cur_len)
                .saturating_mul(bb_heat_eff)
        } else {
            prob_den_base
        };
        // seed site 1/6: device zero/negative-gain acceptance lottery.
        let lcg_x0: u64 = 123456789u64
            .wrapping_add(round as u64)
            .wrapping_add(run_off);
        // wave10: same two numbers as the champion whenever `sched_custom` is
        // false.  In the general branch `rounds_left` hits 0 at
        // `anneal_rounds`, so every round after that has `prob_num == 0` and
        // the kernel's `(x % den) < prob_num` can never fire: the tail of the
        // loop is a pure zero-temperature quench.
        let rounds_left: u64 = if bbs_on {
            bbs_seg_end.saturating_sub(round) as u64
        } else if sched_custom {
            anneal_rounds.saturating_sub(round) as u64
        } else {
            sched_rounds.saturating_sub(round) as u64
        };
        let prob_num: u64 = if sched_custom {
            let mut n = 1u64;
            for _ in 0..run_exp {
                n = n.saturating_mul(rounds_left);
            }
            n
        } else {
            rounds_left * rounds_left
        };
        let mut sp = if probe_split {
            Some(std::time::Instant::now())
        } else {
            None
        };
        unsafe {
            stream
                .launch_builder(&fused_filter_kernel)
                .arg(&(challenge.num_nodes as i32))
                .arg(&(challenge.num_parts as i32))
                .arg(&(challenge.max_part_size as i32))
                .arg(&challenge.d_hyperedge_nodes)
                .arg(&challenge.d_node_hyperedges)
                .arg(&d_partition)
                .arg(&d_nodes_in_part)
                .arg(&d_fe_slots)
                .arg(&n_fe_tasks)
                .arg(&d_fe_giant)
                .arg(&n_fe_giant)
                .arg(&d_mv_slots)
                .arg(&n_mv_tasks)
                .arg(&mut d_edge_flags_pair)
                .arg(&mut d_move_priorities)
                .arg(&mut d_grid_barrier)
                .arg(&barrier_base)
                .arg(&mut d_tabu_until)
                .arg(&d_mark_nodes)
                .arg(&pend_n_a)
                .arg(&pend_until_a)
                .arg(&pend_n_b)
                .arg(&pend_until_b)
                .arg(&round_i32)
                .arg(&lcg_x0)
                .arg(&prob_num)
                .arg(&prob_den_round)
                .arg(&mut d_max_gain)
                .arg(&mut d_blk_elig)
                .arg(&mut d_tile_valid)
                .arg(&ntiles_i32)
                .arg(&mut d_cand)
                .launch(fused_cfg.clone())?;
        }

        if let Some(t) = sp {
            split_launch_ns += t.elapsed().as_nanos() as u64;
            sp = Some(std::time::Instant::now());
        }
        // The kernel already compacted the candidates globally and in node order,
        // so we only need [count][2*count] i32 -- ~26 KB instead of the 148 KB
        // envelope. The copied length is a prediction from the previous round;
        // if it was too small we re-copy with the exact length (rare, and the
        // device data is unchanged, so the result is identical either way).
        {
            let want = std::cmp::min(cand_len, 1 + 2 * cand_cap);
            stream.memcpy_dtoh(&d_cand.slice(0..want), &mut cand_host[..want])?;
        }
        let mut n_cand = std::cmp::min(
            std::cmp::max(cand_host[0], 0) as usize,
            challenge.num_nodes as usize,
        );
        if n_cand > cand_cap {
            let want2 = 1 + 2 * n_cand;
            stream.memcpy_dtoh(&d_cand.slice(0..want2), &mut cand_host[..want2])?;
            n_cand = std::cmp::min(
                std::cmp::max(cand_host[0], 0) as usize,
                challenge.num_nodes as usize,
            );
        }
        // margin over the previous round's count; a miss only costs one extra
        // (correct) copy, so keep it tight.
        cand_cap = std::cmp::min(
            challenge.num_nodes as usize,
            n_cand + (n_cand >> 3) + 256,
        );
        if let Some(t) = sp {
            split_d2h_ns += t.elapsed().as_nanos() as u64;
            sp = Some(std::time::Instant::now());
        }

        mark_buf.clear();
        pend_n_a = 0;
        pend_until_a = 0;
        pend_n_b = 0;
        pend_until_b = 0;

        // ---- wave4 kwhostsel: build packed keys straight out of the D2H buffer ----
        // cand[1..] is already in ascending node order, exactly the order the
        // serial host filter produced; that order only ever mattered as the
        // tie-break of `cmp`, and the low 32 bits of the packed key now carry it
        // explicitly. The per-target-part histogram is folded into this pass: it is
        // the only pass that touches the freshly DMA'd (cold) `cand_host`.
        vm_keys.clear();
        let mut part_cnt = [0usize; 64];
        if bb_active {
            // wave20: a frozen (backbone) node only passes with a strictly
            // improving candidate; the disputed nodes pass unchanged.
            let seg = &cand_host[1..1 + 2 * n_cand];
            for w in seg.chunks_exact(2) {
                let node_u = w[0] as usize;
                let gain_le0: bool = (w[1] >> 16) <= 0;
                if node_u < bb_n && bb_frozen[node_u] && gain_le0 {
                    continue;
                }
                // wave29: closed block.  A vote-stable node sitting in a
                // block part is free only because of the block; its drift
                // (gain <= 0: zero-gain and lottery moves, the kernel zeroes
                // the gain field of an accepted lottery move) may target
                // block parts only.  The target part is the key's low 6 bits.
                if bb_closed_on && gain_le0 && node_u < bb_n {
                    let src = partition_host_refine[node_u];
                    if src >= 0
                        && (src as usize) < num_parts_usize
                        && bb_blk_mask[src as usize]
                        && bb_stable.get(node_u).copied().unwrap_or(false)
                    {
                        let tgt: usize = (w[1] & 63) as usize;
                        if tgt >= num_parts_usize || !bb_blk_mask[tgt] {
                            continue;
                        }
                    }
                }
                let hi = (w[1] as u32) ^ 0x7FFF_FFFFu32;
                part_cnt[((hi as usize) & 63) ^ 63] += 1;
                vm_keys.push(((hi as u64) << 32) | (w[0] as u32 as u64));
            }
        } else {
            let seg = &cand_host[1..1 + 2 * n_cand];
            for w in seg.chunks_exact(2) {
                let hi = (w[1] as u32) ^ 0x7FFF_FFFFu32;
                part_cnt[((hi as usize) & 63) ^ 63] += 1;
                vm_keys.push(((hi as u64) << 32) | (w[0] as u32 as u64));
            }
        }

        if vm_keys.is_empty() {
            break;
        }

        let mut k_base = vm_keys.len();
        let adaptive_limit = if round < 50 {
            move_limit / 2
        } else if round < 200 {
            (move_limit * 3) / 4
        } else {
            move_limit / 4
        };
        if k_base > adaptive_limit {
            k_base = adaptive_limit;
        }

        let k_cand = std::cmp::min(vm_keys.len(), k_base.saturating_add(extra_window));

        let bucket_path = k_base >= vm_keys.len() && k_cand >= vm_keys.len();
        if !bucket_path {
            // Same two statements as before, on packed keys: ascending Ord on u64 IS
            // `cmp`, so no comparator closure is needed.
            if k_cand > 1 {
                vm_keys.select_nth_unstable(k_cand - 1);
            }
            vm_keys[..k_cand].sort_unstable();
        }

        let slack = if round < 64 {
            slack_early
        } else if round < 256 {
            slack_mid
        } else {
            slack_late
        };

        tgt_used.fill(0);
        for p in 0..num_parts_usize {
            let free = (challenge.max_part_size as i32 - nodes_in_part_host[p]).max(0) as usize;
            tgt_quota[p] = std::cmp::max(1, free.saturating_add(slack));
        }

        sorted_move_nodes.clear();
        sorted_move_parts.clear();
        // The kernel packs the target part into 6 bits, so a candidate's target is
        // always < 64; parts >= 64 (if num_parts > 64) can never own a candidate.
        let np64 = std::cmp::min(num_parts_usize, 64usize);
        if bucket_path {
            // (1) counting sort by target part into ONE flat buffer (replaces 64
            //     separate Vec pushes over ~1 MB of bucket storage),
            // (2) per-part bounded selection: the first tgt_quota[p] candidates of
            //     part p in `cmp` order == the quota_p smallest packed keys,
            // (3) one global sort of the ~600 survivors.
            // Step (2) keeps exactly the same SET as the old
            // select_nth_unstable_by(q-1)+truncate(q); step (3) puts that set in
            // `cmp` order, so the emitted sequence is identical.
            let mut part_off = [0usize; 64];
            let mut acc = 0usize;
            for p in 0..64 {
                part_off[p] = acc;
                acc += part_cnt[p];
            }
            let mut cur = part_off;
            for &v in vm_keys.iter() {
                let tgt = (((v >> 32) as usize) & 63) ^ 63;
                let idx = cur[tgt];
                cur[tgt] = idx + 1;
                by_buf[idx] = v;
            }
            let mut m = 0usize;
            for p in 0..np64 {
                let c = part_cnt[p];
                if c == 0 {
                    continue;
                }
                let s = part_off[p];
                let q = tgt_quota[p];
                let keep = if c > q {
                    by_buf[s..s + c].select_nth_unstable(q - 1);
                    q
                } else {
                    c
                };
                sel_keys[m..m + keep].copy_from_slice(&by_buf[s..s + keep]);
                m += keep;
            }
            sel_keys[..m].sort_unstable();
            for &v in sel_keys[..m].iter() {
                sorted_move_nodes.push(v as u32 as i32);
                sorted_move_parts.push(((((v >> 32) as usize) & 63) ^ 63) as i32);
            }
        } else {
            for &v in vm_keys[..k_cand].iter() {
                if sorted_move_nodes.len() >= k_base {
                    break;
                }
                let tgt = (((v >> 32) as usize) & 63) ^ 63;
                if tgt < num_parts_usize && tgt_used[tgt] < tgt_quota[tgt] {
                    tgt_used[tgt] += 1;
                    sorted_move_nodes.push(v as u32 as i32);
                    sorted_move_parts.push(tgt as i32);
                }
            }
        }

        if sorted_move_nodes.is_empty() {
            if bucket_path {
                vm_keys.sort_unstable();
            }
            let take = std::cmp::min(k_base, k_cand);
            for &v in vm_keys[..take].iter() {
                sorted_move_nodes.push(v as u32 as i32);
                sorted_move_parts.push(((((v >> 32) as usize) & 63) ^ 63) as i32);
            }
        }

        let mut moves_executed = simulate_execute_moves(
            &mut partition_host_refine,
            &mut nodes_in_part_host,
            &sorted_move_nodes,
            &sorted_move_parts,
        );

        if moves_executed > 0 {
            stream.memcpy_htod(&partition_host_refine, &mut d_partition)?;
            stream.memcpy_htod(&nodes_in_part_host, &mut d_nodes_in_part)?;
        }

        if moves_executed == 0 && k_cand > k_base {
            let fail_mark_len = std::cmp::min(sorted_move_nodes.len(), tabu_fail_mark_len);
            mark_buf.extend_from_slice(&sorted_move_nodes[..fail_mark_len]);
            pend_n_a = fail_mark_len as i32;
            pend_until_a = (round + tabu_fail_tenure) as i32;

            sorted_move_nodes.clear();
            sorted_move_parts.clear();
            let tail = &vm_keys[k_base..k_cand];
            let take = std::cmp::min(tail.len(), k_base);
            for &v in tail.iter().take(take) {
                sorted_move_nodes.push(v as u32 as i32);
                sorted_move_parts.push(((((v >> 32) as usize) & 63) ^ 63) as i32);
            }

            if !sorted_move_nodes.is_empty() {
                moves_executed = simulate_execute_moves(
                    &mut partition_host_refine,
                    &mut nodes_in_part_host,
                    &sorted_move_nodes,
                    &sorted_move_parts,
                );
                if moves_executed > 0 {
                    stream.memcpy_htod(&partition_host_refine, &mut d_partition)?;
                    stream.memcpy_htod(&nodes_in_part_host, &mut d_nodes_in_part)?;
                }
            }
        }

        total_moves_executed += moves_executed as usize;

        if moves_executed > 0 {
            let mark_len = std::cmp::min(
                sorted_move_nodes.len(),
                std::cmp::max(
                    tabu_mark_base,
                    (moves_executed as usize).saturating_mul(tabu_mark_mult),
                ),
            );
            let until = round + tabu_tenure;
            mark_buf.extend_from_slice(&sorted_move_nodes[..mark_len]);
            pend_n_b = mark_len as i32;
            pend_until_b = until as i32;
        }

        if !mark_buf.is_empty() {
            stream.memcpy_htod(&mark_buf[..], &mut d_mark_nodes)?;
        }

        if let Some(t) = sp {
            split_rest_ns += t.elapsed().as_nanos() as u64;
            split_rounds += 1;
        }
        // wave10: `zero_streak` counts CONSECUTIVE zero-move rounds and, unlike
        // `stagnant_rounds`, is NOT reset by the mini-perturbation.  It drives
        // the optional `run_stag_stop` and the probe's fire-rate statistics.
        if moves_executed == 0 {
            zero_streak += 1;
            probe_zero_rounds += 1;
            if zero_streak > probe_max_zero {
                probe_max_zero = zero_streak;
            }
        } else {
            zero_streak = 0;
        }

        if moves_executed == 0 {
            stagnant_rounds += 1;
            // wave10: `perturb_guard_end == sched_rounds` unless
            // `run_sched_cold_mode == 1`, in which case the mini-perturbation
            // is disarmed for the whole quench so the quench is monotone and
            // the break below becomes reachable.
            if stagnant_rounds >= 3 && round < perturb_guard_end.saturating_sub(50) {
                // seed site 2/6: stagnation mini-perturbation.
                let mini_seed =
                    (987654321u64 + (round as u64) * 123456789u64).wrapping_add(run_off);
                if bb_active && bb_closed_on && !bb_blk_parts.is_empty() {
                    // wave29: in-block kick targets for block-freed nodes.
                    perturb_block_on_host(
                        &mut partition_host_refine,
                        &mut nodes_in_part_host,
                        &bb_free_nodes,
                        &bb_stable,
                        &bb_blk_mask,
                        &bb_blk_parts,
                        bb_perturb_pct as i32,
                        mini_seed,
                    );
                } else if bb_active {
                    // wave20: the kick must not touch the backbone (the
                    // device would never move a kicked frozen node back).
                    perturb_free_on_host(
                        &mut partition_host_refine,
                        &mut nodes_in_part_host,
                        &bb_free_nodes,
                        bb_perturb_pct as i32,
                        mini_seed,
                    );
                } else {
                    perturb_on_host(
                        &mut partition_host_refine,
                        &mut nodes_in_part_host,
                        3i32,
                        mini_seed,
                    );
                }
                stream.memcpy_htod(&partition_host_refine, &mut d_partition)?;
                stream.memcpy_htod(&nodes_in_part_host, &mut d_nodes_in_part)?;
                total_perturbations += 1;
                stagnant_rounds = 0;
            } else if stagnant_rounds > max_stagnant_rounds {
                probe_break_round = round;
                break;
            }
        } else {
            stagnant_rounds = 0;
        }

        // wave10: optional early stop.  `run_stop == 0` by default, so this is
        // a dead branch (one integer compare per round) unless it is asked for.
        if run_stop > 0 && zero_streak >= run_stop {
            probe_break_round = round;
            break;
        }

        if probe_trace && (round % probe_step == 0 || round + 1 == run_rounds) {
            let k = probe_km1!(&partition_host_refine);
            probe_out!(
                "P11 n={:08x} run={} trace round={} km1={} moves={} cum_moves={} perturbs={} ncand={} zero={} prob_num={} prob_den={}\n",
                probe_nonce, bok_run, round, k, moves_executed, total_moves_executed,
                total_perturbations, n_cand, probe_zero_rounds, prob_num, prob_den_round
            );
        }
    }

    if probe_on {
        let us = probe_lap!();
        let k = probe_km1!(&partition_host_refine);
        let rounds_exec = if probe_break_round == usize::MAX {
            run_rounds
        } else {
            probe_break_round + 1
        };
        probe_out!(
            "P11 t={} n={:08x} run={} phase=main_end km1={} us={} rounds_exec={} rounds_budget={} moves={} perturbs={} zero_rounds={} max_zero_streak={} broke={} split_rounds={} launch_ns={} d2h_ns={} rest_ns={} fused_mode={} fused_block={} fused_grid={} ntiles={} barriers={}\n",
            tp.name, probe_nonce, bok_run, k, us, rounds_exec, run_rounds,
            total_moves_executed, total_perturbations, probe_zero_rounds,
            probe_max_zero, (probe_break_round != usize::MAX) as i32,
            split_rounds, split_launch_ns, split_d2h_ns, split_rest_ns,
            fused_mode, fused_block, fused_grid, ntiles, fused_barriers_per_call
        );
    }

    macro_rules! do_swap_phase {
        ($d_partition:expr, $d_nodes_in_part:expr,
         $d_edge_flags_all:expr, $d_edge_flags_double:expr,
         $d_swap_gains:expr, $swap_gains_host:expr,
         $partition_host_swap:expr, $partition_mut_swap:expr,
         $part_to_part:expr, $used_ba_buf:expr,
         $max_rounds:expr, $ngt:expr, $scan_lim:expr, $scan_lim_cyc:expr) => {{
            let num_nodes_i = challenge.num_nodes as i32;
            let num_parts_i = challenge.num_parts as i32;
            let np = num_parts_usize;
            let mut prev_swap_count = usize::MAX;
            let mut stagnant = 0usize;
            let mut total_swaps = 0usize;
            stream.memcpy_dtoh(&mut *$d_partition, $partition_host_swap)?;
            for _swap_round in 0..$max_rounds {
                unsafe {
                    stream
                        .launch_builder(&precompute_edge_flags_kernel)
                        .arg(&(challenge.num_hyperedges as i32))
                        .arg(&num_nodes_i)
                        .arg(&challenge.d_hyperedge_nodes)
                        .arg(&challenge.d_hyperedge_offsets)
                        .arg(&mut *$d_partition)
                        .arg(&mut *$d_edge_flags_all)
                        .arg(&mut *$d_edge_flags_double)
                        .launch(hedge_cfg.clone())?;
                }
                unsafe {
                    stream
                        .launch_builder(&compute_swap_gains_kernel)
                        .arg(&num_nodes_i)
                        .arg(&num_parts_i)
                        .arg(&$ngt)
                        .arg(&challenge.d_node_hyperedges)
                        .arg(&challenge.d_node_offsets)
                        .arg(&mut *$d_partition)
                        .arg(&mut *$d_edge_flags_all)
                        .arg(&mut *$d_edge_flags_double)
                        .arg(&mut *$d_swap_gains)
                        .launch(cfg.clone())?;
                }
                stream.memcpy_dtoh(&mut *$d_swap_gains, $swap_gains_host)?;
                let num_nodes = num_nodes_i as usize;

                for v in $part_to_part.iter_mut() { v.clear(); }
                for node in 0..num_nodes {
                    let src = $partition_host_swap[node] as usize;
                    if src >= np { continue; }
                    for k in 0..3usize {
                        let val = $swap_gains_host[node * 3 + k];
                        if val == 0 { continue; }
                        let tgt = (val & 0xFFFF) as usize;
                        let gain = ((val >> 16) as i16) as i32;
                        if tgt < np && tgt != src {
                            $part_to_part[src * np + tgt].push((node, gain));
                        }
                    }
                }

                $partition_mut_swap.copy_from_slice($partition_host_swap);
                let mut swap_count = 0usize;

                for a in 0..np {
                    for b in (a + 1)..np {
                        let idx_ab = a * np + b;
                        let idx_ba = b * np + a;
                        if $part_to_part[idx_ab].is_empty() || $part_to_part[idx_ba].is_empty() { continue; }
                        $part_to_part[idx_ab].sort_unstable_by(|x, y| y.1.cmp(&x.1));
                        $part_to_part[idx_ba].sort_unstable_by(|x, y| y.1.cmp(&x.1));
                        let lab_len = $part_to_part[idx_ab].len();
                        let lba_len = $part_to_part[idx_ba].len();
                        $used_ba_buf.clear();
                        $used_ba_buf.resize(lba_len, false);
                        for i in 0..lab_len {
                            let (node_a, gain_a) = $part_to_part[idx_ab][i];
                            if $partition_mut_swap[node_a] as usize != a { continue; }
                            let mut best_combined = 0i32;
                            let mut best_j = usize::MAX;
                            let sl = std::cmp::min(lba_len, $scan_lim);
                            for j in 0..sl {
                                if $used_ba_buf[j] { continue; }
                                let (node_b, gain_b) = $part_to_part[idx_ba][j];
                                if $partition_mut_swap[node_b] as usize != b { continue; }
                                let combined = gain_a + gain_b;
                                if combined > best_combined {
                                    best_combined = combined;
                                    best_j = j;
                                }
                            }
                            if best_j < lba_len && best_combined > 0 {
                                let (node_b, _) = $part_to_part[idx_ba][best_j];
                                $partition_mut_swap[node_a] = b as i32;
                                $partition_mut_swap[node_b] = a as i32;
                                $used_ba_buf[best_j] = true;
                                swap_count += 1;
                            }
                        }
                    }
                }

                let cyc_scan = $scan_lim_cyc;
                if cyc_scan > 0 {
                    for a in 0..np {
                        for b in 0..np {
                            if b == a { continue; }
                            let idx_ab = a * np + b;
                            if $part_to_part[idx_ab].is_empty() { continue; }
                            for c in 0..np {
                                if c == a || c == b { continue; }
                                let idx_bc = b * np + c;
                                let idx_ca = c * np + a;
                                if $part_to_part[idx_bc].is_empty() || $part_to_part[idx_ca].is_empty() { continue; }
                                let sl_ab = std::cmp::min($part_to_part[idx_ab].len(), cyc_scan);
                                let sl_bc = std::cmp::min($part_to_part[idx_bc].len(), cyc_scan);
                                let sl_ca = std::cmp::min($part_to_part[idx_ca].len(), cyc_scan);
                                'outer: for i in 0..sl_ab {
                                    let (node_ab, gain_ab) = $part_to_part[idx_ab][i];
                                    if $partition_mut_swap[node_ab] as usize != a { continue; }
                                    for j in 0..sl_bc {
                                        let (node_bc, gain_bc) = $part_to_part[idx_bc][j];
                                        if $partition_mut_swap[node_bc] as usize != b { continue; }
                                        if node_bc == node_ab { continue; }
                                        if gain_ab + gain_bc <= 0 { break; }
                                        for k in 0..sl_ca {
                                            let (node_ca, gain_ca) = $part_to_part[idx_ca][k];
                                            if $partition_mut_swap[node_ca] as usize != c { continue; }
                                            if node_ca == node_ab || node_ca == node_bc { continue; }
                                            if gain_ab + gain_bc + gain_ca > 0 {
                                                $partition_mut_swap[node_ab] = b as i32;
                                                $partition_mut_swap[node_bc] = c as i32;
                                                $partition_mut_swap[node_ca] = a as i32;
                                                swap_count += 1;
                                                break 'outer;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                if swap_count == 0 { break; }
                total_swaps += swap_count;
                if swap_count >= prev_swap_count {
                    stagnant += 1;
                    if stagnant >= 3 { break; }
                } else {
                    stagnant = 0;
                }
                prev_swap_count = swap_count;
                stream.memcpy_htod($partition_mut_swap, &mut *$d_partition)?;
                $partition_host_swap.copy_from_slice($partition_mut_swap);
            }
            anyhow::Ok(total_swaps)
        }};
    }

    do_swap_phase!(
        &mut d_partition, &mut d_nodes_in_part,
        &mut d_edge_flags_all, &mut d_edge_flags_double,
        &mut d_swap_gains, &mut swap_gains_host,
        &mut partition_host_swap, &mut partition_mut_swap,
        &mut part_to_part, &mut used_ba_buf,
        100, neg_gain_thresh, scan_limit_swap, scan_limit_cycle
    )?;
    partition_host_refine.copy_from_slice(&partition_host_swap);
    if probe_on {
        let us = probe_lap!();
        let k = probe_km1!(&partition_host_refine);
        probe_out!(
            "P11 n={:08x} run={} phase=swap1 km1={} us={}\n",
            probe_nonce, bok_run, k, us
        );
    }

    let mut probe_ps30: usize = 0;
    for _post_swap_round in 0..30 {
        probe_ps30 += 1;
        fused_calls = fused_calls.wrapping_add(1);
        let fused_target: u32 = fused_calls.wrapping_mul(fused_grid);
        unsafe {
            stream
                .launch_builder(&fused_moves_kernel)
                .arg(&(challenge.num_hyperedges as i32))
                .arg(&(challenge.num_nodes as i32))
                .arg(&(challenge.num_parts as i32))
                .arg(&(challenge.max_part_size as i32))
                .arg(&challenge.d_hyperedge_nodes)
                .arg(&challenge.d_hyperedge_offsets)
                .arg(&challenge.d_node_hyperedges)
                .arg(&challenge.d_node_offsets)
                .arg(&d_partition)
                .arg(&d_nodes_in_part)
                .arg(&mut d_edge_flags_all)
                .arg(&mut d_edge_flags_double)
                .arg(&mut d_move_priorities)
                .arg(&mut d_grid_barrier)
                .arg(&fused_target)
                .launch(fused_tail_cfg.clone())?;
        }
        stream.memcpy_dtoh(&d_move_priorities, &mut move_keys_host)?;
        valid_moves.clear();
        for (node, &key) in move_keys_host.iter().enumerate() {
            if key as u32 != 0x80000000 {
                let gain = key >> 16;
                if gain > 0 { valid_moves.push((node, key)); }
            }
        }
        if valid_moves.is_empty() { break; }
        let cmp = |a: &(usize, i32), b: &(usize, i32)| b.1.cmp(&a.1).then(a.0.cmp(&b.0));
        let k_base = std::cmp::min(valid_moves.len(), move_limit / 2);
        let k_cand = std::cmp::min(valid_moves.len(), k_base + extra_window / 2);
        if k_cand > 1 {
            valid_moves.select_nth_unstable_by(k_cand - 1, cmp);
            valid_moves[..k_cand].sort_unstable_by(cmp);
        }
        tgt_used.fill(0);
        for p in 0..num_parts_usize {
            let free = (challenge.max_part_size as i32 - nodes_in_part_host[p]).max(0) as usize;
            tgt_quota[p] = std::cmp::max(1, free + slack_mid);
        }
        sorted_move_nodes.clear();
        sorted_move_parts.clear();
        for &(node, key) in valid_moves[..k_cand].iter() {
            if sorted_move_nodes.len() >= k_base { break; }
            let tgt = (key & 63) as usize;
            if tgt < num_parts_usize && tgt_used[tgt] < tgt_quota[tgt] {
                tgt_used[tgt] += 1;
                sorted_move_nodes.push(node as i32);
                sorted_move_parts.push(tgt as i32);
            }
        }
        if sorted_move_nodes.is_empty() {
            let take = std::cmp::min(k_base, k_cand);
            sorted_move_nodes.extend(valid_moves[..take].iter().map(|(n, _)| *n as i32));
            sorted_move_parts.extend(valid_moves[..take].iter().map(|(_, key)| (key & 63) as i32));
        }
        let me = simulate_execute_moves(
            &mut partition_host_refine,
            &mut nodes_in_part_host,
            &sorted_move_nodes,
            &sorted_move_parts,
        );
        if me > 0 {
            stream.memcpy_htod(&partition_host_refine, &mut d_partition)?;
            stream.memcpy_htod(&nodes_in_part_host, &mut d_nodes_in_part)?;
        }
        if me == 0 { break; }
    }
    if probe_on {
        let us = probe_lap!();
        let k = probe_km1!(&partition_host_refine);
        probe_out!(
            "P11 n={:08x} run={} phase=postswap30 km1={} us={} iters={}\n",
            probe_nonce, bok_run, k, us, probe_ps30
        );
    }

    do_swap_phase!(
        &mut d_partition, &mut d_nodes_in_part,
        &mut d_edge_flags_all, &mut d_edge_flags_double,
        &mut d_swap_gains, &mut swap_gains_host,
        &mut partition_host_swap, &mut partition_mut_swap,
        &mut part_to_part, &mut used_ba_buf,
        50, neg_gain_thresh, scan_limit_swap, scan_limit_cycle
    )?;

    let perturb_strength = 3;

    unsafe {
        stream
            .launch_builder(&compute_connectivity_kernel)
            .arg(&(challenge.num_hyperedges as i32))
            .arg(&challenge.d_hyperedge_nodes)
            .arg(&challenge.d_hyperedge_offsets)
            .arg(&d_partition)
            .arg(&mut d_connectivity)
            .launch(hedge_cfg.clone())?;
    }
    let conn_vec = stream.memcpy_dtov(&d_connectivity)?;
    let mut best_connectivity: i32 = conn_vec.iter().sum();
    let mut best_partition_host = stream.memcpy_dtov(&d_partition)?;
    let mut best_nodes_in_part_host = stream.memcpy_dtov(&d_nodes_in_part)?;
    if probe_on {
        let us = probe_lap!();
        probe_out!(
            "P11 n={:08x} run={} phase=ils_in km1={} us={}\n",
            probe_nonce, bok_run, best_connectivity, us
        );
    }

    let num_high_hedges = std::cmp::min(500usize, challenge.num_hyperedges as usize);
    let mut high_hedge_ids_host: Vec<i32> = build_high_hedge_ids(&conn_vec, num_high_hedges);

    let pool_size = ils_iterations;
    let mut elite_scores: Vec<i32> = vec![i32::MAX; pool_size];
    let mut elite_flat_host: Vec<i32> = vec![0i32; pool_size * challenge.num_nodes as usize];
    elite_scores[0] = best_connectivity;
    elite_flat_host[..challenge.num_nodes as usize].copy_from_slice(&best_partition_host);
    let mut elite_count: usize = 1;
    let mut d_elite_flat = stream.alloc_zeros::<i32>(pool_size * challenge.num_nodes as usize)?;

    let mut use_consensus_next = false;

    for ils_iter in 0..ils_iterations {
        let d_partition_restored = stream.memcpy_stod(&best_partition_host)?;
        let d_nodes_in_part_restored = stream.memcpy_stod(&best_nodes_in_part_host)?;
        d_partition = d_partition_restored;
        d_nodes_in_part = d_nodes_in_part_restored;
        partition_host_refine.copy_from_slice(&best_partition_host);
        nodes_in_part_host.copy_from_slice(&best_nodes_in_part_host);

        unsafe {
            stream
                .launch_builder(&precompute_edge_flags_kernel)
                .arg(&(challenge.num_hyperedges as i32))
                .arg(&(challenge.num_nodes as i32))
                .arg(&challenge.d_hyperedge_nodes)
                .arg(&challenge.d_hyperedge_offsets)
                .arg(&d_partition)
                .arg(&mut d_edge_flags_all)
                .arg(&mut d_edge_flags_double)
                .launch(hedge_cfg.clone())?;
        }

        // seed site 3/6: ILS random perturbation.
        let seed =
            (123456789u64 + (ils_iter as u64) * 987654321u64).wrapping_add(run_off);

        let refreshed_host_from_device = if use_consensus_next && elite_count > 1 {
            stream.memcpy_htod(&elite_flat_host, &mut d_elite_flat)?;

            let mut elite_order_host: Vec<i32> = (0..elite_count as i32).collect();
            elite_order_host.sort_unstable_by(|&a, &b| {
                elite_scores[a as usize]
                    .cmp(&elite_scores[b as usize])
                    .then_with(|| a.cmp(&b))
            });
            let d_elite_order = stream.memcpy_stod(&elite_order_host)?;

            unsafe {
                stream
                    .launch_builder(&choose_elite_per_hyperedge_kernel)
                    .arg(&(challenge.num_hyperedges as i32))
                    .arg(&(challenge.num_nodes as i32))
                    .arg(&(challenge.num_parts as i32))
                    .arg(&(elite_count as i32))
                    .arg(&d_elite_flat)
                    .arg(&d_elite_order)
                    .arg(&challenge.d_hyperedge_offsets)
                    .arg(&challenge.d_hyperedge_nodes)
                    .arg(&mut d_hedge_choice)
                    .launch(hedge_cfg.clone())?;
            }

            stream.memset_zeros(&mut d_nodes_in_part)?;

            unsafe {
                stream
                    .launch_builder(&assign_from_elite_votes_kernel)
                    .arg(&(challenge.num_nodes as i32))
                    .arg(&(challenge.num_parts as i32))
                    .arg(&(elite_count as i32))
                    .arg(&d_elite_flat)
                    .arg(&d_hedge_choice)
                    .arg(&challenge.d_node_hyperedges)
                    .arg(&challenge.d_node_offsets)
                    .arg(&d_edge_flags_all)
                    .arg(&mut d_partition)
                    .launch(cfg.clone())?;
            }

            stream.memcpy_dtoh(&d_partition, &mut partition_host_refine)?;
            nodes_in_part_host.fill(0);
            for &p in partition_host_refine.iter() {
                if p >= 0 && (p as usize) < num_parts_usize {
                    nodes_in_part_host[p as usize] += 1;
                }
            }
            stream.memcpy_htod(&nodes_in_part_host, &mut d_nodes_in_part)?;

            unsafe {
                stream
                    .launch_builder(&balance_kernel)
                    .arg(&(challenge.num_nodes as i32))
                    .arg(&(challenge.num_parts as i32))
                    .arg(&1i32)
                    .arg(&(challenge.max_part_size as i32))
                    .arg(&mut d_partition)
                    .arg(&mut d_nodes_in_part)
                    .launch(one_thread_cfg.clone())?;
            }
            true
        } else {
            if ils_iter % 2 == 0 {
                perturb_guided_on_host(
                    &mut partition_host_refine,
                    &mut nodes_in_part_host,
                    &high_hedge_ids_host,
                );
            } else {
                perturb_on_host(
                    &mut partition_host_refine,
                    &mut nodes_in_part_host,
                    perturb_strength,
                    seed,
                );
            }
            stream.memcpy_htod(&partition_host_refine, &mut d_partition)?;
            stream.memcpy_htod(&nodes_in_part_host, &mut d_nodes_in_part)?;
            false
        };

        if refreshed_host_from_device {
            stream.memcpy_dtoh(&d_partition, &mut partition_host_refine)?;
            stream.memcpy_dtoh(&d_nodes_in_part, &mut nodes_in_part_host)?;
        }

        let mut probe_qr: usize = 0;
        for _ in 0..ils_quick_refine {
            probe_qr += 1;
            fused_calls = fused_calls.wrapping_add(1);
            let fused_target: u32 = fused_calls.wrapping_mul(fused_grid);
            unsafe {
                stream
                    .launch_builder(&fused_moves_kernel)
                    .arg(&(challenge.num_hyperedges as i32))
                    .arg(&(challenge.num_nodes as i32))
                    .arg(&(challenge.num_parts as i32))
                    .arg(&(challenge.max_part_size as i32))
                    .arg(&challenge.d_hyperedge_nodes)
                    .arg(&challenge.d_hyperedge_offsets)
                    .arg(&challenge.d_node_hyperedges)
                    .arg(&challenge.d_node_offsets)
                    .arg(&d_partition)
                    .arg(&d_nodes_in_part)
                    .arg(&mut d_edge_flags_all)
                    .arg(&mut d_edge_flags_double)
                    .arg(&mut d_move_priorities)
                    .arg(&mut d_grid_barrier)
                    .arg(&fused_target)
                    .launch(fused_tail_cfg.clone())?;
            }

            stream.memcpy_dtoh(&d_move_priorities, &mut move_keys_host)?;
            valid_moves.clear();
            // seed site 4/6: ILS quick-refine zero-gain lottery.
            let mut rng_state = 987654321u64.wrapping_add(run_off);
            for (node, &key) in move_keys_host.iter().enumerate() {
                if key as u32 != 0x80000000 {
                    let gain = key >> 16;
                    if gain > 0 {
                        valid_moves.push((node, key));
                    } else if gain == 0 {
                        rng_state = rng_state.wrapping_mul(6364136223846793005u64).wrapping_add(1);
                        if (rng_state % 10) < 2 {
                            let new_key = (0 << 16) | (key & 0xFFFF);
                            valid_moves.push((node, new_key));
                        }
                    }
                }
            }
            if valid_moves.is_empty() {
                break;
            }

            let cmp = |a: &(usize, i32), b: &(usize, i32)| b.1.cmp(&a.1).then(a.0.cmp(&b.0));
            let mut k_base = valid_moves.len();
            if k_base > move_limit {
                k_base = move_limit;
            }
            let ils_extra = extra_window / 2;
            let k_cand = std::cmp::min(valid_moves.len(), k_base.saturating_add(ils_extra));

            if k_cand > 1 {
                valid_moves.select_nth_unstable_by(k_cand - 1, cmp);
                valid_moves[..k_cand].sort_unstable_by(cmp);
            } else {
                valid_moves[..k_cand].sort_unstable_by(cmp);
            }

            let slack = slack_mid + 2;

            tgt_used.fill(0);
            for p in 0..num_parts_usize {
                let free = (challenge.max_part_size as i32 - nodes_in_part_host[p]).max(0) as usize;
                tgt_quota[p] = std::cmp::max(1, free.saturating_add(slack));
            }

            sorted_move_nodes.clear();
            sorted_move_parts.clear();
            for &(node, key) in valid_moves[..k_cand].iter() {
                if sorted_move_nodes.len() >= k_base {
                    break;
                }
                let tgt = (key & 63) as usize;
                if tgt < num_parts_usize && tgt_used[tgt] < tgt_quota[tgt] {
                    tgt_used[tgt] += 1;
                    sorted_move_nodes.push(node as i32);
                    sorted_move_parts.push(tgt as i32);
                }
            }
            if sorted_move_nodes.is_empty() {
                let take = std::cmp::min(k_base, k_cand);
                sorted_move_nodes.extend(valid_moves[..take].iter().map(|(n, _)| *n as i32));
                sorted_move_parts.extend(valid_moves[..take].iter().map(|(_, key)| (key & 63) as i32));
            }

            let moves_executed = simulate_execute_moves(
                &mut partition_host_refine,
                &mut nodes_in_part_host,
                &sorted_move_nodes,
                &sorted_move_parts,
            );

            if moves_executed > 0 {
                stream.memcpy_htod(&partition_host_refine, &mut d_partition)?;
                stream.memcpy_htod(&nodes_in_part_host, &mut d_nodes_in_part)?;
            }
            if moves_executed == 0 {
                break;
            }
        }

        do_swap_phase!(
            &mut d_partition, &mut d_nodes_in_part,
            &mut d_edge_flags_all, &mut d_edge_flags_double,
            &mut d_swap_gains, &mut swap_gains_host,
            &mut partition_host_swap, &mut partition_mut_swap,
            &mut part_to_part, &mut used_ba_buf,
            25, neg_gain_thresh, scan_limit_swap, scan_limit_cycle
        )?;

        unsafe {
            stream
                .launch_builder(&compute_connectivity_kernel)
                .arg(&(challenge.num_hyperedges as i32))
                .arg(&challenge.d_hyperedge_nodes)
                .arg(&challenge.d_hyperedge_offsets)
                .arg(&d_partition)
                .arg(&mut d_connectivity)
                .launch(hedge_cfg.clone())?;
        }
        unsafe {
            stream
                .launch_builder(&reduce_connectivity_sum_kernel)
                .arg(&(challenge.num_hyperedges as i32))
                .arg(&d_connectivity)
                .arg(&mut d_total_connectivity)
                .launch(connectivity_reduce_cfg.clone())?;
        }

        let block_sums = stream.memcpy_dtov(&d_total_connectivity)?;
        let num_blocks = ((challenge.num_hyperedges as u32 + block_size * 2 - 1) / (block_size * 2)) as usize;
        let new_connectivity: i32 = block_sums[..num_blocks].iter().sum();

        let mut improved = false;
        if new_connectivity < best_connectivity {
            improved = true;
            best_connectivity = new_connectivity;
            best_partition_host = stream.memcpy_dtov(&d_partition)?;
            best_nodes_in_part_host = stream.memcpy_dtov(&d_nodes_in_part)?;

            let connectivity_vec = stream.memcpy_dtov(&d_connectivity)?;
            high_hedge_ids_host = build_high_hedge_ids(&connectivity_vec, num_high_hedges);
        }

        let slot_opt: Option<usize> = if elite_count < pool_size {
            let s = elite_count;
            elite_count += 1;
            Some(s)
        } else {
            let mut worst_idx = 0usize;
            let mut worst_score = elite_scores[0];
            for i in 1..pool_size {
                if elite_scores[i] > worst_score {
                    worst_score = elite_scores[i];
                    worst_idx = i;
                }
            }
            if new_connectivity < worst_score {
                Some(worst_idx)
            } else {
                None
            }
        };

        if let Some(slot) = slot_opt {
            let src_part: Vec<i32>;
            let src_slice: &[i32] = if improved {
                &best_partition_host
            } else {
                src_part = stream.memcpy_dtov(&d_partition)?;
                &src_part
            };
            let n = challenge.num_nodes as usize;
            elite_flat_host[slot * n..(slot + 1) * n].copy_from_slice(src_slice);
            elite_scores[slot] = new_connectivity;
        }

        if probe_on {
            let us = probe_lap!();
            probe_out!(
                "P11 n={:08x} run={} phase=ils iter={} new_km1={} best_km1={} improved={} consensus={} quick_exec={} elite={} us={}\n",
                probe_nonce, bok_run, ils_iter, new_connectivity, best_connectivity,
                improved as i32, refreshed_host_from_device as i32, probe_qr,
                elite_count, us
            );
        }

        use_consensus_next = !improved;
    }

    let d_partition_final = stream.memcpy_stod(&best_partition_host)?;
    let d_nodes_in_part_final = stream.memcpy_stod(&best_nodes_in_part_host)?;
    d_partition = d_partition_final;
    d_nodes_in_part = d_nodes_in_part_final;
    partition_host_refine.copy_from_slice(&best_partition_host);
    nodes_in_part_host.copy_from_slice(&best_nodes_in_part_host);
    if probe_on {
        let us = probe_lap!();
        probe_out!(
            "P11 n={:08x} run={} phase=ils_end km1={} us={} iters={}\n",
            probe_nonce, bok_run, best_connectivity, us, ils_iterations
        );
    }

    let mut probe_pol: usize = 0;
    for _ in 0..post_ils_polish {
        probe_pol += 1;
        fused_calls = fused_calls.wrapping_add(1);
        let fused_target: u32 = fused_calls.wrapping_mul(fused_grid);
        unsafe {
            stream
                .launch_builder(&fused_moves_kernel)
                .arg(&(challenge.num_hyperedges as i32))
                .arg(&(challenge.num_nodes as i32))
                .arg(&(challenge.num_parts as i32))
                .arg(&(challenge.max_part_size as i32))
                .arg(&challenge.d_hyperedge_nodes)
                .arg(&challenge.d_hyperedge_offsets)
                .arg(&challenge.d_node_hyperedges)
                .arg(&challenge.d_node_offsets)
                .arg(&d_partition)
                .arg(&d_nodes_in_part)
                .arg(&mut d_edge_flags_all)
                .arg(&mut d_edge_flags_double)
                .arg(&mut d_move_priorities)
                .arg(&mut d_grid_barrier)
                .arg(&fused_target)
                .launch(fused_tail_cfg.clone())?;
        }

        stream.memcpy_dtoh(&d_move_priorities, &mut move_keys_host)?;
        valid_moves.clear();
        // seed site 5/6: post-ILS polish zero-gain lottery.
        let mut rng_state = 11223344u64.wrapping_add(run_off);
        for (node, &key) in move_keys_host.iter().enumerate() {
            if key as u32 != 0x80000000 {
                let gain = key >> 16;
                if gain > 0 {
                    valid_moves.push((node, key));
                } else if gain == 0 {
                    rng_state = rng_state.wrapping_mul(6364136223846793005u64).wrapping_add(1);
                    if (rng_state % 20) == 0 {
                        let new_key = (0 << 16) | (key & 0xFFFF);
                        valid_moves.push((node, new_key));
                    }
                }
            }
        }
        if valid_moves.is_empty() {
            break;
        }

        let cmp = |a: &(usize, i32), b: &(usize, i32)| b.1.cmp(&a.1).then(a.0.cmp(&b.0));
        let polish_limit = 100000usize;
        let k_base = std::cmp::min(valid_moves.len(), polish_limit);
        let polish_extra = extra_window / 3;
        let k_cand = std::cmp::min(valid_moves.len(), k_base.saturating_add(polish_extra));

        if k_cand > 1 {
            valid_moves.select_nth_unstable_by(k_cand - 1, cmp);
            valid_moves[..k_cand].sort_unstable_by(cmp);
        } else {
            valid_moves[..k_cand].sort_unstable_by(cmp);
        }

        let slack = slack_mid;

        tgt_used.fill(0);
        for p in 0..num_parts_usize {
            let free = (challenge.max_part_size as i32 - nodes_in_part_host[p]).max(0) as usize;
            tgt_quota[p] = std::cmp::max(1, free.saturating_add(slack));
        }

        sorted_move_nodes.clear();
        sorted_move_parts.clear();
        for &(node, key) in valid_moves[..k_cand].iter() {
            if sorted_move_nodes.len() >= k_base {
                break;
            }
            let tgt = (key & 63) as usize;
            if tgt < num_parts_usize && tgt_used[tgt] < tgt_quota[tgt] {
                tgt_used[tgt] += 1;
                sorted_move_nodes.push(node as i32);
                sorted_move_parts.push(tgt as i32);
            }
        }
        if sorted_move_nodes.is_empty() {
            let take = std::cmp::min(k_base, k_cand);
            sorted_move_nodes.extend(valid_moves[..take].iter().map(|(n, _)| *n as i32));
            sorted_move_parts.extend(valid_moves[..take].iter().map(|(_, key)| (key & 63) as i32));
        }

        let moves_executed = simulate_execute_moves(
            &mut partition_host_refine,
            &mut nodes_in_part_host,
            &sorted_move_nodes,
            &sorted_move_parts,
        );

        if moves_executed > 0 {
            stream.memcpy_htod(&partition_host_refine, &mut d_partition)?;
            stream.memcpy_htod(&nodes_in_part_host, &mut d_nodes_in_part)?;
        }
        if moves_executed == 0 {
            break;
        }
    }

    unsafe {
        stream
            .launch_builder(&balance_kernel)
            .arg(&(challenge.num_nodes as i32))
            .arg(&(challenge.num_parts as i32))
            .arg(&1i32)
            .arg(&(challenge.max_part_size as i32))
            .arg(&mut d_partition)
            .arg(&mut d_nodes_in_part)
            .launch(one_thread_cfg.clone())?;
    }
    stream.memcpy_dtoh(&d_partition, &mut partition_host_refine)?;
    stream.memcpy_dtoh(&d_nodes_in_part, &mut nodes_in_part_host)?;
    if probe_on {
        let us = probe_lap!();
        let k = probe_km1!(&partition_host_refine);
        probe_out!(
            "P11 n={:08x} run={} phase=polish km1={} us={} iters={} budget={}\n",
            probe_nonce, bok_run, k, us, probe_pol, post_ils_polish
        );
    }

    let post_balance_rounds = hyperparameters
        .as_ref()
        .and_then(|p| p.get("post_refinement").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(0, 128) as usize)
        .unwrap_or(base_post_balance);
    // wave9: the last member of the tail budget.  `lo == 0` because the outer
    // clamp already allows 0 here.
    // wave10: `run_pol_pct` rides on top, exactly as for `post_ils_polish`.
    let post_balance_rounds: usize =
        pct_of(pct_of(post_balance_rounds, run_tail_pct, 0), run_pol_pct, 0);

    let mut probe_pb: usize = 0;
    for _ in 0..post_balance_rounds {
        probe_pb += 1;
        fused_calls = fused_calls.wrapping_add(1);
        let fused_target: u32 = fused_calls.wrapping_mul(fused_grid);
        unsafe {
            stream
                .launch_builder(&fused_moves_kernel)
                .arg(&(challenge.num_hyperedges as i32))
                .arg(&(challenge.num_nodes as i32))
                .arg(&(challenge.num_parts as i32))
                .arg(&(challenge.max_part_size as i32))
                .arg(&challenge.d_hyperedge_nodes)
                .arg(&challenge.d_hyperedge_offsets)
                .arg(&challenge.d_node_hyperedges)
                .arg(&challenge.d_node_offsets)
                .arg(&d_partition)
                .arg(&d_nodes_in_part)
                .arg(&mut d_edge_flags_all)
                .arg(&mut d_edge_flags_double)
                .arg(&mut d_move_priorities)
                .arg(&mut d_grid_barrier)
                .arg(&fused_target)
                .launch(fused_tail_cfg.clone())?;
        }

        stream.memcpy_dtoh(&d_move_priorities, &mut move_keys_host)?;
        valid_moves.clear();
        // seed site 6/6: post-balance refinement zero-gain lottery.
        let mut rng_state = 55667788u64.wrapping_add(run_off);
        for (node, &key) in move_keys_host.iter().enumerate() {
            if key as u32 != 0x80000000 {
                let gain = key >> 16;
                if gain > 0 {
                    valid_moves.push((node, key));
                } else if gain == 0 {
                    rng_state = rng_state.wrapping_mul(6364136223846793005u64).wrapping_add(1);
                    if (rng_state % 20) == 0 {
                        let new_key = (0 << 16) | (key & 0xFFFF);
                        valid_moves.push((node, new_key));
                    }
                }
            }
        }
        if valid_moves.is_empty() {
            break;
        }

        let cmp = |a: &(usize, i32), b: &(usize, i32)| b.1.cmp(&a.1).then(a.0.cmp(&b.0));
        let mut k_base = valid_moves.len();
        let adaptive_limit = move_limit / 2;
        if k_base > adaptive_limit {
            k_base = adaptive_limit;
        }

        let post_extra = extra_window / 3;
        let k_cand = std::cmp::min(valid_moves.len(), k_base.saturating_add(post_extra));

        if k_cand > 1 {
            valid_moves.select_nth_unstable_by(k_cand - 1, cmp);
            valid_moves[..k_cand].sort_unstable_by(cmp);
        } else {
            valid_moves[..k_cand].sort_unstable_by(cmp);
        }

        let slack = slack_mid;

        tgt_used.fill(0);
        for p in 0..num_parts_usize {
            let free = (challenge.max_part_size as i32 - nodes_in_part_host[p]).max(0) as usize;
            tgt_quota[p] = std::cmp::max(1, free.saturating_add(slack));
        }

        sorted_move_nodes.clear();
        sorted_move_parts.clear();
        for &(node, key) in valid_moves[..k_cand].iter() {
            if sorted_move_nodes.len() >= k_base {
                break;
            }
            let tgt = (key & 63) as usize;
            if tgt < num_parts_usize && tgt_used[tgt] < tgt_quota[tgt] {
                tgt_used[tgt] += 1;
                sorted_move_nodes.push(node as i32);
                sorted_move_parts.push(tgt as i32);
            }
        }
        if sorted_move_nodes.is_empty() {
            let take = std::cmp::min(k_base, k_cand);
            sorted_move_nodes.extend(valid_moves[..take].iter().map(|(n, _)| *n as i32));
            sorted_move_parts.extend(valid_moves[..take].iter().map(|(_, key)| (key & 63) as i32));
        }

        let mut moves_executed = simulate_execute_moves(
            &mut partition_host_refine,
            &mut nodes_in_part_host,
            &sorted_move_nodes,
            &sorted_move_parts,
        );

        if moves_executed > 0 {
            stream.memcpy_htod(&partition_host_refine, &mut d_partition)?;
            stream.memcpy_htod(&nodes_in_part_host, &mut d_nodes_in_part)?;
        }

        if moves_executed == 0 && k_cand > k_base {
            sorted_move_nodes.clear();
            sorted_move_parts.clear();
            let tail = &valid_moves[k_base..k_cand];
            let take = std::cmp::min(tail.len(), k_base);
            sorted_move_nodes.extend(tail.iter().take(take).map(|(n, _)| *n as i32));
            sorted_move_parts.extend(tail.iter().take(take).map(|(_, key)| (key & 63) as i32));

            if !sorted_move_nodes.is_empty() {
                moves_executed = simulate_execute_moves(
                    &mut partition_host_refine,
                    &mut nodes_in_part_host,
                    &sorted_move_nodes,
                    &sorted_move_parts,
                );
                if moves_executed > 0 {
                    stream.memcpy_htod(&partition_host_refine, &mut d_partition)?;
                    stream.memcpy_htod(&nodes_in_part_host, &mut d_nodes_in_part)?;
                }
            }
        }

        let _ = (moves_executed, total_moves_executed, total_perturbations);
        if moves_executed == 0 {
            break;
        }
    }
    if probe_on {
        let us = probe_lap!();
        let k = probe_km1!(&partition_host_refine);
        probe_out!(
            "P11 n={:08x} run={} phase=postbal km1={} us={} iters={} budget={}\n",
            probe_nonce, bok_run, k, us, probe_pb, post_balance_rounds
        );
    }

    do_swap_phase!(
        &mut d_partition, &mut d_nodes_in_part,
        &mut d_edge_flags_all, &mut d_edge_flags_double,
        &mut d_swap_gains, &mut swap_gains_host,
        &mut partition_host_swap, &mut partition_mut_swap,
        &mut part_to_part, &mut used_ba_buf,
        10, neg_gain_thresh, scan_limit_swap, scan_limit_cycle
    )?;

    let partition = stream.memcpy_dtov(&d_partition)?;
    // ==== END per-run body ====

        // ---- best-of-K selection on the TRUE km1 objective ----
        let run_km1 = eval_km1(
            &partition,
            &hedge_offsets_host,
            &hyperedge_nodes_host,
            challenge.num_hyperedges as usize,
            num_parts_usize,
            &mut bok_seen,
        );
        if probe_on {
            let us = probe_lap!();
            probe_out!(
                "P11 n={:08x} run={} phase=final km1={} us={}\n",
                probe_nonce, bok_run, run_km1, us
            );
        }
        // wave9: evaluated BEFORE the selection below updates `bok_best_km1`,
        // so `bok_best_raw` tracks `bok_best_part` exactly (strict `<`, ties
        // keep the earlier run).
        let bok_improved = run_km1 < bok_best_km1;
        // Strict `<` keeps the EARLIEST best run, so at runs == 1 (and whenever
        // no later run strictly wins) the saved solution is run 0's, i.e. the
        // base solver's, bit for bit.
        if run_km1 < bok_best_km1 {
            bok_best_km1 = run_km1;
            bok_best_part = partition.iter().map(|&x| x as u32).collect();
        }
        // wave9: i32 mirrors for the exploit flavors.  Pure bookkeeping -- it
        // reads nothing and changes no selection; at `runs == 1` it runs once
        // and is never read.
        if bok_improved {
            bok_best_raw.clear();
            bok_best_raw.extend_from_slice(&partition);
        }
        bok_prev_raw.clear();
        bok_prev_raw.extend_from_slice(&partition);
        // wave20/22: backbone vote.  The first finished run defines the
        // reference labels; every later run can only demote a node from
        // stable to disputed.  wave22: a backbone run's final votes too, so
        // the NEXT backbone run re-derives its disputed set over the whole
        // population of finals (iterated fixed-set search).  A backbone final
        // agrees with the reference on every frozen node except the ones a
        // strictly improving candidate moved, so its vote can only demote
        // those -- nodes that evidently were not settled.  The last run's
        // vote is computed and never read, exactly as before.
        if partition.len() == bb_n {
            if bb_ref.len() != bb_n {
                bb_ref.clear();
                bb_ref.extend_from_slice(&partition);
                bb_stable.clear();
                bb_stable.resize(bb_n, true);
            } else {
                for i in 0..bb_n {
                    if partition[i] != bb_ref[i] {
                        bb_stable[i] = false;
                    }
                }
            }
        }
    }

    // ---- wave14: PAIRWISE FM POLISH of the best-of-K incumbent -------------
    // Decomposition (heaviest-cut part pairs) + 2-way FM local search, host
    // only, on the saved partition.  Gated on the TRUE km1 and on balance, so
    // the saved solution can only get better.  `fm_rounds = 0` switches it off.
    let fm_rounds: usize = hyperparameters
        .as_ref()
        .and_then(|p| p.get("fm_rounds").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(0, 16) as usize)
        .unwrap_or(2);
    let fm_pairs: usize = hyperparameters
        .as_ref()
        .and_then(|p| p.get("fm_pairs").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(0, 4096) as usize)
        .unwrap_or(96);
    let fm_passes: usize = hyperparameters
        .as_ref()
        .and_then(|p| p.get("fm_passes").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(1, 16) as usize)
        .unwrap_or(3);
    let fm_nonimp: usize = hyperparameters
        .as_ref()
        .and_then(|p| p.get("fm_nonimp").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(1, 100_000) as usize)
        .unwrap_or(64);
    // wave15: k-way boundary FM after the pairwise decomposition (same gate).
    let kfm_passes: usize = hyperparameters
        .as_ref()
        .and_then(|p| p.get("kfm_passes").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(0, 16) as usize)
        .unwrap_or(4);
    let kfm_nonimp: usize = hyperparameters
        .as_ref()
        .and_then(|p| p.get("kfm_nonimp").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(1, 100_000) as usize)
        .unwrap_or(256);
    // wave16: hyperedge-collapse block moves between the two FM operators
    // (same gate).  `ecol_passes = 0` switches it off.
    let ecol_passes: usize = hyperparameters
        .as_ref()
        .and_then(|p| p.get("ecol_passes").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(0, 16) as usize)
        .unwrap_or(3);
    let ecol_max_size: usize = hyperparameters
        .as_ref()
        .and_then(|p| p.get("ecol_max_size").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(2, 4096) as usize)
        .unwrap_or(16);
    let ecol_max_lambda: usize = hyperparameters
        .as_ref()
        .and_then(|p| p.get("ecol_max_lambda").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(2, 256) as usize)
        .unwrap_or(4);
    // wave17: disjoint part-group decomposition + group-restricted k-way FM,
    // between the pairwise and the collapse operators (same gate).
    // `gfm_rounds = 0` switches it off.
    let gfm_rounds: usize = hyperparameters
        .as_ref()
        .and_then(|p| p.get("gfm_rounds").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(0, 16) as usize)
        .unwrap_or(2);
    let gfm_size: usize = hyperparameters
        .as_ref()
        .and_then(|p| p.get("gfm_size").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(3, 64) as usize)
        .unwrap_or(4);
    let gfm_passes: usize = hyperparameters
        .as_ref()
        .and_then(|p| p.get("gfm_passes").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(1, 16) as usize)
        .unwrap_or(2);
    let gfm_nonimp: usize = hyperparameters
        .as_ref()
        .and_then(|p| p.get("gfm_nonimp").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(1, 100_000) as usize)
        .unwrap_or(128);
    // wave18: multilevel V-cycle (same-part heavy-edge contraction + weighted
    // k-way FM per coarse level) between the collapse and the k-way operators
    // (same gate).  `ml_levels = 0` switches it off.
    let ml_levels: usize = hyperparameters
        .as_ref()
        .and_then(|p| p.get("ml_levels").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(0, 8) as usize)
        .unwrap_or(3);
    let ml_passes: usize = hyperparameters
        .as_ref()
        .and_then(|p| p.get("ml_passes").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(1, 16) as usize)
        .unwrap_or(2);
    let ml_nonimp: usize = hyperparameters
        .as_ref()
        .and_then(|p| p.get("ml_nonimp").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(1, 100_000) as usize)
        .unwrap_or(128);
    let ml_max_edge_size: usize = hyperparameters
        .as_ref()
        .and_then(|p| p.get("ml_max_edge_size").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(2, 4096) as usize)
        .unwrap_or(32);
    // wave19: hypergraph-locality region decomposition + region-restricted
    // k-way FM, after the group operator (same gate).  `rfm_rounds = 0`
    // switches it off.
    let rfm_rounds: usize = hyperparameters
        .as_ref()
        .and_then(|p| p.get("rfm_rounds").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(0, 64) as usize)
        .unwrap_or(4);
    let rfm_regions: usize = hyperparameters
        .as_ref()
        .and_then(|p| p.get("rfm_regions").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(1, 100_000) as usize)
        .unwrap_or(48);
    let rfm_max_nodes: usize = hyperparameters
        .as_ref()
        .and_then(|p| p.get("rfm_max_nodes").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(2, 1_000_000) as usize)
        .unwrap_or(512);
    let rfm_max_edge_size: usize = hyperparameters
        .as_ref()
        .and_then(|p| p.get("rfm_max_edge_size").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(2, 4096) as usize)
        .unwrap_or(64);
    let rfm_passes: usize = hyperparameters
        .as_ref()
        .and_then(|p| p.get("rfm_passes").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(1, 16) as usize)
        .unwrap_or(2);
    let rfm_nonimp: usize = hyperparameters
        .as_ref()
        .and_then(|p| p.get("rfm_nonimp").and_then(|v| v.as_i64()))
        .map(|v| v.clamp(1, 100_000) as usize)
        .unwrap_or(48);
    if fm_rounds > 0 && fm_pairs > 0 && bok_best_part.len() == challenge.num_nodes as usize {
        let fm_fetched: Vec<i32>;
        let node_hedges_fm: &[i32] = if node_hedges_host.is_empty() {
            fm_fetched = stream.memcpy_dtov(&challenge.d_node_hyperedges)?;
            &fm_fetched
        } else {
            &node_hedges_host
        };
        let mut fm_part: Vec<i32> = bok_best_part.iter().map(|&x| x as i32).collect();
        let mut fm_sizes: Vec<i32> = vec![0i32; num_parts_usize];
        let mut fm_valid = true;
        for &p in fm_part.iter() {
            if p >= 0 && (p as usize) < num_parts_usize {
                fm_sizes[p as usize] += 1;
            } else {
                fm_valid = false;
            }
        }
        if fm_valid {
            let _ = pairwise_fm_polish(
                &mut fm_part,
                &mut fm_sizes,
                &hedge_offsets_host,
                &hyperedge_nodes_host,
                &node_offsets_host,
                node_hedges_fm,
                challenge.num_hyperedges as usize,
                num_parts_usize,
                challenge.max_part_size as i32,
                fm_rounds,
                fm_pairs,
                fm_passes,
                fm_nonimp,
            );
            if gfm_rounds > 0 {
                let _ = group_fm_polish(
                    &mut fm_part,
                    &mut fm_sizes,
                    &hedge_offsets_host,
                    &hyperedge_nodes_host,
                    &node_offsets_host,
                    node_hedges_fm,
                    challenge.num_hyperedges as usize,
                    num_parts_usize,
                    challenge.max_part_size as i32,
                    gfm_rounds,
                    gfm_size,
                    gfm_passes,
                    gfm_nonimp,
                );
            }
            if rfm_rounds > 0 {
                let _ = region_fm_polish(
                    &mut fm_part,
                    &mut fm_sizes,
                    &hedge_offsets_host,
                    &hyperedge_nodes_host,
                    &node_offsets_host,
                    node_hedges_fm,
                    challenge.num_hyperedges as usize,
                    num_parts_usize,
                    challenge.max_part_size as i32,
                    rfm_rounds,
                    rfm_regions,
                    rfm_max_nodes,
                    rfm_max_edge_size,
                    rfm_passes,
                    rfm_nonimp,
                );
            }
            if ecol_passes > 0 {
                let _ = hyperedge_collapse_polish(
                    &mut fm_part,
                    &mut fm_sizes,
                    &hedge_offsets_host,
                    &hyperedge_nodes_host,
                    &node_offsets_host,
                    node_hedges_fm,
                    challenge.num_hyperedges as usize,
                    num_parts_usize,
                    challenge.max_part_size as i32,
                    ecol_passes,
                    ecol_max_size,
                    ecol_max_lambda,
                );
            }
            if ml_levels > 0 {
                let _ = multilevel_fm_polish(
                    &mut fm_part,
                    &mut fm_sizes,
                    &hedge_offsets_host,
                    &hyperedge_nodes_host,
                    &node_offsets_host,
                    node_hedges_fm,
                    challenge.num_hyperedges as usize,
                    num_parts_usize,
                    challenge.max_part_size as i32,
                    ml_levels,
                    ml_passes,
                    ml_nonimp,
                    ml_max_edge_size,
                );
            }
            if kfm_passes > 0 {
                let _ = kway_fm_polish(
                    &mut fm_part,
                    &mut fm_sizes,
                    &hedge_offsets_host,
                    &hyperedge_nodes_host,
                    &node_offsets_host,
                    node_hedges_fm,
                    &[],
                    challenge.num_hyperedges as usize,
                    num_parts_usize,
                    challenge.max_part_size as i32,
                    kfm_passes,
                    kfm_nonimp,
                );
            }
            let fm_km1 = eval_km1(
                &fm_part,
                &hedge_offsets_host,
                &hyperedge_nodes_host,
                challenge.num_hyperedges as usize,
                num_parts_usize,
                &mut bok_seen,
            );
            let mut fm_ok = fm_km1 < bok_best_km1;
            for p in 0..num_parts_usize {
                if fm_sizes[p] < 1 || fm_sizes[p] > challenge.max_part_size as i32 {
                    fm_ok = false;
                }
            }
            if fm_ok {
                bok_best_km1 = fm_km1;
                bok_best_part = fm_part.iter().map(|&x| x as u32).collect();
            }
        }
    }
    let _ = bok_best_km1;

    // bok_runs >= 1, so bok_best_part is always populated here.
    save_solution(&Solution {
        partition: bok_best_part,
    })?;
    Ok(())
}

// ---- wave4_kmicro2: degree/size-class warp-task lists ------------------------
// Class c gets FF_LANES[c] lanes per work item: the smallest power of two for
// which the per-lane strided loop runs at most 4 iterations (class 5 runs at
// most 8, because the moves loop caps used_degree at 256).  Homogeneous warps
// are the point: before this, a warp's cost was max(size) over the 32 items its
// lanes happened to land on while the mean size is ~5.
const FF_LANES: [usize; 6] = [1, 2, 4, 8, 16, 32];

#[inline]

fn ff_class_of(sz: i32) -> usize {
    if sz <= 4 {
        0
    } else if sz <= 8 {
        1
    } else if sz <= 16 {
        2
    } else if sz <= 32 {
        3
    } else if sz <= 64 {
        4
    } else {
        5
    }
}

/// `units` = (id, csr_start, size).  Returns (slots, n_tasks):
///
///  * `slots`: flattened int4 records {id, csr_start, size, class}, THIRTY-TWO
///    per warp task, laid out task by task -- slot `t*32 + l` is the work item
///    that LANE l of warp task t owns.  A class-c task gives `FF_LANES[c]`
///    consecutive lanes to one item (`FF_LANES[c] == 1 << c`, so lane l holds
///    item `first + (l >> c)` of that class), which is exactly the mapping the
///    champion's kernel computed as `items[base_item + (lane >> SH)]`.  Making
///    it explicit lets the kernel load the record with an address that depends
///    on nothing (`slots[t*32 + lane]`), so the task-descriptor load and the
///    item load -- two serialized dependent global latencies at 8 warps/SM --
///    collapse into one coalesced 512-B load per warp that can additionally be
///    prefetched a whole task ahead.  Cost: the array is `lane budget` int4s
///    (~23k for the moves phase) instead of `item count` int4s (~18.4k), i.e.
///    ~70 KB more per phase, read coalesced once per round.
///    A class segment is padded to a whole number of tasks by REPEATING its
///    last real entry, exactly as the champion's item list was: recomputing a
///    work item is idempotent (it reads only state that is constant for the
///    phase and stores the identical value to the identical address) and the
///    duplicate lanes are in the SAME warp, so padding cannot change any output.
///
///  * `n_tasks`: the number of warp tasks.
///
/// TASK ORDER (wave6 W7).  A warp grid-strides the task list, so warp w runs
/// tasks w, w + W, ... with W = blocks x warps_per_block (628 here).  With
/// ~716 tasks only `extra = n_tasks - W` ~ 88 warps run a SECOND task, and the
/// champion's heaviest-class-first order handed those extras to precisely the
/// warps that had just run the heaviest first task, so the phase makespan was
/// "one class-5 task + one class-0 task".  Here the first `extra` and the last
/// `extra` positions are both filled with class-0 tasks (the cheapest: one pin
/// quad, one flag gather), so the doubled-up warps run cheap+cheap and every
/// heavy task runs alone.  This is a pure permutation of the task list -- each
/// task still covers the same items with the same lane mapping, and work items
/// are independent of each other -- so it is bit-identical by construction.
fn ff_build_slots(units: &[(i32, i32, i32)], warps: usize) -> (Vec<i32>, i32) {
    let mut by_class: Vec<Vec<(i32, i32, i32)>> = (0..6).map(|_| Vec::new()).collect();
    for &(id, st, sz) in units.iter() {
        by_class[ff_class_of(sz)].push((id, st, sz));
    }
    // one entry per warp task: (class, index of its first item within the class)
    let mut light: Vec<(usize, usize)> = Vec::new();
    let mut heavy: Vec<(usize, usize)> = Vec::new();
    for c in (0..6).rev() {
        let n = by_class[c].len();
        if n == 0 {
            continue;
        }
        let per_task = 32 / FF_LANES[c];
        let ntask = (n + per_task - 1) / per_task;
        for t in 0..ntask {
            if c == 0 {
                light.push((c, t * per_task));
            } else {
                heavy.push((c, t * per_task));
            }
        }
    }
    let n_tasks = light.len() + heavy.len();
    let extra = if n_tasks > warps { n_tasks - warps } else { 0 };
    let mut order: Vec<(usize, usize)> = Vec::with_capacity(n_tasks);
    if extra > 0 && light.len() >= 2 * extra {
        let head = light.len() - extra;
        order.extend_from_slice(&light[head..]);   // positions 0..extra   : cheap
        order.extend_from_slice(&heavy);           // the expensive tasks run alone
        order.extend_from_slice(&light[..head]);   // >= extra cheap tasks, so the
                                                   // last `extra` positions are cheap
    } else {
        order.extend_from_slice(&heavy);
        order.extend_from_slice(&light);
    }
    let mut slots: Vec<i32> = Vec::with_capacity(n_tasks * 32 * 4);
    for &(c, first) in order.iter() {
        let n = by_class[c].len();
        for l in 0..32usize {
            let k = first + (l >> c);              // FF_LANES[c] == 1 << c
            let (id, st, sz) = by_class[c][std::cmp::min(k, n - 1)];
            slots.push(id);
            slots.push(st);
            slots.push(sz);
            slots.push(c as i32);
        }
    }
    (slots, n_tasks as i32)
}
