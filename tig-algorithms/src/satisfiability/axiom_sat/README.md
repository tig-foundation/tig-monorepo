# TIG Code Submission

## Submission Details

* **Challenge Name:** satisfiability
* **Algorithm Name:** axiom_sat
* **Copyright:** 2026 ubelai
* **Identity of Submitter:** ubelai
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

## Target Tracks

| Track | Variables (`n_vars`) | Clause/Var Ratio | Route |
|-------|----------------------|------------------|-------|
| Track 1 | `5,000`   | `4.267` | Pure-clause + unit-propagation reduction (`Pass 1 + Pass 2`) + polarity-partitioned 2-SAT watcher & cached WalkSAT (`ProbSAT` polynomial table) |
| Track 2 | `7,500`   | `4.267` | Adaptive segmented WalkSAT with checkpointed conflict-set swap and polarity-partitioned occurrence scans |
| Track 3 | `10,000`  | `4.267` | Cached critical-ratio WalkSAT with zero-break freebie shortcut and majority-guided restart schedule |
| Track 4 | `100,000` | `4.150` | Thread-local zero-allocation `SatWorkspace` with contiguous `[u32; 3]` clause triples and polarity-partitioned occurrence lists |
| Track 5 | `100,000` | `4.200` | Polarity-partitioned occurrence CSR + cached WalkSAT with direct `true_count == 1` crit-sum maintenance |

## Method Overview

`axiom_sat` is a self-contained 3-SAT (`c001`) solver engineered around cache-line locality, zero per-instance heap churn, and polarity-partitioned occurrence traversal:

1. **Persistent Thread-Local `SatWorkspace`**:
   - Reuses a single `thread_local!` workspace (`SAT_WS`) across seeds inside a bundle, eliminating repeated multi-megabyte OS demand-zero page faults on `N = 100,000` tracks (`Track 4` and `Track 5`) while bounding all slice views strictly to `[..n]` / `[..m]`.
2. **Two-Pass Preprocessing (`Pass 1` Pure-Literal & Unit Cascade + `Pass 2` Subsumption/Reduction)**:
   - Fuses literal counting and contiguous `[u32; 3]` clause packing, cascading unit clauses and pure literals before building CSR occurrence tables.
3. **Polarity-Partitioned Occurrence CSR (`occ_pos_r` / `occ_neg_r` + `occ_data`)**:
   - Separates positive and negative literal occurrences into disjoint contiguous slices (`occ_pos_r[v]` and `occ_neg_r[v]`). Flipping variable `v` scans only the newly falsified polarity slice to increment `crit_v` / `unsat` when `true_count == 1` and only the newly satisfied polarity slice to decrement `crit_v` / `unsat`, eliminating per-clause sign branches and cutting inner-loop memory traffic in half.
4. **Precomputed `PROBSAT_POLY_TABLE` WalkSAT Core**:
   - Evaluates break-score weights via a 64-entry precomputed polynomial lookup table (`cb` scaled by instance regime), taking immediate zero-break freebie flips when available and maintaining an exact O(1) swap-remove `unsat_clauses` buffer with full-assignment verification before invoking `save_solution`.

## Validation Summary

- **50-Seed 5-Track Gauntlet (`Tracks 1..5`, 10 seeds per track)**: **`50 / 50` solved (`100.0%`)** in **`2,522.2 ms`** total wall time (`~304.8 ms` Seed-1 cold+warm switchboard latency).

## References and Acknowledgments

### Academic Papers
- B. Selman, H. A. Kautz, B. Cohen, *"Noise strategies for improving local search"*, AAAI, 1994 (WalkSAT).
- A. Balint, U. Schöning, *"Choosing probability distributions for stochastic local search and the role of make versus break"*, SAT, 2012 (probSAT).

### Code References
- TIG `c001` public submissions (`sat_exact_sp128` / `c001_a127`, `sat_tailwalk_v7` / `c001_a128`, `clausal_ember` / `c001_a121`) under the TIG Open Data License and TIG Innovator Outbound Game License.

## License

The files in this folder are under the following licenses:
* TIG Benchmarker Outbound License
* TIG Commercial License
* TIG Inbound Game License
* TIG Innovator Outbound Game License
* TIG Open Data License
* TIG THV Game License

Copies of the licenses can be obtained at:
https://github.com/tig-foundation/tig-monorepo/tree/main/docs/licenses
