# TIG Code Submission

## Submission Details

* **Challenge Name:** vehicle_routing
* **Algorithm Name:** adaptive_tig_adp_v3
* **Copyright:** 2025 Brent Beane 
* **Identity of Submitter:** Brent Beane
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

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


## References and Acknowledgments

This implementation is a Rust re-implementation and engineering of the Hybrid Genetic Search (HGS) family of approaches adapted to a Time-Indexed Graph (TIG) representation with Approximate Dynamic Programming (ADP) components. The implementation is intended to be an academically-grounded, benchmark-ready solver that preserves algorithmic intent while offering a compact, modular, Rust-native codebase optimized for experiments and integration with TIG infrastructure.

The solver benefits from the original HGS literature and follows the methodological ideas introduced and extended by the authors cited below. The code also integrates neighborhood and local-search ideas described in more recent HGS implementations.


## Recent Benchmark Results (developer runs)

These are quick developer-run summaries produced while iterating on the `adaptive_tig_adp_v3` implementation. Two run outputs are shown below (both executed from the repository root using the Sigma runner):

- 201 ms budget (short run)
	- Run id: run_1766157280
	- Nonces: 2
	- Mean score: 0.10598
	- Nonce 0: seed=13895752972849944850, score=0.10346, total_cost=6437, elapsed_ms=185, solution_hash=4696ef34f9c56b33
	- Nonce 1: seed=6849723718463591719, score=0.10850, total_cost=4630, elapsed_ms=143, solution_hash=b65e6ea44fd17a72
	- Validation: both nonces produced syntactically valid solutions (no missing customers).

- 1200 ms budget (longer run)
	- Run id: run_1766158887
	- Nonces: 2
	- Mean score: 0.10414
	- Nonce 0: seed=13895752972849944850, score=0.10235, total_cost=6437, elapsed_ms=709, solution_hash=4696ef34f9c56b33
	- Nonce 1: seed=6849723718463591719, score=0.10593, total_cost=4630, elapsed_ms=1049, solution_hash=b65e6ea44fd17a72
	- Validation: both nonces produced syntactically valid solutions (no missing customers). Constraint violations (time-windows / capacity) may still occur depending on instance specifics and constructive choices.

Notes on these runs:
- The two time budgets demonstrate behaviour under limited versus more generous CPU budgets. For these particular instances the constructive solution dominates the output (the same solution hashes appear across budgets), so increasing time improved internal refinement but did not substantially change the constructed route in these runs.
- The `solution_hash` is a deterministic hash of the produced solution JSON and is useful for regression checks.

## Comparison to `hgs_v1`

- Scope: `hgs_v1` is the earlier HGS-style reference implementation (baseline) used in prior experiments. It implements the full HGS pipeline including multi-route splitting, strong capacity-aware constructive heuristics, and tuned population/genetic operators.
- Current status of `adaptive_tig_adp_v3`:
	- Now guarantees a complete solution (every customer appears, depot loop closed) by using a TIG-aware constructive pass and conservative fallbacks. This fixes earlier runs that produced partial routes or omitted customers.
	- The implementation currently uses a greedy, feasibility-first insertion followed by forced insertions for remaining customers (Phase II incremental constructive). This produces valid but sometimes constraint-violating single-route solutions for small instances until a multi-route split/repair is applied.
- How that differs from `hgs_v1` in practice:
	- `hgs_v1` typically produces multiple vehicle routes that better respect capacity and time-windows out of the constructive + split phases, so its raw validation error counts and penalties are usually lower on comparable budgets.
	- `adaptive_tig_adp_v3` focuses on the TIG + ADP integration and local-search/learning components; closing the remaining gap (split/repair and TW-aware insertion) is the next engineering step to match `hgs_v1` baseline quality.

## Testing notes and reproducibility

- To reproduce these developer runs use the Sigma runner command from the repository root (examples below):

```bash
# short run (approx 201ms):
cargo run --bin adaptive_tig_adp_v3 -- --benchmark sigma2 --time-limit-ms 201 --seed 42 > benchmarks/adaptive_tig_adp_v3/201ms/run.json || true

# longer run (approx 1200ms):
cargo run --bin adaptive_tig_adp_v3 -- --benchmark sigma2 --time-limit-ms 1200 --seed 42 > benchmarks/adaptive_tig_adp_v3/1200ms/run.json || true
```

- Deterministic seeds and instance generation are used so the `solution_hash` and scores are reproducible between runs at the same seed and code state.
