# TIG Code Submission

## Submission Details

* **Challenge Name:** vehicle_routing
* **Algorithm Name:** adaptive_tig_adp_v4
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

# Adaptive TIG VRPTW Solver (Single-File Submission)

## Overview

This submission provides a Vehicle Routing Problem with Time Windows (VRPTW) solver implemented in Rust, packaged as a single-file submission (`mod.rs`) for compatibility with the TIG innovation framework.

The solver implements an adaptive heuristic architecture combining:

- Approximate Dynamic Programming (ADP)
- Dynamic Learning Tables (DLT)
- Value Function Approximation (VFA)
- Local search with delta tables
- Population-based refinement

The implementation is optimized for short-horizon, high-frequency optimization, aligning with Sigma II execution constraints.

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
---

## Submission Format

This submission intentionally uses a single Rust source file:

```
mod.rs
```

All internal modules (solver, ADP, local search, validation, instance generation, etc.) are consolidated within `mod.rs` using inline `mod {}` declarations.

This layout is intentional and compliant with TIG’s compilation and packaging requirements.

---

## TIG Integration

The solver is exposed under the exact module path expected by the TIG harness:

```
tig_algorithms::vehicle_routing::capacitated_vehicle_routing::Solver
```

This is achieved via explicit re-exports at the bottom of `mod.rs`.

No external glue code is required.

---

## Algorithm Summary

Core components include:

- ADP + DLT for online learning of move values
- Value Function Approximation combining feasibility, slack, and cost signals
- Local search operators (relocate, swap, 2-opt, OR-opt)
- Lightweight population-based diversification
- Full feasibility validation with repair fallback

---

## Determinism & Reproducibility

- Deterministic seeding is used throughout
- Bundle and nonce behavior is reproducible
- No external state is persisted between runs

---

## Dependencies

Uses only standard Rust crates available in the TIG environment:

- serde / serde_json
- rand
- anyhow
- smallvec

---

## Files Included

```
mod.rs
README.md
```
