# c008_a001

## Submission Details

* **Challenge Name:** energy_arbitrage
* **Algorithm Name:** c008_a001
* **Copyright:** 2026 Dale
* **Identity of Submitter:** Dale
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

RT-aware rolling arbitrage heuristic for the Energy Arbitrage challenge.

## Baseline Weaknesses

The in-challenge baseline picks the better of:

- a greedy day-ahead lookahead policy that charges or discharges every battery based on a grid-level future average, then scales actions to satisfy flow limits;
- a conservative policy that avoids negative cumulative profit by scaling down risky actions.

This is robust, but it leaves obvious profit on the table because it mostly ignores the currently revealed real-time nodal price, treats batteries similarly even when their nodes differ, and repairs congestion only after proposing a portfolio-wide action.

## Strategy

This solver keeps the verifier/rules untouched and implements only a policy:

- uses current real-time nodal prices for the executable step;
- uses known day-ahead prices over a 24-step rolling horizon as a legal forecast of future opportunity cost;
- scores charge/discharge candidates per battery using SOC balance, efficiency, transaction cost, and future DA low/average/high prices;
- adds actions incrementally in descending score order;
- accepts only the feasible fraction of each action under PTDF line limits, keeping a small flow headroom;
- falls back to global scaling, and then zero-action feasibility, if a numerical edge case slips through.

The policy is deterministic and does not use hidden future RT prices.

## Benchmark

Run from the repo root in an environment with Rust/Cargo:

```bash
./scripts/benchmark_energy_arbitrage.sh 10
```

That executes 10 seeds for each Energy Arbitrage scenario and reports official baseline profit, candidate profit, percent delta, worst case, and rollout time.

## Measured Results

Measured locally on macOS with:

```bash
./scripts/benchmark_energy_arbitrage.sh 50
```

Across 50 seeds for each of the five scenarios, average baseline profit was `23193.63`, average candidate profit was `178424.47`, and aggregate profit improvement was `669.28%`. There was one losing case out of 250: `baseline seed 26`, where profit was `12119.54` versus baseline `14323.97` (`-15.39%`). Average rollout time was `282.62 ms`.

Additional synthetic stress checks can be run with:

```bash
cargo run -p tig-challenges --features energy_arbitrage --example stress_energy_arbitrage
```

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
