# c008_exp04

## Submission Details

* **Challenge Name:** energy_arbitrage
* **Algorithm Name:** c008_exp04
* **Copyright:** 2026 Dale
* **Identity of Submitter:** Dale
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

## Algorithm Summary

`c008_exp04` is a deterministic rolling dispatch policy for the TIG Energy Arbitrage challenge. It controls batteries across the five active instance families by selecting a track-specialised solver from the number of batteries:

* baseline: up to 15 batteries
* congested: up to 30 batteries
* multiday: up to 50 batteries
* dense: up to 80 batteries
* capstone: up to 150 batteries

The algorithm builds a single-battery Bellman value table over state of charge, then uses network-aware coordinate ascent to choose real-time actions under PTDF line-flow constraints. A final deflator repairs any remaining line-limit pressure before returning the action vector.

## Key Ideas

* Stochastic expected real-time pricing: the DP cache samples the public market model instead of treating day-ahead prices as the exact realized price.
* Network-aware dispatch: each battery has a sparse PTDF footprint, and the action optimiser respects line limits during coordinate ascent.
* Track-specific congestion derating: congested, multiday, dense, and capstone cases use different action intensity and iteration settings.
* Revealed real-time momentum: non-baseline tracks use only current and previous revealed `state.rt_prices` through an EMA bias.
* Baseline momentum gate: the baseline track disables real-time momentum because testing showed the simple network was hurt by reacting to short-term noise.

The solver does not use future hidden real-time prices, randomness in online action selection, wallets, live trading, or changes to the verifier/challenge rules.

## Hyperparameters

The defaults are selected per track. Optional JSON hyperparameters may override:

* `soc_levels`: SOC grid size for the Bellman table.
* `action_grid`: action discretisation used while building the DP cache.
* `asca_iters`: coordinate-ascent sweeps per timestep.
* `ternary_iters`: per-battery action search iterations.
* `convergence_tol`: early-stop tolerance for coordinate ascent.
* `anticipate_lmp`: enable expected congestion premium adjustment.
* `lmp_threshold`: exogenous flow utilisation threshold for LMP anticipation.
* `lmp_premium_scale`: scale for congestion premium estimates.
* `jump_premium`: optional extra sell-side premium.
* `prune_ratio`: optional pruning of small batteries in the ASCA order.
* `deflator_iters`: max repair passes for line-limit violations.
* `flow_margin`: safety margin below line limits.
* `network_derating`: per-track action derating factor.

Recommended submission/default use is `null` hyperparameters.

## Validation Results

Local 5-seed benchmark quality, higher is better:

| Track | Local avg quality |
|---|---:|
| baseline | 4,294,774 |
| congested | 5,445,736 |
| multiday | 9,438,462 |
| dense | 9,583,998 |
| capstone | 9,910,615 |

Official-style Docker `test_algorithm` validation over 50 nonces per track:

| Track | Valid | Invalid | Avg quality |
|---|---:|---:|---:|
| baseline | 50 | 0 | 1,843,282 |
| congested | 50 | 0 | 5,728,889 |
| multiday | 50 | 0 | 8,727,308 |
| dense | 50 | 0 | 9,534,738 |
| capstone | 50 | 0 | 9,754,876 |

Docker dev-image build succeeded for:

```text
tig-algorithms/lib/energy_arbitrage/arm64/c008_exp04.so
```

## References and Acknowledgments

### 1. Academic Papers

* Bellman, R., *Dynamic Programming*, Princeton University Press, 1957.
* Bertsekas, D. P., *Dynamic Programming and Optimal Control*, Athena Scientific, 2017.
* Wood, A. J., Wollenberg, B. F., and Sheble, G. B., *Power Generation, Operation, and Control*, Wiley, 2013.
* Boyd, S. and Vandenberghe, L., *Convex Optimization*, Cambridge University Press, 2004.

### 2. Code References

This implementation is original source for `c008_exp04` and follows the public TIG Energy Arbitrage challenge API.

### 3. Other

The algorithm was developed and benchmarked against the public TIG Energy Arbitrage generator and verifier interfaces without modifying challenge rules.

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
