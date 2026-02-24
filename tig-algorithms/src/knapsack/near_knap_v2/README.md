# TIG Code Submission

## Submission Details

* **Challenge Name:** knapsack
* **Algorithm Name:** near_knap_v2
* **Copyright:** 2026 testing
* **Identity of Submitter:** testing
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

## Additional Details

Multi-start Iterated Local Search (ILS) with Hybrid Basin Discovery for the Quadratic Knapsack Problem.

### Hyperparameters

| Parameter | Type | Default | Range | Affects |
|---|---|---|---|---|
| `effort` | integer | 1 | 1–6 | n<2500 tracks only |
| `stall_limit` | integer | 12 | 1–20 | All tracks |
| `perturbation_strength` | integer | auto | 1–20 | All tracks |
| `perturbation_rounds` | integer | auto | 1–100 | All tracks |

### Usage

Pass hyperparameters as a JSON object, e.g. `{"effort":6}` or `{"perturbation_rounds":25}`.

**`effort`** (1–6, default 1)
Controls overall search intensity for n<2500 tracks. Scales perturbation rounds, VND iterations, and number of construction starts. Has no effect on n≥2500 tracks.
- effort=1: 15 rounds, 350 VND iters
- effort=3: 29 rounds, 420 VND iters, +1 start
- effort=6: 50 rounds, 525 VND iters, +2 starts

**`stall_limit`** (1–20, default 12)
Number of consecutive non-improving perturbation rounds before early exit. Affects all tracks. Higher values allow more persistent search before giving up.

**`perturbation_strength`** (1–20, default auto)
Overrides the number of items removed per perturbation step. When not set, scales automatically with effort and instance size. Affects all tracks including n≥2500.

**`perturbation_rounds`** (1–100, default auto)
Overrides the maximum number of ILS perturbation rounds. When not set, scales with effort for n<2500 and is fixed at 20 for n≥2500. Affects all tracks including n≥2500.

### Track Behaviour

| Track | n | Budget | Notes |
|---|---|---|---|
| 1 | 1,000 | 5% | Responds well to effort and stall_limit |
| 2 | 1,000 | 10% | Responds well to effort and stall_limit |
| 3 | 1,000 | 25% | Responds well to effort and stall_limit |
| 4 | 5,000 | 10% | Use perturbation_rounds / perturbation_strength to tune |
| 5 | 5,000 | 25% | Use perturbation_rounds / perturbation_strength to tune |

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