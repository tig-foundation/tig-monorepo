# TIG Code Submission

## Submission Details

* **Challenge Name:** energy_arbitrage
* **Algorithm Name:** harimox
* **Copyright:** 2026 tilk
* **Identity of Submitter:** tilk
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

## Target Tracks & Recommended Hyperparameters

| Track | Recommended Fuel | Recommended HP | Engine |
|---|---|---|---|
| s=baseline | 5T | `null` | voltpath (c008_a055) track_t1 |
| s=congested | 5T | `null` | wasserwert (c008_a054), new defaults |
| s=multiday | 5T | `null` | voltpath track_t3, new defaults |
| s=dense | 5T | `null` | wasserwert, new defaults |
| s=capstone | 5T | `null` | voltpath track_t5 |

The voltpath and wasserwert hyperparameters still apply to their engines.

Instances are routed by num_batteries with the voltpath thresholds (15 / 30 / 50 / 80 / 150). The engines are
faster and give the same solutions as voltpath / wasserwert on baseline and capstone. Defaults changed on
three tracks; these hyperparameters give the donor solutions:

- multiday: `{"dp_soc_cap":41,"ct_rounds":30,"ct_rebuild":5,"ct_replan":0,"blk_w":4,"blk_exp":1,"ct_fit_levels":0}`
- congested: `{"plan_rounds":8,"lp_seg":4}`
- dense: `{"plan_rounds":3,"lp_seg":4}`

## Results

Runs: `tig-runtime` and `tig-verifier` built with the c008 feature, the codes run on the same
nonces (192 on baseline and multiday, 96 on the others), 5T fuel, null HP. Quality is harimox minus the other
code, time is harimox time / other code time.

| Track | quality vs peakshift | quality vs gridflow | time vs peakshift | time vs gridflow |
|---|---|---|---|---|
| baseline | +0.027 | -0.054 | 0.77 | 1.18 |
| congested | +0.010 | -0.079 | 1.14 | 1.13 |
| multiday | +0.047 | -0.011 | 0.75 | 0.65 |
| dense | +0.077 | -0.038 | 1.91 | 1.81 |
| capstone | +0.013 | +0.056 | 0.56 | 0.57 |

The gain over gridflow is on capstone; quality is above peakshift on every track.

Compared with the donor engine of each track, harimox is about 2.9x faster on baseline, 9x on congested,
1.25x on multiday, 5x on dense and 3.4x on capstone.

## References and Acknowledgments

### Code References
- voltpath (c008_a055), NVX: baseline, multiday and capstone engines. Copyright remains with NVX.
- wasserwert (c008_a054), Kernelsmith: congested and dense engine. Copyright remains with Kernelsmith.

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
