# TIG Code Submission

## Submission Details

* **Challenge Name:** energy_arbitrage
* **Algorithm Name:** harimox_v2
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
| s=dense | 5T | `null` | wasserwert, new defaults, faster LP dispatch |
| s=capstone | 5T | `{"target_quality": T}` | voltpath track_t5, quality target |

The voltpath and wasserwert hyperparameters still apply to their engines.

`target_quality` (capstone only): T = the lowest value of `qualifier_qualities_by_track["s=capstone"]` in
`/get-challenges` divided by 1e6 (3806936 on chain is 3.806936). Default 3.85, taken from the line at block
1404249 (3.807) plus the expected rise once capstone runs on this code. Stop margin 1.0 below T. A missing key uses the
default; 0 turns the target off and gives the harimox solutions. Update T when the line moves by more than
0.1. A stale value costs qualifiers both ways: above the line it gives up nonces that would qualify, below it
spends full effort on nonces that would qualify anyway.

Capstone: early in the run each nonce estimates its final quality. A nonce clearly above or below T finishes with
a lighter setting, a nonce far below T (by more than the stop margin) stops early, and nonces near T run in
full as in harimox, so the bundle median near T is unchanged. Runs stay deterministic.

Instances are routed by num_batteries with the voltpath thresholds (15 / 30 / 50 / 80 / 150). The engines are
faster and give the same solutions as voltpath / wasserwert on baseline. Defaults changed on
three tracks; these hyperparameters give the donor solutions:

- multiday: `{"dp_soc_cap":41,"ct_rounds":30,"ct_rebuild":5,"ct_replan":0,"blk_w":4,"blk_exp":1,"ct_fit_levels":0}`
- congested: `{"plan_rounds":8,"lp_seg":4}`
- dense: `{"plan_rounds":3,"lp_seg":4}`; solutions can differ through the LP solver.

Dense uses a faster LP dispatch with the original LP as fallback; quality is unchanged and schedules stay
deterministic.

## Results

Runs: `tig-runtime` and `tig-verifier` built with the c008 feature, the codes run on the same nonces, 5T fuel,
null HP. Quality is harimox_v2 minus the other code, time is the runtime-plus-verifier CPU ratio. Baseline,
congested and multiday give the harimox solutions (192 nonces on baseline and multiday, 96 on congested).

| Track | quality vs peakshift | quality vs gridflow | time vs peakshift | time vs gridflow |
|---|---|---|---|---|
| baseline | +0.027 | -0.054 | 0.77 | 1.18 |
| congested | +0.010 | -0.079 | 1.14 | 1.13 |
| multiday | +0.047 | -0.011 | 0.75 | 0.65 |
| dense | +0.062 | -0.038 | 1.75 | 1.71 |
| capstone | see below | see below | 0.50 | 0.51 |

Dense: 192 nonces, quality equal to harimox on every nonce, CPU 0.944 of harimox.

Capstone, T 3.85, line 3.804, against harimox: 576 nonces, 259 at or above the line for both codes (also
equal at the line -0.05 and +0.05), no nonce crosses the line, CPU 0.90 of harimox on 192 timed pairs. harimox
capstone quality is +0.013 over peakshift and +0.056 over gridflow at 0.56 / 0.57 of their time; the capstone
times above are those ratios times 0.90.

The gain over gridflow is on capstone; quality is above peakshift on every track.

Compared with the donor engine of each track, harimox_v2 is about 2.9x faster on baseline, 9x on congested,
1.25x on multiday and 5.3x on dense.

## References and Acknowledgments

### Code References
- energy_v1 capstone baseline policy (tig-challenges energy_arbitrage/baselines/energy_v1/capstone.rs), TIG
  Foundation: used to estimate quality on capstone.
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
