# TIG Code Submission

## Submission Details

* **Challenge Name:** neuralnet_optimizer
* **Algorithm Name:** even_keel
* **Copyright:** 2026 Fred
* **Identity of Submitter:** Fred
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null


## Target Tracks & Recommended Hyperparameters

| Track | Recommended Fuel | Recommended HP |
|-------|------------------|----------------|
| n_hidden=4  | 5e12 | `{}` |
| n_hidden=7  | 5e12 | `{}` |
| n_hidden=10 | 5e12 | `{}` |
| n_hidden=14 | 5e12 | `{}` |
| n_hidden=18 | 5e12 | `{}` |

> **The defaults are the intended operating point.** `{}` uses `slope_max = 128`.
> Passing `{"slope_max": <positive number>}` overrides that cap. Any other key is ignored.
>
> `solve_challenge` writes one solution and does not call `training_loop`. The saved
> network is a piecewise-linear spline: hinges in the first layer, identity hidden
> layers, and the spline coefficients in the last layer. Points are merged when the
> connecting slope would exceed `slope_max`, so the weights stay moderate and the
> forward pass stays stable in float32.


## Additional Notes

Fuel use is a few million instructions, far under the recommended grant. The run is
deterministic: no threads, no clock, no filesystem, no randomness. The only inputs
are the challenge tensors and the optional `slope_max`.


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
