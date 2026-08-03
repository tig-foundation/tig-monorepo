# TIG Code Submission

## Submission Details

* **Challenge Name:** knapsack
* **Algorithm Name:** combination_alg
* **Copyright:** 2026 alta
* **Identity of Submitter:** alta
* **Identity of Creator of Algorithmic Method:** alta
* **Unique Algorithm Identifier (UAI):** null

## References and Acknowledgments

### 1. Academic Papers
- Pisinger, *"The Quadratic Knapsack Problem -- a survey"*, Discrete Applied Mathematics 2007

### 2. Code References
- TIG `knap_quality_opt_v11` (`c003_a133`, copyright 2026 NVX). Its five
  track engines are included as private modules so this submission is
  independently compilable.

## Hyperparameters

`track` is the primary hyperparameter. Accepted values are `1000_5`,
`1000_10`, `5000_10`, `1000_25`, and `5000_25`; the corresponding full TIG
track strings such as `n_items=1000,budget=5` are also accepted. The default is
`5000_25`.

Each `track` selects a tuned preset and the implementation appropriate for that
track. Any concrete secondary hyperparameter supplied by the benchmarker
overrides the selected preset. Missing secondary values, JSON `null`, and the
string `"default"` retain the preset value.

The configured `track` must match the challenge instance. This catches a
misconfigured benchmark before it silently runs the wrong implementation.

All implementation files are private sibling modules in this directory. The
submission does not import or depend on another TIG algorithm package.

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
