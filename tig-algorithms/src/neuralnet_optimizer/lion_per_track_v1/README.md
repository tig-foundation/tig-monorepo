# TIG Code Submission

## Submission Details

* **Challenge Name:** neuralnet_optimizer
* **Algorithm Name:** lion_per_track_v1
* **Copyright:** 2026 Nick Dunin
* **Identity of Submitter:** Nick Dunin (player_id `0x85Fb166deeec5c76fFD0cf3A74CCEc958a4E2711`)
* **Identity of Creator of Algorithmic Method:** Lion is from Chen et al., Google Brain (2023). Per-track hyperparameter tuning is original work for the TIG c006 challenge.
* **Unique Algorithm Identifier (UAI):** null

## References

- Chen et al. *"Symbolic Discovery of Optimization Algorithms."* arXiv:2302.06675

## Notes

Same Lion update rule as `neural_baseline_lion_v1`, but with **distinct (lr, β₁, β₂, weight_decay) tuples per track** detected from `param_sizes` shape on init. Per the all-28 c006 recon, per-track tuning is the dominant pattern of winning submissions — single track-agnostic configurations underperform.

Track-detection heuristic: count param tensors of length 256×256 (hidden→hidden weight matrices) → that count ≈ n_hidden + 1 for the c006 MLP.

| Track (n_hidden) | lr | β₁ | β₂ |
|---|---|---|---|
| 4 | 6e-4 | 0.92 | 0.985 |
| 7 | 4e-4 | 0.91 | 0.99 |
| 10 | 3e-4 | 0.9 | 0.99 |
| 14 | 2e-4 | 0.9 | 0.995 |
| 18 | 1.5e-4 | 0.88 | 0.995 |

All defaults overridable via `--hyperparameters` JSON.

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
