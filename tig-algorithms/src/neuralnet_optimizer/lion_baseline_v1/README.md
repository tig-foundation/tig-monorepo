# TIG Code Submission

## Submission Details

* **Challenge Name:** neuralnet_optimizer
* **Algorithm Name:** lion_baseline_v1
* **Copyright:** 2026 Nick Dunin
* **Identity of Submitter:** Nick Dunin (player_id `0x85Fb166deeec5c76fFD0cf3A74CCEc958a4E2711`)
* **Identity of Creator of Algorithmic Method:** The Lion optimizer is from Chen et al., Google Brain (2023). This implementation is an original Rust + CUDA port for the TIG c006 challenge.
* **Unique Algorithm Identifier (UAI):** null

## References and Acknowledgments

### 1. Academic Papers

- Chen, X., Liang, C., Huang, D., Real, E., Wang, K., Liu, Y., Pham, H., Dong, X., Luong, T., Hsieh, C.-J., Lu, Y., & Le, Q. V. *"Symbolic Discovery of Optimization Algorithms."* NeurIPS 2023. arXiv:2302.06675. https://arxiv.org/abs/2302.06675

### 2. Code References

- Reference template: `tig-monorepo/tig-algorithms/src/neuralnet_optimizer/template.rs` (TIG Foundation)

### 3. Other

None.

## Additional Notes

This submission ports the Lion (Evolved Sign Momentum) optimizer to the c006 Neural Net Optimizer challenge interface.

**Mechanism (per element):**

```
blend  = β₁ · m_{t−1} + (1 − β₁) · g_t
update = lr · (sign(blend) + λ · θ_{t−1})
θ_t    = θ_{t−1} − update
m_t    = β₂ · m_{t−1} + (1 − β₂) · g_t
```

**Track strategy:** single track-agnostic step function. Lion's update is invariant to layer count by construction (element-wise + sign), so the same kernel runs for `num_hidden_layers ∈ {4, 7, 10, 14, 18}`. Per-track learning-rate tuning is exposed as a hyperparameter for future iterations but defaults to the paper's values for v1.

**Default hyperparameters:** `lr = 3e-4`, `β₁ = 0.9`, `β₂ = 0.99`, `weight_decay = 0.0`. These are the Lion-paper defaults with `lr` lifted from 1e-4 to 3e-4 for the small-batch (128) regression target.

**Memory footprint:** 1× model params (vs Adam's 2× — no second moment needed).

**Numerical safety:** the sign(·) function clamps NaN and Inf inputs to 0, making the update bounded (`|update[i]| ≤ lr · (1 + |λ| · |θ_i|)`). This is a structural NaN-impossibility property, not a runtime guard.

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
