# TIG Code Submission

## Submission Details

* **Challenge Name:** neuralnet_optimizer
* **Algorithm Name:** neural_sophia_g_v1
* **Copyright:** 2026 Nick Dunin
* **Identity of Submitter:** Nick Dunin (player_id `0x85Fb166deeec5c76fFD0cf3A74CCEc958a4E2711`)
* **Identity of Creator of Algorithmic Method:** The Sophia optimizer is from Liu et al., Stanford (2023). This implementation is an original Rust + CUDA port of the Sophia-G (Gauss-Newton-Bartlett) variant for the TIG c006 challenge.
* **Unique Algorithm Identifier (UAI):** null

## References and Acknowledgments

### 1. Academic Papers

- Liu, H., Li, Z., Hall, D., Liang, P., & Ma, T. *"Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training."* ICLR 2024. arXiv:2305.14342. https://arxiv.org/abs/2305.14342

### 2. Code References

- Reference template: `tig-monorepo/tig-algorithms/src/neuralnet_optimizer/template.rs` (TIG Foundation)

### 3. Other

None.

## Additional Notes

This submission ports the Sophia-G (Gauss-Newton-Bartlett) variant of Sophia to the c006 Neural Net Optimizer challenge.

**Mechanism (per element):**

```
m       = β₁ · m_old + (1 − β₁) · g
h       = β₂ · h_old + (1 − β₂) · g²            (GNB diagonal Hessian estimate)
precond = m / max(h, ε)
clipped = clamp(precond, −ρ, +ρ)
θ      −= lr · (clipped + λ · θ)
```

**Track strategy:** single track-agnostic step function. Sophia-G adapts per-element automatically through the curvature estimate `h`, so no manual per-track tuning is needed. Per-track learning-rate overrides are exposed as hyperparameters for future iterations.

**Default hyperparameters:** `lr = 1e-3`, `β₁ = 0.965`, `β₂ = 0.99`, `ρ = 0.04`, `ε = 1e-12`, `weight_decay = 0.0`. The Sophia paper's GPT-2 defaults adjusted for the small-batch (128) regression target on c006's 256-wide MLP.

**Memory footprint:** 2× model params (m + h, same as AdamW).

**Numerical safety:** four guards layered together — `max(h, ε)` prevents division blow-up, `clamp(precond, −ρ, +ρ)` bounds the preconditioned step magnitude, an explicit `isnan/isinf` clamp on the result handles edge cases, and the final `lr` multiplier scales the entire bounded update. Result: `|update[i]| ≤ lr · (ρ + |λ| · |θ_i|)`, which is a structural guarantee independent of gradient or Hessian values.

**Variant choice (Sophia-G vs Sophia-H):** This is Sophia-G. The Sophia-H variant uses Hutchinson's Hessian-vector product estimator for true second-order curvature, which would more directly exploit the Protocol-0.0.5 weights-visibility surface. We ship Sophia-G first because Sophia-H requires coordinating perturbed-parameter injections across consecutive batches via the `optimizer_query_at_params` hook, which is implementation-risky for a v1 submission. If this v1 lands at ≥5% adoption, we level up to Sophia-H or K-FAC for v2.

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
