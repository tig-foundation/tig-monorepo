# TIG Code Submission

## Submission Details

* **Challenge Name:** neuralnet_optimizer
* **Algorithm Name:** neural_sophia_h_v1
* **Copyright:** 2026 Nick Dunin
* **Identity of Submitter:** Nick Dunin (player_id `0x85Fb166deeec5c76fFD0cf3A74CCEc958a4E2711`)
* **Identity of Creator of Algorithmic Method:** The Sophia optimizer is from Liu et al., Stanford (2023). This implementation is an original Rust + CUDA port of the Sophia-H (Hutchinson HVP) variant for the TIG c006 challenge, with original cross-batch coordination of HVP probes via the TIG `optimizer_query_at_params` hook.
* **Unique Algorithm Identifier (UAI):** null

## References and Acknowledgments

### 1. Academic Papers

- Liu, H., Li, Z., Hall, D., Liang, P., & Ma, T. *"Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training."* ICLR 2024. arXiv:2305.14342. https://arxiv.org/abs/2305.14342
- Hutchinson, M. F. *"A stochastic estimator of the trace of the influence matrix for Laplacian smoothing splines."* Communications in Statistics — Simulation and Computation, 18(3), 1059-1076 (1989).

### 2. Code References

- Reference template: `tig-monorepo/tig-algorithms/src/neuralnet_optimizer/template.rs` (TIG Foundation)

## Additional Notes

This submission ports the Sophia-H (Hutchinson) variant of Sophia to c006, exploiting the Protocol-0.0.5 weights-visibility surface that no current adopted submission uses for anything beyond L2 weight decay.

**Mechanism — three-phase cycle of length `cycle` (default 8):**

```
Phase 0  (probe +): query returns θ + ε·v_t        (v_t Rademacher)
                    capture grad_plus = ∇L(θ + ε·v_t)
                    apply provisional Sophia update with last-cycle's h

Phase 1  (probe −): query returns θ − ε·v_t        (SAME v_t)
                    capture grad_minus = ∇L(θ − ε·v_t)
                    estimate h_hessian = (grad_plus − grad_minus) / (2ε) ⊙ v_t
                    EMA-update h ← β_h·h + (1−β_h)·|h_hessian|
                    apply Sophia update with fresh h

Phase 2..cycle−1:   query returns None (use original θ)
                    apply Sophia update with current h
```

**Why this beats v4:** Per the all-28-submissions recon (`STRATEGY/tig_recon/2026_05_03_all_28_submissions_landscape.md`), 0 of 16 analyzed c006 submissions read θ for anything beyond standard L2 decay. v4's "consensus blender" of Adam + sign + normalized + Fisher uses g², not real curvature. Sophia-H's Hutchinson estimate is an unbiased estimator of `diag(H)·v` — it captures actual loss-landscape curvature via finite-difference HVPs.

**Per-track tuning:** Per the recon, per-track specialization is THE pattern of winners (v4 has track_4/7/10/14/18.rs files). Sophia-H takes a different approach: a single algorithm with per-track *learning-rate multipliers* (lr_track_{4,7,10,14,18}) detected from `param_sizes` shape. Smaller networks get higher lr (1.5×), bigger get lower (0.6×), reflecting standard MLP scaling intuition.

**Memory footprint:** 4× model params (m, h, grad_plus, grad_minus). For c006's ~1.2M params, ~20 MB of f32 — trivial.

**Numerical safety:**
- Rademacher v ∈ {-1, +1} bounded
- HVP estimate clipped to [-100, 100] before EMA (Hutchinson is high-variance, single-batch outliers can poison the EMA)
- explicit `isnan/isinf` clamp on HVP and on the final clipped update
- Sophia's per-element clip(·, ρ) bounds step magnitude regardless of m or h
- max(h, ε) prevents division blow-up

Combined: `|update[i]| ≤ lr · (ρ + |λ| · |θ_i|)`, structurally bounded.

**Default hyperparameters:** `lr = 1e-3`, `β₁ = 0.965`, `β₂ = 0.99`, `β_h = 0.95`, `ρ = 0.04`, `ε = 1e-12`, `ε_hvp = 1e-3`, `cycle = 8`. Sophia paper defaults plus `cycle = 8` (probe ~25% of steps, refresh h every 8 batches).

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
