# TIG Code Submission

## Submission Details

* **Challenge Name:** neuralnet_optimizer
* **Algorithm Name:** neural_sam_lion_v1
* **Copyright:** 2026 Nick Dunin
* **Identity of Submitter:** Nick Dunin (player_id `0x85Fb166deeec5c76fFD0cf3A74CCEc958a4E2711`)
* **Identity of Creator of Algorithmic Method:** SAM is from Foret et al., Google Research (2021). Lion is from Chen et al., Google Brain (2023). The SAM-Lion hybrid for the TIG c006 challenge — using `optimizer_query_at_params` for sign-perturbation SAM probes on alternating batches — is original work.
* **Unique Algorithm Identifier (UAI):** null

## References

- Foret, P., Kleiner, A., Mobahi, H., & Neyshabur, B. *"Sharpness-Aware Minimization for Efficiently Improving Generalization."* ICLR 2021. arXiv:2010.01412.
- Chen et al. *"Symbolic Discovery of Optimization Algorithms."* arXiv:2302.06675.

## Mechanism

```
batch N (vanilla):
    query_at_params returns None
    receive g(θ), save as g_last
    apply Lion update with g(θ)

batch N+1 (SAM probe, only if g_last exists):
    query_at_params returns θ + ε_sam · sign(g_last)
    harness computes g at perturbed point
    apply Lion update with g_perturbed
```

Sign-perturbation rather than full norm-normalised SAM avoids a reduction kernel; the spirit (ascent toward worst-case neighbourhood, then optimise) is preserved. On c006's noisy regression (σ=0.2 added Gaussian noise on RFF labels), flat-minima preference should improve generalisation to the held-out test set.

## Per-track learning rate

Same scheme as `neural_sophia_h_v1` and `neural_lion_per_track_v1` — multiplier ∈ {1.5, 1.2, 1.0, 0.8, 0.6} for tracks {4, 7, 10, 14, 18} respectively.

## Memory

2× param memory (momentum + g_last). Same as Adam/Sophia-G.

## Default hyperparameters

`lr = 3e-4`, `β₁ = 0.9`, `β₂ = 0.99`, `eps_sam = 5e-3`, `sam_period = 2`. All overridable.

## Why this might beat v4

Per the all-28 c006 recon, NO current submission uses `optimizer_query_at_params` for sharpness probing. v4's consensus blender doesn't touch the loss landscape's curvature except via the Fisher matrix on the GRADIENT (not the loss surface). SAM directly probes loss-surface sharpness in the worst-case direction; SAM-style training reliably produces flatter minima with better test-set generalization on noisy regression problems like c006.

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
