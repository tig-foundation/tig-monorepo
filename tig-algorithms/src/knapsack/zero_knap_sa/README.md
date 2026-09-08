# TIG Code Submission

## Submission Details

* **Challenge Name:** knapsack
* **Algorithm Name:** zero_knap_sa
* **Copyright:** 2026 AgentZero
* **Identity of Submitter:** AgentZero
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

## References and Acknowledgments

### 1. Academic Papers
- Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. "Optimization by Simulated Annealing." Science 220, 671-680 (1983).

### 2. Code References
- superfast_knap_v1 — TIG merged code submission (github.com/tig-foundation/tig-monorepo). Track 1/3/4/5 pipelines and the track2 base pipeline are vendored from this submission with credit.

## Additional Notes

Self-contained bundle. track2 adds a staged, fuel-gated multi-seed search over
the superfast pipeline with Metropolis simulated-annealing acceptance in the
deep-polish phase (default sa_t0_bp=100, sa_decay_pm=970). Improvements are
persisted immediately via the save callback; effort is bounded by the runtime
fuel counter.

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
