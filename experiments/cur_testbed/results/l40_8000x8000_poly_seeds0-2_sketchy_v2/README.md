# L40 8000x8000 sketchy_v2 three-seed run

This is the audited result for C3 job `job_1788518486289_9o66c5`, run
sequentially on one NVIDIA L40 on September 4, 2026. Each seed is one CUR
challenge instance containing eight 8000x8000 subinstances. The timed wall
clock starts after CUDA warm-up and covers generation, solving, and canonical
fast-`U` verification.

The `sketchy_v2` defaults were `baseline_trials=3`, `quality_trials=10`,
`sketch_extra=64`, `power_iters=4`, `maxvol_swaps=48`,
`maxvol_tolerance=1.001`, and `crossover_pool=10`.

| Seed | TIG integer quality | Mean score | Generation | Solve | Verify | End-to-end wall clock |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 714,037 | 0.714037098 | 0.299 s | 122.658 s | 0.217 s | 123.174 s |
| 1 | 718,983 | 0.718983139 | 0.303 s | 85.591 s | 0.197 s | 86.092 s |
| 2 | 721,528 | 0.721527643 | 0.303 s | 147.215 s | 0.232 s | 147.752 s |
| Mean | 718,182.7 | 0.718182627 | 0.302 s | 118.488 s | 0.215 s | 119.006 s |

The complete C3 job took 7m54s including dependency installation, compilation,
three separate CUDA warm-ups, and artifact finalization. That job-level time is
not an individual challenge runtime.

