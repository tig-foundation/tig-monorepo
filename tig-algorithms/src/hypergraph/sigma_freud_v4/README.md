# TIG Code Submission

## Submission Details

* **Challenge Name:** hypergraph
* **Algorithm Name:** sigma_freud_v4
* **Copyright:** 2026 Rootz
* **Identity of Submitter:** Rootz
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

## Algorithm Description

**sigma_freud_v4** is a GPU-accelerated hypergraph partitioning algorithm that combines multiple optimization techniques:

### Key Features

- **Track-specific tuning** - Optimized for problem sizes from 10K to 200K hyperedges

### Hyperparameters

The algorithm uses an **effort-based system** for easy tuning:

```
{"effort": 0}  # Fastest (300 refinement rounds, 3 ILS iterations)
{"effort": 2}  # Default balanced setting
{"effort": 5}  # Highest quality (500 refinement rounds, 5 ILS iterations)
```

**Advanced parameters** (override effort defaults):
- `{"refinement": 750}` - Main refinement iterations (50-5000)
- `{"ils_iterations": 8}` - ILS restarts (1-10)
- `{"post_ils_polish": 100}` - Polishing rounds after ILS (20-200)
- `{"post_refinement": 80}` - Final balance passes (0-128)
- `{"move_limit": 200000}` - Max moves per round (256-1000000, auto-adaptive)
- `{"clusters": 64}` - Hyperedge clusters (4-256, must be multiple of 4)

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
