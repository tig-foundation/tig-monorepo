# TIG Code Submission

## Submission Details

* **Challenge Name:** vehicle_routing
* **Algorithm Name:** murelan
* **Copyright:** 2026 tilk
* **Identity of Submitter:** tilk
* **Identity of Creator of Algorithmic Method:** Thibaut Vidal
* **Unique Algorithm Identifier (UAI):** c002_a110

## Target Tracks & Recommended Hyperparameters

| Track | Recommended Fuel | Recommended HP |
|---|---|---|
| n_nodes=600 | 5T | `{"target_quality": T}` |
| n_nodes=700 | 5T | `{"target_quality": T}` |
| n_nodes=800 | 5T | `{"target_quality": T}` |
| n_nodes=900 | 5T | `{"target_quality": T}` |
| n_nodes=1000 | 5T | `{"target_quality": T}` |

- `target_quality`: the qualifying quality T of the track, as a fraction. T is the lowest value of
  `qualifier_qualities_by_track[<track>]` in `/get-challenges` divided by 1e6 (199543 on chain is 0.199543).
  Update it when it is more than 0.004 from the line. A stale value costs qualifiers both ways: above the
  line it gives up nonces that would qualify, below it stops nonces short of the line. A missing or
  non-numeric value takes the default, the expected line in rounds 143-145: the mean line over blocks
  1396750-1397626 plus 0.0025.

  | n_nodes | 600 | 700 | 800 | 900 | 1000 |
  |---|---|---|---|---|---|
  | default `target_quality` | 0.2020 | 0.1956 | 0.1905 | 0.1871 | 0.1862 |

- `early_stop`: 0 turns stopping off and gives the hgs_epure output; missing or any other value keeps it on.
- All hgs_epure hyperparameters still apply.

## Changes from hgs_epure

hgs_epure (c002_a120) with early stopping. A bundle qualifies when its median nonce quality reaches the
qualifying quality T of the track, so between decomposition phases a nonce stops once it is clearly above T
or can no longer reach it. Stopping only depends on solution costs, so runs stay deterministic.

## Results

Qualifiers per unit of CPU time relative to hgs_epure at the same settings, 0.975 age factor applied to
murelan, qualifying line at the default `target_quality`. Default settings: 360 nonces per track.
On-chain settings: the most common hgs_epure settings precommitted for the track (two on n_nodes=600 and 900, one on 800, three on 1000, none on 700), 100 nonces per setting, range
over the settings. No nonce that reached the line with hgs_epure fell below it. The gain is on every
track, largest on n_nodes=1000 at default settings.

| n_nodes | 600 | 700 | 800 | 900 | 1000 |
|---|---|---|---|---|---|
| murelan / hgs_epure, default settings | 2.19 | 2.19 | 2.03 | 2.30 | 2.40 |
| murelan / hgs_epure, on-chain settings | 2.10-2.18 | - | 1.81 | 1.63-1.89 | 1.81-2.29 |

## References and Acknowledgments

This submission builds on hgs_prometheus: a hybrid genetic search for vehicle routing with
capacity and time window constraints, combining a population-based search with local search and
problem decomposition. It has no thread, no clock and no filesystem access; the search is
deterministic and anytime, and feasible solutions are reported through the TIG solution callback.

### Academic Papers

[1] Vidal, T., Crainic, T. G., Gendreau, M., and Prins, C. (2013). *A hybrid genetic algorithm with adaptive diversity management for a large class of vehicle routing problems with time windows*. Computers & Operations Research, 40(1), 475-489. https://doi.org/10.1016/j.cor.2012.07.018

[2] Vidal, T. (2022). *Hybrid genetic search for the CVRP: Open-source implementation and SWAP* neighborhood*. Computers & Operations Research, 140, 105643. https://doi.org/10.1016/j.cor.2021.105643

### Code References

- hgs_epure (c002_a120), FP Labs. Copyright remains with FP Labs.
- TIG baseline (vehicle_routing/baselines/solomon.rs).
- ember_stromboli (job_scheduling) also stops its search once the result is good enough.

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
