# TIG Code Submission

## Submission Details

* **Challenge Name:** vehicle_routing
* **Algorithm Name:** murelan_v2
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
  Update it when it is more than 0.003 from the line. A nonce stops once its quality reaches T plus the stop
  margin: 0.0045 on n_nodes 600, 800 and 1000 and 0.006 on 700 and 900. With the default T, which sits about
  0.0015 above the line, the margin is 0.0045 on every track. A stale value costs qualifiers both ways: above
  the line it gives up nonces that would qualify, below it stops nonces short of the line. A missing or
  non-numeric value takes the default, the expected line in rounds 144-145: the mean line over blocks
  1401433-1403264 plus 0.0015.

  | n_nodes | 600 | 700 | 800 | 900 | 1000 |
  |---|---|---|---|---|---|
  | default `target_quality` | 0.2045 | 0.1981 | 0.1936 | 0.1896 | 0.1920 |

- `early_stop`: 0 turns stopping off and gives the hgs_epure output; missing or any other value keeps it on.
- All hgs_epure hyperparameters still apply. Sending any of them gives the murelan output; the earlier stop of
  murelan_v2 applies only when no key other than `target_quality` and `early_stop` is sent.

## Changes from murelan

murelan (c002_a123) with an earlier stop at default settings and a faster local search on every track. A
bundle qualifies when its median nonce quality reaches T, so a nonce stops once it is clearly above T, or
once T is out of reach; at default settings a nonce that can no longer reach T stops sooner. Stopping only
depends on solution costs, so runs stay deterministic. With hgs_epure settings or `{"early_stop": 0}` the
solutions are murelan's.

## Results

murelan_v2 against murelan, same fresh nonces (48 per track), official tig-runtime and tig-verifier, 5T
fuel, qualifying lines of block 1403832. Qualifying bundles per unit of fuel relative to murelan, from the
bundle-median probabilities and total fuel on the paired runs, with the age factor of a newer code applied
(x0.9775). Default settings: each code's own defaults. Same `target_quality`: 0.2045 / 0.1981 / 0.1936 /
0.1896 / 0.1920 on both.

| n_nodes | 600 | 700 | 800 | 900 | 1000 |
|---|---|---|---|---|---|
| Nonces at or above the line, murelan_v2 / murelan | 7 / 7 | 15 / 15 | 7 / 7 | 14 / 14 | 4 / 4 |
| Qualifying bundles per fuel, default settings | 1.76 | 1.33 | 1.54 | 1.20 | 2.52 |
| Qualifying bundles per fuel, same `target_quality` | 1.44 | 1.24 | 1.36 | 1.10 | 1.47 |

Counts are also equal at the line plus and minus 0.001 on every track. At the line minus 0.0025, one nonce
on n_nodes=1000 reached it with murelan and not with murelan_v2.

murelan against hgs_epure (murelan README): 2.03 to 2.40 qualifiers per unit of CPU at default settings,
1.63 to 2.29 at the most common on-chain settings.

## References and Acknowledgments

This submission builds on hgs_prometheus: a hybrid genetic search for vehicle routing with
capacity and time window constraints, combining a population-based search with local search and
problem decomposition. It has no thread, no clock and no filesystem access; the search is
deterministic and anytime, and feasible solutions are reported through the TIG solution callback.

### Academic Papers

[1] Vidal, T., Crainic, T. G., Gendreau, M., and Prins, C. (2013). *A hybrid genetic algorithm with adaptive diversity management for a large class of vehicle routing problems with time windows*. Computers & Operations Research, 40(1), 475-489. https://doi.org/10.1016/j.cor.2012.07.018

[2] Vidal, T. (2022). *Hybrid genetic search for the CVRP: Open-source implementation and SWAP* neighborhood*. Computers & Operations Research, 140, 105643. https://doi.org/10.1016/j.cor.2021.105643

### Code References

- murelan (c002_a123), tilk.
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
