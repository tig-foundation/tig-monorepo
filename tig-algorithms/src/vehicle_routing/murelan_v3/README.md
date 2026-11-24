# TIG Code Submission

## Submission Details

* **Challenge Name:** vehicle_routing
* **Algorithm Name:** murelan_v3
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
  margin: 0.0045 on n_nodes 600, 800 and 1000 and 0.006 on 700 and 900. Without a numeric target, the stop
  margin is 0.002 on every track, or 0.0035 with other search keys. A stale value costs qualifiers both ways: above
  the line it gives up nonces that would qualify, below it stops nonces short of the line. A missing or
  non-numeric value takes the default: the mean line over blocks 1401433-1403264 plus 0.0015.

  | n_nodes | 600 | 700 | 800 | 900 | 1000 |
  |---|---|---|---|---|---|
  | default `target_quality` | 0.2045 | 0.1981 | 0.1936 | 0.1896 | 0.1920 |

- `early_stop`: 0 turns stopping off; missing or any other value keeps it on.
- All hgs_epure hyperparameters still apply. The earlier stop applies only when no key other than
  `target_quality` and `early_stop` is sent.

## Changes from murelan_v2

murelan_v2 (c002_a125) with a faster local search and a different choice of capacity neighbours for SWAP*:
within the last demand tier that fills the list, the closest customers are kept instead of the first by ID.
Without a numeric `target_quality` the stop margin is 0.0025 lower than murelan_v2's; with one, keys,
defaults and stopping are those of murelan_v2. Runs stay deterministic; solutions differ from
murelan_v2's.

## Results

murelan_v3 against murelan_v2, same fresh nonces (144 per track, 96 on n_nodes=900), official tig-runtime
and tig-verifier, 5T fuel, qualifying lines of block 1426146. Qualifying bundles per unit of fuel relative
to murelan_v2, from the bundle-median probabilities and total fuel on the paired runs, with the age factor
of a newer code applied (x0.977); 95% bootstrap intervals in brackets. Both codes ran the same settings, sent
on chain on each track: `{}` on 700 and 800; on 1000 `{"allow_swap3": true,
"exploration_level": 4}`; on 600 and 900 that plus `"decomp_target_size": 200,
"max_credit_deterioration"` 20 / 40, `"decomp_nb_phases"` 8 / 30, `"granularity"` 30 / 40,
`"granularity2"` 40 / 50 and `"swapstar_capa_filter"` 0.5 / 0.6.

| n_nodes | 600 | 700 | 800 | 900 | 1000 |
|---|---|---|---|---|---|
| Nonces at or above the line, murelan_v3 / murelan_v2 | 38 / 35 | 32 / 34 | 34 / 34 | 16 / 16 | 4 / 5 |
| Fuel, murelan_v3 / murelan_v2 | 0.908 | 0.836 | 0.901 | 0.874 | 0.966 |
| Qualifying bundles per fuel | 2.33 [1.08, 6.93] | 0.98 [0.50, 1.59] | 1.08 [0.39, 3.12] | 1.12 [0.75, 1.93] | 0.52 [0.04, 1.07] |

murelan_v2 against murelan (murelan_v2 README): 1.20 to 2.52 qualifying bundles per fuel at default
settings.

## References and Acknowledgments

This submission builds on hgs_prometheus: a hybrid genetic search for vehicle routing with
capacity and time window constraints, combining a population-based search with local search and
problem decomposition. It has no thread, no clock and no filesystem access; the search is
deterministic and anytime, and feasible solutions are reported through the TIG solution callback.

### Academic Papers

[1] Vidal, T., Crainic, T. G., Gendreau, M., and Prins, C. (2013). *A hybrid genetic algorithm with adaptive diversity management for a large class of vehicle routing problems with time windows*. Computers & Operations Research, 40(1), 475-489. https://doi.org/10.1016/j.cor.2012.07.018

[2] Vidal, T. (2022). *Hybrid genetic search for the CVRP: Open-source implementation and SWAP* neighborhood*. Computers & Operations Research, 140, 105643. https://doi.org/10.1016/j.cor.2021.105643

### Code References

- murelan_v2 (c002_a125) and murelan (c002_a123), tilk.
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
