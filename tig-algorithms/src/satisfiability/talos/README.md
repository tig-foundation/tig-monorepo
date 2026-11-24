# Talos — TIG Code Submission

## Submission Details

* **Challenge Name:** satisfiability
* **Algorithm Name:** talos
* **Copyright:** 2026 zhenglcc; upstream copyrights retained, including tilk, NVX, ChervovNikita, Fred and testing
* **Identity of Submitter:** zhenglcc
* **Identity of Creator of Algorithmic Method:** zhenglcc
* **Unique Algorithm Identifier (UAI):** null

## Supported Tracks and Hyperparameters

Derivative SAT solver; upstream authorship and copyrights are retained below.

| Track | Recommended hyperparameters |
|---|---|
| n_vars=5000,ratio=4267 | {"max_fuel_high":400000000000} |
| n_vars=7500,ratio=4267 | {"max_fuel_high":400000000000} |
| n_vars=10000,ratio=4267 | {"max_fuel_high":125000000000} |
| n_vars=100000,ratio=4150 | null |
| n_vars=100000,ratio=4200 | null |

Use `null` or `{}` for the recommended defaults. A nonempty object is
forwarded without merging defaults; include `max_fuel_high` when adding
parameters on T1/T2/T3 to preserve the recommended budget. T4/T5 use fixed
settings and ignore HP.

### HP parameters

| HP | Type | Applies to | Default / meaning |
|---|---|---|---|
| `max_fuel_high` | Nonnegative number | T1/T2/T3 | T1/T2: 400000000000; T3: 125000000000. Search budget estimate. |
| `check_interval` | Integer >=1 | T1 compatibility, T3 | Automatic, about 97 at the standard density. Flips between progress checks. |
| `stagnation_limit` | Unsigned integer | T1 compatibility | 3; non-progress checks before perturbation. |
| `max_reinits` | Unsigned integer | T1 compatibility | 15; maximum reinitializations, 0 disables them. |
| `sel_variant` | Unsigned integer | T1 compatibility | 2; selector variant: 0 control, 1/2 individual variants, >=3 combined. |
| `probe_max_flips` | Unsigned integer | T1 compatibility | Unset; overrides this kernel's estimated flip budget. |
| `t4_fuel_b` | Unsigned integer | T1 complement | 50000000000; first complement-stage budget. |

### T1 parameter routing

T1 keeps its normal portfolio with `null`, `{}`, or only a valid
`max_fuel_high`. Adding other keys selects compatibility processing.
An object containing only `t4_fuel_b` selects the complement plus cached fallback.

### Defaults for nonempty partial objects

| Track | If a nonempty object omits `max_fuel_high` |
|---|---|
| T1 compatibility or `t4_fuel_b`-only cached fallback | 160000000000 in the main kernel |
| T2 | 250000000000 |
| T3 | 180000000000 for the primary search; 125000000000 for the conditional rescue |
| T4/T5 | HP object ignored |

### HP examples

Pass the following JSON value as HP for the selected Track.

**All Tracks: use the recommended defaults**

```json
null
```

**T1: compatibility settings**

```json
{
  "max_fuel_high": 400000000000,
  "check_interval": 100,
  "stagnation_limit": 3,
  "max_reinits": 15,
  "sel_variant": 2,
  "t4_fuel_b": 50000000000
}
```

**T1: bounded flip-budget example**

```json
{
  "max_fuel_high": 400000000000,
  "probe_max_flips": 1000000,
  "sel_variant": 0,
  "t4_fuel_b": 1000000
}
```

**T3: check interval**

```json
{
  "max_fuel_high": 125000000000,
  "check_interval": 100
}
```

View the same parameter overview from the official developer environment:
```sh
help_algorithm talos
```

This submission contains README.md and seven Rust source files.

## Code attribution

`tekinon` / c001_a135, copyright 2026 tilk: reinforced survey propagation
and probSAT code used by the large-instance solvers.
https://github.com/tig-foundation/tig-monorepo/tree/satisfiability/tekinon/tig-algorithms/src/satisfiability/tekinon

## References and Acknowledgments

### Academic Papers

1. H. Fu, W. Zhang, G. Wu, Y. Xu and J. Liu, *[Improving stochastic local search for uniform k-SAT by generating appropriate initial assignment](https://doi.org/10.1111/coin.12438)*, Computational Intelligence 37(4), 1706–1744 (2021).
2. Adrian Balint and Uwe Schöning, *[Choosing Probability Distributions for Stochastic Local Search and the Role of Make versus Break](https://link.springer.com/chapter/10.1007/978-3-642-31612-8_3)*, SAT 2012, LNCS 7317, 16–29.
3. Alfredo Braunstein, Marc Mézard and Riccardo Zecchina, *[Survey propagation: an algorithm for satisfiability](https://arxiv.org/abs/cs/0212002)*, Random Structures & Algorithms 27, 201–226 (2005).
4. Joel Chavas, Cyril Furtlehner, Marc Mézard and Riccardo Zecchina, *[Survey-propagation decimation through distributed local computations](https://arxiv.org/abs/cond-mat/0512002)*, Journal of Statistical Mechanics: Theory and Experiment (2005), P11016; DOI: [10.1088/1742-5468/2005/11/P11016](https://doi.org/10.1088/1742-5468/2005/11/P11016).

### Code References

- [flipwave / c001_a130](https://github.com/tig-foundation/tig-monorepo/tree/c7c44b1510faeffb84f11031d86d225bbe397945/tig-algorithms/src/satisfiability/flipwave), copyright 2026 NVX: preprocessing, survey propagation and local-search code.
- `sat_tailwalk_v7` / c001_a128, zhenglcc: composite solver structure.

| Public source | Author / upstream attribution | Reused contribution |
|---|---|---|
| [sat_tailwalk_v6 / c001_a114](https://github.com/tig-foundation/tig-monorepo/tree/6a4da2d7da002bea33e8f3b8ec3fe794761e1d41/tig-algorithms/src/satisfiability/sat_tailwalk_v6) | zhenglcc | Dispatcher and preprocessing. |
| [sat_exact / c001_a125](https://github.com/tig-foundation/tig-monorepo/tree/b3c056ad909a65243314cbae24347cc4d0e9bdd6/tig-algorithms/src/satisfiability/sat_exact) | ChervovNikita | Cached SAT helpers and kernels. |
| [sat_hybrid_v3 / c001_a115](https://github.com/tig-foundation/tig-monorepo/tree/84125f66ca7b8a96430a635983aa97a31e913acc/tig-algorithms/src/satisfiability/sat_hybrid_v3) | NVX | Compatibility SAT kernel. |
| [sat_imp_giveup2 / c001_a112](https://github.com/tig-foundation/tig-monorepo/tree/4a0404680950bb11f97a0f1c8e19c4747bd16e9a/tig-algorithms/src/satisfiability/sat_imp_giveup2) | ChervovNikita | Local-search code. |
| [sat_hybrid_v4 / c001_a118](https://github.com/tig-foundation/tig-monorepo/tree/d2f974477490a028c5e411ffc9c303315dbc16d2/tig-algorithms/src/satisfiability/sat_hybrid_v4) | NVX | Complementary SAT and SP kernels. |
| [sat_hybrid_sp128 / c001_a123](https://github.com/tig-foundation/tig-monorepo/tree/c681d549498d6ddb35dd66861d11da29ff4b9e67/tig-algorithms/src/satisfiability/sat_hybrid_sp128) | Fred; donor copyright NVX | Reinforced SP seeding and local search. |
| [sat_imp_v5 / c001_a119](https://github.com/tig-foundation/tig-monorepo/tree/6fe5e28f5597936c9b3300920bc724519c4ac606/tig-algorithms/src/satisfiability/sat_imp_v5) | testing | Scoring and fused scans. |
| [sat_hybrid / c001_a109](https://github.com/tig-foundation/tig-monorepo/tree/c8fe46d5ad5574ae6eb17ed5d6c3e829da01eb27/tig-algorithms/src/satisfiability/sat_hybrid) | NVX | Hybrid SAT engine ancestry. |

* [sat_imp_v4](https://github.com/tig-foundation/tig-monorepo/tree/b8a95b0d9723ac2e12957830d22dd08293fd76bc/tig-algorithms/src/satisfiability/sat_imp_v4)
* [sat_vanguard_v9](https://github.com/tig-foundation/tig-monorepo/tree/6717cd2e740e664f22da06089fbc252b666ce4cb/tig-algorithms/src/satisfiability/sat_vanguard_v9)

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
