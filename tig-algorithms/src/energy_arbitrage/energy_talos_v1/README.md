# TIG Code Submission

## Submission Details

* **Challenge Name:** energy_arbitrage
* **Algorithm Name:** energy_talos_v1
* **Copyright:** 2026 zhenglcc; upstream copyrights retained, including tilk, NVX and Kernelsmith
* **Identity of Submitter:** zhenglcc
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

## Supported Tracks and Hyperparameters

The entry point selects the engine automatically. Use `null` or `{}` for the built-in defaults.

| Track | Setting | Recommended HP |
|---|---|---|
| T1 | `s=baseline` | `null` |
| T2 | `s=congested` | `null` |
| T3 | `s=multiday` | `null` |
| T4 | `s=dense` | `null` |
| T5 | `s=capstone` | `null` |

### HP examples

**All Tracks: built-in defaults**

```json
null
```

**T2 (`s=congested`): explicit default configuration**

```json
{
  "grid": 28,
  "sdp_q": 5,
  "sdp_qj": 2,
  "cong_prem": 1,
  "lp_seg": 1,
  "samples": 4,
  "gate_k": 8
}
```

For T2, supplied keys override the corresponding defaults; omitted keys keep their default values.

## Usage

In the official C008 developer environment:

```sh
build_algorithm energy_talos_v1
help_algorithm energy_talos_v1
```

Keep all nine Rust files together with their original filenames, including `mod.rs`.

## References and Acknowledgments

- tilk, harimox (c008_a057): parent portfolio and routing.
  https://github.com/tig-foundation/tig-monorepo/tree/371432bbd66402d14ab742e4aea2480d40d05011/tig-algorithms/src/energy_arbitrage/harimox
- NVX, voltpath (c008_a055): T1/T3/T5 engine foundations.
- Kernelsmith, wasserwert (c008_a054): T2/T4 engine foundations.

Upstream authorship and copyrights remain with their respective owners.

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
