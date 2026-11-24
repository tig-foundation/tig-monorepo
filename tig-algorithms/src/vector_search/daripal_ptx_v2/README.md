# TIG Code Submission

## Submission Details

* **Challenge Name:** vector_search
* **Algorithm Name:** daripal_ptx_v2
* **Copyright:** 2026 ubelai
* **Identity of Submitter:** ubelai
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

## Method Overview

The kernels extend `c004_a111` (`daripal`) with `lop3.b32` selection, immediate-offset
shared-memory WMMA B-loads, and a hoisted thread offset in global loads, retaining
the fp16 screening and exact fp32 reranking trajectory. Four kernel handles are
cached strongly by module identity (`Weak<CudaModule>` + `Arc::ptr_eq`), bounded to
8 entries with explicit CUDA context rebinding on eviction, and eight thread-local
GPU buffers grow only as required with per-nonce merge counter zeroing.

### Code References
- TIG `c004` public submissions (`daripal` / `c004_a111` by tilk, `there_v11` / `c004_a107` by Rootz) under the TIG Open Data License and TIG Innovator Outbound Game License.

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
