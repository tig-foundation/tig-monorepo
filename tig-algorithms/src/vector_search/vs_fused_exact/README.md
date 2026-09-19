# TIG Code Submission

## Submission Details

* **Challenge Name:** vector_search
* **Algorithm Name:** vs_fused_exact
* **Copyright:** 2026 Kernelsmith
* **Identity of Submitter:** Kernelsmith
* **Identity of Creator of Algorithmic Method:** null
* **Unique Algorithm Identifier (UAI):** null

## Additional Notes

GPU nearest-neighbour search: a fused WMMA GEMM with an in-register argmin epilogue, followed by
exact FP32 re-ranking. The solver visits the full database, keeps a query tile resident in shared
memory and streams database tiles through double buffering. Inputs and accumulators use FP16;
every database chunk writes its winner per query, and a second
kernel re-evaluates all chunk winners within `delta` (default 0.5 in squared-distance units) of the
best FP16 value with the exact FP32 distance on the original vectors. FP16 near-ties between chunks
are thereby resolved exactly; a near-tie inside one 4096-row chunk still follows the FP16 ranking,
which is rare (re-rank windows 0.5 and 8 gave identical answers on 40 test instances).

Built with TIG's build_so LLVM fuel/signature instrumentation and build_ptx targeting
compute_70/sm_70. Measured on an RTX 3060 (12 GB) on 2026-09-19 against the official there_v10,
autovector_i and helix_search binaries: 25 fresh TIG instances per active track (125 in total),
identical settings for all solvers, rotating execution order, hyperparameters `{}`.
The quality equalled the best tested network solver on all 125 instances; all exceeded 68,500.

Paired median solver time relative to the network solvers across the five tracks:
0.51-0.53x there_v10, 0.60-0.62x autovector_i, 0.76-0.79x helix_search. Times exclude instance
generation, PTX JIT and the subsequent verifier. Other GPU architectures were not tested in this
run. The measurements do not establish future network adoption or rewards.

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
