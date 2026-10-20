# TIG Code Submission

## Submission Details

* **Challenge Name:** vector_search
* **Algorithm Name:** tuned_v11
* **Copyright:** 2026 Yash Shah (modifications); 2026 Rootz (original implementation)
* **Identity of Submitter:** Yash Shah
* **Identity of Creator of Algorithmic Method:** Rootz
* **Unique Algorithm Identifier (UAI):** null

## References and Acknowledgments

* **Original implementation:** `there_v11` by Rootz, TIG c004_a107
* **Source:** https://github.com/tig-foundation/tig-monorepo/tree/main/tig-algorithms/src/vector_search/there_v11

## Additional Notes

This solver exposes database chunk size and packing grid size as optional
hyperparameters. The `tau_half`
parameter adjusts the FP16 safety window used to decide whether to rescan an
entire database chunk in FP32. The default value of 0.5 keeps `there_v11`'s
original behavior; lower values can reduce quality and should not be used for
the reported parity results. It keeps the original `there_v11` screening and
FP32 reranking behavior, while omitting the unused second-place index from
the screening kernel. Equal rounded quality alone does not establish exact
nearest neighbors.

The scan uses eight warps and 16 database rows per tile.
It runs on Turing GPUs (compute capability 7.5). Other CUDA architectures use
the original 32-row scan layout because its timing has not been measured there.
On a local Quadro RTX 6000, the final build matched the leader's complete
output and official quality on 180 fresh paired instances across all five
active tracks. In a 12-nonce-per-track run, median paired solver time was
about 9–11% lower by track. Instrumented fuel rose 43–44%, remaining below
the current c004 fuel cap. The GPU was shared during these measurements, so
timing can vary with background load.

TIG's Innovator Outbound Game License permits using a submitted implementation
as a starting point for further in-game innovation. The source and author
above are credited because this prototype modifies that implementation.
See https://docs.tig.foundation/innovators and
https://github.com/tig-foundation/tig-monorepo/blob/main/docs/guides/anatomy.md.

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
