pub use anyhow::{Error as ContextError, Result as ContextResult};
use
{
    tig_structs::
    {
        *,
        core::
        {
            *
        },
        config::
        {
            *
        },
    },
    std::collections::HashSet,
};

pub trait Context
{
    async fn get_block_by_height(
        &self,
        block_height:                   i64,
    )                                           -> ContextResult<Option<Block>>;

    async fn get_block_by_id(
        &self,
        block_id:                       &String,
    )                                           -> ContextResult<Option<Block>>;

    async fn get_algorithm_by_id(
        &self,
        algorithm_id:                   &String,
    )                                           -> ContextResult<Option<Algorithm>>;

    async fn get_algorithm_by_tx_hash(
        &self,
        tx_hash:                        &String,
    )                                           -> ContextResult<Option<Algorithm>>;

    async fn get_challenges_by_id(
        &self,
        challenge_id:                   &String,
    )                                           -> ContextResult<Vec<Challenge>>;

    async fn get_challenge_by_id_and_height(
        &self,
        challenge_id:                   &String,
        block_height:                   u64,
    )                                           -> ContextResult<Option<Challenge>>;

    async fn get_challenge_by_id_and_block_id(
        &self,
        challenge_id:                   &String,
        block_id:                       &String,
    )                                           -> ContextResult<Option<Challenge>>;

    async fn get_precommits_by_settings(
        &self,
        settings:                       &BenchmarkSettings,
    )                                           -> ContextResult<Vec<Precommit>>;

    async fn get_precommits_by_benchmark_id(
        &self,
        benchmark_id:                   &String,
    )                                           -> ContextResult<Vec<Precommit>>;

    async fn get_transaction(
        &self,
        tx_hash:                        &String,
    )                                           -> ContextResult<Transaction>;

    async fn get_topups_by_txid(
        &self,
        tx_hash:                        &String,
    )                                           -> ContextResult<Vec<TopUp>>;

    async fn get_benchmarks_by_id(
        &self,
        benchmark_id:                   &String,
    )                                           -> ContextResult<Vec<Benchmark>>;

    async fn get_proofs_by_benchmark_id(
        &self,
        benchmark_id:                   &String,
    )                                           -> ContextResult<Vec<Proof>>;

    async fn verify_solution(
        &self,
        settings:                       &BenchmarkSettings,
        nonce:                          u64,
        solution:                       &Solution,
    )                                           -> ContextResult<anyhow::Result<()>>;

    async fn compute_solution(
        &self,
        settings:                       &BenchmarkSettings,
        nonce:                          u64,
        wasm_vm_config:                 &WasmVMConfig,
    )                                           -> ContextResult<anyhow::Result<OutputData>>;

    async fn add_precommit_to_mempool(
        &self,
        settings:                       &BenchmarkSettings,
        details:                        &PrecommitDetails,
    )                                           -> ContextResult<String>;

    async fn add_benchmark_to_mempool(
        &self,
        benchmark_id:                   &String,
        details:                        &BenchmarkDetails,
        solution_nonces:                &HashSet<u64>,
    )                                           -> ContextResult<()>;

    async fn add_proof_to_mempool(
        &self,
        benchmark_id:                   &String,
        merkle_proofs:                  &Vec<MerkleProof>,
    )                                           -> ContextResult<()>;

    async fn add_fraud_to_mempool(
        &self,
        benchmark_id:                   &String,
        allegations:                    &String,
    )                                           -> ContextResult<()>;
}
