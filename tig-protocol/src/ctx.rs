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
};


pub trait Context
{
    async fn get_block_by_height(
        &self,
        block_height:                   i64,
        include_data:                   bool,
    )                                           -> ContextResult<Option<Block>>;

    async fn get_block_by_id(
        &self,
        block_id:                       &String,
        include_data:                   bool,
    )                                           -> ContextResult<Option<Block>>;

    async fn get_algorithm_by_id(
        &self,
        algorithm_id:                   &String,
        include_data:                   bool,
    )                                           -> ContextResult<Option<Algorithm>>;

    async fn get_algorithm_by_tx_hash(
        &self,
        tx_hash:                        &String,
        include_data:                   bool,
    )                                           -> ContextResult<Option<Algorithm>>;

    async fn get_challenges_by_id(
        &self,
        challenge_id:                   &String,
    )                                           -> ContextResult<Vec<Challenge>>;

    async fn get_challenge_by_id_and_height(
        &self,
        challenge_id:                   &String,
        block_height:                   u64,
        include_data:                   bool,
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
        include_data:                   bool,
    )                                           -> ContextResult<Vec<Benchmark>>;

    async fn get_proofs_by_benchmark_id(
        &self,
        benchmark_id:                   &String,
        include_data:                   bool,
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
}
