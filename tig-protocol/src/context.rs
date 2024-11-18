use std::collections::HashSet;

pub use anyhow::{Error as ContextError, Result as ContextResult};
use tig_structs::{config::*, core::*};

#[derive(Debug, Clone, PartialEq)]
pub enum SubmissionType {
    Algorithm,
    Benchmark,
    Precommit,
    Proof,
    TopUp,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AlgorithmsFilter {
    Name(String),
    TxHash(String),
    Mempool,
    Confirmed,
}
#[derive(Debug, Clone, PartialEq)]
pub enum BenchmarksFilter {
    Mempool,
    Confirmed,
}
#[derive(Debug, Clone, PartialEq)]
pub enum BlockFilter {
    Current,
    LastConfirmed,
    Height(u32),
    Round(u32),
}
#[derive(Debug, Clone, PartialEq)]
pub enum ChallengesFilter {
    Mempool,
    Confirmed,
}
#[derive(Debug, Clone, PartialEq)]
pub enum FraudsFilter {
    Mempool,
    Confirmed,
}
#[derive(Debug, Clone, PartialEq)]
pub enum PlayersFilter {
    Id(String),
    Name(String),
    Benchmarkers,
    Innovators,
}
#[derive(Debug, Clone, PartialEq)]
pub enum PrecommitsFilter {
    Settings(BenchmarkSettings),
    Mempool,
    Confirmed,
}
#[derive(Debug, Clone, PartialEq)]
pub enum ProofsFilter {
    Mempool,
    Confirmed,
}
#[derive(Debug, Clone, PartialEq)]
pub enum TopUpsFilter {
    PlayerId(String),
    Mempool,
    Confirmed,
}
#[derive(Debug, Clone, PartialEq)]
pub enum WasmsFilter {
    Mempool,
    Confirmed,
}
#[allow(async_fn_in_trait)]
pub trait Context {
    async fn get_algorithm_ids(&self, filter: AlgorithmsFilter) -> Vec<String>;
    async fn get_algorithm_state(&self, algorithm_id: &String) -> Option<AlgorithmState>;
    async fn get_benchmark_ids(&self, filter: BenchmarksFilter) -> Vec<String>;
    async fn get_benchmark_details(&self, benchmark_id: &String) -> Option<BenchmarkDetails>;
    async fn get_benchmark_state(&self, benchmark_id: &String) -> Option<BenchmarkState>;
    async fn confirm_benchmark(
        &self,
        benchmark_id: String,
        details: BenchmarkDetails,
        solution_nonces: HashSet<u64>,
    ) -> ContextResult<()>;
    async fn get_block_id(&self, filter: BlockFilter) -> Option<String>;
    async fn get_block_details(&self, block_id: &String) -> Option<BlockDetails>;
    async fn get_challenge_ids(&self, filter: ChallengesFilter) -> Vec<String>;
    async fn get_challenge_state(&self, challenge_id: &String) -> Option<ChallengeState>;
    async fn get_challenge_block_data(
        &self,
        challenge_id: &String,
        block_id: &String,
    ) -> Option<ChallengeBlockData>;
    async fn get_config(&self) -> ProtocolConfig;
    async fn get_fraud_ids(&self, filter: FraudsFilter) -> Vec<String>;
    async fn get_player_ids(&self, filter: PlayersFilter) -> Vec<String>;
    async fn get_player_state(&self, player_id: &String) -> Option<PlayerState>;
    async fn get_player_block_data(
        &self,
        player_id: &String,
        block_id: &String,
    ) -> Option<PlayerBlockData>;
    async fn get_precommit_ids(&self, filter: PrecommitsFilter) -> Vec<String>;
    async fn get_precommit_settings(&self, benchmark_id: &String) -> Option<BenchmarkSettings>;
    async fn get_precommit_details(&self, benchmark_id: &String) -> Option<PrecommitDetails>;
    async fn confirm_precommit(
        &self,
        settings: BenchmarkSettings,
        details: PrecommitDetails,
    ) -> ContextResult<String>;
    async fn get_proofs_ids(&self, filter: ProofsFilter) -> Vec<String>;
    async fn get_proof_state(&self, benchmark_id: &String) -> Option<ProofState>;
    async fn get_topup_ids(&self, filter: TopUpsFilter) -> Vec<String>;
    async fn get_wasm_ids(&self, filter: WasmsFilter) -> Vec<String>;
    async fn verify_solution(
        &self,
        settings: &BenchmarkSettings,
        nonce: u64,
        solution: &Solution,
    ) -> ContextResult<anyhow::Result<()>>;
    async fn compute_solution(
        &self,
        settings: &BenchmarkSettings,
        nonce: u64,
        wasm_vm_config: &WasmVMConfig,
    ) -> ContextResult<anyhow::Result<OutputData>>;
    async fn get_transaction(&self, tx_hash: &String) -> ContextResult<Transaction>;
    async fn get_latest_eth_block_num(&self) -> ContextResult<String>;
    async fn get_player_deposit(
        &self,
        eth_block_num: &String,
        player_id: &String,
    ) -> ContextResult<Option<PreciseNumber>>;

    // Mempool
    async fn add_block(
        &self,
        details: BlockDetails,
        data: BlockData,
        config: ProtocolConfig,
    ) -> ContextResult<String>;
    async fn add_challenge_to_mempool(&self, details: ChallengeDetails) -> ContextResult<String>;
    async fn add_algorithm_to_mempool(
        &self,
        details: AlgorithmDetails,
        code: String,
    ) -> ContextResult<String>;
    async fn add_benchmark_to_mempool(
        &self,
        benchmark_id: &String,
        details: BenchmarkDetails,
        solution_nonces: HashSet<u64>,
    ) -> ContextResult<()>;
    async fn add_precommit_to_mempool(
        &self,
        settings: BenchmarkSettings,
        details: PrecommitDetails,
    ) -> ContextResult<String>;
    async fn add_proof_to_mempool(
        &self,
        benchmark_id: &String,
        merkle_proofs: Vec<MerkleProof>,
    ) -> ContextResult<()>;
    async fn add_fraud_to_mempool(
        &self,
        benchmark_id: &String,
        allegation: String,
    ) -> ContextResult<()>;
    async fn add_topup_to_mempool(
        &self,
        topup_id: &String,
        details: TopUpDetails,
    ) -> ContextResult<()>;
    async fn add_wasm_to_mempool(
        &self,
        algorithm_id: &String,
        details: WasmDetails,
    ) -> ContextResult<()>;

    // Updates
    async fn update_challenge_state(
        &self,
        challenge_id: &String,
        state: ChallengeState,
    ) -> ContextResult<()>;
    async fn update_challenge_block_data(
        &self,
        challenge_id: &String,
        block_id: &String,
        block_data: ChallengeBlockData,
    ) -> ContextResult<()>;
    async fn update_algorithm_state(
        &self,
        algorithm_id: &String,
        state: AlgorithmState,
    ) -> ContextResult<()>;
    async fn update_algorithm_block_data(
        &self,
        algorithm_id: &String,
        block_id: &String,
        block_data: AlgorithmBlockData,
    ) -> ContextResult<()>;
    async fn update_benchmark_state(
        &self,
        benchmark_id: &String,
        state: BenchmarkState,
    ) -> ContextResult<()>;
    async fn update_player_state(
        &self,
        player_id: &String,
        state: PlayerState,
    ) -> ContextResult<()>;
    async fn update_precommit_state(
        &self,
        benchmark_id: &String,
        state: PrecommitState,
    ) -> ContextResult<()>;
    async fn update_proof_state(
        &self,
        benchmark_id: &String,
        state: ProofState,
    ) -> ContextResult<()>;
    async fn update_fraud_state(
        &self,
        benchmark_id: &String,
        state: FraudState,
    ) -> ContextResult<()>;
    async fn update_topup_state(&self, topup_id: &String, state: TopUpState) -> ContextResult<()>;
    async fn update_player_block_data(
        &self,
        player_id: &String,
        block_id: &String,
        block_data: PlayerBlockData,
    ) -> ContextResult<()>;
    async fn update_wasm_state(&self, algorithm_id: &String, state: WasmState)
        -> ContextResult<()>;
}
