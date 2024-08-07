pub use anyhow::{Error as ContextError, Result as ContextResult};
use tig_structs::{config::*, core::*};

#[derive(Debug, Clone, PartialEq)]
pub enum SubmissionType {
    Algorithm,
    Benchmark,
    Proof,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AlgorithmsFilter {
    Id(String),
    Name(String),
    TxHash(String),
    Mempool,
    Confirmed,
}
#[derive(Debug, Clone, PartialEq)]
pub enum BenchmarksFilter {
    Id(String),
    Settings(BenchmarkSettings),
    Mempool { from_block_started: u32 },
    Confirmed { from_block_started: u32 },
}
#[derive(Debug, Clone, PartialEq)]
pub enum BlockFilter {
    Latest,
    Height(u32),
    Id(String),
    Round(u32),
}
#[derive(Debug, Clone, PartialEq)]
pub enum ChallengesFilter {
    Id(String),
    Name(String),
    Mempool,
    Confirmed,
}
#[derive(Debug, Clone, PartialEq)]
pub enum FraudsFilter {
    BenchmarkId(String),
    Mempool { from_block_started: u32 },
    Confirmed { from_block_started: u32 },
}
#[derive(Debug, Clone, PartialEq)]
pub enum PlayersFilter {
    Id(String),
    Name(String),
    Benchmarkers,
    Innovators,
}
#[derive(Debug, Clone, PartialEq)]
pub enum ProofsFilter {
    BenchmarkId(String),
    Mempool { from_block_started: u32 },
    Confirmed { from_block_started: u32 },
}
#[derive(Debug, Clone, PartialEq)]
pub enum WasmsFilter {
    AlgorithmId(String),
    Mempool,
    Confirmed,
}
#[allow(async_fn_in_trait)]
pub trait Context {
    async fn get_algorithms(
        &self,
        filter: AlgorithmsFilter,
        block_data: Option<BlockFilter>,
        include_data: bool,
    ) -> ContextResult<Vec<Algorithm>>;
    async fn get_benchmarks(
        &self,
        filter: BenchmarksFilter,
        include_data: bool,
    ) -> ContextResult<Vec<Benchmark>>;
    async fn get_block(
        &self,
        filter: BlockFilter,
        include_data: bool,
    ) -> ContextResult<Option<Block>>;
    async fn get_challenges(
        &self,
        filter: ChallengesFilter,
        block_data: Option<BlockFilter>,
    ) -> ContextResult<Vec<Challenge>>;
    async fn get_config(&self) -> ContextResult<ProtocolConfig>;
    async fn get_frauds(
        &self,
        filter: FraudsFilter,
        include_data: bool,
    ) -> ContextResult<Vec<Fraud>>;
    async fn get_players(
        &self,
        filter: PlayersFilter,
        block_data: Option<BlockFilter>,
    ) -> ContextResult<Vec<Player>>;
    async fn get_proofs(
        &self,
        filter: ProofsFilter,
        include_data: bool,
    ) -> ContextResult<Vec<Proof>>;
    async fn get_wasms(&self, filter: WasmsFilter, include_data: bool) -> ContextResult<Vec<Wasm>>;
    async fn verify_solution(
        &self,
        settings: &BenchmarkSettings,
        nonce: u32,
        solution: &Solution,
    ) -> ContextResult<anyhow::Result<()>>;
    async fn compute_solution(
        &self,
        settings: &BenchmarkSettings,
        nonce: u32,
        wasm_vm_config: &WasmVMConfig,
    ) -> ContextResult<anyhow::Result<SolutionData>>;
    async fn get_transaction(&self, tx_hash: &String) -> ContextResult<Transaction>;
    async fn get_multisig_owners(&self, address: &String) -> ContextResult<Vec<String>>;
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
        settings: BenchmarkSettings,
        details: BenchmarkDetails,
        solutions_metadata: Vec<SolutionMetaData>,
        solution_data: SolutionData,
    ) -> ContextResult<String>;
    async fn add_proof_to_mempool(
        &self,
        benchmark_id: &String,
        solutions_data: Vec<SolutionData>,
    ) -> ContextResult<()>;
    async fn add_fraud_to_mempool(
        &self,
        benchmark_id: &String,
        allegation: String,
    ) -> ContextResult<()>;
    async fn add_wasm_to_mempool(
        &self,
        algorithm_id: &String,
        details: WasmDetails,
        wasm_blob: Option<Vec<u8>>,
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
    async fn update_player_block_data(
        &self,
        player_id: &String,
        block_id: &String,
        block_data: PlayerBlockData,
    ) -> ContextResult<()>;
    async fn update_wasm_state(&self, algorithm_id: &String, state: WasmState)
        -> ContextResult<()>;
}
