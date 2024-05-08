pub use anyhow::{Error as ContextError, Result as ContextResult};
use tig_structs::{config::*, core::*};

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
        &mut self,
        filter: AlgorithmsFilter,
        block_data: Option<BlockFilter>,
        include_data: bool,
    ) -> ContextResult<Vec<Algorithm>>;
    async fn get_benchmarks(
        &mut self,
        filter: BenchmarksFilter,
        include_data: bool,
    ) -> ContextResult<Vec<Benchmark>>;
    async fn get_block(
        &mut self,
        filter: BlockFilter,
        include_data: bool,
    ) -> ContextResult<Option<Block>>;
    async fn get_challenges(
        &mut self,
        filter: ChallengesFilter,
        block_data: Option<BlockFilter>,
    ) -> ContextResult<Vec<Challenge>>;
    async fn get_config(&mut self) -> ContextResult<ProtocolConfig>;
    async fn get_frauds(
        &mut self,
        filter: FraudsFilter,
        include_data: bool,
    ) -> ContextResult<Vec<Fraud>>;
    async fn get_players(
        &mut self,
        filter: PlayersFilter,
        block_data: Option<BlockFilter>,
    ) -> ContextResult<Vec<Player>>;
    async fn get_proofs(
        &mut self,
        filter: ProofsFilter,
        include_data: bool,
    ) -> ContextResult<Vec<Proof>>;
    async fn get_wasms(
        &mut self,
        filter: WasmsFilter,
        include_data: bool,
    ) -> ContextResult<Vec<Wasm>>;
    async fn verify_solution(
        &mut self,
        settings: &BenchmarkSettings,
        nonce: u32,
        solution: &Solution,
    ) -> ContextResult<anyhow::Result<()>>;
    async fn compute_solution(
        &mut self,
        settings: &BenchmarkSettings,
        nonce: u32,
        wasm_vm_config: &WasmVMConfig,
    ) -> ContextResult<anyhow::Result<SolutionData>>;
    async fn get_transaction(&mut self, tx_hash: &String) -> ContextResult<Transaction>;
    async fn get_multisig_owners(&mut self, address: &String) -> ContextResult<Vec<String>>;

    // Mempool
    async fn add_block(
        &mut self,
        details: &BlockDetails,
        data: &BlockData,
        config: &ProtocolConfig,
    ) -> ContextResult<String>;
    async fn add_challenge_to_mempool(
        &mut self,
        details: &ChallengeDetails,
    ) -> ContextResult<String>;
    async fn add_algorithm_to_mempool(
        &mut self,
        details: &AlgorithmDetails,
        code: &String,
    ) -> ContextResult<String>;
    async fn add_benchmark_to_mempool(
        &mut self,
        settings: &BenchmarkSettings,
        details: &BenchmarkDetails,
        solutions_metadata: &Vec<SolutionMetaData>,
        solution_data: &SolutionData,
    ) -> ContextResult<String>;
    async fn add_proof_to_mempool(
        &mut self,
        benchmark_id: &String,
        solutions_data: &Vec<SolutionData>,
    ) -> ContextResult<()>;
    async fn add_fraud_to_mempool(
        &mut self,
        benchmark_id: &String,
        allegation: &String,
    ) -> ContextResult<()>;
    async fn add_wasm_to_mempool(
        &mut self,
        algorithm_id: &String,
        details: &WasmDetails,
        wasm_blob: &Option<Vec<u8>>,
    ) -> ContextResult<()>;

    // Updates
    async fn update_challenge_state(
        &mut self,
        challenge_id: &String,
        state: &ChallengeState,
    ) -> ContextResult<()>;
    async fn update_challenge_block_data(
        &mut self,
        challenge_id: &String,
        block_id: &String,
        block_data: &ChallengeBlockData,
    ) -> ContextResult<()>;
    async fn update_algorithm_state(
        &mut self,
        algorithm_id: &String,
        state: &AlgorithmState,
    ) -> ContextResult<()>;
    async fn update_algorithm_block_data(
        &mut self,
        algorithm_id: &String,
        block_id: &String,
        block_data: &AlgorithmBlockData,
    ) -> ContextResult<()>;
    async fn update_benchmark_state(
        &mut self,
        benchmark_id: &String,
        state: &BenchmarkState,
    ) -> ContextResult<()>;
    async fn update_proof_state(
        &mut self,
        benchmark_id: &String,
        state: &ProofState,
    ) -> ContextResult<()>;
    async fn update_fraud_state(
        &mut self,
        benchmark_id: &String,
        state: &FraudState,
    ) -> ContextResult<()>;
    async fn update_player_block_data(
        &mut self,
        player_id: &String,
        block_id: &String,
        block_data: &PlayerBlockData,
    ) -> ContextResult<()>;
    async fn update_wasm_state(
        &mut self,
        algorithm_id: &String,
        state: &WasmState,
    ) -> ContextResult<()>;
}
