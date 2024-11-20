pub use anyhow::Result;
use std::collections::{HashMap, HashSet};
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
    Latest,
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
pub enum BinariesFilter {
    Mempool,
    Confirmed,
}
#[allow(async_fn_in_trait)]
pub trait Context {
    async fn get_algorithm_ids(&self, filter: AlgorithmsFilter) -> Vec<String>;
    async fn get_algorithm_state(&self, algorithm_id: &String) -> Option<AlgorithmState>;
    async fn add_algorithm_to_mempool(&self, details: AlgorithmDetails) -> Result<String>;
    async fn get_benchmark_ids(&self, filter: BenchmarksFilter) -> Vec<String>;
    async fn get_benchmark_details(&self, benchmark_id: &String) -> Option<BenchmarkDetails>;
    async fn get_benchmark_state(&self, benchmark_id: &String) -> Option<BenchmarkState>;
    async fn add_benchmark_to_mempool(
        &self,
        benchmark_id: String,
        details: BenchmarkDetails,
        solution_nonces: HashSet<u64>,
    ) -> Result<()>;
    async fn get_binary_ids(&self, filter: BinariesFilter) -> Vec<String>;
    async fn get_binary_details(&self, algorithm_id: &String) -> Option<BinaryDetails>;
    async fn add_binary_to_mempool(
        &self,
        algorithm_id: String,
        details: BinaryDetails,
    ) -> Result<()>;
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
    async fn add_deposit_to_mempool(&self, details: DepositDetails) -> Result<String>;
    async fn get_fraud_ids(&self, filter: FraudsFilter) -> Vec<String>;
    async fn add_fraud_to_mempool(&self, benchmark_id: String, allegation: String) -> Result<()>;
    async fn get_player_ids(&self, filter: PlayersFilter) -> Vec<String>;
    async fn get_player_state(&self, player_id: &String) -> Option<PlayerState>;
    async fn get_player_block_data(
        &self,
        player_id: &String,
        block_id: &String,
    ) -> Option<PlayerBlockData>;
    async fn set_player_delegatee(&self, player_id: String, delegatee: String) -> Result<()>;
    async fn set_player_reward_share(&self, player_id: String, reward_share: f64) -> Result<()>;
    async fn get_precommit_ids(&self, filter: PrecommitsFilter) -> Vec<String>;
    async fn get_precommit_settings(&self, benchmark_id: &String) -> Option<BenchmarkSettings>;
    async fn get_precommit_details(&self, benchmark_id: &String) -> Option<PrecommitDetails>;
    async fn add_precommit_to_mempool(
        &self,
        settings: BenchmarkSettings,
        details: PrecommitDetails,
    ) -> Result<String>;
    async fn get_proofs_ids(&self, filter: ProofsFilter) -> Vec<String>;
    async fn get_proof_details(&self, benchmark_id: &String) -> Option<ProofDetails>;
    async fn get_proof_state(&self, benchmark_id: &String) -> Option<ProofState>;
    async fn add_proof_to_mempool(
        &self,
        benchmark_id: String,
        merkle_proofs: Vec<MerkleProof>,
    ) -> Result<()>;
    async fn get_topup_ids(&self, filter: TopUpsFilter) -> Vec<String>;
    async fn add_topup_to_mempool(&self, details: TopUpDetails) -> Result<String>;

    async fn build_block_cache(&self) -> AddBlockCache;
    async fn commit_block_cache(&self, cache: AddBlockCache);
}

pub struct AddBlockCache {
    pub config: ProtocolConfig,
    pub block_details: BlockDetails,
    pub block_data: BlockData,
    pub active_deposit_details: HashMap<String, DepositDetails>,
    pub active_players_state: HashMap<String, PlayerState>,
    pub active_players_block_data: HashMap<String, PlayerBlockData>,
    pub active_opow_block_data: HashMap<String, OPoWBlockData>,
    pub active_challenges_state: HashMap<String, ChallengeState>,
    pub active_challenges_block_data: HashMap<String, ChallengeBlockData>,
    pub active_algorithms_state: HashMap<String, AlgorithmState>,
    pub active_algorithms_details: HashMap<String, AlgorithmDetails>,
    pub active_algorithms_block_data: HashMap<String, AlgorithmBlockData>,
    pub prev_algorithms_block_data: HashMap<String, AlgorithmBlockData>,
    pub active_solutions: HashMap<String, (BenchmarkSettings, u32)>,
}
