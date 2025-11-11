pub use anyhow::Result;
use serde_json::{Map, Value};
use std::collections::HashMap;
use tig_structs::{config::*, core::*};

#[allow(async_fn_in_trait)]
pub trait Context {
    async fn get_advance_state(&self, advance_id: &String) -> Option<AdvanceState>;
    async fn add_advance_to_mempool(
        &self,
        details: AdvanceDetails,
        evidence: String,
    ) -> Result<String>;
    async fn get_benchmark_details(&self, benchmark_id: &String) -> Option<BenchmarkDetails>;
    async fn add_benchmark_to_mempool(
        &self,
        benchmark_id: String,
        details: BenchmarkDetails,
        solution_quality: Option<Vec<i32>>,
    ) -> Result<()>;
    async fn get_binary_details(&self, code_id: &String) -> Option<BinaryDetails>;
    async fn add_binary_to_mempool(&self, code_id: String, details: BinaryDetails) -> Result<()>;
    async fn get_latest_block_id(&self) -> String;
    async fn get_block_details(&self, block_id: &String) -> Option<BlockDetails>;
    async fn get_challenge_state(&self, challenge_id: &String) -> Option<ChallengeState>;
    async fn get_challenge_block_data(
        &self,
        challenge_id: &String,
        block_id: &String,
    ) -> Option<ChallengeBlockData>;
    async fn get_code_state(&self, code_id: &String) -> Option<CodeState>;
    async fn add_code_to_mempool(
        &self,
        details: CodeDetails,
        source_code: HashMap<String, String>,
    ) -> Result<String>;
    async fn get_config(&self) -> ProtocolConfig;
    async fn add_deposit_to_mempool(&self, details: DepositDetails) -> Result<String>;
    async fn get_player_details(&self, player_id: &String) -> Option<PlayerDetails>;
    async fn get_player_state(&self, player_id: &String) -> Option<PlayerState>;
    async fn get_player_block_data(
        &self,
        player_id: &String,
        block_id: &String,
    ) -> Option<PlayerBlockData>;
    async fn set_player_delegatees(
        &self,
        player_id: String,
        delegatees: HashMap<String, f64>,
    ) -> Result<()>;
    async fn set_player_reward_share(&self, player_id: String, reward_share: f64) -> Result<()>;
    async fn set_player_coinbase(
        &self,
        player_id: String,
        coinbase: HashMap<String, f64>,
    ) -> Result<()>;
    async fn set_player_vote(&self, player_id: String, advance_id: String, yes: bool)
        -> Result<()>;
    async fn get_precommit_settings(&self, benchmark_id: &String) -> Option<BenchmarkSettings>;
    async fn get_precommit_details(&self, benchmark_id: &String) -> Option<PrecommitDetails>;
    async fn add_precommit_to_mempool(
        &self,
        settings: BenchmarkSettings,
        details: PrecommitDetails,
    ) -> Result<String>;
    async fn get_proof_details(&self, benchmark_id: &String) -> Option<ProofDetails>;
    async fn get_proof_state(&self, benchmark_id: &String) -> Option<ProofState>;
    async fn add_proof_to_mempool(
        &self,
        benchmark_id: String,
        merkle_proofs: Vec<MerkleProof>,
        allegation: Option<String>,
    ) -> Result<()>;
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
    pub active_challenges_block_data: HashMap<String, ChallengeBlockData>,
    pub active_codes_state: HashMap<String, CodeState>,
    pub active_codes_details: HashMap<String, CodeDetails>,
    pub active_codes_block_data: HashMap<String, CodeBlockData>,
    pub voting_advances_state: HashMap<String, AdvanceState>,
    pub active_advances_state: HashMap<String, AdvanceState>,
    pub active_advances_details: HashMap<String, AdvanceDetails>,
    pub active_advances_block_data: HashMap<String, AdvanceBlockData>,
    pub active_benchmarks: Vec<(BenchmarkSettings, i32, u64)>,
}
