use crate::{config::ProtocolConfig, serializable_struct_with_getters};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::{HashMap, HashSet};
use tig_utils::{jsonify, u64s_from_str, u8s_from_str};
pub use tig_utils::{Frontier, MerkleBranch, MerkleHash, Point, PreciseNumber, Transfer, U256};

serializable_struct_with_getters! {
    Algorithm {
        id: String,
        details: AlgorithmDetails,
        state: AlgorithmState,
        block_data: Option<AlgorithmBlockData>,
    }
}
serializable_struct_with_getters! {
    Benchmark {
        id: String,
        details: BenchmarkDetails,
        state: BenchmarkState,
        solution_nonces: Option<HashSet<u64>>,
    }
}
serializable_struct_with_getters! {
    Binary {
        algorithm_id: String,
        details: BinaryDetails,
        state: BinaryState,
    }
}
serializable_struct_with_getters! {
    Block {
        id: String,
        details: BlockDetails,
        data: Option<BlockData>,
        config: Option<ProtocolConfig>,
    }
}
serializable_struct_with_getters! {
    Breakthrough {
        id: String,
        details: BreakthroughDetails,
        state: BreakthroughState,
        block_data: Option<BreakthroughBlockData>,
    }
}
serializable_struct_with_getters! {
    Challenge {
        id: String,
        details: ChallengeDetails,
        state: ChallengeState,
        block_data: Option<ChallengeBlockData>,
    }
}
serializable_struct_with_getters! {
    Deposit {
        id: String,
        details: DepositDetails,
        state: DepositState,
    }
}
serializable_struct_with_getters! {
    Fraud {
        benchmark_id: String,
        state: FraudState,
        allegation: Option<String>,
    }
}
serializable_struct_with_getters! {
    OPoW {
        player_id: String,
        block_data: Option<OPoWBlockData>,
    }
}
serializable_struct_with_getters! {
    Player {
        id: String,
        details: PlayerDetails,
        state: PlayerState,
        block_data: Option<PlayerBlockData>,
    }
}
serializable_struct_with_getters! {
    Precommit {
        benchmark_id: String,
        details: PrecommitDetails,
        settings: BenchmarkSettings,
        state: PrecommitState,
    }
}
serializable_struct_with_getters! {
    MerkleProof {
        leaf: OutputData,
        branch: MerkleBranch,
    }
}
serializable_struct_with_getters! {
    Proof {
        benchmark_id: String,
        details: ProofDetails,
        state: ProofState,
        merkle_proofs: Option<Vec<MerkleProof>>,
    }
}
serializable_struct_with_getters! {
    TopUp {
        id: String,
        details: TopUpDetails,
        state: TopUpState,
    }
}

// Algorithm child structs
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AlgorithmType {
    Wasm,
    Ptx,
}
serializable_struct_with_getters! {
    AlgorithmDetails {
        name: String,
        player_id: String,
        challenge_id: String,
        breakthrough_id: Option<String>,
        r#type: AlgorithmType,
        fee_paid: PreciseNumber,
    }
}
serializable_struct_with_getters! {
    AlgorithmState {
        block_confirmed: u32,
        round_submitted: u32,
        round_pushed: Option<u32>,
        round_active: Option<u32>,
        round_merged: Option<u32>,
        banned: bool,
    }
}
serializable_struct_with_getters! {
    AlgorithmBlockData {
        num_qualifiers_by_player: HashMap<String, u32>,
        adoption: PreciseNumber,
        merge_points: u32,
        reward: PreciseNumber,
    }
}

// Benchmark child structs
serializable_struct_with_getters! {
    BenchmarkSettings {
        player_id: String,
        block_id: String,
        challenge_id: String,
        algorithm_id: String,
        difficulty: Vec<i32>,
    }
}
impl BenchmarkSettings {
    pub fn calc_seed(&self, rand_hash: &String, nonce: u64) -> [u8; 32] {
        u8s_from_str(&format!("{}_{}_{}", jsonify(&self), rand_hash, nonce))
    }
}
serializable_struct_with_getters! {
    BenchmarkDetails {
        num_solutions: u32,
        merkle_root: MerkleHash,
        sampled_nonces: HashSet<u64>,
    }
}
serializable_struct_with_getters! {
    BenchmarkState {
        block_confirmed: u32,
    }
}
serializable_struct_with_getters! {
    OutputMetaData {
        nonce: u64,
        runtime_signature: u64,
        fuel_consumed: u64,
        solution_signature: u64,
    }
}
impl From<OutputData> for OutputMetaData {
    fn from(data: OutputData) -> Self {
        OutputMetaData {
            solution_signature: data.calc_solution_signature(),
            runtime_signature: data.runtime_signature,
            fuel_consumed: data.fuel_consumed,
            nonce: data.nonce,
        }
    }
}
impl From<OutputMetaData> for MerkleHash {
    fn from(data: OutputMetaData) -> Self {
        MerkleHash(u8s_from_str(&jsonify(&data)))
    }
}
impl From<OutputData> for MerkleHash {
    fn from(data: OutputData) -> Self {
        MerkleHash::from(OutputMetaData::from(data))
    }
}

// Binary child structs
serializable_struct_with_getters! {
    BinaryDetails {
        compile_success: bool,
        download_url: Option<String>,
    }
}
serializable_struct_with_getters! {
    BinaryState {
        block_confirmed: u32,
    }
}

// Block child structs
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, Hash)]
#[serde(rename_all = "lowercase")]
pub enum TxType {
    Algorithm,
    Benchmark,
    Binary,
    Breakthrough,
    Challenge,
    Deposit,
    Fraud,
    Precommit,
    Proof,
    TopUp,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, Hash)]
#[serde(rename_all = "lowercase")]
pub enum ActiveType {
    Algorithm,
    Benchmark,
    Breakthrough,
    Challenge,
    Deposit,
    OPoW,
    Player,
}
serializable_struct_with_getters! {
    BlockDetails {
        prev_block_id: String,
        height: u32,
        round: u32,
        num_confirmed: HashMap<TxType, u32>,
        num_active: HashMap<ActiveType, u32>,
        timestamp: u64,
    }
}
serializable_struct_with_getters! {
    BlockData {
        confirmed_ids: HashMap<TxType, HashSet<String>>,
        active_ids: HashMap<ActiveType, HashSet<String>>,
    }
}

// Breakthrough child structs
serializable_struct_with_getters! {
    BreakthroughDetails {
        name: String,
        player_id: String,
        challenge_id: String,
        fee_paid: PreciseNumber,
    }
}
serializable_struct_with_getters! {
    BreakthroughState {
        block_confirmed: u32,
        round_submitted: u32,
        round_pushed: u32,
        round_votes_tallied: u32,
        votes_tally: HashMap<bool, PreciseNumber>,
        round_active: Option<u32>,
        round_merged: Option<u32>,
        banned: bool,
    }
}
serializable_struct_with_getters! {
    BreakthroughBlockData {
        adoption: PreciseNumber,
        merge_points: u32,
        reward: PreciseNumber,
    }
}

// Challenge child structs
serializable_struct_with_getters! {
    ChallengeDetails {
        name: String,
    }
}
serializable_struct_with_getters! {
    ChallengeState {
        round_active: u32,
    }
}
serializable_struct_with_getters! {
    ChallengeBlockData {
        num_qualifiers: u32,
        qualifier_difficulties: HashSet<Point>,
        base_frontier: Frontier,
        scaled_frontier: Frontier,
        scaling_factor: f64,
        base_fee: PreciseNumber,
        per_nonce_fee: PreciseNumber,
    }
}

// Deposit child structs
serializable_struct_with_getters! {
    DepositDetails {
        player_id: String,
        tx_hash: String,
        log_idx: usize,
        amount: PreciseNumber,
        start_timestamp: u64,
        end_timestamp: u64,
    }
}
serializable_struct_with_getters! {
    DepositState {
        block_confirmed: u32,
    }
}

// Fraud child structs
serializable_struct_with_getters! {
    FraudState {
        block_confirmed: u32,
    }
}

// OPoW child structs
serializable_struct_with_getters! {
    OPoWBlockData {
        num_qualifiers_by_challenge: HashMap<String, u32>,
        cutoff: u32,
        self_deposit: PreciseNumber,
        delegated_weighted_deposit: PreciseNumber,
        delegators: HashSet<String>,
        reward_share: f64,
        imbalance: PreciseNumber,
        influence: PreciseNumber,
        reward: PreciseNumber,
    }
}

// Player child structs
serializable_struct_with_getters! {
    PlayerDetails {
        name: Option<String>,
        is_multisig: bool,
    }
}
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct PlayerValue<T> {
    pub value: T,
    pub block_set: u32,
}
serializable_struct_with_getters! {
    PlayerState {
        total_fees_paid: PreciseNumber,
        available_fee_balance: PreciseNumber,
        delegatees: Option<PlayerValue<HashMap<String, f64>>>,
        votes: HashMap<String, PlayerValue<bool>>,
        reward_share: Option<PlayerValue<f64>>,
    }
}
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, Hash)]
#[serde(rename_all = "lowercase")]
pub enum RewardType {
    Benchmarker,
    Algorithm,
    Breakthrough,
    Delegator,
}
serializable_struct_with_getters! {
    PlayerBlockData {
        delegatees: HashMap<String, f64>,
        reward_by_type: HashMap<RewardType, PreciseNumber>,
        deposit_by_locked_period: Vec<PreciseNumber>,
        weighted_deposit: PreciseNumber,
    }
}

// Precommit child structs
serializable_struct_with_getters! {
    PrecommitDetails {
        block_started: u32,
        num_nonces: u32,
        rand_hash: String,
        fee_paid: PreciseNumber,
    }
}
serializable_struct_with_getters! {
    PrecommitState {
        block_confirmed: u32,
    }
}

// Proof child structs
serializable_struct_with_getters! {
    ProofDetails {
        submission_delay: u32,
        block_active: u32,
    }
}
serializable_struct_with_getters! {
    ProofState {
        block_confirmed: u32,
    }
}
pub type Solution = Map<String, Value>;
serializable_struct_with_getters! {
    OutputData {
        nonce: u64,
        runtime_signature: u64,
        fuel_consumed: u64,
        solution: Solution,
    }
}
impl OutputData {
    pub fn calc_solution_signature(&self) -> u64 {
        u64s_from_str(&jsonify(&self.solution))[0]
    }
}

// TopUp child structs
serializable_struct_with_getters! {
    TopUpDetails {
        player_id: String,
        tx_hash: String,
        log_idx: usize,
        amount: PreciseNumber,
    }
}
serializable_struct_with_getters! {
    TopUpState {
        block_confirmed: u32,
    }
}
