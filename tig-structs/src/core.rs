use crate::{config::ProtocolConfig, serializable_struct_with_getters};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::{HashMap, HashSet};
use tig_utils::{jsonify, u32_from_str};
pub use tig_utils::{Frontier, Point, PreciseNumber, Transaction};

serializable_struct_with_getters! {
    Algorithm {
        id: String,
        details: AlgorithmDetails,
        state: Option<AlgorithmState>,
        block_data: Option<AlgorithmBlockData>,
        code: Option<String>,
    }
}
serializable_struct_with_getters! {
    Benchmark {
        id: String,
        settings: BenchmarkSettings,
        details: BenchmarkDetails,
        state: Option<BenchmarkState>,
        solutions_meta_data: Option<Vec<SolutionMetaData>>,
        solution_data: Option<SolutionData>,
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
    Challenge {
        id: String,
        details: ChallengeDetails,
        state: Option<ChallengeState>,
        block_data: Option<ChallengeBlockData>,
    }
}
serializable_struct_with_getters! {
    Player {
        id: String,
        details: PlayerDetails,
        block_data: Option<PlayerBlockData>,
    }
}
serializable_struct_with_getters! {
    Proof {
        benchmark_id: String,
        state: Option<ProofState>,
        solutions_data: Option<Vec<SolutionData>>,
    }
}
serializable_struct_with_getters! {
    Fraud {
        benchmark_id: String,
        state: Option<FraudState>,
        allegation: Option<String>,
    }
}
serializable_struct_with_getters! {
    Wasm {
        algorithm_id: String,
        details: WasmDetails,
        state: Option<WasmState>,
        wasm_blob: Option<Vec<u8>>,
    }
}

// Algorithm child structs
serializable_struct_with_getters! {
    AlgorithmDetails {
        name: String,
        player_id: String,
        challenge_id: String,
        tx_hash: String,
    }
}
serializable_struct_with_getters! {
    AlgorithmState {
        block_confirmed: Option<u32>,
        round_submitted: Option<u32>,
        round_pushed: Option<u32>,
        round_merged: Option<u32>,
    }
}
serializable_struct_with_getters! {
    AlgorithmBlockData {
        num_qualifiers_by_player: Option<HashMap<String, u32>>,
        adoption: Option<PreciseNumber>,
        merge_points: Option<u32>,
        reward: Option<PreciseNumber>,
        round_earnings: Option<PreciseNumber>,
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
    pub fn calc_seed(&self, nonce: u32) -> u32 {
        u32_from_str(jsonify(&self).as_str()) ^ nonce
    }
}
serializable_struct_with_getters! {
    BenchmarkDetails {
        block_started: u32,
        num_solutions: u32,
    }
}
serializable_struct_with_getters! {
    BenchmarkState {
        block_confirmed: Option<u32>,
        sampled_nonces: Option<Vec<u32>>,
    }
}
serializable_struct_with_getters! {
    SolutionMetaData {
        nonce: u32,
        solution_signature: u32,
    }
}
impl From<SolutionData> for SolutionMetaData {
    fn from(data: SolutionData) -> Self {
        SolutionMetaData {
            solution_signature: data.calc_solution_signature(),
            nonce: data.nonce,
        }
    }
}

// Block child structs
serializable_struct_with_getters! {
    BlockDetails {
        prev_block_id: String,
        height: u32,
        round: u32,
    }
}
serializable_struct_with_getters! {
    BlockData {
        mempool_challenge_ids: HashSet<String>,
        mempool_algorithm_ids: HashSet<String>,
        mempool_benchmark_ids: HashSet<String>,
        mempool_proof_ids: HashSet<String>,
        mempool_fraud_ids: HashSet<String>,
        mempool_wasm_ids: HashSet<String>,
        active_challenge_ids: HashSet<String>,
        active_algorithm_ids: HashSet<String>,
        active_benchmark_ids: HashSet<String>,
        active_player_ids: HashSet<String>,
    }
}

// Challenge child structs
serializable_struct_with_getters! {
    ChallengeDetails {
        name: String,
        difficulty_parameters: Vec<DifficultyParameter>,
    }
}
impl ChallengeDetails {
    pub fn min_difficulty(&self) -> Point {
        self.difficulty_parameters
            .iter()
            .map(|p| p.min_value)
            .collect()
    }
    pub fn max_difficulty(&self) -> Point {
        self.difficulty_parameters
            .iter()
            .map(|p| p.max_value)
            .collect()
    }
}
serializable_struct_with_getters! {
    ChallengeState {
        block_confirmed: Option<u32>,
        round_submitted: Option<u32>,
        round_active: Option<u32>,
        round_inactive: Option<u32>,
    }
}
serializable_struct_with_getters! {
    DifficultyParameter {
        name: String,
        min_value: i32,
        max_value: i32,
    }
}
serializable_struct_with_getters! {
    ChallengeBlockData {
        solution_signature_threshold: Option<u32>,
        num_qualifiers: Option<u32>,
        qualifier_difficulties: Option<HashSet<Point>>,
        base_frontier: Option<Frontier>,
        scaled_frontier: Option<Frontier>,
        scaling_factor: Option<f64>,
    }
}

// Player child structs
serializable_struct_with_getters! {
    PlayerDetails {
        name: String,
        is_multisig: bool,
    }
}
serializable_struct_with_getters! {
    PlayerBlockData {
        num_qualifiers_by_challenge: Option<HashMap<String, u32>>,
        cutoff: Option<u32>,
        imbalance: Option<PreciseNumber>,
        imbalance_penalty: Option<PreciseNumber>,
        influence: Option<PreciseNumber>,
        reward: Option<PreciseNumber>,
        round_earnings: Option<PreciseNumber>,
    }
}

// Proof child structs
serializable_struct_with_getters! {
    ProofState {
        block_confirmed: Option<u32>,
        submission_delay: Option<u32>,
    }
}
pub type Solution = Map<String, Value>;
serializable_struct_with_getters! {
    SolutionData {
        nonce: u32,
        runtime_signature: u32,
        fuel_consumed: u64,
        solution: Solution,
    }
}
impl SolutionData {
    pub fn calc_solution_signature(&self) -> u32 {
        u32_from_str(&jsonify(self))
    }
}

// Fraud child structs
serializable_struct_with_getters! {
    FraudState {
        block_confirmed: Option<u32>,
    }
}
// Wasm child structs
serializable_struct_with_getters! {
    WasmDetails {
        compile_success: bool,
        download_url: Option<String>,
        checksum: Option<String>,
    }
}
serializable_struct_with_getters! {
    WasmState {
        block_confirmed: Option<u32>,
    }
}
