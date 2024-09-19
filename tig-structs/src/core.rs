use crate::{config::ProtocolConfig, serializable_struct_with_getters};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::{HashMap, HashSet};
use tig_utils::{jsonify, u64s_from_str};
pub use tig_utils::{Frontier, MerkleBranch, MerkleHash, Point, PreciseNumber, Transaction, U256};

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
        details: BenchmarkDetails,
        state: Option<BenchmarkState>,
        solution_nonces: Option<HashSet<u64>>,
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
        state: Option<PlayerState>,
        block_data: Option<PlayerBlockData>,
    }
}
serializable_struct_with_getters! {
    Precommit {
        benchmark_id: String,
        details: PrecommitDetails,
        settings: BenchmarkSettings,
        state: Option<PrecommitState>,
    }
}
serializable_struct_with_getters! {
    MerkleProof {
        leaf: OutputData,
        branch: Option<MerkleBranch>,
    }
}
serializable_struct_with_getters! {
    Proof {
        benchmark_id: String,
        state: Option<ProofState>,
        merkle_proofs: Option<Vec<MerkleProof>>,
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
    TopUp {
        id: String,
        details: TopUpDetails,
        state: Option<TopUpState>,
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
        banned: bool,
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
    pub fn calc_seeds(&self, nonce: u64) -> [u64; 8] {
        let mut seeds = u64s_from_str(jsonify(&self).as_str());
        for seed in seeds.iter_mut() {
            *seed ^= nonce;
        }
        seeds
    }
}
serializable_struct_with_getters! {
    BenchmarkDetails {
        num_solutions: u32,
        merkle_root: Option<MerkleHash>,
    }
}
serializable_struct_with_getters! {
    BenchmarkState {
        block_confirmed: Option<u32>,
        sampled_nonces: Option<Vec<u64>>,
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
        let u64s = u64s_from_str(&jsonify(&data));
        let mut hash = [0u8; 32];
        hash[0..8].copy_from_slice(&u64s[0].to_le_bytes());
        hash[8..16].copy_from_slice(&u64s[1].to_le_bytes());
        hash[16..24].copy_from_slice(&u64s[2].to_le_bytes());
        hash[24..32].copy_from_slice(&u64s[3].to_le_bytes());
        MerkleHash(hash)
    }
}

// Block child structs
serializable_struct_with_getters! {
    BlockDetails {
        prev_block_id: String,
        height: u32,
        round: u32,
        eth_block_num: Option<String>,
        fees_paid: Option<PreciseNumber>,
        num_confirmed_challenges: Option<u32>,
        num_confirmed_algorithms: Option<u32>,
        num_confirmed_benchmarks: Option<u32>,
        num_confirmed_precommits: Option<u32>,
        num_confirmed_proofs: Option<u32>,
        num_confirmed_frauds: Option<u32>,
        num_confirmed_topups: Option<u32>,
        num_confirmed_wasms: Option<u32>,
        num_active_challenges: Option<u32>,
        num_active_algorithms: Option<u32>,
        num_active_benchmarks: Option<u32>,
        num_active_players: Option<u32>,
    }
}
serializable_struct_with_getters! {
    BlockData {
        confirmed_challenge_ids: HashSet<String>,
        confirmed_algorithm_ids: HashSet<String>,
        confirmed_benchmark_ids: HashSet<String>,
        confirmed_precommit_ids: HashSet<String>,
        confirmed_proof_ids: HashSet<String>,
        confirmed_fraud_ids: HashSet<String>,
        confirmed_topup_ids: HashSet<String>,
        confirmed_wasm_ids: HashSet<String>,
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
    }
}
serializable_struct_with_getters! {
    ChallengeState {
        block_confirmed: Option<u32>,
        round_active: Option<u32>,
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
        base_fee: Option<PreciseNumber>,
        per_nonce_fee: Option<PreciseNumber>,
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
    PlayerState {
        total_fees_paid: Option<PreciseNumber>,
        available_fee_balance: Option<PreciseNumber>,
    }
}
serializable_struct_with_getters! {
    PlayerBlockData {
        num_qualifiers_by_challenge: Option<HashMap<String, u32>>,
        cutoff: Option<u32>,
        deposit: Option<PreciseNumber>,
        rolling_deposit: Option<PreciseNumber>,
        qualifying_percent_rolling_deposit: Option<PreciseNumber>,
        imbalance: Option<PreciseNumber>,
        imbalance_penalty: Option<PreciseNumber>,
        influence: Option<PreciseNumber>,
        reward: Option<PreciseNumber>,
        round_earnings: Option<PreciseNumber>,
    }
}

// Precommit child structs
serializable_struct_with_getters! {
    PrecommitDetails {
        block_started: u32,
        num_nonces: u64,
        fee_paid: PreciseNumber,
    }
}
serializable_struct_with_getters! {
    PrecommitState {
        block_confirmed: Option<u32>,
        rand_hash: Option<String>,
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

// Fraud child structs
serializable_struct_with_getters! {
    FraudState {
        block_confirmed: Option<u32>,
    }
}

// TopUp child structs
serializable_struct_with_getters! {
    TopUpDetails {
        player_id: String,
        amount: PreciseNumber,
    }
}
serializable_struct_with_getters! {
    TopUpState {
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
