pub use anyhow::{Error as ContextError, Result as ContextResult};
use {
    std::sync::Arc,
    tig_structs::{config::*, core::*, *},
};

pub trait BlocksStore {
    fn get(&self, block_id: &String) -> Option<&Block>;

    fn get_details(&self, block_id: &String) -> Option<&BlockDetails>;

    fn get_data(&self, block_id: &String) -> Option<&BlockData>;

    fn get_config(&self, block_id: &String) -> Option<&ProtocolConfig>;

    fn get_next_id(&self) -> String;
}

pub trait AlgorithmsStore {
    fn get(&self, block_id: &String) -> Option<&Block>;

    fn get_details(&self, algorithm_id: &String) -> Option<&AlgorithmDetails>;

    fn get_state(&self, algorithm_id: &String) -> Option<&AlgorithmState>;

    fn get_data(&self, algorithm_id: &String) -> Option<&AlgorithmBlockData>;
}

pub trait PrecommitsStore {
    fn get(&self, benchmark_id: &String) -> Option<&Precommit>;

    fn get_details(&self, benchmark_id: &String) -> Option<&PrecommitDetails>;

    fn get_settings(&self, benchmark_id: &String) -> Option<&BenchmarkSettings>;

    fn get_state(&self, benchmark_id: &String) -> Option<&PrecommitState>;

    fn calc_benchmark_id(&self, settings: &BenchmarkSettings) -> String;
}

pub trait BenchmarksStore {
    fn get(&self, benchmark_id: &String) -> Option<&Benchmark>;

    fn get_details(&self, benchmark_id: &String) -> Option<&BenchmarkDetails>;

    fn get_state(&self, benchmark_id: &String) -> Option<&BenchmarkState>;

    fn get_solution_nonces(&self, benchmark_id: &String) -> Option<&HashSet<u64>>;
}

pub trait ProofsStore {
    fn get(&self, benchmark_id: &String) -> Option<&Proof>;

    fn get_state(&self, benchmark_id: &String) -> Option<&ProofState>;

    fn get_merkle_proofs(&self, benchmark_id: &String) -> Option<&Vec<MerkleProof>>;
}

pub trait FraudsStore {
    fn get(&self, benchmark_id: &String) -> Option<&Fraud>;

    fn get_state(&self, benchmark_id: &String) -> Option<&FraudState>;

    fn get_allegations(&self, benchmark_id: &String) -> Option<&String>;
}

pub trait ChallengesStore {
    fn get(&self, challenge_id: &String) -> Option<&Challenge>;

    fn get_details(&self, challenge_id: &String) -> Option<&ChallengeDetails>;

    fn get_state(&self, challenge_id: &String) -> Option<&ChallengeState>;

    fn get_data(&self, challenge_id: &String) -> Option<&ChallengeBlockData>;
}

pub trait TopUpsStore {
    fn get(&self, tx_hash: &String) -> Option<&TopUp>;

    fn get_details(&self, tx_hash: &String) -> Option<&TopUpDetails>;

    fn get_state(&self, tx_hash: &String) -> Option<&TopUpState>;
}

pub trait WasmsStore {
    fn get(&self, wasm_id: &String) -> Option<&Wasm>;

    fn get_details(&self, wasm_id: &String) -> Option<&WasmDetails>;

    fn get_state(&self, wasm_id: &String) -> Option<&WasmState>;
}

// TODO Vote
// TODO Deposit
// TODO Breakthrough
// TODO Binary

pub struct Context {
    pub blocks: Arc<dyn BlocksStore>,
    pub algorithms: Arc<dyn AlgorithmsStore>,
    pub challenges: Arc<dyn ChallengesStore>,
    pub precommits: Arc<dyn PrecommitsStore>,
    pub benchmarks: Arc<dyn BenchmarksStore>,
    pub proofs: Arc<dyn ProofsStore>,
    pub frauds: Arc<dyn FraudsStore>,
    pub wasms: Arc<dyn WasmsStore>,
}

// pub trait Context {
//     fn get_block_by_height(&self, block_height: i64) -> Option<Block>;

//     fn get_block_by_id(&self, block_id: &String) -> Option<&Block>;

//     fn get_next_block(&self) -> &Block;

//     fn get_algorithm_by_id(&self, algorithm_id: &String) -> Option<Algorithm>;

//     fn get_algorithm_by_tx_hash(&self, tx_hash: &String) -> Option<Algorithm>;

//     fn get_challenges_by_id(&self, challenge_id: &String) -> Vec<Challenge>;

//     fn get_challenge_by_id_and_height(
//         &self,
//         challenge_id: &String,
//         block_height: u64,
//     ) -> Option<Challenge>;

//     fn get_challenge_by_id_and_block_id(
//         &self,
//         challenge_id: &String,
//         block_id: &String,
//     ) -> Option<Challenge>;

//     fn get_precommits_by_settings(&self, settings: &BenchmarkSettings) -> Vec<Precommit>;

//     fn get_precommits_by_benchmark_id(&self, benchmark_id: &String) -> Vec<Precommit>;

//     fn get_transaction(&self, tx_hash: &String) -> Option<Transaction>;

//     fn get_topups_by_tx_hash(&self, tx_hash: &String) -> Vec<TopUp>;

//     fn get_benchmarks_by_id(&self, benchmark_id: &String) -> Vec<Benchmark>;

//     fn get_proofs_by_benchmark_id(&self, benchmark_id: &String) -> Vec<Proof>;

//     fn get_player_deposit(
//         &self,
//         eth_block_num: &String,
//         player_id: &String,
//     ) -> Option<PreciseNumber>;

//     fn verify_solution(
//         &self,
//         settings: &BenchmarkSettings,
//         nonce: u64,
//         solution: &Solution,
//     ) -> ContextResult<anyhow::Result<()>>;

//     fn compute_solution(
//         &self,
//         settings: &BenchmarkSettings,
//         nonce: u64,
//         wasm_vm_config: &WasmVMConfig,
//     ) -> ContextResult<anyhow::Result<OutputData>>;

//     fn add_precommit_to_mempool(
//         &self,
//         settings: &BenchmarkSettings,
//         details: &PrecommitDetails,
//     ) -> ContextResult<String>;

//     fn add_benchmark_to_mempool(
//         &self,
//         benchmark_id: &String,
//         merkle_root: &MerkleHash,
//         solution_nonces: &HashSet<u64>,
//     ) -> ContextResult<()>;

//     fn add_proof_to_mempool(
//         &self,
//         benchmark_id: &String,
//         merkle_proofs: &Vec<MerkleProof>,
//     ) -> ContextResult<()>;

//     fn add_fraud_to_mempool(
//         &self,
//         benchmark_id: &String,
//         allegations: &String,
//     ) -> ContextResult<()>;

//     fn update_precommit_state(
//         &self,
//         benchmark_id: &String,
//         state: &PrecommitState,
//     ) -> ContextResult<()>;

//     fn update_algorithm_state(
//         &self,
//         algorithm_id: &String,
//         state: &AlgorithmState,
//     ) -> ContextResult<()>;

//     fn notify_new_block(&self);

//     fn block_assembled(&self, block: &Block);

//     fn data_committed(&self, block: &Block);
// }
