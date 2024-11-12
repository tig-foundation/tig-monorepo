pub use anyhow::{Error as ContextError, Result as ContextResult};
use {
    std::sync::Arc,
    tig_structs::{config::*, core::*, *},
    std::sync::RwLock,
    std::collections::HashSet,
};

pub trait Context 
{
    fn verify_solution(&self, settings: &BenchmarkSettings, nonce: u64, solution: &Solution)            -> ContextResult<anyhow::Result<()>>;
    fn compute_solution(&self, settings: &BenchmarkSettings, nonce: u64, wasm_vm_config: &WasmVMConfig) -> ContextResult<anyhow::Result<OutputData>>;
  
    fn notify_add_new_block(&self) -> ContextResult<String>;

    fn get_config(&self) -> &ProtocolConfig;

    //add_
    fn add_precommit(&self, settings: &BenchmarkSettings, details: &PrecommitDetails) -> ContextResult<String>;
    fn add_benchmark(&self, benchmark_id: &String, merkle_root: &MerkleHash, solution_nonces: &HashSet<u64>) -> ContextResult<String>;

    // PlayersContract functions
    fn get_player_deposit(&self, eth_block_num: &String, player_id: &String) -> ContextResult<Option<PreciseNumber>>;
    
    fn get_player_state(&self, player_id: &String)      -> Option<&PlayerState>;
    fn get_player_state_mut(&self, player_id: &String)  -> Option<&mut PlayerState>;

    fn get_player_block_data(&self, player_id: &String)     -> Option<&PlayerBlockData>;
    fn get_player_block_data_mut(&self, player_id: &String) -> Option<&mut PlayerBlockData>;

    // BlocksStore functions
    fn get_block_details(&self, block_id: &String)  -> Option<&BlockDetails>;
    fn get_block_data(&self, block_id: &String)     -> Option<&BlockData>;
    fn get_block_config(&self, block_id: &String)   -> Option<&ProtocolConfig>;
    fn get_next_block_id(&self)                     -> &String;

    // AlgorithmsStore functions
    fn get_algorithm_details(&self, algorithm_id: &String)                  -> Option<&AlgorithmDetails>;
    fn get_algorithm_state(&self, algorithm_id: &String, block_id: &String) -> Option<&AlgorithmState>;

    fn get_algorithm_data(&self, algorithm_id: &String, block_id: &String)      -> Option<&AlgorithmBlockData>;
    fn get_algorithm_data_mut(&self, algorithm_id: &String, block_id: &String)  -> Option<&mut AlgorithmBlockData>;

    fn get_confirmed_algorithms(&self)                                          -> Vec<&Algorithm>;
    // PrecommitsStore functions
    fn get_precommit_details(&self, benchmark_id: &String)      -> Option<&PrecommitDetails>;
    fn get_benchmark_settings(&self, benchmark_id: &String)     -> Option<&BenchmarkSettings>;
    fn get_precommit_state(&self, benchmark_id: &String)        -> Option<&PrecommitState>;
    fn calc_benchmark_id(&self, settings: &BenchmarkSettings)   -> &String;

    // BenchmarksStore functions
    fn get_benchmark_details(&self, benchmark_id: &String)  -> Option<&BenchmarkDetails>;
    fn get_benchmark_state(&self, benchmark_id: &String)    -> Option<&BenchmarkState>;
    fn get_solution_nonces(&self, benchmark_id: &String)    -> Option<&HashSet<i32>>;

    // ProofsStore functions
    fn get_proof_state(&self, benchmark_id: &String)    -> Option<&ProofState>;
    fn get_merkle_proofs(&self, benchmark_id: &String)  -> Option<&Vec<&MerkleProof>>;

    fn get_confirmed_proofs_by_height_started(&self, height: u64) -> Vec<&Proof>;
    // FraudsStore functions
    fn get_fraud_state(&self, benchmark_id: &String)        -> Option<&FraudState>;
    fn get_fraud_allegations(&self, benchmark_id: &String)  -> Option<&String>;

    fn get_frauds_by_benchmark_id(&self, benchmark_id: &String) -> Vec<&Fraud>;
    // ChallengesStore functions
    fn get_challenge_details(&self, challenge_id: &String)                      -> Option<&ChallengeDetails>;
    
    fn get_challenge_state(&self, challenge_id: &String, block_id: &String)     -> Option<&ChallengeState>;
    fn get_challenge_state_mut(&self, challenge_id: &String, block_id: &String) -> Option<&mut ChallengeState>;

    fn get_challenge_data(&self, challenge_id: &String, block_id: &String)      -> Option<&ChallengeBlockData>;
    fn get_challenge_data_mut(&self, challenge_id: &String, block_id: &String)  -> Option<&mut ChallengeBlockData>;

    fn get_confirmed_challenges(&self)  -> Vec<&Challenge>;
    // TopUpsStore functions
    fn get_top_up(&self, tx_hash: &String)          -> Option<&TopUp>;
    fn get_top_up_details(&self, tx_hash: &String)  -> Option<&TopUpDetails>;
    fn get_top_up_state(&self, tx_hash: &String)    -> Option<&TopUpState>;

    // WasmsStore functions
    fn get_wasm(&self, wasm_id: &String)            -> Option<&Wasm>;
    fn get_wasm_details(&self, wasm_id: &String)    -> Option<&WasmDetails>;
    fn get_wasm_state(&self, wasm_id: &String)      -> Option<&WasmState>;

    fn get_wasm_by_algorithm_id(&self, algorithm_id: &String) -> Option<&Wasm>;
}
