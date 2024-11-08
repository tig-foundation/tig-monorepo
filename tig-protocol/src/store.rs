use{
    std::collections::HashSet,
    tig_structs::{
        *,
        core::*,
        config::*,
    },
};

pub trait BlocksStore 
{
    fn get(&self, block_id: &String)                                -> Option<&Block>;
    fn get_details(&self, block_id: &String)                        -> Option<&BlockDetails>;
    fn get_data(&self, block_id: &String)                           -> Option<&BlockData>;
    fn get_config(&self, block_id: &String)                         -> Option<&ProtocolConfig>;
    fn get_next_id(&self)                                           -> &String;
}

pub trait AlgorithmsStore 
{
    fn get(&self, algorithm_id: &String)                            -> Option<&Algorithm>;
    fn get_details(&self, algorithm_id: &String)                    -> Option<&AlgorithmDetails>;
    fn get_state(&self, algorithm_id: &String)                      -> Option<&AlgorithmState>;
    fn get_data(&self, algorithm_id: &String)                       -> Option<&AlgorithmBlockData>;
}

pub trait PrecommitsStore 
{
    fn get(&self, benchmark_id: &String)                            -> Option<&Precommit>;
    fn get_details(&self, benchmark_id: &String)                    -> Option<&PrecommitDetails>;
    fn get_settings(&self, benchmark_id: &String)                   -> Option<&BenchmarkSettings>;
    fn get_state(&self, benchmark_id: &String)                      -> Option<&PrecommitState>;
    fn calc_benchmark_id(&self, settings: &BenchmarkSettings)       -> &String;
}

pub trait BenchmarksStore 
{
    fn get(&self, benchmark_id: &String)                            -> Option<&Benchmark>;
    fn get_details(&self, benchmark_id: &String)                    -> Option<&BenchmarkDetails>;
    fn get_state(&self, benchmark_id: &String)                      -> Option<&BenchmarkState>;
    fn get_solution_nonces(&self, benchmark_id: &String)            -> Option<&HashSet<i32>>;
}

pub trait ProofsStore 
{
    fn get(&self, benchmark_id: &String)                            -> Option<&Proof>;
    fn get_state(&self, benchmark_id: &String)                      -> Option<&ProofState>;
    fn get_merkle_proofs(&self, benchmark_id: &String)              -> Option<&Vec<&MerkleProof>>;
}

pub trait FraudsStore 
{
    fn get(&self, benchmark_id: &String)                            -> Option<&Fraud>;
    fn get_state(&self, benchmark_id: &String)                      -> Option<&FraudState>;
    fn get_allegations(&self, benchmark_id: &String)                -> Option<&String>;
}

pub trait ChallengesStore 
{
    fn get(&self, challenge_id: &String)                            -> Option<Challenge>;
    fn get_details(&self, challenge_id: &String)                    -> Option<ChallengeDetails>;
    fn get_state(&self, challenge_id: &String)                      -> Option<ChallengeState>;
    fn get_data(&self, challenge_id: &String)                       -> Option<ChallengeBlockData>;
}

pub trait TopUpsStore 
{
    fn get(&self, tx_hash: &String)                                 -> Option<&TopUp>;
    fn get_details(&self, tx_hash: &String)                         -> Option<&TopUpDetails>;
    fn get_state(&self, tx_hash: &String)                           -> Option<&TopUpState>;
}

pub trait WasmsStore 
{
    fn get(&self, wasm_id: &String)                                 -> Option<&Wasm>;
    fn get_details(&self, wasm_id: &String)                         -> Option<&WasmDetails>;
    fn get_state(&self, wasm_id: &String)                           -> Option<&WasmState>;
}