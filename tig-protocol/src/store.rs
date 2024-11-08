pub trait BlocksStore 
{
    type Block;
    type BlockDetails;
    type BlockData;
    type ProtocolConfig;

    fn get(&self, block_id: &String)                            -> Option<&Self::Block>;
    fn get_details(&self, block_id: &String)                    -> Option<&Self::BlockDetails>;
    fn get_data(&self, block_id: &String)                       -> Option<&Self::BlockData>;
    fn get_config(&self, block_id: &String)                     -> Option<&Self::ProtocolConfig>;
    fn get_next_id(&self)                                       -> String;
}

pub trait AlgorithmsStore 
{
    type Block;
    type AlgorithmDetails;
    type AlgorithmState;
    type AlgorithmBlockData;

    fn get(&self, block_id: &String)                            -> Option<&Self::Block>;
    fn get_details(&self, algorithm_id: &String)                -> Option<&Self::AlgorithmDetails>;
    fn get_state(&self, algorithm_id: &String)                  -> Option<&Self::AlgorithmState>;
    fn get_data(&self, algorithm_id: &String)                   -> Option<&Self::AlgorithmBlockData>;
}

pub trait PrecommitsStore 
{
    type Precommit;
    type PrecommitDetails;
    type BenchmarkSettings;
    type PrecommitState;

    fn get(&self, benchmark_id: &String)                        -> Option<&Self::Precommit>;
    fn get_details(&self, benchmark_id: &String)                -> Option<&Self::PrecommitDetails>;
    fn get_settings(&self, benchmark_id: &String)               -> Option<&Self::BenchmarkSettings>;
    fn get_state(&self, benchmark_id: &String)                  -> Option<&Self::PrecommitState>;
    fn calc_benchmark_id(&self, settings: &Self::BenchmarkSettings)   -> String;
}

pub trait BenchmarksStore 
{
    type Benchmark;
    type BenchmarkDetails;
    type BenchmarkState;
    type HashSet;

    fn get(&self, benchmark_id: &String)                        -> Option<&Self::Benchmark>;
    fn get_details(&self, benchmark_id: &String)                -> Option<&Self::BenchmarkDetails>;
    fn get_state(&self, benchmark_id: &String)                  -> Option<&Self::BenchmarkState>;
    fn get_solution_nonces(&self, benchmark_id: &String)        -> Option<&Self::HashSet>;
}

pub trait ProofsStore 
{
    type Proof;
    type ProofState;
    type MerkleProof;

    fn get(&self, benchmark_id: &String)                        -> Option<&Self::Proof>;
    fn get_state(&self, benchmark_id: &String)                  -> Option<&Self::ProofState>;
    fn get_merkle_proofs(&self, benchmark_id: &String)          -> Option<&Vec<Self::MerkleProof>>;
}

pub trait FraudsStore 
{
    type Fraud;
    type FraudState;

    fn get(&self, benchmark_id: &String)                        -> Option<&Self::Fraud>;
    fn get_state(&self, benchmark_id: &String)                  -> Option<&Self::FraudState>;
    fn get_allegations(&self, benchmark_id: &String)            -> Option<&String>;
}

pub trait ChallengesStore 
{
    type Challenge;
    type ChallengeDetails;
    type ChallengeState;
    type ChallengeBlockData;

    fn get(&self, challenge_id: &String)                        -> Option<&Self::Challenge>;
    fn get_details(&self, challenge_id: &String)                -> Option<&Self::ChallengeDetails>;
    fn get_state(&self, challenge_id: &String)                  -> Option<&Self::ChallengeState>;
    fn get_data(&self, challenge_id: &String)                   -> Option<&Self::ChallengeBlockData>;
}

pub trait TopUpsStore 
{
    type TopUp;
    type TopUpDetails;
    type TopUpState;

    fn get(&self, tx_hash: &String)                            -> Option<&Self::TopUp>;
    fn get_details(&self, tx_hash: &String)                    -> Option<&Self::TopUpDetails>;
    fn get_state(&self, tx_hash: &String)                      -> Option<&Self::TopUpState>;
}

pub trait WasmsStore 
{
    type Wasm;
    type WasmDetails;
    type WasmState;

    fn get(&self, wasm_id: &String)                            -> Option<&Self::Wasm>;
    fn get_details(&self, wasm_id: &String)                    -> Option<&Self::WasmDetails>;
    fn get_state(&self, wasm_id: &String)                      -> Option<&Self::WasmState>;
}