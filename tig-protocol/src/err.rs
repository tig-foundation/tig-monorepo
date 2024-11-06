use tig_structs::{config::DifficultyParameter, core::BenchmarkSettings};
use tig_utils::PreciseNumber;
use std::collections::HashSet;

#[derive(Debug, PartialEq)]
pub enum ProtocolError<'a> 
{
    DifficultyAboveHardestFrontier 
    {
        difficulty:                     &'a Vec<i32>,
    },
    DifficultyBelowEasiestFrontier 
    {
        difficulty:                     &'a Vec<i32>,
    },
    DuplicateBenchmark 
    {
        benchmark_id:                   &'a String,
    },
    DuplicateBenchmarkSettings 
    {
        settings:                       &'a BenchmarkSettings,
    },
    DuplicateNonce 
    {
        nonce:                          u64,
    },
    DuplicateProof 
    {
        benchmark_id:                   &'a String,
    },
    DuplicateTransaction 
    {
        tx_hash:                        &'a String,
    },
    FlaggedAsFraud 
    {
        benchmark_id:                   &'a String,
    },
    InsufficientLifespan,
    InsufficientSolutions 
    {
        min_num_solutions:              usize,
        num_solutions:                  usize,
    },
    InsufficientFeeBalance 
    {
        fee_paid:                       PreciseNumber,
        available_fee_balance:          PreciseNumber,
    },
    InvalidAlgorithm 
    {
        algorithm_id:                   &'a String,
    },
    InvalidBenchmark 
    {
        benchmark_id:                   &'a String,
    },
    InvalidBenchmarkNonce 
    {
        nonce:                          u64,
    },
    InvalidBlock 
    {
        block_id:                       &'a String,
    },
    InvalidChallenge 
    {
        challenge_id:                   &'a String,
    },
    InvalidDifficulty 
    {
        difficulty:                     &'a Vec<i32>,
        difficulty_parameters:          &'a Vec<DifficultyParameter>,
    },
    InvalidMerkleProof 
    {
        nonce:                          u64,
    },
    InvalidNumNonces 
    {
        num_nonces:                     u32,
    },
    InvalidPrecommit 
    {
        benchmark_id:                   &'a String,
    },
    InvalidProofNonces 
    {
        expected_nonces:                &'a HashSet<u64>,
        submitted_nonces:               Vec<u64>,
    },
    InvalidSignatureFromSolutionData 
    {
        actual_signature:               u32,
        nonce:                          u64,
        expected_signature:             u32,
    },
    InvalidSolution 
    {
        nonce:                          u64,
    },
    InvalidSolutionData 
    {
        algorithm_id:                   &'a String,
        nonce:                          u64,
    },
    InvalidSolutionSignature 
    {
        nonce:                          u64,
        solution_signature:             u32,
        threshold:                      u32,
    },
    InvalidSubmittingPlayer 
    {
        expected_player_id:             &'a String,
        actual_player_id:               &'a String,
    },
    InvalidTransactionAmount 
    {
        expected_amount:                PreciseNumber,
        actual_amount:                  PreciseNumber,
        tx_hash:                        &'a String,
    },
    InvalidTransactionReceiver 
    {
        tx_hash:                        &'a String,
        expected_receiver:              String,
        actual_receiver:                String,
    },
    InvalidTransactionSender 
    {
        tx_hash:                        &'a String,
        expected_sender:                String,
        actual_sender:                  String,
    },
    InvalidTransaction 
    {
        tx_hash:                        &'a String,
    },
}

impl<'a> std::fmt::Display for ProtocolError<'_>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result 
    {
        match self 
        {
            ProtocolError::DifficultyAboveHardestFrontier { difficulty } => 
            {
                write!(f, "Difficulty '{:?}' is above the hardest allowed frontier", difficulty)
            },

            ProtocolError::DifficultyBelowEasiestFrontier { difficulty } => 
            {
                write!(f, "Difficulty '{:?}' is below the easiest allowed frontier", difficulty)
            },

            ProtocolError::DuplicateBenchmark { benchmark_id } => 
            {
                write!(f, "Benchmark already submitted for precommit '{}'", benchmark_id)
            }

            ProtocolError::DuplicateBenchmarkSettings { settings } => 
            {
                write!(f, "A benchmark with settings '{:?}' has been submitted before.", settings)
            }

            ProtocolError::DuplicateNonce { nonce } => 
            {
                write!(f, "Nonce '{}' is submitted more than once", nonce)
            },

            ProtocolError::DuplicateProof { benchmark_id } => 
            {
                write!(f, "Proof already submitted for benchmark '{}'", benchmark_id)
            }
            
            ProtocolError::DuplicateTransaction { tx_hash } => 
            {
                write!(f, "Transaction '{}' is already used", tx_hash)
            }
            ProtocolError::FlaggedAsFraud { benchmark_id } => 
            {
                write!(f, "Benchmark '{}' is flagged as fraud", benchmark_id)
            }
            
            ProtocolError::InsufficientFeeBalance { fee_paid, available_fee_balance } => 
            {
                write!(f, "Insufficient fee balance. Fee to be paid: '{}', Available fee balance: '{}'", fee_paid, available_fee_balance)
            },
            ProtocolError::InsufficientLifespan => 
            {
                write!(f, "Benchmark will have no lifespan remaining after submission delay penalty is applied.")
            },
            ProtocolError::InsufficientSolutions { num_solutions, min_num_solutions } => 
            {
                write!(f, "Insufficient number of solutions. Expected: '{}', Actual: '{}'", min_num_solutions, num_solutions)
            },
            ProtocolError::InvalidAlgorithm { algorithm_id } => 
            {
                write!(f, "Algorithm '{}' does not exist, is not yet active, or is banned", algorithm_id)
            },
            ProtocolError::InvalidBenchmark { benchmark_id } => 
            {
                write!(f, "Benchmark '{}' does not exist", benchmark_id)
            },
            ProtocolError::InvalidBenchmarkNonce { nonce } => {
                write!(f, "Benchmark nonce '{}' is invalid. Must exist in solutions_meta_data", nonce)
            },
            ProtocolError::InvalidBlock { block_id } => 
            {
                write!(f, "Block '{}' does not exist", block_id)
            },
            ProtocolError::InvalidChallenge { challenge_id } => 
            {
                write!(f, "Challenge '{}' either does not exist or in not yet active", challenge_id)
            },
            ProtocolError::InvalidDifficulty { difficulty, difficulty_parameters } => 
            {
                write!(f, "Difficulty '{:?}' is invalid. Must match difficulty parameters '{:?}'", difficulty, difficulty_parameters)
            },
            ProtocolError::InvalidMerkleProof { nonce } => 
            {
                write!(f, "Merkle proof for nonce '{}' is invalid", nonce)
            },
            ProtocolError::InvalidNumNonces { num_nonces } => 
            {
                write!(f, "Number of nonces '{}' is invalid", num_nonces)
            },
            ProtocolError::InvalidPrecommit { benchmark_id } =>
            {
                write!(f, "Precommit '{}' does not exist", benchmark_id)
            }
            ProtocolError::InvalidProofNonces { submitted_nonces, expected_nonces: sampled_nonces } => 
            {
                write!(f, "Submitted nonces are invalid. Expected: '{:?}', Submitted '{:?}'", sampled_nonces, submitted_nonces)
            },
            ProtocolError::InvalidSignatureFromSolutionData { nonce, expected_signature, actual_signature } => 
            {
                write!(f, "Solution data for nonce '{}' produces invalid solution signature. Expected: '{}', Actual: '{}'", nonce, expected_signature, actual_signature)
            },
            ProtocolError::InvalidSolution { nonce } => 
            {
                write!(f, "Solution for nonce '{}' is invalid", nonce)
            },
            ProtocolError::InvalidSolutionData { algorithm_id, nonce } => 
            {
                write!(f, "The solution data for nonce '{}' is invalid. Does not match the solution data re-computed using algorithm '{}'.", nonce, algorithm_id)
            },
            ProtocolError::InvalidSolutionSignature { nonce, solution_signature, threshold } => 
            {
                write!(f, "Solution signature '{}' for nonce '{}' is invalid. Must be less than or equal to threshold '{}'", solution_signature, nonce, threshold)
            },
            ProtocolError::InvalidSubmittingPlayer { expected_player_id, actual_player_id } => 
            {
                write!(f, "Submission made by the invalid player. Expected: '{}', Actual: '{}'", expected_player_id, actual_player_id)
            }
            ProtocolError::InvalidTransactionAmount { expected_amount, actual_amount, tx_hash } => 
            {
                write!(f, "Transaction '{}' paid an invalid amount of submission fee. Expected: '{}', Actual: '{}'", tx_hash, expected_amount, actual_amount)
            },
            ProtocolError::InvalidTransactionReceiver { tx_hash, expected_receiver, actual_receiver } => 
            {
                write!(f, "Transaction '{}' has invalid receiver. Expected: '{}', Actual: '{}'", tx_hash, expected_receiver, actual_receiver)
            },
            ProtocolError::InvalidTransactionSender { tx_hash, expected_sender, actual_sender } => 
            {
                write!(f, "Transaction '{}' has invalid sender. Expected: '{}', Actual: '{}'", tx_hash, expected_sender, actual_sender)
            },
            ProtocolError::InvalidTransaction { tx_hash } => 
            {
                write!(f, "Transaction '{}' is invalid", tx_hash)
            },
        }
    }
}

impl std::error::Error for ProtocolError<'_> {}
pub type ProtocolResult<'a, T> = std::result::Result<T, ProtocolError<'a>>;
pub type ContractResult<'a, T> = std::result::Result<T, ProtocolError<'a>>;
