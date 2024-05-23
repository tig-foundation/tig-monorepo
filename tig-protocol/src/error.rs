use tig_structs::{config::DifficultyParameter, core::BenchmarkSettings};

#[derive(Debug, PartialEq)]
pub enum ProtocolError {
    DifficultyAboveHardestFrontier {
        difficulty: Vec<i32>,
    },
    DifficultyBelowEasiestFrontier {
        difficulty: Vec<i32>,
    },
    DuplicateBenchmarkSettings {
        settings: BenchmarkSettings,
    },
    DuplicateNonce {
        nonce: u32,
    },
    DuplicateProof {
        benchmark_id: String,
    },
    DuplicateSubmissionFeeTx {
        tx_hash: String,
    },
    FlaggedAsFraud {
        benchmark_id: String,
    },
    InsufficientLifespan,
    InsufficientSolutions {
        min_num_solutions: usize,
        num_solutions: usize,
    },
    InvalidAlgorithm {
        algorithm_id: String,
    },
    InvalidBenchmark {
        benchmark_id: String,
    },
    InvalidBenchmarkNonce {
        nonce: u32,
    },
    InvalidBlock {
        block_id: String,
    },
    InvalidChallenge {
        challenge_id: String,
    },
    InvalidDifficulty {
        difficulty: Vec<i32>,
        difficulty_parameters: Vec<DifficultyParameter>,
    },
    InvalidProofNonces {
        expected_nonces: Vec<u32>,
        submitted_nonces: Vec<u32>,
    },
    InvalidSignatureFromSolutionData {
        actual_signature: u32,
        nonce: u32,
        expected_signature: u32,
    },
    InvalidSolution {
        nonce: u32,
    },
    InvalidSolutionData {
        algorithm_id: String,
        nonce: u32,
    },
    InvalidSolutionSignature {
        nonce: u32,
        solution_signature: u32,
        threshold: u32,
    },
    InvalidSubmittingPlayer {
        expected_player_id: String,
        actual_player_id: String,
    },
    InvalidSubmissionFeeAmount {
        expected_amount: String,
        actual_amount: String,
        tx_hash: String,
    },
    InvalidSubmissionFeeReceiver {
        tx_hash: String,
        expected_receiver: String,
        actual_receiver: String,
    },
    InvalidSubmissionFeeSender {
        tx_hash: String,
        expected_sender: String,
        actual_sender: String,
    },
    InvalidTransaction {
        tx_hash: String,
    },
}

impl std::fmt::Display for ProtocolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProtocolError::DifficultyAboveHardestFrontier {
                difficulty,
            } => write!(
                f,
                "Difficulty '{:?}' is above the hardest allowed frontier",
                difficulty
            ),
            ProtocolError::DifficultyBelowEasiestFrontier {
                difficulty,
            } => write!(
                f,
                "Difficulty '{:?}' is below the easiest allowed frontier",
                difficulty
            ),
            ProtocolError::DuplicateBenchmarkSettings { settings }=> {
                write!(f, "A benchmark with settings '{:?}' has been submitted before.", settings)
            }
            ProtocolError::DuplicateNonce { nonce } => write!(
                f,
                "Nonce '{}' is submitted more than once",
                nonce
            ),
            ProtocolError::DuplicateProof { benchmark_id } => {
                write!(f, "Proof already submitted for benchmark '{}'", benchmark_id)
            }
            ProtocolError::DuplicateSubmissionFeeTx { tx_hash } => write!(
                f,
                "Transaction '{}' is already used",
                tx_hash
            ),
            ProtocolError::FlaggedAsFraud { benchmark_id } => {
                write!(f, "Benchmark '{}' is flagged as fraud", benchmark_id)
            }
            ProtocolError::InsufficientLifespan => {
                write!(f, "Benchmark will have no lifespan remaining after submission delay penalty is applied.")
            }
            ProtocolError::InsufficientSolutions {
                num_solutions,
                min_num_solutions,
            } => write!(
                f,
                "Insufficient number of solutions. Expected: '{}', Actual: '{}'",
                min_num_solutions, num_solutions
            ),
            ProtocolError::InvalidAlgorithm { algorithm_id } => {
                write!(f, "Algorithm '{}' does not exist or is not yet active", algorithm_id)
            }
            ProtocolError::InvalidBenchmark { benchmark_id } => {
                write!(f, "Benchmark '{}' does not exist", benchmark_id)
            }
            ProtocolError::InvalidBenchmarkNonce { nonce } => {
                write!(f, "Benchmark nonce '{}' is invalid. Must exist in solutions_meta_data", nonce)
            }
            ProtocolError::InvalidBlock { block_id } => {
                write!(f, "Block '{}' does not exist", block_id)
            }
            ProtocolError::InvalidChallenge { challenge_id } => {
                write!(f, "Challenge '{}' either does not exist or in not yet active", challenge_id)
            }
            ProtocolError::InvalidDifficulty {
                difficulty,
                difficulty_parameters,
            } => write!(
                f,
                "Difficulty '{:?}' is invalid. Must match difficulty parameters '{:?}'",
                difficulty, difficulty_parameters
            ),
            ProtocolError::InvalidProofNonces {
                submitted_nonces,
                expected_nonces: sampled_nonces,
            } => write!(
                f,
                "Submitted nonces are invalid. Expected: '{:?}', Submitted '{:?}'",
                sampled_nonces, submitted_nonces
            ),
            ProtocolError::InvalidSignatureFromSolutionData {
                nonce,
                expected_signature,
                actual_signature,
            } => write!(
                f,
                "Solution data for nonce '{}' produces invalid solution signature. Expected: '{}', Actual: '{}'",
                nonce, expected_signature, actual_signature
            ),
            ProtocolError::InvalidSolution { nonce } => {
                write!(f, "Solution for nonce '{}' is invalid", nonce)
            }
            ProtocolError::InvalidSolutionData {
                algorithm_id,
                nonce,
            } => write!(
                f,
                "The solution data for nonce '{}' is invalid. Does not match the solution data re-computed using algorithm '{}'.",
                nonce, algorithm_id
            ),
            ProtocolError::InvalidSolutionSignature {
                nonce,
                solution_signature,
                threshold,
            } => write!(
                f,
                "Solution signature '{}' for nonce '{}' is invalid. Must be less than or equal to threshold '{}'",
                solution_signature, nonce, threshold
            ),
            ProtocolError::InvalidSubmittingPlayer {
                expected_player_id,
                actual_player_id,
            } => write!(
                f,
                "Submission made by the invalid player. Expected: '{}', Actual: '{}'",
                expected_player_id, actual_player_id
            ),
            ProtocolError::InvalidSubmissionFeeAmount {
                expected_amount,
                actual_amount,
                tx_hash,
            } => write!(
                f,
                "Transaction '{}' paid an invalid amount of submission fee. Expected: '{}', Actual: '{}'",
                tx_hash, expected_amount, actual_amount
            ),
            ProtocolError::InvalidSubmissionFeeReceiver { tx_hash, expected_receiver, actual_receiver } => write!(
                f,
                "Transaction '{}' has invalid receiver. Expected: '{}', Actual: '{}'",
                tx_hash, expected_receiver, actual_receiver
            ),
            ProtocolError::InvalidSubmissionFeeSender {
                tx_hash,
                expected_sender,
                actual_sender,
            } => write!(
                f,
                "Transaction '{}' has invalid sender. Expected: '{}', Actual: '{}'",
                tx_hash, expected_sender, actual_sender
            ),
            ProtocolError::InvalidTransaction {
                tx_hash,
            } => write!(
                f,
                "Transaction '{}' is invalid",
                tx_hash
            ),
        }
    }
}

impl std::error::Error for ProtocolError {}

pub type ProtocolResult<T> = std::result::Result<T, ProtocolError>;
