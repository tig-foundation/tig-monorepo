use tig_structs::{config::DifficultyParameter, core::BenchmarkSettings};
use tig_utils::PreciseNumber;
use std::collections::HashSet;

pub type ProtocolError      = String;
pub type ProtocolResult<T>  = std::result::Result<T, ProtocolError>;
pub type ContractResult<T>  = std::result::Result<T, ProtocolError>;
