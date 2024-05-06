use crate::{core::*, serializable_struct_with_getters};
use serde::{Deserialize, Serialize};

serializable_struct_with_getters! {
    RequestApiKeyReq {
        signature: String,
        address: String,
    }
}

serializable_struct_with_getters! {
    RequestApiKeyResp {
        api_key: String,
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub enum PlayerType {
    Benchmarker,
    Innovator,
}

impl PlayerType {
    pub fn to_string(self) -> String {
        match self {
            PlayerType::Benchmarker => "benchmarker".to_string(),
            PlayerType::Innovator => "innovator".to_string(),
        }
    }

    pub fn from_string(s: String) -> Result<Self, String> {
        match s.as_str() {
            "benchmarker" => Ok(PlayerType::Benchmarker),
            "innovator" => Ok(PlayerType::Innovator),
            _ => Err("Invalid PlayerType".to_string()),
        }
    }
}

serializable_struct_with_getters! {
    GetPlayersReq {
        block_id: String,
        player_type: PlayerType,
    }
}

serializable_struct_with_getters! {
    GetPlayersResp {
        block_id: String,
        block_details: BlockDetails,
        players: Vec<Player>,
    }
}

serializable_struct_with_getters! {
    GetBlockReq {
        id: Option<String>,
        round: Option<u32>,
        height: Option<u32>,
        include_data: bool,
    }
}

serializable_struct_with_getters! {
    GetBlockResp {
        block: Option<Block>,
    }
}

serializable_struct_with_getters! {
    GetChallengesReq {
        block_id: String,
    }
}

serializable_struct_with_getters! {
    GetChallengesResp {
        block_id: String,
        block_details: BlockDetails,
        challenges: Vec<Challenge>,
    }
}

serializable_struct_with_getters! {
    GetAlgorithmsReq {
        block_id: String,
    }
}

serializable_struct_with_getters! {
    GetAlgorithmsResp {
        block_id: String,
        block_details: BlockDetails,
        algorithms: Vec<Algorithm>,
        wasms: Vec<Wasm>,
    }
}

serializable_struct_with_getters! {
    GetBenchmarksReq {
        block_id: String,
        player_id: String,
    }
}

serializable_struct_with_getters! {
    GetBenchmarksResp {
        block_id: String,
        block_details: BlockDetails,
        benchmarks: Vec<Benchmark>,
        proofs: Vec<Proof>,
        frauds: Vec<Fraud>,
    }
}

serializable_struct_with_getters! {
    GetBenchmarkDataReq {
        benchmark_id: String,
    }
}

serializable_struct_with_getters! {
    GetBenchmarkDataResp {
        benchmark: Option<Benchmark>,
        proof: Option<Proof>,
        fraud: Option<Fraud>,
    }
}

serializable_struct_with_getters! {
    SubmitBenchmarkReq {
        settings: BenchmarkSettings,
        solutions_meta_data: Vec<SolutionMetaData>,
        solution_data: SolutionData,
    }
}

serializable_struct_with_getters! {
    SubmitBenchmarkResp {
        benchmark_id: String,
        verified: Result<(), String>,
    }
}

serializable_struct_with_getters! {
    SubmitProofReq {
        benchmark_id: String,
        solutions_data: Vec<SolutionData>,
    }
}

serializable_struct_with_getters! {
    SubmitProofResp {
        verified: Result<(), String>,
    }
}

serializable_struct_with_getters! {
    SubmitAlgorithmReq {
        name: String,
        challenge_id: String,
        tx_hash: String,
        code: String,
    }
}

serializable_struct_with_getters! {
    SubmitAlgorithmResp {
        algorithm_id: String,
    }
}
