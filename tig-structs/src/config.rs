use crate::serializable_struct_with_getters;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
pub use tig_utils::Point;
use tig_utils::PreciseNumber;

serializable_struct_with_getters! {
    ProtocolConfig {
        erc20: ERC20Config,
        benchmark_submissions: BenchmarkSubmissionsConfig,
        wasm_vm: WasmVMConfig,
        solution_signature: SolutionSignatureConfig,
        qualifiers: QualifiersConfig,
        difficulty: DifficultyConfig,
        optimisable_proof_of_work: OptimisableProofOfWorkConfig,
        rounds: RoundsConfig,
        algorithm_submissions: AlgorithmSubmissionsConfig,
        rewards: RewardsConfig,
    }
}
serializable_struct_with_getters! {
    ERC20Config {
        rpc_url: String,
        chain_id: String,
        token_address: String,
        burn_address: String,
    }
}
serializable_struct_with_getters! {
    BenchmarkSubmissionsConfig {
        min_num_solutions: u32,
        submission_delay_multiplier: u32,
        max_samples: usize,
        lifespan_period: u32,
    }
}
serializable_struct_with_getters! {
    WasmVMConfig {
        max_memory: u64,
        max_fuel: u64,
    }
}
serializable_struct_with_getters! {
    SolutionSignatureConfig {
        max_percent_delta: f64,
        equilibrium_rate_multiplier: f64,
        percent_error_multiplier: f64,
    }
}
serializable_struct_with_getters! {
    QualifiersConfig {
        cutoff_multiplier: f64,
        total_qualifiers_threshold: u32,
    }
}
serializable_struct_with_getters! {
    DifficultyParameter {
        name: String,
        min_value: i32,
        max_value: i32,
    }
}
pub trait MinMaxDifficulty {
    fn min_difficulty(&self) -> Point;
    fn max_difficulty(&self) -> Point;
}
impl MinMaxDifficulty for Vec<DifficultyParameter> {
    fn min_difficulty(&self) -> Point {
        self.iter().map(|p| p.min_value).collect()
    }
    fn max_difficulty(&self) -> Point {
        self.iter().map(|p| p.max_value).collect()
    }
}
serializable_struct_with_getters! {
    DifficultyConfig {
        max_scaling_factor: f64,
        min_frontiers_gap: Option<f64>,
        parameters: HashMap<String, Vec<DifficultyParameter>>,
    }
}
serializable_struct_with_getters! {
    OptimisableProofOfWorkConfig {
        imbalance_multiplier: f64,
        enable_proof_of_deposit: Option<bool>,
        rolling_deposit_decay: Option<f64>,
    }
}
serializable_struct_with_getters! {
    RoundsConfig {
        blocks_per_round: u32,
    }
}
serializable_struct_with_getters! {
    AlgorithmSubmissionsConfig {
        submission_fee: PreciseNumber,
        adoption_threshold: f64,
        merge_points_threshold: u32,
        push_delay: u32,
    }
}
serializable_struct_with_getters! {
    RewardsConfig {
        distribution: DistributionConfig,
        schedule: Vec<EmissionsConfig>,
    }
}
serializable_struct_with_getters! {
    DistributionConfig {
        benchmarkers: f64,
        optimisations: f64,
        breakthroughs: f64,
    }
}
serializable_struct_with_getters! {
    EmissionsConfig {
        block_reward: f64,
        round_start: u32,
    }
}
