use crate::serializable_struct_with_getters;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
pub use tig_utils::Point;
use tig_utils::PreciseNumber;

serializable_struct_with_getters! {
    ProtocolConfig {
        algorithms: AlgorithmsConfig,
        benchmarks: BenchmarksConfig,
        breakthroughs: BreakthroughsConfig,
        challenges: ChallengesConfig,
        deposits: DepositsConfig,
        erc20: ERC20Config,
        opow: OPoWConfig,
        rounds: RoundsConfig,
        rewards: RewardsConfig,
        runtime: RuntimeConfig,
    }
}

serializable_struct_with_getters! {
    BreakthroughsConfig {
        academic_fund_address: String,
        min_percent_yes_votes: f64,
        vote_period_rounds: u32,
        min_lock_period_to_vote: u32,
        submission_fee: PreciseNumber,
        adoption_threshold: f64,
        merge_points_threshold: u32,
        push_delay: u32,
    }
}
serializable_struct_with_getters! {
    ERC20Config {
        rpc_url: String,
        chain_id: String,
        token_address: String,
    }
}
serializable_struct_with_getters! {
    DepositsConfig {
        lock_address: String,
        min_lock_amount: PreciseNumber,
        min_lock_period_secs: u64,
        max_lock_period_rounds: u32,
        lock_period_multiplier: f64,
        max_reward_share: f64,
        deposit_to_qualifier_ratio: f64,
        period_between_redelegate: u32,
    }
}
serializable_struct_with_getters! {
    BenchmarksConfig {
        min_num_solutions: u32,
        submission_delay_multiplier: f64,
        max_samples: usize,
        max_active_period_blocks: u32,
        min_per_nonce_fee: PreciseNumber,
        min_base_fee: PreciseNumber,
        max_fee_percentage_delta: f64,
        target_num_precommits: u32,
    }
}
serializable_struct_with_getters! {
    TopUpsConfig {
        topup_address: String,
        min_topup_amount: PreciseNumber,
    }
}
serializable_struct_with_getters! {
    RuntimeConfig {
        max_memory: u64,
        max_fuel: u64,
    }
}
serializable_struct_with_getters! {
    ChallengesConfig {
        max_percent_delta: Option<f64>,
        threshold_decay: Option<f64>,
        equilibrium_rate_multiplier: f64,
        percent_error_multiplier: Option<f64>,
        max_scaling_factor: f64,
        difficulty_parameters: HashMap<String, Vec<DifficultyParameter>>,
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
    OPoWConfig {
        imbalance_multiplier: f64,
        enable_proof_of_deposit: Option<bool>,
        cutoff_phase_in_period: Option<u32>,
        cutoff_multiplier: f64,
        total_qualifiers_threshold: u32,
        min_cutoff: Option<u32>,
        deposit_to_cutoff_cap_ratio: f64,
    }
}
serializable_struct_with_getters! {
    RoundsConfig {
        blocks_per_round: u32,
    }
}
serializable_struct_with_getters! {
    AlgorithmsConfig {
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
