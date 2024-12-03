use crate::{core::AlgorithmType, serializable_struct_with_getters};
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
        topups: TopUpsConfig,
    }
}

serializable_struct_with_getters! {
    BreakthroughsConfig {
        bootstrap_address: String,
        min_percent_yes_votes: f64,
        vote_period: u32,
        min_lock_period_to_vote: u32,
        submission_fee: PreciseNumber,
        adoption_threshold: f64,
        merge_points_threshold: u32,
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
        min_lock_period: u32,
        lock_period_cap: u32,
        max_reward_share: f64,
        default_reward_share: f64,
        reward_share_update_period: u32,
        delegatees_update_period: u32,
        delegatee_min_deposit: PreciseNumber,
        max_delegations: usize,
    }
}
serializable_struct_with_getters! {
    RuntimeConfig {
        max_memory: u64,
        max_fuel: u64,
    }
}
serializable_struct_with_getters! {
    BenchmarksConfig {
        min_num_solutions: u32,
        submission_delay_multiplier: f64,
        max_samples: usize,
        lifespan_period: u32,
        min_per_nonce_fee: PreciseNumber,
        min_base_fee: PreciseNumber,
        runtime_configs: HashMap<AlgorithmType, RuntimeConfig>,
    }
}
serializable_struct_with_getters! {
    TopUpsConfig {
        topup_address: String,
        min_topup_amount: PreciseNumber,
    }
}
serializable_struct_with_getters! {
    ChallengesConfig {
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
        cutoff_phase_in_period: u32,
        cutoff_multiplier: f64,
        total_qualifiers_threshold: u32,
        max_deposit_to_qualifier_ratio: f64,
        deposit_multiplier: f64,
        deposit_to_cutoff_ratio: f64,
    }
}
serializable_struct_with_getters! {
    RoundsConfig {
        blocks_per_round: u32,
        seconds_between_blocks: u32,
    }
}
serializable_struct_with_getters! {
    AlgorithmsConfig {
        submission_fee: PreciseNumber,
        adoption_threshold: f64,
        merge_points_threshold: u32,
        push_delay_period: u32,
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
        opow: f64,
        algorithms: f64,
        breakthroughs: f64,
    }
}
serializable_struct_with_getters! {
    EmissionsConfig {
        block_reward: f64,
        round_start: u32,
    }
}
