use crate::serializable_struct_with_getters;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
pub use tig_utils::Frontier;
use tig_utils::PreciseNumber;

serializable_struct_with_getters! {
    ProtocolConfig {
        advances: AdvancesConfig,
        challenges: HashMap<String, ChallengeConfig>,
        codes: CodesConfig,
        deposits: DepositsConfig,
        erc20: ERC20Config,
        opow: OPoWConfig,
        rounds: RoundsConfig,
        rewards: RewardsConfig,
        topups: TopUpsConfig,
    }
}

serializable_struct_with_getters! {
    AdvancesConfig {
        bootstrap_address: String,
        min_percent_yes_votes: f64,
        vote_start_delay: u32,
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
        max_delegations: usize,
        token_locker_weight: u32,
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
        min_num_nonces: u64,
        submission_delay_multiplier: f64,
        max_samples: usize,
        lifespan_period: u32,
        per_nonce_fee: PreciseNumber,
        base_fee: PreciseNumber,
        runtime_config: RuntimeConfig,
    }
}
serializable_struct_with_getters! {
    TopUpsConfig {
        topup_address: String,
        min_topup_amount: PreciseNumber,
    }
}
serializable_struct_with_getters! {
    ChallengeConfig {
        benchmarks: BenchmarksConfig,
        difficulty: DifficultyConfig,
    }
}
serializable_struct_with_getters! {
    DifficultyConfig {
        allowed_sizes: Vec<usize>,
        total_qualifiers_threshold: u64,
    }
}
serializable_struct_with_getters! {
    OPoWConfig {
        imbalance_multiplier: f64,
        cutoff_phase_in_period: u32,
        cutoff_multiplier: f64,
        max_deposit_to_qualifier_ratio: f64,
        challenge_factors_weight: f64,
        max_coinbase_outputs: usize,
        coinbase_update_period: u32,
    }
}
serializable_struct_with_getters! {
    RoundsConfig {
        blocks_per_round: u32,
        seconds_between_blocks: u32,
    }
}
serializable_struct_with_getters! {
    CodesConfig {
        submission_fee: PreciseNumber,
        adoption_threshold: f64,
        merge_points_threshold: u32,
        push_delay_period: u32,
    }
}
serializable_struct_with_getters! {
    RewardsConfig {
        gamma: GammaConfig,
        distribution: DistributionConfig,
        schedule: Vec<EmissionsConfig>,
    }
}
serializable_struct_with_getters! {
    GammaConfig {
        a: f64,
        b: f64,
        c: f64,
    }
}
serializable_struct_with_getters! {
    DistributionConfig {
        opow: f64,
        codes: f64,
        advances: f64,
        challenge_owners: f64,
    }
}
serializable_struct_with_getters! {
    EmissionsConfig {
        block_reward: f64,
        round_start: u32,
    }
}
