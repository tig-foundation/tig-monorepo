use crate::context::*;
use logging_timer::time;

#[time]
pub(crate) async fn update(cache: &mut AddBlockCache) {
    let AddBlockCache {
        config,
        active_challenges_block_data,
        ..
    } = cache;

    for challenge_data in active_challenges_block_data.values_mut() {
        challenge_data.base_fee = config.benchmarks.min_base_fee;
        challenge_data.per_nonce_fee = config.benchmarks.min_per_nonce_fee;
    }
}
