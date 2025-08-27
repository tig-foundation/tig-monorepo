use crate::context::*;
use logging_timer::time;

#[time]
pub(crate) async fn update(cache: &mut AddBlockCache) {
    let AddBlockCache {
        config,
        active_challenges_block_data,
        ..
    } = cache;

    for (challenge_id, challenge_data) in active_challenges_block_data.iter_mut() {
        let benchmarks_config = &config.challenges[challenge_id].benchmarks;
        challenge_data.base_fee = benchmarks_config.min_base_fee;
        challenge_data.per_nonce_fee = benchmarks_config.min_per_nonce_fee;
    }
}
