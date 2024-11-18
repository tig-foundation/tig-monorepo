use super::contracts::*;

async fn add_block() {
    // clone of player_state; internally sets round_merged, etc
    let cache = ctx.build_block_cache().await;
    
    // filter active benchmarks
    benchmarks::update
    
    // deposit calcs
    players.update

    // calc influence
    opow.update

    // calc adoption
    algorithms.update
    
    // calc fees, solution signature
    challenges.update

    // calc rewards
    rewards.update
}
