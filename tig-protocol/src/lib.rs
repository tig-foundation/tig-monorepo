pub mod context;
mod contracts;
use context::*;

pub use contracts::{
    algorithms::{submit_algorithm, submit_binary, submit_breakthrough},
    benchmarks::{submit_benchmark, submit_precommit, submit_proof},
    players::{set_coinbase, set_delegatees, set_reward_share, set_vote},
};

pub async fn add_block<T: Context>(ctx: &T) {
    let mut cache = ctx.build_block_cache().await;
    contracts::players::update(&mut cache).await;
    contracts::opow::update(&mut cache).await;
    contracts::algorithms::update(&mut cache).await;
    contracts::challenges::update(&mut cache).await;
    contracts::rewards::update(&mut cache).await;
    ctx.commit_block_cache(cache).await;
}
