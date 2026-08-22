pub mod context;
mod contracts;
use context::*;

pub use contracts::{
    algorithms::{submit_advance, submit_binary, submit_code},
    benchmarks::{submit_benchmark, submit_precommit, submit_proof},
    players::{set_coinbase, set_delegatees, set_reward_share, set_vote, submit_report},
};

pub async fn add_block<T: Context>(ctx: &T) {
    let mut cache = ctx.build_block_cache().await;
    println!("add_block: start");
    contracts::players::update(&mut cache).await;
    println!("add_block: players updated");
    contracts::opow::update(&mut cache).await;
    println!("add_block: opow updated");
    contracts::algorithms::update(&mut cache).await;
    println!("add_block: algorithms updated");
    contracts::rewards::update(&mut cache).await;
    println!("add_block: rewards updated");
    ctx.commit_block_cache(cache).await;
    println!("add_block: cache committed");
}
