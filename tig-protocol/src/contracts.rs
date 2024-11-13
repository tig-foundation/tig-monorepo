mod algorithms;
mod benchmarks;
mod challenges;
mod players;
mod opow;
mod rewards;

use {
    crate::{
        contracts::{
            algorithms::AlgorithmContract,
            benchmarks::BenchmarkContract,
            challenges::ChallengeContract,
            players::PlayerContract,
            opow::OPoWContract,
            rewards::RewardsContract,
        },
        ctx::Context,
    },
    std::sync::Arc,
};

pub struct Contracts<T: Context + Send + Sync>
{
    pub benchmark:  BenchmarkContract<T>,
    pub challenge:  ChallengeContract<T>,
    pub algorithm:  AlgorithmContract<T>,
    pub player:     PlayerContract<T>,
    pub opow:       OPoWContract<T>,
    pub rewards:    RewardsContract<T>,
}

impl<T: Context + Send + Sync> Contracts<T>
{
    pub fn new() -> Self 
    {
        return Self 
        {
            benchmark:  BenchmarkContract::new(),
            challenge:  ChallengeContract::new(),
            algorithm:  AlgorithmContract::new(),
            player:     PlayerContract::new(),
            opow:       OPoWContract::new(),
            rewards:    RewardsContract::new(),
        };  
    }
}
