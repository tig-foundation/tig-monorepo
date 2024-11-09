mod algorithms;
mod benchmarks;
mod challenges;
mod players;
mod opow;

use {
    crate::{
        contracts::{
            algorithms::AlgorithmContract,
            benchmarks::BenchmarkContract,
            challenges::ChallengeContract,
            players::PlayerContract,
            opow::OPoWContract,
        },
        ctx::Context,
    },
    std::sync::Arc,
};

pub struct Contracts<T: Context>
{
    pub benchmark:  BenchmarkContract<T>,
    pub challenge:  ChallengeContract,
    pub algorithm:  AlgorithmContract,
    pub player:     PlayerContract,
    pub opow:       OPoWContract<T>,
}

impl<T: Context> Contracts<T>
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
        };  
    }
}
