mod algorithm;
mod benchmark;
mod challenge;
mod player;
mod topup;

use {
    crate::{
        contracts::{
            algorithm::AlgorithmContract, benchmark::BenchmarkContract,
            challenge::ChallengeContract, player::PlayerContract, topup::TopUpContract,
        },
        ctx::Context,
    },
    std::sync::Arc,
};

pub struct Contracts<T: Context> {
    pub benchmark: BenchmarkContract<T>,
    pub challenge: ChallengeContract<T>,
    pub algorithm: AlgorithmContract<T>,
    pub player: PlayerContract<T>,
    pub topup: TopUpContract<T>,
}

impl<T: Context> Contracts<T> {
    pub fn new() -> Self {
        return Self {
            benchmark: BenchmarkContract::new(),
            challenge: ChallengeContract::new(),
            algorithm: AlgorithmContract::new(),
            player: PlayerContract::new(),
            topup: TopUpContract::new(),
        };
    }
}
