mod benchmark;
mod challenge;
mod algorithm;
mod player;
mod topup;

use
{
    crate::
    {
        ctx::Context,
        contracts::
        {
            benchmark::BenchmarkContract,
            algorithm::AlgorithmContract,
            challenge::ChallengeContract,
            player::PlayerContract,
            topup::TopUpContract,
        },
    },
    std::
    {
        sync::
        {
            Arc,
        },
    },
};

pub struct Contracts<T: Context>
{
    pub benchmark:          BenchmarkContract<T>,
    pub challenge:          ChallengeContract<T>,
    pub algorithm:          AlgorithmContract<T>,
    pub player:             PlayerContract<T>,
    pub topup:              TopUpContract<T>,
}

impl<T: Context> Contracts<T>
{
    pub fn new()                    -> Self
    {
        return Self 
        { 
            benchmark                   : BenchmarkContract::new(), 
            challenge                   : ChallengeContract::new(), 
            algorithm                   : AlgorithmContract::new(),
            player                      : PlayerContract::new(),
            topup                       : TopUpContract::new(),
        };
    }
}