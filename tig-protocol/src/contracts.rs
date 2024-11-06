mod benchmark;
mod algorithms;
mod challenge;
mod topup;
mod precommits;
mod proof;

use
{
    crate::
    {
        ctx::Context,
        contracts::
        {
            benchmark::BenchmarkContract,
            algorithms::AlgorithmsContract,
            challenge::ChallengeContract,
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
    benchmark:              BenchmarkContract<T>,
    algorithms:             AlgorithmsContract<T>,
    challenge:              ChallengeContract<T>,
}

impl<T: Context> Contracts<T>
{
    pub fn new()                    -> Self
    {
        return Self 
        { 
            benchmark                   : BenchmarkContract::new(), 
            algorithms                  : AlgorithmsContract::new(),
            challenge                   : ChallengeContract::new(),
        };
    }
}