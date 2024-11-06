mod benchmark;
mod algorithms;
mod challenge;

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
        cache::
        {
            Cache,
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
    pub fn new(
        cache:              Arc<Cache<T>>
    )                               -> Self
    {
        return Self 
        { 
            benchmark                   : BenchmarkContract::new(cache.clone()), 
            algorithms                  : AlgorithmsContract::new(cache.clone()),
            challenge                   : ChallengeContract::new(cache.clone()),
        };
    }
}