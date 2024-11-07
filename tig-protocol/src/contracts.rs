mod benchmark;
mod challenge;
mod topup;

use
{
    crate::
    {
        ctx::Context,
        contracts::
        {
            benchmark::BenchmarkContract,
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
    pub benchmark:          BenchmarkContract<T>,
    pub challenge:          ChallengeContract<T>,
}

impl<T: Context> Contracts<T>
{
    pub fn new()                    -> Self
    {
        return Self 
        { 
            benchmark                   : BenchmarkContract::new(), 
            challenge                   : ChallengeContract::new(),
        };
    }
}