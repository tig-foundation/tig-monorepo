pub mod block;
pub mod algorithm;
pub mod challenge;

use
{
    std::
    {
        sync::
        {
            RwLock,
        }
    },
    crate::
    {
        ctx::
        {
            Context,
        },
        cache::
        {
            block::BlockCache,
            algorithm::AlgorithmCache,
            challenge::ChallengeCache,
        },
    }
};

pub struct Cache<T: Context>
{
    pub blocks:                 RwLock<BlockCache<T>>,
    pub algorithms:             RwLock<AlgorithmCache<T>>,
    pub challenges:             RwLock<ChallengeCache<T>>,
}

impl<T: Context> Cache<T>
{
    pub fn new()                        -> Self
    {
        return Self 
        { 
            blocks                          : RwLock::new(BlockCache::new()), 
            algorithms                      : RwLock::new(AlgorithmCache::new()), 
            challenges                      : RwLock::new(ChallengeCache::new()) 
        };
    }
}