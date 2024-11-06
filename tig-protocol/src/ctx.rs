pub use anyhow::{Error as ContextError, Result as ContextResult};
use
{
    crate::
    {
        cache::
        {
            block::
            {
                BlockCache,
            },
        },
    },
    tig_structs::
    {
        *,
        core::
        {
            *
        }
    }
};


pub trait Context
{
    async fn get_block_by_height(
        &self,
        block_height:                   i64,
        include_data:                   bool,
    )                                           -> ContextResult<Option<Block>>;

    async fn get_block_by_id(
        &self,
        block_id:                       &String,
        include_data:                   bool,
    )                                           -> ContextResult<Option<Block>>;

    async fn get_algorithm_by_id(
        &self,
        algorithm_id:                   &String,
        include_data:                   bool,
    )                                           -> ContextResult<Option<Algorithm>>;

    async fn get_challenge_by_id_and_height(
        &self,
        challenge_id:                   &String,
        block_height:                   u64,
        include_data:                   bool,
    )                                           -> ContextResult<Option<Challenge>>;

    async fn get_precommits_by_settings(
        &self,
        settings:                       &BenchmarkSettings,
    )                                           -> ContextResult<Vec<Precommit>>;
}
