use
{
    crate::
    {
        ctx::
        {
            Context,
            ContextResult,
        },
        err::
        {
            ContractResult,
            ProtocolError,
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
            RwLock,
            Arc,
        },
        marker::
        {
            PhantomData,
        },
    },
    tig_structs::
    {
        core::
        {
            *
        }
    },
    tig_utils::
    {
        *
    }
};

pub struct BenchmarkContract<T: Context>
{
    phantom:                            PhantomData<T>,
    cache:                              Arc<Cache<T>>,
}   

impl<T: Context> BenchmarkContract<T>
{
    pub fn new(
        cache:                  Arc<Cache<T>>
    )                                   -> Self
    {
        return Self 
        {
            phantom                         : PhantomData,
            cache                           : cache,
        };
    }

    pub async fn verify_benchmark_settings_are_unique<'a>(
        &self,
        ctx:                    &T,
        settings:               &'a BenchmarkSettings
    )                                   -> ContractResult<'a, ()>
    {
        if ctx
            .get_precommits_by_settings(settings)
            .await
            .unwrap_or_else(|e| panic!("get_precommits error: {:?}", e))
            .first()
            .is_some()
        {
            return Err(ProtocolError::DuplicateBenchmarkSettings 
            {
                settings                    : settings,
            });
        }

        return Ok(());
    }

    pub async fn verify_benchmark_difficulty<'b>(
        difficulty:             &'b Vec<i32>,
        challenge:              &Challenge,
        block:                  &'b Block,
    )                                   -> ContractResult<'b, ()>
    {
        let config                          = block.config();
        let difficulty_parameters           = &config.difficulty.parameters[&challenge.id];

        if difficulty.len() != difficulty_parameters.len()
            || difficulty
                .iter()
                .zip(difficulty_parameters.iter())
                .any(|(d, p)| *d < p.min_value || *d > p.max_value)
        {
            return Err(ProtocolError::InvalidDifficulty 
            {
                difficulty                  : difficulty,
                difficulty_parameters       : difficulty_parameters,
            });
        }

        let challenge_data                  = challenge.block_data();
        let (lower_frontier, upper_frontier) = if *challenge_data.scaling_factor() > 1f64 {
            (
                challenge_data.base_frontier(),
                challenge_data.scaled_frontier(),
            )
        } else {
            (
                challenge_data.scaled_frontier(),
                challenge_data.base_frontier(),
            )
        };
        
        match difficulty.within(lower_frontier, upper_frontier) 
        {
            PointCompareFrontiers::Above => 
            {
                return Err(ProtocolError::DifficultyAboveHardestFrontier 
                {
                    difficulty              : difficulty,
                });
            }
            PointCompareFrontiers::Below => 
            {
                return Err(ProtocolError::DifficultyBelowEasiestFrontier 
                {
                    difficulty              : difficulty,
                });
            }
            PointCompareFrontiers::Within => {}
        }

        return Ok(());
    }
}