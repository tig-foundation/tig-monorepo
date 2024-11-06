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
    },
    logging_timer::time
};

pub struct BenchmarkContract<T: Context>
{
    phantom:                            PhantomData<T>,
}   

impl<T: Context> BenchmarkContract<T>
{
    pub fn new()                        -> Self
    {
        return Self 
        {
            phantom                         : PhantomData,
        };
    }

    #[time]
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

    #[time]
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

    #[time]
    async fn get_benchmark_by_id<'a>(
        ctx:                    &T,
        benchmark_id:           &'a String,
    )                                   -> ContractResult<'a, Benchmark> 
    {
        return ctx.get_benchmarks_by_id(benchmark_id, true)
            .await
            .unwrap_or_else(|e| panic!("add_benchmark_to_mempool error: {:?}", e))
            .pop()
            .filter(|b| b.state.is_some())
            .ok_or_else(|| ProtocolError::InvalidBenchmark 
            {
                benchmark_id                : benchmark_id,
            })
    }

    #[time]
    async fn verify_benchmark_ownership<'a>(
        player:                 &'a Player,
        settings:               &'a BenchmarkSettings,
    )                                   -> ContractResult<'a, ()>
    {
        if player.id != settings.player_id
        {
            return Err(ProtocolError::InvalidSubmittingPlayer 
            {
                actual_player_id            : &player.id,
                expected_player_id          : &settings.player_id,
            });
        }

        return Ok(());
    }

    #[time]
    async fn verify_benchmark_not_already_submitted<'a>(
        ctx:                    &T,
        benchmark_id:           &'a String,
    )                                   -> ContractResult<'a, ()>
    {
        if ctx
            .get_benchmarks_by_id(benchmark_id, false)
            .await
            .unwrap_or_else(|e| panic!("get_benchmarks error: {:?}", e))
            .first()
            .is_some()
        {
            return Err(ProtocolError::DuplicateBenchmark 
            {
                benchmark_id                : benchmark_id,
            });
        }

        return Ok(());
    }

    #[time]
    async fn verify_player_owns_benchmark<'a>(
        player:                 &'a Player,
        settings:               &'a BenchmarkSettings,
    )                                   -> ContractResult<'a, ()>
    {
        if player.id != settings.player_id 
        {
            return Err(ProtocolError::InvalidSubmittingPlayer 
            {
                actual_player_id            : &player.id,
                expected_player_id          : &settings.player_id,
            });
        }

        return Ok(());
    }
}
