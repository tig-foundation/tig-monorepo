use
{
    crate::
    {
        Block,
        BlockFilter,
        Context,
        ProtocolResult,
        ProtocolError,
        PrecommitsFilter
    },
    tig_structs::
    {
        core::
        {
            Player,
            Challenge,
            BenchmarkSettings
        }
    },
    tig_utils::
    {
        PreciseNumber,
        PointOps,
        PointCompareFrontiers
    },
    std::
    {
        future::
        {
            Future
        },
        pin::
        {
            Pin
        }
    }
};

pub struct BenchmarksContract<T: Context>
{
    _phantom:                           std::marker::PhantomData<T>
}

impl<T: Context> BenchmarksContract<T>
{
    pub fn new()                                -> Self 
    {
        return Self { _phantom: std::marker::PhantomData };
    }

    pub fn verify_player_owns_benchmark(
        &self, 
        player:                         &Player, 
        settings:                       &BenchmarkSettings
    )                                           -> ProtocolResult<()>
    {
        if player.id != settings.player_id 
        {
            return Err(ProtocolError::InvalidSubmittingPlayer 
            {
                actual_player_id                    : player.id.clone(),
                expected_player_id                  : settings.player_id.clone(),
            });
        }

        return Ok(());
    }

    pub fn verify_num_nonces(
        &self, 
        num_nonces:                     u32
    )                                           -> ProtocolResult<()>
    {
        if num_nonces == 0 
        {
            return Err(ProtocolError::InvalidNumNonces
            { 
                num_nonces
            });
        }

        return Ok(());
    }

    pub async fn verify_sufficient_lifespan(
        &self,  
        ctx:                            &T,
        block:                          &Block
    )                                           -> ProtocolResult<()>
    {
        let latest_block                        = ctx
        .get_block(BlockFilter::Latest, false)
        .await
        .unwrap_or_else(|e| panic!("get_block error: {:?}", e))
        .expect("Expecting latest block to exist");

        let config                              = block.config();
        let submission_delay                    = latest_block.details.height - block.details.height + 1;
        if (submission_delay as f64 * (config.benchmark_submissions.submission_delay_multiplier + 1.0))
            as u32
            >= config.benchmark_submissions.lifespan_period
        {
            return Err(ProtocolError::InsufficientLifespan);
        }

        return Ok(());
    }

    pub fn verify_benchmark_difficulty(
        &self, 
        difficulty:                     &Vec<i32>,
        challenge:                      &Challenge,
        block:                          &Block
    )                                           -> ProtocolResult<()>
    {
        let config                                  = block.config();
        let difficulty_parameters                   = &config.difficulty.parameters[&challenge.id];

        if difficulty.len() != difficulty_parameters.len()
            || difficulty
                .iter()
                .zip(difficulty_parameters.iter())
                .any(|(d, p)| *d < p.min_value || *d > p.max_value)
        {
            return Err(ProtocolError::InvalidDifficulty 
            {
                difficulty                          : difficulty.clone(),
                difficulty_parameters               : difficulty_parameters.clone(),
            });
        }

        let challenge_data                          = challenge.block_data();
        let (lower_frontier, upper_frontier)        = if *challenge_data.scaling_factor() > 1f64
        {(
            challenge_data.base_frontier(),
            challenge_data.scaled_frontier(),
        )}
        else
        {(
            challenge_data.scaled_frontier(),
            challenge_data.base_frontier(),
        )};

        match difficulty.within(lower_frontier, upper_frontier) 
        {
            PointCompareFrontiers::Above => 
            {
                return Err(ProtocolError::DifficultyAboveHardestFrontier 
                {
                    difficulty                      : difficulty.clone(),
                });
            }

            PointCompareFrontiers::Below => 
            {
                return Err(ProtocolError::DifficultyBelowEasiestFrontier 
                {
                    difficulty: difficulty.clone(),
                });
            }
            
            PointCompareFrontiers::Within => {}
        }

        return Ok(());
    }

    pub async fn verify_benchmark_settings_are_unique(
        &self, 
        ctx:                            &T,
        settings:                       &BenchmarkSettings
    )                                           -> ProtocolResult<()>
    {
        if ctx
            .get_precommits(PrecommitsFilter::Settings(settings.clone()))
            .await
            .unwrap_or_else(|e| panic!("get_precommits error: {:?}", e))
            .first()
            .is_some()
        {
            return Err(ProtocolError::DuplicateBenchmarkSettings 
            {
                settings                        : settings.clone(),
            });
        }
    
        return Ok(());
    }
}