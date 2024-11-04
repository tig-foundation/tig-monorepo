use
{
    crate::
    {
        Block,
        BlockFilter,
        Context,
        ProtocolResult,
        ProtocolError
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

pub struct BenchmarkContext;
impl Context for BenchmarkContext {}

pub trait IBenchmarksContract
{
    fn verify_player_owns_benchmark(
        &self, 
        player:                         &Player, 
        settings:                       &BenchmarkSettings
    )                                           -> ProtocolResult<()>;

    fn verify_num_nonces(
        &self, 
        num_nonces:                     u32
    )                                           -> ProtocolResult<()>;

    fn verify_sufficient_lifespan(
        &self,  
        ctx:                            &BenchmarkContext,
        block:                          &Block
    )                                           -> Pin<Box<dyn Future<Output = ProtocolResult<()>>>>;

    fn get_challenge_by_id(
        &self, 
        challenge_id:                   &String,
        block:                          &Block
    )                                           -> ();

    fn verify_benchmark_difficulty(
        &self, 
        difficulty:                     &Vec<i32>,
        challenge:                      &Challenge,
        block:                          &Block
    )                                           -> ProtocolResult<()>;

    fn verify_benchmark_settings_are_unique(
        &self, 
        ctx:                            &BenchmarkContext,
        settings:                       &BenchmarkSettings
    )                                           -> Pin<Box<dyn Future<Output = ProtocolResult<()>>>>;

    fn get_fee_paid(
        &self, 
        player:                         &Player,
        num_nonces:                     u32,
        challenge:                      &Challenge
    )                                           -> ProtocolResult<PreciseNumber>;

    fn submit_precommit(
        &self
    )                                           -> ();

    fn submit_benchmark(&self)  -> ();
    fn submit_proof(&self)      -> ();
}

impl dyn IBenchmarksContract 
{
    fn verify_player_owns_benchmark(
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
    
    fn verify_num_nonces(
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

    fn verify_sufficient_lifespan(
        &self,  
        ctx:                            &BenchmarkContext,
        block:                          &Block
    )                                           -> Pin<Box<dyn Future<Output = ProtocolResult<()>>>>
    {
        Box::pin(async move 
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
        })
    }

    fn verify_benchmark_difficulty(
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

    fn verify_benchmark_settings_are_unique(
        &self, 
        ctx:                            &BenchmarkContext,
        settings:                       &BenchmarkSettings
    )                                           -> Pin<Box<dyn Future<Output = ProtocolResult<()>>>>
    {
        Box::pin(async move 
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
        })
    }

    fn get_fee_paid(
        &self, 
        player:                         &Player,
        num_nonces:                     u32,
        challenge:                      &Challenge
    )                                           -> ProtocolResult<PreciseNumber>
    {
        let num_nonces                              = PreciseNumber::from(num_nonces);
        let fee_paid                                = challenge.block_data().base_fee().clone()
                                                        + challenge.block_data().per_nonce_fee().clone() * num_nonces;
        if !player
            .state
            .as_ref()
            .is_some_and(|s| *s.available_fee_balance.as_ref().unwrap() >= fee_paid)
        {
            return Err(ProtocolError::InsufficientFeeBalance 
            {
                fee_paid,
                available_fee_balance: player
                    .state
                    .as_ref()
                    .map(|s| s.available_fee_balance().clone())
                    .unwrap_or(PreciseNumber::from(0)),
            });
        }

        return Ok(fee_paid);
    }

    pub fn submit_precommit(
        &self
    )                                           -> ()
    {

    }

    pub fn submit_benchmark(
        &self
    )                                           -> ()
    {

    }

    pub fn submit_proof(
        &self
    )                                           -> ()
    {

    }
}