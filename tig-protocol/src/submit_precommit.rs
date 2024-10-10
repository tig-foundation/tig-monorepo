use crate::{context::*, error::*};
use logging_timer::time;
use tig_structs::core::*;
use tig_utils::*;

#[time]
pub(crate) async fn execute<T: Context>(
    ctx: &T,
    player: &Player,
    settings: BenchmarkSettings,
    num_nonces: u32,
) -> ProtocolResult<String> {
    verify_player_owns_benchmark(player, &settings)?;
    verify_num_nonces(num_nonces)?;
    let block = get_block_by_id(ctx, &settings.block_id).await?;
    verify_sufficient_lifespan(ctx, &block).await?;
    let challenge = get_challenge_by_id(ctx, &settings.challenge_id, &block).await?;
    verify_algorithm(ctx, &settings.algorithm_id, &block).await?;
    verify_benchmark_settings_are_unique(ctx, &settings).await?;
    verify_benchmark_difficulty(&settings.difficulty, &challenge, &block)?;
    let fee_paid = get_fee_paid(&player, num_nonces, &challenge)?;
    let benchmark_id = ctx
        .add_precommit_to_mempool(
            settings,
            PrecommitDetails {
                block_started: block.details.height,
                num_nonces: Some(num_nonces),
                fee_paid: Some(fee_paid),
            },
        )
        .await
        .unwrap_or_else(|e| panic!("add_precommit_to_mempool error: {:?}", e));
    Ok(benchmark_id)
}

#[time]
fn verify_player_owns_benchmark(
    player: &Player,
    settings: &BenchmarkSettings,
) -> ProtocolResult<()> {
    if player.id != settings.player_id {
        return Err(ProtocolError::InvalidSubmittingPlayer {
            actual_player_id: player.id.clone(),
            expected_player_id: settings.player_id.clone(),
        });
    }
    Ok(())
}

#[time]
fn verify_num_nonces(num_nonces: u32) -> ProtocolResult<()> {
    if num_nonces == 0 {
        return Err(ProtocolError::InvalidNumNonces { num_nonces });
    }
    Ok(())
}

#[time]
async fn verify_sufficient_lifespan<T: Context>(ctx: &T, block: &Block) -> ProtocolResult<()> {
    let latest_block = ctx
        .get_block(BlockFilter::Latest, false)
        .await
        .unwrap_or_else(|e| panic!("get_block error: {:?}", e))
        .expect("Expecting latest block to exist");
    let config = block.config();
    let submission_delay = latest_block.details.height - block.details.height + 1;
    if (submission_delay as f64 * (config.benchmark_submissions.submission_delay_multiplier + 1.0))
        as u32
        >= config.benchmark_submissions.lifespan_period
    {
        return Err(ProtocolError::InsufficientLifespan);
    }
    Ok(())
}

#[time]
async fn get_challenge_by_id<T: Context>(
    ctx: &T,
    challenge_id: &String,
    block: &Block,
) -> ProtocolResult<Challenge> {
    if !block.data().active_challenge_ids.contains(challenge_id) {
        return Err(ProtocolError::InvalidChallenge {
            challenge_id: challenge_id.clone(),
        });
    }
    let challenge = ctx
        .get_challenges(
            ChallengesFilter::Id(challenge_id.clone()),
            Some(BlockFilter::Id(block.id.clone())),
        )
        .await
        .unwrap_or_else(|e| panic!("get_challenges error: {:?}", e))
        .first()
        .map(|x| x.to_owned())
        .ok_or_else(|| ProtocolError::InvalidChallenge {
            challenge_id: challenge_id.clone(),
        })?;
    Ok(challenge)
}

#[time]
async fn verify_algorithm<T: Context>(
    ctx: &T,
    algorithm_id: &String,
    block: &Block,
) -> ProtocolResult<()> {
    if !ctx
        .get_algorithms(AlgorithmsFilter::Id(algorithm_id.clone()), None, false)
        .await
        .unwrap_or_else(|e| panic!("get_algorithms error: {:?}", e))
        .pop()
        .is_some_and(|a| a.state.is_some_and(|s| !s.banned))
    {
        return Err(ProtocolError::InvalidAlgorithm {
            algorithm_id: algorithm_id.clone(),
        });
    }
    if !block.data().active_algorithm_ids.contains(algorithm_id) {
        return Err(ProtocolError::InvalidAlgorithm {
            algorithm_id: algorithm_id.clone(),
        });
    }
    Ok(())
}

#[time]
async fn get_block_by_id<T: Context>(ctx: &T, block_id: &String) -> ProtocolResult<Block> {
    ctx.get_block(BlockFilter::Id(block_id.clone()), true)
        .await
        .unwrap_or_else(|e| panic!("get_block error: {:?}", e))
        .ok_or_else(|| ProtocolError::InvalidBlock {
            block_id: block_id.clone(),
        })
}

#[time]
async fn verify_benchmark_settings_are_unique<T: Context>(
    ctx: &T,
    settings: &BenchmarkSettings,
) -> ProtocolResult<()> {
    if ctx
        .get_precommits(PrecommitsFilter::Settings(settings.clone()))
        .await
        .unwrap_or_else(|e| panic!("get_precommits error: {:?}", e))
        .first()
        .is_some()
    {
        return Err(ProtocolError::DuplicateBenchmarkSettings {
            settings: settings.clone(),
        });
    }

    Ok(())
}

#[time]
fn verify_benchmark_difficulty(
    difficulty: &Vec<i32>,
    challenge: &Challenge,
    block: &Block,
) -> ProtocolResult<()> {
    let config = block.config();
    let difficulty_parameters = &config.difficulty.parameters[&challenge.id];

    if difficulty.len() != difficulty_parameters.len()
        || difficulty
            .iter()
            .zip(difficulty_parameters.iter())
            .any(|(d, p)| *d < p.min_value || *d > p.max_value)
    {
        return Err(ProtocolError::InvalidDifficulty {
            difficulty: difficulty.clone(),
            difficulty_parameters: difficulty_parameters.clone(),
        });
    }

    let challenge_data = challenge.block_data();
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
    match difficulty.within(lower_frontier, upper_frontier) {
        PointCompareFrontiers::Above => {
            return Err(ProtocolError::DifficultyAboveHardestFrontier {
                difficulty: difficulty.clone(),
            });
        }
        PointCompareFrontiers::Below => {
            return Err(ProtocolError::DifficultyBelowEasiestFrontier {
                difficulty: difficulty.clone(),
            });
        }
        PointCompareFrontiers::Within => {}
    }

    Ok(())
}

#[time]
fn get_fee_paid(
    player: &Player,
    num_nonces: u32,
    challenge: &Challenge,
) -> ProtocolResult<PreciseNumber> {
    let num_nonces = PreciseNumber::from(num_nonces);
    let fee_paid = challenge.block_data().base_fee().clone()
        + challenge.block_data().per_nonce_fee().clone() * num_nonces;
    if !player
        .state
        .as_ref()
        .is_some_and(|s| *s.available_fee_balance.as_ref().unwrap() >= fee_paid)
    {
        return Err(ProtocolError::InsufficientFeeBalance {
            fee_paid,
            available_fee_balance: player
                .state
                .as_ref()
                .map(|s| s.available_fee_balance().clone())
                .unwrap_or(PreciseNumber::from(0)),
        });
    }
    Ok(fee_paid)
}
