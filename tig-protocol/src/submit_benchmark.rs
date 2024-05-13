use crate::{context::*, error::*};
use std::collections::HashMap;
use tig_structs::core::*;
use tig_utils::*;

pub(crate) async fn execute<T: Context>(
    ctx: &mut T,
    player: &Player,
    settings: &BenchmarkSettings,
    solutions_meta_data: &Vec<SolutionMetaData>,
    solution_data: &SolutionData,
) -> ProtocolResult<(String, Result<(), String>)> {
    verify_player_owns_benchmark(player, settings)?;
    let block = get_block_by_id(ctx, &settings.block_id).await?;
    verify_sufficient_lifespan(ctx, &block).await?;
    let challenge = get_challenge_by_id(ctx, &settings.challenge_id, &block).await?;
    verify_algorithm(ctx, &settings.algorithm_id, &block).await?;
    verify_sufficient_solutions(&block, solutions_meta_data)?;
    verify_benchmark_settings_are_unique(ctx, settings).await?;
    verify_nonces_are_unique(solutions_meta_data)?;
    verify_solutions_signatures(solutions_meta_data, &challenge)?;
    verify_benchmark_difficulty(&settings.difficulty, &challenge)?;
    let benchmark_id = ctx
        .add_benchmark_to_mempool(
            &settings,
            &BenchmarkDetails {
                block_started: block.details.height,
                num_solutions: solutions_meta_data.len() as u32,
            },
            solutions_meta_data,
            solution_data,
        )
        .await
        .unwrap_or_else(|e| panic!("add_benchmark_to_mempool error: {:?}", e));
    let mut verified = Ok(());
    if let Err(e) =
        verify_solution_is_valid(ctx, settings, solutions_meta_data, solution_data).await
    {
        ctx.add_fraud_to_mempool(&benchmark_id, &e.to_string())
            .await
            .unwrap_or_else(|e| panic!("add_fraud_to_mempool error: {:?}", e));
        verified = Err(e.to_string());
    }
    Ok((benchmark_id, verified))
}

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

async fn verify_sufficient_lifespan<T: Context>(ctx: &mut T, block: &Block) -> ProtocolResult<()> {
    let latest_block = ctx
        .get_block(BlockFilter::Latest, false)
        .await
        .unwrap_or_else(|e| panic!("get_block error: {:?}", e))
        .expect("Expecting latest block to exist");
    let config = block.config();
    let submission_delay = latest_block.details.height - block.details.height + 1;
    if submission_delay * (config.benchmark_submissions.submission_delay_multiplier + 1)
        >= config.benchmark_submissions.lifespan_period
    {
        return Err(ProtocolError::InsufficientLifespan);
    }
    Ok(())
}

async fn get_challenge_by_id<T: Context>(
    ctx: &mut T,
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

async fn verify_algorithm<T: Context>(
    ctx: &mut T,
    algorithm_id: &String,
    block: &Block,
) -> ProtocolResult<()> {
    if !ctx
        .get_algorithms(AlgorithmsFilter::Id(algorithm_id.clone()), None, false)
        .await
        .unwrap_or_else(|e| panic!("get_algorithms error: {:?}", e))
        .first()
        .is_some()
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

async fn get_block_by_id<T: Context>(ctx: &mut T, block_id: &String) -> ProtocolResult<Block> {
    ctx.get_block(BlockFilter::Id(block_id.clone()), true)
        .await
        .unwrap_or_else(|e| panic!("get_block error: {:?}", e))
        .ok_or_else(|| ProtocolError::InvalidBlock {
            block_id: block_id.clone(),
        })
}

fn verify_sufficient_solutions(
    block: &Block,
    solutions_meta_data: &Vec<SolutionMetaData>,
) -> ProtocolResult<()> {
    let min_num_solutions = block.config().benchmark_submissions.min_num_solutions as usize;
    if solutions_meta_data.len() < min_num_solutions {
        return Err(ProtocolError::InsufficientSolutions {
            num_solutions: solutions_meta_data.len(),
            min_num_solutions,
        });
    }
    Ok(())
}

async fn verify_benchmark_settings_are_unique<T: Context>(
    ctx: &mut T,
    settings: &BenchmarkSettings,
) -> ProtocolResult<()> {
    if ctx
        .get_benchmarks(BenchmarksFilter::Settings(settings.clone()), false)
        .await
        .unwrap_or_else(|e| panic!("get_benchmarks error: {:?}", e))
        .first()
        .is_some()
    {
        return Err(ProtocolError::DuplicateBenchmarkSettings {
            settings: settings.clone(),
        });
    }

    Ok(())
}

fn verify_nonces_are_unique(solutions_meta_data: &Vec<SolutionMetaData>) -> ProtocolResult<()> {
    let nonces: HashMap<u32, u32> =
        solutions_meta_data
            .iter()
            .fold(HashMap::new(), |mut acc, s| {
                *acc.entry(s.nonce).or_insert(0) += 1;
                acc
            });

    if let Some((&nonce, _)) = nonces.iter().find(|(_, &count)| count > 1) {
        return Err(ProtocolError::DuplicateNonce { nonce });
    }

    Ok(())
}

fn verify_solutions_signatures(
    solutions_meta_data: &Vec<SolutionMetaData>,
    challenge: &Challenge,
) -> ProtocolResult<()> {
    let solution_signature_threshold = *challenge.block_data().solution_signature_threshold();
    if let Some(s) = solutions_meta_data
        .iter()
        .find(|&s| s.solution_signature > solution_signature_threshold)
    {
        return Err(ProtocolError::InvalidSolutionSignature {
            nonce: s.nonce,
            solution_signature: s.solution_signature,
            threshold: solution_signature_threshold,
        });
    }

    Ok(())
}

fn verify_benchmark_difficulty(difficulty: &Vec<i32>, challenge: &Challenge) -> ProtocolResult<()> {
    let challenge_data = challenge.block_data();

    let difficulty_parameters = &challenge.details.difficulty_parameters;
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

async fn verify_solution_is_valid<T: Context>(
    ctx: &mut T,
    settings: &BenchmarkSettings,
    solutions_meta_data: &Vec<SolutionMetaData>,
    solution_data: &SolutionData,
) -> ProtocolResult<()> {
    let solutions_map: HashMap<u32, u32> = solutions_meta_data
        .iter()
        .map(|d| (d.nonce, d.solution_signature))
        .collect();

    if let Some(&expected_signature) = solutions_map.get(&solution_data.nonce) {
        let signature = solution_data.calc_solution_signature();

        if expected_signature != signature {
            return Err(ProtocolError::InvalidSignatureFromSolutionData {
                nonce: solution_data.nonce,
                expected_signature,
                actual_signature: signature,
            });
        }
    } else {
        return Err(ProtocolError::InvalidBenchmarkNonce {
            nonce: solution_data.nonce,
        });
    }

    if ctx
        .verify_solution(settings, solution_data.nonce, &solution_data.solution)
        .await
        .unwrap_or_else(|e| panic!("verify_solution error: {:?}", e))
        .is_err()
    {
        return Err(ProtocolError::InvalidSolution {
            nonce: solution_data.nonce,
        });
    }

    Ok(())
}
