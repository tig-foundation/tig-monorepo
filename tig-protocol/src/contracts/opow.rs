use crate::context::*;
use logging_timer::time;
use std::collections::{HashMap, HashSet};
use tig_structs::{config::*, core::*};
use tig_utils::*;

#[time]
pub(crate) async fn update(cache: &mut AddBlockCache) {
    let AddBlockCache {
        config,
        block_details,
        block_data,
        active_challenges_block_data,
        active_challenges_prev_block_data,
        active_algorithms_state,
        active_algorithms_details,
        active_algorithms_block_data,
        active_solutions,
        active_players_state,
        active_players_block_data,
        active_opow_block_data,
        active_opow_prev_block_data,
        confirmed_num_solutions,
        ..
    } = cache;

    let active_algorithm_ids = &block_data.active_ids[&ActiveType::Algorithm];
    let active_challenge_ids = &block_data.active_ids[&ActiveType::Challenge];
    let active_player_ids = &block_data.active_ids[&ActiveType::Player];
    let active_opow_ids = &block_data.active_ids[&ActiveType::OPoW];

    // update cutoffs
    let phase_in_start = (block_details.round - 1) * config.rounds.blocks_per_round;
    let phase_in_end = phase_in_start + config.opow.challenge_phase_in_period;
    let mut phase_in_challenge_ids: HashSet<String> = active_challenge_ids.clone();
    if phase_in_end <= block_details.height {
        phase_in_challenge_ids.clear();
    } else {
        // keep only challenges where their first algorithms are available this round
        for algorithm_id in active_algorithm_ids.iter() {
            if active_algorithms_state[algorithm_id]
                .round_active
                .as_ref()
                .is_some_and(|r| *r + 1 <= block_details.round)
            {
                phase_in_challenge_ids
                    .remove(&active_algorithms_details[algorithm_id].challenge_id);
            }
        }
    }

    let mut num_solutions_by_player_by_challenge = HashMap::<String, HashMap<String, u32>>::new();
    for (settings, num_solutions, _, _) in active_solutions.iter() {
        *num_solutions_by_player_by_challenge
            .entry(settings.player_id.clone())
            .or_default()
            .entry(settings.challenge_id.clone())
            .or_default() += *num_solutions;
    }
    for (player_id, num_solutions_by_challenge) in num_solutions_by_player_by_challenge.iter() {
        let opow_data = active_opow_block_data.get_mut(player_id).unwrap();

        if num_solutions_by_challenge
            .iter()
            .all(|(challenge_id, &num_solutions)| {
                num_solutions >= config.opow.min_solutions_and_cutoff
                    || phase_in_challenge_ids.contains(challenge_id)
            })
        {
            let prev_ema_percent_qualifiers = active_opow_prev_block_data
                .get(player_id)
                .map(|x| x.ema_percent_qualifiers.clone())
                .unwrap_or_default();
            for challenge_id in active_challenge_ids.iter() {
                let prev_num_qualifiers = active_challenges_prev_block_data
                    .get(challenge_id)
                    .map(|x| x.num_qualifiers.clone())
                    .unwrap_or_default();
                opow_data.cutoff.insert(
                    challenge_id.clone(),
                    config.opow.min_solutions_and_cutoff.max(
                        (prev_ema_percent_qualifiers
                            * config.opow.cutoff_multiplier
                            * config
                                .opow
                                .total_qualifiers_threshold
                                .max(prev_num_qualifiers) as f64)
                            .ceil() as u32,
                    ),
                );
            }
        } else {
            opow_data.cutoff = active_challenge_ids
                .iter()
                .cloned()
                .map(|id| (id, 0))
                .collect();
        }
    }

    // update hash threshold
    let denominator: u64 = 1_000_000_000_000_000;
    let max_delta = U256::MAX / U256::from(denominator)
        * U256::from(
            (config.benchmarks.hash_threshold_max_percent_delta * denominator as f64) as u64,
        );
    for challenge_id in active_challenge_ids.iter() {
        let prev_hash_threshold = active_challenges_prev_block_data
            .get(challenge_id)
            .map(|x| U256::from(x.hash_threshold.clone().0))
            .unwrap_or(U256::MAX);
        let current_solution_rate = *confirmed_num_solutions.get(challenge_id).unwrap_or(&0);
        let target_threshold = if current_solution_rate == 0 {
            U256::MAX
        } else {
            (prev_hash_threshold / U256::from(current_solution_rate))
                .saturating_mul(U256::from(config.benchmarks.target_solution_rate))
        };
        let diff = prev_hash_threshold.abs_diff(target_threshold);
        let delta = (diff / U256::from(100)).min(max_delta);
        let hash_threshold = if prev_hash_threshold > target_threshold {
            prev_hash_threshold.saturating_sub(delta)
        } else {
            prev_hash_threshold.saturating_add(delta)
        };

        let challenge_data = active_challenges_block_data.get_mut(challenge_id).unwrap();
        hash_threshold.to_big_endian(&mut challenge_data.hash_threshold.0);
    }

    // update qualifiers
    let mut solutions_by_challenge =
        HashMap::<String, Vec<(&BenchmarkSettings, &u32, &u32, &u32)>>::new();
    for (settings, num_solutions, num_discarded_solutions, num_nonces) in active_solutions.iter() {
        solutions_by_challenge
            .entry(settings.challenge_id.clone())
            .or_default()
            .push((settings, num_solutions, num_discarded_solutions, num_nonces));
    }

    for challenge_id in active_challenge_ids.iter() {
        if !solutions_by_challenge.contains_key(challenge_id) {
            continue;
        }
        let solutions = solutions_by_challenge.get_mut(challenge_id).unwrap();
        let points = solutions
            .iter()
            .filter(|(_, &num_solutions, _, _)| num_solutions > 0)
            .map(|(settings, _, _, _)| settings.difficulty.clone())
            .collect::<Frontier>();
        let mut frontier_indexes = HashMap::<Point, usize>::new();
        for (frontier_index, frontier) in pareto_algorithm(&points, false).into_iter().enumerate() {
            for point in frontier {
                frontier_indexes.insert(point, frontier_index);
            }
        }
        let mut solutions_by_frontier_idx =
            HashMap::<usize, Vec<(&BenchmarkSettings, &u32, &u32, &u32)>>::new();
        for &x in solutions.iter() {
            if !points.contains(&x.0.difficulty) {
                continue;
            }
            solutions_by_frontier_idx
                .entry(frontier_indexes[&x.0.difficulty])
                .or_default()
                .push(x);
        }

        let cutoff_by_player = active_opow_ids
            .iter()
            .map(|player_id| {
                (
                    player_id.clone(),
                    active_opow_block_data[player_id].cutoff[challenge_id].clone(),
                )
            })
            .collect::<HashMap<String, u32>>();

        let challenge_data = active_challenges_block_data.get_mut(challenge_id).unwrap();
        let mut player_algorithm_solutions = HashMap::<String, HashMap<String, u32>>::new();
        let mut player_solutions = HashMap::<String, u32>::new();
        let mut player_discarded_solutions = HashMap::<String, u32>::new();
        let mut player_nonces = HashMap::<String, u64>::new();

        for frontier_idx in 0..solutions_by_frontier_idx.len() {
            for (settings, &num_solutions, &num_discarded_solutions, &num_nonces) in
                solutions_by_frontier_idx[&frontier_idx].iter()
            {
                let BenchmarkSettings {
                    player_id,
                    algorithm_id,
                    challenge_id,
                    difficulty,
                    ..
                } = settings;

                let difficulty_parameters = &config.challenges.difficulty_parameters[challenge_id];
                let min_difficulty = difficulty_parameters.min_difficulty();
                let max_difficulty = difficulty_parameters.max_difficulty();
                if (0..difficulty.len())
                    .into_iter()
                    .any(|i| difficulty[i] < min_difficulty[i] || difficulty[i] > max_difficulty[i])
                {
                    continue;
                }
                *player_algorithm_solutions
                    .entry(player_id.clone())
                    .or_default()
                    .entry(algorithm_id.clone())
                    .or_default() += num_solutions;
                *player_solutions.entry(player_id.clone()).or_default() += num_solutions;
                *player_discarded_solutions
                    .entry(player_id.clone())
                    .or_default() += num_discarded_solutions;
                *player_nonces.entry(player_id.clone()).or_default() += num_nonces as u64;

                challenge_data
                    .qualifier_difficulties
                    .insert(difficulty.clone());
            }

            // check if we have enough qualifiers
            let player_solution_ratio: HashMap<String, f64> = player_solutions
                .keys()
                .map(|player_id| {
                    (
                        player_id.clone(),
                        (player_solutions[player_id] + player_discarded_solutions[player_id])
                            as f64
                            / player_nonces[player_id] as f64,
                    )
                })
                .collect();
            let player_qualifiers: HashMap<String, u32> = player_solution_ratio
                .keys()
                .map(|player_id| {
                    (
                        player_id.clone(),
                        cutoff_by_player[player_id].min(player_solutions[player_id]),
                    )
                })
                .collect();

            let num_qualifiers = player_qualifiers.values().sum::<u32>();
            if num_qualifiers >= config.opow.total_qualifiers_threshold
                || frontier_idx == solutions_by_frontier_idx.len() - 1
            {
                let mut sum_weighted_solution_ratio = 0.0;
                for player_id in player_qualifiers.keys() {
                    let opow_data = active_opow_block_data.get_mut(player_id).unwrap();
                    opow_data
                        .num_qualifiers_by_challenge
                        .insert(challenge_id.clone(), player_qualifiers[player_id]);
                    opow_data
                        .num_solutions_by_challenge
                        .insert(challenge_id.clone(), player_solutions[player_id]);
                    opow_data
                        .solution_ratio_by_challenge
                        .insert(challenge_id.clone(), player_solution_ratio[player_id]);

                    sum_weighted_solution_ratio +=
                        player_solution_ratio[player_id] * player_qualifiers[player_id] as f64;

                    if player_qualifiers[player_id] > 0 {
                        for algorithm_id in player_algorithm_solutions[player_id].keys() {
                            let algorithm_data =
                                active_algorithms_block_data.get_mut(algorithm_id).unwrap();

                            algorithm_data.num_solutions_by_player.insert(
                                player_id.clone(),
                                player_algorithm_solutions[player_id][algorithm_id].clone(),
                            );
                        }
                    }
                }
                challenge_data.num_qualifiers = num_qualifiers;
                challenge_data.average_solution_ratio = if num_qualifiers == 0 {
                    0.0
                } else {
                    sum_weighted_solution_ratio / num_qualifiers as f64
                };
                break;
            }
        }
    }

    // update frontiers
    for challenge_id in active_challenge_ids.iter() {
        let challenge_data = active_challenges_block_data.get_mut(challenge_id).unwrap();

        let difficulty_parameters = &config.challenges.difficulty_parameters[challenge_id];
        let min_difficulty = difficulty_parameters.min_difficulty();
        let max_difficulty = difficulty_parameters.max_difficulty();

        let points = challenge_data
            .qualifier_difficulties
            .iter()
            .map(|d| d.iter().map(|x| -x).collect()) // mirror the points so easiest difficulties are first
            .collect::<Frontier>();
        let (base_frontier, scaling_factor, scaled_frontier) = if points.len() == 0 {
            let base_frontier: Frontier = vec![min_difficulty.clone()].into_iter().collect();
            let scaling_factor = 1.0;
            let scaled_frontier = base_frontier.clone();
            (base_frontier, scaling_factor, scaled_frontier)
        } else {
            let mut base_frontier = pareto_algorithm(&points, true)
                .pop()
                .unwrap()
                .into_iter()
                .map(|d| d.into_iter().map(|x| -x).collect())
                .collect::<Frontier>(); // mirror the points back;
            base_frontier = extend_frontier(&base_frontier, &min_difficulty, &max_difficulty);

            let mut scaling_factor = (challenge_data.num_qualifiers as f64
                / config.opow.total_qualifiers_threshold as f64)
                .min(config.challenges.max_scaling_factor);

            if scaling_factor < 1.0 {
                base_frontier = scale_frontier(
                    &base_frontier,
                    &min_difficulty,
                    &max_difficulty,
                    scaling_factor,
                );
                base_frontier = extend_frontier(&base_frontier, &min_difficulty, &max_difficulty);
                scaling_factor = (1.0 / scaling_factor).min(config.challenges.max_scaling_factor);
            }

            let mut scaled_frontier = scale_frontier(
                &base_frontier,
                &min_difficulty,
                &max_difficulty,
                scaling_factor,
            );
            scaled_frontier = extend_frontier(&scaled_frontier, &min_difficulty, &max_difficulty);

            (base_frontier, scaling_factor, scaled_frontier)
        };

        challenge_data.base_frontier = base_frontier;
        challenge_data.scaled_frontier = scaled_frontier;
        challenge_data.scaling_factor = scaling_factor;
    }

    // update influence
    if active_opow_ids.len() == 0 {
        return;
    }

    for player_id in active_player_ids.iter() {
        let player_data = active_players_block_data.get_mut(player_id).unwrap();
        let player_state = &active_players_state[player_id];
        if let Some(delegatees) = &player_state.delegatees {
            player_data.delegatees = delegatees
                .value
                .iter()
                .filter(|(delegatee, _)| active_opow_ids.contains(delegatee.as_str()))
                .map(|(delegatee, fraction)| (delegatee.clone(), *fraction))
                .collect();
        } else {
            continue;
        }

        for (delegatee, fraction) in player_data.delegatees.iter() {
            let fraction = PreciseNumber::from_f64(*fraction);
            let opow_data = active_opow_block_data.get_mut(delegatee).unwrap();
            if player_id == delegatee {
                // self deposit
                opow_data.weighted_self_deposit += player_data.weighted_deposit * fraction;
            } else {
                // delegated deposit
                opow_data.delegators.insert(player_id.clone());
                opow_data.weighted_delegated_deposit += player_data.weighted_deposit * fraction;
            }
        }
    }
    let total_weighted_delegated_deposit = active_opow_block_data
        .values()
        .map(|d| d.weighted_delegated_deposit)
        .sum::<PreciseNumber>();
    let total_weighted_self_deposit = active_opow_block_data
        .values()
        .filter(|x| x.cutoff.values().all(|&c| c > 0))
        .map(|d| d.weighted_self_deposit)
        .sum::<PreciseNumber>();

    let zero = PreciseNumber::from(0);
    let one = PreciseNumber::from(1);
    let imbalance_multiplier = PreciseNumber::from_f64(config.opow.imbalance_multiplier);
    let mut factor_weights = active_challenge_ids
        .iter()
        .map(|challenge_id| {
            if phase_in_challenge_ids.contains(challenge_id) {
                PreciseNumber::from(block_details.height - phase_in_start)
                    / PreciseNumber::from(config.opow.challenge_phase_in_period)
            } else {
                one.clone()
            }
        })
        .collect::<Vec<_>>();
    factor_weights.extend(vec![
        PreciseNumber::from_f64(config.opow.deposit_multiplier);
        2
    ]);

    let mut raw_influences = Vec::<PreciseNumber>::new();
    for player_id in active_opow_ids.iter() {
        let opow_data = active_opow_block_data.get_mut(player_id).unwrap();

        let mut factors = Vec::<PreciseNumber>::new();
        for challenge_id in active_challenge_ids.iter() {
            let challenge_data = active_challenges_block_data.get(challenge_id).unwrap();
            factors.push(if challenge_data.num_qualifiers == 0 {
                zero.clone()
            } else {
                PreciseNumber::from(
                    *opow_data
                        .num_qualifiers_by_challenge
                        .get(challenge_id)
                        .unwrap_or(&0),
                ) / PreciseNumber::from(challenge_data.num_qualifiers)
            });
        }

        let weighted_average_challenge_factor = factors
            .iter()
            .zip(factor_weights.iter())
            .map(|(x, w)| x * w)
            .sum::<PreciseNumber>()
            / factor_weights.iter().sum::<PreciseNumber>();
        let max_deposit_to_qualifier_ratio =
            PreciseNumber::from_f64(config.opow.max_deposit_to_qualifier_ratio);

        // exponential moving average = x * (1 - decay) + prev_x * decay
        opow_data.ema_percent_qualifiers = weighted_average_challenge_factor.to_f64()
            * (1.0 - config.opow.ema_decay)
            + active_opow_prev_block_data
                .get(player_id)
                .map(|x| x.ema_percent_qualifiers)
                .unwrap_or_default()
                * config.opow.ema_decay;

        // append deposit factors
        let delegated_deposit_factor = if total_weighted_delegated_deposit == zero {
            zero.clone()
        } else {
            opow_data.weighted_delegated_deposit / total_weighted_delegated_deposit
        };
        if weighted_average_challenge_factor == zero {
            factors.push(zero.clone());
        } else if delegated_deposit_factor / weighted_average_challenge_factor
            > max_deposit_to_qualifier_ratio
        {
            factors.push(weighted_average_challenge_factor * max_deposit_to_qualifier_ratio);
        }

        // self deposit factor
        let self_deposit_factor = if total_weighted_self_deposit == zero {
            zero.clone()
        } else {
            opow_data.weighted_self_deposit / total_weighted_self_deposit
        };
        if weighted_average_challenge_factor == zero {
            factors.push(zero.clone());
        } else if self_deposit_factor / weighted_average_challenge_factor
            > max_deposit_to_qualifier_ratio
        {
            factors.push(weighted_average_challenge_factor * max_deposit_to_qualifier_ratio);
        }

        let weighted_average_factor = factors
            .iter()
            .zip(factor_weights.iter())
            .map(|(x, w)| x * w)
            .sum::<PreciseNumber>()
            / factor_weights.iter().sum::<PreciseNumber>();
        let weighted_variance = factors
            .iter()
            .zip(factor_weights.iter())
            .map(|(x, w)| {
                let diff = if x > weighted_average_factor {
                    x - weighted_average_factor
                } else {
                    weighted_average_factor - x
                };
                diff * diff * w
            })
            .sum::<PreciseNumber>()
            / factor_weights.iter().sum::<PreciseNumber>();

        let cv_sqr = if weighted_average_factor == zero {
            zero.clone()
        } else {
            weighted_variance / (weighted_average_factor * weighted_average_factor)
        };

        let imbalance = cv_sqr / (factor_weights.iter().sum::<PreciseNumber>() - one);
        raw_influences.push(
            weighted_average_factor
                * PreciseNumber::approx_inv_exp(imbalance_multiplier * imbalance),
        );
        opow_data.imbalance = imbalance;
    }

    let influences = raw_influences.normalise();
    for (player_id, &influence) in active_opow_ids.iter().zip(influences.iter()) {
        let data = active_opow_block_data.get_mut(player_id).unwrap();
        data.influence = influence;
    }
}
