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
        confirmed_num_solutions,
        ..
    } = cache;

    let active_algorithm_ids = &block_data.active_ids[&ActiveType::Algorithm];
    let active_challenge_ids = &block_data.active_ids[&ActiveType::Challenge];
    let active_player_ids = &block_data.active_ids[&ActiveType::Player];
    let active_opow_ids = &block_data.active_ids[&ActiveType::OPoW];

    // update cutoffs
    let self_deposit = active_player_ids
        .iter()
        .map(|player_id| {
            (
                player_id.clone(),
                active_players_block_data[player_id]
                    .deposit_by_locked_period
                    .iter()
                    .cloned()
                    .sum(),
            )
        })
        .collect::<HashMap<String, PreciseNumber>>();

    let mut phase_in_challenge_ids: HashSet<String> = active_challenge_ids.clone();
    for algorithm_id in active_algorithm_ids.iter() {
        if active_algorithms_state[algorithm_id]
            .round_active
            .as_ref()
            .is_some_and(|r| *r + 1 <= block_details.round)
        {
            phase_in_challenge_ids.remove(&active_algorithms_details[algorithm_id].challenge_id);
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
    let deposit_to_cutoff_cap_ratio = PreciseNumber::from_f64(config.opow.deposit_to_cutoff_ratio);
    for (player_id, num_solutions_by_challenge) in num_solutions_by_player_by_challenge.iter() {
        let opow_data = active_opow_block_data.get_mut(player_id).unwrap();
        let phase_in_start = (block_details.round - 1) * config.rounds.blocks_per_round;
        let phase_in_period = config.opow.cutoff_phase_in_period;
        let phase_in_end = phase_in_start + phase_in_period;
        let cutoff_cap = (self_deposit[player_id] / deposit_to_cutoff_cap_ratio).to_f64() as u32;
        let min_num_solutions = active_challenge_ids
            .iter()
            .map(|id| num_solutions_by_challenge.get(id).unwrap_or(&0).clone())
            .min()
            .unwrap();
        let mut cutoff = cutoff_cap
            .min((min_num_solutions as f64 * config.opow.cutoff_multiplier).ceil() as u32);
        if phase_in_challenge_ids.len() > 0 && phase_in_end > block_details.height {
            let phase_in_min_num_solutions = active_challenge_ids
                .iter()
                .filter(|&id| !phase_in_challenge_ids.contains(id))
                .map(|id| num_solutions_by_challenge.get(id).unwrap_or(&0).clone())
                .min()
                .unwrap();
            let phase_in_cutoff = cutoff_cap.min(
                (phase_in_min_num_solutions as f64 * config.opow.cutoff_multiplier).ceil() as u32,
            );
            let phase_in_weight =
                (phase_in_end - block_details.height) as f64 / phase_in_period as f64;
            cutoff = (phase_in_cutoff as f64 * phase_in_weight
                + cutoff as f64 * (1.0 - phase_in_weight)) as u32;
        }
        opow_data.cutoff = cutoff;
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

    let max_qualifiers_by_player = active_opow_ids
        .iter()
        .map(|player_id| {
            (
                player_id.clone(),
                active_opow_block_data[player_id].cutoff.clone(),
            )
        })
        .collect::<HashMap<String, u32>>();

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

        let challenge_data = active_challenges_block_data.get_mut(challenge_id).unwrap();
        let min_num_nonces = config.opow.min_num_nonces as u64;
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
                        if player_nonces[player_id] >= min_num_nonces {
                            max_qualifiers_by_player[player_id].min(player_solutions[player_id])
                        } else {
                            0
                        },
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
                        .solution_ratio_by_challenge
                        .insert(challenge_id.clone(), player_solution_ratio[player_id]);

                    sum_weighted_solution_ratio +=
                        player_solution_ratio[player_id] * player_qualifiers[player_id] as f64;

                    if player_qualifiers[player_id] > 0 {
                        for algorithm_id in player_algorithm_solutions[player_id].keys() {
                            let algorithm_data =
                                active_algorithms_block_data.get_mut(algorithm_id).unwrap();

                            algorithm_data.num_qualifiers_by_player.insert(
                                player_id.clone(),
                                (player_qualifiers[player_id] as f64
                                    * player_algorithm_solutions[player_id][algorithm_id] as f64
                                    / player_solutions[player_id] as f64)
                                    .ceil() as u32,
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

    for player_id in active_opow_ids.iter() {
        let opow_data = active_opow_block_data.get_mut(player_id).unwrap();
        opow_data.self_deposit = self_deposit[player_id].clone();
    }

    for player_id in active_player_ids.iter() {
        let player_data = active_players_block_data.get_mut(player_id).unwrap();
        let player_state = &active_players_state[player_id];
        if active_opow_ids.contains(player_id) {
            // benchmarkers self-delegate 100% to themselves
            player_data.delegatees = HashMap::from([(player_id.clone(), 1.0)]);
        } else if let Some(delegatees) = &player_state.delegatees {
            player_data.delegatees = delegatees
                .value
                .iter()
                .filter(|(delegatee, _)| {
                    active_opow_ids.contains(delegatee.as_str())
                        && self_deposit[delegatee.as_str()] >= config.deposits.delegatee_min_deposit
                })
                .map(|(delegatee, fraction)| (delegatee.clone(), *fraction))
                .collect();
        } else {
            continue;
        }

        for (delegatee, fraction) in player_data.delegatees.iter() {
            let fraction = PreciseNumber::from_f64(*fraction);
            let opow_data = active_opow_block_data.get_mut(delegatee).unwrap();
            opow_data.delegators.insert(player_id.clone());
            opow_data.delegated_weighted_deposit += player_data.weighted_deposit * fraction;
        }
    }
    let total_deposit = active_opow_block_data
        .values()
        .map(|d| d.delegated_weighted_deposit)
        .sum::<PreciseNumber>();

    let zero = PreciseNumber::from(0);
    let imbalance_multiplier = PreciseNumber::from_f64(config.opow.imbalance_multiplier);
    let num_challenges = PreciseNumber::from(active_challenge_ids.len());

    let mut weights = Vec::<PreciseNumber>::new();
    for player_id in active_opow_ids.iter() {
        let opow_data = active_opow_block_data.get_mut(player_id).unwrap();

        let mut challenge_factors = Vec::<PreciseNumber>::new();
        for challenge_id in active_challenge_ids.iter() {
            let challenge_data = active_challenges_block_data.get(challenge_id).unwrap();
            challenge_factors.push(if challenge_data.num_qualifiers == 0 {
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

        let mut deposit_factor = if total_deposit == zero {
            zero.clone()
        } else {
            opow_data.delegated_weighted_deposit / total_deposit
        };
        let mean_challenge_factor = challenge_factors.arithmetic_mean();
        let max_deposit_to_qualifier_ratio =
            PreciseNumber::from_f64(config.opow.max_deposit_to_qualifier_ratio);
        if mean_challenge_factor == zero {
            deposit_factor = zero.clone();
        } else if deposit_factor / mean_challenge_factor > max_deposit_to_qualifier_ratio {
            deposit_factor = mean_challenge_factor * max_deposit_to_qualifier_ratio;
        }

        let sum_challenge_factors: PreciseNumber = challenge_factors.iter().cloned().sum();
        let weighted_mean = (sum_challenge_factors
            + deposit_factor * PreciseNumber::from_f64(config.opow.deposit_multiplier))
            / PreciseNumber::from_f64(
                active_challenge_ids.len() as f64 + config.opow.deposit_multiplier,
            );
        let mut all_factors = challenge_factors;
        all_factors.push(deposit_factor);
        let mean = all_factors.arithmetic_mean();
        let variance = all_factors.variance();
        let cv_sqr = if mean == zero {
            zero.clone()
        } else {
            variance / (mean * mean)
        };

        let imbalance = cv_sqr / num_challenges; // no need minus 1, because deposit is extra factor
        weights
            .push(weighted_mean * PreciseNumber::approx_inv_exp(imbalance_multiplier * imbalance));
        opow_data.imbalance = imbalance;
    }

    let influences = weights.normalise();
    for (player_id, &influence) in active_opow_ids.iter().zip(influences.iter()) {
        let data = active_opow_block_data.get_mut(player_id).unwrap();
        data.influence = influence;
    }
}
