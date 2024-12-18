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
        active_algorithms_state,
        active_algorithms_details,
        active_algorithms_block_data,
        active_solutions,
        active_players_state,
        active_players_block_data,
        active_opow_block_data,
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
    for (settings, num_solutions) in active_solutions.values() {
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

    // update qualifiers
    let mut solutions_by_challenge = HashMap::<String, Vec<(&BenchmarkSettings, &u32)>>::new();
    for (settings, num_solutions) in active_solutions.values() {
        solutions_by_challenge
            .entry(settings.challenge_id.clone())
            .or_default()
            .push((settings, num_solutions));
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
            .map(|(settings, _)| settings.difficulty.clone())
            .collect::<Frontier>();
        let mut frontier_indexes = HashMap::<Point, usize>::new();
        for (frontier_index, frontier) in pareto_algorithm(&points, false).into_iter().enumerate() {
            for point in frontier {
                frontier_indexes.insert(point, frontier_index);
            }
        }
        solutions.sort_by(|(a_settings, _), (b_settings, _)| {
            let a_index = frontier_indexes[&a_settings.difficulty];
            let b_index = frontier_indexes[&b_settings.difficulty];
            a_index.cmp(&b_index)
        });

        let mut max_qualifiers_by_player = max_qualifiers_by_player.clone();
        let mut curr_frontier_index = 0;
        let challenge_data = active_challenges_block_data.get_mut(challenge_id).unwrap();
        for (settings, &num_solutions) in solutions.iter() {
            let BenchmarkSettings {
                player_id,
                algorithm_id,
                challenge_id,
                difficulty,
                ..
            } = settings;

            if curr_frontier_index != frontier_indexes[difficulty]
                && challenge_data.num_qualifiers > config.opow.total_qualifiers_threshold
            {
                break;
            }
            let difficulty_parameters = &config.challenges.difficulty_parameters[challenge_id];
            let min_difficulty = difficulty_parameters.min_difficulty();
            let max_difficulty = difficulty_parameters.max_difficulty();
            if (0..difficulty.len())
                .into_iter()
                .any(|i| difficulty[i] < min_difficulty[i] || difficulty[i] > max_difficulty[i])
            {
                continue;
            }
            curr_frontier_index = frontier_indexes[difficulty];
            let player_data = active_opow_block_data.get_mut(player_id).unwrap();
            let algorithm_data = active_algorithms_block_data.get_mut(algorithm_id).unwrap();

            let max_qualifiers = max_qualifiers_by_player.get(player_id).unwrap().clone();
            let num_qualifiers = num_solutions.min(max_qualifiers);
            max_qualifiers_by_player.insert(player_id.clone(), max_qualifiers - num_qualifiers);

            if num_qualifiers > 0 {
                *player_data
                    .num_qualifiers_by_challenge
                    .entry(challenge_id.clone())
                    .or_default() += num_qualifiers;
                *algorithm_data
                    .num_qualifiers_by_player
                    .entry(player_id.clone())
                    .or_default() += num_qualifiers;
                challenge_data.num_qualifiers += num_qualifiers;
            }
            challenge_data
                .qualifier_difficulties
                .insert(difficulty.clone());
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

    let num_qualifiers_by_challenge: HashMap<String, u32> = active_challenge_ids
        .iter()
        .map(|challenge_id| {
            (
                challenge_id.clone(),
                active_challenges_block_data[challenge_id]
                    .num_qualifiers
                    .clone(),
            )
        })
        .collect();

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

        let mut percent_qualifiers = Vec::<PreciseNumber>::new();
        for challenge_id in active_challenge_ids.iter() {
            let num_qualifiers = num_qualifiers_by_challenge[challenge_id];
            let num_qualifiers_by_player = *opow_data
                .num_qualifiers_by_challenge
                .get(challenge_id)
                .unwrap_or(&0);

            percent_qualifiers.push(if num_qualifiers_by_player == 0 {
                PreciseNumber::from(0)
            } else {
                PreciseNumber::from(num_qualifiers_by_player) / PreciseNumber::from(num_qualifiers)
            });
        }

        let mut percent_deposit = if total_deposit == zero {
            zero.clone()
        } else {
            opow_data.delegated_weighted_deposit / total_deposit
        };
        let mean_percent_qualifiers = percent_qualifiers.arithmetic_mean();
        let max_deposit_to_qualifier_ratio =
            PreciseNumber::from_f64(config.opow.max_deposit_to_qualifier_ratio);
        if mean_percent_qualifiers == zero {
            percent_deposit = zero.clone();
        } else if percent_deposit / mean_percent_qualifiers > max_deposit_to_qualifier_ratio {
            percent_deposit = mean_percent_qualifiers * max_deposit_to_qualifier_ratio;
        }

        let sum_percent_qualifiers: PreciseNumber = percent_qualifiers.iter().cloned().sum();
        let weighted_mean = (sum_percent_qualifiers
            + percent_deposit * PreciseNumber::from_f64(config.opow.deposit_multiplier))
            / PreciseNumber::from_f64(
                active_challenge_ids.len() as f64 + config.opow.deposit_multiplier,
            );
        let mut qualifiers_and_deposit = percent_qualifiers;
        qualifiers_and_deposit.push(percent_deposit);
        let mean = qualifiers_and_deposit.arithmetic_mean();
        let variance = qualifiers_and_deposit.variance();
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
