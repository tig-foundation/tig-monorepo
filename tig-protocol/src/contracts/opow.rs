use crate::context::*;
use logging_timer::time;
use std::collections::{HashMap, HashSet};
use tig_structs::{config::*, core::*};
use tig_utils::*;

const EPSILON: f64 = 1e-9;

#[time]
pub(crate) async fn update(cache: &mut AddBlockCache) {
    let AddBlockCache {
        config,
        block_details,
        block_data,
        active_challenges_block_data,
        active_codes_state,
        active_codes_details,
        active_codes_block_data,
        active_benchmarks,
        active_players_state,
        active_players_block_data,
        active_opow_block_data,
        ..
    } = cache;

    let active_code_ids = &block_data.active_ids[&ActiveType::Code];
    let active_challenge_ids = &block_data.active_ids[&ActiveType::Challenge];
    let active_player_ids = &block_data.active_ids[&ActiveType::Player];
    let active_opow_ids = &block_data.active_ids[&ActiveType::OPoW];

    // update cutoffs
    let mut phase_in_challenge_ids: HashSet<String> = active_challenge_ids.clone();
    for algorithm_id in active_code_ids.iter() {
        if active_codes_state[algorithm_id]
            .round_active
            .as_ref()
            .is_some_and(|r| *r + 1 <= block_details.round)
        {
            phase_in_challenge_ids.remove(&active_codes_details[algorithm_id].challenge_id);
        }
    }
    let phase_in_start = (block_details.round - 1) * config.rounds.blocks_per_round;
    let phase_in_period = config.opow.cutoff_phase_in_period;
    let phase_in_end = phase_in_start + phase_in_period;

    let mut num_nonces_by_player_by_challenge = HashMap::<String, HashMap<String, u64>>::new();
    for (settings, _, num_nonces) in active_benchmarks.iter() {
        *num_nonces_by_player_by_challenge
            .entry(settings.player_id.clone())
            .or_default()
            .entry(settings.challenge_id.clone())
            .or_default() += *num_nonces;
    }
    for (player_id, num_nonces_by_challenge) in num_nonces_by_player_by_challenge.iter() {
        let opow_data = active_opow_block_data.get_mut(player_id).unwrap();
        let min_num_nonces = active_challenge_ids
            .iter()
            .map(|id| num_nonces_by_challenge.get(id).cloned().unwrap_or_default())
            .min()
            .unwrap_or_default();
        let mut cutoff = (min_num_nonces as f64 * config.opow.cutoff_multiplier).ceil() as u64;
        if phase_in_challenge_ids.len() > 0 && phase_in_end > block_details.height {
            let phase_in_min_num_nonces = active_challenge_ids
                .iter()
                .filter(|&id| !phase_in_challenge_ids.contains(id))
                .map(|id| num_nonces_by_challenge.get(id).cloned().unwrap_or_default())
                .min()
                .unwrap_or_default();
            let phase_in_cutoff =
                (phase_in_min_num_nonces as f64 * config.opow.cutoff_multiplier).ceil() as u64;
            let phase_in_weight =
                (phase_in_end - block_details.height) as f64 / phase_in_period as f64;
            cutoff = (phase_in_cutoff as f64 * phase_in_weight
                + cutoff as f64 * (1.0 - phase_in_weight)) as u64;
        }
        opow_data.cutoff = cutoff;
    }

    // update qualifiers
    let mut benchmarks_by_challenge =
        HashMap::<String, Vec<(&BenchmarkSettings, &i32, &u64, Point)>>::new();
    for (settings, average_solution_quality, num_nonces) in active_benchmarks.iter() {
        benchmarks_by_challenge
            .entry(settings.challenge_id.clone())
            .or_default()
            .push((
                settings,
                average_solution_quality,
                num_nonces,
                vec![settings.size as i32, average_solution_quality.clone()],
            ));
    }

    let max_qualifiers_by_player = active_opow_ids
        .iter()
        .map(|player_id| {
            (
                player_id.clone(),
                active_opow_block_data[player_id].cutoff.clone(),
            )
        })
        .collect::<HashMap<String, u64>>();

    for challenge_id in active_challenge_ids.iter() {
        if !benchmarks_by_challenge.contains_key(challenge_id) {
            continue;
        }
        let challenge_config = &config.challenges[challenge_id];
        let benchmarks = benchmarks_by_challenge.get_mut(challenge_id).unwrap();
        let points = benchmarks
            .iter()
            .map(|(_, _, _, difficulty)| difficulty.clone())
            .collect::<Frontier>();
        let mut frontier_indexes = HashMap::<Point, usize>::new();
        for (frontier_index, frontier) in pareto_algorithm(&points, false).into_iter().enumerate() {
            for point in frontier {
                frontier_indexes.insert(point, frontier_index);
            }
        }
        let mut benchmarks_by_frontier_idx =
            HashMap::<usize, Vec<&(&BenchmarkSettings, &i32, &u64, Point)>>::new();
        for x in benchmarks.iter() {
            benchmarks_by_frontier_idx
                .entry(frontier_indexes[&x.3])
                .or_default()
                .push(x);
        }

        let challenge_data = active_challenges_block_data.get_mut(challenge_id).unwrap();
        let mut player_code_nonces = HashMap::<String, HashMap<String, u64>>::new();
        let mut player_nonces = HashMap::<String, u64>::new();

        for frontier_idx in 0..benchmarks_by_frontier_idx.len() {
            for (settings, _, &num_nonces, difficulty) in
                benchmarks_by_frontier_idx[&frontier_idx].iter()
            {
                let BenchmarkSettings {
                    player_id,
                    algorithm_id,
                    ..
                } = settings;

                *player_code_nonces
                    .entry(player_id.clone())
                    .or_default()
                    .entry(algorithm_id.clone())
                    .or_default() += num_nonces;
                *player_nonces.entry(player_id.clone()).or_default() += num_nonces;

                challenge_data
                    .qualifier_difficulties
                    .insert(difficulty.clone());
            }

            // check if we have enough qualifiers
            let player_qualifiers: HashMap<String, u64> = player_nonces
                .keys()
                .map(|player_id| {
                    (
                        player_id.clone(),
                        max_qualifiers_by_player[player_id].min(player_nonces[player_id]),
                    )
                })
                .collect();

            let num_qualifiers = player_qualifiers.values().sum::<u64>();
            if num_qualifiers >= challenge_config.difficulty.total_qualifiers_threshold
                || frontier_idx == benchmarks_by_frontier_idx.len() - 1
            {
                for player_id in player_qualifiers.keys() {
                    let opow_data = active_opow_block_data.get_mut(player_id).unwrap();
                    opow_data
                        .num_qualifiers_by_challenge
                        .insert(challenge_id.clone(), player_qualifiers[player_id]);

                    if player_qualifiers[player_id] > 0 {
                        for algorithm_id in player_code_nonces[player_id].keys() {
                            if !active_code_ids.contains(algorithm_id) {
                                continue; // algorithm is banned
                            }
                            let code_data = active_codes_block_data.get_mut(algorithm_id).unwrap();

                            code_data.num_qualifiers_by_player.insert(
                                player_id.clone(),
                                (player_qualifiers[player_id] as f64
                                    * player_code_nonces[player_id][algorithm_id] as f64
                                    / player_nonces[player_id] as f64)
                                    .ceil() as u64,
                            );
                        }
                    }
                }
                challenge_data.num_qualifiers = num_qualifiers;
                break;
            }
        }
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
        }

        // self deposit
        if active_opow_ids.contains(player_id) {
            let opow_data = active_opow_block_data.get_mut(player_id).unwrap();
            let self_deposit_fraction = PreciseNumber::from_f64(
                (1.0 - player_data.delegatees.values().sum::<f64>()).clamp(0.0, 1.0),
            );
            opow_data.weighted_self_deposit += player_data.weighted_deposit * self_deposit_fraction;
        }

        // delegated deposit
        for (delegatee, fraction) in player_data.delegatees.iter() {
            let fraction = PreciseNumber::from_f64(*fraction);
            let opow_data = active_opow_block_data.get_mut(delegatee).unwrap();
            opow_data.delegators.insert(player_id.clone());
            opow_data.weighted_delegated_deposit += player_data.weighted_deposit * fraction;
        }
    }
    let total_weighted_delegated_deposit = active_opow_block_data
        .values()
        .filter(|x| x.cutoff > 0)
        .map(|d| d.weighted_delegated_deposit)
        .sum::<PreciseNumber>();
    let total_weighted_self_deposit = active_opow_block_data
        .values()
        .filter(|x| x.cutoff > 0)
        .map(|d| d.weighted_self_deposit)
        .sum::<PreciseNumber>();

    let zero = PreciseNumber::from(0);
    let one = PreciseNumber::from(1);
    let imbalance_multiplier = PreciseNumber::from_f64(config.opow.imbalance_multiplier);
    let mut factor_weights = active_challenge_ids
        .iter()
        .map(|challenge_id| {
            if phase_in_challenge_ids.contains(challenge_id) && block_details.height < phase_in_end
            {
                PreciseNumber::from(block_details.height - phase_in_start)
                    / PreciseNumber::from(config.opow.cutoff_phase_in_period)
            } else {
                one.clone()
            }
        })
        .collect::<Vec<_>>()
        .normalise()
        .into_iter()
        .map(|x| x * PreciseNumber::from_f64(config.opow.challenge_factors_weight))
        .collect::<Vec<_>>();
    factor_weights.extend(vec![
        PreciseNumber::from_f64(
            (1.0 - config.opow.challenge_factors_weight) / 2.0
        );
        2
    ]);

    let mut weights = Vec::<PreciseNumber>::new();
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
            / PreciseNumber::from_f64(config.opow.challenge_factors_weight);
        let max_deposit_to_qualifier_ratio =
            PreciseNumber::from_f64(config.opow.max_deposit_to_qualifier_ratio);
        for (deposit_factor, total) in [
            (
                &opow_data.weighted_self_deposit,
                &total_weighted_self_deposit,
            ),
            (
                &opow_data.weighted_delegated_deposit,
                &total_weighted_delegated_deposit,
            ),
        ] {
            // if cutoff is 0, then deposit will be limited to 0
            let f = if total.to_f64() <= EPSILON {
                zero.clone()
            } else {
                deposit_factor / total
            };
            factors.push(if weighted_average_challenge_factor.to_f64() <= EPSILON {
                zero.clone()
            } else if f / weighted_average_challenge_factor > max_deposit_to_qualifier_ratio {
                weighted_average_challenge_factor * max_deposit_to_qualifier_ratio
            } else {
                f
            });
        }

        let weighted_average_factor = factors
            .iter()
            .zip(factor_weights.iter())
            .map(|(x, w)| x * w)
            .sum::<PreciseNumber>();
        let weighted_variance_factor = factors
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
            .sum::<PreciseNumber>();
        let imbalance = if weighted_variance_factor.to_f64() <= EPSILON
            || weighted_average_factor.to_f64() <= EPSILON
            || (1.0 - weighted_average_factor.to_f64()) <= EPSILON
        {
            zero.clone()
        } else {
            weighted_variance_factor / (weighted_average_factor * (one - weighted_average_factor))
        };
        weights.push(
            weighted_average_factor
                * PreciseNumber::approx_inv_exp(imbalance_multiplier * imbalance),
        );
        opow_data.imbalance = imbalance;
    }

    let influences = weights.normalise();
    for (player_id, &influence) in active_opow_ids.iter().zip(influences.iter()) {
        let data = active_opow_block_data.get_mut(player_id).unwrap();
        data.influence = influence;
    }
}
