use crate::context::*;
use logging_timer::time;
use rand::seq::SliceRandom;
use std::collections::{HashMap, HashSet};
use tig_structs::core::*;
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

    let mut num_bundles_by_player_by_challenge = HashMap::<String, HashMap<String, u64>>::new();
    for (settings, average_quality_by_bundle) in active_benchmarks.iter() {
        if config.challenges[&settings.challenge_id]
            .active_tracks
            .contains_key(&settings.track_id)
        {
            *num_bundles_by_player_by_challenge
                .entry(settings.player_id.clone())
                .or_default()
                .entry(settings.challenge_id.clone())
                .or_default() += average_quality_by_bundle.len() as u64;
        }
    }
    for (player_id, player_bundles_by_challenge) in num_bundles_by_player_by_challenge.iter() {
        let opow_data = active_opow_block_data.get_mut(player_id).unwrap();
        let min_num_bundles = active_challenge_ids
            .iter()
            .map(|id| {
                player_bundles_by_challenge
                    .get(id)
                    .cloned()
                    .unwrap_or_default()
            })
            .min()
            .unwrap_or_default();
        let mut cutoff = (min_num_bundles as f64 * config.opow.cutoff_multiplier).ceil() as u64;
        if phase_in_challenge_ids.len() > 0 && phase_in_end > block_details.height {
            let phase_in_min_num_groups = active_challenge_ids
                .iter()
                .filter(|&id| !phase_in_challenge_ids.contains(id))
                .map(|id| {
                    player_bundles_by_challenge
                        .get(id)
                        .cloned()
                        .unwrap_or_default()
                })
                .min()
                .unwrap_or_default();
            let phase_in_cutoff =
                (phase_in_min_num_groups as f64 * config.opow.cutoff_multiplier).ceil() as u64;
            let phase_in_weight =
                (phase_in_end - block_details.height) as f64 / phase_in_period as f64;
            cutoff = (phase_in_cutoff as f64 * phase_in_weight
                + cutoff as f64 * (1.0 - phase_in_weight)) as u64;
        }
        opow_data.cutoff = cutoff;
    }

    // update qualifiers
    let mut bundles_by_challenge_by_track =
        HashMap::<String, HashMap<String, Vec<(&BenchmarkSettings, i32)>>>::new();
    for (settings, average_quality_by_bundle) in active_benchmarks.iter() {
        if config.challenges[&settings.challenge_id]
            .active_tracks
            .contains_key(&settings.track_id)
        {
            bundles_by_challenge_by_track
                .entry(settings.challenge_id.clone())
                .or_default()
                .entry(settings.track_id.clone())
                .or_default()
                .extend(
                    average_quality_by_bundle
                        .iter()
                        .map(|&quality| (settings, quality)),
                );
        }
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

    let mut rng = rand::thread_rng();
    for challenge_id in active_challenge_ids.iter() {
        let challenge_config = &config.challenges[challenge_id];
        let challenge_data = active_challenges_block_data.get_mut(challenge_id).unwrap();
        for track_id in challenge_config.active_tracks.keys() {
            challenge_data
                .qualifier_qualities_by_track
                .insert(track_id.clone(), HashSet::new());
            challenge_data
                .num_qualifiers_by_track
                .insert(track_id.clone(), 0u64);
        }

        if let Some(bundles_by_track) = bundles_by_challenge_by_track.remove(challenge_id) {
            let bundles_by_track = bundles_by_track
                .into_iter()
                .filter(|(_, x)| !x.is_empty())
                .map(|(track_id, mut x)| {
                    // shuffle to break ties randomly
                    x.shuffle(&mut rng);
                    x.sort_by_key(|&(_, quality)| -quality);
                    (track_id, x)
                })
                .collect::<HashMap<_, _>>();
            let mut num_qualifiers_by_code_by_track_by_player =
                HashMap::<String, HashMap<String, HashMap<String, u64>>>::new();
            let mut num_qualifiers_by_player_by_track =
                HashMap::<String, HashMap<String, u64>>::new();
            let mut num_qualifiers_by_player = HashMap::<String, u64>::new();

            let mut rank = 0;
            let mut track_ids = bundles_by_track.keys().cloned().collect::<Vec<_>>();
            while !track_ids.is_empty() {
                track_ids.shuffle(&mut rng);
                track_ids.retain(|track_id| {
                    let (
                        BenchmarkSettings {
                            player_id,
                            algorithm_id,
                            ..
                        },
                        quality,
                    ) = &bundles_by_track[track_id][rank];

                    let player_qualifiers = num_qualifiers_by_player
                        .entry(player_id.clone())
                        .or_default();
                    if *player_qualifiers < max_qualifiers_by_player[player_id] {
                        challenge_data
                            .qualifier_qualities_by_track
                            .get_mut(track_id)
                            .unwrap()
                            .insert(*quality);
                        let num_qualifiers = challenge_data
                            .num_qualifiers_by_track
                            .get_mut(track_id)
                            .unwrap();
                        *num_qualifiers += 1;
                        *num_qualifiers_by_code_by_track_by_player
                            .entry(algorithm_id.clone())
                            .or_default()
                            .entry(track_id.clone())
                            .or_default()
                            .entry(player_id.clone())
                            .or_default() += 1;
                        *num_qualifiers_by_player_by_track
                            .entry(player_id.clone())
                            .or_default()
                            .entry(track_id.clone())
                            .or_default() += 1;
                        *player_qualifiers += 1;

                        *num_qualifiers < challenge_config.max_qualifiers_per_track
                            && rank + 1 < bundles_by_track[track_id].len()
                    } else {
                        rank + 1 < bundles_by_track[track_id].len()
                    }
                });
                rank += 1;
            }

            for (player_id, num_qualifiers_by_track) in num_qualifiers_by_player_by_track {
                let opow_data = active_opow_block_data.get_mut(&player_id).unwrap();
                opow_data
                    .num_qualifiers_by_challenge_by_track
                    .insert(challenge_id.clone(), num_qualifiers_by_track);
            }
            for (algorithm_id, num_qualifiers_by_track_by_player) in
                num_qualifiers_by_code_by_track_by_player
            {
                let code_data = active_codes_block_data.get_mut(&algorithm_id).unwrap();
                code_data.num_qualifiers_by_track_by_player = num_qualifiers_by_track_by_player;
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
            let total_qualifiers = challenge_data.num_qualifiers_by_track.values().sum::<u64>();
            factors.push(if total_qualifiers == 0 {
                zero.clone()
            } else {
                PreciseNumber::from(
                    opow_data
                        .num_qualifiers_by_challenge_by_track
                        .get(challenge_id)
                        .map(|x| x.values().sum::<u64>())
                        .unwrap_or_default(),
                ) / PreciseNumber::from(total_qualifiers)
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
