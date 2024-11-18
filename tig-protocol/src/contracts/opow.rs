#[time]
async fn update_cutoffs(block: &Block, cache: &mut AddBlockCache) {
    let config = block.config();
    let mut phase_in_challenge_ids: HashSet<String> =
        cache.active_challenges.keys().cloned().collect();
    for algorithm in cache.active_algorithms.values() {
        if algorithm
            .state()
            .round_pushed
            .is_some_and(|r| r + 1 <= block.details.round)
        {
            phase_in_challenge_ids.remove(&algorithm.details.challenge_id);
        }
    }

    let mut num_solutions_by_player_by_challenge = HashMap::<String, HashMap<String, u32>>::new();
    for (settings, num_solutions) in cache.active_solutions.values() {
        *num_solutions_by_player_by_challenge
            .entry(settings.player_id.clone())
            .or_default()
            .entry(settings.challenge_id.clone())
            .or_default() += *num_solutions;
    }

    for (player_id, num_solutions_by_challenge) in num_solutions_by_player_by_challenge.iter() {
        let data = cache
            .active_players
            .get_mut(player_id)
            .unwrap()
            .block_data
            .as_mut()
            .unwrap();
        let phase_in_start = (block.details.round - 1) * config.rounds.blocks_per_round;
        let phase_in_period = config.qualifiers.cutoff_phase_in_period.unwrap();
        let phase_in_end = phase_in_start + phase_in_period;
        let min_cutoff = config.qualifiers.min_cutoff.clone().unwrap();
        let min_num_solutions = cache
            .active_challenges
            .keys()
            .map(|id| num_solutions_by_challenge.get(id).unwrap_or(&0).clone())
            .min()
            .unwrap();
        let mut cutoff = min_cutoff
            .max((min_num_solutions as f64 * config.qualifiers.cutoff_multiplier).ceil() as u32);
        if phase_in_challenge_ids.len() > 0 && phase_in_end > block.details.height {
            let phase_in_min_num_solutions = cache
                .active_challenges
                .keys()
                .filter(|&id| !phase_in_challenge_ids.contains(id))
                .map(|id| num_solutions_by_challenge.get(id).unwrap_or(&0).clone())
                .min()
                .unwrap();
            let phase_in_cutoff = min_cutoff.max(
                (phase_in_min_num_solutions as f64 * config.qualifiers.cutoff_multiplier).ceil()
                    as u32,
            );
            let phase_in_weight =
                (phase_in_end - block.details.height) as f64 / phase_in_period as f64;
            cutoff = (phase_in_cutoff as f64 * phase_in_weight
                + cutoff as f64 * (1.0 - phase_in_weight)) as u32;
        }
        data.cutoff = Some(cutoff);
    }
}

fn find_smallest_range_dimension(points: &Frontier) -> usize {
    (0..2)
        .min_by_key(|&d| {
            let (min, max) = points
                .iter()
                .map(|p| p[d])
                .fold((i32::MAX, i32::MIN), |(min, max), val| {
                    (min.min(val), max.max(val))
                });
            max - min
        })
        .unwrap()
}

fn pareto_algorithm(points: Frontier, only_one: bool) -> Vec<Frontier> {
    if points.is_empty() {
        return Vec::new();
    }
    let dimension = find_smallest_range_dimension(&points);
    let sort_dimension = 1 - dimension;

    let mut buckets: HashMap<i32, Vec<Point>> = HashMap::new();
    for point in points {
        buckets.entry(point[dimension]).or_default().push(point);
    }
    for (_, group) in buckets.iter_mut() {
        // sort descending
        group.sort_unstable_by(|a, b| b[sort_dimension].cmp(&a[sort_dimension]));
    }
    let mut result = Vec::new();
    while !buckets.is_empty() {
        let points: HashSet<Point> = buckets.values().map(|group| group[0].clone()).collect();
        let frontier = points.pareto_frontier();
        for point in frontier.iter() {
            let bucket = buckets.get_mut(&point[dimension]).unwrap();
            bucket.remove(0);
            if bucket.is_empty() {
                buckets.remove(&point[dimension]);
            }
        }
        result.push(frontier);
        if only_one {
            break;
        }
    }
    result
}

#[time]
async fn update_qualifiers(block: &Block, cache: &mut AddBlockCache) {
    let config = block.config();

    let mut solutions_by_challenge = HashMap::<String, Vec<(&BenchmarkSettings, &u32)>>::new();
    for (settings, num_solutions) in cache.active_solutions.values() {
        solutions_by_challenge
            .entry(settings.challenge_id.clone())
            .or_default()
            .push((settings, num_solutions));
    }

    let mut max_qualifiers_by_player = HashMap::<String, u32>::new();
    for challenge in cache.active_challenges.values_mut() {
        let block_data = challenge.block_data.as_mut().unwrap();
        block_data.num_qualifiers = Some(0);
        block_data.qualifier_difficulties = Some(HashSet::new());
    }
    for algorithm in cache.active_algorithms.values_mut() {
        let block_data = algorithm.block_data.as_mut().unwrap();
        block_data.num_qualifiers_by_player = Some(HashMap::new());
    }
    for player in cache.active_players.values_mut() {
        let block_data = player.block_data.as_mut().unwrap();
        max_qualifiers_by_player.insert(player.id.clone(), *block_data.cutoff());
        block_data.num_qualifiers_by_challenge = Some(HashMap::new());
    }

    for (challenge_id, challenge) in cache.active_challenges.iter_mut() {
        if !solutions_by_challenge.contains_key(challenge_id) {
            continue;
        }
        let solutions = solutions_by_challenge.get_mut(challenge_id).unwrap();
        let points = solutions
            .iter()
            .map(|(settings, _)| settings.difficulty.clone())
            .collect::<Frontier>();
        let mut frontier_indexes = HashMap::<Point, usize>::new();
        for (frontier_index, frontier) in pareto_algorithm(points, false).into_iter().enumerate() {
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
        let challenge_data = challenge.block_data.as_mut().unwrap();
        for (settings, &num_solutions) in solutions.iter() {
            let BenchmarkSettings {
                player_id,
                algorithm_id,
                challenge_id,
                difficulty,
                ..
            } = settings;

            if curr_frontier_index != frontier_indexes[difficulty]
                && *challenge_data.num_qualifiers() > config.qualifiers.total_qualifiers_threshold
            {
                break;
            }
            let difficulty_parameters = &config.difficulty.parameters[challenge_id];
            let min_difficulty = difficulty_parameters.min_difficulty();
            let max_difficulty = difficulty_parameters.max_difficulty();
            if (0..difficulty.len())
                .into_iter()
                .any(|i| difficulty[i] < min_difficulty[i] || difficulty[i] > max_difficulty[i])
            {
                continue;
            }
            curr_frontier_index = frontier_indexes[difficulty];
            let player_data = cache
                .active_players
                .get_mut(player_id)
                .unwrap()
                .block_data
                .as_mut()
                .unwrap();
            let algorithm_data = cache
                .active_algorithms
                .get_mut(algorithm_id)
                .unwrap()
                .block_data
                .as_mut()
                .unwrap();

            let max_qualifiers = max_qualifiers_by_player.get(player_id).unwrap().clone();
            let num_qualifiers = num_solutions.min(max_qualifiers);
            max_qualifiers_by_player.insert(player_id.clone(), max_qualifiers - num_qualifiers);

            if num_qualifiers > 0 {
                *player_data
                    .num_qualifiers_by_challenge
                    .as_mut()
                    .unwrap()
                    .entry(challenge_id.clone())
                    .or_default() += num_qualifiers;
                *algorithm_data
                    .num_qualifiers_by_player
                    .as_mut()
                    .unwrap()
                    .entry(player_id.clone())
                    .or_default() += num_qualifiers;
                *challenge_data.num_qualifiers.as_mut().unwrap() += num_qualifiers;
            }
            challenge_data
                .qualifier_difficulties
                .as_mut()
                .unwrap()
                .insert(difficulty.clone());
        }
    }
}

#[time]
async fn update_frontiers(block: &Block, cache: &mut AddBlockCache) {
    let config = block.config();

    for challenge in cache.active_challenges.values_mut() {
        let block_data = challenge.block_data.as_mut().unwrap();

        let difficulty_parameters = &config.difficulty.parameters[&challenge.id];
        let min_difficulty = difficulty_parameters.min_difficulty();
        let max_difficulty = difficulty_parameters.max_difficulty();

        let points = block_data
            .qualifier_difficulties()
            .iter()
            .map(|d| d.iter().map(|x| -x).collect()) // mirror the points so easiest difficulties are first
            .collect::<Frontier>();
        let (base_frontier, scaling_factor, scaled_frontier) = if points.len() == 0 {
            let base_frontier: Frontier = vec![min_difficulty.clone()].into_iter().collect();
            let scaling_factor = 0.0;
            let scaled_frontier = base_frontier.clone();
            (base_frontier, scaling_factor, scaled_frontier)
        } else {
            let base_frontier = pareto_algorithm(points, true)
                .pop()
                .unwrap()
                .into_iter()
                .map(|d| d.into_iter().map(|x| -x).collect())
                .collect::<Frontier>() // mirror the points back;
                .extend(&min_difficulty, &max_difficulty);
            let scaling_factor = (*block_data.num_qualifiers() as f64
                / config.qualifiers.total_qualifiers_threshold as f64)
                .min(config.difficulty.max_scaling_factor);
            let scaled_frontier = base_frontier
                .scale(&min_difficulty, &max_difficulty, scaling_factor)
                .extend(&min_difficulty, &max_difficulty);
            (base_frontier, scaling_factor, scaled_frontier)
        };

        block_data.base_frontier = Some(base_frontier);
        block_data.scaled_frontier = Some(scaled_frontier);
        block_data.scaling_factor = Some(scaling_factor);
    }
}

#[time]
async fn update_influence(block: &Block, cache: &mut AddBlockCache) {
    let config = block.config();
    let active_player_ids = &block.data().active_player_ids;
    if active_player_ids.len() == 0 {
        return;
    }

    let mut num_qualifiers_by_challenge = HashMap::<String, u32>::new();
    for challenge in cache.active_challenges.values() {
        num_qualifiers_by_challenge.insert(
            challenge.id.clone(),
            *challenge.block_data().num_qualifiers(),
        );
    }

    let total_deposit = cache
        .active_players
        .values()
        .map(|p| p.block_data().deposit.clone().unwrap())
        .sum::<PreciseNumber>();

    let zero = PreciseNumber::from(0);
    let one = PreciseNumber::from(1);
    let imbalance_multiplier =
        PreciseNumber::from_f64(config.optimisable_proof_of_work.imbalance_multiplier);
    let num_challenges = PreciseNumber::from(cache.active_challenges.len());

    let mut weights = Vec::<PreciseNumber>::new();
    for player_id in active_player_ids.iter() {
        let data = cache
            .active_players
            .get_mut(player_id)
            .unwrap()
            .block_data
            .as_mut()
            .unwrap();

        let mut percent_qualifiers = Vec::<PreciseNumber>::new();
        for challenge_id in cache.active_challenges.keys() {
            let num_qualifiers = num_qualifiers_by_challenge[challenge_id];
            let num_qualifiers_by_player = *data
                .num_qualifiers_by_challenge()
                .get(challenge_id)
                .unwrap_or(&0);

            percent_qualifiers.push(if num_qualifiers_by_player == 0 {
                PreciseNumber::from(0)
            } else {
                PreciseNumber::from(num_qualifiers_by_player) / PreciseNumber::from(num_qualifiers)
            });
        }
        let OptimisableProofOfWorkConfig {
            avg_percent_qualifiers_multiplier,
            enable_proof_of_deposit,
            ..
        } = &config.optimisable_proof_of_work;
        if enable_proof_of_deposit.is_some_and(|x| x) {
            let max_percent_rolling_deposit =
                PreciseNumber::from_f64(avg_percent_qualifiers_multiplier.clone().unwrap())
                    * percent_qualifiers.arithmetic_mean();
            let percent_rolling_deposit = if total_deposit == zero {
                zero.clone()
            } else {
                data.deposit.clone().unwrap() / total_deposit
            };
            let qualifying_percent_rolling_deposit =
                if percent_rolling_deposit > max_percent_rolling_deposit {
                    max_percent_rolling_deposit.clone()
                } else {
                    percent_rolling_deposit
                };
            percent_qualifiers.push(qualifying_percent_rolling_deposit.clone());
            data.qualifying_percent_rolling_deposit = Some(qualifying_percent_rolling_deposit);
        }

        let mean = percent_qualifiers.arithmetic_mean();
        let variance = percent_qualifiers.variance();
        let cv_sqr = if mean == zero {
            zero.clone()
        } else {
            variance / (mean * mean)
        };

        let imbalance = cv_sqr / (num_challenges - one);
        let imbalance_penalty =
            one - PreciseNumber::approx_inv_exp(imbalance_multiplier * imbalance);

        weights.push(mean * (one - imbalance_penalty));

        data.imbalance = Some(imbalance);
        data.imbalance_penalty = Some(imbalance_penalty);
    }

    let influences = weights.normalise();
    for (player_id, &influence) in active_player_ids.iter().zip(influences.iter()) {
        let data = cache
            .active_players
            .get_mut(player_id)
            .unwrap()
            .block_data
            .as_mut()
            .unwrap();
        data.influence = Some(influence);
    }
}
