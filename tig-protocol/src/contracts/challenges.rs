#[time]
async fn update_solution_signature_thresholds(block: &Block, cache: &mut AddBlockCache) {
    let config = block.config();

    let confirmed_proof_ids = &block.data().confirmed_proof_ids;
    let mut num_solutions_by_player_by_challenge = HashMap::<String, HashMap<String, u32>>::new();
    let mut new_solutions_by_player_by_challenge = HashMap::<String, HashMap<String, u32>>::new();
    for (benchmark_id, (settings, num_solutions)) in cache.active_solutions.iter() {
        *num_solutions_by_player_by_challenge
            .entry(settings.player_id.clone())
            .or_default()
            .entry(settings.challenge_id.clone())
            .or_default() += *num_solutions;
        if confirmed_proof_ids.contains(benchmark_id) {
            *new_solutions_by_player_by_challenge
                .entry(settings.player_id.clone())
                .or_default()
                .entry(settings.challenge_id.clone())
                .or_default() += *num_solutions;
        }
    }

    let mut solutions_rate_by_challenge = HashMap::<String, u32>::new();
    for (player_id, new_solutions_by_challenge) in new_solutions_by_player_by_challenge.iter() {
        let cutoff = *cache
            .active_players
            .get(player_id)
            .unwrap()
            .block_data()
            .cutoff();
        for (challenge_id, new_solutions) in new_solutions_by_challenge.iter() {
            let num_solutions =
                num_solutions_by_player_by_challenge[player_id][challenge_id].clone();
            *solutions_rate_by_challenge
                .entry(challenge_id.clone())
                .or_default() +=
                new_solutions.saturating_sub(num_solutions - cutoff.min(num_solutions));
        }
    }

    for challenge in cache.active_challenges.values_mut() {
        let max_threshold = u32::MAX as f64;
        let current_threshold = match &cache.prev_challenges.get(&challenge.id).unwrap().block_data
        {
            Some(data) => *data.solution_signature_threshold() as f64,
            None => max_threshold,
        };
        let current_rate = *solutions_rate_by_challenge.get(&challenge.id).unwrap_or(&0) as f64;

        let equilibrium_rate = config.qualifiers.total_qualifiers_threshold as f64
            / config.benchmark_submissions.lifespan_period as f64;
        let target_rate = config.solution_signature.equilibrium_rate_multiplier * equilibrium_rate;
        let target_threshold = if current_rate == 0.0 {
            max_threshold
        } else {
            (current_threshold * target_rate / current_rate).clamp(0.0, max_threshold)
        };

        let threshold_decay = config.solution_signature.threshold_decay.unwrap_or(0.99);
        let block_data = challenge.block_data.as_mut().unwrap();
        block_data.solution_signature_threshold = Some(
            (current_threshold * threshold_decay + target_threshold * (1.0 - threshold_decay))
                .clamp(0.0, max_threshold) as u32,
        );
    }
}

#[time]
async fn update_fees(block: &Block, cache: &mut AddBlockCache) {
    let config = block.config();
    let PrecommitSubmissionsConfig {
        min_base_fee,
        min_per_nonce_fee,
        target_num_precommits,
        max_fee_percentage_delta,
        ..
    } = config.precommit_submissions();
    let num_precommits_by_challenge = cache.mempool_precommits.iter().fold(
        HashMap::<String, u32>::new(),
        |mut map, precommit| {
            *map.entry(precommit.settings.challenge_id.clone())
                .or_default() += 1;
            map
        },
    );
    let target_num_precommits = PreciseNumber::from(*target_num_precommits);
    let max_fee_percent_delta = PreciseNumber::from_f64(*max_fee_percentage_delta);
    let one = PreciseNumber::from(1);
    let zero = PreciseNumber::from(0);
    for challenge in cache.active_challenges.values_mut() {
        let num_precommits = PreciseNumber::from(
            num_precommits_by_challenge
                .get(&challenge.id)
                .unwrap_or(&0)
                .clone(),
        );
        let mut percent_delta = num_precommits / target_num_precommits;
        if num_precommits >= target_num_precommits {
            percent_delta = percent_delta - one;
        } else {
            percent_delta = one - percent_delta;
        }
        if percent_delta > max_fee_percent_delta {
            percent_delta = max_fee_percent_delta;
        }
        let current_base_fee =
            match &cache.prev_challenges.get(&challenge.id).unwrap().block_data {
                Some(data) => data.base_fee.as_ref().unwrap_or(&zero),
                None => &zero,
            }
            .clone();
        let mut base_fee = if num_precommits >= target_num_precommits {
            current_base_fee * (one + percent_delta)
        } else {
            current_base_fee * (one - percent_delta)
        };
        if base_fee < *min_base_fee {
            base_fee = *min_base_fee;
        }
        let block_data = challenge.block_data.as_mut().unwrap();
        block_data.base_fee = Some(base_fee);
        block_data.per_nonce_fee = Some(min_per_nonce_fee.clone());
    }
}
