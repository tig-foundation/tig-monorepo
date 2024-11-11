use {
    crate::{
        ctx::Context,
        err::{ContractResult, ProtocolError},
        block::AddBlockCache,
    },
    logging_timer::time,
    std::{
        marker::PhantomData,
        sync::{Arc, RwLock},
        collections::HashMap,
    },
    tig_structs::{
        config::*,
        core::*,
    },
    rayon::prelude::*,
};

pub struct ChallengeContract<T: Context>
{
    phantom: PhantomData<T>,
}

impl<T: Context> ChallengeContract<T> 
{
    pub fn new() -> Self 
    {
        return Self { phantom: PhantomData };
    }

    pub fn update(&self, ctx: &T, cache: &AddBlockCache, block: &Block)
    {
        // update solution signature thresholds
        {
            let config              = block.config();
            let confirmed_proof_ids = &block.data().confirmed_proof_ids;

            let num_solutions_by_player_by_challenge = Arc::new(RwLock::new(HashMap::<String, HashMap<String, u32>>::new()));
            let new_solutions_by_player_by_challenge = Arc::new(RwLock::new(HashMap::<String, HashMap<String, u32>>::new()));

            cache.active_solutions.read().unwrap().par_iter().for_each(|(benchmark_id, (settings, num_solutions))|
            {
                {
                    let mut map = num_solutions_by_player_by_challenge.write().unwrap();
                    *map.entry(settings.player_id.clone())
                        .or_default()
                        .entry(settings.challenge_id.clone())
                        .or_default() += *num_solutions;
                }

                if confirmed_proof_ids.contains(benchmark_id) {
                    let mut map = new_solutions_by_player_by_challenge.write().unwrap();
                    *map.entry(settings.player_id.clone())
                        .or_default()
                        .entry(settings.challenge_id.clone())
                        .or_default() += *num_solutions;
                }
            });

            let num_solutions_by_player_by_challenge = Arc::try_unwrap(num_solutions_by_player_by_challenge)
                .unwrap()
                .into_inner()
                .unwrap();
            let new_solutions_by_player_by_challenge = Arc::try_unwrap(new_solutions_by_player_by_challenge)
                .unwrap()
                .into_inner()
                .unwrap();

            let solutions_rate_by_challenge = Arc::new(RwLock::new(HashMap::<String, u32>::new()));

            new_solutions_by_player_by_challenge.par_iter().for_each(|(player_id, new_solutions_by_challenge)| 
            {
                let cutoff = *cache
                    .active_players.read().unwrap()
                    .get(player_id).unwrap()
                    .block_data.as_ref().unwrap()
                    .cutoff();

                new_solutions_by_challenge.par_iter().for_each(|(challenge_id, new_solutions)| 
                {
                    let num_solutions   = num_solutions_by_player_by_challenge[player_id][challenge_id].clone();
                    let delta           = new_solutions.saturating_sub(num_solutions - cutoff.min(num_solutions));
                    let mut map         = solutions_rate_by_challenge.write().unwrap();

                    *map.entry(challenge_id.clone()).or_default() += delta;
                });
            });

            let solutions_rate_by_challenge = Arc::try_unwrap(solutions_rate_by_challenge)
                .unwrap()
                .into_inner()
                .unwrap();

            cache.active_challenges.write().unwrap().par_iter().for_each(|(challenge_id, challenge)|
            {
                let max_threshold       = u32::MAX as f64;
                let current_threshold   = match cache.prev_challenges.read().unwrap().get(&challenge.id)
                {
                    Some(data) => *data.block_data.as_ref().unwrap().solution_signature_threshold() as f64,
                    None => max_threshold,
                };

                let current_rate        = *solutions_rate_by_challenge.get(&challenge.id).unwrap_or(&0) as f64;
                let equilibrium_rate    = config.qualifiers.total_qualifiers_threshold as f64
                    / config.benchmark_submissions.lifespan_period as f64;
                    
                let target_rate         = config.solution_signature.equilibrium_rate_multiplier * equilibrium_rate;
                let target_threshold    = if current_rate == 0.0 
                {
                    max_threshold
                } 
                else 
                {
                    (current_threshold * target_rate / current_rate).clamp(0.0, max_threshold)
                };

                let threshold_decay = config.solution_signature.threshold_decay.unwrap_or(0.99);
                //let block_data      = challenge.block_data.as_mut().unwrap();
                /*
                block_data.solution_signature_threshold = Some(
                    (current_threshold * threshold_decay + target_threshold * (1.0 - threshold_decay))
                        .clamp(0.0, max_threshold) as u32,
                );
                */

                cache
                    .commit_challenges_solution_sig_thresholds.write().unwrap()
                    .insert(
                        challenge_id.clone(), 
                        (current_threshold * threshold_decay + target_threshold * (1.0 - threshold_decay)).clamp(0.0, max_threshold) as u32
                    );
            });
        }

        // update fees
        {
            let config = block.config();
            let PrecommitSubmissionsConfig {
                min_base_fee,
                min_per_nonce_fee,
                target_num_precommits,
                max_fee_percentage_delta,
                ..
            } = config.precommit_submissions();

            let num_precommits_by_challenge = cache.mempool_precommits.read().unwrap().iter().fold(
                HashMap::<String, u32>::new(),
                |mut map, precommit| {
                    *map.entry(precommit.settings.challenge_id.clone())
                        .or_default() += 1;
                    map
                },
            );
            let target_num_precommits   = PreciseNumber::from(*target_num_precommits);
            let max_fee_percent_delta   = PreciseNumber::from_f64(*max_fee_percentage_delta);
            let one                     = PreciseNumber::from(1);
            let zero                    = PreciseNumber::from(0);

            cache.active_challenges.read().unwrap().par_iter().for_each(|(challenge_id, challenge)|
            {
                let num_precommits = PreciseNumber::from(
                    num_precommits_by_challenge
                        .get(&challenge.id)
                        .unwrap_or(&0)
                        .clone(),
                );

                let mut percent_delta = num_precommits / target_num_precommits;
                if num_precommits >= target_num_precommits 
                {
                    percent_delta = percent_delta - one;
                } 
                else 
                {
                    percent_delta = one - percent_delta;
                }

                if percent_delta > max_fee_percent_delta 
                {
                    percent_delta = max_fee_percent_delta;
                }

                let current_base_fee =
                    match cache.prev_challenges.read().unwrap().get(&challenge.id)
                    {
                        Some(data)  => data.block_data.as_ref().unwrap().base_fee.as_ref().unwrap_or(&zero),
                        None        => &zero,
                    }
                    .clone();
                let mut base_fee = if num_precommits >= target_num_precommits 
                {
                    current_base_fee * (one + percent_delta)
                } 
                else 
                {
                    current_base_fee * (one - percent_delta)
                };

                if base_fee < *min_base_fee 
                {
                    base_fee = *min_base_fee;
                }

                /*
                let block_data = challenge.block_data.as_mut().unwrap();
                block_data.base_fee = Some(base_fee);
                block_data.per_nonce_fee = Some(min_per_nonce_fee.clone());*/

                cache
                    .commit_challenges_fees.write().unwrap()
                    .insert(challenge.id.clone(), (base_fee, min_per_nonce_fee.clone()));
            });
        }
    }
}
