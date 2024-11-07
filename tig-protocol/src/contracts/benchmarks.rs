use {
    crate::{ctx::Context, err::ContractResult},
    logging_timer::time,
    std::{collections::HashSet, marker::PhantomData},
    tig_structs::core::*,
    tig_utils::*,
    tokio::sync::RwLock,
};

pub struct BenchmarkContract {}

impl BenchmarkContract {
    // pub fn new() -> Self {
    //     return Self {
    //         phantom: PhantomData,
    //     };
    // }

    pub async fn submit_precommit(
        &self,
        ctx: &RwLock<Context>,
        player: Player,
        settings: BenchmarkSettings,
        num_nonces: u32,
    ) -> ContractResult<String> {
        //verify that the player owns the benchmark
        if player.id != settings.player_id {
            return Err(format!("Invalid submitting player: {}", player.id));
        }

        //verify that the num nonces is greater than 0
        if num_nonces == 0 {
            return Err("Invalid num nonces".to_string());
        }

        let ctx = ctx.write().await;
        //make sure that the submission delay is within the lifespan period
        let block_id = &settings.block_id;
        let next_block_id = ctx.blocks.get_next_id();
        let block_details = ctx
            .blocks
            .get_details(block_id)
            .ok_or_else(|| &format!("Expecting benchmark block to exist: {}", block_id))?;
        let next_block_details = ctx.blocks.get_details(&next_block_id).unwrap();

        // verify precommit has sufficient lifespan
        let config = ctx.blocks.get_config(block_id).unwrap();
        let submission_delay = next_block_details.height - block_details.height;
        if (submission_delay as f64
            * (config.benchmark_submissions.submission_delay_multiplier + 1.0)) as u32
            >= config.benchmark_submissions.lifespan_period
        {
            return Err(format!("Insufficient lifespan"));
        }

        // verify challenge and algorithm is active
        let algorithm_id = &settings.algorithm_id;
        let challenge_id = &settings.challenge_id;
        let block_data = ctx.blocks.get_data(block_id).unwrap();
        if !block_data.active_challenge_ids.contains(challenge_id) {
            return Err(format!("Invalid challenge: {}", challenge_id));
        }
        if !block_data.active_algorithm_ids.contains(algorithm_id) {
            return Err(format!("Invalid algorithm: {}", algorithm_id));
        }

        //verify that the algorithm is not banned
        if !ctx
            .algorithms
            .get_state(&settings.algorithm_id)
            .unwrap()
            .banned
        {
            return Err(format!("Banned algorithm: {}", algorithm_id));
        }

        // verify that benchmark settings are unique
        let benchmark_id = ctx.precommits.calc_benchmark_id(settings);
        if ctx.precommits.get(&benchmark_id).is_some() {
            return Err(format!("Duplicate benchmark settings"));
        }

        //verify benchmark difficulty
        let difficulty = &settings.difficulty;
        let difficulty_parameters = &config.difficulty.parameters[challenge_id];
        if difficulty.len() != difficulty_parameters.len()
            || difficulty
                .iter()
                .zip(difficulty_parameters.iter())
                .any(|(d, p)| *d < p.min_value || *d > p.max_value)
        {
            return Err(format!("Invalid difficulty: {:?}", difficulty));
        }

        let challenge_data = ctx.challenges.get_data(challenge_id).unwrap();
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
                return Err(format!(
                    "Difficulty above hardest frontier: {:?}",
                    difficulty
                ));
            }
            PointCompareFrontiers::Below => {
                return Err(format!(
                    "Difficulty below easiest frontier: {:?}",
                    difficulty
                ));
            }
            PointCompareFrontiers::Within => {}
        }

        //verify sufficient balance
        let fee_to_pay = challenge_data.base_fee()
            + challenge_data.per_nonce_fee() * PreciseNumber::from(num_nonces);

        let player_state = ctx.players.get_mut_state(&player.id).unwrap();
        if player_state.available_fee_balance < fee_to_pay {
            return Err(format!("Insufficient fee balance"));
        }
        player_state.available_fee_balance -= fee_to_pay;
        player_state.total_fees_paid += fee_to_pay;

        // confirm precommit
        ctx.precommits.add(
            benchmark_id,
            settings,
            &PrecommitDetails {
                block_started: block_details.height,
                num_nonces: Some(num_nonces),
                fee_paid: Some(fee_to_pay),
            },
            PrecommitState {
                block_confirmed: Some(next_block_details.height),
                rand_hash: Some(next_block_id),
            },
        );

        return Ok(benchmark_id);
    }

    pub async fn submit_benchmark(
        &self,
        ctx: &RwLock<Context>,
        player: Player,
        benchmark_id: String,
        merkle_root: MerkleHash,
        solution_nonces: HashSet<u64>,
    ) -> ContractResult<()> {
        let ctx = ctx.write().await;

        //verify that the benchmark is not already submitted
        if ctx.benchmarks.get(&benchmark_id).is_some() {
            return Err(format!("Duplicate benchmark: {}", benchmark_id));
        }

        //fetch the precommit
        let block_id = ctx.blocks.get_current_id();
        let block_details = ctx.blocks.get_details(&block_id).unwrap();
        let precommit_state = ctx
            .precommits
            .get_state(&benchmark_id)
            .ok_or_else(|| format!("Invalid precommit: {}", benchmark_id))?;
        if precommit_state.block_confirmed.unwrap() > block_details.height {
            return Err(format!("Unconfirmed precommit: {}", benchmark_id));
        }

        //verify that the player owns the precommit
        if player.id != ctx.precommits.get_settings(benchmark_id).unwrap().player_id {
            return Err(format!("Invalid submitting player: {}", player.id));
        }

        //verify that the solution nonces are valid
        let num_nonces = *ctx
            .precommits
            .get_details(&benchmark_id)
            .unwrap()
            .num_nonces
            .as_ref()
            .unwrap() as u64;
        for n in solution_nonces.iter() {
            if *n >= num_nonces {
                return Err(format!("Invalid solution nonce: {}", n));
            }
        }

        let next_block_id = ctx.blocks.get_next_id();
        let next_config = ctx.blocks.get_config(&next_block_id).unwrap();
        let next_block_details = ctx.blocks.get_details(&next_block_id).unwrap();
        let seed = u64s_from_str(&format!("{:?}|{:?}", next_block_id, benchmark_id))[0];
        let mut rng = StdRng::seed_from_u64(seed);
        let mut sampled_nonces = HashSet::new();
        let mut solution_nonces_vec = solution_nonces.iter().cloned().collect::<Vec<u64>>();
        if solution_nonces.len() > 0 {
            solution_nonces.shuffle(&mut rng);
            for nonce in solution_nonces
                .iter()
                .take(next_config.benchmark_submissions.max_samples)
            {
                sampled_nonces.insert(*nonce);
            }
        }
        let mut non_solution_nonces: Vec<u64> = (0..num_nonces as u64)
            .filter(|n| !solution_nonces.contains(n))
            .collect();
        if non_solution_nonces.len() > 0 {
            non_solution_nonces.shuffle(&mut rng);
            for nonce in non_solution_nonces
                .iter()
                .take(next_config.benchmark_submissions.max_samples)
            {
                sampled_nonces.insert(*nonce);
            }
        }

        ctx.benchmarks.add(
            benchmark_id,
            BenchmarkDetails {
                merkle_root: Some(merkle_root.clone()),
                num_solutions: solution_nonces.len() as u32,
            },
            BenchmarkState {
                sampled_nonces: Some(sampled_nonces),
                block_confirmed: Some(next_block_details.height),
            },
            solution_nonces,
        );

        return Ok(());
    }

    pub async fn submit_proof(
        &self,
        ctx: &RwLock<Context>,
        player: Player,
        benchmark_id: String,
        merkle_proofs: Vec<MerkleProof>,
    ) -> ContractResult<Result<(), String>> {
        let ctx = ctx.write().await;

        //verify that the proof is not already submitted
        if ctx.proofs.get(&benchmark_id).is_some() {
            return Err(format!("Duplicate proof: {}", benchmark_id));
        }

        //fetch the benchmark
        let block_id = ctx.blocks.get_current_id();
        let block_details = ctx.blocks.get_details(&block_id).unwrap();
        let benchmark_state = ctx
            .benchmarks
            .get_state(&benchmark_id)
            .ok_or_else(|| format!("Invalid benchmark: {}", benchmark_id))?;
        if benchmark_state.block_confirmed.unwrap() > block_details.height {
            return Err(format!("Unconfirmed benchmark: {}", benchmark_id));
        }

        //verify that the player owns the benchmark
        let settings = ctx.precommits.get_settings(&benchmark_id).unwrap();
        if player.id != settings.player_id {
            return Err(format!("Invalid submitting player: {}", player.id));
        }

        //verify the sampled nonces
        let sampled_nonces = benchmark_state.sampled_nonces();
        let proof_nonces: HashSet<u64> = merkle_proofs.iter().map(|p| p.leaf.nonce).collect();

        if *sampled_nonces != proof_nonces || sampled_nonces.len() != merkle_proofs.len() {
            return Err(format!("Invalid proof nonces"));
        }

        //verify the merkle proofs
        let precommit_details = ctx.precommits.get_details(&benchmark_id).unwrap();
        let benchmark_details = ctx.benchmarks.get_details(&benchmark_id).unwrap();
        let max_branch_len =
            (64 - (*precommit_details.num_nonces.as_ref().unwrap() - 1).leading_zeros()) as usize;
        let expected_merkle_root = benchmark_details.merkle_root.as_ref().unwrap();

        let mut is_fraudulent = None;
        for merkle_proof in merkle_proofs.iter() {
            let branch = merkle_proof.branch.as_ref().unwrap();
            if branch.0.len() > max_branch_len
                || branch.0.iter().any(|(d, _)| *d as usize > max_branch_len)
            {
                is_fraudulent = Some(format!("Invalid merkle proof: {}", merkle_proof.leaf.nonce));

                break;
            }

            let output_meta_data = OutputMetaData::from(merkle_proof.leaf.clone());
            let hash = MerkleHash::from(output_meta_data);
            let result = merkle_proof
                .branch
                .as_ref()
                .unwrap()
                .calc_merkle_root(&hash, merkle_proof.leaf.nonce as usize);

            if !result.is_ok_and(|actual_merkle_root| actual_merkle_root == *expected_merkle_root) {
                is_fraudulent = Some(format!("Invalid merkle proof: {}", merkle_proof.leaf.nonce));

                break;
            }
        }

        //verify the solutions
        if !is_fraudulent.is_some() {
            for p in merkle_proofs.iter() {
                if ctx
                    .proofs
                    .verify_solution(&settings, p.leaf.nonce, &p.leaf.solution)
                    .await
                    .unwrap()
                    .is_err()
                {
                    is_fraudulent = Some(format!("Invalid solution: {}", p.leaf.nonce));

                    break;
                }
            }
        }

        //add the proof to the mempool
        ctx.proofs.add(
            benchmark_id,
            ProofState {
                block_confirmed: Some(next_block_details.height),
                submission_delay: Some(next_block_details.height - precommit_details.block_started),
            },
            merkle_proofs,
        );

        //add fraud to the mempool if the proof is fraudulent
        if is_fraudulent.is_some() {
            self.submit_fraud(benchmark_id, &is_fraudulent.clone().unwrap().to_string())
                .await;

            return Ok(Err(is_fraudulent.unwrap()));
        }

        return Ok(Ok(()));
    }

    async fn submit_fraud() {
        // FIXME
    }
}
