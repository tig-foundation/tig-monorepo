const queries = {
    block: `
    INSERT INTO block (
      id, 
      datetime_added, 
      prev_block_id, 
      height, 
      round, 
      config, 
      eth_block_num
    ) VALUES ($1, $2, $3, $4, $5, $6, $7)
    `,
    block_data: `
    INSERT INTO block_data (
      block_id, 
      mempool_algorithm_ids, 
      mempool_benchmark_ids, 
      mempool_fraud_ids, 
      mempool_proof_ids, 
      mempool_wasm_ids, 
      active_algorithm_ids, 
      active_benchmark_ids, 
      active_challenge_ids,
      active_player_ids,
      mempool_challenge_ids
    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
    `,
    benchmark: `
      INSERT INTO benchmark (
        id, 
        datetime_added, 
        player_id, 
        block_id, 
        challenge_id, 
        algorithm_id, 
        difficulty, 
        block_started, 
        num_solutions
      ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
    `,
    benchmark_state: `
      INSERT INTO benchmark_state (
        benchmark_id, 
        block_confirmed, 
        sampled_nonces
      ) VALUES ($1, $2, $3)
    `,
    benchmark_data: `
      INSERT INTO benchmark_data (
        benchmark_id, 
        solutions_meta_data,
        solution_data
      ) VALUES ($1, $2, $3)
    `,
    proof: `
      INSERT INTO proof (
        benchmark_id, 
        datetime_added
      ) VALUES ($1, $2)
    `,
    proof_state: `
      INSERT INTO proof_state (
        benchmark_id, 
        block_confirmed, 
        submission_delay
      ) VALUES ($1, $2, $3)
    `,
    proof_data: `
      INSERT INTO proof_data (
        benchmark_id, 
        solutions_data
      ) VALUES ($1, $2)
    `,
    fraud: `
      INSERT INTO fraud (
        benchmark_id, 
        datetime_added
      ) VALUES ($1, $2)
    `,
    fraud_state: `
      INSERT INTO fraud_state (
        benchmark_id, 
        block_confirmed
      ) VALUES ($1, $2)
    `,
    fraud_data: `
      INSERT INTO fraud_data (
        benchmark_id, 
        allegation
      ) VALUES ($1, $2)
    `,
    algorithm: `
      INSERT INTO algorithm (
        id,
        datetime_added,
        name,
        player_id,
        challenge_id,
        tx_hash
      ) VALUES ($1, $2, $3, $4, $5, $6)
    `,
    algorithm_state: `
      INSERT INTO algorithm_state (
        algorithm_id, 
        block_confirmed,
        round_submitted,
        round_pushed,
        round_merged,
        banned
      ) VALUES ($1, $2, $3, $4, $5, $6)
      ON CONFLICT (algorithm_id)
      DO UPDATE SET
        block_confirmed = EXCLUDED.block_confirmed,
        round_submitted = EXCLUDED.round_submitted,
        round_pushed = EXCLUDED.round_pushed,
        round_merged = EXCLUDED.round_merged,
        banned = EXCLUDED.banned;
    `,
    algorithm_data: `
      INSERT INTO algorithm_data (
        algorithm_id, 
        code
      ) VALUES ($1, $2)
    `,
    wasm: `
      INSERT INTO wasm (
        algorithm_id, 
        datetime_added,
        download_url,
        checksum,
        compile_success
      ) VALUES ($1, $2, $3, $4, $5)
    `,
    wasm_state: `
      INSERT INTO wasm_state (
        algorithm_id, 
        block_confirmed
      ) VALUES ($1, $2)
    `,
    wasm_data: `
      INSERT INTO wasm_data (
        algorithm_id, 
        wasm_blob
      ) VALUES ($1, $2)
    `,
    challenge: `
      INSERT INTO challenge (
        id, 
        datetime_added, 
        name
      ) VALUES ($1, $2, $3)
    `,
    challenge_state: `
      INSERT INTO challenge_state (
        challenge_id, 
        block_confirmed,
        round_active
      ) VALUES ($1, $2, $3)
      ON CONFLICT (challenge_id) 
      DO UPDATE SET 
        block_confirmed = EXCLUDED.block_confirmed,
        round_active = EXCLUDED.round_active;

    `,
    challenge_block_data: `
      INSERT INTO challenge_block_data (
        challenge_id, 
        block_id,
        solution_signature_threshold,
        num_qualifiers,
        qualifier_difficulties,
        base_frontier,
        scaled_frontier,
        scaling_factor,
        cutoff_frontier
      ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
    `,
    player_block_data: `
      INSERT INTO player_block_data (
        block_id, 
        player_id,
        num_qualifiers_by_challenge,
        cutoff,
        imbalance,
        imbalance_penalty,
        influence,
        reward,
        round_earnings,
        deposit,
        rolling_deposit
      ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
    `,
    algorithm_block_data: `
      INSERT INTO algorithm_block_data (
        block_id,
        algorithm_id,
        num_qualifiers_by_player,
        adoption,
        merge_points,
        reward,
        round_earnings
      ) VALUES ($1, $2, $3, $4, $5, $6, $7)
    `,
  };
  
  module.exports = queries;