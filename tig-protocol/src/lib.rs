mod add_block;
pub mod context;
mod error;
mod submit_algorithm;
mod submit_benchmark;
mod submit_precommit;
mod submit_proof;
mod submit_topup;
mod verify_proof;

mod contract_benchmark;
mod contract_algorithms;
mod contract_challenges;

use context::*;
pub use error::*;
use std::collections::HashSet;
use tig_structs::core::*;

use contract_benchmark::BenchmarksContract;
use contract_algorithms::AlgorithmsContract;
use contract_challenges::ChallengesContract;

pub struct Protocol<T: Context> {
    pub ctx: T,
    pub benchmarks: BenchmarksContract<T>,
    pub algorithms: AlgorithmsContract<T>,
    pub challenges: ChallengesContract<T>,
}

impl<'a, T: Context> Protocol<T> {
    pub fn new(ctx: T) -> Self {
        Self { ctx, benchmarks: BenchmarksContract::new(), algorithms: AlgorithmsContract::new(), challenges: ChallengesContract::new() }
    }

    pub async fn submit_algorithm(
        &self,
        player: &Player,
        details: AlgorithmDetails,
        code: String,
    ) -> ProtocolResult<String> {
        submit_algorithm::execute(&self.ctx, player, details, code).await
    }

    pub async fn submit_benchmark(
        &self,
        player: &Player,
        benchmark_id: &String,
        merkle_root: MerkleHash,
        solution_nonces: HashSet<u64>,
    ) -> ProtocolResult<()> {
        submit_benchmark::execute(
            &self.ctx,
            player,
            benchmark_id,
            merkle_root,
            solution_nonces,
        )
        .await
    }

    pub async fn submit_proof(
        &self,
        player: &Player,
        benchmark_id: &String,
        merkle_proofs: Vec<MerkleProof>,
    ) -> ProtocolResult<Result<(), String>> {
        submit_proof::execute(&self.ctx, player, benchmark_id, merkle_proofs).await
    }

    pub async fn submit_topup(&self, player: &Player, tx_hash: String) -> ProtocolResult<()> {
        submit_topup::execute(&self.ctx, player, tx_hash).await
    }

    pub async fn verify_proof(&self, benchmark_id: &String) -> ProtocolResult<Result<(), String>> {
        verify_proof::execute(&self.ctx, benchmark_id).await
    }

    pub async fn add_block(&self) -> String {
        add_block::execute(&self.ctx).await
    }


    pub async fn submit_precommit(
        &self,
        player:                     &Player,
        settings:                   BenchmarkSettings,
        num_nonces:                 u32,
    )                                       -> ProtocolResult<String> 
    {
        //self.benchmarks.execute(player, &settings, num_nonces).await

        self.benchmarks.verify_player_owns_benchmark(player, &settings)?;
        submit_precommit::verify_num_nonces(num_nonces)?;
        
        let block                               = submit_precommit::get_block_by_id(&self.ctx, &settings.block_id).await?;
        submit_precommit::verify_sufficient_lifespan(&self.ctx, &block).await?;
        
        let challenge                           = self.challenges.get_challenge_by_id(&self.ctx, &settings.challenge_id, &block).await?;
        self.algorithms.verify_algorithm(&self.ctx, &settings.algorithm_id, &block).await?;
        self.benchmarks.verify_benchmark_settings_are_unique(&self.ctx, &settings).await?;
        self.benchmarks.verify_benchmark_difficulty(&settings.difficulty, &challenge, &block)?;

        let fee_paid                            = submit_precommit::get_fee_paid(&player, num_nonces, &challenge)?;
        let benchmark_id                        = self.ctx
        .add_precommit_to_mempool(
            settings,
            PrecommitDetails 
            {
                block_started                   : block.details.height,
                num_nonces                      : Some(num_nonces),
                fee_paid                        : Some(fee_paid),
            },
        )
        .await
        .unwrap_or_else(|e| panic!("add_precommit_to_mempool error: {:?}", e));

        return Ok(benchmark_id);
    }
}
