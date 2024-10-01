mod add_block;
pub mod context;
mod error;
mod submit_algorithm;
mod submit_benchmark;
mod submit_precommit;
mod submit_proof;
mod submit_topup;
mod verify_proof;
use context::*;
pub use error::*;
use std::collections::HashSet;
use tig_structs::core::*;

pub struct Protocol<T: Context> {
    pub ctx: T,
}

impl<'a, T: Context> Protocol<T> {
    pub fn new(ctx: T) -> Self {
        Self { ctx }
    }

    pub async fn submit_algorithm(
        &self,
        player: &Player,
        details: AlgorithmDetails,
        code: String,
    ) -> ProtocolResult<String> {
        submit_algorithm::execute(&self.ctx, player, details, code).await
    }

    pub async fn submit_precommit(
        &self,
        player: &Player,
        settings: BenchmarkSettings,
        num_nonces: u32,
    ) -> ProtocolResult<String> {
        submit_precommit::execute(&self.ctx, player, settings, num_nonces).await
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
}
