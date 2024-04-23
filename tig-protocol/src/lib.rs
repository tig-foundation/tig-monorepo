mod add_block;
pub mod context;
mod error;
mod submit_algorithm;
mod submit_benchmark;
mod submit_proof;
mod verify_proof;
use context::*;
pub use error::*;
use tig_structs::core::*;

pub struct Protocol<T: Context> {
    pub ctx: T,
}

impl<'a, T: Context> Protocol<T> {
    pub fn new(ctx: T) -> Self {
        Self { ctx }
    }

    pub async fn submit_algorithm(
        &mut self,
        player: &Player,
        details: &AlgorithmDetails,
        code: &String,
    ) -> ProtocolResult<String> {
        submit_algorithm::execute(&mut self.ctx, player, details, code).await
    }

    pub async fn submit_benchmark(
        &mut self,
        player: &Player,
        settings: &BenchmarkSettings,
        solutions_meta_data: &Vec<SolutionMetaData>,
        solution_data: &SolutionData,
    ) -> ProtocolResult<(String, Result<(), String>)> {
        submit_benchmark::execute(
            &mut self.ctx,
            player,
            settings,
            solutions_meta_data,
            solution_data,
        )
        .await
    }

    pub async fn submit_proof(
        &mut self,
        player: &Player,
        benchmark_id: &String,
        solutions_data: &Vec<SolutionData>,
    ) -> ProtocolResult<Result<(), String>> {
        submit_proof::execute(&mut self.ctx, player, benchmark_id, solutions_data).await
    }

    pub async fn verify_proof(
        &mut self,
        benchmark_id: &String,
    ) -> ProtocolResult<Result<(), String>> {
        verify_proof::execute(&mut self.ctx, benchmark_id).await
    }

    pub async fn add_block(&mut self) -> String {
        add_block::execute(&mut self.ctx).await
    }
}
