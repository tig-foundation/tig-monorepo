pub mod knapsack;
pub use knapsack as c003;
pub mod satisfiability;
pub use satisfiability as c001;
pub mod vector_search;
pub use vector_search as c004;
pub mod vehicle_routing;
pub use vehicle_routing as c002;

pub use tig_challenges::CudaKernel;

use tig_challenges::{ChallengeTrait, DifficultyTrait, SolutionTrait};
pub trait SolverTrait<const N: usize> {
    type S: SolutionTrait;
    type D: DifficultyTrait<N>;
    type C: ChallengeTrait<Self::S, Self::D, N>;

    fn algorithm_exists(id: &str) -> bool;

    fn get_algorithm(id: &str) -> Option<fn(&Self::C)-> anyhow::Result<Option<Self::S>>>;

    fn solve_challenge_with_algorithm(id: &str, c: &Self::C) -> anyhow::Result<Option<Self::S>> {
        match Self::get_algorithm(id) {
            Some(algo) => algo(c),
            None => Err(anyhow::anyhow!("Algorithm does not exist")),
        }
    }

    fn generate_instance(seeds: [u64; 8], difficulty: &Vec<i32>) -> anyhow::Result<Self::C> {
        Self::C::generate_instance_from_vec(seeds, difficulty)
    }

    fn verify_solution(c: &Self::C, s: &Self::S) -> anyhow::Result<()> {
        Self::C::verify_solution(c, s)
    }
}