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

/// A trait for implementing solvers for specific challenges.
/// 
/// This trait provides the necessary methods to manage different algorithms, generate
/// problem instances, solve them, and verify solutions. The trait is generic over
/// a constant `N`, representing a size of the Difficulty dimensions for the challenge.
pub trait SolverTrait<const N: usize> {
    type S: SolutionTrait;
    type D: DifficultyTrait<N>;
    type C: ChallengeTrait<Self::S, Self::D, N>;

    /// Checks if the solver contains the algorithm with identifier 'id'.
    fn algorithm_exists(id: &str) -> bool;

    /// Checks if the solver contains the algorithm with name 'name'.
    fn algorithm_exists_name(name: &str) -> bool;

    /// Returns the solve_challenge function for an algorithm matching the 'id'.
    /// 
    /// # Parameters
    /// - id: Algorithm id; Example format is c002_a035, the 35th algorithm for challenge 2. 
    fn get_algorithm(id: &str) -> Option<fn(&Self::C)-> anyhow::Result<Option<Self::S>>>;

    /// Solves a challenge instance 'c' with algorithm 'id' if there exists an algorithm for 'id'
    /// 
    /// # Parameters
    /// - id: Algorithm id; Example format is c002_a035, the 35th algorithm for challenge 2. 
    /// - c: Challenge instance of type 'C'. 
    /// 
    /// # Returns
    /// An `anyhow::Result` containing an `Option` with the solution of type 'S' if the algorithm
    /// exists and succeeds, or an error if the algorithm does not exist or fails.
    fn solve_challenge_with_algorithm(id: &str, c: &Self::C) -> anyhow::Result<Option<Self::S>> {
        match Self::get_algorithm(id) {
            Some(algo) => algo(c),
            None => Err(anyhow::anyhow!("Algorithm does not exist")),
        }
    }

    /// Generates a challenge instance based on provided seeds and difficulty settings.
    ///
    /// # Parameters
    /// - `seeds`: An array of 8 `u64` values used to initialize the random generator for the instance.
    /// - `difficulty`: A vector of `i32` values representing the difficulty settings for the challenge.
    ///
    /// # Returns
    /// An `anyhow::Result` containing the generated challenge instance (`Self::C`), or an error if generation fails.
    fn generate_instance(seeds: [u64; 8], difficulty: &Vec<i32>) -> anyhow::Result<Self::C> {
        Self::C::generate_instance_from_vec(seeds, difficulty)
    }

    /// Verifies the provided solution against the given challenge.
    ///
    /// # Parameters
    /// - `c`: A reference to the challenge instance.
    /// - `s`: A reference to the solution that needs to be verified.
    ///
    /// # Returns
    /// An `anyhow::Result` indicating success if the solution is correct, or an error if the verification fails.
    fn verify_solution(c: &Self::C, s: &Self::S) -> anyhow::Result<()> {
        Self::C::verify_solution(c, s)
    }
}