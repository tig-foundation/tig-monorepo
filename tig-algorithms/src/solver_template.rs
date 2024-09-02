use crate::SolverTrait;

struct Solver;

impl SolverTrait<2> for Solver {
    type C = tig_challenges::{CHALLENGE}::Challenge;
    type S = tig_challenges::{CHALLENGE}::Solution;
    type D = tig_challenges::{CHALLENGE}::Difficulty;

    fn algorithm_exists(id: &str) -> bool {
        match id {
            {EXISTING_ALGOS}
            _ => false
        }
    }

    fn get_algorithm(id: &str) -> Option<fn(&Self::C)-> anyhow::Result<Option<Self::S>>> {
        match id {
        {ALGORITHMS}
            _ => None
        }
    }

}