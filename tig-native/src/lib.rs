extern crate tig_challenges;
mod solve;

use {
    tig_challenges::knapsack::{Challenge as KnapsackChallenge, Solution as KnapsackSolution},
    tig_challenges::vector_search::{Challenge as VectorSearchChallenge, Solution as VectorSearchSolution},
    tig_challenges::satisfiability::{Challenge as SatisfiabilityChallenge, Solution as SatisfiabilitySolution},
    tig_challenges::vehicle_routing::{Challenge as VehicleRoutingChallenge, Solution as VehicleRoutingSolution},
    std::panic::catch_unwind,
    solve::solve
};

#[cfg(feature = "knapsack")]
pub type Challenge = KnapsackChallenge;
#[cfg(feature = "knapsack")]
pub type Solution = KnapsackSolution;

#[cfg(feature = "vector_search")]
pub type Challenge = VectorSearchChallenge;
#[cfg(feature = "vector_search")]
pub type Solution = VectorSearchSolution;

#[cfg(feature = "satisfiability")]
pub type Challenge = SatisfiabilityChallenge;
#[cfg(feature = "satisfiability")]
pub type Solution = SatisfiabilitySolution;

#[cfg(feature = "vehicle_routing")]
pub type Challenge = VehicleRoutingChallenge;
#[cfg(feature = "vehicle_routing")]
pub type Solution = VehicleRoutingSolution;

#[cfg(feature = "entry_point")]
#[unsafe(no_mangle)]
pub extern "C" fn entry_point(challenge: Challenge) -> Option<Solution>
{
    return catch_unwind(|| {
        return solve(challenge);
    }).unwrap_or(None);
}