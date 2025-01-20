extern crate tig_challenges;

use {
    tig_challenges::knapsack::{Challenge as KnapsackChallenge, Solution as KnapsackSolution},
    tig_challenges::vector_search::{Challenge as VectorSearchChallenge, Solution as VectorSearchSolution},
    tig_challenges::satisfiability::{Challenge as SatisfiabilityChallenge, Solution as SatisfiabilitySolution},
    tig_challenges::vehicle_routing::{Challenge as VehicleRoutingChallenge, Solution as VehicleRoutingSolution},
};

#[cfg(feature = "knapsack")]
type Challenge = KnapsackChallenge;
#[cfg(feature = "knapsack")]
type Solution = KnapsackSolution;

#[cfg(feature = "vector_search")]
type Challenge = VectorSearchChallenge;
#[cfg(feature = "vector_search")]
type Solution = VectorSearchSolution;

#[cfg(feature = "satisfiability")]
type Challenge = SatisfiabilityChallenge;
#[cfg(feature = "satisfiability")]
type Solution = SatisfiabilitySolution;

#[cfg(feature = "vehicle_routing")]
type Challenge = VehicleRoutingChallenge;
#[cfg(feature = "vehicle_routing")]
type Solution = VehicleRoutingSolution;

#[cfg(feature = "entry_point")]
fn solve(challenge: Challenge) -> Option<Solution>
{
    return Some(Solution { 
        items: vec![0, 1, 2] 
    });
}

#[cfg(feature = "entry_point")]
#[unsafe(no_mangle)]
pub extern "C" fn entry_point(challenge: Challenge) -> Option<Solution>
{
    return solve(challenge);
}
