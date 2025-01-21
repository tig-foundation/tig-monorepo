extern crate tig_challenges;
mod solve;

use {
    tig_challenges::{
        knapsack, vector_search, satisfiability, vehicle_routing,
    },
    std::panic::catch_unwind,
    solve::solve
};

macro_rules! define_challenge_types {
    ($(($feature:literal, $challenge:path, $solution:path)),*) => {
        $(
            #[cfg(feature = $feature)]
            pub type Challenge = $challenge;
            #[cfg(feature = $feature)]
            pub type Solution = $solution;
        )*
    };
}

define_challenge_types!(
    ("knapsack", knapsack::Challenge, knapsack::Solution),
    ("vector_search", vector_search::Challenge, vector_search::Solution),
    ("satisfiability", satisfiability::Challenge, satisfiability::Solution),
    ("vehicle_routing", vehicle_routing::Challenge, vehicle_routing::Solution)
);

#[cfg(feature = "entry_point")]
#[unsafe(no_mangle)]
pub extern "C" fn entry_point(challenge: Challenge) -> Option<Solution>
{
    return catch_unwind(|| {
        return solve(challenge);
    }).unwrap_or(None);
}