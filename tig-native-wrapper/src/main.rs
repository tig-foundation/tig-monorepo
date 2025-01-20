mod dylib;
mod solvers;

use {
    tig_challenges::{
        knapsack::{Challenge as KnapsackChallenge},
        satisfiability::{Challenge as SatisfiabilityChallenge},
        vector_search::{Challenge as VectorSearchChallenge},
        vehicle_routing::{Challenge as VehicleRoutingChallenge},
    },
};

macro_rules! handle_challenge {
    ($challenge_type:expr, $challenge_json:expr, $library_path:expr, $max_fuel:expr, $solver_fn:ident, $challenge:ty) => {
        {
            let challenge: $challenge = serde_json::from_str($challenge_json)
                .map_err(|e| format!("Failed to parse challenge: {}", e))
                .unwrap();

            let result = solvers::$solver_fn(
                $library_path.to_string(),
                challenge,
                Some($max_fuel)
            );

            if result.is_err()
            {
                println!("Error: {:?}", result.err().unwrap());
                return;
            }

            let (solution, runtime_signature, fuel_remaining) = result.unwrap();
            println!("{}", serde_json::json!({
                "solution": solution,
                "runtime_signature": runtime_signature,
                "fuel_consumed": $max_fuel as i64 - fuel_remaining.max(0),
                "nonce": 0,
            }));
        }
    };
}

fn main()
{
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 5
    {
        println!("Usage: {} <library_path> <challenge_type> <challenge_json> <max_fuel>", args[0]);
        println!("Challenge types: knapsack, satisfiability, vector_search, vehicle_routing");
        return;
    }

    let library_path = &args[1];
    let challenge_type = &args[2];
    let challenge_json = &args[3];
    let max_fuel = args[4].parse::<u64>().unwrap();

    match challenge_type.as_str()
    {
        "knapsack" => handle_challenge!(challenge_type, challenge_json, library_path, max_fuel, solve_knapsack, KnapsackChallenge),
        "satisfiability" => handle_challenge!(challenge_type, challenge_json, library_path, max_fuel, solve_satisfiability, SatisfiabilityChallenge),
        "vector_search" => handle_challenge!(challenge_type, challenge_json, library_path, max_fuel, solve_vector_search, VectorSearchChallenge),
        "vehicle_routing" => handle_challenge!(challenge_type, challenge_json, library_path, max_fuel, solve_vehicle_routing, VehicleRoutingChallenge),
        _ =>
        {
            println!("Invalid challenge type");
            return;
        }
    }
}
