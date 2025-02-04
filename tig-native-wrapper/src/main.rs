mod dylib;
mod solvers;

use {
    tig_challenges::{
        knapsack::{Challenge as KnapsackChallenge},
        satisfiability::{Challenge as SatisfiabilityChallenge},
        vector_search::{Challenge as VectorSearchChallenge},
        vehicle_routing::{Challenge as VehicleRoutingChallenge},
    },
    tig_utils::{dejsonify, jsonify},
    tig_structs::core::OutputData,
    std::io::{self, Read},
};

macro_rules! handle_challenge {
    ($challenge_type:expr, $challenge_json:expr, $library_path:expr, $max_fuel:expr, $solver_fn:ident, $challenge:ty) => {
        {
            let challenge: $challenge = dejsonify::<$challenge>($challenge_json)
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

            let (solution, runtime_signature, fuel_remaining, max_memory_usage, total_memory_usage) = result.unwrap();
            let output_data = OutputData {
                nonce: 0,
                runtime_signature,
                fuel_consumed: ($max_fuel as i64 - fuel_remaining.max(0)) as u64,
                solution: if solution.is_some() 
                    { Some(dejsonify(&jsonify(&solution.unwrap())).unwrap()) } 
                else 
                    { None },
            };

            println!("{}", jsonify(&output_data));
        }
    };
}

fn main()
{
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4
    {
        println!("Usage: {} <library_path> <challenge_type> <max_fuel>", args[0]);
        println!("Challenge types: knapsack, satisfiability, vector_search, vehicle_routing");
        return;
    }

    let library_path = &args[1];
    let challenge_type = &args[2];
    let max_fuel = args[3].parse::<u64>().unwrap();

    let mut challenge_json = String::new();
    io::stdin().read_to_string(&mut challenge_json).unwrap();

    match challenge_type.as_str()
    {
        "knapsack" => handle_challenge!(challenge_type, &challenge_json, library_path, max_fuel, solve_knapsack, KnapsackChallenge),
        "satisfiability" => handle_challenge!(challenge_type, &challenge_json, library_path, max_fuel, solve_satisfiability, SatisfiabilityChallenge),
        "vector_search"|"vectorsearch" => handle_challenge!(challenge_type, &challenge_json, library_path, max_fuel, solve_vector_search, VectorSearchChallenge),
        "vehicle_routing"|"vehiclerouting" => handle_challenge!(challenge_type, &challenge_json, library_path, max_fuel, solve_vehicle_routing, VehicleRoutingChallenge),
        _ =>
        {
            println!("Invalid challenge type");
            return;
        }
    }
}
