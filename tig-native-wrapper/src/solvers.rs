use {
    std::path::PathBuf,
    tig_challenges::{
        knapsack::{Challenge as KnapsackChallenge, Solution as KnapsackSolution},
        satisfiability::{Challenge as SatisfiabilityChallenge, Solution as SatisfiabilitySolution},
        vector_search::{Challenge as VectorSearchChallenge, Solution as VectorSearchSolution},
        vehicle_routing::{Challenge as VehicleRoutingChallenge, Solution as VehicleRoutingSolution},
    },
    crate::dylib::{load_module}
};

macro_rules! generate_solvers {
    ($(($name:ident, $challenge:ty, $solution:ty)),* $(,)?) => {
        $(
            pub fn $name(
                library_path: String,
                challenge: $challenge,
                fuel: Option<u64>
            ) -> Result<(Option<$solution>, u64, i64, u64, u64), String>
            {
                let library = load_module(&PathBuf::from(library_path))?;
                let solve_fn = unsafe { library.get::<fn($challenge, Option<u64>) -> Option<$solution>>(b"entry_point").map_err(|e| e.to_string())? };

                if fuel.is_some()
                {
                    let fuel_remaining_ptr = unsafe { *library.get::<*mut i64>(b"__fuel_remaining").map_err(|e| e.to_string())? };
                    unsafe { *fuel_remaining_ptr = fuel.unwrap() as i64 };
                }

                let solution = solve_fn(challenge, fuel);

                let fuel_remaining_ptr = unsafe { *library.get::<*const i64>(b"__fuel_remaining").map_err(|e| e.to_string())? };
                let runtime_signature_ptr = unsafe { *library.get::<*const u64>(b"__runtime_signature").map_err(|e| e.to_string())? };
                let max_memory_usage_ptr = unsafe { *library.get::<*const u64>(b"__max_memory_usage").map_err(|e| e.to_string())? };
                let total_memory_usage_ptr = unsafe { *library.get::<*const u64>(b"__total_memory_usage").map_err(|e| e.to_string())? };
                let fuel_remaining = unsafe { *fuel_remaining_ptr };
                let runtime_signature = unsafe { *runtime_signature_ptr };
                let max_memory_usage = unsafe { *max_memory_usage_ptr };
                let total_memory_usage = unsafe { *total_memory_usage_ptr };

                Ok((solution, runtime_signature, fuel_remaining, max_memory_usage, total_memory_usage))
            }
        )*
    };
}

generate_solvers!(
    (solve_knapsack, KnapsackChallenge, KnapsackSolution),
    (solve_satisfiability, SatisfiabilityChallenge, SatisfiabilitySolution),
    (solve_vector_search, VectorSearchChallenge, VectorSearchSolution),
    (solve_vehicle_routing, VehicleRoutingChallenge, VehicleRoutingSolution)
);