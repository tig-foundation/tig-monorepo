use anyhow::{anyhow, Result};
use tig_challenges::{knapsack, satisfiability, vehicle_routing, ChallengeTrait};
use tig_structs::core::*;
use tig_utils::decompress_obj;
use wasmi::{Config, Engine, Linker, Module, Store, StoreLimitsBuilder};

const BUFFER_SIZE: usize = u16::MAX as usize;

pub fn compute_solution(
    settings: &BenchmarkSettings,
    nonce: u32,
    wasm: &[u8],
    max_memory: u64,
    max_fuel: u64,
) -> Result<Result<SolutionData>> {
    if settings.difficulty.len() != 2 {
        return Err(anyhow!("Unsupported difficulty length"));
    }

    let mut config = Config::default();
    config.update_runtime_signature(true);
    config.consume_fuel(true);

    let limits = StoreLimitsBuilder::new()
        .memory_size(max_memory as usize)
        .memories(1)
        .trap_on_grow_failure(true)
        .build();
    // Setup instance of wasm module
    let engine = Engine::new(&config);
    let mut store = Store::new(&engine, limits);
    store.add_fuel(max_fuel).unwrap();
    let linker = Linker::new(&engine);
    let module = Module::new(store.engine(), wasm)
        .map_err(|e| anyhow!("Failed to instantiate module: {}", e))?;
    let instance = &linker
        .instantiate(&mut store, &module)
        .map_err(|e| anyhow!("Failed to instantiate linker: {}", e))?
        .start(&mut store)
        .map_err(|e| anyhow!("Failed to start module: {}", e))?;

    // Create memory for entry_point to write solution to
    let mut buffer = [0u8; BUFFER_SIZE];
    let memory = instance
        .get_memory(&store, "memory")
        .ok_or_else(|| anyhow!("Failed to find memory"))?;
    memory
        .write(&mut store, 0, &buffer)
        .map_err(|e| anyhow!("Failed to write to memory: {}", e))?;

    // Run algorithm
    let func = instance
        .get_func(&store, "entry_point")
        .ok_or_else(|| anyhow!("Failed to find entry_point"))?;
    let seed = settings.calc_seed(nonce);
    store.set_runtime_signature(seed as u64);
    if let Err(e) = func
        .typed::<(u32, i32, i32, i32, i32), ()>(&store)
        .map_err(|e| anyhow!("Failed to instantiate function: {}", e))?
        .call(
            &mut store,
            (
                seed,
                settings.difficulty[0],
                settings.difficulty[1],
                0,
                BUFFER_SIZE as i32,
            ),
        )
    {
        return Ok(Err(anyhow!("Error occured during execution: {}", e)));
    }
    // Get runtime signature
    let runtime_signature_u64 = store.get_runtime_signature();
    let runtime_signature = (runtime_signature_u64 as u32) ^ ((runtime_signature_u64 >> 32) as u32);
    let fuel_consumed = store.fuel_consumed().unwrap();
    // Read solution from memory
    memory
        .read(&store, 0, &mut buffer)
        .map_err(|e| anyhow!("Failed to read from memory: {}", e))?;
    let solution_len = u32::from_be_bytes(buffer[0..4].try_into().unwrap()) as usize;
    if solution_len == 0 {
        return Ok(Err(anyhow!(
            "No solution found (runtime_signature: {}, fuel_consumed: {})",
            runtime_signature,
            fuel_consumed
        )));
    }
    if solution_len > BUFFER_SIZE - 4 {
        return Ok(Err(anyhow!(
            "Solution too large (solution_size: {}, runtime_signature: {}, fuel_consumed: {})",
            solution_len,
            runtime_signature,
            fuel_consumed
        )));
    }
    let solution = decompress_obj(&buffer[4..4 + solution_len])
        .map_err(|e| anyhow!("Failed to convert buffer to solution: {}", e.to_string()))?;

    Ok(Ok(SolutionData {
        nonce,
        runtime_signature,
        fuel_consumed,
        solution,
    }))
}

pub fn verify_solution(
    settings: &BenchmarkSettings,
    nonce: u32,
    solution: &Solution,
) -> Result<Result<()>> {
    let seed = settings.calc_seed(nonce);
    match settings.challenge_id.as_str() {
        "c001" => {
            let challenge =
                satisfiability::Challenge::generate_instance_from_vec(seed, &settings.difficulty)
                    .map_err(|e| {
                    anyhow!(
                        "satisfiability::Challenge::generate_instance_from_vec error: {}",
                        e
                    )
                })?;
            match satisfiability::Solution::try_from(solution.clone()) {
                Ok(solution) => Ok(challenge.verify_solution(&solution)),
                Err(_) => Ok(Err(anyhow!(
                    "Invalid solution. Cannot convert to satisfiability::Solution"
                ))),
            }
        }
        "c002" => {
            let challenge =
                vehicle_routing::Challenge::generate_instance_from_vec(seed, &settings.difficulty)
                    .map_err(|e| {
                        anyhow!(
                            "vehicle_routing::Challenge::generate_instance_from_vec error: {}",
                            e
                        )
                    })?;
            match vehicle_routing::Solution::try_from(solution.clone()) {
                Ok(solution) => Ok(challenge.verify_solution(&solution)),
                Err(_) => Ok(Err(anyhow!(
                    "Invalid solution. Cannot convert to vehicle_routing::Solution"
                ))),
            }
        }
        "c003" => {
            let challenge =
                knapsack::Challenge::generate_instance_from_vec(seed, &settings.difficulty)
                    .map_err(|e| {
                        anyhow!(
                            "knapsack::Challenge::generate_instance_from_vec error: {}",
                            e
                        )
                    })?;
            match knapsack::Solution::try_from(solution.clone()) {
                Ok(solution) => Ok(challenge.verify_solution(&solution)),
                Err(_) => Ok(Err(anyhow!(
                    "Invalid solution. Cannot convert to knapsack::Solution"
                ))),
            }
        }
        _ => panic!("Unknown challenge"),
    }
}
