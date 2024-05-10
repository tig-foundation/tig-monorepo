use anyhow::{anyhow, Result};
use tig_challenges::{knapsack, satisfiability, vehicle_routing, ChallengeTrait};
pub use tig_structs::core::{BenchmarkSettings, Solution, SolutionData};
use tig_utils::decompress_obj;
use wasmi::{Config, Engine, Linker, Module, Store, StoreLimitsBuilder};

const BUFFER_SIZE: usize = u16::MAX as usize;

#[derive(Debug, Clone, PartialEq)]
pub enum ComputeResult {
    NoSolution(SolutionData),
    InvalidSolution(SolutionData),
    ValidSolution(SolutionData),
    SolutionTooLarge(SolutionData),
}

pub fn compute_solution(
    settings: &BenchmarkSettings,
    nonce: u32,
    wasm: &[u8],
    max_memory: u64,
    max_fuel: u64,
) -> Result<ComputeResult> {
    assert_eq!(
        settings.difficulty.len(),
        2,
        "Unsupported difficulty length"
    );

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
    let module = Module::new(store.engine(), wasm).expect("Failed to instantiate module");
    let instance = &linker
        .instantiate(&mut store, &module)
        .expect("Failed to instantiate linker")
        .start(&mut store)
        .expect("Failed to start module");

    // Create memory for entry_point to write solution to
    let mut buffer = [0u8; BUFFER_SIZE];
    let memory = instance
        .get_memory(&store, "memory")
        .expect("Failed to find memory");
    memory
        .write(&mut store, 0, &buffer)
        .expect("Failed to write to memory");

    // Run algorithm
    let func = instance
        .get_func(&store, "entry_point")
        .expect("Failed to find entry_point");
    let seed = settings.calc_seed(nonce);
    store.set_runtime_signature(seed as u64);
    if let Err(e) = func
        .typed::<(u32, i32, i32, i32, i32), ()>(&store)
        .expect("Failed to instantiate function")
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
        return Err(anyhow!("Error occured during execution: {}", e));
    }
    // Get runtime signature
    let runtime_signature_u64 = store.get_runtime_signature();
    let runtime_signature = (runtime_signature_u64 as u32) ^ ((runtime_signature_u64 >> 32) as u32);
    let fuel_consumed = store.fuel_consumed().unwrap();
    // Read solution from memory
    memory
        .read(&store, 0, &mut buffer)
        .expect("Failed to read from memory");
    let valid_solution = buffer[0] == 1;
    let solution_len = u32::from_be_bytes(buffer[1..5].try_into().unwrap()) as usize;
    let mut solution_data = SolutionData {
        nonce,
        runtime_signature,
        fuel_consumed,
        solution: Solution::new(),
    };
    if solution_len == 0 {
        return Ok(ComputeResult::NoSolution(solution_data));
    }
    if solution_len > BUFFER_SIZE - 5 {
        return Ok(ComputeResult::SolutionTooLarge(solution_data));
    }
    solution_data.solution =
        decompress_obj(&buffer[5..5 + solution_len]).expect("Failed to convert buffer to solution");

    match valid_solution {
        true => Ok(ComputeResult::ValidSolution(solution_data)),
        false => Ok(ComputeResult::InvalidSolution(solution_data)),
    }
}

pub fn verify_solution(
    settings: &BenchmarkSettings,
    nonce: u32,
    solution: &Solution,
) -> Result<()> {
    let seed = settings.calc_seed(nonce);
    match settings.challenge_id.as_str() {
        "c001" => {
            let challenge =
                satisfiability::Challenge::generate_instance_from_vec(seed, &settings.difficulty)
                    .expect("Failed to generate satisfiability instance");
            match satisfiability::Solution::try_from(solution.clone()) {
                Ok(solution) => challenge.verify_solution(&solution),
                Err(_) => Err(anyhow!(
                    "Invalid solution. Cannot convert to satisfiability::Solution"
                )),
            }
        }
        "c002" => {
            let challenge =
                vehicle_routing::Challenge::generate_instance_from_vec(seed, &settings.difficulty)
                    .expect("Failed to generate vehicle_routing instance");
            match vehicle_routing::Solution::try_from(solution.clone()) {
                Ok(solution) => challenge.verify_solution(&solution),
                Err(_) => Err(anyhow!(
                    "Invalid solution. Cannot convert to vehicle_routing::Solution"
                )),
            }
        }
        "c003" => {
            let challenge =
                knapsack::Challenge::generate_instance_from_vec(seed, &settings.difficulty)
                    .expect("Failed to generate knapsack instance");
            match knapsack::Solution::try_from(solution.clone()) {
                Ok(solution) => challenge.verify_solution(&solution),
                Err(_) => Err(anyhow!(
                    "Invalid solution. Cannot convert to knapsack::Solution"
                )),
            }
        }
        _ => panic!("Unknown challenge"),
    }
}
