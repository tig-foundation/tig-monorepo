use cudarc::{
    driver::{safe::LaunchConfig, CudaModule, CudaStream, PushKernelArg},
    runtime::sys::cudaDeviceProp,
};
use std::sync::Arc;
use tig_challenges::vector_search::*;

pub fn solve_challenge(
    challenge: &Challenge,
    module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    prop: &cudaDeviceProp,
) -> anyhow::Result<Option<Solution>> {
    let d_seed = stream.memcpy_stod(&challenge.seed.to_vec()).unwrap();
    let d_solution_indexes = stream
        .alloc_zeros::<u32>(challenge.difficulty.num_queries as usize)
        .unwrap();

    let simple_search_kernel = module.load_function("simple_search").unwrap();

    let threads_per_block = prop.maxThreadsPerBlock as u32;
    let blocks = challenge.difficulty.num_queries;

    let cfg = LaunchConfig {
        grid_dim: (blocks as u32, 1, 1),
        block_dim: (threads_per_block as u32, 1, 1),
        shared_mem_bytes: (threads_per_block + 250) * 4,
    };

    let mut builder = stream.launch_builder(&simple_search_kernel);
    unsafe {
        builder
            .arg(&d_seed)
            .arg(&(challenge.vector_dims as u32))
            .arg(&(challenge.database_size as u32))
            .arg(&(challenge.difficulty.num_queries as u32))
            .arg(&d_solution_indexes)
            .launch(cfg)
            .unwrap();
    }

    let indexes = stream
        .memcpy_dtov(&d_solution_indexes)
        .unwrap()
        .into_iter()
        .map(|x| x as usize)
        .collect::<Vec<_>>();

    Ok(Some(Solution { indexes }))
}

#[test]
fn test_simple_search() {
    let ptx = cudarc::nvrtc::Ptx::from_file("/home/ubuntu/cuda_test/asd.ptx");
    let ctx = cudarc::driver::CudaContext::new(0).unwrap();
    let module = ctx.load_module(ptx).unwrap();
    let stream = ctx.default_stream();
    let prop = cudarc::runtime::result::device::get_device_prop(0).unwrap();

    let c = Challenge::generate_instance(
        [0; 32],
        &Difficulty {
            num_queries: 10000,
            better_than_baseline: 0,
        },
        module.clone(),
        stream.clone(),
        &prop,
    )
    .unwrap();

    let solution = solve_challenge(&c, module.clone(), stream.clone(), &prop)
        .unwrap()
        .unwrap();
    println!("indexes: {:?}", solution.indexes);
    println!(
        "verify: {:?}",
        c.verify_solution(&solution, module.clone(), stream.clone(), &prop)
    );
}
