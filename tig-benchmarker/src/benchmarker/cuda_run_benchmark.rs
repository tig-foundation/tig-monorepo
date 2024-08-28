use super::{Job, NonceIterator};
use crate::future_utils;
use cudarc::driver::*;
use cudarc::nvrtc::{compile_ptx, Ptx};
use future_utils::{spawn, time, yield_now, Mutex};
use once_cell::sync::OnceCell;
use std::collections::HashMap;
use std::sync::Arc;
use tig_algorithms::{c001, c002, c003, c004, CudaKernel};
use tig_challenges::ChallengeTrait;
use tig_worker::{compute_solution, verify_solution, SolutionData};

static PTX_CACHE: OnceCell<Mutex<HashMap<String, Ptx>>> = OnceCell::new();

pub async fn get_or_compile_cuda(
    key: &String,
    kernel: &Option<CudaKernel>,
    dev: &Arc<CudaDevice>,
) -> HashMap<&'static str, CudaFunction> {
    if kernel.is_none() {
        println!("No CUDA optimisations available for '{}'", key);
        return HashMap::new();
    }
    let kernel = kernel.as_ref().unwrap();
    let ptx_cache = PTX_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut ptx_cache = ptx_cache.lock().await;
    if !ptx_cache.contains_key(key) {
        let start = std::time::Instant::now();
        println!("Compiling CUDA kernels for '{}'", key);
        ptx_cache.insert(
            key.clone(),
            compile_ptx(kernel.src)
                .map_err(|e| format!("Cuda Kernel failed to compile\nError: {}", e))
                .unwrap_or_else(|err_msg| panic!("{}", err_msg)),
        );
        println!(
            "CUDA kernels for '{}' compiled in {}ms",
            key,
            start.elapsed().as_millis()
        );
    }
    let ptx = ptx_cache.get(key).cloned().unwrap();
    dev.load_ptx(ptx, key, &kernel.funcs)
        .map_err(|e| {
            format!(
                "Failed to load functions '{:?}' from CUDA kernel '{}'\nError: {}",
                kernel.funcs, key, e
            )
        })
        .unwrap_or_else(|err_msg| panic!("{}", err_msg));
    kernel
        .funcs
        .iter()
        .map(|&name| (name, dev.get_func(key, name).unwrap()))
        .collect()
}

pub async fn execute(
    nonce_iters: Vec<Arc<Mutex<NonceIterator>>>,
    job: &Job,
    wasm: &Vec<u8>,
    solutions_data: Arc<Mutex<Vec<SolutionData>>>,
    solutions_count: Arc<Mutex<u32>>,
) {
    for nonce_iter in nonce_iters {
        let job = job.clone();
        let wasm = wasm.clone();
        let solutions_data = solutions_data.clone();
        let solutions_count = solutions_count.clone();
        spawn(async move {
            let mut last_yield = time();
            let dev = CudaDevice::new(0).expect("Failed to create CudaDevice");
            let mut challenge_cuda_funcs: Option<HashMap<&'static str, CudaFunction>> = None;
            let mut algorithm_cuda_funcs: Option<HashMap<&'static str, CudaFunction>> = None;
            loop {
                match {
                    let mut nonce_iter = (*nonce_iter).lock().await;
                    (*nonce_iter).next()
                } {
                    None => break,
                    Some(nonce) => {
                        let now = time();
                        if now - last_yield > 25 {
                            yield_now().await;
                            last_yield = now;
                        }
                        let seeds = job.settings.calc_seeds(nonce);
                        let skip = match job.settings.challenge_id.as_str() {
                            "c001" => {
                                type CudaSolveChallengeFn =
                                    fn(
                                        &tig_challenges::c001::Challenge,
                                        &Arc<CudaDevice>,
                                        HashMap<&'static str, CudaFunction>,
                                    )
                                        -> anyhow::Result<Option<tig_challenges::c001::Solution>>;

                                let filter_step = match job.settings.algorithm_id.as_str() {
                                    // #[cfg(feature = "c001_a001")]
                                    // "c001_a001" => Some((c001::c001_a001::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a001::KERNEL)),

                                    // #[cfg(feature = "c001_a002")]
                                    // "c001_a002" => Some((c001::c001_a002::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a002::KERNEL)),

                                    // #[cfg(feature = "c001_a003")]
                                    // "c001_a003" => Some((c001::c001_a003::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a003::KERNEL)),

                                    // #[cfg(feature = "c001_a004")]
                                    // "c001_a004" => Some((c001::c001_a004::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a004::KERNEL)),

                                    // #[cfg(feature = "c001_a005")]
                                    // "c001_a005" => Some((c001::c001_a005::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a005::KERNEL)),

                                    // #[cfg(feature = "c001_a006")]
                                    // "c001_a006" => Some((c001::c001_a006::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a006::KERNEL)),

                                    // #[cfg(feature = "c001_a007")]
                                    // "c001_a007" => Some((c001::c001_a007::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a007::KERNEL)),

                                    // #[cfg(feature = "c001_a008")]
                                    // "c001_a008" => Some((c001::c001_a008::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a008::KERNEL)),

                                    // #[cfg(feature = "c001_a009")]
                                    // "c001_a009" => Some((c001::c001_a009::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a009::KERNEL)),

                                    // #[cfg(feature = "c001_a010")]
                                    // "c001_a010" => Some((c001::c001_a010::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a010::KERNEL)),

                                    #[cfg(feature = "c001_a011")]
                                    "c001_a011" => Some((c001::c001_a011::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a011::KERNEL)),

                                    // #[cfg(feature = "c001_a012")]
                                    // "c001_a012" => Some((c001::c001_a012::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a012::KERNEL)),

                                    // #[cfg(feature = "c001_a013")]
                                    // "c001_a013" => Some((c001::c001_a013::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a013::KERNEL)),

                                    // #[cfg(feature = "c001_a014")]
                                    // "c001_a014" => Some((c001::c001_a014::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a014::KERNEL)),

                                    // #[cfg(feature = "c001_a015")]
                                    // "c001_a015" => Some((c001::c001_a015::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a015::KERNEL)),

                                    // #[cfg(feature = "c001_a016")]
                                    // "c001_a016" => Some((c001::c001_a016::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a016::KERNEL)),

                                    // #[cfg(feature = "c001_a017")]
                                    // "c001_a017" => Some((c001::c001_a017::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a017::KERNEL)),

                                    // #[cfg(feature = "c001_a018")]
                                    // "c001_a018" => Some((c001::c001_a018::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a018::KERNEL)),

                                    // #[cfg(feature = "c001_a019")]
                                    // "c001_a019" => Some((c001::c001_a019::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a019::KERNEL)),

                                    // #[cfg(feature = "c001_a020")]
                                    // "c001_a020" => Some((c001::c001_a020::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a020::KERNEL)),

                                    // #[cfg(feature = "c001_a021")]
                                    // "c001_a021" => Some((c001::c001_a021::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a021::KERNEL)),

                                    // #[cfg(feature = "c001_a022")]
                                    // "c001_a022" => Some((c001::c001_a022::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a022::KERNEL)),

                                    // #[cfg(feature = "c001_a023")]
                                    // "c001_a023" => Some((c001::c001_a023::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a023::KERNEL)),

                                    // #[cfg(feature = "c001_a024")]
                                    // "c001_a024" => Some((c001::c001_a024::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a024::KERNEL)),

                                    // #[cfg(feature = "c001_a025")]
                                    // "c001_a025" => Some((c001::c001_a025::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a025::KERNEL)),

                                    // #[cfg(feature = "c001_a026")]
                                    // "c001_a026" => Some((c001::c001_a026::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a026::KERNEL)),

                                    // #[cfg(feature = "c001_a027")]
                                    // "c001_a027" => Some((c001::c001_a027::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a027::KERNEL)),

                                    // #[cfg(feature = "c001_a028")]
                                    // "c001_a028" => Some((c001::c001_a028::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a028::KERNEL)),

                                    // #[cfg(feature = "c001_a029")]
                                    // "c001_a029" => Some((c001::c001_a029::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a029::KERNEL)),

                                    // #[cfg(feature = "c001_a030")]
                                    // "c001_a030" => Some((c001::c001_a030::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a030::KERNEL)),

                                    // #[cfg(feature = "c001_a031")]
                                    // "c001_a031" => Some((c001::c001_a031::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a031::KERNEL)),

                                    // #[cfg(feature = "c001_a032")]
                                    // "c001_a032" => Some((c001::c001_a032::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a032::KERNEL)),

                                    // #[cfg(feature = "c001_a033")]
                                    // "c001_a033" => Some((c001::c001_a033::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a033::KERNEL)),

                                    // #[cfg(feature = "c001_a034")]
                                    // "c001_a034" => Some((c001::c001_a034::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a034::KERNEL)),

                                    // #[cfg(feature = "c001_a035")]
                                    // "c001_a035" => Some((c001::c001_a035::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a035::KERNEL)),

                                    // #[cfg(feature = "c001_a036")]
                                    // "c001_a036" => Some((c001::c001_a036::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a036::KERNEL)),

                                    // #[cfg(feature = "c001_a037")]
                                    // "c001_a037" => Some((c001::c001_a037::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a037::KERNEL)),

                                    // #[cfg(feature = "c001_a038")]
                                    // "c001_a038" => Some((c001::c001_a038::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a038::KERNEL)),

                                    // #[cfg(feature = "c001_a039")]
                                    // "c001_a039" => Some((c001::c001_a039::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a039::KERNEL)),

                                    // #[cfg(feature = "c001_a040")]
                                    // "c001_a040" => Some((c001::c001_a040::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a040::KERNEL)),

                                    // #[cfg(feature = "c001_a041")]
                                    // "c001_a041" => Some((c001::c001_a041::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a041::KERNEL)),

                                    // #[cfg(feature = "c001_a042")]
                                    // "c001_a042" => Some((c001::c001_a042::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a042::KERNEL)),

                                    // #[cfg(feature = "c001_a043")]
                                    // "c001_a043" => Some((c001::c001_a043::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a043::KERNEL)),

                                    // #[cfg(feature = "c001_a044")]
                                    // "c001_a044" => Some((c001::c001_a044::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a044::KERNEL)),

                                    // #[cfg(feature = "c001_a045")]
                                    // "c001_a045" => Some((c001::c001_a045::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a045::KERNEL)),

                                    // #[cfg(feature = "c001_a046")]
                                    // "c001_a046" => Some((c001::c001_a046::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a046::KERNEL)),

                                    // #[cfg(feature = "c001_a047")]
                                    // "c001_a047" => Some((c001::c001_a047::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a047::KERNEL)),

                                    // #[cfg(feature = "c001_a048")]
                                    // "c001_a048" => Some((c001::c001_a048::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a048::KERNEL)),

                                    // #[cfg(feature = "c001_a049")]
                                    // "c001_a049" => Some((c001::c001_a049::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a049::KERNEL)),

                                    // #[cfg(feature = "c001_a050")]
                                    // "c001_a050" => Some((c001::c001_a050::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a050::KERNEL)),

                                    // #[cfg(feature = "c001_a051")]
                                    // "c001_a051" => Some((c001::c001_a051::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a051::KERNEL)),

                                    // #[cfg(feature = "c001_a052")]
                                    // "c001_a052" => Some((c001::c001_a052::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a052::KERNEL)),

                                    // #[cfg(feature = "c001_a053")]
                                    // "c001_a053" => Some((c001::c001_a053::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a053::KERNEL)),

                                    // #[cfg(feature = "c001_a054")]
                                    // "c001_a054" => Some((c001::c001_a054::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a054::KERNEL)),

                                    // #[cfg(feature = "c001_a055")]
                                    // "c001_a055" => Some((c001::c001_a055::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a055::KERNEL)),

                                    // #[cfg(feature = "c001_a056")]
                                    // "c001_a056" => Some((c001::c001_a056::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a056::KERNEL)),

                                    // #[cfg(feature = "c001_a057")]
                                    // "c001_a057" => Some((c001::c001_a057::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a057::KERNEL)),

                                    // #[cfg(feature = "c001_a058")]
                                    // "c001_a058" => Some((c001::c001_a058::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a058::KERNEL)),

                                    // #[cfg(feature = "c001_a059")]
                                    // "c001_a059" => Some((c001::c001_a059::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a059::KERNEL)),

                                    // #[cfg(feature = "c001_a060")]
                                    // "c001_a060" => Some((c001::c001_a060::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a060::KERNEL)),

                                    // #[cfg(feature = "c001_a061")]
                                    // "c001_a061" => Some((c001::c001_a061::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a061::KERNEL)),

                                    // #[cfg(feature = "c001_a062")]
                                    // "c001_a062" => Some((c001::c001_a062::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a062::KERNEL)),

                                    // #[cfg(feature = "c001_a063")]
                                    // "c001_a063" => Some((c001::c001_a063::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a063::KERNEL)),

                                    // #[cfg(feature = "c001_a064")]
                                    // "c001_a064" => Some((c001::c001_a064::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a064::KERNEL)),

                                    // #[cfg(feature = "c001_a065")]
                                    // "c001_a065" => Some((c001::c001_a065::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a065::KERNEL)),

                                    // #[cfg(feature = "c001_a066")]
                                    // "c001_a066" => Some((c001::c001_a066::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a066::KERNEL)),

                                    // #[cfg(feature = "c001_a067")]
                                    // "c001_a067" => Some((c001::c001_a067::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a067::KERNEL)),

                                    // #[cfg(feature = "c001_a068")]
                                    // "c001_a068" => Some((c001::c001_a068::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a068::KERNEL)),

                                    // #[cfg(feature = "c001_a069")]
                                    // "c001_a069" => Some((c001::c001_a069::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a069::KERNEL)),

                                    // #[cfg(feature = "c001_a070")]
                                    // "c001_a070" => Some((c001::c001_a070::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a070::KERNEL)),

                                    // #[cfg(feature = "c001_a071")]
                                    // "c001_a071" => Some((c001::c001_a071::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a071::KERNEL)),

                                    // #[cfg(feature = "c001_a072")]
                                    // "c001_a072" => Some((c001::c001_a072::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a072::KERNEL)),

                                    // #[cfg(feature = "c001_a073")]
                                    // "c001_a073" => Some((c001::c001_a073::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a073::KERNEL)),

                                    // #[cfg(feature = "c001_a074")]
                                    // "c001_a074" => Some((c001::c001_a074::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a074::KERNEL)),

                                    // #[cfg(feature = "c001_a075")]
                                    // "c001_a075" => Some((c001::c001_a075::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a075::KERNEL)),

                                    // #[cfg(feature = "c001_a076")]
                                    // "c001_a076" => Some((c001::c001_a076::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a076::KERNEL)),

                                    // #[cfg(feature = "c001_a077")]
                                    // "c001_a077" => Some((c001::c001_a077::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a077::KERNEL)),

                                    // #[cfg(feature = "c001_a078")]
                                    // "c001_a078" => Some((c001::c001_a078::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a078::KERNEL)),

                                    // #[cfg(feature = "c001_a079")]
                                    // "c001_a079" => Some((c001::c001_a079::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a079::KERNEL)),

                                    // #[cfg(feature = "c001_a080")]
                                    // "c001_a080" => Some((c001::c001_a080::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a080::KERNEL)),

                                    // #[cfg(feature = "c001_a081")]
                                    // "c001_a081" => Some((c001::c001_a081::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a081::KERNEL)),

                                    // #[cfg(feature = "c001_a082")]
                                    // "c001_a082" => Some((c001::c001_a082::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a082::KERNEL)),

                                    // #[cfg(feature = "c001_a083")]
                                    // "c001_a083" => Some((c001::c001_a083::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a083::KERNEL)),

                                    // #[cfg(feature = "c001_a084")]
                                    // "c001_a084" => Some((c001::c001_a084::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a084::KERNEL)),

                                    // #[cfg(feature = "c001_a085")]
                                    // "c001_a085" => Some((c001::c001_a085::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a085::KERNEL)),

                                    // #[cfg(feature = "c001_a086")]
                                    // "c001_a086" => Some((c001::c001_a086::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a086::KERNEL)),

                                    // #[cfg(feature = "c001_a087")]
                                    // "c001_a087" => Some((c001::c001_a087::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a087::KERNEL)),

                                    // #[cfg(feature = "c001_a088")]
                                    // "c001_a088" => Some((c001::c001_a088::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a088::KERNEL)),

                                    // #[cfg(feature = "c001_a089")]
                                    // "c001_a089" => Some((c001::c001_a089::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a089::KERNEL)),

                                    // #[cfg(feature = "c001_a090")]
                                    // "c001_a090" => Some((c001::c001_a090::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a090::KERNEL)),

                                    // #[cfg(feature = "c001_a091")]
                                    // "c001_a091" => Some((c001::c001_a091::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a091::KERNEL)),

                                    // #[cfg(feature = "c001_a092")]
                                    // "c001_a092" => Some((c001::c001_a092::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a092::KERNEL)),

                                    // #[cfg(feature = "c001_a093")]
                                    // "c001_a093" => Some((c001::c001_a093::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a093::KERNEL)),

                                    // #[cfg(feature = "c001_a094")]
                                    // "c001_a094" => Some((c001::c001_a094::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a094::KERNEL)),

                                    // #[cfg(feature = "c001_a095")]
                                    // "c001_a095" => Some((c001::c001_a095::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a095::KERNEL)),

                                    // #[cfg(feature = "c001_a096")]
                                    // "c001_a096" => Some((c001::c001_a096::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a096::KERNEL)),

                                    // #[cfg(feature = "c001_a097")]
                                    // "c001_a097" => Some((c001::c001_a097::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a097::KERNEL)),

                                    // #[cfg(feature = "c001_a098")]
                                    // "c001_a098" => Some((c001::c001_a098::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a098::KERNEL)),

                                    // #[cfg(feature = "c001_a099")]
                                    // "c001_a099" => Some((c001::c001_a099::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a099::KERNEL)),

                                    // #[cfg(feature = "c001_a100")]
                                    // "c001_a100" => Some((c001::c001_a100::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a100::KERNEL)),

                                    // #[cfg(feature = "c001_a101")]
                                    // "c001_a101" => Some((c001::c001_a101::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a101::KERNEL)),

                                    // #[cfg(feature = "c001_a102")]
                                    // "c001_a102" => Some((c001::c001_a102::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a102::KERNEL)),

                                    // #[cfg(feature = "c001_a103")]
                                    // "c001_a103" => Some((c001::c001_a103::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a103::KERNEL)),

                                    // #[cfg(feature = "c001_a104")]
                                    // "c001_a104" => Some((c001::c001_a104::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a104::KERNEL)),

                                    // #[cfg(feature = "c001_a105")]
                                    // "c001_a105" => Some((c001::c001_a105::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a105::KERNEL)),

                                    // #[cfg(feature = "c001_a106")]
                                    // "c001_a106" => Some((c001::c001_a106::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a106::KERNEL)),

                                    // #[cfg(feature = "c001_a107")]
                                    // "c001_a107" => Some((c001::c001_a107::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a107::KERNEL)),

                                    // #[cfg(feature = "c001_a108")]
                                    // "c001_a108" => Some((c001::c001_a108::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a108::KERNEL)),

                                    // #[cfg(feature = "c001_a109")]
                                    // "c001_a109" => Some((c001::c001_a109::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a109::KERNEL)),

                                    // #[cfg(feature = "c001_a110")]
                                    // "c001_a110" => Some((c001::c001_a110::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a110::KERNEL)),

                                    // #[cfg(feature = "c001_a111")]
                                    // "c001_a111" => Some((c001::c001_a111::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a111::KERNEL)),

                                    // #[cfg(feature = "c001_a112")]
                                    // "c001_a112" => Some((c001::c001_a112::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a112::KERNEL)),

                                    // #[cfg(feature = "c001_a113")]
                                    // "c001_a113" => Some((c001::c001_a113::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a113::KERNEL)),

                                    // #[cfg(feature = "c001_a114")]
                                    // "c001_a114" => Some((c001::c001_a114::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a114::KERNEL)),

                                    // #[cfg(feature = "c001_a115")]
                                    // "c001_a115" => Some((c001::c001_a115::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a115::KERNEL)),

                                    // #[cfg(feature = "c001_a116")]
                                    // "c001_a116" => Some((c001::c001_a116::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a116::KERNEL)),

                                    // #[cfg(feature = "c001_a117")]
                                    // "c001_a117" => Some((c001::c001_a117::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a117::KERNEL)),

                                    // #[cfg(feature = "c001_a118")]
                                    // "c001_a118" => Some((c001::c001_a118::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a118::KERNEL)),

                                    // #[cfg(feature = "c001_a119")]
                                    // "c001_a119" => Some((c001::c001_a119::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a119::KERNEL)),

                                    // #[cfg(feature = "c001_a120")]
                                    // "c001_a120" => Some((c001::c001_a120::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a120::KERNEL)),

                                    // #[cfg(feature = "c001_a121")]
                                    // "c001_a121" => Some((c001::c001_a121::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a121::KERNEL)),

                                    // #[cfg(feature = "c001_a122")]
                                    // "c001_a122" => Some((c001::c001_a122::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a122::KERNEL)),

                                    // #[cfg(feature = "c001_a123")]
                                    // "c001_a123" => Some((c001::c001_a123::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a123::KERNEL)),

                                    // #[cfg(feature = "c001_a124")]
                                    // "c001_a124" => Some((c001::c001_a124::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a124::KERNEL)),

                                    // #[cfg(feature = "c001_a125")]
                                    // "c001_a125" => Some((c001::c001_a125::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a125::KERNEL)),

                                    // #[cfg(feature = "c001_a126")]
                                    // "c001_a126" => Some((c001::c001_a126::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a126::KERNEL)),

                                    // #[cfg(feature = "c001_a127")]
                                    // "c001_a127" => Some((c001::c001_a127::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a127::KERNEL)),

                                    // #[cfg(feature = "c001_a128")]
                                    // "c001_a128" => Some((c001::c001_a128::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a128::KERNEL)),

                                    // #[cfg(feature = "c001_a129")]
                                    // "c001_a129" => Some((c001::c001_a129::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a129::KERNEL)),

                                    // #[cfg(feature = "c001_a130")]
                                    // "c001_a130" => Some((c001::c001_a130::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a130::KERNEL)),

                                    // #[cfg(feature = "c001_a131")]
                                    // "c001_a131" => Some((c001::c001_a131::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a131::KERNEL)),

                                    // #[cfg(feature = "c001_a132")]
                                    // "c001_a132" => Some((c001::c001_a132::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a132::KERNEL)),

                                    // #[cfg(feature = "c001_a133")]
                                    // "c001_a133" => Some((c001::c001_a133::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a133::KERNEL)),

                                    // #[cfg(feature = "c001_a134")]
                                    // "c001_a134" => Some((c001::c001_a134::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a134::KERNEL)),

                                    // #[cfg(feature = "c001_a135")]
                                    // "c001_a135" => Some((c001::c001_a135::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a135::KERNEL)),

                                    // #[cfg(feature = "c001_a136")]
                                    // "c001_a136" => Some((c001::c001_a136::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a136::KERNEL)),

                                    // #[cfg(feature = "c001_a137")]
                                    // "c001_a137" => Some((c001::c001_a137::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a137::KERNEL)),

                                    // #[cfg(feature = "c001_a138")]
                                    // "c001_a138" => Some((c001::c001_a138::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a138::KERNEL)),

                                    // #[cfg(feature = "c001_a139")]
                                    // "c001_a139" => Some((c001::c001_a139::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a139::KERNEL)),

                                    // #[cfg(feature = "c001_a140")]
                                    // "c001_a140" => Some((c001::c001_a140::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a140::KERNEL)),

                                    // #[cfg(feature = "c001_a141")]
                                    // "c001_a141" => Some((c001::c001_a141::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a141::KERNEL)),

                                    // #[cfg(feature = "c001_a142")]
                                    // "c001_a142" => Some((c001::c001_a142::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a142::KERNEL)),

                                    // #[cfg(feature = "c001_a143")]
                                    // "c001_a143" => Some((c001::c001_a143::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a143::KERNEL)),

                                    // #[cfg(feature = "c001_a144")]
                                    // "c001_a144" => Some((c001::c001_a144::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a144::KERNEL)),

                                    // #[cfg(feature = "c001_a145")]
                                    // "c001_a145" => Some((c001::c001_a145::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a145::KERNEL)),

                                    // #[cfg(feature = "c001_a146")]
                                    // "c001_a146" => Some((c001::c001_a146::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a146::KERNEL)),

                                    // #[cfg(feature = "c001_a147")]
                                    // "c001_a147" => Some((c001::c001_a147::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a147::KERNEL)),

                                    // #[cfg(feature = "c001_a148")]
                                    // "c001_a148" => Some((c001::c001_a148::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a148::KERNEL)),

                                    // #[cfg(feature = "c001_a149")]
                                    // "c001_a149" => Some((c001::c001_a149::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a149::KERNEL)),

                                    // #[cfg(feature = "c001_a150")]
                                    // "c001_a150" => Some((c001::c001_a150::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a150::KERNEL)),

                                    // #[cfg(feature = "c001_a151")]
                                    // "c001_a151" => Some((c001::c001_a151::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a151::KERNEL)),

                                    // #[cfg(feature = "c001_a152")]
                                    // "c001_a152" => Some((c001::c001_a152::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a152::KERNEL)),

                                    // #[cfg(feature = "c001_a153")]
                                    // "c001_a153" => Some((c001::c001_a153::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a153::KERNEL)),

                                    // #[cfg(feature = "c001_a154")]
                                    // "c001_a154" => Some((c001::c001_a154::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a154::KERNEL)),

                                    // #[cfg(feature = "c001_a155")]
                                    // "c001_a155" => Some((c001::c001_a155::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a155::KERNEL)),

                                    // #[cfg(feature = "c001_a156")]
                                    // "c001_a156" => Some((c001::c001_a156::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a156::KERNEL)),

                                    // #[cfg(feature = "c001_a157")]
                                    // "c001_a157" => Some((c001::c001_a157::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a157::KERNEL)),

                                    // #[cfg(feature = "c001_a158")]
                                    // "c001_a158" => Some((c001::c001_a158::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a158::KERNEL)),

                                    // #[cfg(feature = "c001_a159")]
                                    // "c001_a159" => Some((c001::c001_a159::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a159::KERNEL)),

                                    // #[cfg(feature = "c001_a160")]
                                    // "c001_a160" => Some((c001::c001_a160::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a160::KERNEL)),

                                    // #[cfg(feature = "c001_a161")]
                                    // "c001_a161" => Some((c001::c001_a161::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a161::KERNEL)),

                                    // #[cfg(feature = "c001_a162")]
                                    // "c001_a162" => Some((c001::c001_a162::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a162::KERNEL)),

                                    // #[cfg(feature = "c001_a163")]
                                    // "c001_a163" => Some((c001::c001_a163::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a163::KERNEL)),

                                    // #[cfg(feature = "c001_a164")]
                                    // "c001_a164" => Some((c001::c001_a164::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a164::KERNEL)),

                                    // #[cfg(feature = "c001_a165")]
                                    // "c001_a165" => Some((c001::c001_a165::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a165::KERNEL)),

                                    // #[cfg(feature = "c001_a166")]
                                    // "c001_a166" => Some((c001::c001_a166::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a166::KERNEL)),

                                    // #[cfg(feature = "c001_a167")]
                                    // "c001_a167" => Some((c001::c001_a167::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a167::KERNEL)),

                                    // #[cfg(feature = "c001_a168")]
                                    // "c001_a168" => Some((c001::c001_a168::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a168::KERNEL)),

                                    // #[cfg(feature = "c001_a169")]
                                    // "c001_a169" => Some((c001::c001_a169::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a169::KERNEL)),

                                    // #[cfg(feature = "c001_a170")]
                                    // "c001_a170" => Some((c001::c001_a170::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a170::KERNEL)),

                                    // #[cfg(feature = "c001_a171")]
                                    // "c001_a171" => Some((c001::c001_a171::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a171::KERNEL)),

                                    // #[cfg(feature = "c001_a172")]
                                    // "c001_a172" => Some((c001::c001_a172::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a172::KERNEL)),

                                    // #[cfg(feature = "c001_a173")]
                                    // "c001_a173" => Some((c001::c001_a173::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a173::KERNEL)),

                                    // #[cfg(feature = "c001_a174")]
                                    // "c001_a174" => Some((c001::c001_a174::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a174::KERNEL)),

                                    // #[cfg(feature = "c001_a175")]
                                    // "c001_a175" => Some((c001::c001_a175::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a175::KERNEL)),

                                    // #[cfg(feature = "c001_a176")]
                                    // "c001_a176" => Some((c001::c001_a176::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a176::KERNEL)),

                                    // #[cfg(feature = "c001_a177")]
                                    // "c001_a177" => Some((c001::c001_a177::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a177::KERNEL)),

                                    // #[cfg(feature = "c001_a178")]
                                    // "c001_a178" => Some((c001::c001_a178::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a178::KERNEL)),

                                    // #[cfg(feature = "c001_a179")]
                                    // "c001_a179" => Some((c001::c001_a179::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a179::KERNEL)),

                                    // #[cfg(feature = "c001_a180")]
                                    // "c001_a180" => Some((c001::c001_a180::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a180::KERNEL)),

                                    // #[cfg(feature = "c001_a181")]
                                    // "c001_a181" => Some((c001::c001_a181::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a181::KERNEL)),

                                    // #[cfg(feature = "c001_a182")]
                                    // "c001_a182" => Some((c001::c001_a182::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a182::KERNEL)),

                                    // #[cfg(feature = "c001_a183")]
                                    // "c001_a183" => Some((c001::c001_a183::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a183::KERNEL)),

                                    // #[cfg(feature = "c001_a184")]
                                    // "c001_a184" => Some((c001::c001_a184::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a184::KERNEL)),

                                    // #[cfg(feature = "c001_a185")]
                                    // "c001_a185" => Some((c001::c001_a185::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a185::KERNEL)),

                                    // #[cfg(feature = "c001_a186")]
                                    // "c001_a186" => Some((c001::c001_a186::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a186::KERNEL)),

                                    // #[cfg(feature = "c001_a187")]
                                    // "c001_a187" => Some((c001::c001_a187::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a187::KERNEL)),

                                    // #[cfg(feature = "c001_a188")]
                                    // "c001_a188" => Some((c001::c001_a188::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a188::KERNEL)),

                                    // #[cfg(feature = "c001_a189")]
                                    // "c001_a189" => Some((c001::c001_a189::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a189::KERNEL)),

                                    // #[cfg(feature = "c001_a190")]
                                    // "c001_a190" => Some((c001::c001_a190::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a190::KERNEL)),

                                    // #[cfg(feature = "c001_a191")]
                                    // "c001_a191" => Some((c001::c001_a191::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a191::KERNEL)),

                                    // #[cfg(feature = "c001_a192")]
                                    // "c001_a192" => Some((c001::c001_a192::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a192::KERNEL)),

                                    // #[cfg(feature = "c001_a193")]
                                    // "c001_a193" => Some((c001::c001_a193::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a193::KERNEL)),

                                    // #[cfg(feature = "c001_a194")]
                                    // "c001_a194" => Some((c001::c001_a194::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a194::KERNEL)),

                                    // #[cfg(feature = "c001_a195")]
                                    // "c001_a195" => Some((c001::c001_a195::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a195::KERNEL)),

                                    // #[cfg(feature = "c001_a196")]
                                    // "c001_a196" => Some((c001::c001_a196::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a196::KERNEL)),

                                    // #[cfg(feature = "c001_a197")]
                                    // "c001_a197" => Some((c001::c001_a197::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a197::KERNEL)),

                                    // #[cfg(feature = "c001_a198")]
                                    // "c001_a198" => Some((c001::c001_a198::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a198::KERNEL)),

                                    // #[cfg(feature = "c001_a199")]
                                    // "c001_a199" => Some((c001::c001_a199::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a199::KERNEL)),

                                    // #[cfg(feature = "c001_a200")]
                                    // "c001_a200" => Some((c001::c001_a200::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a200::KERNEL)),

                                    // #[cfg(feature = "c001_a201")]
                                    // "c001_a201" => Some((c001::c001_a201::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a201::KERNEL)),

                                    // #[cfg(feature = "c001_a202")]
                                    // "c001_a202" => Some((c001::c001_a202::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a202::KERNEL)),

                                    // #[cfg(feature = "c001_a203")]
                                    // "c001_a203" => Some((c001::c001_a203::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a203::KERNEL)),

                                    // #[cfg(feature = "c001_a204")]
                                    // "c001_a204" => Some((c001::c001_a204::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a204::KERNEL)),

                                    // #[cfg(feature = "c001_a205")]
                                    // "c001_a205" => Some((c001::c001_a205::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a205::KERNEL)),

                                    // #[cfg(feature = "c001_a206")]
                                    // "c001_a206" => Some((c001::c001_a206::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a206::KERNEL)),

                                    // #[cfg(feature = "c001_a207")]
                                    // "c001_a207" => Some((c001::c001_a207::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a207::KERNEL)),

                                    // #[cfg(feature = "c001_a208")]
                                    // "c001_a208" => Some((c001::c001_a208::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a208::KERNEL)),

                                    // #[cfg(feature = "c001_a209")]
                                    // "c001_a209" => Some((c001::c001_a209::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a209::KERNEL)),

                                    // #[cfg(feature = "c001_a210")]
                                    // "c001_a210" => Some((c001::c001_a210::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a210::KERNEL)),

                                    // #[cfg(feature = "c001_a211")]
                                    // "c001_a211" => Some((c001::c001_a211::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a211::KERNEL)),

                                    // #[cfg(feature = "c001_a212")]
                                    // "c001_a212" => Some((c001::c001_a212::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a212::KERNEL)),

                                    // #[cfg(feature = "c001_a213")]
                                    // "c001_a213" => Some((c001::c001_a213::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a213::KERNEL)),

                                    // #[cfg(feature = "c001_a214")]
                                    // "c001_a214" => Some((c001::c001_a214::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a214::KERNEL)),

                                    // #[cfg(feature = "c001_a215")]
                                    // "c001_a215" => Some((c001::c001_a215::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a215::KERNEL)),

                                    // #[cfg(feature = "c001_a216")]
                                    // "c001_a216" => Some((c001::c001_a216::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a216::KERNEL)),

                                    // #[cfg(feature = "c001_a217")]
                                    // "c001_a217" => Some((c001::c001_a217::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a217::KERNEL)),

                                    // #[cfg(feature = "c001_a218")]
                                    // "c001_a218" => Some((c001::c001_a218::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a218::KERNEL)),

                                    // #[cfg(feature = "c001_a219")]
                                    // "c001_a219" => Some((c001::c001_a219::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a219::KERNEL)),

                                    // #[cfg(feature = "c001_a220")]
                                    // "c001_a220" => Some((c001::c001_a220::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a220::KERNEL)),

                                    // #[cfg(feature = "c001_a221")]
                                    // "c001_a221" => Some((c001::c001_a221::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a221::KERNEL)),

                                    // #[cfg(feature = "c001_a222")]
                                    // "c001_a222" => Some((c001::c001_a222::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a222::KERNEL)),

                                    // #[cfg(feature = "c001_a223")]
                                    // "c001_a223" => Some((c001::c001_a223::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a223::KERNEL)),

                                    // #[cfg(feature = "c001_a224")]
                                    // "c001_a224" => Some((c001::c001_a224::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a224::KERNEL)),

                                    // #[cfg(feature = "c001_a225")]
                                    // "c001_a225" => Some((c001::c001_a225::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a225::KERNEL)),

                                    // #[cfg(feature = "c001_a226")]
                                    // "c001_a226" => Some((c001::c001_a226::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a226::KERNEL)),

                                    // #[cfg(feature = "c001_a227")]
                                    // "c001_a227" => Some((c001::c001_a227::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a227::KERNEL)),

                                    // #[cfg(feature = "c001_a228")]
                                    // "c001_a228" => Some((c001::c001_a228::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a228::KERNEL)),

                                    // #[cfg(feature = "c001_a229")]
                                    // "c001_a229" => Some((c001::c001_a229::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a229::KERNEL)),

                                    // #[cfg(feature = "c001_a230")]
                                    // "c001_a230" => Some((c001::c001_a230::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a230::KERNEL)),

                                    // #[cfg(feature = "c001_a231")]
                                    // "c001_a231" => Some((c001::c001_a231::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a231::KERNEL)),

                                    // #[cfg(feature = "c001_a232")]
                                    // "c001_a232" => Some((c001::c001_a232::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a232::KERNEL)),

                                    // #[cfg(feature = "c001_a233")]
                                    // "c001_a233" => Some((c001::c001_a233::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a233::KERNEL)),

                                    // #[cfg(feature = "c001_a234")]
                                    // "c001_a234" => Some((c001::c001_a234::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a234::KERNEL)),

                                    // #[cfg(feature = "c001_a235")]
                                    // "c001_a235" => Some((c001::c001_a235::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a235::KERNEL)),

                                    // #[cfg(feature = "c001_a236")]
                                    // "c001_a236" => Some((c001::c001_a236::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a236::KERNEL)),

                                    // #[cfg(feature = "c001_a237")]
                                    // "c001_a237" => Some((c001::c001_a237::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a237::KERNEL)),

                                    // #[cfg(feature = "c001_a238")]
                                    // "c001_a238" => Some((c001::c001_a238::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a238::KERNEL)),

                                    // #[cfg(feature = "c001_a239")]
                                    // "c001_a239" => Some((c001::c001_a239::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a239::KERNEL)),

                                    // #[cfg(feature = "c001_a240")]
                                    // "c001_a240" => Some((c001::c001_a240::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a240::KERNEL)),

                                    // #[cfg(feature = "c001_a241")]
                                    // "c001_a241" => Some((c001::c001_a241::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a241::KERNEL)),

                                    // #[cfg(feature = "c001_a242")]
                                    // "c001_a242" => Some((c001::c001_a242::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a242::KERNEL)),

                                    // #[cfg(feature = "c001_a243")]
                                    // "c001_a243" => Some((c001::c001_a243::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a243::KERNEL)),

                                    // #[cfg(feature = "c001_a244")]
                                    // "c001_a244" => Some((c001::c001_a244::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a244::KERNEL)),

                                    // #[cfg(feature = "c001_a245")]
                                    // "c001_a245" => Some((c001::c001_a245::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a245::KERNEL)),

                                    // #[cfg(feature = "c001_a246")]
                                    // "c001_a246" => Some((c001::c001_a246::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a246::KERNEL)),

                                    // #[cfg(feature = "c001_a247")]
                                    // "c001_a247" => Some((c001::c001_a247::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a247::KERNEL)),

                                    // #[cfg(feature = "c001_a248")]
                                    // "c001_a248" => Some((c001::c001_a248::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a248::KERNEL)),

                                    // #[cfg(feature = "c001_a249")]
                                    // "c001_a249" => Some((c001::c001_a249::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a249::KERNEL)),

                                    // #[cfg(feature = "c001_a250")]
                                    // "c001_a250" => Some((c001::c001_a250::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a250::KERNEL)),

                                    // #[cfg(feature = "c001_a251")]
                                    // "c001_a251" => Some((c001::c001_a251::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a251::KERNEL)),

                                    // #[cfg(feature = "c001_a252")]
                                    // "c001_a252" => Some((c001::c001_a252::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a252::KERNEL)),

                                    // #[cfg(feature = "c001_a253")]
                                    // "c001_a253" => Some((c001::c001_a253::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a253::KERNEL)),

                                    // #[cfg(feature = "c001_a254")]
                                    // "c001_a254" => Some((c001::c001_a254::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a254::KERNEL)),

                                    // #[cfg(feature = "c001_a255")]
                                    // "c001_a255" => Some((c001::c001_a255::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a255::KERNEL)),

                                    // #[cfg(feature = "c001_a256")]
                                    // "c001_a256" => Some((c001::c001_a256::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a256::KERNEL)),

                                    // #[cfg(feature = "c001_a257")]
                                    // "c001_a257" => Some((c001::c001_a257::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a257::KERNEL)),

                                    // #[cfg(feature = "c001_a258")]
                                    // "c001_a258" => Some((c001::c001_a258::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a258::KERNEL)),

                                    // #[cfg(feature = "c001_a259")]
                                    // "c001_a259" => Some((c001::c001_a259::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a259::KERNEL)),

                                    // #[cfg(feature = "c001_a260")]
                                    // "c001_a260" => Some((c001::c001_a260::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a260::KERNEL)),

                                    // #[cfg(feature = "c001_a261")]
                                    // "c001_a261" => Some((c001::c001_a261::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a261::KERNEL)),

                                    // #[cfg(feature = "c001_a262")]
                                    // "c001_a262" => Some((c001::c001_a262::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a262::KERNEL)),

                                    // #[cfg(feature = "c001_a263")]
                                    // "c001_a263" => Some((c001::c001_a263::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a263::KERNEL)),

                                    // #[cfg(feature = "c001_a264")]
                                    // "c001_a264" => Some((c001::c001_a264::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a264::KERNEL)),

                                    // #[cfg(feature = "c001_a265")]
                                    // "c001_a265" => Some((c001::c001_a265::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a265::KERNEL)),

                                    // #[cfg(feature = "c001_a266")]
                                    // "c001_a266" => Some((c001::c001_a266::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a266::KERNEL)),

                                    // #[cfg(feature = "c001_a267")]
                                    // "c001_a267" => Some((c001::c001_a267::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a267::KERNEL)),

                                    // #[cfg(feature = "c001_a268")]
                                    // "c001_a268" => Some((c001::c001_a268::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a268::KERNEL)),

                                    // #[cfg(feature = "c001_a269")]
                                    // "c001_a269" => Some((c001::c001_a269::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a269::KERNEL)),

                                    // #[cfg(feature = "c001_a270")]
                                    // "c001_a270" => Some((c001::c001_a270::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a270::KERNEL)),

                                    // #[cfg(feature = "c001_a271")]
                                    // "c001_a271" => Some((c001::c001_a271::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a271::KERNEL)),

                                    // #[cfg(feature = "c001_a272")]
                                    // "c001_a272" => Some((c001::c001_a272::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a272::KERNEL)),

                                    // #[cfg(feature = "c001_a273")]
                                    // "c001_a273" => Some((c001::c001_a273::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a273::KERNEL)),

                                    // #[cfg(feature = "c001_a274")]
                                    // "c001_a274" => Some((c001::c001_a274::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a274::KERNEL)),

                                    // #[cfg(feature = "c001_a275")]
                                    // "c001_a275" => Some((c001::c001_a275::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a275::KERNEL)),

                                    // #[cfg(feature = "c001_a276")]
                                    // "c001_a276" => Some((c001::c001_a276::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a276::KERNEL)),

                                    // #[cfg(feature = "c001_a277")]
                                    // "c001_a277" => Some((c001::c001_a277::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a277::KERNEL)),

                                    // #[cfg(feature = "c001_a278")]
                                    // "c001_a278" => Some((c001::c001_a278::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a278::KERNEL)),

                                    // #[cfg(feature = "c001_a279")]
                                    // "c001_a279" => Some((c001::c001_a279::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a279::KERNEL)),

                                    // #[cfg(feature = "c001_a280")]
                                    // "c001_a280" => Some((c001::c001_a280::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a280::KERNEL)),

                                    // #[cfg(feature = "c001_a281")]
                                    // "c001_a281" => Some((c001::c001_a281::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a281::KERNEL)),

                                    // #[cfg(feature = "c001_a282")]
                                    // "c001_a282" => Some((c001::c001_a282::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a282::KERNEL)),

                                    // #[cfg(feature = "c001_a283")]
                                    // "c001_a283" => Some((c001::c001_a283::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a283::KERNEL)),

                                    // #[cfg(feature = "c001_a284")]
                                    // "c001_a284" => Some((c001::c001_a284::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a284::KERNEL)),

                                    // #[cfg(feature = "c001_a285")]
                                    // "c001_a285" => Some((c001::c001_a285::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a285::KERNEL)),

                                    // #[cfg(feature = "c001_a286")]
                                    // "c001_a286" => Some((c001::c001_a286::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a286::KERNEL)),

                                    // #[cfg(feature = "c001_a287")]
                                    // "c001_a287" => Some((c001::c001_a287::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a287::KERNEL)),

                                    // #[cfg(feature = "c001_a288")]
                                    // "c001_a288" => Some((c001::c001_a288::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a288::KERNEL)),

                                    // #[cfg(feature = "c001_a289")]
                                    // "c001_a289" => Some((c001::c001_a289::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a289::KERNEL)),

                                    // #[cfg(feature = "c001_a290")]
                                    // "c001_a290" => Some((c001::c001_a290::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a290::KERNEL)),

                                    // #[cfg(feature = "c001_a291")]
                                    // "c001_a291" => Some((c001::c001_a291::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a291::KERNEL)),

                                    // #[cfg(feature = "c001_a292")]
                                    // "c001_a292" => Some((c001::c001_a292::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a292::KERNEL)),

                                    // #[cfg(feature = "c001_a293")]
                                    // "c001_a293" => Some((c001::c001_a293::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a293::KERNEL)),

                                    // #[cfg(feature = "c001_a294")]
                                    // "c001_a294" => Some((c001::c001_a294::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a294::KERNEL)),

                                    // #[cfg(feature = "c001_a295")]
                                    // "c001_a295" => Some((c001::c001_a295::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a295::KERNEL)),

                                    // #[cfg(feature = "c001_a296")]
                                    // "c001_a296" => Some((c001::c001_a296::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a296::KERNEL)),

                                    // #[cfg(feature = "c001_a297")]
                                    // "c001_a297" => Some((c001::c001_a297::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a297::KERNEL)),

                                    // #[cfg(feature = "c001_a298")]
                                    // "c001_a298" => Some((c001::c001_a298::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a298::KERNEL)),

                                    // #[cfg(feature = "c001_a299")]
                                    // "c001_a299" => Some((c001::c001_a299::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a299::KERNEL)),

                                    // #[cfg(feature = "c001_a300")]
                                    // "c001_a300" => Some((c001::c001_a300::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a300::KERNEL)),

                                    // #[cfg(feature = "c001_a301")]
                                    // "c001_a301" => Some((c001::c001_a301::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a301::KERNEL)),

                                    // #[cfg(feature = "c001_a302")]
                                    // "c001_a302" => Some((c001::c001_a302::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a302::KERNEL)),

                                    // #[cfg(feature = "c001_a303")]
                                    // "c001_a303" => Some((c001::c001_a303::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a303::KERNEL)),

                                    // #[cfg(feature = "c001_a304")]
                                    // "c001_a304" => Some((c001::c001_a304::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a304::KERNEL)),

                                    // #[cfg(feature = "c001_a305")]
                                    // "c001_a305" => Some((c001::c001_a305::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a305::KERNEL)),

                                    // #[cfg(feature = "c001_a306")]
                                    // "c001_a306" => Some((c001::c001_a306::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a306::KERNEL)),

                                    // #[cfg(feature = "c001_a307")]
                                    // "c001_a307" => Some((c001::c001_a307::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a307::KERNEL)),

                                    // #[cfg(feature = "c001_a308")]
                                    // "c001_a308" => Some((c001::c001_a308::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a308::KERNEL)),

                                    // #[cfg(feature = "c001_a309")]
                                    // "c001_a309" => Some((c001::c001_a309::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a309::KERNEL)),

                                    // #[cfg(feature = "c001_a310")]
                                    // "c001_a310" => Some((c001::c001_a310::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a310::KERNEL)),

                                    // #[cfg(feature = "c001_a311")]
                                    // "c001_a311" => Some((c001::c001_a311::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a311::KERNEL)),

                                    // #[cfg(feature = "c001_a312")]
                                    // "c001_a312" => Some((c001::c001_a312::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a312::KERNEL)),

                                    // #[cfg(feature = "c001_a313")]
                                    // "c001_a313" => Some((c001::c001_a313::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a313::KERNEL)),

                                    // #[cfg(feature = "c001_a314")]
                                    // "c001_a314" => Some((c001::c001_a314::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a314::KERNEL)),

                                    // #[cfg(feature = "c001_a315")]
                                    // "c001_a315" => Some((c001::c001_a315::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a315::KERNEL)),

                                    // #[cfg(feature = "c001_a316")]
                                    // "c001_a316" => Some((c001::c001_a316::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a316::KERNEL)),

                                    // #[cfg(feature = "c001_a317")]
                                    // "c001_a317" => Some((c001::c001_a317::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a317::KERNEL)),

                                    // #[cfg(feature = "c001_a318")]
                                    // "c001_a318" => Some((c001::c001_a318::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a318::KERNEL)),

                                    // #[cfg(feature = "c001_a319")]
                                    // "c001_a319" => Some((c001::c001_a319::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a319::KERNEL)),

                                    // #[cfg(feature = "c001_a320")]
                                    // "c001_a320" => Some((c001::c001_a320::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a320::KERNEL)),

                                    // #[cfg(feature = "c001_a321")]
                                    // "c001_a321" => Some((c001::c001_a321::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a321::KERNEL)),

                                    // #[cfg(feature = "c001_a322")]
                                    // "c001_a322" => Some((c001::c001_a322::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a322::KERNEL)),

                                    // #[cfg(feature = "c001_a323")]
                                    // "c001_a323" => Some((c001::c001_a323::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a323::KERNEL)),

                                    // #[cfg(feature = "c001_a324")]
                                    // "c001_a324" => Some((c001::c001_a324::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a324::KERNEL)),

                                    // #[cfg(feature = "c001_a325")]
                                    // "c001_a325" => Some((c001::c001_a325::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a325::KERNEL)),

                                    // #[cfg(feature = "c001_a326")]
                                    // "c001_a326" => Some((c001::c001_a326::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a326::KERNEL)),

                                    // #[cfg(feature = "c001_a327")]
                                    // "c001_a327" => Some((c001::c001_a327::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a327::KERNEL)),

                                    // #[cfg(feature = "c001_a328")]
                                    // "c001_a328" => Some((c001::c001_a328::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a328::KERNEL)),

                                    // #[cfg(feature = "c001_a329")]
                                    // "c001_a329" => Some((c001::c001_a329::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a329::KERNEL)),

                                    // #[cfg(feature = "c001_a330")]
                                    // "c001_a330" => Some((c001::c001_a330::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a330::KERNEL)),

                                    // #[cfg(feature = "c001_a331")]
                                    // "c001_a331" => Some((c001::c001_a331::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a331::KERNEL)),

                                    // #[cfg(feature = "c001_a332")]
                                    // "c001_a332" => Some((c001::c001_a332::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a332::KERNEL)),

                                    // #[cfg(feature = "c001_a333")]
                                    // "c001_a333" => Some((c001::c001_a333::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a333::KERNEL)),

                                    // #[cfg(feature = "c001_a334")]
                                    // "c001_a334" => Some((c001::c001_a334::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a334::KERNEL)),

                                    // #[cfg(feature = "c001_a335")]
                                    // "c001_a335" => Some((c001::c001_a335::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a335::KERNEL)),

                                    // #[cfg(feature = "c001_a336")]
                                    // "c001_a336" => Some((c001::c001_a336::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a336::KERNEL)),

                                    // #[cfg(feature = "c001_a337")]
                                    // "c001_a337" => Some((c001::c001_a337::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a337::KERNEL)),

                                    // #[cfg(feature = "c001_a338")]
                                    // "c001_a338" => Some((c001::c001_a338::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a338::KERNEL)),

                                    // #[cfg(feature = "c001_a339")]
                                    // "c001_a339" => Some((c001::c001_a339::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a339::KERNEL)),

                                    // #[cfg(feature = "c001_a340")]
                                    // "c001_a340" => Some((c001::c001_a340::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a340::KERNEL)),

                                    // #[cfg(feature = "c001_a341")]
                                    // "c001_a341" => Some((c001::c001_a341::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a341::KERNEL)),

                                    // #[cfg(feature = "c001_a342")]
                                    // "c001_a342" => Some((c001::c001_a342::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a342::KERNEL)),

                                    // #[cfg(feature = "c001_a343")]
                                    // "c001_a343" => Some((c001::c001_a343::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a343::KERNEL)),

                                    // #[cfg(feature = "c001_a344")]
                                    // "c001_a344" => Some((c001::c001_a344::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a344::KERNEL)),

                                    // #[cfg(feature = "c001_a345")]
                                    // "c001_a345" => Some((c001::c001_a345::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a345::KERNEL)),

                                    // #[cfg(feature = "c001_a346")]
                                    // "c001_a346" => Some((c001::c001_a346::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a346::KERNEL)),

                                    // #[cfg(feature = "c001_a347")]
                                    // "c001_a347" => Some((c001::c001_a347::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a347::KERNEL)),

                                    // #[cfg(feature = "c001_a348")]
                                    // "c001_a348" => Some((c001::c001_a348::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a348::KERNEL)),

                                    // #[cfg(feature = "c001_a349")]
                                    // "c001_a349" => Some((c001::c001_a349::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a349::KERNEL)),

                                    // #[cfg(feature = "c001_a350")]
                                    // "c001_a350" => Some((c001::c001_a350::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a350::KERNEL)),

                                    // #[cfg(feature = "c001_a351")]
                                    // "c001_a351" => Some((c001::c001_a351::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a351::KERNEL)),

                                    // #[cfg(feature = "c001_a352")]
                                    // "c001_a352" => Some((c001::c001_a352::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a352::KERNEL)),

                                    // #[cfg(feature = "c001_a353")]
                                    // "c001_a353" => Some((c001::c001_a353::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a353::KERNEL)),

                                    // #[cfg(feature = "c001_a354")]
                                    // "c001_a354" => Some((c001::c001_a354::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a354::KERNEL)),

                                    // #[cfg(feature = "c001_a355")]
                                    // "c001_a355" => Some((c001::c001_a355::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a355::KERNEL)),

                                    // #[cfg(feature = "c001_a356")]
                                    // "c001_a356" => Some((c001::c001_a356::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a356::KERNEL)),

                                    // #[cfg(feature = "c001_a357")]
                                    // "c001_a357" => Some((c001::c001_a357::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a357::KERNEL)),

                                    // #[cfg(feature = "c001_a358")]
                                    // "c001_a358" => Some((c001::c001_a358::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a358::KERNEL)),

                                    // #[cfg(feature = "c001_a359")]
                                    // "c001_a359" => Some((c001::c001_a359::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a359::KERNEL)),

                                    // #[cfg(feature = "c001_a360")]
                                    // "c001_a360" => Some((c001::c001_a360::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a360::KERNEL)),

                                    // #[cfg(feature = "c001_a361")]
                                    // "c001_a361" => Some((c001::c001_a361::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a361::KERNEL)),

                                    // #[cfg(feature = "c001_a362")]
                                    // "c001_a362" => Some((c001::c001_a362::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a362::KERNEL)),

                                    // #[cfg(feature = "c001_a363")]
                                    // "c001_a363" => Some((c001::c001_a363::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a363::KERNEL)),

                                    // #[cfg(feature = "c001_a364")]
                                    // "c001_a364" => Some((c001::c001_a364::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a364::KERNEL)),

                                    // #[cfg(feature = "c001_a365")]
                                    // "c001_a365" => Some((c001::c001_a365::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a365::KERNEL)),

                                    // #[cfg(feature = "c001_a366")]
                                    // "c001_a366" => Some((c001::c001_a366::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a366::KERNEL)),

                                    // #[cfg(feature = "c001_a367")]
                                    // "c001_a367" => Some((c001::c001_a367::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a367::KERNEL)),

                                    // #[cfg(feature = "c001_a368")]
                                    // "c001_a368" => Some((c001::c001_a368::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a368::KERNEL)),

                                    // #[cfg(feature = "c001_a369")]
                                    // "c001_a369" => Some((c001::c001_a369::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a369::KERNEL)),

                                    // #[cfg(feature = "c001_a370")]
                                    // "c001_a370" => Some((c001::c001_a370::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a370::KERNEL)),

                                    // #[cfg(feature = "c001_a371")]
                                    // "c001_a371" => Some((c001::c001_a371::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a371::KERNEL)),

                                    // #[cfg(feature = "c001_a372")]
                                    // "c001_a372" => Some((c001::c001_a372::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a372::KERNEL)),

                                    // #[cfg(feature = "c001_a373")]
                                    // "c001_a373" => Some((c001::c001_a373::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a373::KERNEL)),

                                    // #[cfg(feature = "c001_a374")]
                                    // "c001_a374" => Some((c001::c001_a374::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a374::KERNEL)),

                                    // #[cfg(feature = "c001_a375")]
                                    // "c001_a375" => Some((c001::c001_a375::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a375::KERNEL)),

                                    // #[cfg(feature = "c001_a376")]
                                    // "c001_a376" => Some((c001::c001_a376::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a376::KERNEL)),

                                    // #[cfg(feature = "c001_a377")]
                                    // "c001_a377" => Some((c001::c001_a377::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a377::KERNEL)),

                                    // #[cfg(feature = "c001_a378")]
                                    // "c001_a378" => Some((c001::c001_a378::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a378::KERNEL)),

                                    // #[cfg(feature = "c001_a379")]
                                    // "c001_a379" => Some((c001::c001_a379::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a379::KERNEL)),

                                    // #[cfg(feature = "c001_a380")]
                                    // "c001_a380" => Some((c001::c001_a380::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a380::KERNEL)),

                                    // #[cfg(feature = "c001_a381")]
                                    // "c001_a381" => Some((c001::c001_a381::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a381::KERNEL)),

                                    // #[cfg(feature = "c001_a382")]
                                    // "c001_a382" => Some((c001::c001_a382::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a382::KERNEL)),

                                    // #[cfg(feature = "c001_a383")]
                                    // "c001_a383" => Some((c001::c001_a383::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a383::KERNEL)),

                                    // #[cfg(feature = "c001_a384")]
                                    // "c001_a384" => Some((c001::c001_a384::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a384::KERNEL)),

                                    // #[cfg(feature = "c001_a385")]
                                    // "c001_a385" => Some((c001::c001_a385::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a385::KERNEL)),

                                    // #[cfg(feature = "c001_a386")]
                                    // "c001_a386" => Some((c001::c001_a386::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a386::KERNEL)),

                                    // #[cfg(feature = "c001_a387")]
                                    // "c001_a387" => Some((c001::c001_a387::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a387::KERNEL)),

                                    // #[cfg(feature = "c001_a388")]
                                    // "c001_a388" => Some((c001::c001_a388::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a388::KERNEL)),

                                    // #[cfg(feature = "c001_a389")]
                                    // "c001_a389" => Some((c001::c001_a389::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a389::KERNEL)),

                                    // #[cfg(feature = "c001_a390")]
                                    // "c001_a390" => Some((c001::c001_a390::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a390::KERNEL)),

                                    // #[cfg(feature = "c001_a391")]
                                    // "c001_a391" => Some((c001::c001_a391::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a391::KERNEL)),

                                    // #[cfg(feature = "c001_a392")]
                                    // "c001_a392" => Some((c001::c001_a392::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a392::KERNEL)),

                                    // #[cfg(feature = "c001_a393")]
                                    // "c001_a393" => Some((c001::c001_a393::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a393::KERNEL)),

                                    // #[cfg(feature = "c001_a394")]
                                    // "c001_a394" => Some((c001::c001_a394::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a394::KERNEL)),

                                    // #[cfg(feature = "c001_a395")]
                                    // "c001_a395" => Some((c001::c001_a395::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a395::KERNEL)),

                                    // #[cfg(feature = "c001_a396")]
                                    // "c001_a396" => Some((c001::c001_a396::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a396::KERNEL)),

                                    // #[cfg(feature = "c001_a397")]
                                    // "c001_a397" => Some((c001::c001_a397::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a397::KERNEL)),

                                    // #[cfg(feature = "c001_a398")]
                                    // "c001_a398" => Some((c001::c001_a398::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a398::KERNEL)),

                                    // #[cfg(feature = "c001_a399")]
                                    // "c001_a399" => Some((c001::c001_a399::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a399::KERNEL)),

                                    // #[cfg(feature = "c001_a400")]
                                    // "c001_a400" => Some((c001::c001_a400::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a400::KERNEL)),

                                    // #[cfg(feature = "c001_a401")]
                                    // "c001_a401" => Some((c001::c001_a401::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a401::KERNEL)),

                                    // #[cfg(feature = "c001_a402")]
                                    // "c001_a402" => Some((c001::c001_a402::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a402::KERNEL)),

                                    // #[cfg(feature = "c001_a403")]
                                    // "c001_a403" => Some((c001::c001_a403::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a403::KERNEL)),

                                    // #[cfg(feature = "c001_a404")]
                                    // "c001_a404" => Some((c001::c001_a404::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a404::KERNEL)),

                                    // #[cfg(feature = "c001_a405")]
                                    // "c001_a405" => Some((c001::c001_a405::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a405::KERNEL)),

                                    // #[cfg(feature = "c001_a406")]
                                    // "c001_a406" => Some((c001::c001_a406::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a406::KERNEL)),

                                    // #[cfg(feature = "c001_a407")]
                                    // "c001_a407" => Some((c001::c001_a407::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a407::KERNEL)),

                                    // #[cfg(feature = "c001_a408")]
                                    // "c001_a408" => Some((c001::c001_a408::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a408::KERNEL)),

                                    // #[cfg(feature = "c001_a409")]
                                    // "c001_a409" => Some((c001::c001_a409::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a409::KERNEL)),

                                    // #[cfg(feature = "c001_a410")]
                                    // "c001_a410" => Some((c001::c001_a410::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a410::KERNEL)),

                                    // #[cfg(feature = "c001_a411")]
                                    // "c001_a411" => Some((c001::c001_a411::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a411::KERNEL)),

                                    // #[cfg(feature = "c001_a412")]
                                    // "c001_a412" => Some((c001::c001_a412::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a412::KERNEL)),

                                    // #[cfg(feature = "c001_a413")]
                                    // "c001_a413" => Some((c001::c001_a413::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a413::KERNEL)),

                                    // #[cfg(feature = "c001_a414")]
                                    // "c001_a414" => Some((c001::c001_a414::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a414::KERNEL)),

                                    // #[cfg(feature = "c001_a415")]
                                    // "c001_a415" => Some((c001::c001_a415::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a415::KERNEL)),

                                    // #[cfg(feature = "c001_a416")]
                                    // "c001_a416" => Some((c001::c001_a416::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a416::KERNEL)),

                                    // #[cfg(feature = "c001_a417")]
                                    // "c001_a417" => Some((c001::c001_a417::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a417::KERNEL)),

                                    // #[cfg(feature = "c001_a418")]
                                    // "c001_a418" => Some((c001::c001_a418::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a418::KERNEL)),

                                    // #[cfg(feature = "c001_a419")]
                                    // "c001_a419" => Some((c001::c001_a419::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a419::KERNEL)),

                                    // #[cfg(feature = "c001_a420")]
                                    // "c001_a420" => Some((c001::c001_a420::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a420::KERNEL)),

                                    // #[cfg(feature = "c001_a421")]
                                    // "c001_a421" => Some((c001::c001_a421::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a421::KERNEL)),

                                    // #[cfg(feature = "c001_a422")]
                                    // "c001_a422" => Some((c001::c001_a422::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a422::KERNEL)),

                                    // #[cfg(feature = "c001_a423")]
                                    // "c001_a423" => Some((c001::c001_a423::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a423::KERNEL)),

                                    // #[cfg(feature = "c001_a424")]
                                    // "c001_a424" => Some((c001::c001_a424::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a424::KERNEL)),

                                    // #[cfg(feature = "c001_a425")]
                                    // "c001_a425" => Some((c001::c001_a425::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a425::KERNEL)),

                                    // #[cfg(feature = "c001_a426")]
                                    // "c001_a426" => Some((c001::c001_a426::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a426::KERNEL)),

                                    // #[cfg(feature = "c001_a427")]
                                    // "c001_a427" => Some((c001::c001_a427::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a427::KERNEL)),

                                    // #[cfg(feature = "c001_a428")]
                                    // "c001_a428" => Some((c001::c001_a428::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a428::KERNEL)),

                                    // #[cfg(feature = "c001_a429")]
                                    // "c001_a429" => Some((c001::c001_a429::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a429::KERNEL)),

                                    // #[cfg(feature = "c001_a430")]
                                    // "c001_a430" => Some((c001::c001_a430::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a430::KERNEL)),

                                    // #[cfg(feature = "c001_a431")]
                                    // "c001_a431" => Some((c001::c001_a431::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a431::KERNEL)),

                                    // #[cfg(feature = "c001_a432")]
                                    // "c001_a432" => Some((c001::c001_a432::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a432::KERNEL)),

                                    // #[cfg(feature = "c001_a433")]
                                    // "c001_a433" => Some((c001::c001_a433::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a433::KERNEL)),

                                    // #[cfg(feature = "c001_a434")]
                                    // "c001_a434" => Some((c001::c001_a434::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a434::KERNEL)),

                                    // #[cfg(feature = "c001_a435")]
                                    // "c001_a435" => Some((c001::c001_a435::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a435::KERNEL)),

                                    // #[cfg(feature = "c001_a436")]
                                    // "c001_a436" => Some((c001::c001_a436::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a436::KERNEL)),

                                    // #[cfg(feature = "c001_a437")]
                                    // "c001_a437" => Some((c001::c001_a437::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a437::KERNEL)),

                                    // #[cfg(feature = "c001_a438")]
                                    // "c001_a438" => Some((c001::c001_a438::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a438::KERNEL)),

                                    // #[cfg(feature = "c001_a439")]
                                    // "c001_a439" => Some((c001::c001_a439::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a439::KERNEL)),

                                    // #[cfg(feature = "c001_a440")]
                                    // "c001_a440" => Some((c001::c001_a440::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a440::KERNEL)),

                                    // #[cfg(feature = "c001_a441")]
                                    // "c001_a441" => Some((c001::c001_a441::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a441::KERNEL)),

                                    // #[cfg(feature = "c001_a442")]
                                    // "c001_a442" => Some((c001::c001_a442::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a442::KERNEL)),

                                    // #[cfg(feature = "c001_a443")]
                                    // "c001_a443" => Some((c001::c001_a443::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a443::KERNEL)),

                                    // #[cfg(feature = "c001_a444")]
                                    // "c001_a444" => Some((c001::c001_a444::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a444::KERNEL)),

                                    // #[cfg(feature = "c001_a445")]
                                    // "c001_a445" => Some((c001::c001_a445::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a445::KERNEL)),

                                    // #[cfg(feature = "c001_a446")]
                                    // "c001_a446" => Some((c001::c001_a446::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a446::KERNEL)),

                                    // #[cfg(feature = "c001_a447")]
                                    // "c001_a447" => Some((c001::c001_a447::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a447::KERNEL)),

                                    // #[cfg(feature = "c001_a448")]
                                    // "c001_a448" => Some((c001::c001_a448::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a448::KERNEL)),

                                    // #[cfg(feature = "c001_a449")]
                                    // "c001_a449" => Some((c001::c001_a449::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a449::KERNEL)),

                                    // #[cfg(feature = "c001_a450")]
                                    // "c001_a450" => Some((c001::c001_a450::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a450::KERNEL)),

                                    // #[cfg(feature = "c001_a451")]
                                    // "c001_a451" => Some((c001::c001_a451::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a451::KERNEL)),

                                    // #[cfg(feature = "c001_a452")]
                                    // "c001_a452" => Some((c001::c001_a452::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a452::KERNEL)),

                                    // #[cfg(feature = "c001_a453")]
                                    // "c001_a453" => Some((c001::c001_a453::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a453::KERNEL)),

                                    // #[cfg(feature = "c001_a454")]
                                    // "c001_a454" => Some((c001::c001_a454::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a454::KERNEL)),

                                    // #[cfg(feature = "c001_a455")]
                                    // "c001_a455" => Some((c001::c001_a455::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a455::KERNEL)),

                                    // #[cfg(feature = "c001_a456")]
                                    // "c001_a456" => Some((c001::c001_a456::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a456::KERNEL)),

                                    // #[cfg(feature = "c001_a457")]
                                    // "c001_a457" => Some((c001::c001_a457::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a457::KERNEL)),

                                    // #[cfg(feature = "c001_a458")]
                                    // "c001_a458" => Some((c001::c001_a458::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a458::KERNEL)),

                                    // #[cfg(feature = "c001_a459")]
                                    // "c001_a459" => Some((c001::c001_a459::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a459::KERNEL)),

                                    // #[cfg(feature = "c001_a460")]
                                    // "c001_a460" => Some((c001::c001_a460::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a460::KERNEL)),

                                    // #[cfg(feature = "c001_a461")]
                                    // "c001_a461" => Some((c001::c001_a461::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a461::KERNEL)),

                                    // #[cfg(feature = "c001_a462")]
                                    // "c001_a462" => Some((c001::c001_a462::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a462::KERNEL)),

                                    // #[cfg(feature = "c001_a463")]
                                    // "c001_a463" => Some((c001::c001_a463::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a463::KERNEL)),

                                    // #[cfg(feature = "c001_a464")]
                                    // "c001_a464" => Some((c001::c001_a464::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a464::KERNEL)),

                                    // #[cfg(feature = "c001_a465")]
                                    // "c001_a465" => Some((c001::c001_a465::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a465::KERNEL)),

                                    // #[cfg(feature = "c001_a466")]
                                    // "c001_a466" => Some((c001::c001_a466::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a466::KERNEL)),

                                    // #[cfg(feature = "c001_a467")]
                                    // "c001_a467" => Some((c001::c001_a467::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a467::KERNEL)),

                                    // #[cfg(feature = "c001_a468")]
                                    // "c001_a468" => Some((c001::c001_a468::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a468::KERNEL)),

                                    // #[cfg(feature = "c001_a469")]
                                    // "c001_a469" => Some((c001::c001_a469::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a469::KERNEL)),

                                    // #[cfg(feature = "c001_a470")]
                                    // "c001_a470" => Some((c001::c001_a470::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a470::KERNEL)),

                                    // #[cfg(feature = "c001_a471")]
                                    // "c001_a471" => Some((c001::c001_a471::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a471::KERNEL)),

                                    // #[cfg(feature = "c001_a472")]
                                    // "c001_a472" => Some((c001::c001_a472::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a472::KERNEL)),

                                    // #[cfg(feature = "c001_a473")]
                                    // "c001_a473" => Some((c001::c001_a473::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a473::KERNEL)),

                                    // #[cfg(feature = "c001_a474")]
                                    // "c001_a474" => Some((c001::c001_a474::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a474::KERNEL)),

                                    // #[cfg(feature = "c001_a475")]
                                    // "c001_a475" => Some((c001::c001_a475::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a475::KERNEL)),

                                    // #[cfg(feature = "c001_a476")]
                                    // "c001_a476" => Some((c001::c001_a476::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a476::KERNEL)),

                                    // #[cfg(feature = "c001_a477")]
                                    // "c001_a477" => Some((c001::c001_a477::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a477::KERNEL)),

                                    // #[cfg(feature = "c001_a478")]
                                    // "c001_a478" => Some((c001::c001_a478::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a478::KERNEL)),

                                    // #[cfg(feature = "c001_a479")]
                                    // "c001_a479" => Some((c001::c001_a479::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a479::KERNEL)),

                                    // #[cfg(feature = "c001_a480")]
                                    // "c001_a480" => Some((c001::c001_a480::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a480::KERNEL)),

                                    // #[cfg(feature = "c001_a481")]
                                    // "c001_a481" => Some((c001::c001_a481::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a481::KERNEL)),

                                    // #[cfg(feature = "c001_a482")]
                                    // "c001_a482" => Some((c001::c001_a482::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a482::KERNEL)),

                                    // #[cfg(feature = "c001_a483")]
                                    // "c001_a483" => Some((c001::c001_a483::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a483::KERNEL)),

                                    // #[cfg(feature = "c001_a484")]
                                    // "c001_a484" => Some((c001::c001_a484::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a484::KERNEL)),

                                    // #[cfg(feature = "c001_a485")]
                                    // "c001_a485" => Some((c001::c001_a485::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a485::KERNEL)),

                                    // #[cfg(feature = "c001_a486")]
                                    // "c001_a486" => Some((c001::c001_a486::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a486::KERNEL)),

                                    // #[cfg(feature = "c001_a487")]
                                    // "c001_a487" => Some((c001::c001_a487::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a487::KERNEL)),

                                    // #[cfg(feature = "c001_a488")]
                                    // "c001_a488" => Some((c001::c001_a488::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a488::KERNEL)),

                                    // #[cfg(feature = "c001_a489")]
                                    // "c001_a489" => Some((c001::c001_a489::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a489::KERNEL)),

                                    // #[cfg(feature = "c001_a490")]
                                    // "c001_a490" => Some((c001::c001_a490::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a490::KERNEL)),

                                    // #[cfg(feature = "c001_a491")]
                                    // "c001_a491" => Some((c001::c001_a491::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a491::KERNEL)),

                                    // #[cfg(feature = "c001_a492")]
                                    // "c001_a492" => Some((c001::c001_a492::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a492::KERNEL)),

                                    // #[cfg(feature = "c001_a493")]
                                    // "c001_a493" => Some((c001::c001_a493::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a493::KERNEL)),

                                    // #[cfg(feature = "c001_a494")]
                                    // "c001_a494" => Some((c001::c001_a494::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a494::KERNEL)),

                                    // #[cfg(feature = "c001_a495")]
                                    // "c001_a495" => Some((c001::c001_a495::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a495::KERNEL)),

                                    // #[cfg(feature = "c001_a496")]
                                    // "c001_a496" => Some((c001::c001_a496::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a496::KERNEL)),

                                    // #[cfg(feature = "c001_a497")]
                                    // "c001_a497" => Some((c001::c001_a497::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a497::KERNEL)),

                                    // #[cfg(feature = "c001_a498")]
                                    // "c001_a498" => Some((c001::c001_a498::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a498::KERNEL)),

                                    // #[cfg(feature = "c001_a499")]
                                    // "c001_a499" => Some((c001::c001_a499::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a499::KERNEL)),

                                    // #[cfg(feature = "c001_a500")]
                                    // "c001_a500" => Some((c001::c001_a500::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a500::KERNEL)),

                                    // #[cfg(feature = "c001_a501")]
                                    // "c001_a501" => Some((c001::c001_a501::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a501::KERNEL)),

                                    // #[cfg(feature = "c001_a502")]
                                    // "c001_a502" => Some((c001::c001_a502::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a502::KERNEL)),

                                    // #[cfg(feature = "c001_a503")]
                                    // "c001_a503" => Some((c001::c001_a503::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a503::KERNEL)),

                                    // #[cfg(feature = "c001_a504")]
                                    // "c001_a504" => Some((c001::c001_a504::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a504::KERNEL)),

                                    // #[cfg(feature = "c001_a505")]
                                    // "c001_a505" => Some((c001::c001_a505::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a505::KERNEL)),

                                    // #[cfg(feature = "c001_a506")]
                                    // "c001_a506" => Some((c001::c001_a506::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a506::KERNEL)),

                                    // #[cfg(feature = "c001_a507")]
                                    // "c001_a507" => Some((c001::c001_a507::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a507::KERNEL)),

                                    // #[cfg(feature = "c001_a508")]
                                    // "c001_a508" => Some((c001::c001_a508::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a508::KERNEL)),

                                    // #[cfg(feature = "c001_a509")]
                                    // "c001_a509" => Some((c001::c001_a509::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a509::KERNEL)),

                                    // #[cfg(feature = "c001_a510")]
                                    // "c001_a510" => Some((c001::c001_a510::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a510::KERNEL)),

                                    // #[cfg(feature = "c001_a511")]
                                    // "c001_a511" => Some((c001::c001_a511::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a511::KERNEL)),

                                    // #[cfg(feature = "c001_a512")]
                                    // "c001_a512" => Some((c001::c001_a512::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a512::KERNEL)),

                                    // #[cfg(feature = "c001_a513")]
                                    // "c001_a513" => Some((c001::c001_a513::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a513::KERNEL)),

                                    // #[cfg(feature = "c001_a514")]
                                    // "c001_a514" => Some((c001::c001_a514::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a514::KERNEL)),

                                    // #[cfg(feature = "c001_a515")]
                                    // "c001_a515" => Some((c001::c001_a515::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a515::KERNEL)),

                                    // #[cfg(feature = "c001_a516")]
                                    // "c001_a516" => Some((c001::c001_a516::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a516::KERNEL)),

                                    // #[cfg(feature = "c001_a517")]
                                    // "c001_a517" => Some((c001::c001_a517::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a517::KERNEL)),

                                    // #[cfg(feature = "c001_a518")]
                                    // "c001_a518" => Some((c001::c001_a518::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a518::KERNEL)),

                                    // #[cfg(feature = "c001_a519")]
                                    // "c001_a519" => Some((c001::c001_a519::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a519::KERNEL)),

                                    // #[cfg(feature = "c001_a520")]
                                    // "c001_a520" => Some((c001::c001_a520::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a520::KERNEL)),

                                    // #[cfg(feature = "c001_a521")]
                                    // "c001_a521" => Some((c001::c001_a521::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a521::KERNEL)),

                                    // #[cfg(feature = "c001_a522")]
                                    // "c001_a522" => Some((c001::c001_a522::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a522::KERNEL)),

                                    // #[cfg(feature = "c001_a523")]
                                    // "c001_a523" => Some((c001::c001_a523::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a523::KERNEL)),

                                    // #[cfg(feature = "c001_a524")]
                                    // "c001_a524" => Some((c001::c001_a524::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a524::KERNEL)),

                                    // #[cfg(feature = "c001_a525")]
                                    // "c001_a525" => Some((c001::c001_a525::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a525::KERNEL)),

                                    // #[cfg(feature = "c001_a526")]
                                    // "c001_a526" => Some((c001::c001_a526::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a526::KERNEL)),

                                    // #[cfg(feature = "c001_a527")]
                                    // "c001_a527" => Some((c001::c001_a527::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a527::KERNEL)),

                                    // #[cfg(feature = "c001_a528")]
                                    // "c001_a528" => Some((c001::c001_a528::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a528::KERNEL)),

                                    // #[cfg(feature = "c001_a529")]
                                    // "c001_a529" => Some((c001::c001_a529::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a529::KERNEL)),

                                    // #[cfg(feature = "c001_a530")]
                                    // "c001_a530" => Some((c001::c001_a530::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a530::KERNEL)),

                                    // #[cfg(feature = "c001_a531")]
                                    // "c001_a531" => Some((c001::c001_a531::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a531::KERNEL)),

                                    // #[cfg(feature = "c001_a532")]
                                    // "c001_a532" => Some((c001::c001_a532::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a532::KERNEL)),

                                    // #[cfg(feature = "c001_a533")]
                                    // "c001_a533" => Some((c001::c001_a533::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a533::KERNEL)),

                                    // #[cfg(feature = "c001_a534")]
                                    // "c001_a534" => Some((c001::c001_a534::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a534::KERNEL)),

                                    // #[cfg(feature = "c001_a535")]
                                    // "c001_a535" => Some((c001::c001_a535::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a535::KERNEL)),

                                    // #[cfg(feature = "c001_a536")]
                                    // "c001_a536" => Some((c001::c001_a536::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a536::KERNEL)),

                                    // #[cfg(feature = "c001_a537")]
                                    // "c001_a537" => Some((c001::c001_a537::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a537::KERNEL)),

                                    // #[cfg(feature = "c001_a538")]
                                    // "c001_a538" => Some((c001::c001_a538::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a538::KERNEL)),

                                    // #[cfg(feature = "c001_a539")]
                                    // "c001_a539" => Some((c001::c001_a539::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a539::KERNEL)),

                                    // #[cfg(feature = "c001_a540")]
                                    // "c001_a540" => Some((c001::c001_a540::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a540::KERNEL)),

                                    // #[cfg(feature = "c001_a541")]
                                    // "c001_a541" => Some((c001::c001_a541::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a541::KERNEL)),

                                    // #[cfg(feature = "c001_a542")]
                                    // "c001_a542" => Some((c001::c001_a542::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a542::KERNEL)),

                                    // #[cfg(feature = "c001_a543")]
                                    // "c001_a543" => Some((c001::c001_a543::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a543::KERNEL)),

                                    // #[cfg(feature = "c001_a544")]
                                    // "c001_a544" => Some((c001::c001_a544::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a544::KERNEL)),

                                    // #[cfg(feature = "c001_a545")]
                                    // "c001_a545" => Some((c001::c001_a545::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a545::KERNEL)),

                                    // #[cfg(feature = "c001_a546")]
                                    // "c001_a546" => Some((c001::c001_a546::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a546::KERNEL)),

                                    // #[cfg(feature = "c001_a547")]
                                    // "c001_a547" => Some((c001::c001_a547::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a547::KERNEL)),

                                    // #[cfg(feature = "c001_a548")]
                                    // "c001_a548" => Some((c001::c001_a548::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a548::KERNEL)),

                                    // #[cfg(feature = "c001_a549")]
                                    // "c001_a549" => Some((c001::c001_a549::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a549::KERNEL)),

                                    // #[cfg(feature = "c001_a550")]
                                    // "c001_a550" => Some((c001::c001_a550::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a550::KERNEL)),

                                    // #[cfg(feature = "c001_a551")]
                                    // "c001_a551" => Some((c001::c001_a551::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a551::KERNEL)),

                                    // #[cfg(feature = "c001_a552")]
                                    // "c001_a552" => Some((c001::c001_a552::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a552::KERNEL)),

                                    // #[cfg(feature = "c001_a553")]
                                    // "c001_a553" => Some((c001::c001_a553::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a553::KERNEL)),

                                    // #[cfg(feature = "c001_a554")]
                                    // "c001_a554" => Some((c001::c001_a554::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a554::KERNEL)),

                                    // #[cfg(feature = "c001_a555")]
                                    // "c001_a555" => Some((c001::c001_a555::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a555::KERNEL)),

                                    // #[cfg(feature = "c001_a556")]
                                    // "c001_a556" => Some((c001::c001_a556::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a556::KERNEL)),

                                    // #[cfg(feature = "c001_a557")]
                                    // "c001_a557" => Some((c001::c001_a557::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a557::KERNEL)),

                                    // #[cfg(feature = "c001_a558")]
                                    // "c001_a558" => Some((c001::c001_a558::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a558::KERNEL)),

                                    // #[cfg(feature = "c001_a559")]
                                    // "c001_a559" => Some((c001::c001_a559::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a559::KERNEL)),

                                    // #[cfg(feature = "c001_a560")]
                                    // "c001_a560" => Some((c001::c001_a560::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a560::KERNEL)),

                                    // #[cfg(feature = "c001_a561")]
                                    // "c001_a561" => Some((c001::c001_a561::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a561::KERNEL)),

                                    // #[cfg(feature = "c001_a562")]
                                    // "c001_a562" => Some((c001::c001_a562::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a562::KERNEL)),

                                    // #[cfg(feature = "c001_a563")]
                                    // "c001_a563" => Some((c001::c001_a563::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a563::KERNEL)),

                                    // #[cfg(feature = "c001_a564")]
                                    // "c001_a564" => Some((c001::c001_a564::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a564::KERNEL)),

                                    // #[cfg(feature = "c001_a565")]
                                    // "c001_a565" => Some((c001::c001_a565::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a565::KERNEL)),

                                    // #[cfg(feature = "c001_a566")]
                                    // "c001_a566" => Some((c001::c001_a566::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a566::KERNEL)),

                                    // #[cfg(feature = "c001_a567")]
                                    // "c001_a567" => Some((c001::c001_a567::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a567::KERNEL)),

                                    // #[cfg(feature = "c001_a568")]
                                    // "c001_a568" => Some((c001::c001_a568::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a568::KERNEL)),

                                    // #[cfg(feature = "c001_a569")]
                                    // "c001_a569" => Some((c001::c001_a569::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a569::KERNEL)),

                                    // #[cfg(feature = "c001_a570")]
                                    // "c001_a570" => Some((c001::c001_a570::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a570::KERNEL)),

                                    // #[cfg(feature = "c001_a571")]
                                    // "c001_a571" => Some((c001::c001_a571::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a571::KERNEL)),

                                    // #[cfg(feature = "c001_a572")]
                                    // "c001_a572" => Some((c001::c001_a572::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a572::KERNEL)),

                                    // #[cfg(feature = "c001_a573")]
                                    // "c001_a573" => Some((c001::c001_a573::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a573::KERNEL)),

                                    // #[cfg(feature = "c001_a574")]
                                    // "c001_a574" => Some((c001::c001_a574::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a574::KERNEL)),

                                    // #[cfg(feature = "c001_a575")]
                                    // "c001_a575" => Some((c001::c001_a575::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a575::KERNEL)),

                                    // #[cfg(feature = "c001_a576")]
                                    // "c001_a576" => Some((c001::c001_a576::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a576::KERNEL)),

                                    // #[cfg(feature = "c001_a577")]
                                    // "c001_a577" => Some((c001::c001_a577::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a577::KERNEL)),

                                    // #[cfg(feature = "c001_a578")]
                                    // "c001_a578" => Some((c001::c001_a578::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a578::KERNEL)),

                                    // #[cfg(feature = "c001_a579")]
                                    // "c001_a579" => Some((c001::c001_a579::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a579::KERNEL)),

                                    // #[cfg(feature = "c001_a580")]
                                    // "c001_a580" => Some((c001::c001_a580::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a580::KERNEL)),

                                    // #[cfg(feature = "c001_a581")]
                                    // "c001_a581" => Some((c001::c001_a581::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a581::KERNEL)),

                                    // #[cfg(feature = "c001_a582")]
                                    // "c001_a582" => Some((c001::c001_a582::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a582::KERNEL)),

                                    // #[cfg(feature = "c001_a583")]
                                    // "c001_a583" => Some((c001::c001_a583::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a583::KERNEL)),

                                    // #[cfg(feature = "c001_a584")]
                                    // "c001_a584" => Some((c001::c001_a584::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a584::KERNEL)),

                                    // #[cfg(feature = "c001_a585")]
                                    // "c001_a585" => Some((c001::c001_a585::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a585::KERNEL)),

                                    // #[cfg(feature = "c001_a586")]
                                    // "c001_a586" => Some((c001::c001_a586::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a586::KERNEL)),

                                    // #[cfg(feature = "c001_a587")]
                                    // "c001_a587" => Some((c001::c001_a587::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a587::KERNEL)),

                                    // #[cfg(feature = "c001_a588")]
                                    // "c001_a588" => Some((c001::c001_a588::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a588::KERNEL)),

                                    // #[cfg(feature = "c001_a589")]
                                    // "c001_a589" => Some((c001::c001_a589::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a589::KERNEL)),

                                    // #[cfg(feature = "c001_a590")]
                                    // "c001_a590" => Some((c001::c001_a590::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a590::KERNEL)),

                                    // #[cfg(feature = "c001_a591")]
                                    // "c001_a591" => Some((c001::c001_a591::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a591::KERNEL)),

                                    // #[cfg(feature = "c001_a592")]
                                    // "c001_a592" => Some((c001::c001_a592::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a592::KERNEL)),

                                    // #[cfg(feature = "c001_a593")]
                                    // "c001_a593" => Some((c001::c001_a593::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a593::KERNEL)),

                                    // #[cfg(feature = "c001_a594")]
                                    // "c001_a594" => Some((c001::c001_a594::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a594::KERNEL)),

                                    // #[cfg(feature = "c001_a595")]
                                    // "c001_a595" => Some((c001::c001_a595::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a595::KERNEL)),

                                    // #[cfg(feature = "c001_a596")]
                                    // "c001_a596" => Some((c001::c001_a596::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a596::KERNEL)),

                                    // #[cfg(feature = "c001_a597")]
                                    // "c001_a597" => Some((c001::c001_a597::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a597::KERNEL)),

                                    // #[cfg(feature = "c001_a598")]
                                    // "c001_a598" => Some((c001::c001_a598::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a598::KERNEL)),

                                    // #[cfg(feature = "c001_a599")]
                                    // "c001_a599" => Some((c001::c001_a599::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a599::KERNEL)),

                                    // #[cfg(feature = "c001_a600")]
                                    // "c001_a600" => Some((c001::c001_a600::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a600::KERNEL)),

                                    // #[cfg(feature = "c001_a601")]
                                    // "c001_a601" => Some((c001::c001_a601::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a601::KERNEL)),

                                    // #[cfg(feature = "c001_a602")]
                                    // "c001_a602" => Some((c001::c001_a602::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a602::KERNEL)),

                                    // #[cfg(feature = "c001_a603")]
                                    // "c001_a603" => Some((c001::c001_a603::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a603::KERNEL)),

                                    // #[cfg(feature = "c001_a604")]
                                    // "c001_a604" => Some((c001::c001_a604::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a604::KERNEL)),

                                    // #[cfg(feature = "c001_a605")]
                                    // "c001_a605" => Some((c001::c001_a605::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a605::KERNEL)),

                                    // #[cfg(feature = "c001_a606")]
                                    // "c001_a606" => Some((c001::c001_a606::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a606::KERNEL)),

                                    // #[cfg(feature = "c001_a607")]
                                    // "c001_a607" => Some((c001::c001_a607::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a607::KERNEL)),

                                    // #[cfg(feature = "c001_a608")]
                                    // "c001_a608" => Some((c001::c001_a608::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a608::KERNEL)),

                                    // #[cfg(feature = "c001_a609")]
                                    // "c001_a609" => Some((c001::c001_a609::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a609::KERNEL)),

                                    // #[cfg(feature = "c001_a610")]
                                    // "c001_a610" => Some((c001::c001_a610::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a610::KERNEL)),

                                    // #[cfg(feature = "c001_a611")]
                                    // "c001_a611" => Some((c001::c001_a611::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a611::KERNEL)),

                                    // #[cfg(feature = "c001_a612")]
                                    // "c001_a612" => Some((c001::c001_a612::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a612::KERNEL)),

                                    // #[cfg(feature = "c001_a613")]
                                    // "c001_a613" => Some((c001::c001_a613::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a613::KERNEL)),

                                    // #[cfg(feature = "c001_a614")]
                                    // "c001_a614" => Some((c001::c001_a614::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a614::KERNEL)),

                                    // #[cfg(feature = "c001_a615")]
                                    // "c001_a615" => Some((c001::c001_a615::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a615::KERNEL)),

                                    // #[cfg(feature = "c001_a616")]
                                    // "c001_a616" => Some((c001::c001_a616::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a616::KERNEL)),

                                    // #[cfg(feature = "c001_a617")]
                                    // "c001_a617" => Some((c001::c001_a617::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a617::KERNEL)),

                                    // #[cfg(feature = "c001_a618")]
                                    // "c001_a618" => Some((c001::c001_a618::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a618::KERNEL)),

                                    // #[cfg(feature = "c001_a619")]
                                    // "c001_a619" => Some((c001::c001_a619::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a619::KERNEL)),

                                    // #[cfg(feature = "c001_a620")]
                                    // "c001_a620" => Some((c001::c001_a620::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a620::KERNEL)),

                                    // #[cfg(feature = "c001_a621")]
                                    // "c001_a621" => Some((c001::c001_a621::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a621::KERNEL)),

                                    // #[cfg(feature = "c001_a622")]
                                    // "c001_a622" => Some((c001::c001_a622::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a622::KERNEL)),

                                    // #[cfg(feature = "c001_a623")]
                                    // "c001_a623" => Some((c001::c001_a623::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a623::KERNEL)),

                                    // #[cfg(feature = "c001_a624")]
                                    // "c001_a624" => Some((c001::c001_a624::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a624::KERNEL)),

                                    // #[cfg(feature = "c001_a625")]
                                    // "c001_a625" => Some((c001::c001_a625::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a625::KERNEL)),

                                    // #[cfg(feature = "c001_a626")]
                                    // "c001_a626" => Some((c001::c001_a626::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a626::KERNEL)),

                                    // #[cfg(feature = "c001_a627")]
                                    // "c001_a627" => Some((c001::c001_a627::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a627::KERNEL)),

                                    // #[cfg(feature = "c001_a628")]
                                    // "c001_a628" => Some((c001::c001_a628::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a628::KERNEL)),

                                    // #[cfg(feature = "c001_a629")]
                                    // "c001_a629" => Some((c001::c001_a629::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a629::KERNEL)),

                                    // #[cfg(feature = "c001_a630")]
                                    // "c001_a630" => Some((c001::c001_a630::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a630::KERNEL)),

                                    // #[cfg(feature = "c001_a631")]
                                    // "c001_a631" => Some((c001::c001_a631::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a631::KERNEL)),

                                    // #[cfg(feature = "c001_a632")]
                                    // "c001_a632" => Some((c001::c001_a632::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a632::KERNEL)),

                                    // #[cfg(feature = "c001_a633")]
                                    // "c001_a633" => Some((c001::c001_a633::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a633::KERNEL)),

                                    // #[cfg(feature = "c001_a634")]
                                    // "c001_a634" => Some((c001::c001_a634::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a634::KERNEL)),

                                    // #[cfg(feature = "c001_a635")]
                                    // "c001_a635" => Some((c001::c001_a635::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a635::KERNEL)),

                                    // #[cfg(feature = "c001_a636")]
                                    // "c001_a636" => Some((c001::c001_a636::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a636::KERNEL)),

                                    // #[cfg(feature = "c001_a637")]
                                    // "c001_a637" => Some((c001::c001_a637::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a637::KERNEL)),

                                    // #[cfg(feature = "c001_a638")]
                                    // "c001_a638" => Some((c001::c001_a638::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a638::KERNEL)),

                                    // #[cfg(feature = "c001_a639")]
                                    // "c001_a639" => Some((c001::c001_a639::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a639::KERNEL)),

                                    // #[cfg(feature = "c001_a640")]
                                    // "c001_a640" => Some((c001::c001_a640::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a640::KERNEL)),

                                    // #[cfg(feature = "c001_a641")]
                                    // "c001_a641" => Some((c001::c001_a641::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a641::KERNEL)),

                                    // #[cfg(feature = "c001_a642")]
                                    // "c001_a642" => Some((c001::c001_a642::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a642::KERNEL)),

                                    // #[cfg(feature = "c001_a643")]
                                    // "c001_a643" => Some((c001::c001_a643::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a643::KERNEL)),

                                    // #[cfg(feature = "c001_a644")]
                                    // "c001_a644" => Some((c001::c001_a644::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a644::KERNEL)),

                                    // #[cfg(feature = "c001_a645")]
                                    // "c001_a645" => Some((c001::c001_a645::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a645::KERNEL)),

                                    // #[cfg(feature = "c001_a646")]
                                    // "c001_a646" => Some((c001::c001_a646::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a646::KERNEL)),

                                    // #[cfg(feature = "c001_a647")]
                                    // "c001_a647" => Some((c001::c001_a647::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a647::KERNEL)),

                                    // #[cfg(feature = "c001_a648")]
                                    // "c001_a648" => Some((c001::c001_a648::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a648::KERNEL)),

                                    // #[cfg(feature = "c001_a649")]
                                    // "c001_a649" => Some((c001::c001_a649::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a649::KERNEL)),

                                    // #[cfg(feature = "c001_a650")]
                                    // "c001_a650" => Some((c001::c001_a650::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a650::KERNEL)),

                                    // #[cfg(feature = "c001_a651")]
                                    // "c001_a651" => Some((c001::c001_a651::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a651::KERNEL)),

                                    // #[cfg(feature = "c001_a652")]
                                    // "c001_a652" => Some((c001::c001_a652::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a652::KERNEL)),

                                    // #[cfg(feature = "c001_a653")]
                                    // "c001_a653" => Some((c001::c001_a653::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a653::KERNEL)),

                                    // #[cfg(feature = "c001_a654")]
                                    // "c001_a654" => Some((c001::c001_a654::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a654::KERNEL)),

                                    // #[cfg(feature = "c001_a655")]
                                    // "c001_a655" => Some((c001::c001_a655::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a655::KERNEL)),

                                    // #[cfg(feature = "c001_a656")]
                                    // "c001_a656" => Some((c001::c001_a656::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a656::KERNEL)),

                                    // #[cfg(feature = "c001_a657")]
                                    // "c001_a657" => Some((c001::c001_a657::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a657::KERNEL)),

                                    // #[cfg(feature = "c001_a658")]
                                    // "c001_a658" => Some((c001::c001_a658::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a658::KERNEL)),

                                    // #[cfg(feature = "c001_a659")]
                                    // "c001_a659" => Some((c001::c001_a659::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a659::KERNEL)),

                                    // #[cfg(feature = "c001_a660")]
                                    // "c001_a660" => Some((c001::c001_a660::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a660::KERNEL)),

                                    // #[cfg(feature = "c001_a661")]
                                    // "c001_a661" => Some((c001::c001_a661::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a661::KERNEL)),

                                    // #[cfg(feature = "c001_a662")]
                                    // "c001_a662" => Some((c001::c001_a662::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a662::KERNEL)),

                                    // #[cfg(feature = "c001_a663")]
                                    // "c001_a663" => Some((c001::c001_a663::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a663::KERNEL)),

                                    // #[cfg(feature = "c001_a664")]
                                    // "c001_a664" => Some((c001::c001_a664::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a664::KERNEL)),

                                    // #[cfg(feature = "c001_a665")]
                                    // "c001_a665" => Some((c001::c001_a665::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a665::KERNEL)),

                                    // #[cfg(feature = "c001_a666")]
                                    // "c001_a666" => Some((c001::c001_a666::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a666::KERNEL)),

                                    // #[cfg(feature = "c001_a667")]
                                    // "c001_a667" => Some((c001::c001_a667::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a667::KERNEL)),

                                    // #[cfg(feature = "c001_a668")]
                                    // "c001_a668" => Some((c001::c001_a668::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a668::KERNEL)),

                                    // #[cfg(feature = "c001_a669")]
                                    // "c001_a669" => Some((c001::c001_a669::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a669::KERNEL)),

                                    // #[cfg(feature = "c001_a670")]
                                    // "c001_a670" => Some((c001::c001_a670::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a670::KERNEL)),

                                    // #[cfg(feature = "c001_a671")]
                                    // "c001_a671" => Some((c001::c001_a671::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a671::KERNEL)),

                                    // #[cfg(feature = "c001_a672")]
                                    // "c001_a672" => Some((c001::c001_a672::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a672::KERNEL)),

                                    // #[cfg(feature = "c001_a673")]
                                    // "c001_a673" => Some((c001::c001_a673::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a673::KERNEL)),

                                    // #[cfg(feature = "c001_a674")]
                                    // "c001_a674" => Some((c001::c001_a674::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a674::KERNEL)),

                                    // #[cfg(feature = "c001_a675")]
                                    // "c001_a675" => Some((c001::c001_a675::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a675::KERNEL)),

                                    // #[cfg(feature = "c001_a676")]
                                    // "c001_a676" => Some((c001::c001_a676::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a676::KERNEL)),

                                    // #[cfg(feature = "c001_a677")]
                                    // "c001_a677" => Some((c001::c001_a677::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a677::KERNEL)),

                                    // #[cfg(feature = "c001_a678")]
                                    // "c001_a678" => Some((c001::c001_a678::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a678::KERNEL)),

                                    // #[cfg(feature = "c001_a679")]
                                    // "c001_a679" => Some((c001::c001_a679::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a679::KERNEL)),

                                    // #[cfg(feature = "c001_a680")]
                                    // "c001_a680" => Some((c001::c001_a680::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a680::KERNEL)),

                                    // #[cfg(feature = "c001_a681")]
                                    // "c001_a681" => Some((c001::c001_a681::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a681::KERNEL)),

                                    // #[cfg(feature = "c001_a682")]
                                    // "c001_a682" => Some((c001::c001_a682::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a682::KERNEL)),

                                    // #[cfg(feature = "c001_a683")]
                                    // "c001_a683" => Some((c001::c001_a683::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a683::KERNEL)),

                                    // #[cfg(feature = "c001_a684")]
                                    // "c001_a684" => Some((c001::c001_a684::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a684::KERNEL)),

                                    // #[cfg(feature = "c001_a685")]
                                    // "c001_a685" => Some((c001::c001_a685::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a685::KERNEL)),

                                    // #[cfg(feature = "c001_a686")]
                                    // "c001_a686" => Some((c001::c001_a686::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a686::KERNEL)),

                                    // #[cfg(feature = "c001_a687")]
                                    // "c001_a687" => Some((c001::c001_a687::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a687::KERNEL)),

                                    // #[cfg(feature = "c001_a688")]
                                    // "c001_a688" => Some((c001::c001_a688::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a688::KERNEL)),

                                    // #[cfg(feature = "c001_a689")]
                                    // "c001_a689" => Some((c001::c001_a689::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a689::KERNEL)),

                                    // #[cfg(feature = "c001_a690")]
                                    // "c001_a690" => Some((c001::c001_a690::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a690::KERNEL)),

                                    // #[cfg(feature = "c001_a691")]
                                    // "c001_a691" => Some((c001::c001_a691::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a691::KERNEL)),

                                    // #[cfg(feature = "c001_a692")]
                                    // "c001_a692" => Some((c001::c001_a692::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a692::KERNEL)),

                                    // #[cfg(feature = "c001_a693")]
                                    // "c001_a693" => Some((c001::c001_a693::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a693::KERNEL)),

                                    // #[cfg(feature = "c001_a694")]
                                    // "c001_a694" => Some((c001::c001_a694::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a694::KERNEL)),

                                    // #[cfg(feature = "c001_a695")]
                                    // "c001_a695" => Some((c001::c001_a695::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a695::KERNEL)),

                                    // #[cfg(feature = "c001_a696")]
                                    // "c001_a696" => Some((c001::c001_a696::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a696::KERNEL)),

                                    // #[cfg(feature = "c001_a697")]
                                    // "c001_a697" => Some((c001::c001_a697::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a697::KERNEL)),

                                    // #[cfg(feature = "c001_a698")]
                                    // "c001_a698" => Some((c001::c001_a698::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a698::KERNEL)),

                                    // #[cfg(feature = "c001_a699")]
                                    // "c001_a699" => Some((c001::c001_a699::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a699::KERNEL)),

                                    // #[cfg(feature = "c001_a700")]
                                    // "c001_a700" => Some((c001::c001_a700::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a700::KERNEL)),

                                    // #[cfg(feature = "c001_a701")]
                                    // "c001_a701" => Some((c001::c001_a701::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a701::KERNEL)),

                                    // #[cfg(feature = "c001_a702")]
                                    // "c001_a702" => Some((c001::c001_a702::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a702::KERNEL)),

                                    // #[cfg(feature = "c001_a703")]
                                    // "c001_a703" => Some((c001::c001_a703::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a703::KERNEL)),

                                    // #[cfg(feature = "c001_a704")]
                                    // "c001_a704" => Some((c001::c001_a704::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a704::KERNEL)),

                                    // #[cfg(feature = "c001_a705")]
                                    // "c001_a705" => Some((c001::c001_a705::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a705::KERNEL)),

                                    // #[cfg(feature = "c001_a706")]
                                    // "c001_a706" => Some((c001::c001_a706::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a706::KERNEL)),

                                    // #[cfg(feature = "c001_a707")]
                                    // "c001_a707" => Some((c001::c001_a707::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a707::KERNEL)),

                                    // #[cfg(feature = "c001_a708")]
                                    // "c001_a708" => Some((c001::c001_a708::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a708::KERNEL)),

                                    // #[cfg(feature = "c001_a709")]
                                    // "c001_a709" => Some((c001::c001_a709::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a709::KERNEL)),

                                    // #[cfg(feature = "c001_a710")]
                                    // "c001_a710" => Some((c001::c001_a710::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a710::KERNEL)),

                                    // #[cfg(feature = "c001_a711")]
                                    // "c001_a711" => Some((c001::c001_a711::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a711::KERNEL)),

                                    // #[cfg(feature = "c001_a712")]
                                    // "c001_a712" => Some((c001::c001_a712::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a712::KERNEL)),

                                    // #[cfg(feature = "c001_a713")]
                                    // "c001_a713" => Some((c001::c001_a713::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a713::KERNEL)),

                                    // #[cfg(feature = "c001_a714")]
                                    // "c001_a714" => Some((c001::c001_a714::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a714::KERNEL)),

                                    // #[cfg(feature = "c001_a715")]
                                    // "c001_a715" => Some((c001::c001_a715::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a715::KERNEL)),

                                    // #[cfg(feature = "c001_a716")]
                                    // "c001_a716" => Some((c001::c001_a716::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a716::KERNEL)),

                                    // #[cfg(feature = "c001_a717")]
                                    // "c001_a717" => Some((c001::c001_a717::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a717::KERNEL)),

                                    // #[cfg(feature = "c001_a718")]
                                    // "c001_a718" => Some((c001::c001_a718::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a718::KERNEL)),

                                    // #[cfg(feature = "c001_a719")]
                                    // "c001_a719" => Some((c001::c001_a719::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a719::KERNEL)),

                                    // #[cfg(feature = "c001_a720")]
                                    // "c001_a720" => Some((c001::c001_a720::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a720::KERNEL)),

                                    // #[cfg(feature = "c001_a721")]
                                    // "c001_a721" => Some((c001::c001_a721::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a721::KERNEL)),

                                    // #[cfg(feature = "c001_a722")]
                                    // "c001_a722" => Some((c001::c001_a722::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a722::KERNEL)),

                                    // #[cfg(feature = "c001_a723")]
                                    // "c001_a723" => Some((c001::c001_a723::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a723::KERNEL)),

                                    // #[cfg(feature = "c001_a724")]
                                    // "c001_a724" => Some((c001::c001_a724::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a724::KERNEL)),

                                    // #[cfg(feature = "c001_a725")]
                                    // "c001_a725" => Some((c001::c001_a725::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a725::KERNEL)),

                                    // #[cfg(feature = "c001_a726")]
                                    // "c001_a726" => Some((c001::c001_a726::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a726::KERNEL)),

                                    // #[cfg(feature = "c001_a727")]
                                    // "c001_a727" => Some((c001::c001_a727::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a727::KERNEL)),

                                    // #[cfg(feature = "c001_a728")]
                                    // "c001_a728" => Some((c001::c001_a728::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a728::KERNEL)),

                                    // #[cfg(feature = "c001_a729")]
                                    // "c001_a729" => Some((c001::c001_a729::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a729::KERNEL)),

                                    // #[cfg(feature = "c001_a730")]
                                    // "c001_a730" => Some((c001::c001_a730::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a730::KERNEL)),

                                    // #[cfg(feature = "c001_a731")]
                                    // "c001_a731" => Some((c001::c001_a731::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a731::KERNEL)),

                                    // #[cfg(feature = "c001_a732")]
                                    // "c001_a732" => Some((c001::c001_a732::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a732::KERNEL)),

                                    // #[cfg(feature = "c001_a733")]
                                    // "c001_a733" => Some((c001::c001_a733::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a733::KERNEL)),

                                    // #[cfg(feature = "c001_a734")]
                                    // "c001_a734" => Some((c001::c001_a734::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a734::KERNEL)),

                                    // #[cfg(feature = "c001_a735")]
                                    // "c001_a735" => Some((c001::c001_a735::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a735::KERNEL)),

                                    // #[cfg(feature = "c001_a736")]
                                    // "c001_a736" => Some((c001::c001_a736::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a736::KERNEL)),

                                    // #[cfg(feature = "c001_a737")]
                                    // "c001_a737" => Some((c001::c001_a737::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a737::KERNEL)),

                                    // #[cfg(feature = "c001_a738")]
                                    // "c001_a738" => Some((c001::c001_a738::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a738::KERNEL)),

                                    // #[cfg(feature = "c001_a739")]
                                    // "c001_a739" => Some((c001::c001_a739::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a739::KERNEL)),

                                    // #[cfg(feature = "c001_a740")]
                                    // "c001_a740" => Some((c001::c001_a740::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a740::KERNEL)),

                                    // #[cfg(feature = "c001_a741")]
                                    // "c001_a741" => Some((c001::c001_a741::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a741::KERNEL)),

                                    // #[cfg(feature = "c001_a742")]
                                    // "c001_a742" => Some((c001::c001_a742::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a742::KERNEL)),

                                    // #[cfg(feature = "c001_a743")]
                                    // "c001_a743" => Some((c001::c001_a743::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a743::KERNEL)),

                                    // #[cfg(feature = "c001_a744")]
                                    // "c001_a744" => Some((c001::c001_a744::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a744::KERNEL)),

                                    // #[cfg(feature = "c001_a745")]
                                    // "c001_a745" => Some((c001::c001_a745::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a745::KERNEL)),

                                    // #[cfg(feature = "c001_a746")]
                                    // "c001_a746" => Some((c001::c001_a746::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a746::KERNEL)),

                                    // #[cfg(feature = "c001_a747")]
                                    // "c001_a747" => Some((c001::c001_a747::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a747::KERNEL)),

                                    // #[cfg(feature = "c001_a748")]
                                    // "c001_a748" => Some((c001::c001_a748::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a748::KERNEL)),

                                    // #[cfg(feature = "c001_a749")]
                                    // "c001_a749" => Some((c001::c001_a749::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a749::KERNEL)),

                                    // #[cfg(feature = "c001_a750")]
                                    // "c001_a750" => Some((c001::c001_a750::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a750::KERNEL)),

                                    // #[cfg(feature = "c001_a751")]
                                    // "c001_a751" => Some((c001::c001_a751::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a751::KERNEL)),

                                    // #[cfg(feature = "c001_a752")]
                                    // "c001_a752" => Some((c001::c001_a752::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a752::KERNEL)),

                                    // #[cfg(feature = "c001_a753")]
                                    // "c001_a753" => Some((c001::c001_a753::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a753::KERNEL)),

                                    // #[cfg(feature = "c001_a754")]
                                    // "c001_a754" => Some((c001::c001_a754::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a754::KERNEL)),

                                    // #[cfg(feature = "c001_a755")]
                                    // "c001_a755" => Some((c001::c001_a755::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a755::KERNEL)),

                                    // #[cfg(feature = "c001_a756")]
                                    // "c001_a756" => Some((c001::c001_a756::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a756::KERNEL)),

                                    // #[cfg(feature = "c001_a757")]
                                    // "c001_a757" => Some((c001::c001_a757::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a757::KERNEL)),

                                    // #[cfg(feature = "c001_a758")]
                                    // "c001_a758" => Some((c001::c001_a758::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a758::KERNEL)),

                                    // #[cfg(feature = "c001_a759")]
                                    // "c001_a759" => Some((c001::c001_a759::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a759::KERNEL)),

                                    // #[cfg(feature = "c001_a760")]
                                    // "c001_a760" => Some((c001::c001_a760::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a760::KERNEL)),

                                    // #[cfg(feature = "c001_a761")]
                                    // "c001_a761" => Some((c001::c001_a761::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a761::KERNEL)),

                                    // #[cfg(feature = "c001_a762")]
                                    // "c001_a762" => Some((c001::c001_a762::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a762::KERNEL)),

                                    // #[cfg(feature = "c001_a763")]
                                    // "c001_a763" => Some((c001::c001_a763::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a763::KERNEL)),

                                    // #[cfg(feature = "c001_a764")]
                                    // "c001_a764" => Some((c001::c001_a764::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a764::KERNEL)),

                                    // #[cfg(feature = "c001_a765")]
                                    // "c001_a765" => Some((c001::c001_a765::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a765::KERNEL)),

                                    // #[cfg(feature = "c001_a766")]
                                    // "c001_a766" => Some((c001::c001_a766::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a766::KERNEL)),

                                    // #[cfg(feature = "c001_a767")]
                                    // "c001_a767" => Some((c001::c001_a767::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a767::KERNEL)),

                                    // #[cfg(feature = "c001_a768")]
                                    // "c001_a768" => Some((c001::c001_a768::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a768::KERNEL)),

                                    // #[cfg(feature = "c001_a769")]
                                    // "c001_a769" => Some((c001::c001_a769::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a769::KERNEL)),

                                    // #[cfg(feature = "c001_a770")]
                                    // "c001_a770" => Some((c001::c001_a770::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a770::KERNEL)),

                                    // #[cfg(feature = "c001_a771")]
                                    // "c001_a771" => Some((c001::c001_a771::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a771::KERNEL)),

                                    // #[cfg(feature = "c001_a772")]
                                    // "c001_a772" => Some((c001::c001_a772::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a772::KERNEL)),

                                    // #[cfg(feature = "c001_a773")]
                                    // "c001_a773" => Some((c001::c001_a773::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a773::KERNEL)),

                                    // #[cfg(feature = "c001_a774")]
                                    // "c001_a774" => Some((c001::c001_a774::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a774::KERNEL)),

                                    // #[cfg(feature = "c001_a775")]
                                    // "c001_a775" => Some((c001::c001_a775::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a775::KERNEL)),

                                    // #[cfg(feature = "c001_a776")]
                                    // "c001_a776" => Some((c001::c001_a776::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a776::KERNEL)),

                                    // #[cfg(feature = "c001_a777")]
                                    // "c001_a777" => Some((c001::c001_a777::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a777::KERNEL)),

                                    // #[cfg(feature = "c001_a778")]
                                    // "c001_a778" => Some((c001::c001_a778::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a778::KERNEL)),

                                    // #[cfg(feature = "c001_a779")]
                                    // "c001_a779" => Some((c001::c001_a779::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a779::KERNEL)),

                                    // #[cfg(feature = "c001_a780")]
                                    // "c001_a780" => Some((c001::c001_a780::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a780::KERNEL)),

                                    // #[cfg(feature = "c001_a781")]
                                    // "c001_a781" => Some((c001::c001_a781::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a781::KERNEL)),

                                    // #[cfg(feature = "c001_a782")]
                                    // "c001_a782" => Some((c001::c001_a782::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a782::KERNEL)),

                                    // #[cfg(feature = "c001_a783")]
                                    // "c001_a783" => Some((c001::c001_a783::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a783::KERNEL)),

                                    // #[cfg(feature = "c001_a784")]
                                    // "c001_a784" => Some((c001::c001_a784::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a784::KERNEL)),

                                    // #[cfg(feature = "c001_a785")]
                                    // "c001_a785" => Some((c001::c001_a785::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a785::KERNEL)),

                                    // #[cfg(feature = "c001_a786")]
                                    // "c001_a786" => Some((c001::c001_a786::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a786::KERNEL)),

                                    // #[cfg(feature = "c001_a787")]
                                    // "c001_a787" => Some((c001::c001_a787::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a787::KERNEL)),

                                    // #[cfg(feature = "c001_a788")]
                                    // "c001_a788" => Some((c001::c001_a788::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a788::KERNEL)),

                                    // #[cfg(feature = "c001_a789")]
                                    // "c001_a789" => Some((c001::c001_a789::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a789::KERNEL)),

                                    // #[cfg(feature = "c001_a790")]
                                    // "c001_a790" => Some((c001::c001_a790::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a790::KERNEL)),

                                    // #[cfg(feature = "c001_a791")]
                                    // "c001_a791" => Some((c001::c001_a791::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a791::KERNEL)),

                                    // #[cfg(feature = "c001_a792")]
                                    // "c001_a792" => Some((c001::c001_a792::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a792::KERNEL)),

                                    // #[cfg(feature = "c001_a793")]
                                    // "c001_a793" => Some((c001::c001_a793::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a793::KERNEL)),

                                    // #[cfg(feature = "c001_a794")]
                                    // "c001_a794" => Some((c001::c001_a794::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a794::KERNEL)),

                                    // #[cfg(feature = "c001_a795")]
                                    // "c001_a795" => Some((c001::c001_a795::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a795::KERNEL)),

                                    // #[cfg(feature = "c001_a796")]
                                    // "c001_a796" => Some((c001::c001_a796::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a796::KERNEL)),

                                    // #[cfg(feature = "c001_a797")]
                                    // "c001_a797" => Some((c001::c001_a797::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a797::KERNEL)),

                                    // #[cfg(feature = "c001_a798")]
                                    // "c001_a798" => Some((c001::c001_a798::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a798::KERNEL)),

                                    // #[cfg(feature = "c001_a799")]
                                    // "c001_a799" => Some((c001::c001_a799::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a799::KERNEL)),

                                    // #[cfg(feature = "c001_a800")]
                                    // "c001_a800" => Some((c001::c001_a800::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a800::KERNEL)),

                                    // #[cfg(feature = "c001_a801")]
                                    // "c001_a801" => Some((c001::c001_a801::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a801::KERNEL)),

                                    // #[cfg(feature = "c001_a802")]
                                    // "c001_a802" => Some((c001::c001_a802::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a802::KERNEL)),

                                    // #[cfg(feature = "c001_a803")]
                                    // "c001_a803" => Some((c001::c001_a803::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a803::KERNEL)),

                                    // #[cfg(feature = "c001_a804")]
                                    // "c001_a804" => Some((c001::c001_a804::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a804::KERNEL)),

                                    // #[cfg(feature = "c001_a805")]
                                    // "c001_a805" => Some((c001::c001_a805::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a805::KERNEL)),

                                    // #[cfg(feature = "c001_a806")]
                                    // "c001_a806" => Some((c001::c001_a806::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a806::KERNEL)),

                                    // #[cfg(feature = "c001_a807")]
                                    // "c001_a807" => Some((c001::c001_a807::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a807::KERNEL)),

                                    // #[cfg(feature = "c001_a808")]
                                    // "c001_a808" => Some((c001::c001_a808::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a808::KERNEL)),

                                    // #[cfg(feature = "c001_a809")]
                                    // "c001_a809" => Some((c001::c001_a809::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a809::KERNEL)),

                                    // #[cfg(feature = "c001_a810")]
                                    // "c001_a810" => Some((c001::c001_a810::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a810::KERNEL)),

                                    // #[cfg(feature = "c001_a811")]
                                    // "c001_a811" => Some((c001::c001_a811::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a811::KERNEL)),

                                    // #[cfg(feature = "c001_a812")]
                                    // "c001_a812" => Some((c001::c001_a812::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a812::KERNEL)),

                                    // #[cfg(feature = "c001_a813")]
                                    // "c001_a813" => Some((c001::c001_a813::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a813::KERNEL)),

                                    // #[cfg(feature = "c001_a814")]
                                    // "c001_a814" => Some((c001::c001_a814::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a814::KERNEL)),

                                    // #[cfg(feature = "c001_a815")]
                                    // "c001_a815" => Some((c001::c001_a815::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a815::KERNEL)),

                                    // #[cfg(feature = "c001_a816")]
                                    // "c001_a816" => Some((c001::c001_a816::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a816::KERNEL)),

                                    // #[cfg(feature = "c001_a817")]
                                    // "c001_a817" => Some((c001::c001_a817::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a817::KERNEL)),

                                    // #[cfg(feature = "c001_a818")]
                                    // "c001_a818" => Some((c001::c001_a818::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a818::KERNEL)),

                                    // #[cfg(feature = "c001_a819")]
                                    // "c001_a819" => Some((c001::c001_a819::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a819::KERNEL)),

                                    // #[cfg(feature = "c001_a820")]
                                    // "c001_a820" => Some((c001::c001_a820::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a820::KERNEL)),

                                    // #[cfg(feature = "c001_a821")]
                                    // "c001_a821" => Some((c001::c001_a821::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a821::KERNEL)),

                                    // #[cfg(feature = "c001_a822")]
                                    // "c001_a822" => Some((c001::c001_a822::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a822::KERNEL)),

                                    // #[cfg(feature = "c001_a823")]
                                    // "c001_a823" => Some((c001::c001_a823::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a823::KERNEL)),

                                    // #[cfg(feature = "c001_a824")]
                                    // "c001_a824" => Some((c001::c001_a824::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a824::KERNEL)),

                                    // #[cfg(feature = "c001_a825")]
                                    // "c001_a825" => Some((c001::c001_a825::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a825::KERNEL)),

                                    // #[cfg(feature = "c001_a826")]
                                    // "c001_a826" => Some((c001::c001_a826::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a826::KERNEL)),

                                    // #[cfg(feature = "c001_a827")]
                                    // "c001_a827" => Some((c001::c001_a827::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a827::KERNEL)),

                                    // #[cfg(feature = "c001_a828")]
                                    // "c001_a828" => Some((c001::c001_a828::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a828::KERNEL)),

                                    // #[cfg(feature = "c001_a829")]
                                    // "c001_a829" => Some((c001::c001_a829::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a829::KERNEL)),

                                    // #[cfg(feature = "c001_a830")]
                                    // "c001_a830" => Some((c001::c001_a830::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a830::KERNEL)),

                                    // #[cfg(feature = "c001_a831")]
                                    // "c001_a831" => Some((c001::c001_a831::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a831::KERNEL)),

                                    // #[cfg(feature = "c001_a832")]
                                    // "c001_a832" => Some((c001::c001_a832::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a832::KERNEL)),

                                    // #[cfg(feature = "c001_a833")]
                                    // "c001_a833" => Some((c001::c001_a833::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a833::KERNEL)),

                                    // #[cfg(feature = "c001_a834")]
                                    // "c001_a834" => Some((c001::c001_a834::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a834::KERNEL)),

                                    // #[cfg(feature = "c001_a835")]
                                    // "c001_a835" => Some((c001::c001_a835::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a835::KERNEL)),

                                    // #[cfg(feature = "c001_a836")]
                                    // "c001_a836" => Some((c001::c001_a836::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a836::KERNEL)),

                                    // #[cfg(feature = "c001_a837")]
                                    // "c001_a837" => Some((c001::c001_a837::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a837::KERNEL)),

                                    // #[cfg(feature = "c001_a838")]
                                    // "c001_a838" => Some((c001::c001_a838::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a838::KERNEL)),

                                    // #[cfg(feature = "c001_a839")]
                                    // "c001_a839" => Some((c001::c001_a839::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a839::KERNEL)),

                                    // #[cfg(feature = "c001_a840")]
                                    // "c001_a840" => Some((c001::c001_a840::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a840::KERNEL)),

                                    // #[cfg(feature = "c001_a841")]
                                    // "c001_a841" => Some((c001::c001_a841::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a841::KERNEL)),

                                    // #[cfg(feature = "c001_a842")]
                                    // "c001_a842" => Some((c001::c001_a842::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a842::KERNEL)),

                                    // #[cfg(feature = "c001_a843")]
                                    // "c001_a843" => Some((c001::c001_a843::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a843::KERNEL)),

                                    // #[cfg(feature = "c001_a844")]
                                    // "c001_a844" => Some((c001::c001_a844::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a844::KERNEL)),

                                    // #[cfg(feature = "c001_a845")]
                                    // "c001_a845" => Some((c001::c001_a845::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a845::KERNEL)),

                                    // #[cfg(feature = "c001_a846")]
                                    // "c001_a846" => Some((c001::c001_a846::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a846::KERNEL)),

                                    // #[cfg(feature = "c001_a847")]
                                    // "c001_a847" => Some((c001::c001_a847::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a847::KERNEL)),

                                    // #[cfg(feature = "c001_a848")]
                                    // "c001_a848" => Some((c001::c001_a848::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a848::KERNEL)),

                                    // #[cfg(feature = "c001_a849")]
                                    // "c001_a849" => Some((c001::c001_a849::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a849::KERNEL)),

                                    // #[cfg(feature = "c001_a850")]
                                    // "c001_a850" => Some((c001::c001_a850::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a850::KERNEL)),

                                    // #[cfg(feature = "c001_a851")]
                                    // "c001_a851" => Some((c001::c001_a851::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a851::KERNEL)),

                                    // #[cfg(feature = "c001_a852")]
                                    // "c001_a852" => Some((c001::c001_a852::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a852::KERNEL)),

                                    // #[cfg(feature = "c001_a853")]
                                    // "c001_a853" => Some((c001::c001_a853::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a853::KERNEL)),

                                    // #[cfg(feature = "c001_a854")]
                                    // "c001_a854" => Some((c001::c001_a854::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a854::KERNEL)),

                                    // #[cfg(feature = "c001_a855")]
                                    // "c001_a855" => Some((c001::c001_a855::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a855::KERNEL)),

                                    // #[cfg(feature = "c001_a856")]
                                    // "c001_a856" => Some((c001::c001_a856::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a856::KERNEL)),

                                    // #[cfg(feature = "c001_a857")]
                                    // "c001_a857" => Some((c001::c001_a857::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a857::KERNEL)),

                                    // #[cfg(feature = "c001_a858")]
                                    // "c001_a858" => Some((c001::c001_a858::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a858::KERNEL)),

                                    // #[cfg(feature = "c001_a859")]
                                    // "c001_a859" => Some((c001::c001_a859::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a859::KERNEL)),

                                    // #[cfg(feature = "c001_a860")]
                                    // "c001_a860" => Some((c001::c001_a860::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a860::KERNEL)),

                                    // #[cfg(feature = "c001_a861")]
                                    // "c001_a861" => Some((c001::c001_a861::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a861::KERNEL)),

                                    // #[cfg(feature = "c001_a862")]
                                    // "c001_a862" => Some((c001::c001_a862::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a862::KERNEL)),

                                    // #[cfg(feature = "c001_a863")]
                                    // "c001_a863" => Some((c001::c001_a863::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a863::KERNEL)),

                                    // #[cfg(feature = "c001_a864")]
                                    // "c001_a864" => Some((c001::c001_a864::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a864::KERNEL)),

                                    // #[cfg(feature = "c001_a865")]
                                    // "c001_a865" => Some((c001::c001_a865::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a865::KERNEL)),

                                    // #[cfg(feature = "c001_a866")]
                                    // "c001_a866" => Some((c001::c001_a866::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a866::KERNEL)),

                                    // #[cfg(feature = "c001_a867")]
                                    // "c001_a867" => Some((c001::c001_a867::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a867::KERNEL)),

                                    // #[cfg(feature = "c001_a868")]
                                    // "c001_a868" => Some((c001::c001_a868::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a868::KERNEL)),

                                    // #[cfg(feature = "c001_a869")]
                                    // "c001_a869" => Some((c001::c001_a869::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a869::KERNEL)),

                                    // #[cfg(feature = "c001_a870")]
                                    // "c001_a870" => Some((c001::c001_a870::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a870::KERNEL)),

                                    // #[cfg(feature = "c001_a871")]
                                    // "c001_a871" => Some((c001::c001_a871::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a871::KERNEL)),

                                    // #[cfg(feature = "c001_a872")]
                                    // "c001_a872" => Some((c001::c001_a872::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a872::KERNEL)),

                                    // #[cfg(feature = "c001_a873")]
                                    // "c001_a873" => Some((c001::c001_a873::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a873::KERNEL)),

                                    // #[cfg(feature = "c001_a874")]
                                    // "c001_a874" => Some((c001::c001_a874::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a874::KERNEL)),

                                    // #[cfg(feature = "c001_a875")]
                                    // "c001_a875" => Some((c001::c001_a875::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a875::KERNEL)),

                                    // #[cfg(feature = "c001_a876")]
                                    // "c001_a876" => Some((c001::c001_a876::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a876::KERNEL)),

                                    // #[cfg(feature = "c001_a877")]
                                    // "c001_a877" => Some((c001::c001_a877::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a877::KERNEL)),

                                    // #[cfg(feature = "c001_a878")]
                                    // "c001_a878" => Some((c001::c001_a878::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a878::KERNEL)),

                                    // #[cfg(feature = "c001_a879")]
                                    // "c001_a879" => Some((c001::c001_a879::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a879::KERNEL)),

                                    // #[cfg(feature = "c001_a880")]
                                    // "c001_a880" => Some((c001::c001_a880::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a880::KERNEL)),

                                    // #[cfg(feature = "c001_a881")]
                                    // "c001_a881" => Some((c001::c001_a881::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a881::KERNEL)),

                                    // #[cfg(feature = "c001_a882")]
                                    // "c001_a882" => Some((c001::c001_a882::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a882::KERNEL)),

                                    // #[cfg(feature = "c001_a883")]
                                    // "c001_a883" => Some((c001::c001_a883::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a883::KERNEL)),

                                    // #[cfg(feature = "c001_a884")]
                                    // "c001_a884" => Some((c001::c001_a884::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a884::KERNEL)),

                                    // #[cfg(feature = "c001_a885")]
                                    // "c001_a885" => Some((c001::c001_a885::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a885::KERNEL)),

                                    // #[cfg(feature = "c001_a886")]
                                    // "c001_a886" => Some((c001::c001_a886::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a886::KERNEL)),

                                    // #[cfg(feature = "c001_a887")]
                                    // "c001_a887" => Some((c001::c001_a887::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a887::KERNEL)),

                                    // #[cfg(feature = "c001_a888")]
                                    // "c001_a888" => Some((c001::c001_a888::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a888::KERNEL)),

                                    // #[cfg(feature = "c001_a889")]
                                    // "c001_a889" => Some((c001::c001_a889::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a889::KERNEL)),

                                    // #[cfg(feature = "c001_a890")]
                                    // "c001_a890" => Some((c001::c001_a890::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a890::KERNEL)),

                                    // #[cfg(feature = "c001_a891")]
                                    // "c001_a891" => Some((c001::c001_a891::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a891::KERNEL)),

                                    // #[cfg(feature = "c001_a892")]
                                    // "c001_a892" => Some((c001::c001_a892::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a892::KERNEL)),

                                    // #[cfg(feature = "c001_a893")]
                                    // "c001_a893" => Some((c001::c001_a893::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a893::KERNEL)),

                                    // #[cfg(feature = "c001_a894")]
                                    // "c001_a894" => Some((c001::c001_a894::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a894::KERNEL)),

                                    // #[cfg(feature = "c001_a895")]
                                    // "c001_a895" => Some((c001::c001_a895::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a895::KERNEL)),

                                    // #[cfg(feature = "c001_a896")]
                                    // "c001_a896" => Some((c001::c001_a896::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a896::KERNEL)),

                                    // #[cfg(feature = "c001_a897")]
                                    // "c001_a897" => Some((c001::c001_a897::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a897::KERNEL)),

                                    // #[cfg(feature = "c001_a898")]
                                    // "c001_a898" => Some((c001::c001_a898::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a898::KERNEL)),

                                    // #[cfg(feature = "c001_a899")]
                                    // "c001_a899" => Some((c001::c001_a899::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a899::KERNEL)),

                                    // #[cfg(feature = "c001_a900")]
                                    // "c001_a900" => Some((c001::c001_a900::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a900::KERNEL)),

                                    // #[cfg(feature = "c001_a901")]
                                    // "c001_a901" => Some((c001::c001_a901::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a901::KERNEL)),

                                    // #[cfg(feature = "c001_a902")]
                                    // "c001_a902" => Some((c001::c001_a902::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a902::KERNEL)),

                                    // #[cfg(feature = "c001_a903")]
                                    // "c001_a903" => Some((c001::c001_a903::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a903::KERNEL)),

                                    // #[cfg(feature = "c001_a904")]
                                    // "c001_a904" => Some((c001::c001_a904::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a904::KERNEL)),

                                    // #[cfg(feature = "c001_a905")]
                                    // "c001_a905" => Some((c001::c001_a905::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a905::KERNEL)),

                                    // #[cfg(feature = "c001_a906")]
                                    // "c001_a906" => Some((c001::c001_a906::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a906::KERNEL)),

                                    // #[cfg(feature = "c001_a907")]
                                    // "c001_a907" => Some((c001::c001_a907::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a907::KERNEL)),

                                    // #[cfg(feature = "c001_a908")]
                                    // "c001_a908" => Some((c001::c001_a908::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a908::KERNEL)),

                                    // #[cfg(feature = "c001_a909")]
                                    // "c001_a909" => Some((c001::c001_a909::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a909::KERNEL)),

                                    // #[cfg(feature = "c001_a910")]
                                    // "c001_a910" => Some((c001::c001_a910::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a910::KERNEL)),

                                    // #[cfg(feature = "c001_a911")]
                                    // "c001_a911" => Some((c001::c001_a911::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a911::KERNEL)),

                                    // #[cfg(feature = "c001_a912")]
                                    // "c001_a912" => Some((c001::c001_a912::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a912::KERNEL)),

                                    // #[cfg(feature = "c001_a913")]
                                    // "c001_a913" => Some((c001::c001_a913::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a913::KERNEL)),

                                    // #[cfg(feature = "c001_a914")]
                                    // "c001_a914" => Some((c001::c001_a914::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a914::KERNEL)),

                                    // #[cfg(feature = "c001_a915")]
                                    // "c001_a915" => Some((c001::c001_a915::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a915::KERNEL)),

                                    // #[cfg(feature = "c001_a916")]
                                    // "c001_a916" => Some((c001::c001_a916::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a916::KERNEL)),

                                    // #[cfg(feature = "c001_a917")]
                                    // "c001_a917" => Some((c001::c001_a917::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a917::KERNEL)),

                                    // #[cfg(feature = "c001_a918")]
                                    // "c001_a918" => Some((c001::c001_a918::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a918::KERNEL)),

                                    // #[cfg(feature = "c001_a919")]
                                    // "c001_a919" => Some((c001::c001_a919::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a919::KERNEL)),

                                    // #[cfg(feature = "c001_a920")]
                                    // "c001_a920" => Some((c001::c001_a920::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a920::KERNEL)),

                                    // #[cfg(feature = "c001_a921")]
                                    // "c001_a921" => Some((c001::c001_a921::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a921::KERNEL)),

                                    // #[cfg(feature = "c001_a922")]
                                    // "c001_a922" => Some((c001::c001_a922::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a922::KERNEL)),

                                    // #[cfg(feature = "c001_a923")]
                                    // "c001_a923" => Some((c001::c001_a923::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a923::KERNEL)),

                                    // #[cfg(feature = "c001_a924")]
                                    // "c001_a924" => Some((c001::c001_a924::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a924::KERNEL)),

                                    // #[cfg(feature = "c001_a925")]
                                    // "c001_a925" => Some((c001::c001_a925::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a925::KERNEL)),

                                    // #[cfg(feature = "c001_a926")]
                                    // "c001_a926" => Some((c001::c001_a926::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a926::KERNEL)),

                                    // #[cfg(feature = "c001_a927")]
                                    // "c001_a927" => Some((c001::c001_a927::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a927::KERNEL)),

                                    // #[cfg(feature = "c001_a928")]
                                    // "c001_a928" => Some((c001::c001_a928::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a928::KERNEL)),

                                    // #[cfg(feature = "c001_a929")]
                                    // "c001_a929" => Some((c001::c001_a929::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a929::KERNEL)),

                                    // #[cfg(feature = "c001_a930")]
                                    // "c001_a930" => Some((c001::c001_a930::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a930::KERNEL)),

                                    // #[cfg(feature = "c001_a931")]
                                    // "c001_a931" => Some((c001::c001_a931::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a931::KERNEL)),

                                    // #[cfg(feature = "c001_a932")]
                                    // "c001_a932" => Some((c001::c001_a932::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a932::KERNEL)),

                                    // #[cfg(feature = "c001_a933")]
                                    // "c001_a933" => Some((c001::c001_a933::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a933::KERNEL)),

                                    // #[cfg(feature = "c001_a934")]
                                    // "c001_a934" => Some((c001::c001_a934::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a934::KERNEL)),

                                    // #[cfg(feature = "c001_a935")]
                                    // "c001_a935" => Some((c001::c001_a935::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a935::KERNEL)),

                                    // #[cfg(feature = "c001_a936")]
                                    // "c001_a936" => Some((c001::c001_a936::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a936::KERNEL)),

                                    // #[cfg(feature = "c001_a937")]
                                    // "c001_a937" => Some((c001::c001_a937::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a937::KERNEL)),

                                    // #[cfg(feature = "c001_a938")]
                                    // "c001_a938" => Some((c001::c001_a938::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a938::KERNEL)),

                                    // #[cfg(feature = "c001_a939")]
                                    // "c001_a939" => Some((c001::c001_a939::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a939::KERNEL)),

                                    // #[cfg(feature = "c001_a940")]
                                    // "c001_a940" => Some((c001::c001_a940::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a940::KERNEL)),

                                    // #[cfg(feature = "c001_a941")]
                                    // "c001_a941" => Some((c001::c001_a941::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a941::KERNEL)),

                                    // #[cfg(feature = "c001_a942")]
                                    // "c001_a942" => Some((c001::c001_a942::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a942::KERNEL)),

                                    // #[cfg(feature = "c001_a943")]
                                    // "c001_a943" => Some((c001::c001_a943::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a943::KERNEL)),

                                    // #[cfg(feature = "c001_a944")]
                                    // "c001_a944" => Some((c001::c001_a944::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a944::KERNEL)),

                                    // #[cfg(feature = "c001_a945")]
                                    // "c001_a945" => Some((c001::c001_a945::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a945::KERNEL)),

                                    // #[cfg(feature = "c001_a946")]
                                    // "c001_a946" => Some((c001::c001_a946::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a946::KERNEL)),

                                    // #[cfg(feature = "c001_a947")]
                                    // "c001_a947" => Some((c001::c001_a947::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a947::KERNEL)),

                                    // #[cfg(feature = "c001_a948")]
                                    // "c001_a948" => Some((c001::c001_a948::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a948::KERNEL)),

                                    // #[cfg(feature = "c001_a949")]
                                    // "c001_a949" => Some((c001::c001_a949::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a949::KERNEL)),

                                    // #[cfg(feature = "c001_a950")]
                                    // "c001_a950" => Some((c001::c001_a950::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a950::KERNEL)),

                                    // #[cfg(feature = "c001_a951")]
                                    // "c001_a951" => Some((c001::c001_a951::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a951::KERNEL)),

                                    // #[cfg(feature = "c001_a952")]
                                    // "c001_a952" => Some((c001::c001_a952::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a952::KERNEL)),

                                    // #[cfg(feature = "c001_a953")]
                                    // "c001_a953" => Some((c001::c001_a953::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a953::KERNEL)),

                                    // #[cfg(feature = "c001_a954")]
                                    // "c001_a954" => Some((c001::c001_a954::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a954::KERNEL)),

                                    // #[cfg(feature = "c001_a955")]
                                    // "c001_a955" => Some((c001::c001_a955::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a955::KERNEL)),

                                    // #[cfg(feature = "c001_a956")]
                                    // "c001_a956" => Some((c001::c001_a956::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a956::KERNEL)),

                                    // #[cfg(feature = "c001_a957")]
                                    // "c001_a957" => Some((c001::c001_a957::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a957::KERNEL)),

                                    // #[cfg(feature = "c001_a958")]
                                    // "c001_a958" => Some((c001::c001_a958::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a958::KERNEL)),

                                    // #[cfg(feature = "c001_a959")]
                                    // "c001_a959" => Some((c001::c001_a959::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a959::KERNEL)),

                                    // #[cfg(feature = "c001_a960")]
                                    // "c001_a960" => Some((c001::c001_a960::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a960::KERNEL)),

                                    // #[cfg(feature = "c001_a961")]
                                    // "c001_a961" => Some((c001::c001_a961::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a961::KERNEL)),

                                    // #[cfg(feature = "c001_a962")]
                                    // "c001_a962" => Some((c001::c001_a962::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a962::KERNEL)),

                                    // #[cfg(feature = "c001_a963")]
                                    // "c001_a963" => Some((c001::c001_a963::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a963::KERNEL)),

                                    // #[cfg(feature = "c001_a964")]
                                    // "c001_a964" => Some((c001::c001_a964::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a964::KERNEL)),

                                    // #[cfg(feature = "c001_a965")]
                                    // "c001_a965" => Some((c001::c001_a965::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a965::KERNEL)),

                                    // #[cfg(feature = "c001_a966")]
                                    // "c001_a966" => Some((c001::c001_a966::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a966::KERNEL)),

                                    // #[cfg(feature = "c001_a967")]
                                    // "c001_a967" => Some((c001::c001_a967::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a967::KERNEL)),

                                    // #[cfg(feature = "c001_a968")]
                                    // "c001_a968" => Some((c001::c001_a968::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a968::KERNEL)),

                                    // #[cfg(feature = "c001_a969")]
                                    // "c001_a969" => Some((c001::c001_a969::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a969::KERNEL)),

                                    // #[cfg(feature = "c001_a970")]
                                    // "c001_a970" => Some((c001::c001_a970::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a970::KERNEL)),

                                    // #[cfg(feature = "c001_a971")]
                                    // "c001_a971" => Some((c001::c001_a971::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a971::KERNEL)),

                                    // #[cfg(feature = "c001_a972")]
                                    // "c001_a972" => Some((c001::c001_a972::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a972::KERNEL)),

                                    // #[cfg(feature = "c001_a973")]
                                    // "c001_a973" => Some((c001::c001_a973::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a973::KERNEL)),

                                    // #[cfg(feature = "c001_a974")]
                                    // "c001_a974" => Some((c001::c001_a974::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a974::KERNEL)),

                                    // #[cfg(feature = "c001_a975")]
                                    // "c001_a975" => Some((c001::c001_a975::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a975::KERNEL)),

                                    // #[cfg(feature = "c001_a976")]
                                    // "c001_a976" => Some((c001::c001_a976::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a976::KERNEL)),

                                    // #[cfg(feature = "c001_a977")]
                                    // "c001_a977" => Some((c001::c001_a977::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a977::KERNEL)),

                                    // #[cfg(feature = "c001_a978")]
                                    // "c001_a978" => Some((c001::c001_a978::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a978::KERNEL)),

                                    // #[cfg(feature = "c001_a979")]
                                    // "c001_a979" => Some((c001::c001_a979::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a979::KERNEL)),

                                    // #[cfg(feature = "c001_a980")]
                                    // "c001_a980" => Some((c001::c001_a980::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a980::KERNEL)),

                                    // #[cfg(feature = "c001_a981")]
                                    // "c001_a981" => Some((c001::c001_a981::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a981::KERNEL)),

                                    // #[cfg(feature = "c001_a982")]
                                    // "c001_a982" => Some((c001::c001_a982::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a982::KERNEL)),

                                    // #[cfg(feature = "c001_a983")]
                                    // "c001_a983" => Some((c001::c001_a983::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a983::KERNEL)),

                                    // #[cfg(feature = "c001_a984")]
                                    // "c001_a984" => Some((c001::c001_a984::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a984::KERNEL)),

                                    // #[cfg(feature = "c001_a985")]
                                    // "c001_a985" => Some((c001::c001_a985::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a985::KERNEL)),

                                    // #[cfg(feature = "c001_a986")]
                                    // "c001_a986" => Some((c001::c001_a986::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a986::KERNEL)),

                                    // #[cfg(feature = "c001_a987")]
                                    // "c001_a987" => Some((c001::c001_a987::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a987::KERNEL)),

                                    // #[cfg(feature = "c001_a988")]
                                    // "c001_a988" => Some((c001::c001_a988::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a988::KERNEL)),

                                    // #[cfg(feature = "c001_a989")]
                                    // "c001_a989" => Some((c001::c001_a989::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a989::KERNEL)),

                                    // #[cfg(feature = "c001_a990")]
                                    // "c001_a990" => Some((c001::c001_a990::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a990::KERNEL)),

                                    // #[cfg(feature = "c001_a991")]
                                    // "c001_a991" => Some((c001::c001_a991::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a991::KERNEL)),

                                    // #[cfg(feature = "c001_a992")]
                                    // "c001_a992" => Some((c001::c001_a992::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a992::KERNEL)),

                                    // #[cfg(feature = "c001_a993")]
                                    // "c001_a993" => Some((c001::c001_a993::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a993::KERNEL)),

                                    // #[cfg(feature = "c001_a994")]
                                    // "c001_a994" => Some((c001::c001_a994::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a994::KERNEL)),

                                    // #[cfg(feature = "c001_a995")]
                                    // "c001_a995" => Some((c001::c001_a995::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a995::KERNEL)),

                                    // #[cfg(feature = "c001_a996")]
                                    // "c001_a996" => Some((c001::c001_a996::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a996::KERNEL)),

                                    // #[cfg(feature = "c001_a997")]
                                    // "c001_a997" => Some((c001::c001_a997::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a997::KERNEL)),

                                    // #[cfg(feature = "c001_a998")]
                                    // "c001_a998" => Some((c001::c001_a998::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a998::KERNEL)),

                                    // #[cfg(feature = "c001_a999")]
                                    // "c001_a999" => Some((c001::c001_a999::cuda_solve_challenge as CudaSolveChallengeFn, &c001::c001_a999::KERNEL)),
                                    _ => {
                                        Option::<(CudaSolveChallengeFn, &Option<CudaKernel>)>::None
                                    }
                                };

                                match filter_step {
                                    Some((cuda_solve_challenge, algo_kernel)) => {
                                        let challenge_cuda_funcs = challenge_cuda_funcs
                                            .get_or_insert(
                                                get_or_compile_cuda(
                                                    &job.settings.challenge_id,
                                                    &tig_challenges::c001::KERNEL,
                                                    &dev,
                                                )
                                                .await,
                                            )
                                            .clone();
                                        let algorithm_cuda_funcs = algorithm_cuda_funcs
                                            .get_or_insert(
                                                get_or_compile_cuda(
                                                    &job.settings.algorithm_id,
                                                    algo_kernel,
                                                    &dev,
                                                )
                                                .await,
                                            )
                                            .clone();
                                        let challenge =
                                            tig_challenges::c001::Challenge::cuda_generate_instance_from_vec(
                                                seeds,
                                                &job.settings.difficulty,
                                                &dev,
                                                challenge_cuda_funcs,
                                            )
                                            .unwrap();
                                        match cuda_solve_challenge(
                                            &challenge,
                                            &dev,
                                            algorithm_cuda_funcs,
                                        ) {
                                            Ok(Some(solution)) => {
                                                challenge.verify_solution(&solution).is_err()
                                            }
                                            _ => true,
                                        }
                                    }
                                    None => false,
                                }
                            }
                            "c002" => {
                                type CudaSolveChallengeFn =
                                    fn(
                                        &tig_challenges::c002::Challenge,
                                        &Arc<CudaDevice>,
                                        HashMap<&'static str, CudaFunction>,
                                    )
                                        -> anyhow::Result<Option<tig_challenges::c002::Solution>>;

                                let filter_step = match job.settings.algorithm_id.as_str() {
                                    // #[cfg(feature = "c002_a001")]
                                    // "c002_a001" => Some((c002::c002_a001::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a001::KERNEL)),

                                    // #[cfg(feature = "c002_a002")]
                                    // "c002_a002" => Some((c002::c002_a002::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a002::KERNEL)),

                                    // #[cfg(feature = "c002_a003")]
                                    // "c002_a003" => Some((c002::c002_a003::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a003::KERNEL)),

                                    // #[cfg(feature = "c002_a004")]
                                    // "c002_a004" => Some((c002::c002_a004::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a004::KERNEL)),

                                    // #[cfg(feature = "c002_a005")]
                                    // "c002_a005" => Some((c002::c002_a005::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a005::KERNEL)),

                                    // #[cfg(feature = "c002_a006")]
                                    // "c002_a006" => Some((c002::c002_a006::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a006::KERNEL)),

                                    // #[cfg(feature = "c002_a007")]
                                    // "c002_a007" => Some((c002::c002_a007::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a007::KERNEL)),

                                    // #[cfg(feature = "c002_a008")]
                                    // "c002_a008" => Some((c002::c002_a008::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a008::KERNEL)),

                                    // #[cfg(feature = "c002_a009")]
                                    // "c002_a009" => Some((c002::c002_a009::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a009::KERNEL)),

                                    // #[cfg(feature = "c002_a010")]
                                    // "c002_a010" => Some((c002::c002_a010::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a010::KERNEL)),

                                    // #[cfg(feature = "c002_a011")]
                                    // "c002_a011" => Some((c002::c002_a011::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a011::KERNEL)),

                                    // #[cfg(feature = "c002_a012")]
                                    // "c002_a012" => Some((c002::c002_a012::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a012::KERNEL)),

                                    // #[cfg(feature = "c002_a013")]
                                    // "c002_a013" => Some((c002::c002_a013::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a013::KERNEL)),

                                    // #[cfg(feature = "c002_a014")]
                                    // "c002_a014" => Some((c002::c002_a014::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a014::KERNEL)),

                                    // #[cfg(feature = "c002_a015")]
                                    // "c002_a015" => Some((c002::c002_a015::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a015::KERNEL)),

                                    // #[cfg(feature = "c002_a016")]
                                    // "c002_a016" => Some((c002::c002_a016::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a016::KERNEL)),

                                    // #[cfg(feature = "c002_a017")]
                                    // "c002_a017" => Some((c002::c002_a017::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a017::KERNEL)),

                                    // #[cfg(feature = "c002_a018")]
                                    // "c002_a018" => Some((c002::c002_a018::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a018::KERNEL)),

                                    // #[cfg(feature = "c002_a019")]
                                    // "c002_a019" => Some((c002::c002_a019::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a019::KERNEL)),

                                    // #[cfg(feature = "c002_a020")]
                                    // "c002_a020" => Some((c002::c002_a020::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a020::KERNEL)),

                                    // #[cfg(feature = "c002_a021")]
                                    // "c002_a021" => Some((c002::c002_a021::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a021::KERNEL)),

                                    // #[cfg(feature = "c002_a022")]
                                    // "c002_a022" => Some((c002::c002_a022::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a022::KERNEL)),

                                    // #[cfg(feature = "c002_a023")]
                                    // "c002_a023" => Some((c002::c002_a023::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a023::KERNEL)),

                                    // #[cfg(feature = "c002_a024")]
                                    // "c002_a024" => Some((c002::c002_a024::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a024::KERNEL)),

                                    // #[cfg(feature = "c002_a025")]
                                    // "c002_a025" => Some((c002::c002_a025::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a025::KERNEL)),

                                    // #[cfg(feature = "c002_a026")]
                                    // "c002_a026" => Some((c002::c002_a026::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a026::KERNEL)),

                                    // #[cfg(feature = "c002_a027")]
                                    // "c002_a027" => Some((c002::c002_a027::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a027::KERNEL)),

                                    // #[cfg(feature = "c002_a028")]
                                    // "c002_a028" => Some((c002::c002_a028::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a028::KERNEL)),

                                    // #[cfg(feature = "c002_a029")]
                                    // "c002_a029" => Some((c002::c002_a029::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a029::KERNEL)),

                                    // #[cfg(feature = "c002_a030")]
                                    // "c002_a030" => Some((c002::c002_a030::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a030::KERNEL)),

                                    // #[cfg(feature = "c002_a031")]
                                    // "c002_a031" => Some((c002::c002_a031::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a031::KERNEL)),

                                    // #[cfg(feature = "c002_a032")]
                                    // "c002_a032" => Some((c002::c002_a032::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a032::KERNEL)),

                                    // #[cfg(feature = "c002_a033")]
                                    // "c002_a033" => Some((c002::c002_a033::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a033::KERNEL)),

                                    // #[cfg(feature = "c002_a034")]
                                    // "c002_a034" => Some((c002::c002_a034::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a034::KERNEL)),

                                    // #[cfg(feature = "c002_a035")]
                                    // "c002_a035" => Some((c002::c002_a035::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a035::KERNEL)),

                                    // #[cfg(feature = "c002_a036")]
                                    // "c002_a036" => Some((c002::c002_a036::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a036::KERNEL)),

                                    // #[cfg(feature = "c002_a037")]
                                    // "c002_a037" => Some((c002::c002_a037::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a037::KERNEL)),

                                    // #[cfg(feature = "c002_a038")]
                                    // "c002_a038" => Some((c002::c002_a038::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a038::KERNEL)),

                                    // #[cfg(feature = "c002_a039")]
                                    // "c002_a039" => Some((c002::c002_a039::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a039::KERNEL)),

                                    // #[cfg(feature = "c002_a040")]
                                    // "c002_a040" => Some((c002::c002_a040::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a040::KERNEL)),

                                    // #[cfg(feature = "c002_a041")]
                                    // "c002_a041" => Some((c002::c002_a041::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a041::KERNEL)),

                                    // #[cfg(feature = "c002_a042")]
                                    // "c002_a042" => Some((c002::c002_a042::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a042::KERNEL)),

                                    // #[cfg(feature = "c002_a043")]
                                    // "c002_a043" => Some((c002::c002_a043::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a043::KERNEL)),

                                    // #[cfg(feature = "c002_a044")]
                                    // "c002_a044" => Some((c002::c002_a044::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a044::KERNEL)),

                                    // #[cfg(feature = "c002_a045")]
                                    // "c002_a045" => Some((c002::c002_a045::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a045::KERNEL)),

                                    // #[cfg(feature = "c002_a046")]
                                    // "c002_a046" => Some((c002::c002_a046::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a046::KERNEL)),

                                    // #[cfg(feature = "c002_a047")]
                                    // "c002_a047" => Some((c002::c002_a047::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a047::KERNEL)),

                                    // #[cfg(feature = "c002_a048")]
                                    // "c002_a048" => Some((c002::c002_a048::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a048::KERNEL)),

                                    // #[cfg(feature = "c002_a049")]
                                    // "c002_a049" => Some((c002::c002_a049::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a049::KERNEL)),

                                    // #[cfg(feature = "c002_a050")]
                                    // "c002_a050" => Some((c002::c002_a050::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a050::KERNEL)),

                                    // #[cfg(feature = "c002_a051")]
                                    // "c002_a051" => Some((c002::c002_a051::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a051::KERNEL)),

                                    // #[cfg(feature = "c002_a052")]
                                    // "c002_a052" => Some((c002::c002_a052::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a052::KERNEL)),

                                    // #[cfg(feature = "c002_a053")]
                                    // "c002_a053" => Some((c002::c002_a053::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a053::KERNEL)),

                                    // #[cfg(feature = "c002_a054")]
                                    // "c002_a054" => Some((c002::c002_a054::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a054::KERNEL)),

                                    // #[cfg(feature = "c002_a055")]
                                    // "c002_a055" => Some((c002::c002_a055::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a055::KERNEL)),

                                    // #[cfg(feature = "c002_a056")]
                                    // "c002_a056" => Some((c002::c002_a056::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a056::KERNEL)),

                                    // #[cfg(feature = "c002_a057")]
                                    // "c002_a057" => Some((c002::c002_a057::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a057::KERNEL)),

                                    // #[cfg(feature = "c002_a058")]
                                    // "c002_a058" => Some((c002::c002_a058::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a058::KERNEL)),

                                    // #[cfg(feature = "c002_a059")]
                                    // "c002_a059" => Some((c002::c002_a059::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a059::KERNEL)),

                                    // #[cfg(feature = "c002_a060")]
                                    // "c002_a060" => Some((c002::c002_a060::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a060::KERNEL)),

                                    // #[cfg(feature = "c002_a061")]
                                    // "c002_a061" => Some((c002::c002_a061::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a061::KERNEL)),

                                    // #[cfg(feature = "c002_a062")]
                                    // "c002_a062" => Some((c002::c002_a062::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a062::KERNEL)),

                                    // #[cfg(feature = "c002_a063")]
                                    // "c002_a063" => Some((c002::c002_a063::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a063::KERNEL)),

                                    // #[cfg(feature = "c002_a064")]
                                    // "c002_a064" => Some((c002::c002_a064::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a064::KERNEL)),

                                    // #[cfg(feature = "c002_a065")]
                                    // "c002_a065" => Some((c002::c002_a065::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a065::KERNEL)),

                                    // #[cfg(feature = "c002_a066")]
                                    // "c002_a066" => Some((c002::c002_a066::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a066::KERNEL)),

                                    // #[cfg(feature = "c002_a067")]
                                    // "c002_a067" => Some((c002::c002_a067::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a067::KERNEL)),

                                    // #[cfg(feature = "c002_a068")]
                                    // "c002_a068" => Some((c002::c002_a068::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a068::KERNEL)),

                                    // #[cfg(feature = "c002_a069")]
                                    // "c002_a069" => Some((c002::c002_a069::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a069::KERNEL)),

                                    // #[cfg(feature = "c002_a070")]
                                    // "c002_a070" => Some((c002::c002_a070::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a070::KERNEL)),

                                    // #[cfg(feature = "c002_a071")]
                                    // "c002_a071" => Some((c002::c002_a071::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a071::KERNEL)),

                                    // #[cfg(feature = "c002_a072")]
                                    // "c002_a072" => Some((c002::c002_a072::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a072::KERNEL)),

                                    // #[cfg(feature = "c002_a073")]
                                    // "c002_a073" => Some((c002::c002_a073::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a073::KERNEL)),

                                    // #[cfg(feature = "c002_a074")]
                                    // "c002_a074" => Some((c002::c002_a074::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a074::KERNEL)),

                                    // #[cfg(feature = "c002_a075")]
                                    // "c002_a075" => Some((c002::c002_a075::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a075::KERNEL)),

                                    // #[cfg(feature = "c002_a076")]
                                    // "c002_a076" => Some((c002::c002_a076::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a076::KERNEL)),

                                    // #[cfg(feature = "c002_a077")]
                                    // "c002_a077" => Some((c002::c002_a077::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a077::KERNEL)),

                                    // #[cfg(feature = "c002_a078")]
                                    // "c002_a078" => Some((c002::c002_a078::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a078::KERNEL)),

                                    // #[cfg(feature = "c002_a079")]
                                    // "c002_a079" => Some((c002::c002_a079::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a079::KERNEL)),

                                    // #[cfg(feature = "c002_a080")]
                                    // "c002_a080" => Some((c002::c002_a080::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a080::KERNEL)),

                                    // #[cfg(feature = "c002_a081")]
                                    // "c002_a081" => Some((c002::c002_a081::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a081::KERNEL)),

                                    // #[cfg(feature = "c002_a082")]
                                    // "c002_a082" => Some((c002::c002_a082::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a082::KERNEL)),

                                    // #[cfg(feature = "c002_a083")]
                                    // "c002_a083" => Some((c002::c002_a083::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a083::KERNEL)),

                                    // #[cfg(feature = "c002_a084")]
                                    // "c002_a084" => Some((c002::c002_a084::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a084::KERNEL)),

                                    // #[cfg(feature = "c002_a085")]
                                    // "c002_a085" => Some((c002::c002_a085::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a085::KERNEL)),

                                    // #[cfg(feature = "c002_a086")]
                                    // "c002_a086" => Some((c002::c002_a086::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a086::KERNEL)),

                                    // #[cfg(feature = "c002_a087")]
                                    // "c002_a087" => Some((c002::c002_a087::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a087::KERNEL)),

                                    // #[cfg(feature = "c002_a088")]
                                    // "c002_a088" => Some((c002::c002_a088::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a088::KERNEL)),

                                    // #[cfg(feature = "c002_a089")]
                                    // "c002_a089" => Some((c002::c002_a089::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a089::KERNEL)),

                                    // #[cfg(feature = "c002_a090")]
                                    // "c002_a090" => Some((c002::c002_a090::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a090::KERNEL)),

                                    // #[cfg(feature = "c002_a091")]
                                    // "c002_a091" => Some((c002::c002_a091::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a091::KERNEL)),

                                    // #[cfg(feature = "c002_a092")]
                                    // "c002_a092" => Some((c002::c002_a092::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a092::KERNEL)),

                                    // #[cfg(feature = "c002_a093")]
                                    // "c002_a093" => Some((c002::c002_a093::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a093::KERNEL)),

                                    // #[cfg(feature = "c002_a094")]
                                    // "c002_a094" => Some((c002::c002_a094::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a094::KERNEL)),

                                    // #[cfg(feature = "c002_a095")]
                                    // "c002_a095" => Some((c002::c002_a095::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a095::KERNEL)),

                                    // #[cfg(feature = "c002_a096")]
                                    // "c002_a096" => Some((c002::c002_a096::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a096::KERNEL)),

                                    // #[cfg(feature = "c002_a097")]
                                    // "c002_a097" => Some((c002::c002_a097::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a097::KERNEL)),

                                    // #[cfg(feature = "c002_a098")]
                                    // "c002_a098" => Some((c002::c002_a098::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a098::KERNEL)),

                                    // #[cfg(feature = "c002_a099")]
                                    // "c002_a099" => Some((c002::c002_a099::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a099::KERNEL)),

                                    // #[cfg(feature = "c002_a100")]
                                    // "c002_a100" => Some((c002::c002_a100::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a100::KERNEL)),

                                    // #[cfg(feature = "c002_a101")]
                                    // "c002_a101" => Some((c002::c002_a101::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a101::KERNEL)),

                                    // #[cfg(feature = "c002_a102")]
                                    // "c002_a102" => Some((c002::c002_a102::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a102::KERNEL)),

                                    // #[cfg(feature = "c002_a103")]
                                    // "c002_a103" => Some((c002::c002_a103::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a103::KERNEL)),

                                    // #[cfg(feature = "c002_a104")]
                                    // "c002_a104" => Some((c002::c002_a104::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a104::KERNEL)),

                                    // #[cfg(feature = "c002_a105")]
                                    // "c002_a105" => Some((c002::c002_a105::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a105::KERNEL)),

                                    // #[cfg(feature = "c002_a106")]
                                    // "c002_a106" => Some((c002::c002_a106::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a106::KERNEL)),

                                    // #[cfg(feature = "c002_a107")]
                                    // "c002_a107" => Some((c002::c002_a107::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a107::KERNEL)),

                                    // #[cfg(feature = "c002_a108")]
                                    // "c002_a108" => Some((c002::c002_a108::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a108::KERNEL)),

                                    // #[cfg(feature = "c002_a109")]
                                    // "c002_a109" => Some((c002::c002_a109::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a109::KERNEL)),

                                    // #[cfg(feature = "c002_a110")]
                                    // "c002_a110" => Some((c002::c002_a110::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a110::KERNEL)),

                                    // #[cfg(feature = "c002_a111")]
                                    // "c002_a111" => Some((c002::c002_a111::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a111::KERNEL)),

                                    // #[cfg(feature = "c002_a112")]
                                    // "c002_a112" => Some((c002::c002_a112::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a112::KERNEL)),

                                    // #[cfg(feature = "c002_a113")]
                                    // "c002_a113" => Some((c002::c002_a113::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a113::KERNEL)),

                                    // #[cfg(feature = "c002_a114")]
                                    // "c002_a114" => Some((c002::c002_a114::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a114::KERNEL)),

                                    // #[cfg(feature = "c002_a115")]
                                    // "c002_a115" => Some((c002::c002_a115::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a115::KERNEL)),

                                    // #[cfg(feature = "c002_a116")]
                                    // "c002_a116" => Some((c002::c002_a116::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a116::KERNEL)),

                                    // #[cfg(feature = "c002_a117")]
                                    // "c002_a117" => Some((c002::c002_a117::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a117::KERNEL)),

                                    // #[cfg(feature = "c002_a118")]
                                    // "c002_a118" => Some((c002::c002_a118::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a118::KERNEL)),

                                    // #[cfg(feature = "c002_a119")]
                                    // "c002_a119" => Some((c002::c002_a119::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a119::KERNEL)),

                                    // #[cfg(feature = "c002_a120")]
                                    // "c002_a120" => Some((c002::c002_a120::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a120::KERNEL)),

                                    // #[cfg(feature = "c002_a121")]
                                    // "c002_a121" => Some((c002::c002_a121::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a121::KERNEL)),

                                    // #[cfg(feature = "c002_a122")]
                                    // "c002_a122" => Some((c002::c002_a122::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a122::KERNEL)),

                                    // #[cfg(feature = "c002_a123")]
                                    // "c002_a123" => Some((c002::c002_a123::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a123::KERNEL)),

                                    // #[cfg(feature = "c002_a124")]
                                    // "c002_a124" => Some((c002::c002_a124::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a124::KERNEL)),

                                    // #[cfg(feature = "c002_a125")]
                                    // "c002_a125" => Some((c002::c002_a125::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a125::KERNEL)),

                                    // #[cfg(feature = "c002_a126")]
                                    // "c002_a126" => Some((c002::c002_a126::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a126::KERNEL)),

                                    // #[cfg(feature = "c002_a127")]
                                    // "c002_a127" => Some((c002::c002_a127::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a127::KERNEL)),

                                    // #[cfg(feature = "c002_a128")]
                                    // "c002_a128" => Some((c002::c002_a128::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a128::KERNEL)),

                                    // #[cfg(feature = "c002_a129")]
                                    // "c002_a129" => Some((c002::c002_a129::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a129::KERNEL)),

                                    // #[cfg(feature = "c002_a130")]
                                    // "c002_a130" => Some((c002::c002_a130::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a130::KERNEL)),

                                    // #[cfg(feature = "c002_a131")]
                                    // "c002_a131" => Some((c002::c002_a131::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a131::KERNEL)),

                                    // #[cfg(feature = "c002_a132")]
                                    // "c002_a132" => Some((c002::c002_a132::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a132::KERNEL)),

                                    // #[cfg(feature = "c002_a133")]
                                    // "c002_a133" => Some((c002::c002_a133::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a133::KERNEL)),

                                    // #[cfg(feature = "c002_a134")]
                                    // "c002_a134" => Some((c002::c002_a134::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a134::KERNEL)),

                                    // #[cfg(feature = "c002_a135")]
                                    // "c002_a135" => Some((c002::c002_a135::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a135::KERNEL)),

                                    // #[cfg(feature = "c002_a136")]
                                    // "c002_a136" => Some((c002::c002_a136::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a136::KERNEL)),

                                    // #[cfg(feature = "c002_a137")]
                                    // "c002_a137" => Some((c002::c002_a137::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a137::KERNEL)),

                                    // #[cfg(feature = "c002_a138")]
                                    // "c002_a138" => Some((c002::c002_a138::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a138::KERNEL)),

                                    // #[cfg(feature = "c002_a139")]
                                    // "c002_a139" => Some((c002::c002_a139::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a139::KERNEL)),

                                    // #[cfg(feature = "c002_a140")]
                                    // "c002_a140" => Some((c002::c002_a140::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a140::KERNEL)),

                                    // #[cfg(feature = "c002_a141")]
                                    // "c002_a141" => Some((c002::c002_a141::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a141::KERNEL)),

                                    // #[cfg(feature = "c002_a142")]
                                    // "c002_a142" => Some((c002::c002_a142::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a142::KERNEL)),

                                    // #[cfg(feature = "c002_a143")]
                                    // "c002_a143" => Some((c002::c002_a143::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a143::KERNEL)),

                                    // #[cfg(feature = "c002_a144")]
                                    // "c002_a144" => Some((c002::c002_a144::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a144::KERNEL)),

                                    // #[cfg(feature = "c002_a145")]
                                    // "c002_a145" => Some((c002::c002_a145::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a145::KERNEL)),

                                    // #[cfg(feature = "c002_a146")]
                                    // "c002_a146" => Some((c002::c002_a146::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a146::KERNEL)),

                                    // #[cfg(feature = "c002_a147")]
                                    // "c002_a147" => Some((c002::c002_a147::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a147::KERNEL)),

                                    // #[cfg(feature = "c002_a148")]
                                    // "c002_a148" => Some((c002::c002_a148::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a148::KERNEL)),

                                    // #[cfg(feature = "c002_a149")]
                                    // "c002_a149" => Some((c002::c002_a149::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a149::KERNEL)),

                                    // #[cfg(feature = "c002_a150")]
                                    // "c002_a150" => Some((c002::c002_a150::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a150::KERNEL)),

                                    // #[cfg(feature = "c002_a151")]
                                    // "c002_a151" => Some((c002::c002_a151::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a151::KERNEL)),

                                    // #[cfg(feature = "c002_a152")]
                                    // "c002_a152" => Some((c002::c002_a152::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a152::KERNEL)),

                                    // #[cfg(feature = "c002_a153")]
                                    // "c002_a153" => Some((c002::c002_a153::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a153::KERNEL)),

                                    // #[cfg(feature = "c002_a154")]
                                    // "c002_a154" => Some((c002::c002_a154::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a154::KERNEL)),

                                    // #[cfg(feature = "c002_a155")]
                                    // "c002_a155" => Some((c002::c002_a155::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a155::KERNEL)),

                                    // #[cfg(feature = "c002_a156")]
                                    // "c002_a156" => Some((c002::c002_a156::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a156::KERNEL)),

                                    // #[cfg(feature = "c002_a157")]
                                    // "c002_a157" => Some((c002::c002_a157::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a157::KERNEL)),

                                    // #[cfg(feature = "c002_a158")]
                                    // "c002_a158" => Some((c002::c002_a158::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a158::KERNEL)),

                                    // #[cfg(feature = "c002_a159")]
                                    // "c002_a159" => Some((c002::c002_a159::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a159::KERNEL)),

                                    // #[cfg(feature = "c002_a160")]
                                    // "c002_a160" => Some((c002::c002_a160::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a160::KERNEL)),

                                    // #[cfg(feature = "c002_a161")]
                                    // "c002_a161" => Some((c002::c002_a161::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a161::KERNEL)),

                                    // #[cfg(feature = "c002_a162")]
                                    // "c002_a162" => Some((c002::c002_a162::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a162::KERNEL)),

                                    // #[cfg(feature = "c002_a163")]
                                    // "c002_a163" => Some((c002::c002_a163::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a163::KERNEL)),

                                    // #[cfg(feature = "c002_a164")]
                                    // "c002_a164" => Some((c002::c002_a164::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a164::KERNEL)),

                                    // #[cfg(feature = "c002_a165")]
                                    // "c002_a165" => Some((c002::c002_a165::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a165::KERNEL)),

                                    // #[cfg(feature = "c002_a166")]
                                    // "c002_a166" => Some((c002::c002_a166::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a166::KERNEL)),

                                    // #[cfg(feature = "c002_a167")]
                                    // "c002_a167" => Some((c002::c002_a167::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a167::KERNEL)),

                                    // #[cfg(feature = "c002_a168")]
                                    // "c002_a168" => Some((c002::c002_a168::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a168::KERNEL)),

                                    // #[cfg(feature = "c002_a169")]
                                    // "c002_a169" => Some((c002::c002_a169::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a169::KERNEL)),

                                    // #[cfg(feature = "c002_a170")]
                                    // "c002_a170" => Some((c002::c002_a170::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a170::KERNEL)),

                                    // #[cfg(feature = "c002_a171")]
                                    // "c002_a171" => Some((c002::c002_a171::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a171::KERNEL)),

                                    // #[cfg(feature = "c002_a172")]
                                    // "c002_a172" => Some((c002::c002_a172::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a172::KERNEL)),

                                    // #[cfg(feature = "c002_a173")]
                                    // "c002_a173" => Some((c002::c002_a173::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a173::KERNEL)),

                                    // #[cfg(feature = "c002_a174")]
                                    // "c002_a174" => Some((c002::c002_a174::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a174::KERNEL)),

                                    // #[cfg(feature = "c002_a175")]
                                    // "c002_a175" => Some((c002::c002_a175::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a175::KERNEL)),

                                    // #[cfg(feature = "c002_a176")]
                                    // "c002_a176" => Some((c002::c002_a176::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a176::KERNEL)),

                                    // #[cfg(feature = "c002_a177")]
                                    // "c002_a177" => Some((c002::c002_a177::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a177::KERNEL)),

                                    // #[cfg(feature = "c002_a178")]
                                    // "c002_a178" => Some((c002::c002_a178::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a178::KERNEL)),

                                    // #[cfg(feature = "c002_a179")]
                                    // "c002_a179" => Some((c002::c002_a179::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a179::KERNEL)),

                                    // #[cfg(feature = "c002_a180")]
                                    // "c002_a180" => Some((c002::c002_a180::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a180::KERNEL)),

                                    // #[cfg(feature = "c002_a181")]
                                    // "c002_a181" => Some((c002::c002_a181::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a181::KERNEL)),

                                    // #[cfg(feature = "c002_a182")]
                                    // "c002_a182" => Some((c002::c002_a182::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a182::KERNEL)),

                                    // #[cfg(feature = "c002_a183")]
                                    // "c002_a183" => Some((c002::c002_a183::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a183::KERNEL)),

                                    // #[cfg(feature = "c002_a184")]
                                    // "c002_a184" => Some((c002::c002_a184::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a184::KERNEL)),

                                    // #[cfg(feature = "c002_a185")]
                                    // "c002_a185" => Some((c002::c002_a185::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a185::KERNEL)),

                                    // #[cfg(feature = "c002_a186")]
                                    // "c002_a186" => Some((c002::c002_a186::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a186::KERNEL)),

                                    // #[cfg(feature = "c002_a187")]
                                    // "c002_a187" => Some((c002::c002_a187::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a187::KERNEL)),

                                    // #[cfg(feature = "c002_a188")]
                                    // "c002_a188" => Some((c002::c002_a188::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a188::KERNEL)),

                                    // #[cfg(feature = "c002_a189")]
                                    // "c002_a189" => Some((c002::c002_a189::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a189::KERNEL)),

                                    // #[cfg(feature = "c002_a190")]
                                    // "c002_a190" => Some((c002::c002_a190::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a190::KERNEL)),

                                    // #[cfg(feature = "c002_a191")]
                                    // "c002_a191" => Some((c002::c002_a191::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a191::KERNEL)),

                                    // #[cfg(feature = "c002_a192")]
                                    // "c002_a192" => Some((c002::c002_a192::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a192::KERNEL)),

                                    // #[cfg(feature = "c002_a193")]
                                    // "c002_a193" => Some((c002::c002_a193::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a193::KERNEL)),

                                    // #[cfg(feature = "c002_a194")]
                                    // "c002_a194" => Some((c002::c002_a194::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a194::KERNEL)),

                                    // #[cfg(feature = "c002_a195")]
                                    // "c002_a195" => Some((c002::c002_a195::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a195::KERNEL)),

                                    // #[cfg(feature = "c002_a196")]
                                    // "c002_a196" => Some((c002::c002_a196::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a196::KERNEL)),

                                    // #[cfg(feature = "c002_a197")]
                                    // "c002_a197" => Some((c002::c002_a197::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a197::KERNEL)),

                                    // #[cfg(feature = "c002_a198")]
                                    // "c002_a198" => Some((c002::c002_a198::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a198::KERNEL)),

                                    // #[cfg(feature = "c002_a199")]
                                    // "c002_a199" => Some((c002::c002_a199::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a199::KERNEL)),

                                    // #[cfg(feature = "c002_a200")]
                                    // "c002_a200" => Some((c002::c002_a200::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a200::KERNEL)),

                                    // #[cfg(feature = "c002_a201")]
                                    // "c002_a201" => Some((c002::c002_a201::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a201::KERNEL)),

                                    // #[cfg(feature = "c002_a202")]
                                    // "c002_a202" => Some((c002::c002_a202::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a202::KERNEL)),

                                    // #[cfg(feature = "c002_a203")]
                                    // "c002_a203" => Some((c002::c002_a203::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a203::KERNEL)),

                                    // #[cfg(feature = "c002_a204")]
                                    // "c002_a204" => Some((c002::c002_a204::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a204::KERNEL)),

                                    // #[cfg(feature = "c002_a205")]
                                    // "c002_a205" => Some((c002::c002_a205::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a205::KERNEL)),

                                    // #[cfg(feature = "c002_a206")]
                                    // "c002_a206" => Some((c002::c002_a206::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a206::KERNEL)),

                                    // #[cfg(feature = "c002_a207")]
                                    // "c002_a207" => Some((c002::c002_a207::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a207::KERNEL)),

                                    // #[cfg(feature = "c002_a208")]
                                    // "c002_a208" => Some((c002::c002_a208::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a208::KERNEL)),

                                    // #[cfg(feature = "c002_a209")]
                                    // "c002_a209" => Some((c002::c002_a209::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a209::KERNEL)),

                                    // #[cfg(feature = "c002_a210")]
                                    // "c002_a210" => Some((c002::c002_a210::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a210::KERNEL)),

                                    // #[cfg(feature = "c002_a211")]
                                    // "c002_a211" => Some((c002::c002_a211::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a211::KERNEL)),

                                    // #[cfg(feature = "c002_a212")]
                                    // "c002_a212" => Some((c002::c002_a212::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a212::KERNEL)),

                                    // #[cfg(feature = "c002_a213")]
                                    // "c002_a213" => Some((c002::c002_a213::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a213::KERNEL)),

                                    // #[cfg(feature = "c002_a214")]
                                    // "c002_a214" => Some((c002::c002_a214::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a214::KERNEL)),

                                    // #[cfg(feature = "c002_a215")]
                                    // "c002_a215" => Some((c002::c002_a215::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a215::KERNEL)),

                                    // #[cfg(feature = "c002_a216")]
                                    // "c002_a216" => Some((c002::c002_a216::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a216::KERNEL)),

                                    // #[cfg(feature = "c002_a217")]
                                    // "c002_a217" => Some((c002::c002_a217::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a217::KERNEL)),

                                    // #[cfg(feature = "c002_a218")]
                                    // "c002_a218" => Some((c002::c002_a218::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a218::KERNEL)),

                                    // #[cfg(feature = "c002_a219")]
                                    // "c002_a219" => Some((c002::c002_a219::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a219::KERNEL)),

                                    // #[cfg(feature = "c002_a220")]
                                    // "c002_a220" => Some((c002::c002_a220::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a220::KERNEL)),

                                    // #[cfg(feature = "c002_a221")]
                                    // "c002_a221" => Some((c002::c002_a221::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a221::KERNEL)),

                                    // #[cfg(feature = "c002_a222")]
                                    // "c002_a222" => Some((c002::c002_a222::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a222::KERNEL)),

                                    // #[cfg(feature = "c002_a223")]
                                    // "c002_a223" => Some((c002::c002_a223::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a223::KERNEL)),

                                    // #[cfg(feature = "c002_a224")]
                                    // "c002_a224" => Some((c002::c002_a224::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a224::KERNEL)),

                                    // #[cfg(feature = "c002_a225")]
                                    // "c002_a225" => Some((c002::c002_a225::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a225::KERNEL)),

                                    // #[cfg(feature = "c002_a226")]
                                    // "c002_a226" => Some((c002::c002_a226::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a226::KERNEL)),

                                    // #[cfg(feature = "c002_a227")]
                                    // "c002_a227" => Some((c002::c002_a227::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a227::KERNEL)),

                                    // #[cfg(feature = "c002_a228")]
                                    // "c002_a228" => Some((c002::c002_a228::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a228::KERNEL)),

                                    // #[cfg(feature = "c002_a229")]
                                    // "c002_a229" => Some((c002::c002_a229::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a229::KERNEL)),

                                    // #[cfg(feature = "c002_a230")]
                                    // "c002_a230" => Some((c002::c002_a230::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a230::KERNEL)),

                                    // #[cfg(feature = "c002_a231")]
                                    // "c002_a231" => Some((c002::c002_a231::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a231::KERNEL)),

                                    // #[cfg(feature = "c002_a232")]
                                    // "c002_a232" => Some((c002::c002_a232::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a232::KERNEL)),

                                    // #[cfg(feature = "c002_a233")]
                                    // "c002_a233" => Some((c002::c002_a233::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a233::KERNEL)),

                                    // #[cfg(feature = "c002_a234")]
                                    // "c002_a234" => Some((c002::c002_a234::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a234::KERNEL)),

                                    // #[cfg(feature = "c002_a235")]
                                    // "c002_a235" => Some((c002::c002_a235::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a235::KERNEL)),

                                    // #[cfg(feature = "c002_a236")]
                                    // "c002_a236" => Some((c002::c002_a236::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a236::KERNEL)),

                                    // #[cfg(feature = "c002_a237")]
                                    // "c002_a237" => Some((c002::c002_a237::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a237::KERNEL)),

                                    // #[cfg(feature = "c002_a238")]
                                    // "c002_a238" => Some((c002::c002_a238::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a238::KERNEL)),

                                    // #[cfg(feature = "c002_a239")]
                                    // "c002_a239" => Some((c002::c002_a239::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a239::KERNEL)),

                                    // #[cfg(feature = "c002_a240")]
                                    // "c002_a240" => Some((c002::c002_a240::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a240::KERNEL)),

                                    // #[cfg(feature = "c002_a241")]
                                    // "c002_a241" => Some((c002::c002_a241::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a241::KERNEL)),

                                    // #[cfg(feature = "c002_a242")]
                                    // "c002_a242" => Some((c002::c002_a242::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a242::KERNEL)),

                                    // #[cfg(feature = "c002_a243")]
                                    // "c002_a243" => Some((c002::c002_a243::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a243::KERNEL)),

                                    // #[cfg(feature = "c002_a244")]
                                    // "c002_a244" => Some((c002::c002_a244::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a244::KERNEL)),

                                    // #[cfg(feature = "c002_a245")]
                                    // "c002_a245" => Some((c002::c002_a245::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a245::KERNEL)),

                                    // #[cfg(feature = "c002_a246")]
                                    // "c002_a246" => Some((c002::c002_a246::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a246::KERNEL)),

                                    // #[cfg(feature = "c002_a247")]
                                    // "c002_a247" => Some((c002::c002_a247::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a247::KERNEL)),

                                    // #[cfg(feature = "c002_a248")]
                                    // "c002_a248" => Some((c002::c002_a248::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a248::KERNEL)),

                                    // #[cfg(feature = "c002_a249")]
                                    // "c002_a249" => Some((c002::c002_a249::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a249::KERNEL)),

                                    // #[cfg(feature = "c002_a250")]
                                    // "c002_a250" => Some((c002::c002_a250::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a250::KERNEL)),

                                    // #[cfg(feature = "c002_a251")]
                                    // "c002_a251" => Some((c002::c002_a251::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a251::KERNEL)),

                                    // #[cfg(feature = "c002_a252")]
                                    // "c002_a252" => Some((c002::c002_a252::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a252::KERNEL)),

                                    // #[cfg(feature = "c002_a253")]
                                    // "c002_a253" => Some((c002::c002_a253::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a253::KERNEL)),

                                    // #[cfg(feature = "c002_a254")]
                                    // "c002_a254" => Some((c002::c002_a254::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a254::KERNEL)),

                                    // #[cfg(feature = "c002_a255")]
                                    // "c002_a255" => Some((c002::c002_a255::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a255::KERNEL)),

                                    // #[cfg(feature = "c002_a256")]
                                    // "c002_a256" => Some((c002::c002_a256::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a256::KERNEL)),

                                    // #[cfg(feature = "c002_a257")]
                                    // "c002_a257" => Some((c002::c002_a257::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a257::KERNEL)),

                                    // #[cfg(feature = "c002_a258")]
                                    // "c002_a258" => Some((c002::c002_a258::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a258::KERNEL)),

                                    // #[cfg(feature = "c002_a259")]
                                    // "c002_a259" => Some((c002::c002_a259::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a259::KERNEL)),

                                    // #[cfg(feature = "c002_a260")]
                                    // "c002_a260" => Some((c002::c002_a260::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a260::KERNEL)),

                                    // #[cfg(feature = "c002_a261")]
                                    // "c002_a261" => Some((c002::c002_a261::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a261::KERNEL)),

                                    // #[cfg(feature = "c002_a262")]
                                    // "c002_a262" => Some((c002::c002_a262::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a262::KERNEL)),

                                    // #[cfg(feature = "c002_a263")]
                                    // "c002_a263" => Some((c002::c002_a263::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a263::KERNEL)),

                                    // #[cfg(feature = "c002_a264")]
                                    // "c002_a264" => Some((c002::c002_a264::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a264::KERNEL)),

                                    // #[cfg(feature = "c002_a265")]
                                    // "c002_a265" => Some((c002::c002_a265::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a265::KERNEL)),

                                    // #[cfg(feature = "c002_a266")]
                                    // "c002_a266" => Some((c002::c002_a266::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a266::KERNEL)),

                                    // #[cfg(feature = "c002_a267")]
                                    // "c002_a267" => Some((c002::c002_a267::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a267::KERNEL)),

                                    // #[cfg(feature = "c002_a268")]
                                    // "c002_a268" => Some((c002::c002_a268::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a268::KERNEL)),

                                    // #[cfg(feature = "c002_a269")]
                                    // "c002_a269" => Some((c002::c002_a269::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a269::KERNEL)),

                                    // #[cfg(feature = "c002_a270")]
                                    // "c002_a270" => Some((c002::c002_a270::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a270::KERNEL)),

                                    // #[cfg(feature = "c002_a271")]
                                    // "c002_a271" => Some((c002::c002_a271::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a271::KERNEL)),

                                    // #[cfg(feature = "c002_a272")]
                                    // "c002_a272" => Some((c002::c002_a272::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a272::KERNEL)),

                                    // #[cfg(feature = "c002_a273")]
                                    // "c002_a273" => Some((c002::c002_a273::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a273::KERNEL)),

                                    // #[cfg(feature = "c002_a274")]
                                    // "c002_a274" => Some((c002::c002_a274::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a274::KERNEL)),

                                    // #[cfg(feature = "c002_a275")]
                                    // "c002_a275" => Some((c002::c002_a275::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a275::KERNEL)),

                                    // #[cfg(feature = "c002_a276")]
                                    // "c002_a276" => Some((c002::c002_a276::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a276::KERNEL)),

                                    // #[cfg(feature = "c002_a277")]
                                    // "c002_a277" => Some((c002::c002_a277::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a277::KERNEL)),

                                    // #[cfg(feature = "c002_a278")]
                                    // "c002_a278" => Some((c002::c002_a278::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a278::KERNEL)),

                                    // #[cfg(feature = "c002_a279")]
                                    // "c002_a279" => Some((c002::c002_a279::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a279::KERNEL)),

                                    // #[cfg(feature = "c002_a280")]
                                    // "c002_a280" => Some((c002::c002_a280::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a280::KERNEL)),

                                    // #[cfg(feature = "c002_a281")]
                                    // "c002_a281" => Some((c002::c002_a281::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a281::KERNEL)),

                                    // #[cfg(feature = "c002_a282")]
                                    // "c002_a282" => Some((c002::c002_a282::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a282::KERNEL)),

                                    // #[cfg(feature = "c002_a283")]
                                    // "c002_a283" => Some((c002::c002_a283::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a283::KERNEL)),

                                    // #[cfg(feature = "c002_a284")]
                                    // "c002_a284" => Some((c002::c002_a284::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a284::KERNEL)),

                                    // #[cfg(feature = "c002_a285")]
                                    // "c002_a285" => Some((c002::c002_a285::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a285::KERNEL)),

                                    // #[cfg(feature = "c002_a286")]
                                    // "c002_a286" => Some((c002::c002_a286::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a286::KERNEL)),

                                    // #[cfg(feature = "c002_a287")]
                                    // "c002_a287" => Some((c002::c002_a287::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a287::KERNEL)),

                                    // #[cfg(feature = "c002_a288")]
                                    // "c002_a288" => Some((c002::c002_a288::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a288::KERNEL)),

                                    // #[cfg(feature = "c002_a289")]
                                    // "c002_a289" => Some((c002::c002_a289::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a289::KERNEL)),

                                    // #[cfg(feature = "c002_a290")]
                                    // "c002_a290" => Some((c002::c002_a290::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a290::KERNEL)),

                                    // #[cfg(feature = "c002_a291")]
                                    // "c002_a291" => Some((c002::c002_a291::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a291::KERNEL)),

                                    // #[cfg(feature = "c002_a292")]
                                    // "c002_a292" => Some((c002::c002_a292::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a292::KERNEL)),

                                    // #[cfg(feature = "c002_a293")]
                                    // "c002_a293" => Some((c002::c002_a293::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a293::KERNEL)),

                                    // #[cfg(feature = "c002_a294")]
                                    // "c002_a294" => Some((c002::c002_a294::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a294::KERNEL)),

                                    // #[cfg(feature = "c002_a295")]
                                    // "c002_a295" => Some((c002::c002_a295::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a295::KERNEL)),

                                    // #[cfg(feature = "c002_a296")]
                                    // "c002_a296" => Some((c002::c002_a296::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a296::KERNEL)),

                                    // #[cfg(feature = "c002_a297")]
                                    // "c002_a297" => Some((c002::c002_a297::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a297::KERNEL)),

                                    // #[cfg(feature = "c002_a298")]
                                    // "c002_a298" => Some((c002::c002_a298::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a298::KERNEL)),

                                    // #[cfg(feature = "c002_a299")]
                                    // "c002_a299" => Some((c002::c002_a299::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a299::KERNEL)),

                                    // #[cfg(feature = "c002_a300")]
                                    // "c002_a300" => Some((c002::c002_a300::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a300::KERNEL)),

                                    // #[cfg(feature = "c002_a301")]
                                    // "c002_a301" => Some((c002::c002_a301::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a301::KERNEL)),

                                    // #[cfg(feature = "c002_a302")]
                                    // "c002_a302" => Some((c002::c002_a302::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a302::KERNEL)),

                                    // #[cfg(feature = "c002_a303")]
                                    // "c002_a303" => Some((c002::c002_a303::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a303::KERNEL)),

                                    // #[cfg(feature = "c002_a304")]
                                    // "c002_a304" => Some((c002::c002_a304::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a304::KERNEL)),

                                    // #[cfg(feature = "c002_a305")]
                                    // "c002_a305" => Some((c002::c002_a305::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a305::KERNEL)),

                                    // #[cfg(feature = "c002_a306")]
                                    // "c002_a306" => Some((c002::c002_a306::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a306::KERNEL)),

                                    // #[cfg(feature = "c002_a307")]
                                    // "c002_a307" => Some((c002::c002_a307::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a307::KERNEL)),

                                    // #[cfg(feature = "c002_a308")]
                                    // "c002_a308" => Some((c002::c002_a308::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a308::KERNEL)),

                                    // #[cfg(feature = "c002_a309")]
                                    // "c002_a309" => Some((c002::c002_a309::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a309::KERNEL)),

                                    // #[cfg(feature = "c002_a310")]
                                    // "c002_a310" => Some((c002::c002_a310::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a310::KERNEL)),

                                    // #[cfg(feature = "c002_a311")]
                                    // "c002_a311" => Some((c002::c002_a311::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a311::KERNEL)),

                                    // #[cfg(feature = "c002_a312")]
                                    // "c002_a312" => Some((c002::c002_a312::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a312::KERNEL)),

                                    // #[cfg(feature = "c002_a313")]
                                    // "c002_a313" => Some((c002::c002_a313::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a313::KERNEL)),

                                    // #[cfg(feature = "c002_a314")]
                                    // "c002_a314" => Some((c002::c002_a314::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a314::KERNEL)),

                                    // #[cfg(feature = "c002_a315")]
                                    // "c002_a315" => Some((c002::c002_a315::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a315::KERNEL)),

                                    // #[cfg(feature = "c002_a316")]
                                    // "c002_a316" => Some((c002::c002_a316::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a316::KERNEL)),

                                    // #[cfg(feature = "c002_a317")]
                                    // "c002_a317" => Some((c002::c002_a317::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a317::KERNEL)),

                                    // #[cfg(feature = "c002_a318")]
                                    // "c002_a318" => Some((c002::c002_a318::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a318::KERNEL)),

                                    // #[cfg(feature = "c002_a319")]
                                    // "c002_a319" => Some((c002::c002_a319::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a319::KERNEL)),

                                    // #[cfg(feature = "c002_a320")]
                                    // "c002_a320" => Some((c002::c002_a320::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a320::KERNEL)),

                                    // #[cfg(feature = "c002_a321")]
                                    // "c002_a321" => Some((c002::c002_a321::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a321::KERNEL)),

                                    // #[cfg(feature = "c002_a322")]
                                    // "c002_a322" => Some((c002::c002_a322::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a322::KERNEL)),

                                    // #[cfg(feature = "c002_a323")]
                                    // "c002_a323" => Some((c002::c002_a323::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a323::KERNEL)),

                                    // #[cfg(feature = "c002_a324")]
                                    // "c002_a324" => Some((c002::c002_a324::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a324::KERNEL)),

                                    // #[cfg(feature = "c002_a325")]
                                    // "c002_a325" => Some((c002::c002_a325::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a325::KERNEL)),

                                    // #[cfg(feature = "c002_a326")]
                                    // "c002_a326" => Some((c002::c002_a326::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a326::KERNEL)),

                                    // #[cfg(feature = "c002_a327")]
                                    // "c002_a327" => Some((c002::c002_a327::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a327::KERNEL)),

                                    // #[cfg(feature = "c002_a328")]
                                    // "c002_a328" => Some((c002::c002_a328::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a328::KERNEL)),

                                    // #[cfg(feature = "c002_a329")]
                                    // "c002_a329" => Some((c002::c002_a329::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a329::KERNEL)),

                                    // #[cfg(feature = "c002_a330")]
                                    // "c002_a330" => Some((c002::c002_a330::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a330::KERNEL)),

                                    // #[cfg(feature = "c002_a331")]
                                    // "c002_a331" => Some((c002::c002_a331::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a331::KERNEL)),

                                    // #[cfg(feature = "c002_a332")]
                                    // "c002_a332" => Some((c002::c002_a332::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a332::KERNEL)),

                                    // #[cfg(feature = "c002_a333")]
                                    // "c002_a333" => Some((c002::c002_a333::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a333::KERNEL)),

                                    // #[cfg(feature = "c002_a334")]
                                    // "c002_a334" => Some((c002::c002_a334::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a334::KERNEL)),

                                    // #[cfg(feature = "c002_a335")]
                                    // "c002_a335" => Some((c002::c002_a335::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a335::KERNEL)),

                                    // #[cfg(feature = "c002_a336")]
                                    // "c002_a336" => Some((c002::c002_a336::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a336::KERNEL)),

                                    // #[cfg(feature = "c002_a337")]
                                    // "c002_a337" => Some((c002::c002_a337::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a337::KERNEL)),

                                    // #[cfg(feature = "c002_a338")]
                                    // "c002_a338" => Some((c002::c002_a338::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a338::KERNEL)),

                                    // #[cfg(feature = "c002_a339")]
                                    // "c002_a339" => Some((c002::c002_a339::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a339::KERNEL)),

                                    // #[cfg(feature = "c002_a340")]
                                    // "c002_a340" => Some((c002::c002_a340::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a340::KERNEL)),

                                    // #[cfg(feature = "c002_a341")]
                                    // "c002_a341" => Some((c002::c002_a341::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a341::KERNEL)),

                                    // #[cfg(feature = "c002_a342")]
                                    // "c002_a342" => Some((c002::c002_a342::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a342::KERNEL)),

                                    // #[cfg(feature = "c002_a343")]
                                    // "c002_a343" => Some((c002::c002_a343::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a343::KERNEL)),

                                    // #[cfg(feature = "c002_a344")]
                                    // "c002_a344" => Some((c002::c002_a344::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a344::KERNEL)),

                                    // #[cfg(feature = "c002_a345")]
                                    // "c002_a345" => Some((c002::c002_a345::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a345::KERNEL)),

                                    // #[cfg(feature = "c002_a346")]
                                    // "c002_a346" => Some((c002::c002_a346::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a346::KERNEL)),

                                    // #[cfg(feature = "c002_a347")]
                                    // "c002_a347" => Some((c002::c002_a347::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a347::KERNEL)),

                                    // #[cfg(feature = "c002_a348")]
                                    // "c002_a348" => Some((c002::c002_a348::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a348::KERNEL)),

                                    // #[cfg(feature = "c002_a349")]
                                    // "c002_a349" => Some((c002::c002_a349::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a349::KERNEL)),

                                    // #[cfg(feature = "c002_a350")]
                                    // "c002_a350" => Some((c002::c002_a350::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a350::KERNEL)),

                                    // #[cfg(feature = "c002_a351")]
                                    // "c002_a351" => Some((c002::c002_a351::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a351::KERNEL)),

                                    // #[cfg(feature = "c002_a352")]
                                    // "c002_a352" => Some((c002::c002_a352::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a352::KERNEL)),

                                    // #[cfg(feature = "c002_a353")]
                                    // "c002_a353" => Some((c002::c002_a353::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a353::KERNEL)),

                                    // #[cfg(feature = "c002_a354")]
                                    // "c002_a354" => Some((c002::c002_a354::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a354::KERNEL)),

                                    // #[cfg(feature = "c002_a355")]
                                    // "c002_a355" => Some((c002::c002_a355::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a355::KERNEL)),

                                    // #[cfg(feature = "c002_a356")]
                                    // "c002_a356" => Some((c002::c002_a356::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a356::KERNEL)),

                                    // #[cfg(feature = "c002_a357")]
                                    // "c002_a357" => Some((c002::c002_a357::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a357::KERNEL)),

                                    // #[cfg(feature = "c002_a358")]
                                    // "c002_a358" => Some((c002::c002_a358::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a358::KERNEL)),

                                    // #[cfg(feature = "c002_a359")]
                                    // "c002_a359" => Some((c002::c002_a359::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a359::KERNEL)),

                                    // #[cfg(feature = "c002_a360")]
                                    // "c002_a360" => Some((c002::c002_a360::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a360::KERNEL)),

                                    // #[cfg(feature = "c002_a361")]
                                    // "c002_a361" => Some((c002::c002_a361::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a361::KERNEL)),

                                    // #[cfg(feature = "c002_a362")]
                                    // "c002_a362" => Some((c002::c002_a362::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a362::KERNEL)),

                                    // #[cfg(feature = "c002_a363")]
                                    // "c002_a363" => Some((c002::c002_a363::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a363::KERNEL)),

                                    // #[cfg(feature = "c002_a364")]
                                    // "c002_a364" => Some((c002::c002_a364::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a364::KERNEL)),

                                    // #[cfg(feature = "c002_a365")]
                                    // "c002_a365" => Some((c002::c002_a365::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a365::KERNEL)),

                                    // #[cfg(feature = "c002_a366")]
                                    // "c002_a366" => Some((c002::c002_a366::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a366::KERNEL)),

                                    // #[cfg(feature = "c002_a367")]
                                    // "c002_a367" => Some((c002::c002_a367::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a367::KERNEL)),

                                    // #[cfg(feature = "c002_a368")]
                                    // "c002_a368" => Some((c002::c002_a368::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a368::KERNEL)),

                                    // #[cfg(feature = "c002_a369")]
                                    // "c002_a369" => Some((c002::c002_a369::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a369::KERNEL)),

                                    // #[cfg(feature = "c002_a370")]
                                    // "c002_a370" => Some((c002::c002_a370::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a370::KERNEL)),

                                    // #[cfg(feature = "c002_a371")]
                                    // "c002_a371" => Some((c002::c002_a371::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a371::KERNEL)),

                                    // #[cfg(feature = "c002_a372")]
                                    // "c002_a372" => Some((c002::c002_a372::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a372::KERNEL)),

                                    // #[cfg(feature = "c002_a373")]
                                    // "c002_a373" => Some((c002::c002_a373::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a373::KERNEL)),

                                    // #[cfg(feature = "c002_a374")]
                                    // "c002_a374" => Some((c002::c002_a374::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a374::KERNEL)),

                                    // #[cfg(feature = "c002_a375")]
                                    // "c002_a375" => Some((c002::c002_a375::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a375::KERNEL)),

                                    // #[cfg(feature = "c002_a376")]
                                    // "c002_a376" => Some((c002::c002_a376::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a376::KERNEL)),

                                    // #[cfg(feature = "c002_a377")]
                                    // "c002_a377" => Some((c002::c002_a377::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a377::KERNEL)),

                                    // #[cfg(feature = "c002_a378")]
                                    // "c002_a378" => Some((c002::c002_a378::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a378::KERNEL)),

                                    // #[cfg(feature = "c002_a379")]
                                    // "c002_a379" => Some((c002::c002_a379::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a379::KERNEL)),

                                    // #[cfg(feature = "c002_a380")]
                                    // "c002_a380" => Some((c002::c002_a380::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a380::KERNEL)),

                                    // #[cfg(feature = "c002_a381")]
                                    // "c002_a381" => Some((c002::c002_a381::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a381::KERNEL)),

                                    // #[cfg(feature = "c002_a382")]
                                    // "c002_a382" => Some((c002::c002_a382::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a382::KERNEL)),

                                    // #[cfg(feature = "c002_a383")]
                                    // "c002_a383" => Some((c002::c002_a383::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a383::KERNEL)),

                                    // #[cfg(feature = "c002_a384")]
                                    // "c002_a384" => Some((c002::c002_a384::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a384::KERNEL)),

                                    // #[cfg(feature = "c002_a385")]
                                    // "c002_a385" => Some((c002::c002_a385::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a385::KERNEL)),

                                    // #[cfg(feature = "c002_a386")]
                                    // "c002_a386" => Some((c002::c002_a386::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a386::KERNEL)),

                                    // #[cfg(feature = "c002_a387")]
                                    // "c002_a387" => Some((c002::c002_a387::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a387::KERNEL)),

                                    // #[cfg(feature = "c002_a388")]
                                    // "c002_a388" => Some((c002::c002_a388::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a388::KERNEL)),

                                    // #[cfg(feature = "c002_a389")]
                                    // "c002_a389" => Some((c002::c002_a389::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a389::KERNEL)),

                                    // #[cfg(feature = "c002_a390")]
                                    // "c002_a390" => Some((c002::c002_a390::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a390::KERNEL)),

                                    // #[cfg(feature = "c002_a391")]
                                    // "c002_a391" => Some((c002::c002_a391::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a391::KERNEL)),

                                    // #[cfg(feature = "c002_a392")]
                                    // "c002_a392" => Some((c002::c002_a392::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a392::KERNEL)),

                                    // #[cfg(feature = "c002_a393")]
                                    // "c002_a393" => Some((c002::c002_a393::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a393::KERNEL)),

                                    // #[cfg(feature = "c002_a394")]
                                    // "c002_a394" => Some((c002::c002_a394::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a394::KERNEL)),

                                    // #[cfg(feature = "c002_a395")]
                                    // "c002_a395" => Some((c002::c002_a395::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a395::KERNEL)),

                                    // #[cfg(feature = "c002_a396")]
                                    // "c002_a396" => Some((c002::c002_a396::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a396::KERNEL)),

                                    // #[cfg(feature = "c002_a397")]
                                    // "c002_a397" => Some((c002::c002_a397::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a397::KERNEL)),

                                    // #[cfg(feature = "c002_a398")]
                                    // "c002_a398" => Some((c002::c002_a398::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a398::KERNEL)),

                                    // #[cfg(feature = "c002_a399")]
                                    // "c002_a399" => Some((c002::c002_a399::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a399::KERNEL)),

                                    // #[cfg(feature = "c002_a400")]
                                    // "c002_a400" => Some((c002::c002_a400::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a400::KERNEL)),

                                    // #[cfg(feature = "c002_a401")]
                                    // "c002_a401" => Some((c002::c002_a401::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a401::KERNEL)),

                                    // #[cfg(feature = "c002_a402")]
                                    // "c002_a402" => Some((c002::c002_a402::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a402::KERNEL)),

                                    // #[cfg(feature = "c002_a403")]
                                    // "c002_a403" => Some((c002::c002_a403::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a403::KERNEL)),

                                    // #[cfg(feature = "c002_a404")]
                                    // "c002_a404" => Some((c002::c002_a404::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a404::KERNEL)),

                                    // #[cfg(feature = "c002_a405")]
                                    // "c002_a405" => Some((c002::c002_a405::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a405::KERNEL)),

                                    // #[cfg(feature = "c002_a406")]
                                    // "c002_a406" => Some((c002::c002_a406::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a406::KERNEL)),

                                    // #[cfg(feature = "c002_a407")]
                                    // "c002_a407" => Some((c002::c002_a407::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a407::KERNEL)),

                                    // #[cfg(feature = "c002_a408")]
                                    // "c002_a408" => Some((c002::c002_a408::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a408::KERNEL)),

                                    // #[cfg(feature = "c002_a409")]
                                    // "c002_a409" => Some((c002::c002_a409::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a409::KERNEL)),

                                    // #[cfg(feature = "c002_a410")]
                                    // "c002_a410" => Some((c002::c002_a410::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a410::KERNEL)),

                                    // #[cfg(feature = "c002_a411")]
                                    // "c002_a411" => Some((c002::c002_a411::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a411::KERNEL)),

                                    // #[cfg(feature = "c002_a412")]
                                    // "c002_a412" => Some((c002::c002_a412::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a412::KERNEL)),

                                    // #[cfg(feature = "c002_a413")]
                                    // "c002_a413" => Some((c002::c002_a413::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a413::KERNEL)),

                                    // #[cfg(feature = "c002_a414")]
                                    // "c002_a414" => Some((c002::c002_a414::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a414::KERNEL)),

                                    // #[cfg(feature = "c002_a415")]
                                    // "c002_a415" => Some((c002::c002_a415::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a415::KERNEL)),

                                    // #[cfg(feature = "c002_a416")]
                                    // "c002_a416" => Some((c002::c002_a416::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a416::KERNEL)),

                                    // #[cfg(feature = "c002_a417")]
                                    // "c002_a417" => Some((c002::c002_a417::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a417::KERNEL)),

                                    // #[cfg(feature = "c002_a418")]
                                    // "c002_a418" => Some((c002::c002_a418::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a418::KERNEL)),

                                    // #[cfg(feature = "c002_a419")]
                                    // "c002_a419" => Some((c002::c002_a419::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a419::KERNEL)),

                                    // #[cfg(feature = "c002_a420")]
                                    // "c002_a420" => Some((c002::c002_a420::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a420::KERNEL)),

                                    // #[cfg(feature = "c002_a421")]
                                    // "c002_a421" => Some((c002::c002_a421::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a421::KERNEL)),

                                    // #[cfg(feature = "c002_a422")]
                                    // "c002_a422" => Some((c002::c002_a422::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a422::KERNEL)),

                                    // #[cfg(feature = "c002_a423")]
                                    // "c002_a423" => Some((c002::c002_a423::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a423::KERNEL)),

                                    // #[cfg(feature = "c002_a424")]
                                    // "c002_a424" => Some((c002::c002_a424::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a424::KERNEL)),

                                    // #[cfg(feature = "c002_a425")]
                                    // "c002_a425" => Some((c002::c002_a425::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a425::KERNEL)),

                                    // #[cfg(feature = "c002_a426")]
                                    // "c002_a426" => Some((c002::c002_a426::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a426::KERNEL)),

                                    // #[cfg(feature = "c002_a427")]
                                    // "c002_a427" => Some((c002::c002_a427::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a427::KERNEL)),

                                    // #[cfg(feature = "c002_a428")]
                                    // "c002_a428" => Some((c002::c002_a428::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a428::KERNEL)),

                                    // #[cfg(feature = "c002_a429")]
                                    // "c002_a429" => Some((c002::c002_a429::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a429::KERNEL)),

                                    // #[cfg(feature = "c002_a430")]
                                    // "c002_a430" => Some((c002::c002_a430::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a430::KERNEL)),

                                    // #[cfg(feature = "c002_a431")]
                                    // "c002_a431" => Some((c002::c002_a431::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a431::KERNEL)),

                                    // #[cfg(feature = "c002_a432")]
                                    // "c002_a432" => Some((c002::c002_a432::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a432::KERNEL)),

                                    // #[cfg(feature = "c002_a433")]
                                    // "c002_a433" => Some((c002::c002_a433::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a433::KERNEL)),

                                    // #[cfg(feature = "c002_a434")]
                                    // "c002_a434" => Some((c002::c002_a434::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a434::KERNEL)),

                                    // #[cfg(feature = "c002_a435")]
                                    // "c002_a435" => Some((c002::c002_a435::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a435::KERNEL)),

                                    // #[cfg(feature = "c002_a436")]
                                    // "c002_a436" => Some((c002::c002_a436::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a436::KERNEL)),

                                    // #[cfg(feature = "c002_a437")]
                                    // "c002_a437" => Some((c002::c002_a437::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a437::KERNEL)),

                                    // #[cfg(feature = "c002_a438")]
                                    // "c002_a438" => Some((c002::c002_a438::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a438::KERNEL)),

                                    // #[cfg(feature = "c002_a439")]
                                    // "c002_a439" => Some((c002::c002_a439::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a439::KERNEL)),

                                    // #[cfg(feature = "c002_a440")]
                                    // "c002_a440" => Some((c002::c002_a440::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a440::KERNEL)),

                                    // #[cfg(feature = "c002_a441")]
                                    // "c002_a441" => Some((c002::c002_a441::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a441::KERNEL)),

                                    // #[cfg(feature = "c002_a442")]
                                    // "c002_a442" => Some((c002::c002_a442::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a442::KERNEL)),

                                    // #[cfg(feature = "c002_a443")]
                                    // "c002_a443" => Some((c002::c002_a443::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a443::KERNEL)),

                                    // #[cfg(feature = "c002_a444")]
                                    // "c002_a444" => Some((c002::c002_a444::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a444::KERNEL)),

                                    // #[cfg(feature = "c002_a445")]
                                    // "c002_a445" => Some((c002::c002_a445::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a445::KERNEL)),

                                    // #[cfg(feature = "c002_a446")]
                                    // "c002_a446" => Some((c002::c002_a446::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a446::KERNEL)),

                                    // #[cfg(feature = "c002_a447")]
                                    // "c002_a447" => Some((c002::c002_a447::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a447::KERNEL)),

                                    // #[cfg(feature = "c002_a448")]
                                    // "c002_a448" => Some((c002::c002_a448::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a448::KERNEL)),

                                    // #[cfg(feature = "c002_a449")]
                                    // "c002_a449" => Some((c002::c002_a449::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a449::KERNEL)),

                                    // #[cfg(feature = "c002_a450")]
                                    // "c002_a450" => Some((c002::c002_a450::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a450::KERNEL)),

                                    // #[cfg(feature = "c002_a451")]
                                    // "c002_a451" => Some((c002::c002_a451::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a451::KERNEL)),

                                    // #[cfg(feature = "c002_a452")]
                                    // "c002_a452" => Some((c002::c002_a452::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a452::KERNEL)),

                                    // #[cfg(feature = "c002_a453")]
                                    // "c002_a453" => Some((c002::c002_a453::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a453::KERNEL)),

                                    // #[cfg(feature = "c002_a454")]
                                    // "c002_a454" => Some((c002::c002_a454::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a454::KERNEL)),

                                    // #[cfg(feature = "c002_a455")]
                                    // "c002_a455" => Some((c002::c002_a455::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a455::KERNEL)),

                                    // #[cfg(feature = "c002_a456")]
                                    // "c002_a456" => Some((c002::c002_a456::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a456::KERNEL)),

                                    // #[cfg(feature = "c002_a457")]
                                    // "c002_a457" => Some((c002::c002_a457::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a457::KERNEL)),

                                    // #[cfg(feature = "c002_a458")]
                                    // "c002_a458" => Some((c002::c002_a458::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a458::KERNEL)),

                                    // #[cfg(feature = "c002_a459")]
                                    // "c002_a459" => Some((c002::c002_a459::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a459::KERNEL)),

                                    // #[cfg(feature = "c002_a460")]
                                    // "c002_a460" => Some((c002::c002_a460::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a460::KERNEL)),

                                    // #[cfg(feature = "c002_a461")]
                                    // "c002_a461" => Some((c002::c002_a461::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a461::KERNEL)),

                                    // #[cfg(feature = "c002_a462")]
                                    // "c002_a462" => Some((c002::c002_a462::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a462::KERNEL)),

                                    // #[cfg(feature = "c002_a463")]
                                    // "c002_a463" => Some((c002::c002_a463::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a463::KERNEL)),

                                    // #[cfg(feature = "c002_a464")]
                                    // "c002_a464" => Some((c002::c002_a464::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a464::KERNEL)),

                                    // #[cfg(feature = "c002_a465")]
                                    // "c002_a465" => Some((c002::c002_a465::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a465::KERNEL)),

                                    // #[cfg(feature = "c002_a466")]
                                    // "c002_a466" => Some((c002::c002_a466::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a466::KERNEL)),

                                    // #[cfg(feature = "c002_a467")]
                                    // "c002_a467" => Some((c002::c002_a467::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a467::KERNEL)),

                                    // #[cfg(feature = "c002_a468")]
                                    // "c002_a468" => Some((c002::c002_a468::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a468::KERNEL)),

                                    // #[cfg(feature = "c002_a469")]
                                    // "c002_a469" => Some((c002::c002_a469::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a469::KERNEL)),

                                    // #[cfg(feature = "c002_a470")]
                                    // "c002_a470" => Some((c002::c002_a470::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a470::KERNEL)),

                                    // #[cfg(feature = "c002_a471")]
                                    // "c002_a471" => Some((c002::c002_a471::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a471::KERNEL)),

                                    // #[cfg(feature = "c002_a472")]
                                    // "c002_a472" => Some((c002::c002_a472::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a472::KERNEL)),

                                    // #[cfg(feature = "c002_a473")]
                                    // "c002_a473" => Some((c002::c002_a473::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a473::KERNEL)),

                                    // #[cfg(feature = "c002_a474")]
                                    // "c002_a474" => Some((c002::c002_a474::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a474::KERNEL)),

                                    // #[cfg(feature = "c002_a475")]
                                    // "c002_a475" => Some((c002::c002_a475::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a475::KERNEL)),

                                    // #[cfg(feature = "c002_a476")]
                                    // "c002_a476" => Some((c002::c002_a476::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a476::KERNEL)),

                                    // #[cfg(feature = "c002_a477")]
                                    // "c002_a477" => Some((c002::c002_a477::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a477::KERNEL)),

                                    // #[cfg(feature = "c002_a478")]
                                    // "c002_a478" => Some((c002::c002_a478::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a478::KERNEL)),

                                    // #[cfg(feature = "c002_a479")]
                                    // "c002_a479" => Some((c002::c002_a479::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a479::KERNEL)),

                                    // #[cfg(feature = "c002_a480")]
                                    // "c002_a480" => Some((c002::c002_a480::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a480::KERNEL)),

                                    // #[cfg(feature = "c002_a481")]
                                    // "c002_a481" => Some((c002::c002_a481::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a481::KERNEL)),

                                    // #[cfg(feature = "c002_a482")]
                                    // "c002_a482" => Some((c002::c002_a482::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a482::KERNEL)),

                                    // #[cfg(feature = "c002_a483")]
                                    // "c002_a483" => Some((c002::c002_a483::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a483::KERNEL)),

                                    // #[cfg(feature = "c002_a484")]
                                    // "c002_a484" => Some((c002::c002_a484::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a484::KERNEL)),

                                    // #[cfg(feature = "c002_a485")]
                                    // "c002_a485" => Some((c002::c002_a485::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a485::KERNEL)),

                                    // #[cfg(feature = "c002_a486")]
                                    // "c002_a486" => Some((c002::c002_a486::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a486::KERNEL)),

                                    // #[cfg(feature = "c002_a487")]
                                    // "c002_a487" => Some((c002::c002_a487::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a487::KERNEL)),

                                    // #[cfg(feature = "c002_a488")]
                                    // "c002_a488" => Some((c002::c002_a488::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a488::KERNEL)),

                                    // #[cfg(feature = "c002_a489")]
                                    // "c002_a489" => Some((c002::c002_a489::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a489::KERNEL)),

                                    // #[cfg(feature = "c002_a490")]
                                    // "c002_a490" => Some((c002::c002_a490::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a490::KERNEL)),

                                    // #[cfg(feature = "c002_a491")]
                                    // "c002_a491" => Some((c002::c002_a491::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a491::KERNEL)),

                                    // #[cfg(feature = "c002_a492")]
                                    // "c002_a492" => Some((c002::c002_a492::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a492::KERNEL)),

                                    // #[cfg(feature = "c002_a493")]
                                    // "c002_a493" => Some((c002::c002_a493::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a493::KERNEL)),

                                    // #[cfg(feature = "c002_a494")]
                                    // "c002_a494" => Some((c002::c002_a494::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a494::KERNEL)),

                                    // #[cfg(feature = "c002_a495")]
                                    // "c002_a495" => Some((c002::c002_a495::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a495::KERNEL)),

                                    // #[cfg(feature = "c002_a496")]
                                    // "c002_a496" => Some((c002::c002_a496::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a496::KERNEL)),

                                    // #[cfg(feature = "c002_a497")]
                                    // "c002_a497" => Some((c002::c002_a497::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a497::KERNEL)),

                                    // #[cfg(feature = "c002_a498")]
                                    // "c002_a498" => Some((c002::c002_a498::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a498::KERNEL)),

                                    // #[cfg(feature = "c002_a499")]
                                    // "c002_a499" => Some((c002::c002_a499::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a499::KERNEL)),

                                    // #[cfg(feature = "c002_a500")]
                                    // "c002_a500" => Some((c002::c002_a500::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a500::KERNEL)),

                                    // #[cfg(feature = "c002_a501")]
                                    // "c002_a501" => Some((c002::c002_a501::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a501::KERNEL)),

                                    // #[cfg(feature = "c002_a502")]
                                    // "c002_a502" => Some((c002::c002_a502::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a502::KERNEL)),

                                    // #[cfg(feature = "c002_a503")]
                                    // "c002_a503" => Some((c002::c002_a503::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a503::KERNEL)),

                                    // #[cfg(feature = "c002_a504")]
                                    // "c002_a504" => Some((c002::c002_a504::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a504::KERNEL)),

                                    // #[cfg(feature = "c002_a505")]
                                    // "c002_a505" => Some((c002::c002_a505::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a505::KERNEL)),

                                    // #[cfg(feature = "c002_a506")]
                                    // "c002_a506" => Some((c002::c002_a506::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a506::KERNEL)),

                                    // #[cfg(feature = "c002_a507")]
                                    // "c002_a507" => Some((c002::c002_a507::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a507::KERNEL)),

                                    // #[cfg(feature = "c002_a508")]
                                    // "c002_a508" => Some((c002::c002_a508::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a508::KERNEL)),

                                    // #[cfg(feature = "c002_a509")]
                                    // "c002_a509" => Some((c002::c002_a509::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a509::KERNEL)),

                                    // #[cfg(feature = "c002_a510")]
                                    // "c002_a510" => Some((c002::c002_a510::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a510::KERNEL)),

                                    // #[cfg(feature = "c002_a511")]
                                    // "c002_a511" => Some((c002::c002_a511::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a511::KERNEL)),

                                    // #[cfg(feature = "c002_a512")]
                                    // "c002_a512" => Some((c002::c002_a512::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a512::KERNEL)),

                                    // #[cfg(feature = "c002_a513")]
                                    // "c002_a513" => Some((c002::c002_a513::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a513::KERNEL)),

                                    // #[cfg(feature = "c002_a514")]
                                    // "c002_a514" => Some((c002::c002_a514::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a514::KERNEL)),

                                    // #[cfg(feature = "c002_a515")]
                                    // "c002_a515" => Some((c002::c002_a515::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a515::KERNEL)),

                                    // #[cfg(feature = "c002_a516")]
                                    // "c002_a516" => Some((c002::c002_a516::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a516::KERNEL)),

                                    // #[cfg(feature = "c002_a517")]
                                    // "c002_a517" => Some((c002::c002_a517::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a517::KERNEL)),

                                    // #[cfg(feature = "c002_a518")]
                                    // "c002_a518" => Some((c002::c002_a518::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a518::KERNEL)),

                                    // #[cfg(feature = "c002_a519")]
                                    // "c002_a519" => Some((c002::c002_a519::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a519::KERNEL)),

                                    // #[cfg(feature = "c002_a520")]
                                    // "c002_a520" => Some((c002::c002_a520::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a520::KERNEL)),

                                    // #[cfg(feature = "c002_a521")]
                                    // "c002_a521" => Some((c002::c002_a521::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a521::KERNEL)),

                                    // #[cfg(feature = "c002_a522")]
                                    // "c002_a522" => Some((c002::c002_a522::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a522::KERNEL)),

                                    // #[cfg(feature = "c002_a523")]
                                    // "c002_a523" => Some((c002::c002_a523::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a523::KERNEL)),

                                    // #[cfg(feature = "c002_a524")]
                                    // "c002_a524" => Some((c002::c002_a524::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a524::KERNEL)),

                                    // #[cfg(feature = "c002_a525")]
                                    // "c002_a525" => Some((c002::c002_a525::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a525::KERNEL)),

                                    // #[cfg(feature = "c002_a526")]
                                    // "c002_a526" => Some((c002::c002_a526::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a526::KERNEL)),

                                    // #[cfg(feature = "c002_a527")]
                                    // "c002_a527" => Some((c002::c002_a527::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a527::KERNEL)),

                                    // #[cfg(feature = "c002_a528")]
                                    // "c002_a528" => Some((c002::c002_a528::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a528::KERNEL)),

                                    // #[cfg(feature = "c002_a529")]
                                    // "c002_a529" => Some((c002::c002_a529::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a529::KERNEL)),

                                    // #[cfg(feature = "c002_a530")]
                                    // "c002_a530" => Some((c002::c002_a530::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a530::KERNEL)),

                                    // #[cfg(feature = "c002_a531")]
                                    // "c002_a531" => Some((c002::c002_a531::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a531::KERNEL)),

                                    // #[cfg(feature = "c002_a532")]
                                    // "c002_a532" => Some((c002::c002_a532::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a532::KERNEL)),

                                    // #[cfg(feature = "c002_a533")]
                                    // "c002_a533" => Some((c002::c002_a533::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a533::KERNEL)),

                                    // #[cfg(feature = "c002_a534")]
                                    // "c002_a534" => Some((c002::c002_a534::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a534::KERNEL)),

                                    // #[cfg(feature = "c002_a535")]
                                    // "c002_a535" => Some((c002::c002_a535::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a535::KERNEL)),

                                    // #[cfg(feature = "c002_a536")]
                                    // "c002_a536" => Some((c002::c002_a536::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a536::KERNEL)),

                                    // #[cfg(feature = "c002_a537")]
                                    // "c002_a537" => Some((c002::c002_a537::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a537::KERNEL)),

                                    // #[cfg(feature = "c002_a538")]
                                    // "c002_a538" => Some((c002::c002_a538::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a538::KERNEL)),

                                    // #[cfg(feature = "c002_a539")]
                                    // "c002_a539" => Some((c002::c002_a539::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a539::KERNEL)),

                                    // #[cfg(feature = "c002_a540")]
                                    // "c002_a540" => Some((c002::c002_a540::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a540::KERNEL)),

                                    // #[cfg(feature = "c002_a541")]
                                    // "c002_a541" => Some((c002::c002_a541::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a541::KERNEL)),

                                    // #[cfg(feature = "c002_a542")]
                                    // "c002_a542" => Some((c002::c002_a542::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a542::KERNEL)),

                                    // #[cfg(feature = "c002_a543")]
                                    // "c002_a543" => Some((c002::c002_a543::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a543::KERNEL)),

                                    // #[cfg(feature = "c002_a544")]
                                    // "c002_a544" => Some((c002::c002_a544::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a544::KERNEL)),

                                    // #[cfg(feature = "c002_a545")]
                                    // "c002_a545" => Some((c002::c002_a545::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a545::KERNEL)),

                                    // #[cfg(feature = "c002_a546")]
                                    // "c002_a546" => Some((c002::c002_a546::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a546::KERNEL)),

                                    // #[cfg(feature = "c002_a547")]
                                    // "c002_a547" => Some((c002::c002_a547::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a547::KERNEL)),

                                    // #[cfg(feature = "c002_a548")]
                                    // "c002_a548" => Some((c002::c002_a548::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a548::KERNEL)),

                                    // #[cfg(feature = "c002_a549")]
                                    // "c002_a549" => Some((c002::c002_a549::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a549::KERNEL)),

                                    // #[cfg(feature = "c002_a550")]
                                    // "c002_a550" => Some((c002::c002_a550::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a550::KERNEL)),

                                    // #[cfg(feature = "c002_a551")]
                                    // "c002_a551" => Some((c002::c002_a551::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a551::KERNEL)),

                                    // #[cfg(feature = "c002_a552")]
                                    // "c002_a552" => Some((c002::c002_a552::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a552::KERNEL)),

                                    // #[cfg(feature = "c002_a553")]
                                    // "c002_a553" => Some((c002::c002_a553::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a553::KERNEL)),

                                    // #[cfg(feature = "c002_a554")]
                                    // "c002_a554" => Some((c002::c002_a554::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a554::KERNEL)),

                                    // #[cfg(feature = "c002_a555")]
                                    // "c002_a555" => Some((c002::c002_a555::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a555::KERNEL)),

                                    // #[cfg(feature = "c002_a556")]
                                    // "c002_a556" => Some((c002::c002_a556::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a556::KERNEL)),

                                    // #[cfg(feature = "c002_a557")]
                                    // "c002_a557" => Some((c002::c002_a557::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a557::KERNEL)),

                                    // #[cfg(feature = "c002_a558")]
                                    // "c002_a558" => Some((c002::c002_a558::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a558::KERNEL)),

                                    // #[cfg(feature = "c002_a559")]
                                    // "c002_a559" => Some((c002::c002_a559::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a559::KERNEL)),

                                    // #[cfg(feature = "c002_a560")]
                                    // "c002_a560" => Some((c002::c002_a560::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a560::KERNEL)),

                                    // #[cfg(feature = "c002_a561")]
                                    // "c002_a561" => Some((c002::c002_a561::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a561::KERNEL)),

                                    // #[cfg(feature = "c002_a562")]
                                    // "c002_a562" => Some((c002::c002_a562::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a562::KERNEL)),

                                    // #[cfg(feature = "c002_a563")]
                                    // "c002_a563" => Some((c002::c002_a563::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a563::KERNEL)),

                                    // #[cfg(feature = "c002_a564")]
                                    // "c002_a564" => Some((c002::c002_a564::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a564::KERNEL)),

                                    // #[cfg(feature = "c002_a565")]
                                    // "c002_a565" => Some((c002::c002_a565::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a565::KERNEL)),

                                    // #[cfg(feature = "c002_a566")]
                                    // "c002_a566" => Some((c002::c002_a566::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a566::KERNEL)),

                                    // #[cfg(feature = "c002_a567")]
                                    // "c002_a567" => Some((c002::c002_a567::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a567::KERNEL)),

                                    // #[cfg(feature = "c002_a568")]
                                    // "c002_a568" => Some((c002::c002_a568::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a568::KERNEL)),

                                    // #[cfg(feature = "c002_a569")]
                                    // "c002_a569" => Some((c002::c002_a569::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a569::KERNEL)),

                                    // #[cfg(feature = "c002_a570")]
                                    // "c002_a570" => Some((c002::c002_a570::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a570::KERNEL)),

                                    // #[cfg(feature = "c002_a571")]
                                    // "c002_a571" => Some((c002::c002_a571::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a571::KERNEL)),

                                    // #[cfg(feature = "c002_a572")]
                                    // "c002_a572" => Some((c002::c002_a572::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a572::KERNEL)),

                                    // #[cfg(feature = "c002_a573")]
                                    // "c002_a573" => Some((c002::c002_a573::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a573::KERNEL)),

                                    // #[cfg(feature = "c002_a574")]
                                    // "c002_a574" => Some((c002::c002_a574::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a574::KERNEL)),

                                    // #[cfg(feature = "c002_a575")]
                                    // "c002_a575" => Some((c002::c002_a575::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a575::KERNEL)),

                                    // #[cfg(feature = "c002_a576")]
                                    // "c002_a576" => Some((c002::c002_a576::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a576::KERNEL)),

                                    // #[cfg(feature = "c002_a577")]
                                    // "c002_a577" => Some((c002::c002_a577::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a577::KERNEL)),

                                    // #[cfg(feature = "c002_a578")]
                                    // "c002_a578" => Some((c002::c002_a578::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a578::KERNEL)),

                                    // #[cfg(feature = "c002_a579")]
                                    // "c002_a579" => Some((c002::c002_a579::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a579::KERNEL)),

                                    // #[cfg(feature = "c002_a580")]
                                    // "c002_a580" => Some((c002::c002_a580::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a580::KERNEL)),

                                    // #[cfg(feature = "c002_a581")]
                                    // "c002_a581" => Some((c002::c002_a581::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a581::KERNEL)),

                                    // #[cfg(feature = "c002_a582")]
                                    // "c002_a582" => Some((c002::c002_a582::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a582::KERNEL)),

                                    // #[cfg(feature = "c002_a583")]
                                    // "c002_a583" => Some((c002::c002_a583::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a583::KERNEL)),

                                    // #[cfg(feature = "c002_a584")]
                                    // "c002_a584" => Some((c002::c002_a584::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a584::KERNEL)),

                                    // #[cfg(feature = "c002_a585")]
                                    // "c002_a585" => Some((c002::c002_a585::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a585::KERNEL)),

                                    // #[cfg(feature = "c002_a586")]
                                    // "c002_a586" => Some((c002::c002_a586::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a586::KERNEL)),

                                    // #[cfg(feature = "c002_a587")]
                                    // "c002_a587" => Some((c002::c002_a587::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a587::KERNEL)),

                                    // #[cfg(feature = "c002_a588")]
                                    // "c002_a588" => Some((c002::c002_a588::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a588::KERNEL)),

                                    // #[cfg(feature = "c002_a589")]
                                    // "c002_a589" => Some((c002::c002_a589::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a589::KERNEL)),

                                    // #[cfg(feature = "c002_a590")]
                                    // "c002_a590" => Some((c002::c002_a590::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a590::KERNEL)),

                                    // #[cfg(feature = "c002_a591")]
                                    // "c002_a591" => Some((c002::c002_a591::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a591::KERNEL)),

                                    // #[cfg(feature = "c002_a592")]
                                    // "c002_a592" => Some((c002::c002_a592::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a592::KERNEL)),

                                    // #[cfg(feature = "c002_a593")]
                                    // "c002_a593" => Some((c002::c002_a593::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a593::KERNEL)),

                                    // #[cfg(feature = "c002_a594")]
                                    // "c002_a594" => Some((c002::c002_a594::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a594::KERNEL)),

                                    // #[cfg(feature = "c002_a595")]
                                    // "c002_a595" => Some((c002::c002_a595::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a595::KERNEL)),

                                    // #[cfg(feature = "c002_a596")]
                                    // "c002_a596" => Some((c002::c002_a596::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a596::KERNEL)),

                                    // #[cfg(feature = "c002_a597")]
                                    // "c002_a597" => Some((c002::c002_a597::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a597::KERNEL)),

                                    // #[cfg(feature = "c002_a598")]
                                    // "c002_a598" => Some((c002::c002_a598::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a598::KERNEL)),

                                    // #[cfg(feature = "c002_a599")]
                                    // "c002_a599" => Some((c002::c002_a599::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a599::KERNEL)),

                                    // #[cfg(feature = "c002_a600")]
                                    // "c002_a600" => Some((c002::c002_a600::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a600::KERNEL)),

                                    // #[cfg(feature = "c002_a601")]
                                    // "c002_a601" => Some((c002::c002_a601::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a601::KERNEL)),

                                    // #[cfg(feature = "c002_a602")]
                                    // "c002_a602" => Some((c002::c002_a602::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a602::KERNEL)),

                                    // #[cfg(feature = "c002_a603")]
                                    // "c002_a603" => Some((c002::c002_a603::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a603::KERNEL)),

                                    // #[cfg(feature = "c002_a604")]
                                    // "c002_a604" => Some((c002::c002_a604::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a604::KERNEL)),

                                    // #[cfg(feature = "c002_a605")]
                                    // "c002_a605" => Some((c002::c002_a605::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a605::KERNEL)),

                                    // #[cfg(feature = "c002_a606")]
                                    // "c002_a606" => Some((c002::c002_a606::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a606::KERNEL)),

                                    // #[cfg(feature = "c002_a607")]
                                    // "c002_a607" => Some((c002::c002_a607::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a607::KERNEL)),

                                    // #[cfg(feature = "c002_a608")]
                                    // "c002_a608" => Some((c002::c002_a608::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a608::KERNEL)),

                                    // #[cfg(feature = "c002_a609")]
                                    // "c002_a609" => Some((c002::c002_a609::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a609::KERNEL)),

                                    // #[cfg(feature = "c002_a610")]
                                    // "c002_a610" => Some((c002::c002_a610::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a610::KERNEL)),

                                    // #[cfg(feature = "c002_a611")]
                                    // "c002_a611" => Some((c002::c002_a611::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a611::KERNEL)),

                                    // #[cfg(feature = "c002_a612")]
                                    // "c002_a612" => Some((c002::c002_a612::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a612::KERNEL)),

                                    // #[cfg(feature = "c002_a613")]
                                    // "c002_a613" => Some((c002::c002_a613::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a613::KERNEL)),

                                    // #[cfg(feature = "c002_a614")]
                                    // "c002_a614" => Some((c002::c002_a614::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a614::KERNEL)),

                                    // #[cfg(feature = "c002_a615")]
                                    // "c002_a615" => Some((c002::c002_a615::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a615::KERNEL)),

                                    // #[cfg(feature = "c002_a616")]
                                    // "c002_a616" => Some((c002::c002_a616::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a616::KERNEL)),

                                    // #[cfg(feature = "c002_a617")]
                                    // "c002_a617" => Some((c002::c002_a617::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a617::KERNEL)),

                                    // #[cfg(feature = "c002_a618")]
                                    // "c002_a618" => Some((c002::c002_a618::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a618::KERNEL)),

                                    // #[cfg(feature = "c002_a619")]
                                    // "c002_a619" => Some((c002::c002_a619::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a619::KERNEL)),

                                    // #[cfg(feature = "c002_a620")]
                                    // "c002_a620" => Some((c002::c002_a620::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a620::KERNEL)),

                                    // #[cfg(feature = "c002_a621")]
                                    // "c002_a621" => Some((c002::c002_a621::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a621::KERNEL)),

                                    // #[cfg(feature = "c002_a622")]
                                    // "c002_a622" => Some((c002::c002_a622::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a622::KERNEL)),

                                    // #[cfg(feature = "c002_a623")]
                                    // "c002_a623" => Some((c002::c002_a623::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a623::KERNEL)),

                                    // #[cfg(feature = "c002_a624")]
                                    // "c002_a624" => Some((c002::c002_a624::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a624::KERNEL)),

                                    // #[cfg(feature = "c002_a625")]
                                    // "c002_a625" => Some((c002::c002_a625::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a625::KERNEL)),

                                    // #[cfg(feature = "c002_a626")]
                                    // "c002_a626" => Some((c002::c002_a626::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a626::KERNEL)),

                                    // #[cfg(feature = "c002_a627")]
                                    // "c002_a627" => Some((c002::c002_a627::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a627::KERNEL)),

                                    // #[cfg(feature = "c002_a628")]
                                    // "c002_a628" => Some((c002::c002_a628::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a628::KERNEL)),

                                    // #[cfg(feature = "c002_a629")]
                                    // "c002_a629" => Some((c002::c002_a629::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a629::KERNEL)),

                                    // #[cfg(feature = "c002_a630")]
                                    // "c002_a630" => Some((c002::c002_a630::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a630::KERNEL)),

                                    // #[cfg(feature = "c002_a631")]
                                    // "c002_a631" => Some((c002::c002_a631::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a631::KERNEL)),

                                    // #[cfg(feature = "c002_a632")]
                                    // "c002_a632" => Some((c002::c002_a632::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a632::KERNEL)),

                                    // #[cfg(feature = "c002_a633")]
                                    // "c002_a633" => Some((c002::c002_a633::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a633::KERNEL)),

                                    // #[cfg(feature = "c002_a634")]
                                    // "c002_a634" => Some((c002::c002_a634::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a634::KERNEL)),

                                    // #[cfg(feature = "c002_a635")]
                                    // "c002_a635" => Some((c002::c002_a635::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a635::KERNEL)),

                                    // #[cfg(feature = "c002_a636")]
                                    // "c002_a636" => Some((c002::c002_a636::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a636::KERNEL)),

                                    // #[cfg(feature = "c002_a637")]
                                    // "c002_a637" => Some((c002::c002_a637::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a637::KERNEL)),

                                    // #[cfg(feature = "c002_a638")]
                                    // "c002_a638" => Some((c002::c002_a638::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a638::KERNEL)),

                                    // #[cfg(feature = "c002_a639")]
                                    // "c002_a639" => Some((c002::c002_a639::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a639::KERNEL)),

                                    // #[cfg(feature = "c002_a640")]
                                    // "c002_a640" => Some((c002::c002_a640::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a640::KERNEL)),

                                    // #[cfg(feature = "c002_a641")]
                                    // "c002_a641" => Some((c002::c002_a641::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a641::KERNEL)),

                                    // #[cfg(feature = "c002_a642")]
                                    // "c002_a642" => Some((c002::c002_a642::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a642::KERNEL)),

                                    // #[cfg(feature = "c002_a643")]
                                    // "c002_a643" => Some((c002::c002_a643::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a643::KERNEL)),

                                    // #[cfg(feature = "c002_a644")]
                                    // "c002_a644" => Some((c002::c002_a644::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a644::KERNEL)),

                                    // #[cfg(feature = "c002_a645")]
                                    // "c002_a645" => Some((c002::c002_a645::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a645::KERNEL)),

                                    // #[cfg(feature = "c002_a646")]
                                    // "c002_a646" => Some((c002::c002_a646::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a646::KERNEL)),

                                    // #[cfg(feature = "c002_a647")]
                                    // "c002_a647" => Some((c002::c002_a647::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a647::KERNEL)),

                                    // #[cfg(feature = "c002_a648")]
                                    // "c002_a648" => Some((c002::c002_a648::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a648::KERNEL)),

                                    // #[cfg(feature = "c002_a649")]
                                    // "c002_a649" => Some((c002::c002_a649::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a649::KERNEL)),

                                    // #[cfg(feature = "c002_a650")]
                                    // "c002_a650" => Some((c002::c002_a650::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a650::KERNEL)),

                                    // #[cfg(feature = "c002_a651")]
                                    // "c002_a651" => Some((c002::c002_a651::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a651::KERNEL)),

                                    // #[cfg(feature = "c002_a652")]
                                    // "c002_a652" => Some((c002::c002_a652::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a652::KERNEL)),

                                    // #[cfg(feature = "c002_a653")]
                                    // "c002_a653" => Some((c002::c002_a653::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a653::KERNEL)),

                                    // #[cfg(feature = "c002_a654")]
                                    // "c002_a654" => Some((c002::c002_a654::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a654::KERNEL)),

                                    // #[cfg(feature = "c002_a655")]
                                    // "c002_a655" => Some((c002::c002_a655::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a655::KERNEL)),

                                    // #[cfg(feature = "c002_a656")]
                                    // "c002_a656" => Some((c002::c002_a656::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a656::KERNEL)),

                                    // #[cfg(feature = "c002_a657")]
                                    // "c002_a657" => Some((c002::c002_a657::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a657::KERNEL)),

                                    // #[cfg(feature = "c002_a658")]
                                    // "c002_a658" => Some((c002::c002_a658::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a658::KERNEL)),

                                    // #[cfg(feature = "c002_a659")]
                                    // "c002_a659" => Some((c002::c002_a659::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a659::KERNEL)),

                                    // #[cfg(feature = "c002_a660")]
                                    // "c002_a660" => Some((c002::c002_a660::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a660::KERNEL)),

                                    // #[cfg(feature = "c002_a661")]
                                    // "c002_a661" => Some((c002::c002_a661::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a661::KERNEL)),

                                    // #[cfg(feature = "c002_a662")]
                                    // "c002_a662" => Some((c002::c002_a662::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a662::KERNEL)),

                                    // #[cfg(feature = "c002_a663")]
                                    // "c002_a663" => Some((c002::c002_a663::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a663::KERNEL)),

                                    // #[cfg(feature = "c002_a664")]
                                    // "c002_a664" => Some((c002::c002_a664::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a664::KERNEL)),

                                    // #[cfg(feature = "c002_a665")]
                                    // "c002_a665" => Some((c002::c002_a665::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a665::KERNEL)),

                                    // #[cfg(feature = "c002_a666")]
                                    // "c002_a666" => Some((c002::c002_a666::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a666::KERNEL)),

                                    // #[cfg(feature = "c002_a667")]
                                    // "c002_a667" => Some((c002::c002_a667::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a667::KERNEL)),

                                    // #[cfg(feature = "c002_a668")]
                                    // "c002_a668" => Some((c002::c002_a668::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a668::KERNEL)),

                                    // #[cfg(feature = "c002_a669")]
                                    // "c002_a669" => Some((c002::c002_a669::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a669::KERNEL)),

                                    // #[cfg(feature = "c002_a670")]
                                    // "c002_a670" => Some((c002::c002_a670::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a670::KERNEL)),

                                    // #[cfg(feature = "c002_a671")]
                                    // "c002_a671" => Some((c002::c002_a671::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a671::KERNEL)),

                                    // #[cfg(feature = "c002_a672")]
                                    // "c002_a672" => Some((c002::c002_a672::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a672::KERNEL)),

                                    // #[cfg(feature = "c002_a673")]
                                    // "c002_a673" => Some((c002::c002_a673::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a673::KERNEL)),

                                    // #[cfg(feature = "c002_a674")]
                                    // "c002_a674" => Some((c002::c002_a674::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a674::KERNEL)),

                                    // #[cfg(feature = "c002_a675")]
                                    // "c002_a675" => Some((c002::c002_a675::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a675::KERNEL)),

                                    // #[cfg(feature = "c002_a676")]
                                    // "c002_a676" => Some((c002::c002_a676::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a676::KERNEL)),

                                    // #[cfg(feature = "c002_a677")]
                                    // "c002_a677" => Some((c002::c002_a677::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a677::KERNEL)),

                                    // #[cfg(feature = "c002_a678")]
                                    // "c002_a678" => Some((c002::c002_a678::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a678::KERNEL)),

                                    // #[cfg(feature = "c002_a679")]
                                    // "c002_a679" => Some((c002::c002_a679::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a679::KERNEL)),

                                    // #[cfg(feature = "c002_a680")]
                                    // "c002_a680" => Some((c002::c002_a680::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a680::KERNEL)),

                                    // #[cfg(feature = "c002_a681")]
                                    // "c002_a681" => Some((c002::c002_a681::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a681::KERNEL)),

                                    // #[cfg(feature = "c002_a682")]
                                    // "c002_a682" => Some((c002::c002_a682::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a682::KERNEL)),

                                    // #[cfg(feature = "c002_a683")]
                                    // "c002_a683" => Some((c002::c002_a683::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a683::KERNEL)),

                                    // #[cfg(feature = "c002_a684")]
                                    // "c002_a684" => Some((c002::c002_a684::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a684::KERNEL)),

                                    // #[cfg(feature = "c002_a685")]
                                    // "c002_a685" => Some((c002::c002_a685::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a685::KERNEL)),

                                    // #[cfg(feature = "c002_a686")]
                                    // "c002_a686" => Some((c002::c002_a686::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a686::KERNEL)),

                                    // #[cfg(feature = "c002_a687")]
                                    // "c002_a687" => Some((c002::c002_a687::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a687::KERNEL)),

                                    // #[cfg(feature = "c002_a688")]
                                    // "c002_a688" => Some((c002::c002_a688::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a688::KERNEL)),

                                    // #[cfg(feature = "c002_a689")]
                                    // "c002_a689" => Some((c002::c002_a689::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a689::KERNEL)),

                                    // #[cfg(feature = "c002_a690")]
                                    // "c002_a690" => Some((c002::c002_a690::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a690::KERNEL)),

                                    // #[cfg(feature = "c002_a691")]
                                    // "c002_a691" => Some((c002::c002_a691::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a691::KERNEL)),

                                    // #[cfg(feature = "c002_a692")]
                                    // "c002_a692" => Some((c002::c002_a692::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a692::KERNEL)),

                                    // #[cfg(feature = "c002_a693")]
                                    // "c002_a693" => Some((c002::c002_a693::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a693::KERNEL)),

                                    // #[cfg(feature = "c002_a694")]
                                    // "c002_a694" => Some((c002::c002_a694::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a694::KERNEL)),

                                    // #[cfg(feature = "c002_a695")]
                                    // "c002_a695" => Some((c002::c002_a695::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a695::KERNEL)),

                                    // #[cfg(feature = "c002_a696")]
                                    // "c002_a696" => Some((c002::c002_a696::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a696::KERNEL)),

                                    // #[cfg(feature = "c002_a697")]
                                    // "c002_a697" => Some((c002::c002_a697::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a697::KERNEL)),

                                    // #[cfg(feature = "c002_a698")]
                                    // "c002_a698" => Some((c002::c002_a698::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a698::KERNEL)),

                                    // #[cfg(feature = "c002_a699")]
                                    // "c002_a699" => Some((c002::c002_a699::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a699::KERNEL)),

                                    // #[cfg(feature = "c002_a700")]
                                    // "c002_a700" => Some((c002::c002_a700::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a700::KERNEL)),

                                    // #[cfg(feature = "c002_a701")]
                                    // "c002_a701" => Some((c002::c002_a701::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a701::KERNEL)),

                                    // #[cfg(feature = "c002_a702")]
                                    // "c002_a702" => Some((c002::c002_a702::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a702::KERNEL)),

                                    // #[cfg(feature = "c002_a703")]
                                    // "c002_a703" => Some((c002::c002_a703::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a703::KERNEL)),

                                    // #[cfg(feature = "c002_a704")]
                                    // "c002_a704" => Some((c002::c002_a704::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a704::KERNEL)),

                                    // #[cfg(feature = "c002_a705")]
                                    // "c002_a705" => Some((c002::c002_a705::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a705::KERNEL)),

                                    // #[cfg(feature = "c002_a706")]
                                    // "c002_a706" => Some((c002::c002_a706::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a706::KERNEL)),

                                    // #[cfg(feature = "c002_a707")]
                                    // "c002_a707" => Some((c002::c002_a707::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a707::KERNEL)),

                                    // #[cfg(feature = "c002_a708")]
                                    // "c002_a708" => Some((c002::c002_a708::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a708::KERNEL)),

                                    // #[cfg(feature = "c002_a709")]
                                    // "c002_a709" => Some((c002::c002_a709::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a709::KERNEL)),

                                    // #[cfg(feature = "c002_a710")]
                                    // "c002_a710" => Some((c002::c002_a710::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a710::KERNEL)),

                                    // #[cfg(feature = "c002_a711")]
                                    // "c002_a711" => Some((c002::c002_a711::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a711::KERNEL)),

                                    // #[cfg(feature = "c002_a712")]
                                    // "c002_a712" => Some((c002::c002_a712::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a712::KERNEL)),

                                    // #[cfg(feature = "c002_a713")]
                                    // "c002_a713" => Some((c002::c002_a713::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a713::KERNEL)),

                                    // #[cfg(feature = "c002_a714")]
                                    // "c002_a714" => Some((c002::c002_a714::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a714::KERNEL)),

                                    // #[cfg(feature = "c002_a715")]
                                    // "c002_a715" => Some((c002::c002_a715::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a715::KERNEL)),

                                    // #[cfg(feature = "c002_a716")]
                                    // "c002_a716" => Some((c002::c002_a716::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a716::KERNEL)),

                                    // #[cfg(feature = "c002_a717")]
                                    // "c002_a717" => Some((c002::c002_a717::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a717::KERNEL)),

                                    // #[cfg(feature = "c002_a718")]
                                    // "c002_a718" => Some((c002::c002_a718::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a718::KERNEL)),

                                    // #[cfg(feature = "c002_a719")]
                                    // "c002_a719" => Some((c002::c002_a719::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a719::KERNEL)),

                                    // #[cfg(feature = "c002_a720")]
                                    // "c002_a720" => Some((c002::c002_a720::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a720::KERNEL)),

                                    // #[cfg(feature = "c002_a721")]
                                    // "c002_a721" => Some((c002::c002_a721::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a721::KERNEL)),

                                    // #[cfg(feature = "c002_a722")]
                                    // "c002_a722" => Some((c002::c002_a722::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a722::KERNEL)),

                                    // #[cfg(feature = "c002_a723")]
                                    // "c002_a723" => Some((c002::c002_a723::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a723::KERNEL)),

                                    // #[cfg(feature = "c002_a724")]
                                    // "c002_a724" => Some((c002::c002_a724::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a724::KERNEL)),

                                    // #[cfg(feature = "c002_a725")]
                                    // "c002_a725" => Some((c002::c002_a725::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a725::KERNEL)),

                                    // #[cfg(feature = "c002_a726")]
                                    // "c002_a726" => Some((c002::c002_a726::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a726::KERNEL)),

                                    // #[cfg(feature = "c002_a727")]
                                    // "c002_a727" => Some((c002::c002_a727::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a727::KERNEL)),

                                    // #[cfg(feature = "c002_a728")]
                                    // "c002_a728" => Some((c002::c002_a728::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a728::KERNEL)),

                                    // #[cfg(feature = "c002_a729")]
                                    // "c002_a729" => Some((c002::c002_a729::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a729::KERNEL)),

                                    // #[cfg(feature = "c002_a730")]
                                    // "c002_a730" => Some((c002::c002_a730::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a730::KERNEL)),

                                    // #[cfg(feature = "c002_a731")]
                                    // "c002_a731" => Some((c002::c002_a731::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a731::KERNEL)),

                                    // #[cfg(feature = "c002_a732")]
                                    // "c002_a732" => Some((c002::c002_a732::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a732::KERNEL)),

                                    // #[cfg(feature = "c002_a733")]
                                    // "c002_a733" => Some((c002::c002_a733::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a733::KERNEL)),

                                    // #[cfg(feature = "c002_a734")]
                                    // "c002_a734" => Some((c002::c002_a734::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a734::KERNEL)),

                                    // #[cfg(feature = "c002_a735")]
                                    // "c002_a735" => Some((c002::c002_a735::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a735::KERNEL)),

                                    // #[cfg(feature = "c002_a736")]
                                    // "c002_a736" => Some((c002::c002_a736::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a736::KERNEL)),

                                    // #[cfg(feature = "c002_a737")]
                                    // "c002_a737" => Some((c002::c002_a737::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a737::KERNEL)),

                                    // #[cfg(feature = "c002_a738")]
                                    // "c002_a738" => Some((c002::c002_a738::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a738::KERNEL)),

                                    // #[cfg(feature = "c002_a739")]
                                    // "c002_a739" => Some((c002::c002_a739::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a739::KERNEL)),

                                    // #[cfg(feature = "c002_a740")]
                                    // "c002_a740" => Some((c002::c002_a740::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a740::KERNEL)),

                                    // #[cfg(feature = "c002_a741")]
                                    // "c002_a741" => Some((c002::c002_a741::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a741::KERNEL)),

                                    // #[cfg(feature = "c002_a742")]
                                    // "c002_a742" => Some((c002::c002_a742::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a742::KERNEL)),

                                    // #[cfg(feature = "c002_a743")]
                                    // "c002_a743" => Some((c002::c002_a743::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a743::KERNEL)),

                                    // #[cfg(feature = "c002_a744")]
                                    // "c002_a744" => Some((c002::c002_a744::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a744::KERNEL)),

                                    // #[cfg(feature = "c002_a745")]
                                    // "c002_a745" => Some((c002::c002_a745::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a745::KERNEL)),

                                    // #[cfg(feature = "c002_a746")]
                                    // "c002_a746" => Some((c002::c002_a746::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a746::KERNEL)),

                                    // #[cfg(feature = "c002_a747")]
                                    // "c002_a747" => Some((c002::c002_a747::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a747::KERNEL)),

                                    // #[cfg(feature = "c002_a748")]
                                    // "c002_a748" => Some((c002::c002_a748::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a748::KERNEL)),

                                    // #[cfg(feature = "c002_a749")]
                                    // "c002_a749" => Some((c002::c002_a749::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a749::KERNEL)),

                                    // #[cfg(feature = "c002_a750")]
                                    // "c002_a750" => Some((c002::c002_a750::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a750::KERNEL)),

                                    // #[cfg(feature = "c002_a751")]
                                    // "c002_a751" => Some((c002::c002_a751::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a751::KERNEL)),

                                    // #[cfg(feature = "c002_a752")]
                                    // "c002_a752" => Some((c002::c002_a752::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a752::KERNEL)),

                                    // #[cfg(feature = "c002_a753")]
                                    // "c002_a753" => Some((c002::c002_a753::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a753::KERNEL)),

                                    // #[cfg(feature = "c002_a754")]
                                    // "c002_a754" => Some((c002::c002_a754::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a754::KERNEL)),

                                    // #[cfg(feature = "c002_a755")]
                                    // "c002_a755" => Some((c002::c002_a755::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a755::KERNEL)),

                                    // #[cfg(feature = "c002_a756")]
                                    // "c002_a756" => Some((c002::c002_a756::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a756::KERNEL)),

                                    // #[cfg(feature = "c002_a757")]
                                    // "c002_a757" => Some((c002::c002_a757::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a757::KERNEL)),

                                    // #[cfg(feature = "c002_a758")]
                                    // "c002_a758" => Some((c002::c002_a758::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a758::KERNEL)),

                                    // #[cfg(feature = "c002_a759")]
                                    // "c002_a759" => Some((c002::c002_a759::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a759::KERNEL)),

                                    // #[cfg(feature = "c002_a760")]
                                    // "c002_a760" => Some((c002::c002_a760::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a760::KERNEL)),

                                    // #[cfg(feature = "c002_a761")]
                                    // "c002_a761" => Some((c002::c002_a761::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a761::KERNEL)),

                                    // #[cfg(feature = "c002_a762")]
                                    // "c002_a762" => Some((c002::c002_a762::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a762::KERNEL)),

                                    // #[cfg(feature = "c002_a763")]
                                    // "c002_a763" => Some((c002::c002_a763::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a763::KERNEL)),

                                    // #[cfg(feature = "c002_a764")]
                                    // "c002_a764" => Some((c002::c002_a764::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a764::KERNEL)),

                                    // #[cfg(feature = "c002_a765")]
                                    // "c002_a765" => Some((c002::c002_a765::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a765::KERNEL)),

                                    // #[cfg(feature = "c002_a766")]
                                    // "c002_a766" => Some((c002::c002_a766::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a766::KERNEL)),

                                    // #[cfg(feature = "c002_a767")]
                                    // "c002_a767" => Some((c002::c002_a767::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a767::KERNEL)),

                                    // #[cfg(feature = "c002_a768")]
                                    // "c002_a768" => Some((c002::c002_a768::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a768::KERNEL)),

                                    // #[cfg(feature = "c002_a769")]
                                    // "c002_a769" => Some((c002::c002_a769::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a769::KERNEL)),

                                    // #[cfg(feature = "c002_a770")]
                                    // "c002_a770" => Some((c002::c002_a770::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a770::KERNEL)),

                                    // #[cfg(feature = "c002_a771")]
                                    // "c002_a771" => Some((c002::c002_a771::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a771::KERNEL)),

                                    // #[cfg(feature = "c002_a772")]
                                    // "c002_a772" => Some((c002::c002_a772::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a772::KERNEL)),

                                    // #[cfg(feature = "c002_a773")]
                                    // "c002_a773" => Some((c002::c002_a773::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a773::KERNEL)),

                                    // #[cfg(feature = "c002_a774")]
                                    // "c002_a774" => Some((c002::c002_a774::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a774::KERNEL)),

                                    // #[cfg(feature = "c002_a775")]
                                    // "c002_a775" => Some((c002::c002_a775::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a775::KERNEL)),

                                    // #[cfg(feature = "c002_a776")]
                                    // "c002_a776" => Some((c002::c002_a776::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a776::KERNEL)),

                                    // #[cfg(feature = "c002_a777")]
                                    // "c002_a777" => Some((c002::c002_a777::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a777::KERNEL)),

                                    // #[cfg(feature = "c002_a778")]
                                    // "c002_a778" => Some((c002::c002_a778::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a778::KERNEL)),

                                    // #[cfg(feature = "c002_a779")]
                                    // "c002_a779" => Some((c002::c002_a779::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a779::KERNEL)),

                                    // #[cfg(feature = "c002_a780")]
                                    // "c002_a780" => Some((c002::c002_a780::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a780::KERNEL)),

                                    // #[cfg(feature = "c002_a781")]
                                    // "c002_a781" => Some((c002::c002_a781::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a781::KERNEL)),

                                    // #[cfg(feature = "c002_a782")]
                                    // "c002_a782" => Some((c002::c002_a782::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a782::KERNEL)),

                                    // #[cfg(feature = "c002_a783")]
                                    // "c002_a783" => Some((c002::c002_a783::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a783::KERNEL)),

                                    // #[cfg(feature = "c002_a784")]
                                    // "c002_a784" => Some((c002::c002_a784::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a784::KERNEL)),

                                    // #[cfg(feature = "c002_a785")]
                                    // "c002_a785" => Some((c002::c002_a785::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a785::KERNEL)),

                                    // #[cfg(feature = "c002_a786")]
                                    // "c002_a786" => Some((c002::c002_a786::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a786::KERNEL)),

                                    // #[cfg(feature = "c002_a787")]
                                    // "c002_a787" => Some((c002::c002_a787::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a787::KERNEL)),

                                    // #[cfg(feature = "c002_a788")]
                                    // "c002_a788" => Some((c002::c002_a788::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a788::KERNEL)),

                                    // #[cfg(feature = "c002_a789")]
                                    // "c002_a789" => Some((c002::c002_a789::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a789::KERNEL)),

                                    // #[cfg(feature = "c002_a790")]
                                    // "c002_a790" => Some((c002::c002_a790::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a790::KERNEL)),

                                    // #[cfg(feature = "c002_a791")]
                                    // "c002_a791" => Some((c002::c002_a791::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a791::KERNEL)),

                                    // #[cfg(feature = "c002_a792")]
                                    // "c002_a792" => Some((c002::c002_a792::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a792::KERNEL)),

                                    // #[cfg(feature = "c002_a793")]
                                    // "c002_a793" => Some((c002::c002_a793::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a793::KERNEL)),

                                    // #[cfg(feature = "c002_a794")]
                                    // "c002_a794" => Some((c002::c002_a794::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a794::KERNEL)),

                                    // #[cfg(feature = "c002_a795")]
                                    // "c002_a795" => Some((c002::c002_a795::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a795::KERNEL)),

                                    // #[cfg(feature = "c002_a796")]
                                    // "c002_a796" => Some((c002::c002_a796::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a796::KERNEL)),

                                    // #[cfg(feature = "c002_a797")]
                                    // "c002_a797" => Some((c002::c002_a797::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a797::KERNEL)),

                                    // #[cfg(feature = "c002_a798")]
                                    // "c002_a798" => Some((c002::c002_a798::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a798::KERNEL)),

                                    // #[cfg(feature = "c002_a799")]
                                    // "c002_a799" => Some((c002::c002_a799::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a799::KERNEL)),

                                    // #[cfg(feature = "c002_a800")]
                                    // "c002_a800" => Some((c002::c002_a800::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a800::KERNEL)),

                                    // #[cfg(feature = "c002_a801")]
                                    // "c002_a801" => Some((c002::c002_a801::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a801::KERNEL)),

                                    // #[cfg(feature = "c002_a802")]
                                    // "c002_a802" => Some((c002::c002_a802::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a802::KERNEL)),

                                    // #[cfg(feature = "c002_a803")]
                                    // "c002_a803" => Some((c002::c002_a803::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a803::KERNEL)),

                                    // #[cfg(feature = "c002_a804")]
                                    // "c002_a804" => Some((c002::c002_a804::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a804::KERNEL)),

                                    // #[cfg(feature = "c002_a805")]
                                    // "c002_a805" => Some((c002::c002_a805::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a805::KERNEL)),

                                    // #[cfg(feature = "c002_a806")]
                                    // "c002_a806" => Some((c002::c002_a806::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a806::KERNEL)),

                                    // #[cfg(feature = "c002_a807")]
                                    // "c002_a807" => Some((c002::c002_a807::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a807::KERNEL)),

                                    // #[cfg(feature = "c002_a808")]
                                    // "c002_a808" => Some((c002::c002_a808::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a808::KERNEL)),

                                    // #[cfg(feature = "c002_a809")]
                                    // "c002_a809" => Some((c002::c002_a809::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a809::KERNEL)),

                                    // #[cfg(feature = "c002_a810")]
                                    // "c002_a810" => Some((c002::c002_a810::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a810::KERNEL)),

                                    // #[cfg(feature = "c002_a811")]
                                    // "c002_a811" => Some((c002::c002_a811::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a811::KERNEL)),

                                    // #[cfg(feature = "c002_a812")]
                                    // "c002_a812" => Some((c002::c002_a812::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a812::KERNEL)),

                                    // #[cfg(feature = "c002_a813")]
                                    // "c002_a813" => Some((c002::c002_a813::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a813::KERNEL)),

                                    // #[cfg(feature = "c002_a814")]
                                    // "c002_a814" => Some((c002::c002_a814::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a814::KERNEL)),

                                    // #[cfg(feature = "c002_a815")]
                                    // "c002_a815" => Some((c002::c002_a815::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a815::KERNEL)),

                                    // #[cfg(feature = "c002_a816")]
                                    // "c002_a816" => Some((c002::c002_a816::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a816::KERNEL)),

                                    // #[cfg(feature = "c002_a817")]
                                    // "c002_a817" => Some((c002::c002_a817::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a817::KERNEL)),

                                    // #[cfg(feature = "c002_a818")]
                                    // "c002_a818" => Some((c002::c002_a818::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a818::KERNEL)),

                                    // #[cfg(feature = "c002_a819")]
                                    // "c002_a819" => Some((c002::c002_a819::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a819::KERNEL)),

                                    // #[cfg(feature = "c002_a820")]
                                    // "c002_a820" => Some((c002::c002_a820::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a820::KERNEL)),

                                    // #[cfg(feature = "c002_a821")]
                                    // "c002_a821" => Some((c002::c002_a821::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a821::KERNEL)),

                                    // #[cfg(feature = "c002_a822")]
                                    // "c002_a822" => Some((c002::c002_a822::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a822::KERNEL)),

                                    // #[cfg(feature = "c002_a823")]
                                    // "c002_a823" => Some((c002::c002_a823::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a823::KERNEL)),

                                    // #[cfg(feature = "c002_a824")]
                                    // "c002_a824" => Some((c002::c002_a824::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a824::KERNEL)),

                                    // #[cfg(feature = "c002_a825")]
                                    // "c002_a825" => Some((c002::c002_a825::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a825::KERNEL)),

                                    // #[cfg(feature = "c002_a826")]
                                    // "c002_a826" => Some((c002::c002_a826::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a826::KERNEL)),

                                    // #[cfg(feature = "c002_a827")]
                                    // "c002_a827" => Some((c002::c002_a827::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a827::KERNEL)),

                                    // #[cfg(feature = "c002_a828")]
                                    // "c002_a828" => Some((c002::c002_a828::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a828::KERNEL)),

                                    // #[cfg(feature = "c002_a829")]
                                    // "c002_a829" => Some((c002::c002_a829::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a829::KERNEL)),

                                    // #[cfg(feature = "c002_a830")]
                                    // "c002_a830" => Some((c002::c002_a830::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a830::KERNEL)),

                                    // #[cfg(feature = "c002_a831")]
                                    // "c002_a831" => Some((c002::c002_a831::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a831::KERNEL)),

                                    // #[cfg(feature = "c002_a832")]
                                    // "c002_a832" => Some((c002::c002_a832::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a832::KERNEL)),

                                    // #[cfg(feature = "c002_a833")]
                                    // "c002_a833" => Some((c002::c002_a833::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a833::KERNEL)),

                                    // #[cfg(feature = "c002_a834")]
                                    // "c002_a834" => Some((c002::c002_a834::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a834::KERNEL)),

                                    // #[cfg(feature = "c002_a835")]
                                    // "c002_a835" => Some((c002::c002_a835::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a835::KERNEL)),

                                    // #[cfg(feature = "c002_a836")]
                                    // "c002_a836" => Some((c002::c002_a836::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a836::KERNEL)),

                                    // #[cfg(feature = "c002_a837")]
                                    // "c002_a837" => Some((c002::c002_a837::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a837::KERNEL)),

                                    // #[cfg(feature = "c002_a838")]
                                    // "c002_a838" => Some((c002::c002_a838::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a838::KERNEL)),

                                    // #[cfg(feature = "c002_a839")]
                                    // "c002_a839" => Some((c002::c002_a839::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a839::KERNEL)),

                                    // #[cfg(feature = "c002_a840")]
                                    // "c002_a840" => Some((c002::c002_a840::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a840::KERNEL)),

                                    // #[cfg(feature = "c002_a841")]
                                    // "c002_a841" => Some((c002::c002_a841::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a841::KERNEL)),

                                    // #[cfg(feature = "c002_a842")]
                                    // "c002_a842" => Some((c002::c002_a842::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a842::KERNEL)),

                                    // #[cfg(feature = "c002_a843")]
                                    // "c002_a843" => Some((c002::c002_a843::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a843::KERNEL)),

                                    // #[cfg(feature = "c002_a844")]
                                    // "c002_a844" => Some((c002::c002_a844::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a844::KERNEL)),

                                    // #[cfg(feature = "c002_a845")]
                                    // "c002_a845" => Some((c002::c002_a845::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a845::KERNEL)),

                                    // #[cfg(feature = "c002_a846")]
                                    // "c002_a846" => Some((c002::c002_a846::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a846::KERNEL)),

                                    // #[cfg(feature = "c002_a847")]
                                    // "c002_a847" => Some((c002::c002_a847::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a847::KERNEL)),

                                    // #[cfg(feature = "c002_a848")]
                                    // "c002_a848" => Some((c002::c002_a848::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a848::KERNEL)),

                                    // #[cfg(feature = "c002_a849")]
                                    // "c002_a849" => Some((c002::c002_a849::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a849::KERNEL)),

                                    // #[cfg(feature = "c002_a850")]
                                    // "c002_a850" => Some((c002::c002_a850::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a850::KERNEL)),

                                    // #[cfg(feature = "c002_a851")]
                                    // "c002_a851" => Some((c002::c002_a851::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a851::KERNEL)),

                                    // #[cfg(feature = "c002_a852")]
                                    // "c002_a852" => Some((c002::c002_a852::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a852::KERNEL)),

                                    // #[cfg(feature = "c002_a853")]
                                    // "c002_a853" => Some((c002::c002_a853::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a853::KERNEL)),

                                    // #[cfg(feature = "c002_a854")]
                                    // "c002_a854" => Some((c002::c002_a854::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a854::KERNEL)),

                                    // #[cfg(feature = "c002_a855")]
                                    // "c002_a855" => Some((c002::c002_a855::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a855::KERNEL)),

                                    // #[cfg(feature = "c002_a856")]
                                    // "c002_a856" => Some((c002::c002_a856::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a856::KERNEL)),

                                    // #[cfg(feature = "c002_a857")]
                                    // "c002_a857" => Some((c002::c002_a857::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a857::KERNEL)),

                                    // #[cfg(feature = "c002_a858")]
                                    // "c002_a858" => Some((c002::c002_a858::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a858::KERNEL)),

                                    // #[cfg(feature = "c002_a859")]
                                    // "c002_a859" => Some((c002::c002_a859::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a859::KERNEL)),

                                    // #[cfg(feature = "c002_a860")]
                                    // "c002_a860" => Some((c002::c002_a860::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a860::KERNEL)),

                                    // #[cfg(feature = "c002_a861")]
                                    // "c002_a861" => Some((c002::c002_a861::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a861::KERNEL)),

                                    // #[cfg(feature = "c002_a862")]
                                    // "c002_a862" => Some((c002::c002_a862::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a862::KERNEL)),

                                    // #[cfg(feature = "c002_a863")]
                                    // "c002_a863" => Some((c002::c002_a863::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a863::KERNEL)),

                                    // #[cfg(feature = "c002_a864")]
                                    // "c002_a864" => Some((c002::c002_a864::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a864::KERNEL)),

                                    // #[cfg(feature = "c002_a865")]
                                    // "c002_a865" => Some((c002::c002_a865::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a865::KERNEL)),

                                    // #[cfg(feature = "c002_a866")]
                                    // "c002_a866" => Some((c002::c002_a866::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a866::KERNEL)),

                                    // #[cfg(feature = "c002_a867")]
                                    // "c002_a867" => Some((c002::c002_a867::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a867::KERNEL)),

                                    // #[cfg(feature = "c002_a868")]
                                    // "c002_a868" => Some((c002::c002_a868::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a868::KERNEL)),

                                    // #[cfg(feature = "c002_a869")]
                                    // "c002_a869" => Some((c002::c002_a869::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a869::KERNEL)),

                                    // #[cfg(feature = "c002_a870")]
                                    // "c002_a870" => Some((c002::c002_a870::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a870::KERNEL)),

                                    // #[cfg(feature = "c002_a871")]
                                    // "c002_a871" => Some((c002::c002_a871::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a871::KERNEL)),

                                    // #[cfg(feature = "c002_a872")]
                                    // "c002_a872" => Some((c002::c002_a872::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a872::KERNEL)),

                                    // #[cfg(feature = "c002_a873")]
                                    // "c002_a873" => Some((c002::c002_a873::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a873::KERNEL)),

                                    // #[cfg(feature = "c002_a874")]
                                    // "c002_a874" => Some((c002::c002_a874::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a874::KERNEL)),

                                    // #[cfg(feature = "c002_a875")]
                                    // "c002_a875" => Some((c002::c002_a875::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a875::KERNEL)),

                                    // #[cfg(feature = "c002_a876")]
                                    // "c002_a876" => Some((c002::c002_a876::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a876::KERNEL)),

                                    // #[cfg(feature = "c002_a877")]
                                    // "c002_a877" => Some((c002::c002_a877::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a877::KERNEL)),

                                    // #[cfg(feature = "c002_a878")]
                                    // "c002_a878" => Some((c002::c002_a878::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a878::KERNEL)),

                                    // #[cfg(feature = "c002_a879")]
                                    // "c002_a879" => Some((c002::c002_a879::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a879::KERNEL)),

                                    // #[cfg(feature = "c002_a880")]
                                    // "c002_a880" => Some((c002::c002_a880::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a880::KERNEL)),

                                    // #[cfg(feature = "c002_a881")]
                                    // "c002_a881" => Some((c002::c002_a881::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a881::KERNEL)),

                                    // #[cfg(feature = "c002_a882")]
                                    // "c002_a882" => Some((c002::c002_a882::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a882::KERNEL)),

                                    // #[cfg(feature = "c002_a883")]
                                    // "c002_a883" => Some((c002::c002_a883::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a883::KERNEL)),

                                    // #[cfg(feature = "c002_a884")]
                                    // "c002_a884" => Some((c002::c002_a884::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a884::KERNEL)),

                                    // #[cfg(feature = "c002_a885")]
                                    // "c002_a885" => Some((c002::c002_a885::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a885::KERNEL)),

                                    // #[cfg(feature = "c002_a886")]
                                    // "c002_a886" => Some((c002::c002_a886::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a886::KERNEL)),

                                    // #[cfg(feature = "c002_a887")]
                                    // "c002_a887" => Some((c002::c002_a887::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a887::KERNEL)),

                                    // #[cfg(feature = "c002_a888")]
                                    // "c002_a888" => Some((c002::c002_a888::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a888::KERNEL)),

                                    // #[cfg(feature = "c002_a889")]
                                    // "c002_a889" => Some((c002::c002_a889::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a889::KERNEL)),

                                    // #[cfg(feature = "c002_a890")]
                                    // "c002_a890" => Some((c002::c002_a890::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a890::KERNEL)),

                                    // #[cfg(feature = "c002_a891")]
                                    // "c002_a891" => Some((c002::c002_a891::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a891::KERNEL)),

                                    // #[cfg(feature = "c002_a892")]
                                    // "c002_a892" => Some((c002::c002_a892::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a892::KERNEL)),

                                    // #[cfg(feature = "c002_a893")]
                                    // "c002_a893" => Some((c002::c002_a893::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a893::KERNEL)),

                                    // #[cfg(feature = "c002_a894")]
                                    // "c002_a894" => Some((c002::c002_a894::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a894::KERNEL)),

                                    // #[cfg(feature = "c002_a895")]
                                    // "c002_a895" => Some((c002::c002_a895::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a895::KERNEL)),

                                    // #[cfg(feature = "c002_a896")]
                                    // "c002_a896" => Some((c002::c002_a896::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a896::KERNEL)),

                                    // #[cfg(feature = "c002_a897")]
                                    // "c002_a897" => Some((c002::c002_a897::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a897::KERNEL)),

                                    // #[cfg(feature = "c002_a898")]
                                    // "c002_a898" => Some((c002::c002_a898::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a898::KERNEL)),

                                    // #[cfg(feature = "c002_a899")]
                                    // "c002_a899" => Some((c002::c002_a899::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a899::KERNEL)),

                                    // #[cfg(feature = "c002_a900")]
                                    // "c002_a900" => Some((c002::c002_a900::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a900::KERNEL)),

                                    // #[cfg(feature = "c002_a901")]
                                    // "c002_a901" => Some((c002::c002_a901::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a901::KERNEL)),

                                    // #[cfg(feature = "c002_a902")]
                                    // "c002_a902" => Some((c002::c002_a902::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a902::KERNEL)),

                                    // #[cfg(feature = "c002_a903")]
                                    // "c002_a903" => Some((c002::c002_a903::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a903::KERNEL)),

                                    // #[cfg(feature = "c002_a904")]
                                    // "c002_a904" => Some((c002::c002_a904::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a904::KERNEL)),

                                    // #[cfg(feature = "c002_a905")]
                                    // "c002_a905" => Some((c002::c002_a905::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a905::KERNEL)),

                                    // #[cfg(feature = "c002_a906")]
                                    // "c002_a906" => Some((c002::c002_a906::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a906::KERNEL)),

                                    // #[cfg(feature = "c002_a907")]
                                    // "c002_a907" => Some((c002::c002_a907::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a907::KERNEL)),

                                    // #[cfg(feature = "c002_a908")]
                                    // "c002_a908" => Some((c002::c002_a908::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a908::KERNEL)),

                                    // #[cfg(feature = "c002_a909")]
                                    // "c002_a909" => Some((c002::c002_a909::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a909::KERNEL)),

                                    // #[cfg(feature = "c002_a910")]
                                    // "c002_a910" => Some((c002::c002_a910::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a910::KERNEL)),

                                    // #[cfg(feature = "c002_a911")]
                                    // "c002_a911" => Some((c002::c002_a911::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a911::KERNEL)),

                                    // #[cfg(feature = "c002_a912")]
                                    // "c002_a912" => Some((c002::c002_a912::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a912::KERNEL)),

                                    // #[cfg(feature = "c002_a913")]
                                    // "c002_a913" => Some((c002::c002_a913::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a913::KERNEL)),

                                    // #[cfg(feature = "c002_a914")]
                                    // "c002_a914" => Some((c002::c002_a914::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a914::KERNEL)),

                                    // #[cfg(feature = "c002_a915")]
                                    // "c002_a915" => Some((c002::c002_a915::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a915::KERNEL)),

                                    // #[cfg(feature = "c002_a916")]
                                    // "c002_a916" => Some((c002::c002_a916::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a916::KERNEL)),

                                    // #[cfg(feature = "c002_a917")]
                                    // "c002_a917" => Some((c002::c002_a917::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a917::KERNEL)),

                                    // #[cfg(feature = "c002_a918")]
                                    // "c002_a918" => Some((c002::c002_a918::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a918::KERNEL)),

                                    // #[cfg(feature = "c002_a919")]
                                    // "c002_a919" => Some((c002::c002_a919::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a919::KERNEL)),

                                    // #[cfg(feature = "c002_a920")]
                                    // "c002_a920" => Some((c002::c002_a920::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a920::KERNEL)),

                                    // #[cfg(feature = "c002_a921")]
                                    // "c002_a921" => Some((c002::c002_a921::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a921::KERNEL)),

                                    // #[cfg(feature = "c002_a922")]
                                    // "c002_a922" => Some((c002::c002_a922::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a922::KERNEL)),

                                    // #[cfg(feature = "c002_a923")]
                                    // "c002_a923" => Some((c002::c002_a923::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a923::KERNEL)),

                                    // #[cfg(feature = "c002_a924")]
                                    // "c002_a924" => Some((c002::c002_a924::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a924::KERNEL)),

                                    // #[cfg(feature = "c002_a925")]
                                    // "c002_a925" => Some((c002::c002_a925::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a925::KERNEL)),

                                    // #[cfg(feature = "c002_a926")]
                                    // "c002_a926" => Some((c002::c002_a926::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a926::KERNEL)),

                                    // #[cfg(feature = "c002_a927")]
                                    // "c002_a927" => Some((c002::c002_a927::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a927::KERNEL)),

                                    // #[cfg(feature = "c002_a928")]
                                    // "c002_a928" => Some((c002::c002_a928::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a928::KERNEL)),

                                    // #[cfg(feature = "c002_a929")]
                                    // "c002_a929" => Some((c002::c002_a929::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a929::KERNEL)),

                                    // #[cfg(feature = "c002_a930")]
                                    // "c002_a930" => Some((c002::c002_a930::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a930::KERNEL)),

                                    // #[cfg(feature = "c002_a931")]
                                    // "c002_a931" => Some((c002::c002_a931::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a931::KERNEL)),

                                    // #[cfg(feature = "c002_a932")]
                                    // "c002_a932" => Some((c002::c002_a932::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a932::KERNEL)),

                                    // #[cfg(feature = "c002_a933")]
                                    // "c002_a933" => Some((c002::c002_a933::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a933::KERNEL)),

                                    // #[cfg(feature = "c002_a934")]
                                    // "c002_a934" => Some((c002::c002_a934::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a934::KERNEL)),

                                    // #[cfg(feature = "c002_a935")]
                                    // "c002_a935" => Some((c002::c002_a935::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a935::KERNEL)),

                                    // #[cfg(feature = "c002_a936")]
                                    // "c002_a936" => Some((c002::c002_a936::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a936::KERNEL)),

                                    // #[cfg(feature = "c002_a937")]
                                    // "c002_a937" => Some((c002::c002_a937::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a937::KERNEL)),

                                    // #[cfg(feature = "c002_a938")]
                                    // "c002_a938" => Some((c002::c002_a938::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a938::KERNEL)),

                                    // #[cfg(feature = "c002_a939")]
                                    // "c002_a939" => Some((c002::c002_a939::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a939::KERNEL)),

                                    // #[cfg(feature = "c002_a940")]
                                    // "c002_a940" => Some((c002::c002_a940::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a940::KERNEL)),

                                    // #[cfg(feature = "c002_a941")]
                                    // "c002_a941" => Some((c002::c002_a941::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a941::KERNEL)),

                                    // #[cfg(feature = "c002_a942")]
                                    // "c002_a942" => Some((c002::c002_a942::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a942::KERNEL)),

                                    // #[cfg(feature = "c002_a943")]
                                    // "c002_a943" => Some((c002::c002_a943::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a943::KERNEL)),

                                    // #[cfg(feature = "c002_a944")]
                                    // "c002_a944" => Some((c002::c002_a944::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a944::KERNEL)),

                                    // #[cfg(feature = "c002_a945")]
                                    // "c002_a945" => Some((c002::c002_a945::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a945::KERNEL)),

                                    // #[cfg(feature = "c002_a946")]
                                    // "c002_a946" => Some((c002::c002_a946::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a946::KERNEL)),

                                    // #[cfg(feature = "c002_a947")]
                                    // "c002_a947" => Some((c002::c002_a947::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a947::KERNEL)),

                                    // #[cfg(feature = "c002_a948")]
                                    // "c002_a948" => Some((c002::c002_a948::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a948::KERNEL)),

                                    // #[cfg(feature = "c002_a949")]
                                    // "c002_a949" => Some((c002::c002_a949::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a949::KERNEL)),

                                    // #[cfg(feature = "c002_a950")]
                                    // "c002_a950" => Some((c002::c002_a950::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a950::KERNEL)),

                                    // #[cfg(feature = "c002_a951")]
                                    // "c002_a951" => Some((c002::c002_a951::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a951::KERNEL)),

                                    // #[cfg(feature = "c002_a952")]
                                    // "c002_a952" => Some((c002::c002_a952::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a952::KERNEL)),

                                    // #[cfg(feature = "c002_a953")]
                                    // "c002_a953" => Some((c002::c002_a953::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a953::KERNEL)),

                                    // #[cfg(feature = "c002_a954")]
                                    // "c002_a954" => Some((c002::c002_a954::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a954::KERNEL)),

                                    // #[cfg(feature = "c002_a955")]
                                    // "c002_a955" => Some((c002::c002_a955::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a955::KERNEL)),

                                    // #[cfg(feature = "c002_a956")]
                                    // "c002_a956" => Some((c002::c002_a956::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a956::KERNEL)),

                                    // #[cfg(feature = "c002_a957")]
                                    // "c002_a957" => Some((c002::c002_a957::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a957::KERNEL)),

                                    // #[cfg(feature = "c002_a958")]
                                    // "c002_a958" => Some((c002::c002_a958::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a958::KERNEL)),

                                    // #[cfg(feature = "c002_a959")]
                                    // "c002_a959" => Some((c002::c002_a959::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a959::KERNEL)),

                                    // #[cfg(feature = "c002_a960")]
                                    // "c002_a960" => Some((c002::c002_a960::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a960::KERNEL)),

                                    // #[cfg(feature = "c002_a961")]
                                    // "c002_a961" => Some((c002::c002_a961::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a961::KERNEL)),

                                    // #[cfg(feature = "c002_a962")]
                                    // "c002_a962" => Some((c002::c002_a962::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a962::KERNEL)),

                                    // #[cfg(feature = "c002_a963")]
                                    // "c002_a963" => Some((c002::c002_a963::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a963::KERNEL)),

                                    // #[cfg(feature = "c002_a964")]
                                    // "c002_a964" => Some((c002::c002_a964::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a964::KERNEL)),

                                    // #[cfg(feature = "c002_a965")]
                                    // "c002_a965" => Some((c002::c002_a965::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a965::KERNEL)),

                                    // #[cfg(feature = "c002_a966")]
                                    // "c002_a966" => Some((c002::c002_a966::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a966::KERNEL)),

                                    // #[cfg(feature = "c002_a967")]
                                    // "c002_a967" => Some((c002::c002_a967::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a967::KERNEL)),

                                    // #[cfg(feature = "c002_a968")]
                                    // "c002_a968" => Some((c002::c002_a968::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a968::KERNEL)),

                                    // #[cfg(feature = "c002_a969")]
                                    // "c002_a969" => Some((c002::c002_a969::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a969::KERNEL)),

                                    // #[cfg(feature = "c002_a970")]
                                    // "c002_a970" => Some((c002::c002_a970::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a970::KERNEL)),

                                    // #[cfg(feature = "c002_a971")]
                                    // "c002_a971" => Some((c002::c002_a971::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a971::KERNEL)),

                                    // #[cfg(feature = "c002_a972")]
                                    // "c002_a972" => Some((c002::c002_a972::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a972::KERNEL)),

                                    // #[cfg(feature = "c002_a973")]
                                    // "c002_a973" => Some((c002::c002_a973::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a973::KERNEL)),

                                    // #[cfg(feature = "c002_a974")]
                                    // "c002_a974" => Some((c002::c002_a974::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a974::KERNEL)),

                                    // #[cfg(feature = "c002_a975")]
                                    // "c002_a975" => Some((c002::c002_a975::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a975::KERNEL)),

                                    // #[cfg(feature = "c002_a976")]
                                    // "c002_a976" => Some((c002::c002_a976::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a976::KERNEL)),

                                    // #[cfg(feature = "c002_a977")]
                                    // "c002_a977" => Some((c002::c002_a977::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a977::KERNEL)),

                                    // #[cfg(feature = "c002_a978")]
                                    // "c002_a978" => Some((c002::c002_a978::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a978::KERNEL)),

                                    // #[cfg(feature = "c002_a979")]
                                    // "c002_a979" => Some((c002::c002_a979::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a979::KERNEL)),

                                    // #[cfg(feature = "c002_a980")]
                                    // "c002_a980" => Some((c002::c002_a980::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a980::KERNEL)),

                                    // #[cfg(feature = "c002_a981")]
                                    // "c002_a981" => Some((c002::c002_a981::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a981::KERNEL)),

                                    // #[cfg(feature = "c002_a982")]
                                    // "c002_a982" => Some((c002::c002_a982::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a982::KERNEL)),

                                    // #[cfg(feature = "c002_a983")]
                                    // "c002_a983" => Some((c002::c002_a983::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a983::KERNEL)),

                                    // #[cfg(feature = "c002_a984")]
                                    // "c002_a984" => Some((c002::c002_a984::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a984::KERNEL)),

                                    // #[cfg(feature = "c002_a985")]
                                    // "c002_a985" => Some((c002::c002_a985::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a985::KERNEL)),

                                    // #[cfg(feature = "c002_a986")]
                                    // "c002_a986" => Some((c002::c002_a986::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a986::KERNEL)),

                                    // #[cfg(feature = "c002_a987")]
                                    // "c002_a987" => Some((c002::c002_a987::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a987::KERNEL)),

                                    // #[cfg(feature = "c002_a988")]
                                    // "c002_a988" => Some((c002::c002_a988::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a988::KERNEL)),

                                    // #[cfg(feature = "c002_a989")]
                                    // "c002_a989" => Some((c002::c002_a989::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a989::KERNEL)),

                                    // #[cfg(feature = "c002_a990")]
                                    // "c002_a990" => Some((c002::c002_a990::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a990::KERNEL)),

                                    // #[cfg(feature = "c002_a991")]
                                    // "c002_a991" => Some((c002::c002_a991::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a991::KERNEL)),

                                    // #[cfg(feature = "c002_a992")]
                                    // "c002_a992" => Some((c002::c002_a992::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a992::KERNEL)),

                                    // #[cfg(feature = "c002_a993")]
                                    // "c002_a993" => Some((c002::c002_a993::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a993::KERNEL)),

                                    // #[cfg(feature = "c002_a994")]
                                    // "c002_a994" => Some((c002::c002_a994::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a994::KERNEL)),

                                    // #[cfg(feature = "c002_a995")]
                                    // "c002_a995" => Some((c002::c002_a995::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a995::KERNEL)),

                                    // #[cfg(feature = "c002_a996")]
                                    // "c002_a996" => Some((c002::c002_a996::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a996::KERNEL)),

                                    // #[cfg(feature = "c002_a997")]
                                    // "c002_a997" => Some((c002::c002_a997::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a997::KERNEL)),

                                    // #[cfg(feature = "c002_a998")]
                                    // "c002_a998" => Some((c002::c002_a998::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a998::KERNEL)),

                                    // #[cfg(feature = "c002_a999")]
                                    // "c002_a999" => Some((c002::c002_a999::cuda_solve_challenge as CudaSolveChallengeFn, &c002::c002_a999::KERNEL)),
                                    _ => {
                                        Option::<(CudaSolveChallengeFn, &Option<CudaKernel>)>::None
                                    }
                                };

                                match filter_step {
                                    Some((cuda_solve_challenge, algo_kernel)) => {
                                        let challenge_cuda_funcs = challenge_cuda_funcs
                                            .get_or_insert(
                                                get_or_compile_cuda(
                                                    &job.settings.challenge_id,
                                                    &tig_challenges::c002::KERNEL,
                                                    &dev,
                                                )
                                                .await,
                                            )
                                            .clone();
                                        let algorithm_cuda_funcs = algorithm_cuda_funcs
                                            .get_or_insert(
                                                get_or_compile_cuda(
                                                    &job.settings.algorithm_id,
                                                    algo_kernel,
                                                    &dev,
                                                )
                                                .await,
                                            )
                                            .clone();
                                        let challenge =
                                            tig_challenges::c002::Challenge::cuda_generate_instance_from_vec(
                                                seeds,
                                                &job.settings.difficulty,
                                                &dev,
                                                challenge_cuda_funcs,
                                            )
                                            .unwrap();
                                        match cuda_solve_challenge(
                                            &challenge,
                                            &dev,
                                            algorithm_cuda_funcs,
                                        ) {
                                            Ok(Some(solution)) => {
                                                challenge.verify_solution(&solution).is_err()
                                            }
                                            _ => true,
                                        }
                                    }
                                    None => false,
                                }
                            }
                            "c003" => {
                                type CudaSolveChallengeFn =
                                    fn(
                                        &tig_challenges::c003::Challenge,
                                        &Arc<CudaDevice>,
                                        HashMap<&'static str, CudaFunction>,
                                    )
                                        -> anyhow::Result<Option<tig_challenges::c003::Solution>>;

                                let filter_step = match job.settings.algorithm_id.as_str() {
                                    // #[cfg(feature = "c003_a001")]
                                    // "c003_a001" => Some((c003::c003_a001::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a001::KERNEL)),

                                    // #[cfg(feature = "c003_a002")]
                                    // "c003_a002" => Some((c003::c003_a002::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a002::KERNEL)),

                                    // #[cfg(feature = "c003_a003")]
                                    // "c003_a003" => Some((c003::c003_a003::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a003::KERNEL)),

                                    // #[cfg(feature = "c003_a004")]
                                    // "c003_a004" => Some((c003::c003_a004::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a004::KERNEL)),

                                    // #[cfg(feature = "c003_a005")]
                                    // "c003_a005" => Some((c003::c003_a005::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a005::KERNEL)),

                                    // #[cfg(feature = "c003_a006")]
                                    // "c003_a006" => Some((c003::c003_a006::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a006::KERNEL)),

                                    // #[cfg(feature = "c003_a007")]
                                    // "c003_a007" => Some((c003::c003_a007::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a007::KERNEL)),

                                    // #[cfg(feature = "c003_a008")]
                                    // "c003_a008" => Some((c003::c003_a008::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a008::KERNEL)),

                                    // #[cfg(feature = "c003_a009")]
                                    // "c003_a009" => Some((c003::c003_a009::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a009::KERNEL)),

                                    // #[cfg(feature = "c003_a010")]
                                    // "c003_a010" => Some((c003::c003_a010::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a010::KERNEL)),

                                    // #[cfg(feature = "c003_a011")]
                                    // "c003_a011" => Some((c003::c003_a011::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a011::KERNEL)),

                                    // #[cfg(feature = "c003_a012")]
                                    // "c003_a012" => Some((c003::c003_a012::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a012::KERNEL)),

                                    // #[cfg(feature = "c003_a013")]
                                    // "c003_a013" => Some((c003::c003_a013::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a013::KERNEL)),

                                    // #[cfg(feature = "c003_a014")]
                                    // "c003_a014" => Some((c003::c003_a014::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a014::KERNEL)),

                                    // #[cfg(feature = "c003_a015")]
                                    // "c003_a015" => Some((c003::c003_a015::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a015::KERNEL)),

                                    // #[cfg(feature = "c003_a016")]
                                    // "c003_a016" => Some((c003::c003_a016::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a016::KERNEL)),

                                    // #[cfg(feature = "c003_a017")]
                                    // "c003_a017" => Some((c003::c003_a017::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a017::KERNEL)),

                                    // #[cfg(feature = "c003_a018")]
                                    // "c003_a018" => Some((c003::c003_a018::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a018::KERNEL)),

                                    // #[cfg(feature = "c003_a019")]
                                    // "c003_a019" => Some((c003::c003_a019::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a019::KERNEL)),

                                    // #[cfg(feature = "c003_a020")]
                                    // "c003_a020" => Some((c003::c003_a020::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a020::KERNEL)),

                                    // #[cfg(feature = "c003_a021")]
                                    // "c003_a021" => Some((c003::c003_a021::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a021::KERNEL)),

                                    // #[cfg(feature = "c003_a022")]
                                    // "c003_a022" => Some((c003::c003_a022::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a022::KERNEL)),

                                    // #[cfg(feature = "c003_a023")]
                                    // "c003_a023" => Some((c003::c003_a023::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a023::KERNEL)),

                                    // #[cfg(feature = "c003_a024")]
                                    // "c003_a024" => Some((c003::c003_a024::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a024::KERNEL)),

                                    // #[cfg(feature = "c003_a025")]
                                    // "c003_a025" => Some((c003::c003_a025::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a025::KERNEL)),

                                    // #[cfg(feature = "c003_a026")]
                                    // "c003_a026" => Some((c003::c003_a026::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a026::KERNEL)),

                                    // #[cfg(feature = "c003_a027")]
                                    // "c003_a027" => Some((c003::c003_a027::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a027::KERNEL)),

                                    // #[cfg(feature = "c003_a028")]
                                    // "c003_a028" => Some((c003::c003_a028::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a028::KERNEL)),

                                    // #[cfg(feature = "c003_a029")]
                                    // "c003_a029" => Some((c003::c003_a029::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a029::KERNEL)),

                                    // #[cfg(feature = "c003_a030")]
                                    // "c003_a030" => Some((c003::c003_a030::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a030::KERNEL)),

                                    // #[cfg(feature = "c003_a031")]
                                    // "c003_a031" => Some((c003::c003_a031::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a031::KERNEL)),

                                    // #[cfg(feature = "c003_a032")]
                                    // "c003_a032" => Some((c003::c003_a032::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a032::KERNEL)),

                                    // #[cfg(feature = "c003_a033")]
                                    // "c003_a033" => Some((c003::c003_a033::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a033::KERNEL)),

                                    // #[cfg(feature = "c003_a034")]
                                    // "c003_a034" => Some((c003::c003_a034::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a034::KERNEL)),

                                    // #[cfg(feature = "c003_a035")]
                                    // "c003_a035" => Some((c003::c003_a035::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a035::KERNEL)),

                                    // #[cfg(feature = "c003_a036")]
                                    // "c003_a036" => Some((c003::c003_a036::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a036::KERNEL)),

                                    // #[cfg(feature = "c003_a037")]
                                    // "c003_a037" => Some((c003::c003_a037::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a037::KERNEL)),

                                    // #[cfg(feature = "c003_a038")]
                                    // "c003_a038" => Some((c003::c003_a038::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a038::KERNEL)),

                                    // #[cfg(feature = "c003_a039")]
                                    // "c003_a039" => Some((c003::c003_a039::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a039::KERNEL)),

                                    // #[cfg(feature = "c003_a040")]
                                    // "c003_a040" => Some((c003::c003_a040::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a040::KERNEL)),

                                    // #[cfg(feature = "c003_a041")]
                                    // "c003_a041" => Some((c003::c003_a041::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a041::KERNEL)),

                                    // #[cfg(feature = "c003_a042")]
                                    // "c003_a042" => Some((c003::c003_a042::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a042::KERNEL)),

                                    // #[cfg(feature = "c003_a043")]
                                    // "c003_a043" => Some((c003::c003_a043::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a043::KERNEL)),

                                    // #[cfg(feature = "c003_a044")]
                                    // "c003_a044" => Some((c003::c003_a044::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a044::KERNEL)),

                                    // #[cfg(feature = "c003_a045")]
                                    // "c003_a045" => Some((c003::c003_a045::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a045::KERNEL)),

                                    // #[cfg(feature = "c003_a046")]
                                    // "c003_a046" => Some((c003::c003_a046::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a046::KERNEL)),

                                    // #[cfg(feature = "c003_a047")]
                                    // "c003_a047" => Some((c003::c003_a047::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a047::KERNEL)),

                                    // #[cfg(feature = "c003_a048")]
                                    // "c003_a048" => Some((c003::c003_a048::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a048::KERNEL)),

                                    // #[cfg(feature = "c003_a049")]
                                    // "c003_a049" => Some((c003::c003_a049::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a049::KERNEL)),

                                    // #[cfg(feature = "c003_a050")]
                                    // "c003_a050" => Some((c003::c003_a050::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a050::KERNEL)),

                                    // #[cfg(feature = "c003_a051")]
                                    // "c003_a051" => Some((c003::c003_a051::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a051::KERNEL)),

                                    // #[cfg(feature = "c003_a052")]
                                    // "c003_a052" => Some((c003::c003_a052::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a052::KERNEL)),

                                    // #[cfg(feature = "c003_a053")]
                                    // "c003_a053" => Some((c003::c003_a053::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a053::KERNEL)),

                                    // #[cfg(feature = "c003_a054")]
                                    // "c003_a054" => Some((c003::c003_a054::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a054::KERNEL)),

                                    // #[cfg(feature = "c003_a055")]
                                    // "c003_a055" => Some((c003::c003_a055::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a055::KERNEL)),

                                    // #[cfg(feature = "c003_a056")]
                                    // "c003_a056" => Some((c003::c003_a056::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a056::KERNEL)),

                                    // #[cfg(feature = "c003_a057")]
                                    // "c003_a057" => Some((c003::c003_a057::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a057::KERNEL)),

                                    // #[cfg(feature = "c003_a058")]
                                    // "c003_a058" => Some((c003::c003_a058::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a058::KERNEL)),

                                    // #[cfg(feature = "c003_a059")]
                                    // "c003_a059" => Some((c003::c003_a059::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a059::KERNEL)),

                                    // #[cfg(feature = "c003_a060")]
                                    // "c003_a060" => Some((c003::c003_a060::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a060::KERNEL)),

                                    // #[cfg(feature = "c003_a061")]
                                    // "c003_a061" => Some((c003::c003_a061::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a061::KERNEL)),

                                    // #[cfg(feature = "c003_a062")]
                                    // "c003_a062" => Some((c003::c003_a062::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a062::KERNEL)),

                                    // #[cfg(feature = "c003_a063")]
                                    // "c003_a063" => Some((c003::c003_a063::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a063::KERNEL)),

                                    // #[cfg(feature = "c003_a064")]
                                    // "c003_a064" => Some((c003::c003_a064::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a064::KERNEL)),

                                    // #[cfg(feature = "c003_a065")]
                                    // "c003_a065" => Some((c003::c003_a065::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a065::KERNEL)),

                                    // #[cfg(feature = "c003_a066")]
                                    // "c003_a066" => Some((c003::c003_a066::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a066::KERNEL)),

                                    // #[cfg(feature = "c003_a067")]
                                    // "c003_a067" => Some((c003::c003_a067::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a067::KERNEL)),

                                    // #[cfg(feature = "c003_a068")]
                                    // "c003_a068" => Some((c003::c003_a068::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a068::KERNEL)),

                                    // #[cfg(feature = "c003_a069")]
                                    // "c003_a069" => Some((c003::c003_a069::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a069::KERNEL)),

                                    // #[cfg(feature = "c003_a070")]
                                    // "c003_a070" => Some((c003::c003_a070::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a070::KERNEL)),

                                    // #[cfg(feature = "c003_a071")]
                                    // "c003_a071" => Some((c003::c003_a071::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a071::KERNEL)),

                                    // #[cfg(feature = "c003_a072")]
                                    // "c003_a072" => Some((c003::c003_a072::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a072::KERNEL)),

                                    // #[cfg(feature = "c003_a073")]
                                    // "c003_a073" => Some((c003::c003_a073::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a073::KERNEL)),

                                    // #[cfg(feature = "c003_a074")]
                                    // "c003_a074" => Some((c003::c003_a074::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a074::KERNEL)),

                                    // #[cfg(feature = "c003_a075")]
                                    // "c003_a075" => Some((c003::c003_a075::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a075::KERNEL)),

                                    // #[cfg(feature = "c003_a076")]
                                    // "c003_a076" => Some((c003::c003_a076::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a076::KERNEL)),

                                    // #[cfg(feature = "c003_a077")]
                                    // "c003_a077" => Some((c003::c003_a077::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a077::KERNEL)),

                                    // #[cfg(feature = "c003_a078")]
                                    // "c003_a078" => Some((c003::c003_a078::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a078::KERNEL)),

                                    // #[cfg(feature = "c003_a079")]
                                    // "c003_a079" => Some((c003::c003_a079::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a079::KERNEL)),

                                    // #[cfg(feature = "c003_a080")]
                                    // "c003_a080" => Some((c003::c003_a080::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a080::KERNEL)),

                                    // #[cfg(feature = "c003_a081")]
                                    // "c003_a081" => Some((c003::c003_a081::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a081::KERNEL)),

                                    // #[cfg(feature = "c003_a082")]
                                    // "c003_a082" => Some((c003::c003_a082::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a082::KERNEL)),

                                    // #[cfg(feature = "c003_a083")]
                                    // "c003_a083" => Some((c003::c003_a083::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a083::KERNEL)),

                                    // #[cfg(feature = "c003_a084")]
                                    // "c003_a084" => Some((c003::c003_a084::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a084::KERNEL)),

                                    // #[cfg(feature = "c003_a085")]
                                    // "c003_a085" => Some((c003::c003_a085::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a085::KERNEL)),

                                    // #[cfg(feature = "c003_a086")]
                                    // "c003_a086" => Some((c003::c003_a086::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a086::KERNEL)),

                                    // #[cfg(feature = "c003_a087")]
                                    // "c003_a087" => Some((c003::c003_a087::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a087::KERNEL)),

                                    // #[cfg(feature = "c003_a088")]
                                    // "c003_a088" => Some((c003::c003_a088::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a088::KERNEL)),

                                    // #[cfg(feature = "c003_a089")]
                                    // "c003_a089" => Some((c003::c003_a089::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a089::KERNEL)),

                                    // #[cfg(feature = "c003_a090")]
                                    // "c003_a090" => Some((c003::c003_a090::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a090::KERNEL)),

                                    // #[cfg(feature = "c003_a091")]
                                    // "c003_a091" => Some((c003::c003_a091::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a091::KERNEL)),

                                    // #[cfg(feature = "c003_a092")]
                                    // "c003_a092" => Some((c003::c003_a092::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a092::KERNEL)),

                                    // #[cfg(feature = "c003_a093")]
                                    // "c003_a093" => Some((c003::c003_a093::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a093::KERNEL)),

                                    // #[cfg(feature = "c003_a094")]
                                    // "c003_a094" => Some((c003::c003_a094::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a094::KERNEL)),

                                    // #[cfg(feature = "c003_a095")]
                                    // "c003_a095" => Some((c003::c003_a095::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a095::KERNEL)),

                                    // #[cfg(feature = "c003_a096")]
                                    // "c003_a096" => Some((c003::c003_a096::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a096::KERNEL)),

                                    // #[cfg(feature = "c003_a097")]
                                    // "c003_a097" => Some((c003::c003_a097::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a097::KERNEL)),

                                    // #[cfg(feature = "c003_a098")]
                                    // "c003_a098" => Some((c003::c003_a098::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a098::KERNEL)),

                                    // #[cfg(feature = "c003_a099")]
                                    // "c003_a099" => Some((c003::c003_a099::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a099::KERNEL)),

                                    // #[cfg(feature = "c003_a100")]
                                    // "c003_a100" => Some((c003::c003_a100::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a100::KERNEL)),

                                    // #[cfg(feature = "c003_a101")]
                                    // "c003_a101" => Some((c003::c003_a101::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a101::KERNEL)),

                                    // #[cfg(feature = "c003_a102")]
                                    // "c003_a102" => Some((c003::c003_a102::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a102::KERNEL)),

                                    // #[cfg(feature = "c003_a103")]
                                    // "c003_a103" => Some((c003::c003_a103::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a103::KERNEL)),

                                    // #[cfg(feature = "c003_a104")]
                                    // "c003_a104" => Some((c003::c003_a104::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a104::KERNEL)),

                                    // #[cfg(feature = "c003_a105")]
                                    // "c003_a105" => Some((c003::c003_a105::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a105::KERNEL)),

                                    // #[cfg(feature = "c003_a106")]
                                    // "c003_a106" => Some((c003::c003_a106::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a106::KERNEL)),

                                    // #[cfg(feature = "c003_a107")]
                                    // "c003_a107" => Some((c003::c003_a107::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a107::KERNEL)),

                                    // #[cfg(feature = "c003_a108")]
                                    // "c003_a108" => Some((c003::c003_a108::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a108::KERNEL)),

                                    // #[cfg(feature = "c003_a109")]
                                    // "c003_a109" => Some((c003::c003_a109::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a109::KERNEL)),

                                    // #[cfg(feature = "c003_a110")]
                                    // "c003_a110" => Some((c003::c003_a110::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a110::KERNEL)),

                                    // #[cfg(feature = "c003_a111")]
                                    // "c003_a111" => Some((c003::c003_a111::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a111::KERNEL)),

                                    // #[cfg(feature = "c003_a112")]
                                    // "c003_a112" => Some((c003::c003_a112::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a112::KERNEL)),

                                    // #[cfg(feature = "c003_a113")]
                                    // "c003_a113" => Some((c003::c003_a113::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a113::KERNEL)),

                                    // #[cfg(feature = "c003_a114")]
                                    // "c003_a114" => Some((c003::c003_a114::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a114::KERNEL)),

                                    // #[cfg(feature = "c003_a115")]
                                    // "c003_a115" => Some((c003::c003_a115::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a115::KERNEL)),

                                    // #[cfg(feature = "c003_a116")]
                                    // "c003_a116" => Some((c003::c003_a116::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a116::KERNEL)),

                                    // #[cfg(feature = "c003_a117")]
                                    // "c003_a117" => Some((c003::c003_a117::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a117::KERNEL)),

                                    // #[cfg(feature = "c003_a118")]
                                    // "c003_a118" => Some((c003::c003_a118::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a118::KERNEL)),

                                    // #[cfg(feature = "c003_a119")]
                                    // "c003_a119" => Some((c003::c003_a119::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a119::KERNEL)),

                                    // #[cfg(feature = "c003_a120")]
                                    // "c003_a120" => Some((c003::c003_a120::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a120::KERNEL)),

                                    // #[cfg(feature = "c003_a121")]
                                    // "c003_a121" => Some((c003::c003_a121::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a121::KERNEL)),

                                    // #[cfg(feature = "c003_a122")]
                                    // "c003_a122" => Some((c003::c003_a122::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a122::KERNEL)),

                                    // #[cfg(feature = "c003_a123")]
                                    // "c003_a123" => Some((c003::c003_a123::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a123::KERNEL)),

                                    // #[cfg(feature = "c003_a124")]
                                    // "c003_a124" => Some((c003::c003_a124::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a124::KERNEL)),

                                    // #[cfg(feature = "c003_a125")]
                                    // "c003_a125" => Some((c003::c003_a125::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a125::KERNEL)),

                                    // #[cfg(feature = "c003_a126")]
                                    // "c003_a126" => Some((c003::c003_a126::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a126::KERNEL)),

                                    // #[cfg(feature = "c003_a127")]
                                    // "c003_a127" => Some((c003::c003_a127::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a127::KERNEL)),

                                    // #[cfg(feature = "c003_a128")]
                                    // "c003_a128" => Some((c003::c003_a128::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a128::KERNEL)),

                                    // #[cfg(feature = "c003_a129")]
                                    // "c003_a129" => Some((c003::c003_a129::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a129::KERNEL)),

                                    // #[cfg(feature = "c003_a130")]
                                    // "c003_a130" => Some((c003::c003_a130::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a130::KERNEL)),

                                    // #[cfg(feature = "c003_a131")]
                                    // "c003_a131" => Some((c003::c003_a131::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a131::KERNEL)),

                                    // #[cfg(feature = "c003_a132")]
                                    // "c003_a132" => Some((c003::c003_a132::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a132::KERNEL)),

                                    // #[cfg(feature = "c003_a133")]
                                    // "c003_a133" => Some((c003::c003_a133::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a133::KERNEL)),

                                    // #[cfg(feature = "c003_a134")]
                                    // "c003_a134" => Some((c003::c003_a134::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a134::KERNEL)),

                                    // #[cfg(feature = "c003_a135")]
                                    // "c003_a135" => Some((c003::c003_a135::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a135::KERNEL)),

                                    // #[cfg(feature = "c003_a136")]
                                    // "c003_a136" => Some((c003::c003_a136::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a136::KERNEL)),

                                    // #[cfg(feature = "c003_a137")]
                                    // "c003_a137" => Some((c003::c003_a137::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a137::KERNEL)),

                                    // #[cfg(feature = "c003_a138")]
                                    // "c003_a138" => Some((c003::c003_a138::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a138::KERNEL)),

                                    // #[cfg(feature = "c003_a139")]
                                    // "c003_a139" => Some((c003::c003_a139::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a139::KERNEL)),

                                    // #[cfg(feature = "c003_a140")]
                                    // "c003_a140" => Some((c003::c003_a140::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a140::KERNEL)),

                                    // #[cfg(feature = "c003_a141")]
                                    // "c003_a141" => Some((c003::c003_a141::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a141::KERNEL)),

                                    // #[cfg(feature = "c003_a142")]
                                    // "c003_a142" => Some((c003::c003_a142::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a142::KERNEL)),

                                    // #[cfg(feature = "c003_a143")]
                                    // "c003_a143" => Some((c003::c003_a143::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a143::KERNEL)),

                                    // #[cfg(feature = "c003_a144")]
                                    // "c003_a144" => Some((c003::c003_a144::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a144::KERNEL)),

                                    // #[cfg(feature = "c003_a145")]
                                    // "c003_a145" => Some((c003::c003_a145::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a145::KERNEL)),

                                    // #[cfg(feature = "c003_a146")]
                                    // "c003_a146" => Some((c003::c003_a146::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a146::KERNEL)),

                                    // #[cfg(feature = "c003_a147")]
                                    // "c003_a147" => Some((c003::c003_a147::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a147::KERNEL)),

                                    // #[cfg(feature = "c003_a148")]
                                    // "c003_a148" => Some((c003::c003_a148::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a148::KERNEL)),

                                    // #[cfg(feature = "c003_a149")]
                                    // "c003_a149" => Some((c003::c003_a149::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a149::KERNEL)),

                                    // #[cfg(feature = "c003_a150")]
                                    // "c003_a150" => Some((c003::c003_a150::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a150::KERNEL)),

                                    // #[cfg(feature = "c003_a151")]
                                    // "c003_a151" => Some((c003::c003_a151::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a151::KERNEL)),

                                    // #[cfg(feature = "c003_a152")]
                                    // "c003_a152" => Some((c003::c003_a152::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a152::KERNEL)),

                                    // #[cfg(feature = "c003_a153")]
                                    // "c003_a153" => Some((c003::c003_a153::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a153::KERNEL)),

                                    // #[cfg(feature = "c003_a154")]
                                    // "c003_a154" => Some((c003::c003_a154::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a154::KERNEL)),

                                    // #[cfg(feature = "c003_a155")]
                                    // "c003_a155" => Some((c003::c003_a155::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a155::KERNEL)),

                                    // #[cfg(feature = "c003_a156")]
                                    // "c003_a156" => Some((c003::c003_a156::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a156::KERNEL)),

                                    // #[cfg(feature = "c003_a157")]
                                    // "c003_a157" => Some((c003::c003_a157::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a157::KERNEL)),

                                    // #[cfg(feature = "c003_a158")]
                                    // "c003_a158" => Some((c003::c003_a158::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a158::KERNEL)),

                                    // #[cfg(feature = "c003_a159")]
                                    // "c003_a159" => Some((c003::c003_a159::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a159::KERNEL)),

                                    // #[cfg(feature = "c003_a160")]
                                    // "c003_a160" => Some((c003::c003_a160::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a160::KERNEL)),

                                    // #[cfg(feature = "c003_a161")]
                                    // "c003_a161" => Some((c003::c003_a161::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a161::KERNEL)),

                                    // #[cfg(feature = "c003_a162")]
                                    // "c003_a162" => Some((c003::c003_a162::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a162::KERNEL)),

                                    // #[cfg(feature = "c003_a163")]
                                    // "c003_a163" => Some((c003::c003_a163::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a163::KERNEL)),

                                    // #[cfg(feature = "c003_a164")]
                                    // "c003_a164" => Some((c003::c003_a164::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a164::KERNEL)),

                                    // #[cfg(feature = "c003_a165")]
                                    // "c003_a165" => Some((c003::c003_a165::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a165::KERNEL)),

                                    // #[cfg(feature = "c003_a166")]
                                    // "c003_a166" => Some((c003::c003_a166::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a166::KERNEL)),

                                    // #[cfg(feature = "c003_a167")]
                                    // "c003_a167" => Some((c003::c003_a167::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a167::KERNEL)),

                                    // #[cfg(feature = "c003_a168")]
                                    // "c003_a168" => Some((c003::c003_a168::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a168::KERNEL)),

                                    // #[cfg(feature = "c003_a169")]
                                    // "c003_a169" => Some((c003::c003_a169::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a169::KERNEL)),

                                    // #[cfg(feature = "c003_a170")]
                                    // "c003_a170" => Some((c003::c003_a170::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a170::KERNEL)),

                                    // #[cfg(feature = "c003_a171")]
                                    // "c003_a171" => Some((c003::c003_a171::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a171::KERNEL)),

                                    // #[cfg(feature = "c003_a172")]
                                    // "c003_a172" => Some((c003::c003_a172::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a172::KERNEL)),

                                    // #[cfg(feature = "c003_a173")]
                                    // "c003_a173" => Some((c003::c003_a173::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a173::KERNEL)),

                                    // #[cfg(feature = "c003_a174")]
                                    // "c003_a174" => Some((c003::c003_a174::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a174::KERNEL)),

                                    // #[cfg(feature = "c003_a175")]
                                    // "c003_a175" => Some((c003::c003_a175::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a175::KERNEL)),

                                    // #[cfg(feature = "c003_a176")]
                                    // "c003_a176" => Some((c003::c003_a176::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a176::KERNEL)),

                                    // #[cfg(feature = "c003_a177")]
                                    // "c003_a177" => Some((c003::c003_a177::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a177::KERNEL)),

                                    // #[cfg(feature = "c003_a178")]
                                    // "c003_a178" => Some((c003::c003_a178::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a178::KERNEL)),

                                    // #[cfg(feature = "c003_a179")]
                                    // "c003_a179" => Some((c003::c003_a179::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a179::KERNEL)),

                                    // #[cfg(feature = "c003_a180")]
                                    // "c003_a180" => Some((c003::c003_a180::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a180::KERNEL)),

                                    // #[cfg(feature = "c003_a181")]
                                    // "c003_a181" => Some((c003::c003_a181::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a181::KERNEL)),

                                    // #[cfg(feature = "c003_a182")]
                                    // "c003_a182" => Some((c003::c003_a182::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a182::KERNEL)),

                                    // #[cfg(feature = "c003_a183")]
                                    // "c003_a183" => Some((c003::c003_a183::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a183::KERNEL)),

                                    // #[cfg(feature = "c003_a184")]
                                    // "c003_a184" => Some((c003::c003_a184::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a184::KERNEL)),

                                    // #[cfg(feature = "c003_a185")]
                                    // "c003_a185" => Some((c003::c003_a185::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a185::KERNEL)),

                                    // #[cfg(feature = "c003_a186")]
                                    // "c003_a186" => Some((c003::c003_a186::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a186::KERNEL)),

                                    // #[cfg(feature = "c003_a187")]
                                    // "c003_a187" => Some((c003::c003_a187::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a187::KERNEL)),

                                    // #[cfg(feature = "c003_a188")]
                                    // "c003_a188" => Some((c003::c003_a188::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a188::KERNEL)),

                                    // #[cfg(feature = "c003_a189")]
                                    // "c003_a189" => Some((c003::c003_a189::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a189::KERNEL)),

                                    // #[cfg(feature = "c003_a190")]
                                    // "c003_a190" => Some((c003::c003_a190::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a190::KERNEL)),

                                    // #[cfg(feature = "c003_a191")]
                                    // "c003_a191" => Some((c003::c003_a191::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a191::KERNEL)),

                                    // #[cfg(feature = "c003_a192")]
                                    // "c003_a192" => Some((c003::c003_a192::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a192::KERNEL)),

                                    // #[cfg(feature = "c003_a193")]
                                    // "c003_a193" => Some((c003::c003_a193::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a193::KERNEL)),

                                    // #[cfg(feature = "c003_a194")]
                                    // "c003_a194" => Some((c003::c003_a194::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a194::KERNEL)),

                                    // #[cfg(feature = "c003_a195")]
                                    // "c003_a195" => Some((c003::c003_a195::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a195::KERNEL)),

                                    // #[cfg(feature = "c003_a196")]
                                    // "c003_a196" => Some((c003::c003_a196::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a196::KERNEL)),

                                    // #[cfg(feature = "c003_a197")]
                                    // "c003_a197" => Some((c003::c003_a197::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a197::KERNEL)),

                                    // #[cfg(feature = "c003_a198")]
                                    // "c003_a198" => Some((c003::c003_a198::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a198::KERNEL)),

                                    // #[cfg(feature = "c003_a199")]
                                    // "c003_a199" => Some((c003::c003_a199::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a199::KERNEL)),

                                    // #[cfg(feature = "c003_a200")]
                                    // "c003_a200" => Some((c003::c003_a200::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a200::KERNEL)),

                                    // #[cfg(feature = "c003_a201")]
                                    // "c003_a201" => Some((c003::c003_a201::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a201::KERNEL)),

                                    // #[cfg(feature = "c003_a202")]
                                    // "c003_a202" => Some((c003::c003_a202::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a202::KERNEL)),

                                    // #[cfg(feature = "c003_a203")]
                                    // "c003_a203" => Some((c003::c003_a203::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a203::KERNEL)),

                                    // #[cfg(feature = "c003_a204")]
                                    // "c003_a204" => Some((c003::c003_a204::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a204::KERNEL)),

                                    // #[cfg(feature = "c003_a205")]
                                    // "c003_a205" => Some((c003::c003_a205::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a205::KERNEL)),

                                    // #[cfg(feature = "c003_a206")]
                                    // "c003_a206" => Some((c003::c003_a206::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a206::KERNEL)),

                                    // #[cfg(feature = "c003_a207")]
                                    // "c003_a207" => Some((c003::c003_a207::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a207::KERNEL)),

                                    // #[cfg(feature = "c003_a208")]
                                    // "c003_a208" => Some((c003::c003_a208::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a208::KERNEL)),

                                    // #[cfg(feature = "c003_a209")]
                                    // "c003_a209" => Some((c003::c003_a209::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a209::KERNEL)),

                                    // #[cfg(feature = "c003_a210")]
                                    // "c003_a210" => Some((c003::c003_a210::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a210::KERNEL)),

                                    // #[cfg(feature = "c003_a211")]
                                    // "c003_a211" => Some((c003::c003_a211::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a211::KERNEL)),

                                    // #[cfg(feature = "c003_a212")]
                                    // "c003_a212" => Some((c003::c003_a212::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a212::KERNEL)),

                                    // #[cfg(feature = "c003_a213")]
                                    // "c003_a213" => Some((c003::c003_a213::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a213::KERNEL)),

                                    // #[cfg(feature = "c003_a214")]
                                    // "c003_a214" => Some((c003::c003_a214::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a214::KERNEL)),

                                    // #[cfg(feature = "c003_a215")]
                                    // "c003_a215" => Some((c003::c003_a215::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a215::KERNEL)),

                                    // #[cfg(feature = "c003_a216")]
                                    // "c003_a216" => Some((c003::c003_a216::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a216::KERNEL)),

                                    // #[cfg(feature = "c003_a217")]
                                    // "c003_a217" => Some((c003::c003_a217::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a217::KERNEL)),

                                    // #[cfg(feature = "c003_a218")]
                                    // "c003_a218" => Some((c003::c003_a218::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a218::KERNEL)),

                                    // #[cfg(feature = "c003_a219")]
                                    // "c003_a219" => Some((c003::c003_a219::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a219::KERNEL)),

                                    // #[cfg(feature = "c003_a220")]
                                    // "c003_a220" => Some((c003::c003_a220::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a220::KERNEL)),

                                    // #[cfg(feature = "c003_a221")]
                                    // "c003_a221" => Some((c003::c003_a221::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a221::KERNEL)),

                                    // #[cfg(feature = "c003_a222")]
                                    // "c003_a222" => Some((c003::c003_a222::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a222::KERNEL)),

                                    // #[cfg(feature = "c003_a223")]
                                    // "c003_a223" => Some((c003::c003_a223::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a223::KERNEL)),

                                    // #[cfg(feature = "c003_a224")]
                                    // "c003_a224" => Some((c003::c003_a224::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a224::KERNEL)),

                                    // #[cfg(feature = "c003_a225")]
                                    // "c003_a225" => Some((c003::c003_a225::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a225::KERNEL)),

                                    // #[cfg(feature = "c003_a226")]
                                    // "c003_a226" => Some((c003::c003_a226::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a226::KERNEL)),

                                    // #[cfg(feature = "c003_a227")]
                                    // "c003_a227" => Some((c003::c003_a227::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a227::KERNEL)),

                                    // #[cfg(feature = "c003_a228")]
                                    // "c003_a228" => Some((c003::c003_a228::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a228::KERNEL)),

                                    // #[cfg(feature = "c003_a229")]
                                    // "c003_a229" => Some((c003::c003_a229::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a229::KERNEL)),

                                    // #[cfg(feature = "c003_a230")]
                                    // "c003_a230" => Some((c003::c003_a230::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a230::KERNEL)),

                                    // #[cfg(feature = "c003_a231")]
                                    // "c003_a231" => Some((c003::c003_a231::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a231::KERNEL)),

                                    // #[cfg(feature = "c003_a232")]
                                    // "c003_a232" => Some((c003::c003_a232::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a232::KERNEL)),

                                    // #[cfg(feature = "c003_a233")]
                                    // "c003_a233" => Some((c003::c003_a233::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a233::KERNEL)),

                                    // #[cfg(feature = "c003_a234")]
                                    // "c003_a234" => Some((c003::c003_a234::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a234::KERNEL)),

                                    // #[cfg(feature = "c003_a235")]
                                    // "c003_a235" => Some((c003::c003_a235::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a235::KERNEL)),

                                    // #[cfg(feature = "c003_a236")]
                                    // "c003_a236" => Some((c003::c003_a236::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a236::KERNEL)),

                                    // #[cfg(feature = "c003_a237")]
                                    // "c003_a237" => Some((c003::c003_a237::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a237::KERNEL)),

                                    // #[cfg(feature = "c003_a238")]
                                    // "c003_a238" => Some((c003::c003_a238::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a238::KERNEL)),

                                    // #[cfg(feature = "c003_a239")]
                                    // "c003_a239" => Some((c003::c003_a239::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a239::KERNEL)),

                                    // #[cfg(feature = "c003_a240")]
                                    // "c003_a240" => Some((c003::c003_a240::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a240::KERNEL)),

                                    // #[cfg(feature = "c003_a241")]
                                    // "c003_a241" => Some((c003::c003_a241::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a241::KERNEL)),

                                    // #[cfg(feature = "c003_a242")]
                                    // "c003_a242" => Some((c003::c003_a242::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a242::KERNEL)),

                                    // #[cfg(feature = "c003_a243")]
                                    // "c003_a243" => Some((c003::c003_a243::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a243::KERNEL)),

                                    // #[cfg(feature = "c003_a244")]
                                    // "c003_a244" => Some((c003::c003_a244::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a244::KERNEL)),

                                    // #[cfg(feature = "c003_a245")]
                                    // "c003_a245" => Some((c003::c003_a245::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a245::KERNEL)),

                                    // #[cfg(feature = "c003_a246")]
                                    // "c003_a246" => Some((c003::c003_a246::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a246::KERNEL)),

                                    // #[cfg(feature = "c003_a247")]
                                    // "c003_a247" => Some((c003::c003_a247::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a247::KERNEL)),

                                    // #[cfg(feature = "c003_a248")]
                                    // "c003_a248" => Some((c003::c003_a248::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a248::KERNEL)),

                                    // #[cfg(feature = "c003_a249")]
                                    // "c003_a249" => Some((c003::c003_a249::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a249::KERNEL)),

                                    // #[cfg(feature = "c003_a250")]
                                    // "c003_a250" => Some((c003::c003_a250::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a250::KERNEL)),

                                    // #[cfg(feature = "c003_a251")]
                                    // "c003_a251" => Some((c003::c003_a251::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a251::KERNEL)),

                                    // #[cfg(feature = "c003_a252")]
                                    // "c003_a252" => Some((c003::c003_a252::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a252::KERNEL)),

                                    // #[cfg(feature = "c003_a253")]
                                    // "c003_a253" => Some((c003::c003_a253::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a253::KERNEL)),

                                    // #[cfg(feature = "c003_a254")]
                                    // "c003_a254" => Some((c003::c003_a254::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a254::KERNEL)),

                                    // #[cfg(feature = "c003_a255")]
                                    // "c003_a255" => Some((c003::c003_a255::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a255::KERNEL)),

                                    // #[cfg(feature = "c003_a256")]
                                    // "c003_a256" => Some((c003::c003_a256::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a256::KERNEL)),

                                    // #[cfg(feature = "c003_a257")]
                                    // "c003_a257" => Some((c003::c003_a257::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a257::KERNEL)),

                                    // #[cfg(feature = "c003_a258")]
                                    // "c003_a258" => Some((c003::c003_a258::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a258::KERNEL)),

                                    // #[cfg(feature = "c003_a259")]
                                    // "c003_a259" => Some((c003::c003_a259::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a259::KERNEL)),

                                    // #[cfg(feature = "c003_a260")]
                                    // "c003_a260" => Some((c003::c003_a260::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a260::KERNEL)),

                                    // #[cfg(feature = "c003_a261")]
                                    // "c003_a261" => Some((c003::c003_a261::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a261::KERNEL)),

                                    // #[cfg(feature = "c003_a262")]
                                    // "c003_a262" => Some((c003::c003_a262::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a262::KERNEL)),

                                    // #[cfg(feature = "c003_a263")]
                                    // "c003_a263" => Some((c003::c003_a263::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a263::KERNEL)),

                                    // #[cfg(feature = "c003_a264")]
                                    // "c003_a264" => Some((c003::c003_a264::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a264::KERNEL)),

                                    // #[cfg(feature = "c003_a265")]
                                    // "c003_a265" => Some((c003::c003_a265::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a265::KERNEL)),

                                    // #[cfg(feature = "c003_a266")]
                                    // "c003_a266" => Some((c003::c003_a266::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a266::KERNEL)),

                                    // #[cfg(feature = "c003_a267")]
                                    // "c003_a267" => Some((c003::c003_a267::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a267::KERNEL)),

                                    // #[cfg(feature = "c003_a268")]
                                    // "c003_a268" => Some((c003::c003_a268::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a268::KERNEL)),

                                    // #[cfg(feature = "c003_a269")]
                                    // "c003_a269" => Some((c003::c003_a269::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a269::KERNEL)),

                                    // #[cfg(feature = "c003_a270")]
                                    // "c003_a270" => Some((c003::c003_a270::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a270::KERNEL)),

                                    // #[cfg(feature = "c003_a271")]
                                    // "c003_a271" => Some((c003::c003_a271::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a271::KERNEL)),

                                    // #[cfg(feature = "c003_a272")]
                                    // "c003_a272" => Some((c003::c003_a272::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a272::KERNEL)),

                                    // #[cfg(feature = "c003_a273")]
                                    // "c003_a273" => Some((c003::c003_a273::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a273::KERNEL)),

                                    // #[cfg(feature = "c003_a274")]
                                    // "c003_a274" => Some((c003::c003_a274::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a274::KERNEL)),

                                    // #[cfg(feature = "c003_a275")]
                                    // "c003_a275" => Some((c003::c003_a275::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a275::KERNEL)),

                                    // #[cfg(feature = "c003_a276")]
                                    // "c003_a276" => Some((c003::c003_a276::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a276::KERNEL)),

                                    // #[cfg(feature = "c003_a277")]
                                    // "c003_a277" => Some((c003::c003_a277::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a277::KERNEL)),

                                    // #[cfg(feature = "c003_a278")]
                                    // "c003_a278" => Some((c003::c003_a278::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a278::KERNEL)),

                                    // #[cfg(feature = "c003_a279")]
                                    // "c003_a279" => Some((c003::c003_a279::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a279::KERNEL)),

                                    // #[cfg(feature = "c003_a280")]
                                    // "c003_a280" => Some((c003::c003_a280::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a280::KERNEL)),

                                    // #[cfg(feature = "c003_a281")]
                                    // "c003_a281" => Some((c003::c003_a281::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a281::KERNEL)),

                                    // #[cfg(feature = "c003_a282")]
                                    // "c003_a282" => Some((c003::c003_a282::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a282::KERNEL)),

                                    // #[cfg(feature = "c003_a283")]
                                    // "c003_a283" => Some((c003::c003_a283::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a283::KERNEL)),

                                    // #[cfg(feature = "c003_a284")]
                                    // "c003_a284" => Some((c003::c003_a284::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a284::KERNEL)),

                                    // #[cfg(feature = "c003_a285")]
                                    // "c003_a285" => Some((c003::c003_a285::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a285::KERNEL)),

                                    // #[cfg(feature = "c003_a286")]
                                    // "c003_a286" => Some((c003::c003_a286::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a286::KERNEL)),

                                    // #[cfg(feature = "c003_a287")]
                                    // "c003_a287" => Some((c003::c003_a287::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a287::KERNEL)),

                                    // #[cfg(feature = "c003_a288")]
                                    // "c003_a288" => Some((c003::c003_a288::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a288::KERNEL)),

                                    // #[cfg(feature = "c003_a289")]
                                    // "c003_a289" => Some((c003::c003_a289::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a289::KERNEL)),

                                    // #[cfg(feature = "c003_a290")]
                                    // "c003_a290" => Some((c003::c003_a290::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a290::KERNEL)),

                                    // #[cfg(feature = "c003_a291")]
                                    // "c003_a291" => Some((c003::c003_a291::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a291::KERNEL)),

                                    // #[cfg(feature = "c003_a292")]
                                    // "c003_a292" => Some((c003::c003_a292::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a292::KERNEL)),

                                    // #[cfg(feature = "c003_a293")]
                                    // "c003_a293" => Some((c003::c003_a293::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a293::KERNEL)),

                                    // #[cfg(feature = "c003_a294")]
                                    // "c003_a294" => Some((c003::c003_a294::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a294::KERNEL)),

                                    // #[cfg(feature = "c003_a295")]
                                    // "c003_a295" => Some((c003::c003_a295::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a295::KERNEL)),

                                    // #[cfg(feature = "c003_a296")]
                                    // "c003_a296" => Some((c003::c003_a296::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a296::KERNEL)),

                                    // #[cfg(feature = "c003_a297")]
                                    // "c003_a297" => Some((c003::c003_a297::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a297::KERNEL)),

                                    // #[cfg(feature = "c003_a298")]
                                    // "c003_a298" => Some((c003::c003_a298::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a298::KERNEL)),

                                    // #[cfg(feature = "c003_a299")]
                                    // "c003_a299" => Some((c003::c003_a299::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a299::KERNEL)),

                                    // #[cfg(feature = "c003_a300")]
                                    // "c003_a300" => Some((c003::c003_a300::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a300::KERNEL)),

                                    // #[cfg(feature = "c003_a301")]
                                    // "c003_a301" => Some((c003::c003_a301::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a301::KERNEL)),

                                    // #[cfg(feature = "c003_a302")]
                                    // "c003_a302" => Some((c003::c003_a302::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a302::KERNEL)),

                                    // #[cfg(feature = "c003_a303")]
                                    // "c003_a303" => Some((c003::c003_a303::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a303::KERNEL)),

                                    // #[cfg(feature = "c003_a304")]
                                    // "c003_a304" => Some((c003::c003_a304::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a304::KERNEL)),

                                    // #[cfg(feature = "c003_a305")]
                                    // "c003_a305" => Some((c003::c003_a305::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a305::KERNEL)),

                                    // #[cfg(feature = "c003_a306")]
                                    // "c003_a306" => Some((c003::c003_a306::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a306::KERNEL)),

                                    // #[cfg(feature = "c003_a307")]
                                    // "c003_a307" => Some((c003::c003_a307::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a307::KERNEL)),

                                    // #[cfg(feature = "c003_a308")]
                                    // "c003_a308" => Some((c003::c003_a308::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a308::KERNEL)),

                                    // #[cfg(feature = "c003_a309")]
                                    // "c003_a309" => Some((c003::c003_a309::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a309::KERNEL)),

                                    // #[cfg(feature = "c003_a310")]
                                    // "c003_a310" => Some((c003::c003_a310::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a310::KERNEL)),

                                    // #[cfg(feature = "c003_a311")]
                                    // "c003_a311" => Some((c003::c003_a311::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a311::KERNEL)),

                                    // #[cfg(feature = "c003_a312")]
                                    // "c003_a312" => Some((c003::c003_a312::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a312::KERNEL)),

                                    // #[cfg(feature = "c003_a313")]
                                    // "c003_a313" => Some((c003::c003_a313::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a313::KERNEL)),

                                    // #[cfg(feature = "c003_a314")]
                                    // "c003_a314" => Some((c003::c003_a314::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a314::KERNEL)),

                                    // #[cfg(feature = "c003_a315")]
                                    // "c003_a315" => Some((c003::c003_a315::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a315::KERNEL)),

                                    // #[cfg(feature = "c003_a316")]
                                    // "c003_a316" => Some((c003::c003_a316::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a316::KERNEL)),

                                    // #[cfg(feature = "c003_a317")]
                                    // "c003_a317" => Some((c003::c003_a317::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a317::KERNEL)),

                                    // #[cfg(feature = "c003_a318")]
                                    // "c003_a318" => Some((c003::c003_a318::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a318::KERNEL)),

                                    // #[cfg(feature = "c003_a319")]
                                    // "c003_a319" => Some((c003::c003_a319::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a319::KERNEL)),

                                    // #[cfg(feature = "c003_a320")]
                                    // "c003_a320" => Some((c003::c003_a320::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a320::KERNEL)),

                                    // #[cfg(feature = "c003_a321")]
                                    // "c003_a321" => Some((c003::c003_a321::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a321::KERNEL)),

                                    // #[cfg(feature = "c003_a322")]
                                    // "c003_a322" => Some((c003::c003_a322::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a322::KERNEL)),

                                    // #[cfg(feature = "c003_a323")]
                                    // "c003_a323" => Some((c003::c003_a323::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a323::KERNEL)),

                                    // #[cfg(feature = "c003_a324")]
                                    // "c003_a324" => Some((c003::c003_a324::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a324::KERNEL)),

                                    // #[cfg(feature = "c003_a325")]
                                    // "c003_a325" => Some((c003::c003_a325::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a325::KERNEL)),

                                    // #[cfg(feature = "c003_a326")]
                                    // "c003_a326" => Some((c003::c003_a326::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a326::KERNEL)),

                                    // #[cfg(feature = "c003_a327")]
                                    // "c003_a327" => Some((c003::c003_a327::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a327::KERNEL)),

                                    // #[cfg(feature = "c003_a328")]
                                    // "c003_a328" => Some((c003::c003_a328::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a328::KERNEL)),

                                    // #[cfg(feature = "c003_a329")]
                                    // "c003_a329" => Some((c003::c003_a329::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a329::KERNEL)),

                                    // #[cfg(feature = "c003_a330")]
                                    // "c003_a330" => Some((c003::c003_a330::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a330::KERNEL)),

                                    // #[cfg(feature = "c003_a331")]
                                    // "c003_a331" => Some((c003::c003_a331::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a331::KERNEL)),

                                    // #[cfg(feature = "c003_a332")]
                                    // "c003_a332" => Some((c003::c003_a332::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a332::KERNEL)),

                                    // #[cfg(feature = "c003_a333")]
                                    // "c003_a333" => Some((c003::c003_a333::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a333::KERNEL)),

                                    // #[cfg(feature = "c003_a334")]
                                    // "c003_a334" => Some((c003::c003_a334::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a334::KERNEL)),

                                    // #[cfg(feature = "c003_a335")]
                                    // "c003_a335" => Some((c003::c003_a335::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a335::KERNEL)),

                                    // #[cfg(feature = "c003_a336")]
                                    // "c003_a336" => Some((c003::c003_a336::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a336::KERNEL)),

                                    // #[cfg(feature = "c003_a337")]
                                    // "c003_a337" => Some((c003::c003_a337::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a337::KERNEL)),

                                    // #[cfg(feature = "c003_a338")]
                                    // "c003_a338" => Some((c003::c003_a338::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a338::KERNEL)),

                                    // #[cfg(feature = "c003_a339")]
                                    // "c003_a339" => Some((c003::c003_a339::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a339::KERNEL)),

                                    // #[cfg(feature = "c003_a340")]
                                    // "c003_a340" => Some((c003::c003_a340::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a340::KERNEL)),

                                    // #[cfg(feature = "c003_a341")]
                                    // "c003_a341" => Some((c003::c003_a341::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a341::KERNEL)),

                                    // #[cfg(feature = "c003_a342")]
                                    // "c003_a342" => Some((c003::c003_a342::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a342::KERNEL)),

                                    // #[cfg(feature = "c003_a343")]
                                    // "c003_a343" => Some((c003::c003_a343::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a343::KERNEL)),

                                    // #[cfg(feature = "c003_a344")]
                                    // "c003_a344" => Some((c003::c003_a344::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a344::KERNEL)),

                                    // #[cfg(feature = "c003_a345")]
                                    // "c003_a345" => Some((c003::c003_a345::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a345::KERNEL)),

                                    // #[cfg(feature = "c003_a346")]
                                    // "c003_a346" => Some((c003::c003_a346::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a346::KERNEL)),

                                    // #[cfg(feature = "c003_a347")]
                                    // "c003_a347" => Some((c003::c003_a347::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a347::KERNEL)),

                                    // #[cfg(feature = "c003_a348")]
                                    // "c003_a348" => Some((c003::c003_a348::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a348::KERNEL)),

                                    // #[cfg(feature = "c003_a349")]
                                    // "c003_a349" => Some((c003::c003_a349::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a349::KERNEL)),

                                    // #[cfg(feature = "c003_a350")]
                                    // "c003_a350" => Some((c003::c003_a350::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a350::KERNEL)),

                                    // #[cfg(feature = "c003_a351")]
                                    // "c003_a351" => Some((c003::c003_a351::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a351::KERNEL)),

                                    // #[cfg(feature = "c003_a352")]
                                    // "c003_a352" => Some((c003::c003_a352::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a352::KERNEL)),

                                    // #[cfg(feature = "c003_a353")]
                                    // "c003_a353" => Some((c003::c003_a353::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a353::KERNEL)),

                                    // #[cfg(feature = "c003_a354")]
                                    // "c003_a354" => Some((c003::c003_a354::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a354::KERNEL)),

                                    // #[cfg(feature = "c003_a355")]
                                    // "c003_a355" => Some((c003::c003_a355::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a355::KERNEL)),

                                    // #[cfg(feature = "c003_a356")]
                                    // "c003_a356" => Some((c003::c003_a356::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a356::KERNEL)),

                                    // #[cfg(feature = "c003_a357")]
                                    // "c003_a357" => Some((c003::c003_a357::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a357::KERNEL)),

                                    // #[cfg(feature = "c003_a358")]
                                    // "c003_a358" => Some((c003::c003_a358::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a358::KERNEL)),

                                    // #[cfg(feature = "c003_a359")]
                                    // "c003_a359" => Some((c003::c003_a359::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a359::KERNEL)),

                                    // #[cfg(feature = "c003_a360")]
                                    // "c003_a360" => Some((c003::c003_a360::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a360::KERNEL)),

                                    // #[cfg(feature = "c003_a361")]
                                    // "c003_a361" => Some((c003::c003_a361::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a361::KERNEL)),

                                    // #[cfg(feature = "c003_a362")]
                                    // "c003_a362" => Some((c003::c003_a362::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a362::KERNEL)),

                                    // #[cfg(feature = "c003_a363")]
                                    // "c003_a363" => Some((c003::c003_a363::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a363::KERNEL)),

                                    // #[cfg(feature = "c003_a364")]
                                    // "c003_a364" => Some((c003::c003_a364::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a364::KERNEL)),

                                    // #[cfg(feature = "c003_a365")]
                                    // "c003_a365" => Some((c003::c003_a365::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a365::KERNEL)),

                                    // #[cfg(feature = "c003_a366")]
                                    // "c003_a366" => Some((c003::c003_a366::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a366::KERNEL)),

                                    // #[cfg(feature = "c003_a367")]
                                    // "c003_a367" => Some((c003::c003_a367::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a367::KERNEL)),

                                    // #[cfg(feature = "c003_a368")]
                                    // "c003_a368" => Some((c003::c003_a368::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a368::KERNEL)),

                                    // #[cfg(feature = "c003_a369")]
                                    // "c003_a369" => Some((c003::c003_a369::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a369::KERNEL)),

                                    // #[cfg(feature = "c003_a370")]
                                    // "c003_a370" => Some((c003::c003_a370::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a370::KERNEL)),

                                    // #[cfg(feature = "c003_a371")]
                                    // "c003_a371" => Some((c003::c003_a371::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a371::KERNEL)),

                                    // #[cfg(feature = "c003_a372")]
                                    // "c003_a372" => Some((c003::c003_a372::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a372::KERNEL)),

                                    // #[cfg(feature = "c003_a373")]
                                    // "c003_a373" => Some((c003::c003_a373::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a373::KERNEL)),

                                    // #[cfg(feature = "c003_a374")]
                                    // "c003_a374" => Some((c003::c003_a374::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a374::KERNEL)),

                                    // #[cfg(feature = "c003_a375")]
                                    // "c003_a375" => Some((c003::c003_a375::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a375::KERNEL)),

                                    // #[cfg(feature = "c003_a376")]
                                    // "c003_a376" => Some((c003::c003_a376::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a376::KERNEL)),

                                    // #[cfg(feature = "c003_a377")]
                                    // "c003_a377" => Some((c003::c003_a377::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a377::KERNEL)),

                                    // #[cfg(feature = "c003_a378")]
                                    // "c003_a378" => Some((c003::c003_a378::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a378::KERNEL)),

                                    // #[cfg(feature = "c003_a379")]
                                    // "c003_a379" => Some((c003::c003_a379::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a379::KERNEL)),

                                    // #[cfg(feature = "c003_a380")]
                                    // "c003_a380" => Some((c003::c003_a380::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a380::KERNEL)),

                                    // #[cfg(feature = "c003_a381")]
                                    // "c003_a381" => Some((c003::c003_a381::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a381::KERNEL)),

                                    // #[cfg(feature = "c003_a382")]
                                    // "c003_a382" => Some((c003::c003_a382::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a382::KERNEL)),

                                    // #[cfg(feature = "c003_a383")]
                                    // "c003_a383" => Some((c003::c003_a383::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a383::KERNEL)),

                                    // #[cfg(feature = "c003_a384")]
                                    // "c003_a384" => Some((c003::c003_a384::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a384::KERNEL)),

                                    // #[cfg(feature = "c003_a385")]
                                    // "c003_a385" => Some((c003::c003_a385::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a385::KERNEL)),

                                    // #[cfg(feature = "c003_a386")]
                                    // "c003_a386" => Some((c003::c003_a386::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a386::KERNEL)),

                                    // #[cfg(feature = "c003_a387")]
                                    // "c003_a387" => Some((c003::c003_a387::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a387::KERNEL)),

                                    // #[cfg(feature = "c003_a388")]
                                    // "c003_a388" => Some((c003::c003_a388::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a388::KERNEL)),

                                    // #[cfg(feature = "c003_a389")]
                                    // "c003_a389" => Some((c003::c003_a389::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a389::KERNEL)),

                                    // #[cfg(feature = "c003_a390")]
                                    // "c003_a390" => Some((c003::c003_a390::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a390::KERNEL)),

                                    // #[cfg(feature = "c003_a391")]
                                    // "c003_a391" => Some((c003::c003_a391::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a391::KERNEL)),

                                    // #[cfg(feature = "c003_a392")]
                                    // "c003_a392" => Some((c003::c003_a392::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a392::KERNEL)),

                                    // #[cfg(feature = "c003_a393")]
                                    // "c003_a393" => Some((c003::c003_a393::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a393::KERNEL)),

                                    // #[cfg(feature = "c003_a394")]
                                    // "c003_a394" => Some((c003::c003_a394::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a394::KERNEL)),

                                    // #[cfg(feature = "c003_a395")]
                                    // "c003_a395" => Some((c003::c003_a395::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a395::KERNEL)),

                                    // #[cfg(feature = "c003_a396")]
                                    // "c003_a396" => Some((c003::c003_a396::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a396::KERNEL)),

                                    // #[cfg(feature = "c003_a397")]
                                    // "c003_a397" => Some((c003::c003_a397::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a397::KERNEL)),

                                    // #[cfg(feature = "c003_a398")]
                                    // "c003_a398" => Some((c003::c003_a398::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a398::KERNEL)),

                                    // #[cfg(feature = "c003_a399")]
                                    // "c003_a399" => Some((c003::c003_a399::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a399::KERNEL)),

                                    // #[cfg(feature = "c003_a400")]
                                    // "c003_a400" => Some((c003::c003_a400::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a400::KERNEL)),

                                    // #[cfg(feature = "c003_a401")]
                                    // "c003_a401" => Some((c003::c003_a401::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a401::KERNEL)),

                                    // #[cfg(feature = "c003_a402")]
                                    // "c003_a402" => Some((c003::c003_a402::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a402::KERNEL)),

                                    // #[cfg(feature = "c003_a403")]
                                    // "c003_a403" => Some((c003::c003_a403::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a403::KERNEL)),

                                    // #[cfg(feature = "c003_a404")]
                                    // "c003_a404" => Some((c003::c003_a404::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a404::KERNEL)),

                                    // #[cfg(feature = "c003_a405")]
                                    // "c003_a405" => Some((c003::c003_a405::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a405::KERNEL)),

                                    // #[cfg(feature = "c003_a406")]
                                    // "c003_a406" => Some((c003::c003_a406::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a406::KERNEL)),

                                    // #[cfg(feature = "c003_a407")]
                                    // "c003_a407" => Some((c003::c003_a407::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a407::KERNEL)),

                                    // #[cfg(feature = "c003_a408")]
                                    // "c003_a408" => Some((c003::c003_a408::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a408::KERNEL)),

                                    // #[cfg(feature = "c003_a409")]
                                    // "c003_a409" => Some((c003::c003_a409::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a409::KERNEL)),

                                    // #[cfg(feature = "c003_a410")]
                                    // "c003_a410" => Some((c003::c003_a410::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a410::KERNEL)),

                                    // #[cfg(feature = "c003_a411")]
                                    // "c003_a411" => Some((c003::c003_a411::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a411::KERNEL)),

                                    // #[cfg(feature = "c003_a412")]
                                    // "c003_a412" => Some((c003::c003_a412::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a412::KERNEL)),

                                    // #[cfg(feature = "c003_a413")]
                                    // "c003_a413" => Some((c003::c003_a413::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a413::KERNEL)),

                                    // #[cfg(feature = "c003_a414")]
                                    // "c003_a414" => Some((c003::c003_a414::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a414::KERNEL)),

                                    // #[cfg(feature = "c003_a415")]
                                    // "c003_a415" => Some((c003::c003_a415::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a415::KERNEL)),

                                    // #[cfg(feature = "c003_a416")]
                                    // "c003_a416" => Some((c003::c003_a416::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a416::KERNEL)),

                                    // #[cfg(feature = "c003_a417")]
                                    // "c003_a417" => Some((c003::c003_a417::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a417::KERNEL)),

                                    // #[cfg(feature = "c003_a418")]
                                    // "c003_a418" => Some((c003::c003_a418::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a418::KERNEL)),

                                    // #[cfg(feature = "c003_a419")]
                                    // "c003_a419" => Some((c003::c003_a419::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a419::KERNEL)),

                                    // #[cfg(feature = "c003_a420")]
                                    // "c003_a420" => Some((c003::c003_a420::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a420::KERNEL)),

                                    // #[cfg(feature = "c003_a421")]
                                    // "c003_a421" => Some((c003::c003_a421::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a421::KERNEL)),

                                    // #[cfg(feature = "c003_a422")]
                                    // "c003_a422" => Some((c003::c003_a422::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a422::KERNEL)),

                                    // #[cfg(feature = "c003_a423")]
                                    // "c003_a423" => Some((c003::c003_a423::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a423::KERNEL)),

                                    // #[cfg(feature = "c003_a424")]
                                    // "c003_a424" => Some((c003::c003_a424::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a424::KERNEL)),

                                    // #[cfg(feature = "c003_a425")]
                                    // "c003_a425" => Some((c003::c003_a425::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a425::KERNEL)),

                                    // #[cfg(feature = "c003_a426")]
                                    // "c003_a426" => Some((c003::c003_a426::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a426::KERNEL)),

                                    // #[cfg(feature = "c003_a427")]
                                    // "c003_a427" => Some((c003::c003_a427::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a427::KERNEL)),

                                    // #[cfg(feature = "c003_a428")]
                                    // "c003_a428" => Some((c003::c003_a428::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a428::KERNEL)),

                                    // #[cfg(feature = "c003_a429")]
                                    // "c003_a429" => Some((c003::c003_a429::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a429::KERNEL)),

                                    // #[cfg(feature = "c003_a430")]
                                    // "c003_a430" => Some((c003::c003_a430::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a430::KERNEL)),

                                    // #[cfg(feature = "c003_a431")]
                                    // "c003_a431" => Some((c003::c003_a431::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a431::KERNEL)),

                                    // #[cfg(feature = "c003_a432")]
                                    // "c003_a432" => Some((c003::c003_a432::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a432::KERNEL)),

                                    // #[cfg(feature = "c003_a433")]
                                    // "c003_a433" => Some((c003::c003_a433::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a433::KERNEL)),

                                    // #[cfg(feature = "c003_a434")]
                                    // "c003_a434" => Some((c003::c003_a434::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a434::KERNEL)),

                                    // #[cfg(feature = "c003_a435")]
                                    // "c003_a435" => Some((c003::c003_a435::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a435::KERNEL)),

                                    // #[cfg(feature = "c003_a436")]
                                    // "c003_a436" => Some((c003::c003_a436::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a436::KERNEL)),

                                    // #[cfg(feature = "c003_a437")]
                                    // "c003_a437" => Some((c003::c003_a437::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a437::KERNEL)),

                                    // #[cfg(feature = "c003_a438")]
                                    // "c003_a438" => Some((c003::c003_a438::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a438::KERNEL)),

                                    // #[cfg(feature = "c003_a439")]
                                    // "c003_a439" => Some((c003::c003_a439::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a439::KERNEL)),

                                    // #[cfg(feature = "c003_a440")]
                                    // "c003_a440" => Some((c003::c003_a440::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a440::KERNEL)),

                                    // #[cfg(feature = "c003_a441")]
                                    // "c003_a441" => Some((c003::c003_a441::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a441::KERNEL)),

                                    // #[cfg(feature = "c003_a442")]
                                    // "c003_a442" => Some((c003::c003_a442::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a442::KERNEL)),

                                    // #[cfg(feature = "c003_a443")]
                                    // "c003_a443" => Some((c003::c003_a443::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a443::KERNEL)),

                                    // #[cfg(feature = "c003_a444")]
                                    // "c003_a444" => Some((c003::c003_a444::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a444::KERNEL)),

                                    // #[cfg(feature = "c003_a445")]
                                    // "c003_a445" => Some((c003::c003_a445::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a445::KERNEL)),

                                    // #[cfg(feature = "c003_a446")]
                                    // "c003_a446" => Some((c003::c003_a446::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a446::KERNEL)),

                                    // #[cfg(feature = "c003_a447")]
                                    // "c003_a447" => Some((c003::c003_a447::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a447::KERNEL)),

                                    // #[cfg(feature = "c003_a448")]
                                    // "c003_a448" => Some((c003::c003_a448::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a448::KERNEL)),

                                    // #[cfg(feature = "c003_a449")]
                                    // "c003_a449" => Some((c003::c003_a449::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a449::KERNEL)),

                                    // #[cfg(feature = "c003_a450")]
                                    // "c003_a450" => Some((c003::c003_a450::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a450::KERNEL)),

                                    // #[cfg(feature = "c003_a451")]
                                    // "c003_a451" => Some((c003::c003_a451::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a451::KERNEL)),

                                    // #[cfg(feature = "c003_a452")]
                                    // "c003_a452" => Some((c003::c003_a452::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a452::KERNEL)),

                                    // #[cfg(feature = "c003_a453")]
                                    // "c003_a453" => Some((c003::c003_a453::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a453::KERNEL)),

                                    // #[cfg(feature = "c003_a454")]
                                    // "c003_a454" => Some((c003::c003_a454::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a454::KERNEL)),

                                    // #[cfg(feature = "c003_a455")]
                                    // "c003_a455" => Some((c003::c003_a455::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a455::KERNEL)),

                                    // #[cfg(feature = "c003_a456")]
                                    // "c003_a456" => Some((c003::c003_a456::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a456::KERNEL)),

                                    // #[cfg(feature = "c003_a457")]
                                    // "c003_a457" => Some((c003::c003_a457::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a457::KERNEL)),

                                    // #[cfg(feature = "c003_a458")]
                                    // "c003_a458" => Some((c003::c003_a458::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a458::KERNEL)),

                                    // #[cfg(feature = "c003_a459")]
                                    // "c003_a459" => Some((c003::c003_a459::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a459::KERNEL)),

                                    // #[cfg(feature = "c003_a460")]
                                    // "c003_a460" => Some((c003::c003_a460::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a460::KERNEL)),

                                    // #[cfg(feature = "c003_a461")]
                                    // "c003_a461" => Some((c003::c003_a461::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a461::KERNEL)),

                                    // #[cfg(feature = "c003_a462")]
                                    // "c003_a462" => Some((c003::c003_a462::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a462::KERNEL)),

                                    // #[cfg(feature = "c003_a463")]
                                    // "c003_a463" => Some((c003::c003_a463::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a463::KERNEL)),

                                    // #[cfg(feature = "c003_a464")]
                                    // "c003_a464" => Some((c003::c003_a464::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a464::KERNEL)),

                                    // #[cfg(feature = "c003_a465")]
                                    // "c003_a465" => Some((c003::c003_a465::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a465::KERNEL)),

                                    // #[cfg(feature = "c003_a466")]
                                    // "c003_a466" => Some((c003::c003_a466::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a466::KERNEL)),

                                    // #[cfg(feature = "c003_a467")]
                                    // "c003_a467" => Some((c003::c003_a467::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a467::KERNEL)),

                                    // #[cfg(feature = "c003_a468")]
                                    // "c003_a468" => Some((c003::c003_a468::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a468::KERNEL)),

                                    // #[cfg(feature = "c003_a469")]
                                    // "c003_a469" => Some((c003::c003_a469::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a469::KERNEL)),

                                    // #[cfg(feature = "c003_a470")]
                                    // "c003_a470" => Some((c003::c003_a470::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a470::KERNEL)),

                                    // #[cfg(feature = "c003_a471")]
                                    // "c003_a471" => Some((c003::c003_a471::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a471::KERNEL)),

                                    // #[cfg(feature = "c003_a472")]
                                    // "c003_a472" => Some((c003::c003_a472::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a472::KERNEL)),

                                    // #[cfg(feature = "c003_a473")]
                                    // "c003_a473" => Some((c003::c003_a473::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a473::KERNEL)),

                                    // #[cfg(feature = "c003_a474")]
                                    // "c003_a474" => Some((c003::c003_a474::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a474::KERNEL)),

                                    // #[cfg(feature = "c003_a475")]
                                    // "c003_a475" => Some((c003::c003_a475::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a475::KERNEL)),

                                    // #[cfg(feature = "c003_a476")]
                                    // "c003_a476" => Some((c003::c003_a476::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a476::KERNEL)),

                                    // #[cfg(feature = "c003_a477")]
                                    // "c003_a477" => Some((c003::c003_a477::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a477::KERNEL)),

                                    // #[cfg(feature = "c003_a478")]
                                    // "c003_a478" => Some((c003::c003_a478::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a478::KERNEL)),

                                    // #[cfg(feature = "c003_a479")]
                                    // "c003_a479" => Some((c003::c003_a479::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a479::KERNEL)),

                                    // #[cfg(feature = "c003_a480")]
                                    // "c003_a480" => Some((c003::c003_a480::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a480::KERNEL)),

                                    // #[cfg(feature = "c003_a481")]
                                    // "c003_a481" => Some((c003::c003_a481::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a481::KERNEL)),

                                    // #[cfg(feature = "c003_a482")]
                                    // "c003_a482" => Some((c003::c003_a482::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a482::KERNEL)),

                                    // #[cfg(feature = "c003_a483")]
                                    // "c003_a483" => Some((c003::c003_a483::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a483::KERNEL)),

                                    // #[cfg(feature = "c003_a484")]
                                    // "c003_a484" => Some((c003::c003_a484::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a484::KERNEL)),

                                    // #[cfg(feature = "c003_a485")]
                                    // "c003_a485" => Some((c003::c003_a485::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a485::KERNEL)),

                                    // #[cfg(feature = "c003_a486")]
                                    // "c003_a486" => Some((c003::c003_a486::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a486::KERNEL)),

                                    // #[cfg(feature = "c003_a487")]
                                    // "c003_a487" => Some((c003::c003_a487::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a487::KERNEL)),

                                    // #[cfg(feature = "c003_a488")]
                                    // "c003_a488" => Some((c003::c003_a488::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a488::KERNEL)),

                                    // #[cfg(feature = "c003_a489")]
                                    // "c003_a489" => Some((c003::c003_a489::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a489::KERNEL)),

                                    // #[cfg(feature = "c003_a490")]
                                    // "c003_a490" => Some((c003::c003_a490::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a490::KERNEL)),

                                    // #[cfg(feature = "c003_a491")]
                                    // "c003_a491" => Some((c003::c003_a491::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a491::KERNEL)),

                                    // #[cfg(feature = "c003_a492")]
                                    // "c003_a492" => Some((c003::c003_a492::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a492::KERNEL)),

                                    // #[cfg(feature = "c003_a493")]
                                    // "c003_a493" => Some((c003::c003_a493::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a493::KERNEL)),

                                    // #[cfg(feature = "c003_a494")]
                                    // "c003_a494" => Some((c003::c003_a494::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a494::KERNEL)),

                                    // #[cfg(feature = "c003_a495")]
                                    // "c003_a495" => Some((c003::c003_a495::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a495::KERNEL)),

                                    // #[cfg(feature = "c003_a496")]
                                    // "c003_a496" => Some((c003::c003_a496::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a496::KERNEL)),

                                    // #[cfg(feature = "c003_a497")]
                                    // "c003_a497" => Some((c003::c003_a497::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a497::KERNEL)),

                                    // #[cfg(feature = "c003_a498")]
                                    // "c003_a498" => Some((c003::c003_a498::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a498::KERNEL)),

                                    // #[cfg(feature = "c003_a499")]
                                    // "c003_a499" => Some((c003::c003_a499::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a499::KERNEL)),

                                    // #[cfg(feature = "c003_a500")]
                                    // "c003_a500" => Some((c003::c003_a500::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a500::KERNEL)),

                                    // #[cfg(feature = "c003_a501")]
                                    // "c003_a501" => Some((c003::c003_a501::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a501::KERNEL)),

                                    // #[cfg(feature = "c003_a502")]
                                    // "c003_a502" => Some((c003::c003_a502::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a502::KERNEL)),

                                    // #[cfg(feature = "c003_a503")]
                                    // "c003_a503" => Some((c003::c003_a503::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a503::KERNEL)),

                                    // #[cfg(feature = "c003_a504")]
                                    // "c003_a504" => Some((c003::c003_a504::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a504::KERNEL)),

                                    // #[cfg(feature = "c003_a505")]
                                    // "c003_a505" => Some((c003::c003_a505::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a505::KERNEL)),

                                    // #[cfg(feature = "c003_a506")]
                                    // "c003_a506" => Some((c003::c003_a506::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a506::KERNEL)),

                                    // #[cfg(feature = "c003_a507")]
                                    // "c003_a507" => Some((c003::c003_a507::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a507::KERNEL)),

                                    // #[cfg(feature = "c003_a508")]
                                    // "c003_a508" => Some((c003::c003_a508::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a508::KERNEL)),

                                    // #[cfg(feature = "c003_a509")]
                                    // "c003_a509" => Some((c003::c003_a509::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a509::KERNEL)),

                                    // #[cfg(feature = "c003_a510")]
                                    // "c003_a510" => Some((c003::c003_a510::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a510::KERNEL)),

                                    // #[cfg(feature = "c003_a511")]
                                    // "c003_a511" => Some((c003::c003_a511::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a511::KERNEL)),

                                    // #[cfg(feature = "c003_a512")]
                                    // "c003_a512" => Some((c003::c003_a512::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a512::KERNEL)),

                                    // #[cfg(feature = "c003_a513")]
                                    // "c003_a513" => Some((c003::c003_a513::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a513::KERNEL)),

                                    // #[cfg(feature = "c003_a514")]
                                    // "c003_a514" => Some((c003::c003_a514::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a514::KERNEL)),

                                    // #[cfg(feature = "c003_a515")]
                                    // "c003_a515" => Some((c003::c003_a515::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a515::KERNEL)),

                                    // #[cfg(feature = "c003_a516")]
                                    // "c003_a516" => Some((c003::c003_a516::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a516::KERNEL)),

                                    // #[cfg(feature = "c003_a517")]
                                    // "c003_a517" => Some((c003::c003_a517::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a517::KERNEL)),

                                    // #[cfg(feature = "c003_a518")]
                                    // "c003_a518" => Some((c003::c003_a518::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a518::KERNEL)),

                                    // #[cfg(feature = "c003_a519")]
                                    // "c003_a519" => Some((c003::c003_a519::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a519::KERNEL)),

                                    // #[cfg(feature = "c003_a520")]
                                    // "c003_a520" => Some((c003::c003_a520::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a520::KERNEL)),

                                    // #[cfg(feature = "c003_a521")]
                                    // "c003_a521" => Some((c003::c003_a521::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a521::KERNEL)),

                                    // #[cfg(feature = "c003_a522")]
                                    // "c003_a522" => Some((c003::c003_a522::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a522::KERNEL)),

                                    // #[cfg(feature = "c003_a523")]
                                    // "c003_a523" => Some((c003::c003_a523::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a523::KERNEL)),

                                    // #[cfg(feature = "c003_a524")]
                                    // "c003_a524" => Some((c003::c003_a524::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a524::KERNEL)),

                                    // #[cfg(feature = "c003_a525")]
                                    // "c003_a525" => Some((c003::c003_a525::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a525::KERNEL)),

                                    // #[cfg(feature = "c003_a526")]
                                    // "c003_a526" => Some((c003::c003_a526::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a526::KERNEL)),

                                    // #[cfg(feature = "c003_a527")]
                                    // "c003_a527" => Some((c003::c003_a527::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a527::KERNEL)),

                                    // #[cfg(feature = "c003_a528")]
                                    // "c003_a528" => Some((c003::c003_a528::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a528::KERNEL)),

                                    // #[cfg(feature = "c003_a529")]
                                    // "c003_a529" => Some((c003::c003_a529::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a529::KERNEL)),

                                    // #[cfg(feature = "c003_a530")]
                                    // "c003_a530" => Some((c003::c003_a530::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a530::KERNEL)),

                                    // #[cfg(feature = "c003_a531")]
                                    // "c003_a531" => Some((c003::c003_a531::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a531::KERNEL)),

                                    // #[cfg(feature = "c003_a532")]
                                    // "c003_a532" => Some((c003::c003_a532::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a532::KERNEL)),

                                    // #[cfg(feature = "c003_a533")]
                                    // "c003_a533" => Some((c003::c003_a533::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a533::KERNEL)),

                                    // #[cfg(feature = "c003_a534")]
                                    // "c003_a534" => Some((c003::c003_a534::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a534::KERNEL)),

                                    // #[cfg(feature = "c003_a535")]
                                    // "c003_a535" => Some((c003::c003_a535::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a535::KERNEL)),

                                    // #[cfg(feature = "c003_a536")]
                                    // "c003_a536" => Some((c003::c003_a536::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a536::KERNEL)),

                                    // #[cfg(feature = "c003_a537")]
                                    // "c003_a537" => Some((c003::c003_a537::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a537::KERNEL)),

                                    // #[cfg(feature = "c003_a538")]
                                    // "c003_a538" => Some((c003::c003_a538::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a538::KERNEL)),

                                    // #[cfg(feature = "c003_a539")]
                                    // "c003_a539" => Some((c003::c003_a539::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a539::KERNEL)),

                                    // #[cfg(feature = "c003_a540")]
                                    // "c003_a540" => Some((c003::c003_a540::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a540::KERNEL)),

                                    // #[cfg(feature = "c003_a541")]
                                    // "c003_a541" => Some((c003::c003_a541::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a541::KERNEL)),

                                    // #[cfg(feature = "c003_a542")]
                                    // "c003_a542" => Some((c003::c003_a542::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a542::KERNEL)),

                                    // #[cfg(feature = "c003_a543")]
                                    // "c003_a543" => Some((c003::c003_a543::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a543::KERNEL)),

                                    // #[cfg(feature = "c003_a544")]
                                    // "c003_a544" => Some((c003::c003_a544::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a544::KERNEL)),

                                    // #[cfg(feature = "c003_a545")]
                                    // "c003_a545" => Some((c003::c003_a545::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a545::KERNEL)),

                                    // #[cfg(feature = "c003_a546")]
                                    // "c003_a546" => Some((c003::c003_a546::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a546::KERNEL)),

                                    // #[cfg(feature = "c003_a547")]
                                    // "c003_a547" => Some((c003::c003_a547::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a547::KERNEL)),

                                    // #[cfg(feature = "c003_a548")]
                                    // "c003_a548" => Some((c003::c003_a548::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a548::KERNEL)),

                                    // #[cfg(feature = "c003_a549")]
                                    // "c003_a549" => Some((c003::c003_a549::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a549::KERNEL)),

                                    // #[cfg(feature = "c003_a550")]
                                    // "c003_a550" => Some((c003::c003_a550::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a550::KERNEL)),

                                    // #[cfg(feature = "c003_a551")]
                                    // "c003_a551" => Some((c003::c003_a551::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a551::KERNEL)),

                                    // #[cfg(feature = "c003_a552")]
                                    // "c003_a552" => Some((c003::c003_a552::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a552::KERNEL)),

                                    // #[cfg(feature = "c003_a553")]
                                    // "c003_a553" => Some((c003::c003_a553::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a553::KERNEL)),

                                    // #[cfg(feature = "c003_a554")]
                                    // "c003_a554" => Some((c003::c003_a554::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a554::KERNEL)),

                                    // #[cfg(feature = "c003_a555")]
                                    // "c003_a555" => Some((c003::c003_a555::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a555::KERNEL)),

                                    // #[cfg(feature = "c003_a556")]
                                    // "c003_a556" => Some((c003::c003_a556::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a556::KERNEL)),

                                    // #[cfg(feature = "c003_a557")]
                                    // "c003_a557" => Some((c003::c003_a557::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a557::KERNEL)),

                                    // #[cfg(feature = "c003_a558")]
                                    // "c003_a558" => Some((c003::c003_a558::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a558::KERNEL)),

                                    // #[cfg(feature = "c003_a559")]
                                    // "c003_a559" => Some((c003::c003_a559::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a559::KERNEL)),

                                    // #[cfg(feature = "c003_a560")]
                                    // "c003_a560" => Some((c003::c003_a560::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a560::KERNEL)),

                                    // #[cfg(feature = "c003_a561")]
                                    // "c003_a561" => Some((c003::c003_a561::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a561::KERNEL)),

                                    // #[cfg(feature = "c003_a562")]
                                    // "c003_a562" => Some((c003::c003_a562::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a562::KERNEL)),

                                    // #[cfg(feature = "c003_a563")]
                                    // "c003_a563" => Some((c003::c003_a563::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a563::KERNEL)),

                                    // #[cfg(feature = "c003_a564")]
                                    // "c003_a564" => Some((c003::c003_a564::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a564::KERNEL)),

                                    // #[cfg(feature = "c003_a565")]
                                    // "c003_a565" => Some((c003::c003_a565::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a565::KERNEL)),

                                    // #[cfg(feature = "c003_a566")]
                                    // "c003_a566" => Some((c003::c003_a566::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a566::KERNEL)),

                                    // #[cfg(feature = "c003_a567")]
                                    // "c003_a567" => Some((c003::c003_a567::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a567::KERNEL)),

                                    // #[cfg(feature = "c003_a568")]
                                    // "c003_a568" => Some((c003::c003_a568::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a568::KERNEL)),

                                    // #[cfg(feature = "c003_a569")]
                                    // "c003_a569" => Some((c003::c003_a569::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a569::KERNEL)),

                                    // #[cfg(feature = "c003_a570")]
                                    // "c003_a570" => Some((c003::c003_a570::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a570::KERNEL)),

                                    // #[cfg(feature = "c003_a571")]
                                    // "c003_a571" => Some((c003::c003_a571::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a571::KERNEL)),

                                    // #[cfg(feature = "c003_a572")]
                                    // "c003_a572" => Some((c003::c003_a572::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a572::KERNEL)),

                                    // #[cfg(feature = "c003_a573")]
                                    // "c003_a573" => Some((c003::c003_a573::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a573::KERNEL)),

                                    // #[cfg(feature = "c003_a574")]
                                    // "c003_a574" => Some((c003::c003_a574::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a574::KERNEL)),

                                    // #[cfg(feature = "c003_a575")]
                                    // "c003_a575" => Some((c003::c003_a575::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a575::KERNEL)),

                                    // #[cfg(feature = "c003_a576")]
                                    // "c003_a576" => Some((c003::c003_a576::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a576::KERNEL)),

                                    // #[cfg(feature = "c003_a577")]
                                    // "c003_a577" => Some((c003::c003_a577::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a577::KERNEL)),

                                    // #[cfg(feature = "c003_a578")]
                                    // "c003_a578" => Some((c003::c003_a578::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a578::KERNEL)),

                                    // #[cfg(feature = "c003_a579")]
                                    // "c003_a579" => Some((c003::c003_a579::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a579::KERNEL)),

                                    // #[cfg(feature = "c003_a580")]
                                    // "c003_a580" => Some((c003::c003_a580::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a580::KERNEL)),

                                    // #[cfg(feature = "c003_a581")]
                                    // "c003_a581" => Some((c003::c003_a581::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a581::KERNEL)),

                                    // #[cfg(feature = "c003_a582")]
                                    // "c003_a582" => Some((c003::c003_a582::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a582::KERNEL)),

                                    // #[cfg(feature = "c003_a583")]
                                    // "c003_a583" => Some((c003::c003_a583::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a583::KERNEL)),

                                    // #[cfg(feature = "c003_a584")]
                                    // "c003_a584" => Some((c003::c003_a584::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a584::KERNEL)),

                                    // #[cfg(feature = "c003_a585")]
                                    // "c003_a585" => Some((c003::c003_a585::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a585::KERNEL)),

                                    // #[cfg(feature = "c003_a586")]
                                    // "c003_a586" => Some((c003::c003_a586::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a586::KERNEL)),

                                    // #[cfg(feature = "c003_a587")]
                                    // "c003_a587" => Some((c003::c003_a587::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a587::KERNEL)),

                                    // #[cfg(feature = "c003_a588")]
                                    // "c003_a588" => Some((c003::c003_a588::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a588::KERNEL)),

                                    // #[cfg(feature = "c003_a589")]
                                    // "c003_a589" => Some((c003::c003_a589::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a589::KERNEL)),

                                    // #[cfg(feature = "c003_a590")]
                                    // "c003_a590" => Some((c003::c003_a590::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a590::KERNEL)),

                                    // #[cfg(feature = "c003_a591")]
                                    // "c003_a591" => Some((c003::c003_a591::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a591::KERNEL)),

                                    // #[cfg(feature = "c003_a592")]
                                    // "c003_a592" => Some((c003::c003_a592::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a592::KERNEL)),

                                    // #[cfg(feature = "c003_a593")]
                                    // "c003_a593" => Some((c003::c003_a593::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a593::KERNEL)),

                                    // #[cfg(feature = "c003_a594")]
                                    // "c003_a594" => Some((c003::c003_a594::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a594::KERNEL)),

                                    // #[cfg(feature = "c003_a595")]
                                    // "c003_a595" => Some((c003::c003_a595::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a595::KERNEL)),

                                    // #[cfg(feature = "c003_a596")]
                                    // "c003_a596" => Some((c003::c003_a596::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a596::KERNEL)),

                                    // #[cfg(feature = "c003_a597")]
                                    // "c003_a597" => Some((c003::c003_a597::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a597::KERNEL)),

                                    // #[cfg(feature = "c003_a598")]
                                    // "c003_a598" => Some((c003::c003_a598::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a598::KERNEL)),

                                    // #[cfg(feature = "c003_a599")]
                                    // "c003_a599" => Some((c003::c003_a599::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a599::KERNEL)),

                                    // #[cfg(feature = "c003_a600")]
                                    // "c003_a600" => Some((c003::c003_a600::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a600::KERNEL)),

                                    // #[cfg(feature = "c003_a601")]
                                    // "c003_a601" => Some((c003::c003_a601::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a601::KERNEL)),

                                    // #[cfg(feature = "c003_a602")]
                                    // "c003_a602" => Some((c003::c003_a602::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a602::KERNEL)),

                                    // #[cfg(feature = "c003_a603")]
                                    // "c003_a603" => Some((c003::c003_a603::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a603::KERNEL)),

                                    // #[cfg(feature = "c003_a604")]
                                    // "c003_a604" => Some((c003::c003_a604::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a604::KERNEL)),

                                    // #[cfg(feature = "c003_a605")]
                                    // "c003_a605" => Some((c003::c003_a605::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a605::KERNEL)),

                                    // #[cfg(feature = "c003_a606")]
                                    // "c003_a606" => Some((c003::c003_a606::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a606::KERNEL)),

                                    // #[cfg(feature = "c003_a607")]
                                    // "c003_a607" => Some((c003::c003_a607::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a607::KERNEL)),

                                    // #[cfg(feature = "c003_a608")]
                                    // "c003_a608" => Some((c003::c003_a608::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a608::KERNEL)),

                                    // #[cfg(feature = "c003_a609")]
                                    // "c003_a609" => Some((c003::c003_a609::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a609::KERNEL)),

                                    // #[cfg(feature = "c003_a610")]
                                    // "c003_a610" => Some((c003::c003_a610::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a610::KERNEL)),

                                    // #[cfg(feature = "c003_a611")]
                                    // "c003_a611" => Some((c003::c003_a611::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a611::KERNEL)),

                                    // #[cfg(feature = "c003_a612")]
                                    // "c003_a612" => Some((c003::c003_a612::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a612::KERNEL)),

                                    // #[cfg(feature = "c003_a613")]
                                    // "c003_a613" => Some((c003::c003_a613::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a613::KERNEL)),

                                    // #[cfg(feature = "c003_a614")]
                                    // "c003_a614" => Some((c003::c003_a614::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a614::KERNEL)),

                                    // #[cfg(feature = "c003_a615")]
                                    // "c003_a615" => Some((c003::c003_a615::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a615::KERNEL)),

                                    // #[cfg(feature = "c003_a616")]
                                    // "c003_a616" => Some((c003::c003_a616::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a616::KERNEL)),

                                    // #[cfg(feature = "c003_a617")]
                                    // "c003_a617" => Some((c003::c003_a617::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a617::KERNEL)),

                                    // #[cfg(feature = "c003_a618")]
                                    // "c003_a618" => Some((c003::c003_a618::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a618::KERNEL)),

                                    // #[cfg(feature = "c003_a619")]
                                    // "c003_a619" => Some((c003::c003_a619::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a619::KERNEL)),

                                    // #[cfg(feature = "c003_a620")]
                                    // "c003_a620" => Some((c003::c003_a620::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a620::KERNEL)),

                                    // #[cfg(feature = "c003_a621")]
                                    // "c003_a621" => Some((c003::c003_a621::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a621::KERNEL)),

                                    // #[cfg(feature = "c003_a622")]
                                    // "c003_a622" => Some((c003::c003_a622::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a622::KERNEL)),

                                    // #[cfg(feature = "c003_a623")]
                                    // "c003_a623" => Some((c003::c003_a623::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a623::KERNEL)),

                                    // #[cfg(feature = "c003_a624")]
                                    // "c003_a624" => Some((c003::c003_a624::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a624::KERNEL)),

                                    // #[cfg(feature = "c003_a625")]
                                    // "c003_a625" => Some((c003::c003_a625::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a625::KERNEL)),

                                    // #[cfg(feature = "c003_a626")]
                                    // "c003_a626" => Some((c003::c003_a626::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a626::KERNEL)),

                                    // #[cfg(feature = "c003_a627")]
                                    // "c003_a627" => Some((c003::c003_a627::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a627::KERNEL)),

                                    // #[cfg(feature = "c003_a628")]
                                    // "c003_a628" => Some((c003::c003_a628::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a628::KERNEL)),

                                    // #[cfg(feature = "c003_a629")]
                                    // "c003_a629" => Some((c003::c003_a629::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a629::KERNEL)),

                                    // #[cfg(feature = "c003_a630")]
                                    // "c003_a630" => Some((c003::c003_a630::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a630::KERNEL)),

                                    // #[cfg(feature = "c003_a631")]
                                    // "c003_a631" => Some((c003::c003_a631::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a631::KERNEL)),

                                    // #[cfg(feature = "c003_a632")]
                                    // "c003_a632" => Some((c003::c003_a632::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a632::KERNEL)),

                                    // #[cfg(feature = "c003_a633")]
                                    // "c003_a633" => Some((c003::c003_a633::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a633::KERNEL)),

                                    // #[cfg(feature = "c003_a634")]
                                    // "c003_a634" => Some((c003::c003_a634::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a634::KERNEL)),

                                    // #[cfg(feature = "c003_a635")]
                                    // "c003_a635" => Some((c003::c003_a635::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a635::KERNEL)),

                                    // #[cfg(feature = "c003_a636")]
                                    // "c003_a636" => Some((c003::c003_a636::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a636::KERNEL)),

                                    // #[cfg(feature = "c003_a637")]
                                    // "c003_a637" => Some((c003::c003_a637::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a637::KERNEL)),

                                    // #[cfg(feature = "c003_a638")]
                                    // "c003_a638" => Some((c003::c003_a638::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a638::KERNEL)),

                                    // #[cfg(feature = "c003_a639")]
                                    // "c003_a639" => Some((c003::c003_a639::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a639::KERNEL)),

                                    // #[cfg(feature = "c003_a640")]
                                    // "c003_a640" => Some((c003::c003_a640::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a640::KERNEL)),

                                    // #[cfg(feature = "c003_a641")]
                                    // "c003_a641" => Some((c003::c003_a641::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a641::KERNEL)),

                                    // #[cfg(feature = "c003_a642")]
                                    // "c003_a642" => Some((c003::c003_a642::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a642::KERNEL)),

                                    // #[cfg(feature = "c003_a643")]
                                    // "c003_a643" => Some((c003::c003_a643::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a643::KERNEL)),

                                    // #[cfg(feature = "c003_a644")]
                                    // "c003_a644" => Some((c003::c003_a644::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a644::KERNEL)),

                                    // #[cfg(feature = "c003_a645")]
                                    // "c003_a645" => Some((c003::c003_a645::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a645::KERNEL)),

                                    // #[cfg(feature = "c003_a646")]
                                    // "c003_a646" => Some((c003::c003_a646::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a646::KERNEL)),

                                    // #[cfg(feature = "c003_a647")]
                                    // "c003_a647" => Some((c003::c003_a647::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a647::KERNEL)),

                                    // #[cfg(feature = "c003_a648")]
                                    // "c003_a648" => Some((c003::c003_a648::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a648::KERNEL)),

                                    // #[cfg(feature = "c003_a649")]
                                    // "c003_a649" => Some((c003::c003_a649::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a649::KERNEL)),

                                    // #[cfg(feature = "c003_a650")]
                                    // "c003_a650" => Some((c003::c003_a650::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a650::KERNEL)),

                                    // #[cfg(feature = "c003_a651")]
                                    // "c003_a651" => Some((c003::c003_a651::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a651::KERNEL)),

                                    // #[cfg(feature = "c003_a652")]
                                    // "c003_a652" => Some((c003::c003_a652::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a652::KERNEL)),

                                    // #[cfg(feature = "c003_a653")]
                                    // "c003_a653" => Some((c003::c003_a653::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a653::KERNEL)),

                                    // #[cfg(feature = "c003_a654")]
                                    // "c003_a654" => Some((c003::c003_a654::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a654::KERNEL)),

                                    // #[cfg(feature = "c003_a655")]
                                    // "c003_a655" => Some((c003::c003_a655::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a655::KERNEL)),

                                    // #[cfg(feature = "c003_a656")]
                                    // "c003_a656" => Some((c003::c003_a656::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a656::KERNEL)),

                                    // #[cfg(feature = "c003_a657")]
                                    // "c003_a657" => Some((c003::c003_a657::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a657::KERNEL)),

                                    // #[cfg(feature = "c003_a658")]
                                    // "c003_a658" => Some((c003::c003_a658::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a658::KERNEL)),

                                    // #[cfg(feature = "c003_a659")]
                                    // "c003_a659" => Some((c003::c003_a659::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a659::KERNEL)),

                                    // #[cfg(feature = "c003_a660")]
                                    // "c003_a660" => Some((c003::c003_a660::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a660::KERNEL)),

                                    // #[cfg(feature = "c003_a661")]
                                    // "c003_a661" => Some((c003::c003_a661::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a661::KERNEL)),

                                    // #[cfg(feature = "c003_a662")]
                                    // "c003_a662" => Some((c003::c003_a662::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a662::KERNEL)),

                                    // #[cfg(feature = "c003_a663")]
                                    // "c003_a663" => Some((c003::c003_a663::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a663::KERNEL)),

                                    // #[cfg(feature = "c003_a664")]
                                    // "c003_a664" => Some((c003::c003_a664::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a664::KERNEL)),

                                    // #[cfg(feature = "c003_a665")]
                                    // "c003_a665" => Some((c003::c003_a665::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a665::KERNEL)),

                                    // #[cfg(feature = "c003_a666")]
                                    // "c003_a666" => Some((c003::c003_a666::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a666::KERNEL)),

                                    // #[cfg(feature = "c003_a667")]
                                    // "c003_a667" => Some((c003::c003_a667::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a667::KERNEL)),

                                    // #[cfg(feature = "c003_a668")]
                                    // "c003_a668" => Some((c003::c003_a668::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a668::KERNEL)),

                                    // #[cfg(feature = "c003_a669")]
                                    // "c003_a669" => Some((c003::c003_a669::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a669::KERNEL)),

                                    // #[cfg(feature = "c003_a670")]
                                    // "c003_a670" => Some((c003::c003_a670::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a670::KERNEL)),

                                    // #[cfg(feature = "c003_a671")]
                                    // "c003_a671" => Some((c003::c003_a671::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a671::KERNEL)),

                                    // #[cfg(feature = "c003_a672")]
                                    // "c003_a672" => Some((c003::c003_a672::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a672::KERNEL)),

                                    // #[cfg(feature = "c003_a673")]
                                    // "c003_a673" => Some((c003::c003_a673::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a673::KERNEL)),

                                    // #[cfg(feature = "c003_a674")]
                                    // "c003_a674" => Some((c003::c003_a674::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a674::KERNEL)),

                                    // #[cfg(feature = "c003_a675")]
                                    // "c003_a675" => Some((c003::c003_a675::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a675::KERNEL)),

                                    // #[cfg(feature = "c003_a676")]
                                    // "c003_a676" => Some((c003::c003_a676::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a676::KERNEL)),

                                    // #[cfg(feature = "c003_a677")]
                                    // "c003_a677" => Some((c003::c003_a677::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a677::KERNEL)),

                                    // #[cfg(feature = "c003_a678")]
                                    // "c003_a678" => Some((c003::c003_a678::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a678::KERNEL)),

                                    // #[cfg(feature = "c003_a679")]
                                    // "c003_a679" => Some((c003::c003_a679::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a679::KERNEL)),

                                    // #[cfg(feature = "c003_a680")]
                                    // "c003_a680" => Some((c003::c003_a680::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a680::KERNEL)),

                                    // #[cfg(feature = "c003_a681")]
                                    // "c003_a681" => Some((c003::c003_a681::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a681::KERNEL)),

                                    // #[cfg(feature = "c003_a682")]
                                    // "c003_a682" => Some((c003::c003_a682::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a682::KERNEL)),

                                    // #[cfg(feature = "c003_a683")]
                                    // "c003_a683" => Some((c003::c003_a683::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a683::KERNEL)),

                                    // #[cfg(feature = "c003_a684")]
                                    // "c003_a684" => Some((c003::c003_a684::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a684::KERNEL)),

                                    // #[cfg(feature = "c003_a685")]
                                    // "c003_a685" => Some((c003::c003_a685::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a685::KERNEL)),

                                    // #[cfg(feature = "c003_a686")]
                                    // "c003_a686" => Some((c003::c003_a686::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a686::KERNEL)),

                                    // #[cfg(feature = "c003_a687")]
                                    // "c003_a687" => Some((c003::c003_a687::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a687::KERNEL)),

                                    // #[cfg(feature = "c003_a688")]
                                    // "c003_a688" => Some((c003::c003_a688::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a688::KERNEL)),

                                    // #[cfg(feature = "c003_a689")]
                                    // "c003_a689" => Some((c003::c003_a689::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a689::KERNEL)),

                                    // #[cfg(feature = "c003_a690")]
                                    // "c003_a690" => Some((c003::c003_a690::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a690::KERNEL)),

                                    // #[cfg(feature = "c003_a691")]
                                    // "c003_a691" => Some((c003::c003_a691::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a691::KERNEL)),

                                    // #[cfg(feature = "c003_a692")]
                                    // "c003_a692" => Some((c003::c003_a692::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a692::KERNEL)),

                                    // #[cfg(feature = "c003_a693")]
                                    // "c003_a693" => Some((c003::c003_a693::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a693::KERNEL)),

                                    // #[cfg(feature = "c003_a694")]
                                    // "c003_a694" => Some((c003::c003_a694::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a694::KERNEL)),

                                    // #[cfg(feature = "c003_a695")]
                                    // "c003_a695" => Some((c003::c003_a695::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a695::KERNEL)),

                                    // #[cfg(feature = "c003_a696")]
                                    // "c003_a696" => Some((c003::c003_a696::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a696::KERNEL)),

                                    // #[cfg(feature = "c003_a697")]
                                    // "c003_a697" => Some((c003::c003_a697::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a697::KERNEL)),

                                    // #[cfg(feature = "c003_a698")]
                                    // "c003_a698" => Some((c003::c003_a698::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a698::KERNEL)),

                                    // #[cfg(feature = "c003_a699")]
                                    // "c003_a699" => Some((c003::c003_a699::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a699::KERNEL)),

                                    // #[cfg(feature = "c003_a700")]
                                    // "c003_a700" => Some((c003::c003_a700::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a700::KERNEL)),

                                    // #[cfg(feature = "c003_a701")]
                                    // "c003_a701" => Some((c003::c003_a701::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a701::KERNEL)),

                                    // #[cfg(feature = "c003_a702")]
                                    // "c003_a702" => Some((c003::c003_a702::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a702::KERNEL)),

                                    // #[cfg(feature = "c003_a703")]
                                    // "c003_a703" => Some((c003::c003_a703::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a703::KERNEL)),

                                    // #[cfg(feature = "c003_a704")]
                                    // "c003_a704" => Some((c003::c003_a704::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a704::KERNEL)),

                                    // #[cfg(feature = "c003_a705")]
                                    // "c003_a705" => Some((c003::c003_a705::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a705::KERNEL)),

                                    // #[cfg(feature = "c003_a706")]
                                    // "c003_a706" => Some((c003::c003_a706::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a706::KERNEL)),

                                    // #[cfg(feature = "c003_a707")]
                                    // "c003_a707" => Some((c003::c003_a707::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a707::KERNEL)),

                                    // #[cfg(feature = "c003_a708")]
                                    // "c003_a708" => Some((c003::c003_a708::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a708::KERNEL)),

                                    // #[cfg(feature = "c003_a709")]
                                    // "c003_a709" => Some((c003::c003_a709::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a709::KERNEL)),

                                    // #[cfg(feature = "c003_a710")]
                                    // "c003_a710" => Some((c003::c003_a710::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a710::KERNEL)),

                                    // #[cfg(feature = "c003_a711")]
                                    // "c003_a711" => Some((c003::c003_a711::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a711::KERNEL)),

                                    // #[cfg(feature = "c003_a712")]
                                    // "c003_a712" => Some((c003::c003_a712::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a712::KERNEL)),

                                    // #[cfg(feature = "c003_a713")]
                                    // "c003_a713" => Some((c003::c003_a713::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a713::KERNEL)),

                                    // #[cfg(feature = "c003_a714")]
                                    // "c003_a714" => Some((c003::c003_a714::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a714::KERNEL)),

                                    // #[cfg(feature = "c003_a715")]
                                    // "c003_a715" => Some((c003::c003_a715::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a715::KERNEL)),

                                    // #[cfg(feature = "c003_a716")]
                                    // "c003_a716" => Some((c003::c003_a716::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a716::KERNEL)),

                                    // #[cfg(feature = "c003_a717")]
                                    // "c003_a717" => Some((c003::c003_a717::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a717::KERNEL)),

                                    // #[cfg(feature = "c003_a718")]
                                    // "c003_a718" => Some((c003::c003_a718::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a718::KERNEL)),

                                    // #[cfg(feature = "c003_a719")]
                                    // "c003_a719" => Some((c003::c003_a719::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a719::KERNEL)),

                                    // #[cfg(feature = "c003_a720")]
                                    // "c003_a720" => Some((c003::c003_a720::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a720::KERNEL)),

                                    // #[cfg(feature = "c003_a721")]
                                    // "c003_a721" => Some((c003::c003_a721::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a721::KERNEL)),

                                    // #[cfg(feature = "c003_a722")]
                                    // "c003_a722" => Some((c003::c003_a722::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a722::KERNEL)),

                                    // #[cfg(feature = "c003_a723")]
                                    // "c003_a723" => Some((c003::c003_a723::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a723::KERNEL)),

                                    // #[cfg(feature = "c003_a724")]
                                    // "c003_a724" => Some((c003::c003_a724::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a724::KERNEL)),

                                    // #[cfg(feature = "c003_a725")]
                                    // "c003_a725" => Some((c003::c003_a725::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a725::KERNEL)),

                                    // #[cfg(feature = "c003_a726")]
                                    // "c003_a726" => Some((c003::c003_a726::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a726::KERNEL)),

                                    // #[cfg(feature = "c003_a727")]
                                    // "c003_a727" => Some((c003::c003_a727::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a727::KERNEL)),

                                    // #[cfg(feature = "c003_a728")]
                                    // "c003_a728" => Some((c003::c003_a728::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a728::KERNEL)),

                                    // #[cfg(feature = "c003_a729")]
                                    // "c003_a729" => Some((c003::c003_a729::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a729::KERNEL)),

                                    // #[cfg(feature = "c003_a730")]
                                    // "c003_a730" => Some((c003::c003_a730::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a730::KERNEL)),

                                    // #[cfg(feature = "c003_a731")]
                                    // "c003_a731" => Some((c003::c003_a731::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a731::KERNEL)),

                                    // #[cfg(feature = "c003_a732")]
                                    // "c003_a732" => Some((c003::c003_a732::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a732::KERNEL)),

                                    // #[cfg(feature = "c003_a733")]
                                    // "c003_a733" => Some((c003::c003_a733::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a733::KERNEL)),

                                    // #[cfg(feature = "c003_a734")]
                                    // "c003_a734" => Some((c003::c003_a734::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a734::KERNEL)),

                                    // #[cfg(feature = "c003_a735")]
                                    // "c003_a735" => Some((c003::c003_a735::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a735::KERNEL)),

                                    // #[cfg(feature = "c003_a736")]
                                    // "c003_a736" => Some((c003::c003_a736::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a736::KERNEL)),

                                    // #[cfg(feature = "c003_a737")]
                                    // "c003_a737" => Some((c003::c003_a737::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a737::KERNEL)),

                                    // #[cfg(feature = "c003_a738")]
                                    // "c003_a738" => Some((c003::c003_a738::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a738::KERNEL)),

                                    // #[cfg(feature = "c003_a739")]
                                    // "c003_a739" => Some((c003::c003_a739::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a739::KERNEL)),

                                    // #[cfg(feature = "c003_a740")]
                                    // "c003_a740" => Some((c003::c003_a740::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a740::KERNEL)),

                                    // #[cfg(feature = "c003_a741")]
                                    // "c003_a741" => Some((c003::c003_a741::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a741::KERNEL)),

                                    // #[cfg(feature = "c003_a742")]
                                    // "c003_a742" => Some((c003::c003_a742::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a742::KERNEL)),

                                    // #[cfg(feature = "c003_a743")]
                                    // "c003_a743" => Some((c003::c003_a743::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a743::KERNEL)),

                                    // #[cfg(feature = "c003_a744")]
                                    // "c003_a744" => Some((c003::c003_a744::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a744::KERNEL)),

                                    // #[cfg(feature = "c003_a745")]
                                    // "c003_a745" => Some((c003::c003_a745::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a745::KERNEL)),

                                    // #[cfg(feature = "c003_a746")]
                                    // "c003_a746" => Some((c003::c003_a746::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a746::KERNEL)),

                                    // #[cfg(feature = "c003_a747")]
                                    // "c003_a747" => Some((c003::c003_a747::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a747::KERNEL)),

                                    // #[cfg(feature = "c003_a748")]
                                    // "c003_a748" => Some((c003::c003_a748::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a748::KERNEL)),

                                    // #[cfg(feature = "c003_a749")]
                                    // "c003_a749" => Some((c003::c003_a749::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a749::KERNEL)),

                                    // #[cfg(feature = "c003_a750")]
                                    // "c003_a750" => Some((c003::c003_a750::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a750::KERNEL)),

                                    // #[cfg(feature = "c003_a751")]
                                    // "c003_a751" => Some((c003::c003_a751::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a751::KERNEL)),

                                    // #[cfg(feature = "c003_a752")]
                                    // "c003_a752" => Some((c003::c003_a752::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a752::KERNEL)),

                                    // #[cfg(feature = "c003_a753")]
                                    // "c003_a753" => Some((c003::c003_a753::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a753::KERNEL)),

                                    // #[cfg(feature = "c003_a754")]
                                    // "c003_a754" => Some((c003::c003_a754::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a754::KERNEL)),

                                    // #[cfg(feature = "c003_a755")]
                                    // "c003_a755" => Some((c003::c003_a755::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a755::KERNEL)),

                                    // #[cfg(feature = "c003_a756")]
                                    // "c003_a756" => Some((c003::c003_a756::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a756::KERNEL)),

                                    // #[cfg(feature = "c003_a757")]
                                    // "c003_a757" => Some((c003::c003_a757::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a757::KERNEL)),

                                    // #[cfg(feature = "c003_a758")]
                                    // "c003_a758" => Some((c003::c003_a758::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a758::KERNEL)),

                                    // #[cfg(feature = "c003_a759")]
                                    // "c003_a759" => Some((c003::c003_a759::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a759::KERNEL)),

                                    // #[cfg(feature = "c003_a760")]
                                    // "c003_a760" => Some((c003::c003_a760::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a760::KERNEL)),

                                    // #[cfg(feature = "c003_a761")]
                                    // "c003_a761" => Some((c003::c003_a761::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a761::KERNEL)),

                                    // #[cfg(feature = "c003_a762")]
                                    // "c003_a762" => Some((c003::c003_a762::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a762::KERNEL)),

                                    // #[cfg(feature = "c003_a763")]
                                    // "c003_a763" => Some((c003::c003_a763::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a763::KERNEL)),

                                    // #[cfg(feature = "c003_a764")]
                                    // "c003_a764" => Some((c003::c003_a764::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a764::KERNEL)),

                                    // #[cfg(feature = "c003_a765")]
                                    // "c003_a765" => Some((c003::c003_a765::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a765::KERNEL)),

                                    // #[cfg(feature = "c003_a766")]
                                    // "c003_a766" => Some((c003::c003_a766::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a766::KERNEL)),

                                    // #[cfg(feature = "c003_a767")]
                                    // "c003_a767" => Some((c003::c003_a767::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a767::KERNEL)),

                                    // #[cfg(feature = "c003_a768")]
                                    // "c003_a768" => Some((c003::c003_a768::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a768::KERNEL)),

                                    // #[cfg(feature = "c003_a769")]
                                    // "c003_a769" => Some((c003::c003_a769::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a769::KERNEL)),

                                    // #[cfg(feature = "c003_a770")]
                                    // "c003_a770" => Some((c003::c003_a770::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a770::KERNEL)),

                                    // #[cfg(feature = "c003_a771")]
                                    // "c003_a771" => Some((c003::c003_a771::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a771::KERNEL)),

                                    // #[cfg(feature = "c003_a772")]
                                    // "c003_a772" => Some((c003::c003_a772::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a772::KERNEL)),

                                    // #[cfg(feature = "c003_a773")]
                                    // "c003_a773" => Some((c003::c003_a773::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a773::KERNEL)),

                                    // #[cfg(feature = "c003_a774")]
                                    // "c003_a774" => Some((c003::c003_a774::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a774::KERNEL)),

                                    // #[cfg(feature = "c003_a775")]
                                    // "c003_a775" => Some((c003::c003_a775::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a775::KERNEL)),

                                    // #[cfg(feature = "c003_a776")]
                                    // "c003_a776" => Some((c003::c003_a776::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a776::KERNEL)),

                                    // #[cfg(feature = "c003_a777")]
                                    // "c003_a777" => Some((c003::c003_a777::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a777::KERNEL)),

                                    // #[cfg(feature = "c003_a778")]
                                    // "c003_a778" => Some((c003::c003_a778::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a778::KERNEL)),

                                    // #[cfg(feature = "c003_a779")]
                                    // "c003_a779" => Some((c003::c003_a779::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a779::KERNEL)),

                                    // #[cfg(feature = "c003_a780")]
                                    // "c003_a780" => Some((c003::c003_a780::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a780::KERNEL)),

                                    // #[cfg(feature = "c003_a781")]
                                    // "c003_a781" => Some((c003::c003_a781::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a781::KERNEL)),

                                    // #[cfg(feature = "c003_a782")]
                                    // "c003_a782" => Some((c003::c003_a782::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a782::KERNEL)),

                                    // #[cfg(feature = "c003_a783")]
                                    // "c003_a783" => Some((c003::c003_a783::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a783::KERNEL)),

                                    // #[cfg(feature = "c003_a784")]
                                    // "c003_a784" => Some((c003::c003_a784::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a784::KERNEL)),

                                    // #[cfg(feature = "c003_a785")]
                                    // "c003_a785" => Some((c003::c003_a785::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a785::KERNEL)),

                                    // #[cfg(feature = "c003_a786")]
                                    // "c003_a786" => Some((c003::c003_a786::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a786::KERNEL)),

                                    // #[cfg(feature = "c003_a787")]
                                    // "c003_a787" => Some((c003::c003_a787::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a787::KERNEL)),

                                    // #[cfg(feature = "c003_a788")]
                                    // "c003_a788" => Some((c003::c003_a788::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a788::KERNEL)),

                                    // #[cfg(feature = "c003_a789")]
                                    // "c003_a789" => Some((c003::c003_a789::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a789::KERNEL)),

                                    // #[cfg(feature = "c003_a790")]
                                    // "c003_a790" => Some((c003::c003_a790::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a790::KERNEL)),

                                    // #[cfg(feature = "c003_a791")]
                                    // "c003_a791" => Some((c003::c003_a791::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a791::KERNEL)),

                                    // #[cfg(feature = "c003_a792")]
                                    // "c003_a792" => Some((c003::c003_a792::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a792::KERNEL)),

                                    // #[cfg(feature = "c003_a793")]
                                    // "c003_a793" => Some((c003::c003_a793::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a793::KERNEL)),

                                    // #[cfg(feature = "c003_a794")]
                                    // "c003_a794" => Some((c003::c003_a794::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a794::KERNEL)),

                                    // #[cfg(feature = "c003_a795")]
                                    // "c003_a795" => Some((c003::c003_a795::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a795::KERNEL)),

                                    // #[cfg(feature = "c003_a796")]
                                    // "c003_a796" => Some((c003::c003_a796::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a796::KERNEL)),

                                    // #[cfg(feature = "c003_a797")]
                                    // "c003_a797" => Some((c003::c003_a797::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a797::KERNEL)),

                                    // #[cfg(feature = "c003_a798")]
                                    // "c003_a798" => Some((c003::c003_a798::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a798::KERNEL)),

                                    // #[cfg(feature = "c003_a799")]
                                    // "c003_a799" => Some((c003::c003_a799::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a799::KERNEL)),

                                    // #[cfg(feature = "c003_a800")]
                                    // "c003_a800" => Some((c003::c003_a800::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a800::KERNEL)),

                                    // #[cfg(feature = "c003_a801")]
                                    // "c003_a801" => Some((c003::c003_a801::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a801::KERNEL)),

                                    // #[cfg(feature = "c003_a802")]
                                    // "c003_a802" => Some((c003::c003_a802::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a802::KERNEL)),

                                    // #[cfg(feature = "c003_a803")]
                                    // "c003_a803" => Some((c003::c003_a803::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a803::KERNEL)),

                                    // #[cfg(feature = "c003_a804")]
                                    // "c003_a804" => Some((c003::c003_a804::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a804::KERNEL)),

                                    // #[cfg(feature = "c003_a805")]
                                    // "c003_a805" => Some((c003::c003_a805::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a805::KERNEL)),

                                    // #[cfg(feature = "c003_a806")]
                                    // "c003_a806" => Some((c003::c003_a806::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a806::KERNEL)),

                                    // #[cfg(feature = "c003_a807")]
                                    // "c003_a807" => Some((c003::c003_a807::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a807::KERNEL)),

                                    // #[cfg(feature = "c003_a808")]
                                    // "c003_a808" => Some((c003::c003_a808::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a808::KERNEL)),

                                    // #[cfg(feature = "c003_a809")]
                                    // "c003_a809" => Some((c003::c003_a809::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a809::KERNEL)),

                                    // #[cfg(feature = "c003_a810")]
                                    // "c003_a810" => Some((c003::c003_a810::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a810::KERNEL)),

                                    // #[cfg(feature = "c003_a811")]
                                    // "c003_a811" => Some((c003::c003_a811::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a811::KERNEL)),

                                    // #[cfg(feature = "c003_a812")]
                                    // "c003_a812" => Some((c003::c003_a812::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a812::KERNEL)),

                                    // #[cfg(feature = "c003_a813")]
                                    // "c003_a813" => Some((c003::c003_a813::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a813::KERNEL)),

                                    // #[cfg(feature = "c003_a814")]
                                    // "c003_a814" => Some((c003::c003_a814::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a814::KERNEL)),

                                    // #[cfg(feature = "c003_a815")]
                                    // "c003_a815" => Some((c003::c003_a815::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a815::KERNEL)),

                                    // #[cfg(feature = "c003_a816")]
                                    // "c003_a816" => Some((c003::c003_a816::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a816::KERNEL)),

                                    // #[cfg(feature = "c003_a817")]
                                    // "c003_a817" => Some((c003::c003_a817::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a817::KERNEL)),

                                    // #[cfg(feature = "c003_a818")]
                                    // "c003_a818" => Some((c003::c003_a818::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a818::KERNEL)),

                                    // #[cfg(feature = "c003_a819")]
                                    // "c003_a819" => Some((c003::c003_a819::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a819::KERNEL)),

                                    // #[cfg(feature = "c003_a820")]
                                    // "c003_a820" => Some((c003::c003_a820::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a820::KERNEL)),

                                    // #[cfg(feature = "c003_a821")]
                                    // "c003_a821" => Some((c003::c003_a821::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a821::KERNEL)),

                                    // #[cfg(feature = "c003_a822")]
                                    // "c003_a822" => Some((c003::c003_a822::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a822::KERNEL)),

                                    // #[cfg(feature = "c003_a823")]
                                    // "c003_a823" => Some((c003::c003_a823::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a823::KERNEL)),

                                    // #[cfg(feature = "c003_a824")]
                                    // "c003_a824" => Some((c003::c003_a824::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a824::KERNEL)),

                                    // #[cfg(feature = "c003_a825")]
                                    // "c003_a825" => Some((c003::c003_a825::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a825::KERNEL)),

                                    // #[cfg(feature = "c003_a826")]
                                    // "c003_a826" => Some((c003::c003_a826::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a826::KERNEL)),

                                    // #[cfg(feature = "c003_a827")]
                                    // "c003_a827" => Some((c003::c003_a827::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a827::KERNEL)),

                                    // #[cfg(feature = "c003_a828")]
                                    // "c003_a828" => Some((c003::c003_a828::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a828::KERNEL)),

                                    // #[cfg(feature = "c003_a829")]
                                    // "c003_a829" => Some((c003::c003_a829::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a829::KERNEL)),

                                    // #[cfg(feature = "c003_a830")]
                                    // "c003_a830" => Some((c003::c003_a830::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a830::KERNEL)),

                                    // #[cfg(feature = "c003_a831")]
                                    // "c003_a831" => Some((c003::c003_a831::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a831::KERNEL)),

                                    // #[cfg(feature = "c003_a832")]
                                    // "c003_a832" => Some((c003::c003_a832::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a832::KERNEL)),

                                    // #[cfg(feature = "c003_a833")]
                                    // "c003_a833" => Some((c003::c003_a833::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a833::KERNEL)),

                                    // #[cfg(feature = "c003_a834")]
                                    // "c003_a834" => Some((c003::c003_a834::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a834::KERNEL)),

                                    // #[cfg(feature = "c003_a835")]
                                    // "c003_a835" => Some((c003::c003_a835::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a835::KERNEL)),

                                    // #[cfg(feature = "c003_a836")]
                                    // "c003_a836" => Some((c003::c003_a836::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a836::KERNEL)),

                                    // #[cfg(feature = "c003_a837")]
                                    // "c003_a837" => Some((c003::c003_a837::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a837::KERNEL)),

                                    // #[cfg(feature = "c003_a838")]
                                    // "c003_a838" => Some((c003::c003_a838::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a838::KERNEL)),

                                    // #[cfg(feature = "c003_a839")]
                                    // "c003_a839" => Some((c003::c003_a839::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a839::KERNEL)),

                                    // #[cfg(feature = "c003_a840")]
                                    // "c003_a840" => Some((c003::c003_a840::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a840::KERNEL)),

                                    // #[cfg(feature = "c003_a841")]
                                    // "c003_a841" => Some((c003::c003_a841::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a841::KERNEL)),

                                    // #[cfg(feature = "c003_a842")]
                                    // "c003_a842" => Some((c003::c003_a842::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a842::KERNEL)),

                                    // #[cfg(feature = "c003_a843")]
                                    // "c003_a843" => Some((c003::c003_a843::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a843::KERNEL)),

                                    // #[cfg(feature = "c003_a844")]
                                    // "c003_a844" => Some((c003::c003_a844::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a844::KERNEL)),

                                    // #[cfg(feature = "c003_a845")]
                                    // "c003_a845" => Some((c003::c003_a845::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a845::KERNEL)),

                                    // #[cfg(feature = "c003_a846")]
                                    // "c003_a846" => Some((c003::c003_a846::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a846::KERNEL)),

                                    // #[cfg(feature = "c003_a847")]
                                    // "c003_a847" => Some((c003::c003_a847::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a847::KERNEL)),

                                    // #[cfg(feature = "c003_a848")]
                                    // "c003_a848" => Some((c003::c003_a848::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a848::KERNEL)),

                                    // #[cfg(feature = "c003_a849")]
                                    // "c003_a849" => Some((c003::c003_a849::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a849::KERNEL)),

                                    // #[cfg(feature = "c003_a850")]
                                    // "c003_a850" => Some((c003::c003_a850::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a850::KERNEL)),

                                    // #[cfg(feature = "c003_a851")]
                                    // "c003_a851" => Some((c003::c003_a851::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a851::KERNEL)),

                                    // #[cfg(feature = "c003_a852")]
                                    // "c003_a852" => Some((c003::c003_a852::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a852::KERNEL)),

                                    // #[cfg(feature = "c003_a853")]
                                    // "c003_a853" => Some((c003::c003_a853::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a853::KERNEL)),

                                    // #[cfg(feature = "c003_a854")]
                                    // "c003_a854" => Some((c003::c003_a854::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a854::KERNEL)),

                                    // #[cfg(feature = "c003_a855")]
                                    // "c003_a855" => Some((c003::c003_a855::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a855::KERNEL)),

                                    // #[cfg(feature = "c003_a856")]
                                    // "c003_a856" => Some((c003::c003_a856::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a856::KERNEL)),

                                    // #[cfg(feature = "c003_a857")]
                                    // "c003_a857" => Some((c003::c003_a857::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a857::KERNEL)),

                                    // #[cfg(feature = "c003_a858")]
                                    // "c003_a858" => Some((c003::c003_a858::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a858::KERNEL)),

                                    // #[cfg(feature = "c003_a859")]
                                    // "c003_a859" => Some((c003::c003_a859::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a859::KERNEL)),

                                    // #[cfg(feature = "c003_a860")]
                                    // "c003_a860" => Some((c003::c003_a860::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a860::KERNEL)),

                                    // #[cfg(feature = "c003_a861")]
                                    // "c003_a861" => Some((c003::c003_a861::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a861::KERNEL)),

                                    // #[cfg(feature = "c003_a862")]
                                    // "c003_a862" => Some((c003::c003_a862::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a862::KERNEL)),

                                    // #[cfg(feature = "c003_a863")]
                                    // "c003_a863" => Some((c003::c003_a863::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a863::KERNEL)),

                                    // #[cfg(feature = "c003_a864")]
                                    // "c003_a864" => Some((c003::c003_a864::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a864::KERNEL)),

                                    // #[cfg(feature = "c003_a865")]
                                    // "c003_a865" => Some((c003::c003_a865::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a865::KERNEL)),

                                    // #[cfg(feature = "c003_a866")]
                                    // "c003_a866" => Some((c003::c003_a866::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a866::KERNEL)),

                                    // #[cfg(feature = "c003_a867")]
                                    // "c003_a867" => Some((c003::c003_a867::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a867::KERNEL)),

                                    // #[cfg(feature = "c003_a868")]
                                    // "c003_a868" => Some((c003::c003_a868::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a868::KERNEL)),

                                    // #[cfg(feature = "c003_a869")]
                                    // "c003_a869" => Some((c003::c003_a869::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a869::KERNEL)),

                                    // #[cfg(feature = "c003_a870")]
                                    // "c003_a870" => Some((c003::c003_a870::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a870::KERNEL)),

                                    // #[cfg(feature = "c003_a871")]
                                    // "c003_a871" => Some((c003::c003_a871::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a871::KERNEL)),

                                    // #[cfg(feature = "c003_a872")]
                                    // "c003_a872" => Some((c003::c003_a872::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a872::KERNEL)),

                                    // #[cfg(feature = "c003_a873")]
                                    // "c003_a873" => Some((c003::c003_a873::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a873::KERNEL)),

                                    // #[cfg(feature = "c003_a874")]
                                    // "c003_a874" => Some((c003::c003_a874::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a874::KERNEL)),

                                    // #[cfg(feature = "c003_a875")]
                                    // "c003_a875" => Some((c003::c003_a875::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a875::KERNEL)),

                                    // #[cfg(feature = "c003_a876")]
                                    // "c003_a876" => Some((c003::c003_a876::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a876::KERNEL)),

                                    // #[cfg(feature = "c003_a877")]
                                    // "c003_a877" => Some((c003::c003_a877::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a877::KERNEL)),

                                    // #[cfg(feature = "c003_a878")]
                                    // "c003_a878" => Some((c003::c003_a878::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a878::KERNEL)),

                                    // #[cfg(feature = "c003_a879")]
                                    // "c003_a879" => Some((c003::c003_a879::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a879::KERNEL)),

                                    // #[cfg(feature = "c003_a880")]
                                    // "c003_a880" => Some((c003::c003_a880::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a880::KERNEL)),

                                    // #[cfg(feature = "c003_a881")]
                                    // "c003_a881" => Some((c003::c003_a881::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a881::KERNEL)),

                                    // #[cfg(feature = "c003_a882")]
                                    // "c003_a882" => Some((c003::c003_a882::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a882::KERNEL)),

                                    // #[cfg(feature = "c003_a883")]
                                    // "c003_a883" => Some((c003::c003_a883::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a883::KERNEL)),

                                    // #[cfg(feature = "c003_a884")]
                                    // "c003_a884" => Some((c003::c003_a884::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a884::KERNEL)),

                                    // #[cfg(feature = "c003_a885")]
                                    // "c003_a885" => Some((c003::c003_a885::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a885::KERNEL)),

                                    // #[cfg(feature = "c003_a886")]
                                    // "c003_a886" => Some((c003::c003_a886::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a886::KERNEL)),

                                    // #[cfg(feature = "c003_a887")]
                                    // "c003_a887" => Some((c003::c003_a887::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a887::KERNEL)),

                                    // #[cfg(feature = "c003_a888")]
                                    // "c003_a888" => Some((c003::c003_a888::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a888::KERNEL)),

                                    // #[cfg(feature = "c003_a889")]
                                    // "c003_a889" => Some((c003::c003_a889::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a889::KERNEL)),

                                    // #[cfg(feature = "c003_a890")]
                                    // "c003_a890" => Some((c003::c003_a890::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a890::KERNEL)),

                                    // #[cfg(feature = "c003_a891")]
                                    // "c003_a891" => Some((c003::c003_a891::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a891::KERNEL)),

                                    // #[cfg(feature = "c003_a892")]
                                    // "c003_a892" => Some((c003::c003_a892::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a892::KERNEL)),

                                    // #[cfg(feature = "c003_a893")]
                                    // "c003_a893" => Some((c003::c003_a893::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a893::KERNEL)),

                                    // #[cfg(feature = "c003_a894")]
                                    // "c003_a894" => Some((c003::c003_a894::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a894::KERNEL)),

                                    // #[cfg(feature = "c003_a895")]
                                    // "c003_a895" => Some((c003::c003_a895::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a895::KERNEL)),

                                    // #[cfg(feature = "c003_a896")]
                                    // "c003_a896" => Some((c003::c003_a896::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a896::KERNEL)),

                                    // #[cfg(feature = "c003_a897")]
                                    // "c003_a897" => Some((c003::c003_a897::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a897::KERNEL)),

                                    // #[cfg(feature = "c003_a898")]
                                    // "c003_a898" => Some((c003::c003_a898::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a898::KERNEL)),

                                    // #[cfg(feature = "c003_a899")]
                                    // "c003_a899" => Some((c003::c003_a899::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a899::KERNEL)),

                                    // #[cfg(feature = "c003_a900")]
                                    // "c003_a900" => Some((c003::c003_a900::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a900::KERNEL)),

                                    // #[cfg(feature = "c003_a901")]
                                    // "c003_a901" => Some((c003::c003_a901::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a901::KERNEL)),

                                    // #[cfg(feature = "c003_a902")]
                                    // "c003_a902" => Some((c003::c003_a902::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a902::KERNEL)),

                                    // #[cfg(feature = "c003_a903")]
                                    // "c003_a903" => Some((c003::c003_a903::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a903::KERNEL)),

                                    // #[cfg(feature = "c003_a904")]
                                    // "c003_a904" => Some((c003::c003_a904::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a904::KERNEL)),

                                    // #[cfg(feature = "c003_a905")]
                                    // "c003_a905" => Some((c003::c003_a905::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a905::KERNEL)),

                                    // #[cfg(feature = "c003_a906")]
                                    // "c003_a906" => Some((c003::c003_a906::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a906::KERNEL)),

                                    // #[cfg(feature = "c003_a907")]
                                    // "c003_a907" => Some((c003::c003_a907::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a907::KERNEL)),

                                    // #[cfg(feature = "c003_a908")]
                                    // "c003_a908" => Some((c003::c003_a908::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a908::KERNEL)),

                                    // #[cfg(feature = "c003_a909")]
                                    // "c003_a909" => Some((c003::c003_a909::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a909::KERNEL)),

                                    // #[cfg(feature = "c003_a910")]
                                    // "c003_a910" => Some((c003::c003_a910::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a910::KERNEL)),

                                    // #[cfg(feature = "c003_a911")]
                                    // "c003_a911" => Some((c003::c003_a911::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a911::KERNEL)),

                                    // #[cfg(feature = "c003_a912")]
                                    // "c003_a912" => Some((c003::c003_a912::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a912::KERNEL)),

                                    // #[cfg(feature = "c003_a913")]
                                    // "c003_a913" => Some((c003::c003_a913::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a913::KERNEL)),

                                    // #[cfg(feature = "c003_a914")]
                                    // "c003_a914" => Some((c003::c003_a914::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a914::KERNEL)),

                                    // #[cfg(feature = "c003_a915")]
                                    // "c003_a915" => Some((c003::c003_a915::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a915::KERNEL)),

                                    // #[cfg(feature = "c003_a916")]
                                    // "c003_a916" => Some((c003::c003_a916::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a916::KERNEL)),

                                    // #[cfg(feature = "c003_a917")]
                                    // "c003_a917" => Some((c003::c003_a917::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a917::KERNEL)),

                                    // #[cfg(feature = "c003_a918")]
                                    // "c003_a918" => Some((c003::c003_a918::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a918::KERNEL)),

                                    // #[cfg(feature = "c003_a919")]
                                    // "c003_a919" => Some((c003::c003_a919::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a919::KERNEL)),

                                    // #[cfg(feature = "c003_a920")]
                                    // "c003_a920" => Some((c003::c003_a920::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a920::KERNEL)),

                                    // #[cfg(feature = "c003_a921")]
                                    // "c003_a921" => Some((c003::c003_a921::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a921::KERNEL)),

                                    // #[cfg(feature = "c003_a922")]
                                    // "c003_a922" => Some((c003::c003_a922::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a922::KERNEL)),

                                    // #[cfg(feature = "c003_a923")]
                                    // "c003_a923" => Some((c003::c003_a923::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a923::KERNEL)),

                                    // #[cfg(feature = "c003_a924")]
                                    // "c003_a924" => Some((c003::c003_a924::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a924::KERNEL)),

                                    // #[cfg(feature = "c003_a925")]
                                    // "c003_a925" => Some((c003::c003_a925::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a925::KERNEL)),

                                    // #[cfg(feature = "c003_a926")]
                                    // "c003_a926" => Some((c003::c003_a926::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a926::KERNEL)),

                                    // #[cfg(feature = "c003_a927")]
                                    // "c003_a927" => Some((c003::c003_a927::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a927::KERNEL)),

                                    // #[cfg(feature = "c003_a928")]
                                    // "c003_a928" => Some((c003::c003_a928::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a928::KERNEL)),

                                    // #[cfg(feature = "c003_a929")]
                                    // "c003_a929" => Some((c003::c003_a929::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a929::KERNEL)),

                                    // #[cfg(feature = "c003_a930")]
                                    // "c003_a930" => Some((c003::c003_a930::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a930::KERNEL)),

                                    // #[cfg(feature = "c003_a931")]
                                    // "c003_a931" => Some((c003::c003_a931::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a931::KERNEL)),

                                    // #[cfg(feature = "c003_a932")]
                                    // "c003_a932" => Some((c003::c003_a932::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a932::KERNEL)),

                                    // #[cfg(feature = "c003_a933")]
                                    // "c003_a933" => Some((c003::c003_a933::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a933::KERNEL)),

                                    // #[cfg(feature = "c003_a934")]
                                    // "c003_a934" => Some((c003::c003_a934::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a934::KERNEL)),

                                    // #[cfg(feature = "c003_a935")]
                                    // "c003_a935" => Some((c003::c003_a935::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a935::KERNEL)),

                                    // #[cfg(feature = "c003_a936")]
                                    // "c003_a936" => Some((c003::c003_a936::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a936::KERNEL)),

                                    // #[cfg(feature = "c003_a937")]
                                    // "c003_a937" => Some((c003::c003_a937::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a937::KERNEL)),

                                    // #[cfg(feature = "c003_a938")]
                                    // "c003_a938" => Some((c003::c003_a938::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a938::KERNEL)),

                                    // #[cfg(feature = "c003_a939")]
                                    // "c003_a939" => Some((c003::c003_a939::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a939::KERNEL)),

                                    // #[cfg(feature = "c003_a940")]
                                    // "c003_a940" => Some((c003::c003_a940::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a940::KERNEL)),

                                    // #[cfg(feature = "c003_a941")]
                                    // "c003_a941" => Some((c003::c003_a941::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a941::KERNEL)),

                                    // #[cfg(feature = "c003_a942")]
                                    // "c003_a942" => Some((c003::c003_a942::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a942::KERNEL)),

                                    // #[cfg(feature = "c003_a943")]
                                    // "c003_a943" => Some((c003::c003_a943::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a943::KERNEL)),

                                    // #[cfg(feature = "c003_a944")]
                                    // "c003_a944" => Some((c003::c003_a944::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a944::KERNEL)),

                                    // #[cfg(feature = "c003_a945")]
                                    // "c003_a945" => Some((c003::c003_a945::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a945::KERNEL)),

                                    // #[cfg(feature = "c003_a946")]
                                    // "c003_a946" => Some((c003::c003_a946::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a946::KERNEL)),

                                    // #[cfg(feature = "c003_a947")]
                                    // "c003_a947" => Some((c003::c003_a947::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a947::KERNEL)),

                                    // #[cfg(feature = "c003_a948")]
                                    // "c003_a948" => Some((c003::c003_a948::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a948::KERNEL)),

                                    // #[cfg(feature = "c003_a949")]
                                    // "c003_a949" => Some((c003::c003_a949::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a949::KERNEL)),

                                    // #[cfg(feature = "c003_a950")]
                                    // "c003_a950" => Some((c003::c003_a950::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a950::KERNEL)),

                                    // #[cfg(feature = "c003_a951")]
                                    // "c003_a951" => Some((c003::c003_a951::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a951::KERNEL)),

                                    // #[cfg(feature = "c003_a952")]
                                    // "c003_a952" => Some((c003::c003_a952::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a952::KERNEL)),

                                    // #[cfg(feature = "c003_a953")]
                                    // "c003_a953" => Some((c003::c003_a953::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a953::KERNEL)),

                                    // #[cfg(feature = "c003_a954")]
                                    // "c003_a954" => Some((c003::c003_a954::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a954::KERNEL)),

                                    // #[cfg(feature = "c003_a955")]
                                    // "c003_a955" => Some((c003::c003_a955::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a955::KERNEL)),

                                    // #[cfg(feature = "c003_a956")]
                                    // "c003_a956" => Some((c003::c003_a956::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a956::KERNEL)),

                                    // #[cfg(feature = "c003_a957")]
                                    // "c003_a957" => Some((c003::c003_a957::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a957::KERNEL)),

                                    // #[cfg(feature = "c003_a958")]
                                    // "c003_a958" => Some((c003::c003_a958::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a958::KERNEL)),

                                    // #[cfg(feature = "c003_a959")]
                                    // "c003_a959" => Some((c003::c003_a959::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a959::KERNEL)),

                                    // #[cfg(feature = "c003_a960")]
                                    // "c003_a960" => Some((c003::c003_a960::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a960::KERNEL)),

                                    // #[cfg(feature = "c003_a961")]
                                    // "c003_a961" => Some((c003::c003_a961::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a961::KERNEL)),

                                    // #[cfg(feature = "c003_a962")]
                                    // "c003_a962" => Some((c003::c003_a962::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a962::KERNEL)),

                                    // #[cfg(feature = "c003_a963")]
                                    // "c003_a963" => Some((c003::c003_a963::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a963::KERNEL)),

                                    // #[cfg(feature = "c003_a964")]
                                    // "c003_a964" => Some((c003::c003_a964::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a964::KERNEL)),

                                    // #[cfg(feature = "c003_a965")]
                                    // "c003_a965" => Some((c003::c003_a965::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a965::KERNEL)),

                                    // #[cfg(feature = "c003_a966")]
                                    // "c003_a966" => Some((c003::c003_a966::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a966::KERNEL)),

                                    // #[cfg(feature = "c003_a967")]
                                    // "c003_a967" => Some((c003::c003_a967::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a967::KERNEL)),

                                    // #[cfg(feature = "c003_a968")]
                                    // "c003_a968" => Some((c003::c003_a968::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a968::KERNEL)),

                                    // #[cfg(feature = "c003_a969")]
                                    // "c003_a969" => Some((c003::c003_a969::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a969::KERNEL)),

                                    // #[cfg(feature = "c003_a970")]
                                    // "c003_a970" => Some((c003::c003_a970::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a970::KERNEL)),

                                    // #[cfg(feature = "c003_a971")]
                                    // "c003_a971" => Some((c003::c003_a971::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a971::KERNEL)),

                                    // #[cfg(feature = "c003_a972")]
                                    // "c003_a972" => Some((c003::c003_a972::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a972::KERNEL)),

                                    // #[cfg(feature = "c003_a973")]
                                    // "c003_a973" => Some((c003::c003_a973::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a973::KERNEL)),

                                    // #[cfg(feature = "c003_a974")]
                                    // "c003_a974" => Some((c003::c003_a974::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a974::KERNEL)),

                                    // #[cfg(feature = "c003_a975")]
                                    // "c003_a975" => Some((c003::c003_a975::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a975::KERNEL)),

                                    // #[cfg(feature = "c003_a976")]
                                    // "c003_a976" => Some((c003::c003_a976::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a976::KERNEL)),

                                    // #[cfg(feature = "c003_a977")]
                                    // "c003_a977" => Some((c003::c003_a977::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a977::KERNEL)),

                                    // #[cfg(feature = "c003_a978")]
                                    // "c003_a978" => Some((c003::c003_a978::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a978::KERNEL)),

                                    // #[cfg(feature = "c003_a979")]
                                    // "c003_a979" => Some((c003::c003_a979::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a979::KERNEL)),

                                    // #[cfg(feature = "c003_a980")]
                                    // "c003_a980" => Some((c003::c003_a980::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a980::KERNEL)),

                                    // #[cfg(feature = "c003_a981")]
                                    // "c003_a981" => Some((c003::c003_a981::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a981::KERNEL)),

                                    // #[cfg(feature = "c003_a982")]
                                    // "c003_a982" => Some((c003::c003_a982::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a982::KERNEL)),

                                    // #[cfg(feature = "c003_a983")]
                                    // "c003_a983" => Some((c003::c003_a983::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a983::KERNEL)),

                                    // #[cfg(feature = "c003_a984")]
                                    // "c003_a984" => Some((c003::c003_a984::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a984::KERNEL)),

                                    // #[cfg(feature = "c003_a985")]
                                    // "c003_a985" => Some((c003::c003_a985::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a985::KERNEL)),

                                    // #[cfg(feature = "c003_a986")]
                                    // "c003_a986" => Some((c003::c003_a986::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a986::KERNEL)),

                                    // #[cfg(feature = "c003_a987")]
                                    // "c003_a987" => Some((c003::c003_a987::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a987::KERNEL)),

                                    // #[cfg(feature = "c003_a988")]
                                    // "c003_a988" => Some((c003::c003_a988::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a988::KERNEL)),

                                    // #[cfg(feature = "c003_a989")]
                                    // "c003_a989" => Some((c003::c003_a989::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a989::KERNEL)),

                                    // #[cfg(feature = "c003_a990")]
                                    // "c003_a990" => Some((c003::c003_a990::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a990::KERNEL)),

                                    // #[cfg(feature = "c003_a991")]
                                    // "c003_a991" => Some((c003::c003_a991::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a991::KERNEL)),

                                    // #[cfg(feature = "c003_a992")]
                                    // "c003_a992" => Some((c003::c003_a992::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a992::KERNEL)),

                                    // #[cfg(feature = "c003_a993")]
                                    // "c003_a993" => Some((c003::c003_a993::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a993::KERNEL)),

                                    // #[cfg(feature = "c003_a994")]
                                    // "c003_a994" => Some((c003::c003_a994::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a994::KERNEL)),

                                    // #[cfg(feature = "c003_a995")]
                                    // "c003_a995" => Some((c003::c003_a995::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a995::KERNEL)),

                                    // #[cfg(feature = "c003_a996")]
                                    // "c003_a996" => Some((c003::c003_a996::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a996::KERNEL)),

                                    // #[cfg(feature = "c003_a997")]
                                    // "c003_a997" => Some((c003::c003_a997::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a997::KERNEL)),

                                    // #[cfg(feature = "c003_a998")]
                                    // "c003_a998" => Some((c003::c003_a998::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a998::KERNEL)),

                                    // #[cfg(feature = "c003_a999")]
                                    // "c003_a999" => Some((c003::c003_a999::cuda_solve_challenge as CudaSolveChallengeFn, &c003::c003_a999::KERNEL)),
                                    _ => {
                                        Option::<(CudaSolveChallengeFn, &Option<CudaKernel>)>::None
                                    }
                                };

                                match filter_step {
                                    Some((cuda_solve_challenge, algo_kernel)) => {
                                        let challenge_cuda_funcs = challenge_cuda_funcs
                                            .get_or_insert(
                                                get_or_compile_cuda(
                                                    &job.settings.challenge_id,
                                                    &tig_challenges::c003::KERNEL,
                                                    &dev,
                                                )
                                                .await,
                                            )
                                            .clone();
                                        let algorithm_cuda_funcs = algorithm_cuda_funcs
                                            .get_or_insert(
                                                get_or_compile_cuda(
                                                    &job.settings.algorithm_id,
                                                    algo_kernel,
                                                    &dev,
                                                )
                                                .await,
                                            )
                                            .clone();
                                        let challenge =
                                            tig_challenges::c003::Challenge::cuda_generate_instance_from_vec(
                                                seeds,
                                                &job.settings.difficulty,
                                                &dev,
                                                challenge_cuda_funcs,
                                            )
                                            .unwrap();
                                        match cuda_solve_challenge(
                                            &challenge,
                                            &dev,
                                            algorithm_cuda_funcs,
                                        ) {
                                            Ok(Some(solution)) => {
                                                challenge.verify_solution(&solution).is_err()
                                            }
                                            _ => true,
                                        }
                                    }
                                    None => false,
                                }
                            }
                            "c004" => {
                                type CudaSolveChallengeFn =
                                    fn(
                                        &tig_challenges::c004::Challenge,
                                        &Arc<CudaDevice>,
                                        HashMap<&'static str, CudaFunction>,
                                    )
                                        -> anyhow::Result<Option<tig_challenges::c004::Solution>>;

                                let filter_step = match job.settings.algorithm_id.as_str() {
                                    // #[cfg(feature = "c004_a001")]
                                    // "c004_a001" => Some((c004::c004_a001::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a001::KERNEL)),

                                    // #[cfg(feature = "c004_a002")]
                                    // "c004_a002" => Some((c004::c004_a002::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a002::KERNEL)),

                                    // #[cfg(feature = "c004_a003")]
                                    // "c004_a003" => Some((c004::c004_a003::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a003::KERNEL)),

                                    // #[cfg(feature = "c004_a004")]
                                    // "c004_a004" => Some((c004::c004_a004::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a004::KERNEL)),

                                    // #[cfg(feature = "c004_a005")]
                                    // "c004_a005" => Some((c004::c004_a005::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a005::KERNEL)),

                                    // #[cfg(feature = "c004_a006")]
                                    // "c004_a006" => Some((c004::c004_a006::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a006::KERNEL)),

                                    // #[cfg(feature = "c004_a007")]
                                    // "c004_a007" => Some((c004::c004_a007::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a007::KERNEL)),

                                    // #[cfg(feature = "c004_a008")]
                                    // "c004_a008" => Some((c004::c004_a008::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a008::KERNEL)),

                                    // #[cfg(feature = "c004_a009")]
                                    // "c004_a009" => Some((c004::c004_a009::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a009::KERNEL)),

                                    // #[cfg(feature = "c004_a010")]
                                    // "c004_a010" => Some((c004::c004_a010::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a010::KERNEL)),

                                    // #[cfg(feature = "c004_a011")]
                                    // "c004_a011" => Some((c004::c004_a011::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a011::KERNEL)),

                                    // #[cfg(feature = "c004_a012")]
                                    // "c004_a012" => Some((c004::c004_a012::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a012::KERNEL)),

                                    // #[cfg(feature = "c004_a013")]
                                    // "c004_a013" => Some((c004::c004_a013::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a013::KERNEL)),

                                    // #[cfg(feature = "c004_a014")]
                                    // "c004_a014" => Some((c004::c004_a014::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a014::KERNEL)),

                                    // #[cfg(feature = "c004_a015")]
                                    // "c004_a015" => Some((c004::c004_a015::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a015::KERNEL)),

                                    // #[cfg(feature = "c004_a016")]
                                    // "c004_a016" => Some((c004::c004_a016::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a016::KERNEL)),

                                    // #[cfg(feature = "c004_a017")]
                                    // "c004_a017" => Some((c004::c004_a017::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a017::KERNEL)),

                                    // #[cfg(feature = "c004_a018")]
                                    // "c004_a018" => Some((c004::c004_a018::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a018::KERNEL)),

                                    // #[cfg(feature = "c004_a019")]
                                    // "c004_a019" => Some((c004::c004_a019::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a019::KERNEL)),

                                    // #[cfg(feature = "c004_a020")]
                                    // "c004_a020" => Some((c004::c004_a020::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a020::KERNEL)),

                                    // #[cfg(feature = "c004_a021")]
                                    // "c004_a021" => Some((c004::c004_a021::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a021::KERNEL)),

                                    // #[cfg(feature = "c004_a022")]
                                    // "c004_a022" => Some((c004::c004_a022::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a022::KERNEL)),

                                    // #[cfg(feature = "c004_a023")]
                                    // "c004_a023" => Some((c004::c004_a023::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a023::KERNEL)),

                                    // #[cfg(feature = "c004_a024")]
                                    // "c004_a024" => Some((c004::c004_a024::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a024::KERNEL)),

                                    // #[cfg(feature = "c004_a025")]
                                    // "c004_a025" => Some((c004::c004_a025::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a025::KERNEL)),

                                    // #[cfg(feature = "c004_a026")]
                                    // "c004_a026" => Some((c004::c004_a026::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a026::KERNEL)),

                                    // #[cfg(feature = "c004_a027")]
                                    // "c004_a027" => Some((c004::c004_a027::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a027::KERNEL)),

                                    // #[cfg(feature = "c004_a028")]
                                    // "c004_a028" => Some((c004::c004_a028::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a028::KERNEL)),

                                    // #[cfg(feature = "c004_a029")]
                                    // "c004_a029" => Some((c004::c004_a029::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a029::KERNEL)),

                                    // #[cfg(feature = "c004_a030")]
                                    // "c004_a030" => Some((c004::c004_a030::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a030::KERNEL)),

                                    // #[cfg(feature = "c004_a031")]
                                    // "c004_a031" => Some((c004::c004_a031::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a031::KERNEL)),

                                    // #[cfg(feature = "c004_a032")]
                                    // "c004_a032" => Some((c004::c004_a032::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a032::KERNEL)),

                                    // #[cfg(feature = "c004_a033")]
                                    // "c004_a033" => Some((c004::c004_a033::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a033::KERNEL)),

                                    // #[cfg(feature = "c004_a034")]
                                    // "c004_a034" => Some((c004::c004_a034::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a034::KERNEL)),

                                    // #[cfg(feature = "c004_a035")]
                                    // "c004_a035" => Some((c004::c004_a035::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a035::KERNEL)),

                                    // #[cfg(feature = "c004_a036")]
                                    // "c004_a036" => Some((c004::c004_a036::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a036::KERNEL)),

                                    // #[cfg(feature = "c004_a037")]
                                    // "c004_a037" => Some((c004::c004_a037::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a037::KERNEL)),

                                    // #[cfg(feature = "c004_a038")]
                                    // "c004_a038" => Some((c004::c004_a038::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a038::KERNEL)),

                                    // #[cfg(feature = "c004_a039")]
                                    // "c004_a039" => Some((c004::c004_a039::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a039::KERNEL)),

                                    // #[cfg(feature = "c004_a040")]
                                    // "c004_a040" => Some((c004::c004_a040::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a040::KERNEL)),

                                    // #[cfg(feature = "c004_a041")]
                                    // "c004_a041" => Some((c004::c004_a041::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a041::KERNEL)),

                                    // #[cfg(feature = "c004_a042")]
                                    // "c004_a042" => Some((c004::c004_a042::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a042::KERNEL)),

                                    // #[cfg(feature = "c004_a043")]
                                    // "c004_a043" => Some((c004::c004_a043::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a043::KERNEL)),

                                    // #[cfg(feature = "c004_a044")]
                                    // "c004_a044" => Some((c004::c004_a044::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a044::KERNEL)),

                                    // #[cfg(feature = "c004_a045")]
                                    // "c004_a045" => Some((c004::c004_a045::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a045::KERNEL)),

                                    // #[cfg(feature = "c004_a046")]
                                    // "c004_a046" => Some((c004::c004_a046::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a046::KERNEL)),

                                    // #[cfg(feature = "c004_a047")]
                                    // "c004_a047" => Some((c004::c004_a047::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a047::KERNEL)),

                                    // #[cfg(feature = "c004_a048")]
                                    // "c004_a048" => Some((c004::c004_a048::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a048::KERNEL)),

                                    // #[cfg(feature = "c004_a049")]
                                    // "c004_a049" => Some((c004::c004_a049::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a049::KERNEL)),

                                    // #[cfg(feature = "c004_a050")]
                                    // "c004_a050" => Some((c004::c004_a050::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a050::KERNEL)),

                                    // #[cfg(feature = "c004_a051")]
                                    // "c004_a051" => Some((c004::c004_a051::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a051::KERNEL)),

                                    // #[cfg(feature = "c004_a052")]
                                    // "c004_a052" => Some((c004::c004_a052::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a052::KERNEL)),

                                    // #[cfg(feature = "c004_a053")]
                                    // "c004_a053" => Some((c004::c004_a053::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a053::KERNEL)),

                                    // #[cfg(feature = "c004_a054")]
                                    // "c004_a054" => Some((c004::c004_a054::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a054::KERNEL)),

                                    // #[cfg(feature = "c004_a055")]
                                    // "c004_a055" => Some((c004::c004_a055::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a055::KERNEL)),

                                    // #[cfg(feature = "c004_a056")]
                                    // "c004_a056" => Some((c004::c004_a056::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a056::KERNEL)),

                                    // #[cfg(feature = "c004_a057")]
                                    // "c004_a057" => Some((c004::c004_a057::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a057::KERNEL)),

                                    // #[cfg(feature = "c004_a058")]
                                    // "c004_a058" => Some((c004::c004_a058::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a058::KERNEL)),

                                    // #[cfg(feature = "c004_a059")]
                                    // "c004_a059" => Some((c004::c004_a059::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a059::KERNEL)),

                                    // #[cfg(feature = "c004_a060")]
                                    // "c004_a060" => Some((c004::c004_a060::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a060::KERNEL)),

                                    // #[cfg(feature = "c004_a061")]
                                    // "c004_a061" => Some((c004::c004_a061::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a061::KERNEL)),

                                    // #[cfg(feature = "c004_a062")]
                                    // "c004_a062" => Some((c004::c004_a062::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a062::KERNEL)),

                                    // #[cfg(feature = "c004_a063")]
                                    // "c004_a063" => Some((c004::c004_a063::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a063::KERNEL)),

                                    // #[cfg(feature = "c004_a064")]
                                    // "c004_a064" => Some((c004::c004_a064::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a064::KERNEL)),

                                    // #[cfg(feature = "c004_a065")]
                                    // "c004_a065" => Some((c004::c004_a065::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a065::KERNEL)),

                                    // #[cfg(feature = "c004_a066")]
                                    // "c004_a066" => Some((c004::c004_a066::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a066::KERNEL)),

                                    // #[cfg(feature = "c004_a067")]
                                    // "c004_a067" => Some((c004::c004_a067::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a067::KERNEL)),

                                    // #[cfg(feature = "c004_a068")]
                                    // "c004_a068" => Some((c004::c004_a068::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a068::KERNEL)),

                                    // #[cfg(feature = "c004_a069")]
                                    // "c004_a069" => Some((c004::c004_a069::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a069::KERNEL)),

                                    // #[cfg(feature = "c004_a070")]
                                    // "c004_a070" => Some((c004::c004_a070::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a070::KERNEL)),

                                    // #[cfg(feature = "c004_a071")]
                                    // "c004_a071" => Some((c004::c004_a071::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a071::KERNEL)),

                                    // #[cfg(feature = "c004_a072")]
                                    // "c004_a072" => Some((c004::c004_a072::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a072::KERNEL)),

                                    // #[cfg(feature = "c004_a073")]
                                    // "c004_a073" => Some((c004::c004_a073::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a073::KERNEL)),

                                    // #[cfg(feature = "c004_a074")]
                                    // "c004_a074" => Some((c004::c004_a074::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a074::KERNEL)),

                                    // #[cfg(feature = "c004_a075")]
                                    // "c004_a075" => Some((c004::c004_a075::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a075::KERNEL)),

                                    // #[cfg(feature = "c004_a076")]
                                    // "c004_a076" => Some((c004::c004_a076::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a076::KERNEL)),

                                    // #[cfg(feature = "c004_a077")]
                                    // "c004_a077" => Some((c004::c004_a077::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a077::KERNEL)),

                                    // #[cfg(feature = "c004_a078")]
                                    // "c004_a078" => Some((c004::c004_a078::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a078::KERNEL)),

                                    // #[cfg(feature = "c004_a079")]
                                    // "c004_a079" => Some((c004::c004_a079::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a079::KERNEL)),

                                    // #[cfg(feature = "c004_a080")]
                                    // "c004_a080" => Some((c004::c004_a080::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a080::KERNEL)),

                                    // #[cfg(feature = "c004_a081")]
                                    // "c004_a081" => Some((c004::c004_a081::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a081::KERNEL)),

                                    // #[cfg(feature = "c004_a082")]
                                    // "c004_a082" => Some((c004::c004_a082::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a082::KERNEL)),

                                    // #[cfg(feature = "c004_a083")]
                                    // "c004_a083" => Some((c004::c004_a083::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a083::KERNEL)),

                                    // #[cfg(feature = "c004_a084")]
                                    // "c004_a084" => Some((c004::c004_a084::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a084::KERNEL)),

                                    // #[cfg(feature = "c004_a085")]
                                    // "c004_a085" => Some((c004::c004_a085::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a085::KERNEL)),

                                    // #[cfg(feature = "c004_a086")]
                                    // "c004_a086" => Some((c004::c004_a086::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a086::KERNEL)),

                                    // #[cfg(feature = "c004_a087")]
                                    // "c004_a087" => Some((c004::c004_a087::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a087::KERNEL)),

                                    // #[cfg(feature = "c004_a088")]
                                    // "c004_a088" => Some((c004::c004_a088::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a088::KERNEL)),

                                    // #[cfg(feature = "c004_a089")]
                                    // "c004_a089" => Some((c004::c004_a089::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a089::KERNEL)),

                                    // #[cfg(feature = "c004_a090")]
                                    // "c004_a090" => Some((c004::c004_a090::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a090::KERNEL)),

                                    // #[cfg(feature = "c004_a091")]
                                    // "c004_a091" => Some((c004::c004_a091::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a091::KERNEL)),

                                    // #[cfg(feature = "c004_a092")]
                                    // "c004_a092" => Some((c004::c004_a092::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a092::KERNEL)),

                                    // #[cfg(feature = "c004_a093")]
                                    // "c004_a093" => Some((c004::c004_a093::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a093::KERNEL)),

                                    // #[cfg(feature = "c004_a094")]
                                    // "c004_a094" => Some((c004::c004_a094::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a094::KERNEL)),

                                    // #[cfg(feature = "c004_a095")]
                                    // "c004_a095" => Some((c004::c004_a095::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a095::KERNEL)),

                                    // #[cfg(feature = "c004_a096")]
                                    // "c004_a096" => Some((c004::c004_a096::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a096::KERNEL)),

                                    // #[cfg(feature = "c004_a097")]
                                    // "c004_a097" => Some((c004::c004_a097::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a097::KERNEL)),

                                    // #[cfg(feature = "c004_a098")]
                                    // "c004_a098" => Some((c004::c004_a098::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a098::KERNEL)),

                                    // #[cfg(feature = "c004_a099")]
                                    // "c004_a099" => Some((c004::c004_a099::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a099::KERNEL)),

                                    // #[cfg(feature = "c004_a100")]
                                    // "c004_a100" => Some((c004::c004_a100::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a100::KERNEL)),

                                    // #[cfg(feature = "c004_a101")]
                                    // "c004_a101" => Some((c004::c004_a101::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a101::KERNEL)),

                                    // #[cfg(feature = "c004_a102")]
                                    // "c004_a102" => Some((c004::c004_a102::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a102::KERNEL)),

                                    // #[cfg(feature = "c004_a103")]
                                    // "c004_a103" => Some((c004::c004_a103::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a103::KERNEL)),

                                    // #[cfg(feature = "c004_a104")]
                                    // "c004_a104" => Some((c004::c004_a104::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a104::KERNEL)),

                                    // #[cfg(feature = "c004_a105")]
                                    // "c004_a105" => Some((c004::c004_a105::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a105::KERNEL)),

                                    // #[cfg(feature = "c004_a106")]
                                    // "c004_a106" => Some((c004::c004_a106::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a106::KERNEL)),

                                    // #[cfg(feature = "c004_a107")]
                                    // "c004_a107" => Some((c004::c004_a107::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a107::KERNEL)),

                                    // #[cfg(feature = "c004_a108")]
                                    // "c004_a108" => Some((c004::c004_a108::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a108::KERNEL)),

                                    // #[cfg(feature = "c004_a109")]
                                    // "c004_a109" => Some((c004::c004_a109::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a109::KERNEL)),

                                    // #[cfg(feature = "c004_a110")]
                                    // "c004_a110" => Some((c004::c004_a110::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a110::KERNEL)),

                                    // #[cfg(feature = "c004_a111")]
                                    // "c004_a111" => Some((c004::c004_a111::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a111::KERNEL)),

                                    // #[cfg(feature = "c004_a112")]
                                    // "c004_a112" => Some((c004::c004_a112::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a112::KERNEL)),

                                    // #[cfg(feature = "c004_a113")]
                                    // "c004_a113" => Some((c004::c004_a113::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a113::KERNEL)),

                                    // #[cfg(feature = "c004_a114")]
                                    // "c004_a114" => Some((c004::c004_a114::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a114::KERNEL)),

                                    // #[cfg(feature = "c004_a115")]
                                    // "c004_a115" => Some((c004::c004_a115::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a115::KERNEL)),

                                    // #[cfg(feature = "c004_a116")]
                                    // "c004_a116" => Some((c004::c004_a116::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a116::KERNEL)),

                                    // #[cfg(feature = "c004_a117")]
                                    // "c004_a117" => Some((c004::c004_a117::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a117::KERNEL)),

                                    // #[cfg(feature = "c004_a118")]
                                    // "c004_a118" => Some((c004::c004_a118::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a118::KERNEL)),

                                    // #[cfg(feature = "c004_a119")]
                                    // "c004_a119" => Some((c004::c004_a119::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a119::KERNEL)),

                                    // #[cfg(feature = "c004_a120")]
                                    // "c004_a120" => Some((c004::c004_a120::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a120::KERNEL)),

                                    // #[cfg(feature = "c004_a121")]
                                    // "c004_a121" => Some((c004::c004_a121::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a121::KERNEL)),

                                    // #[cfg(feature = "c004_a122")]
                                    // "c004_a122" => Some((c004::c004_a122::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a122::KERNEL)),

                                    // #[cfg(feature = "c004_a123")]
                                    // "c004_a123" => Some((c004::c004_a123::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a123::KERNEL)),

                                    // #[cfg(feature = "c004_a124")]
                                    // "c004_a124" => Some((c004::c004_a124::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a124::KERNEL)),

                                    // #[cfg(feature = "c004_a125")]
                                    // "c004_a125" => Some((c004::c004_a125::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a125::KERNEL)),

                                    // #[cfg(feature = "c004_a126")]
                                    // "c004_a126" => Some((c004::c004_a126::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a126::KERNEL)),

                                    // #[cfg(feature = "c004_a127")]
                                    // "c004_a127" => Some((c004::c004_a127::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a127::KERNEL)),

                                    // #[cfg(feature = "c004_a128")]
                                    // "c004_a128" => Some((c004::c004_a128::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a128::KERNEL)),

                                    // #[cfg(feature = "c004_a129")]
                                    // "c004_a129" => Some((c004::c004_a129::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a129::KERNEL)),

                                    // #[cfg(feature = "c004_a130")]
                                    // "c004_a130" => Some((c004::c004_a130::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a130::KERNEL)),

                                    // #[cfg(feature = "c004_a131")]
                                    // "c004_a131" => Some((c004::c004_a131::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a131::KERNEL)),

                                    // #[cfg(feature = "c004_a132")]
                                    // "c004_a132" => Some((c004::c004_a132::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a132::KERNEL)),

                                    // #[cfg(feature = "c004_a133")]
                                    // "c004_a133" => Some((c004::c004_a133::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a133::KERNEL)),

                                    // #[cfg(feature = "c004_a134")]
                                    // "c004_a134" => Some((c004::c004_a134::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a134::KERNEL)),

                                    // #[cfg(feature = "c004_a135")]
                                    // "c004_a135" => Some((c004::c004_a135::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a135::KERNEL)),

                                    // #[cfg(feature = "c004_a136")]
                                    // "c004_a136" => Some((c004::c004_a136::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a136::KERNEL)),

                                    // #[cfg(feature = "c004_a137")]
                                    // "c004_a137" => Some((c004::c004_a137::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a137::KERNEL)),

                                    // #[cfg(feature = "c004_a138")]
                                    // "c004_a138" => Some((c004::c004_a138::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a138::KERNEL)),

                                    // #[cfg(feature = "c004_a139")]
                                    // "c004_a139" => Some((c004::c004_a139::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a139::KERNEL)),

                                    // #[cfg(feature = "c004_a140")]
                                    // "c004_a140" => Some((c004::c004_a140::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a140::KERNEL)),

                                    // #[cfg(feature = "c004_a141")]
                                    // "c004_a141" => Some((c004::c004_a141::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a141::KERNEL)),

                                    // #[cfg(feature = "c004_a142")]
                                    // "c004_a142" => Some((c004::c004_a142::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a142::KERNEL)),

                                    // #[cfg(feature = "c004_a143")]
                                    // "c004_a143" => Some((c004::c004_a143::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a143::KERNEL)),

                                    // #[cfg(feature = "c004_a144")]
                                    // "c004_a144" => Some((c004::c004_a144::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a144::KERNEL)),

                                    // #[cfg(feature = "c004_a145")]
                                    // "c004_a145" => Some((c004::c004_a145::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a145::KERNEL)),

                                    // #[cfg(feature = "c004_a146")]
                                    // "c004_a146" => Some((c004::c004_a146::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a146::KERNEL)),

                                    // #[cfg(feature = "c004_a147")]
                                    // "c004_a147" => Some((c004::c004_a147::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a147::KERNEL)),

                                    // #[cfg(feature = "c004_a148")]
                                    // "c004_a148" => Some((c004::c004_a148::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a148::KERNEL)),

                                    // #[cfg(feature = "c004_a149")]
                                    // "c004_a149" => Some((c004::c004_a149::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a149::KERNEL)),

                                    // #[cfg(feature = "c004_a150")]
                                    // "c004_a150" => Some((c004::c004_a150::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a150::KERNEL)),

                                    // #[cfg(feature = "c004_a151")]
                                    // "c004_a151" => Some((c004::c004_a151::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a151::KERNEL)),

                                    // #[cfg(feature = "c004_a152")]
                                    // "c004_a152" => Some((c004::c004_a152::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a152::KERNEL)),

                                    // #[cfg(feature = "c004_a153")]
                                    // "c004_a153" => Some((c004::c004_a153::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a153::KERNEL)),

                                    // #[cfg(feature = "c004_a154")]
                                    // "c004_a154" => Some((c004::c004_a154::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a154::KERNEL)),

                                    // #[cfg(feature = "c004_a155")]
                                    // "c004_a155" => Some((c004::c004_a155::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a155::KERNEL)),

                                    // #[cfg(feature = "c004_a156")]
                                    // "c004_a156" => Some((c004::c004_a156::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a156::KERNEL)),

                                    // #[cfg(feature = "c004_a157")]
                                    // "c004_a157" => Some((c004::c004_a157::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a157::KERNEL)),

                                    // #[cfg(feature = "c004_a158")]
                                    // "c004_a158" => Some((c004::c004_a158::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a158::KERNEL)),

                                    // #[cfg(feature = "c004_a159")]
                                    // "c004_a159" => Some((c004::c004_a159::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a159::KERNEL)),

                                    // #[cfg(feature = "c004_a160")]
                                    // "c004_a160" => Some((c004::c004_a160::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a160::KERNEL)),

                                    // #[cfg(feature = "c004_a161")]
                                    // "c004_a161" => Some((c004::c004_a161::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a161::KERNEL)),

                                    // #[cfg(feature = "c004_a162")]
                                    // "c004_a162" => Some((c004::c004_a162::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a162::KERNEL)),

                                    // #[cfg(feature = "c004_a163")]
                                    // "c004_a163" => Some((c004::c004_a163::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a163::KERNEL)),

                                    // #[cfg(feature = "c004_a164")]
                                    // "c004_a164" => Some((c004::c004_a164::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a164::KERNEL)),

                                    // #[cfg(feature = "c004_a165")]
                                    // "c004_a165" => Some((c004::c004_a165::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a165::KERNEL)),

                                    // #[cfg(feature = "c004_a166")]
                                    // "c004_a166" => Some((c004::c004_a166::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a166::KERNEL)),

                                    // #[cfg(feature = "c004_a167")]
                                    // "c004_a167" => Some((c004::c004_a167::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a167::KERNEL)),

                                    // #[cfg(feature = "c004_a168")]
                                    // "c004_a168" => Some((c004::c004_a168::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a168::KERNEL)),

                                    // #[cfg(feature = "c004_a169")]
                                    // "c004_a169" => Some((c004::c004_a169::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a169::KERNEL)),

                                    // #[cfg(feature = "c004_a170")]
                                    // "c004_a170" => Some((c004::c004_a170::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a170::KERNEL)),

                                    // #[cfg(feature = "c004_a171")]
                                    // "c004_a171" => Some((c004::c004_a171::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a171::KERNEL)),

                                    // #[cfg(feature = "c004_a172")]
                                    // "c004_a172" => Some((c004::c004_a172::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a172::KERNEL)),

                                    // #[cfg(feature = "c004_a173")]
                                    // "c004_a173" => Some((c004::c004_a173::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a173::KERNEL)),

                                    // #[cfg(feature = "c004_a174")]
                                    // "c004_a174" => Some((c004::c004_a174::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a174::KERNEL)),

                                    // #[cfg(feature = "c004_a175")]
                                    // "c004_a175" => Some((c004::c004_a175::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a175::KERNEL)),

                                    // #[cfg(feature = "c004_a176")]
                                    // "c004_a176" => Some((c004::c004_a176::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a176::KERNEL)),

                                    // #[cfg(feature = "c004_a177")]
                                    // "c004_a177" => Some((c004::c004_a177::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a177::KERNEL)),

                                    // #[cfg(feature = "c004_a178")]
                                    // "c004_a178" => Some((c004::c004_a178::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a178::KERNEL)),

                                    // #[cfg(feature = "c004_a179")]
                                    // "c004_a179" => Some((c004::c004_a179::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a179::KERNEL)),

                                    // #[cfg(feature = "c004_a180")]
                                    // "c004_a180" => Some((c004::c004_a180::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a180::KERNEL)),

                                    // #[cfg(feature = "c004_a181")]
                                    // "c004_a181" => Some((c004::c004_a181::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a181::KERNEL)),

                                    // #[cfg(feature = "c004_a182")]
                                    // "c004_a182" => Some((c004::c004_a182::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a182::KERNEL)),

                                    // #[cfg(feature = "c004_a183")]
                                    // "c004_a183" => Some((c004::c004_a183::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a183::KERNEL)),

                                    // #[cfg(feature = "c004_a184")]
                                    // "c004_a184" => Some((c004::c004_a184::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a184::KERNEL)),

                                    // #[cfg(feature = "c004_a185")]
                                    // "c004_a185" => Some((c004::c004_a185::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a185::KERNEL)),

                                    // #[cfg(feature = "c004_a186")]
                                    // "c004_a186" => Some((c004::c004_a186::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a186::KERNEL)),

                                    // #[cfg(feature = "c004_a187")]
                                    // "c004_a187" => Some((c004::c004_a187::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a187::KERNEL)),

                                    // #[cfg(feature = "c004_a188")]
                                    // "c004_a188" => Some((c004::c004_a188::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a188::KERNEL)),

                                    // #[cfg(feature = "c004_a189")]
                                    // "c004_a189" => Some((c004::c004_a189::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a189::KERNEL)),

                                    // #[cfg(feature = "c004_a190")]
                                    // "c004_a190" => Some((c004::c004_a190::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a190::KERNEL)),

                                    // #[cfg(feature = "c004_a191")]
                                    // "c004_a191" => Some((c004::c004_a191::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a191::KERNEL)),

                                    // #[cfg(feature = "c004_a192")]
                                    // "c004_a192" => Some((c004::c004_a192::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a192::KERNEL)),

                                    // #[cfg(feature = "c004_a193")]
                                    // "c004_a193" => Some((c004::c004_a193::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a193::KERNEL)),

                                    // #[cfg(feature = "c004_a194")]
                                    // "c004_a194" => Some((c004::c004_a194::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a194::KERNEL)),

                                    // #[cfg(feature = "c004_a195")]
                                    // "c004_a195" => Some((c004::c004_a195::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a195::KERNEL)),

                                    // #[cfg(feature = "c004_a196")]
                                    // "c004_a196" => Some((c004::c004_a196::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a196::KERNEL)),

                                    // #[cfg(feature = "c004_a197")]
                                    // "c004_a197" => Some((c004::c004_a197::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a197::KERNEL)),

                                    // #[cfg(feature = "c004_a198")]
                                    // "c004_a198" => Some((c004::c004_a198::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a198::KERNEL)),

                                    // #[cfg(feature = "c004_a199")]
                                    // "c004_a199" => Some((c004::c004_a199::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a199::KERNEL)),

                                    // #[cfg(feature = "c004_a200")]
                                    // "c004_a200" => Some((c004::c004_a200::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a200::KERNEL)),

                                    // #[cfg(feature = "c004_a201")]
                                    // "c004_a201" => Some((c004::c004_a201::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a201::KERNEL)),

                                    // #[cfg(feature = "c004_a202")]
                                    // "c004_a202" => Some((c004::c004_a202::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a202::KERNEL)),

                                    // #[cfg(feature = "c004_a203")]
                                    // "c004_a203" => Some((c004::c004_a203::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a203::KERNEL)),

                                    // #[cfg(feature = "c004_a204")]
                                    // "c004_a204" => Some((c004::c004_a204::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a204::KERNEL)),

                                    // #[cfg(feature = "c004_a205")]
                                    // "c004_a205" => Some((c004::c004_a205::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a205::KERNEL)),

                                    // #[cfg(feature = "c004_a206")]
                                    // "c004_a206" => Some((c004::c004_a206::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a206::KERNEL)),

                                    // #[cfg(feature = "c004_a207")]
                                    // "c004_a207" => Some((c004::c004_a207::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a207::KERNEL)),

                                    // #[cfg(feature = "c004_a208")]
                                    // "c004_a208" => Some((c004::c004_a208::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a208::KERNEL)),

                                    // #[cfg(feature = "c004_a209")]
                                    // "c004_a209" => Some((c004::c004_a209::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a209::KERNEL)),

                                    // #[cfg(feature = "c004_a210")]
                                    // "c004_a210" => Some((c004::c004_a210::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a210::KERNEL)),

                                    // #[cfg(feature = "c004_a211")]
                                    // "c004_a211" => Some((c004::c004_a211::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a211::KERNEL)),

                                    // #[cfg(feature = "c004_a212")]
                                    // "c004_a212" => Some((c004::c004_a212::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a212::KERNEL)),

                                    // #[cfg(feature = "c004_a213")]
                                    // "c004_a213" => Some((c004::c004_a213::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a213::KERNEL)),

                                    // #[cfg(feature = "c004_a214")]
                                    // "c004_a214" => Some((c004::c004_a214::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a214::KERNEL)),

                                    // #[cfg(feature = "c004_a215")]
                                    // "c004_a215" => Some((c004::c004_a215::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a215::KERNEL)),

                                    // #[cfg(feature = "c004_a216")]
                                    // "c004_a216" => Some((c004::c004_a216::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a216::KERNEL)),

                                    // #[cfg(feature = "c004_a217")]
                                    // "c004_a217" => Some((c004::c004_a217::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a217::KERNEL)),

                                    // #[cfg(feature = "c004_a218")]
                                    // "c004_a218" => Some((c004::c004_a218::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a218::KERNEL)),

                                    // #[cfg(feature = "c004_a219")]
                                    // "c004_a219" => Some((c004::c004_a219::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a219::KERNEL)),

                                    // #[cfg(feature = "c004_a220")]
                                    // "c004_a220" => Some((c004::c004_a220::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a220::KERNEL)),

                                    // #[cfg(feature = "c004_a221")]
                                    // "c004_a221" => Some((c004::c004_a221::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a221::KERNEL)),

                                    // #[cfg(feature = "c004_a222")]
                                    // "c004_a222" => Some((c004::c004_a222::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a222::KERNEL)),

                                    // #[cfg(feature = "c004_a223")]
                                    // "c004_a223" => Some((c004::c004_a223::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a223::KERNEL)),

                                    // #[cfg(feature = "c004_a224")]
                                    // "c004_a224" => Some((c004::c004_a224::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a224::KERNEL)),

                                    // #[cfg(feature = "c004_a225")]
                                    // "c004_a225" => Some((c004::c004_a225::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a225::KERNEL)),

                                    // #[cfg(feature = "c004_a226")]
                                    // "c004_a226" => Some((c004::c004_a226::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a226::KERNEL)),

                                    // #[cfg(feature = "c004_a227")]
                                    // "c004_a227" => Some((c004::c004_a227::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a227::KERNEL)),

                                    // #[cfg(feature = "c004_a228")]
                                    // "c004_a228" => Some((c004::c004_a228::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a228::KERNEL)),

                                    // #[cfg(feature = "c004_a229")]
                                    // "c004_a229" => Some((c004::c004_a229::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a229::KERNEL)),

                                    // #[cfg(feature = "c004_a230")]
                                    // "c004_a230" => Some((c004::c004_a230::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a230::KERNEL)),

                                    // #[cfg(feature = "c004_a231")]
                                    // "c004_a231" => Some((c004::c004_a231::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a231::KERNEL)),

                                    // #[cfg(feature = "c004_a232")]
                                    // "c004_a232" => Some((c004::c004_a232::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a232::KERNEL)),

                                    // #[cfg(feature = "c004_a233")]
                                    // "c004_a233" => Some((c004::c004_a233::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a233::KERNEL)),

                                    // #[cfg(feature = "c004_a234")]
                                    // "c004_a234" => Some((c004::c004_a234::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a234::KERNEL)),

                                    // #[cfg(feature = "c004_a235")]
                                    // "c004_a235" => Some((c004::c004_a235::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a235::KERNEL)),

                                    // #[cfg(feature = "c004_a236")]
                                    // "c004_a236" => Some((c004::c004_a236::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a236::KERNEL)),

                                    // #[cfg(feature = "c004_a237")]
                                    // "c004_a237" => Some((c004::c004_a237::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a237::KERNEL)),

                                    // #[cfg(feature = "c004_a238")]
                                    // "c004_a238" => Some((c004::c004_a238::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a238::KERNEL)),

                                    // #[cfg(feature = "c004_a239")]
                                    // "c004_a239" => Some((c004::c004_a239::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a239::KERNEL)),

                                    // #[cfg(feature = "c004_a240")]
                                    // "c004_a240" => Some((c004::c004_a240::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a240::KERNEL)),

                                    // #[cfg(feature = "c004_a241")]
                                    // "c004_a241" => Some((c004::c004_a241::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a241::KERNEL)),

                                    // #[cfg(feature = "c004_a242")]
                                    // "c004_a242" => Some((c004::c004_a242::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a242::KERNEL)),

                                    // #[cfg(feature = "c004_a243")]
                                    // "c004_a243" => Some((c004::c004_a243::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a243::KERNEL)),

                                    // #[cfg(feature = "c004_a244")]
                                    // "c004_a244" => Some((c004::c004_a244::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a244::KERNEL)),

                                    // #[cfg(feature = "c004_a245")]
                                    // "c004_a245" => Some((c004::c004_a245::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a245::KERNEL)),

                                    // #[cfg(feature = "c004_a246")]
                                    // "c004_a246" => Some((c004::c004_a246::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a246::KERNEL)),

                                    // #[cfg(feature = "c004_a247")]
                                    // "c004_a247" => Some((c004::c004_a247::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a247::KERNEL)),

                                    // #[cfg(feature = "c004_a248")]
                                    // "c004_a248" => Some((c004::c004_a248::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a248::KERNEL)),

                                    // #[cfg(feature = "c004_a249")]
                                    // "c004_a249" => Some((c004::c004_a249::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a249::KERNEL)),

                                    // #[cfg(feature = "c004_a250")]
                                    // "c004_a250" => Some((c004::c004_a250::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a250::KERNEL)),

                                    // #[cfg(feature = "c004_a251")]
                                    // "c004_a251" => Some((c004::c004_a251::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a251::KERNEL)),

                                    // #[cfg(feature = "c004_a252")]
                                    // "c004_a252" => Some((c004::c004_a252::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a252::KERNEL)),

                                    // #[cfg(feature = "c004_a253")]
                                    // "c004_a253" => Some((c004::c004_a253::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a253::KERNEL)),

                                    // #[cfg(feature = "c004_a254")]
                                    // "c004_a254" => Some((c004::c004_a254::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a254::KERNEL)),

                                    // #[cfg(feature = "c004_a255")]
                                    // "c004_a255" => Some((c004::c004_a255::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a255::KERNEL)),

                                    // #[cfg(feature = "c004_a256")]
                                    // "c004_a256" => Some((c004::c004_a256::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a256::KERNEL)),

                                    // #[cfg(feature = "c004_a257")]
                                    // "c004_a257" => Some((c004::c004_a257::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a257::KERNEL)),

                                    // #[cfg(feature = "c004_a258")]
                                    // "c004_a258" => Some((c004::c004_a258::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a258::KERNEL)),

                                    // #[cfg(feature = "c004_a259")]
                                    // "c004_a259" => Some((c004::c004_a259::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a259::KERNEL)),

                                    // #[cfg(feature = "c004_a260")]
                                    // "c004_a260" => Some((c004::c004_a260::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a260::KERNEL)),

                                    // #[cfg(feature = "c004_a261")]
                                    // "c004_a261" => Some((c004::c004_a261::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a261::KERNEL)),

                                    // #[cfg(feature = "c004_a262")]
                                    // "c004_a262" => Some((c004::c004_a262::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a262::KERNEL)),

                                    // #[cfg(feature = "c004_a263")]
                                    // "c004_a263" => Some((c004::c004_a263::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a263::KERNEL)),

                                    // #[cfg(feature = "c004_a264")]
                                    // "c004_a264" => Some((c004::c004_a264::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a264::KERNEL)),

                                    // #[cfg(feature = "c004_a265")]
                                    // "c004_a265" => Some((c004::c004_a265::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a265::KERNEL)),

                                    // #[cfg(feature = "c004_a266")]
                                    // "c004_a266" => Some((c004::c004_a266::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a266::KERNEL)),

                                    // #[cfg(feature = "c004_a267")]
                                    // "c004_a267" => Some((c004::c004_a267::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a267::KERNEL)),

                                    // #[cfg(feature = "c004_a268")]
                                    // "c004_a268" => Some((c004::c004_a268::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a268::KERNEL)),

                                    // #[cfg(feature = "c004_a269")]
                                    // "c004_a269" => Some((c004::c004_a269::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a269::KERNEL)),

                                    // #[cfg(feature = "c004_a270")]
                                    // "c004_a270" => Some((c004::c004_a270::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a270::KERNEL)),

                                    // #[cfg(feature = "c004_a271")]
                                    // "c004_a271" => Some((c004::c004_a271::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a271::KERNEL)),

                                    // #[cfg(feature = "c004_a272")]
                                    // "c004_a272" => Some((c004::c004_a272::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a272::KERNEL)),

                                    // #[cfg(feature = "c004_a273")]
                                    // "c004_a273" => Some((c004::c004_a273::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a273::KERNEL)),

                                    // #[cfg(feature = "c004_a274")]
                                    // "c004_a274" => Some((c004::c004_a274::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a274::KERNEL)),

                                    // #[cfg(feature = "c004_a275")]
                                    // "c004_a275" => Some((c004::c004_a275::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a275::KERNEL)),

                                    // #[cfg(feature = "c004_a276")]
                                    // "c004_a276" => Some((c004::c004_a276::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a276::KERNEL)),

                                    // #[cfg(feature = "c004_a277")]
                                    // "c004_a277" => Some((c004::c004_a277::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a277::KERNEL)),

                                    // #[cfg(feature = "c004_a278")]
                                    // "c004_a278" => Some((c004::c004_a278::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a278::KERNEL)),

                                    // #[cfg(feature = "c004_a279")]
                                    // "c004_a279" => Some((c004::c004_a279::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a279::KERNEL)),

                                    // #[cfg(feature = "c004_a280")]
                                    // "c004_a280" => Some((c004::c004_a280::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a280::KERNEL)),

                                    // #[cfg(feature = "c004_a281")]
                                    // "c004_a281" => Some((c004::c004_a281::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a281::KERNEL)),

                                    // #[cfg(feature = "c004_a282")]
                                    // "c004_a282" => Some((c004::c004_a282::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a282::KERNEL)),

                                    // #[cfg(feature = "c004_a283")]
                                    // "c004_a283" => Some((c004::c004_a283::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a283::KERNEL)),

                                    // #[cfg(feature = "c004_a284")]
                                    // "c004_a284" => Some((c004::c004_a284::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a284::KERNEL)),

                                    // #[cfg(feature = "c004_a285")]
                                    // "c004_a285" => Some((c004::c004_a285::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a285::KERNEL)),

                                    // #[cfg(feature = "c004_a286")]
                                    // "c004_a286" => Some((c004::c004_a286::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a286::KERNEL)),

                                    // #[cfg(feature = "c004_a287")]
                                    // "c004_a287" => Some((c004::c004_a287::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a287::KERNEL)),

                                    // #[cfg(feature = "c004_a288")]
                                    // "c004_a288" => Some((c004::c004_a288::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a288::KERNEL)),

                                    // #[cfg(feature = "c004_a289")]
                                    // "c004_a289" => Some((c004::c004_a289::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a289::KERNEL)),

                                    // #[cfg(feature = "c004_a290")]
                                    // "c004_a290" => Some((c004::c004_a290::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a290::KERNEL)),

                                    // #[cfg(feature = "c004_a291")]
                                    // "c004_a291" => Some((c004::c004_a291::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a291::KERNEL)),

                                    // #[cfg(feature = "c004_a292")]
                                    // "c004_a292" => Some((c004::c004_a292::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a292::KERNEL)),

                                    // #[cfg(feature = "c004_a293")]
                                    // "c004_a293" => Some((c004::c004_a293::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a293::KERNEL)),

                                    // #[cfg(feature = "c004_a294")]
                                    // "c004_a294" => Some((c004::c004_a294::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a294::KERNEL)),

                                    // #[cfg(feature = "c004_a295")]
                                    // "c004_a295" => Some((c004::c004_a295::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a295::KERNEL)),

                                    // #[cfg(feature = "c004_a296")]
                                    // "c004_a296" => Some((c004::c004_a296::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a296::KERNEL)),

                                    // #[cfg(feature = "c004_a297")]
                                    // "c004_a297" => Some((c004::c004_a297::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a297::KERNEL)),

                                    // #[cfg(feature = "c004_a298")]
                                    // "c004_a298" => Some((c004::c004_a298::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a298::KERNEL)),

                                    // #[cfg(feature = "c004_a299")]
                                    // "c004_a299" => Some((c004::c004_a299::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a299::KERNEL)),

                                    // #[cfg(feature = "c004_a300")]
                                    // "c004_a300" => Some((c004::c004_a300::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a300::KERNEL)),

                                    // #[cfg(feature = "c004_a301")]
                                    // "c004_a301" => Some((c004::c004_a301::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a301::KERNEL)),

                                    // #[cfg(feature = "c004_a302")]
                                    // "c004_a302" => Some((c004::c004_a302::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a302::KERNEL)),

                                    // #[cfg(feature = "c004_a303")]
                                    // "c004_a303" => Some((c004::c004_a303::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a303::KERNEL)),

                                    // #[cfg(feature = "c004_a304")]
                                    // "c004_a304" => Some((c004::c004_a304::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a304::KERNEL)),

                                    // #[cfg(feature = "c004_a305")]
                                    // "c004_a305" => Some((c004::c004_a305::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a305::KERNEL)),

                                    // #[cfg(feature = "c004_a306")]
                                    // "c004_a306" => Some((c004::c004_a306::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a306::KERNEL)),

                                    // #[cfg(feature = "c004_a307")]
                                    // "c004_a307" => Some((c004::c004_a307::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a307::KERNEL)),

                                    // #[cfg(feature = "c004_a308")]
                                    // "c004_a308" => Some((c004::c004_a308::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a308::KERNEL)),

                                    // #[cfg(feature = "c004_a309")]
                                    // "c004_a309" => Some((c004::c004_a309::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a309::KERNEL)),

                                    // #[cfg(feature = "c004_a310")]
                                    // "c004_a310" => Some((c004::c004_a310::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a310::KERNEL)),

                                    // #[cfg(feature = "c004_a311")]
                                    // "c004_a311" => Some((c004::c004_a311::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a311::KERNEL)),

                                    // #[cfg(feature = "c004_a312")]
                                    // "c004_a312" => Some((c004::c004_a312::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a312::KERNEL)),

                                    // #[cfg(feature = "c004_a313")]
                                    // "c004_a313" => Some((c004::c004_a313::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a313::KERNEL)),

                                    // #[cfg(feature = "c004_a314")]
                                    // "c004_a314" => Some((c004::c004_a314::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a314::KERNEL)),

                                    // #[cfg(feature = "c004_a315")]
                                    // "c004_a315" => Some((c004::c004_a315::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a315::KERNEL)),

                                    // #[cfg(feature = "c004_a316")]
                                    // "c004_a316" => Some((c004::c004_a316::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a316::KERNEL)),

                                    // #[cfg(feature = "c004_a317")]
                                    // "c004_a317" => Some((c004::c004_a317::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a317::KERNEL)),

                                    // #[cfg(feature = "c004_a318")]
                                    // "c004_a318" => Some((c004::c004_a318::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a318::KERNEL)),

                                    // #[cfg(feature = "c004_a319")]
                                    // "c004_a319" => Some((c004::c004_a319::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a319::KERNEL)),

                                    // #[cfg(feature = "c004_a320")]
                                    // "c004_a320" => Some((c004::c004_a320::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a320::KERNEL)),

                                    // #[cfg(feature = "c004_a321")]
                                    // "c004_a321" => Some((c004::c004_a321::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a321::KERNEL)),

                                    // #[cfg(feature = "c004_a322")]
                                    // "c004_a322" => Some((c004::c004_a322::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a322::KERNEL)),

                                    // #[cfg(feature = "c004_a323")]
                                    // "c004_a323" => Some((c004::c004_a323::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a323::KERNEL)),

                                    // #[cfg(feature = "c004_a324")]
                                    // "c004_a324" => Some((c004::c004_a324::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a324::KERNEL)),

                                    // #[cfg(feature = "c004_a325")]
                                    // "c004_a325" => Some((c004::c004_a325::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a325::KERNEL)),

                                    // #[cfg(feature = "c004_a326")]
                                    // "c004_a326" => Some((c004::c004_a326::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a326::KERNEL)),

                                    // #[cfg(feature = "c004_a327")]
                                    // "c004_a327" => Some((c004::c004_a327::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a327::KERNEL)),

                                    // #[cfg(feature = "c004_a328")]
                                    // "c004_a328" => Some((c004::c004_a328::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a328::KERNEL)),

                                    // #[cfg(feature = "c004_a329")]
                                    // "c004_a329" => Some((c004::c004_a329::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a329::KERNEL)),

                                    // #[cfg(feature = "c004_a330")]
                                    // "c004_a330" => Some((c004::c004_a330::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a330::KERNEL)),

                                    // #[cfg(feature = "c004_a331")]
                                    // "c004_a331" => Some((c004::c004_a331::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a331::KERNEL)),

                                    // #[cfg(feature = "c004_a332")]
                                    // "c004_a332" => Some((c004::c004_a332::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a332::KERNEL)),

                                    // #[cfg(feature = "c004_a333")]
                                    // "c004_a333" => Some((c004::c004_a333::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a333::KERNEL)),

                                    // #[cfg(feature = "c004_a334")]
                                    // "c004_a334" => Some((c004::c004_a334::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a334::KERNEL)),

                                    // #[cfg(feature = "c004_a335")]
                                    // "c004_a335" => Some((c004::c004_a335::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a335::KERNEL)),

                                    // #[cfg(feature = "c004_a336")]
                                    // "c004_a336" => Some((c004::c004_a336::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a336::KERNEL)),

                                    // #[cfg(feature = "c004_a337")]
                                    // "c004_a337" => Some((c004::c004_a337::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a337::KERNEL)),

                                    // #[cfg(feature = "c004_a338")]
                                    // "c004_a338" => Some((c004::c004_a338::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a338::KERNEL)),

                                    // #[cfg(feature = "c004_a339")]
                                    // "c004_a339" => Some((c004::c004_a339::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a339::KERNEL)),

                                    // #[cfg(feature = "c004_a340")]
                                    // "c004_a340" => Some((c004::c004_a340::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a340::KERNEL)),

                                    // #[cfg(feature = "c004_a341")]
                                    // "c004_a341" => Some((c004::c004_a341::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a341::KERNEL)),

                                    // #[cfg(feature = "c004_a342")]
                                    // "c004_a342" => Some((c004::c004_a342::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a342::KERNEL)),

                                    // #[cfg(feature = "c004_a343")]
                                    // "c004_a343" => Some((c004::c004_a343::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a343::KERNEL)),

                                    // #[cfg(feature = "c004_a344")]
                                    // "c004_a344" => Some((c004::c004_a344::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a344::KERNEL)),

                                    // #[cfg(feature = "c004_a345")]
                                    // "c004_a345" => Some((c004::c004_a345::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a345::KERNEL)),

                                    // #[cfg(feature = "c004_a346")]
                                    // "c004_a346" => Some((c004::c004_a346::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a346::KERNEL)),

                                    // #[cfg(feature = "c004_a347")]
                                    // "c004_a347" => Some((c004::c004_a347::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a347::KERNEL)),

                                    // #[cfg(feature = "c004_a348")]
                                    // "c004_a348" => Some((c004::c004_a348::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a348::KERNEL)),

                                    // #[cfg(feature = "c004_a349")]
                                    // "c004_a349" => Some((c004::c004_a349::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a349::KERNEL)),

                                    // #[cfg(feature = "c004_a350")]
                                    // "c004_a350" => Some((c004::c004_a350::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a350::KERNEL)),

                                    // #[cfg(feature = "c004_a351")]
                                    // "c004_a351" => Some((c004::c004_a351::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a351::KERNEL)),

                                    // #[cfg(feature = "c004_a352")]
                                    // "c004_a352" => Some((c004::c004_a352::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a352::KERNEL)),

                                    // #[cfg(feature = "c004_a353")]
                                    // "c004_a353" => Some((c004::c004_a353::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a353::KERNEL)),

                                    // #[cfg(feature = "c004_a354")]
                                    // "c004_a354" => Some((c004::c004_a354::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a354::KERNEL)),

                                    // #[cfg(feature = "c004_a355")]
                                    // "c004_a355" => Some((c004::c004_a355::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a355::KERNEL)),

                                    // #[cfg(feature = "c004_a356")]
                                    // "c004_a356" => Some((c004::c004_a356::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a356::KERNEL)),

                                    // #[cfg(feature = "c004_a357")]
                                    // "c004_a357" => Some((c004::c004_a357::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a357::KERNEL)),

                                    // #[cfg(feature = "c004_a358")]
                                    // "c004_a358" => Some((c004::c004_a358::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a358::KERNEL)),

                                    // #[cfg(feature = "c004_a359")]
                                    // "c004_a359" => Some((c004::c004_a359::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a359::KERNEL)),

                                    // #[cfg(feature = "c004_a360")]
                                    // "c004_a360" => Some((c004::c004_a360::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a360::KERNEL)),

                                    // #[cfg(feature = "c004_a361")]
                                    // "c004_a361" => Some((c004::c004_a361::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a361::KERNEL)),

                                    // #[cfg(feature = "c004_a362")]
                                    // "c004_a362" => Some((c004::c004_a362::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a362::KERNEL)),

                                    // #[cfg(feature = "c004_a363")]
                                    // "c004_a363" => Some((c004::c004_a363::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a363::KERNEL)),

                                    // #[cfg(feature = "c004_a364")]
                                    // "c004_a364" => Some((c004::c004_a364::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a364::KERNEL)),

                                    // #[cfg(feature = "c004_a365")]
                                    // "c004_a365" => Some((c004::c004_a365::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a365::KERNEL)),

                                    // #[cfg(feature = "c004_a366")]
                                    // "c004_a366" => Some((c004::c004_a366::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a366::KERNEL)),

                                    // #[cfg(feature = "c004_a367")]
                                    // "c004_a367" => Some((c004::c004_a367::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a367::KERNEL)),

                                    // #[cfg(feature = "c004_a368")]
                                    // "c004_a368" => Some((c004::c004_a368::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a368::KERNEL)),

                                    // #[cfg(feature = "c004_a369")]
                                    // "c004_a369" => Some((c004::c004_a369::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a369::KERNEL)),

                                    // #[cfg(feature = "c004_a370")]
                                    // "c004_a370" => Some((c004::c004_a370::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a370::KERNEL)),

                                    // #[cfg(feature = "c004_a371")]
                                    // "c004_a371" => Some((c004::c004_a371::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a371::KERNEL)),

                                    // #[cfg(feature = "c004_a372")]
                                    // "c004_a372" => Some((c004::c004_a372::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a372::KERNEL)),

                                    // #[cfg(feature = "c004_a373")]
                                    // "c004_a373" => Some((c004::c004_a373::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a373::KERNEL)),

                                    // #[cfg(feature = "c004_a374")]
                                    // "c004_a374" => Some((c004::c004_a374::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a374::KERNEL)),

                                    // #[cfg(feature = "c004_a375")]
                                    // "c004_a375" => Some((c004::c004_a375::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a375::KERNEL)),

                                    // #[cfg(feature = "c004_a376")]
                                    // "c004_a376" => Some((c004::c004_a376::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a376::KERNEL)),

                                    // #[cfg(feature = "c004_a377")]
                                    // "c004_a377" => Some((c004::c004_a377::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a377::KERNEL)),

                                    // #[cfg(feature = "c004_a378")]
                                    // "c004_a378" => Some((c004::c004_a378::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a378::KERNEL)),

                                    // #[cfg(feature = "c004_a379")]
                                    // "c004_a379" => Some((c004::c004_a379::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a379::KERNEL)),

                                    // #[cfg(feature = "c004_a380")]
                                    // "c004_a380" => Some((c004::c004_a380::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a380::KERNEL)),

                                    // #[cfg(feature = "c004_a381")]
                                    // "c004_a381" => Some((c004::c004_a381::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a381::KERNEL)),

                                    // #[cfg(feature = "c004_a382")]
                                    // "c004_a382" => Some((c004::c004_a382::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a382::KERNEL)),

                                    // #[cfg(feature = "c004_a383")]
                                    // "c004_a383" => Some((c004::c004_a383::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a383::KERNEL)),

                                    // #[cfg(feature = "c004_a384")]
                                    // "c004_a384" => Some((c004::c004_a384::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a384::KERNEL)),

                                    // #[cfg(feature = "c004_a385")]
                                    // "c004_a385" => Some((c004::c004_a385::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a385::KERNEL)),

                                    // #[cfg(feature = "c004_a386")]
                                    // "c004_a386" => Some((c004::c004_a386::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a386::KERNEL)),

                                    // #[cfg(feature = "c004_a387")]
                                    // "c004_a387" => Some((c004::c004_a387::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a387::KERNEL)),

                                    // #[cfg(feature = "c004_a388")]
                                    // "c004_a388" => Some((c004::c004_a388::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a388::KERNEL)),

                                    // #[cfg(feature = "c004_a389")]
                                    // "c004_a389" => Some((c004::c004_a389::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a389::KERNEL)),

                                    // #[cfg(feature = "c004_a390")]
                                    // "c004_a390" => Some((c004::c004_a390::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a390::KERNEL)),

                                    // #[cfg(feature = "c004_a391")]
                                    // "c004_a391" => Some((c004::c004_a391::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a391::KERNEL)),

                                    // #[cfg(feature = "c004_a392")]
                                    // "c004_a392" => Some((c004::c004_a392::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a392::KERNEL)),

                                    // #[cfg(feature = "c004_a393")]
                                    // "c004_a393" => Some((c004::c004_a393::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a393::KERNEL)),

                                    // #[cfg(feature = "c004_a394")]
                                    // "c004_a394" => Some((c004::c004_a394::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a394::KERNEL)),

                                    // #[cfg(feature = "c004_a395")]
                                    // "c004_a395" => Some((c004::c004_a395::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a395::KERNEL)),

                                    // #[cfg(feature = "c004_a396")]
                                    // "c004_a396" => Some((c004::c004_a396::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a396::KERNEL)),

                                    // #[cfg(feature = "c004_a397")]
                                    // "c004_a397" => Some((c004::c004_a397::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a397::KERNEL)),

                                    // #[cfg(feature = "c004_a398")]
                                    // "c004_a398" => Some((c004::c004_a398::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a398::KERNEL)),

                                    // #[cfg(feature = "c004_a399")]
                                    // "c004_a399" => Some((c004::c004_a399::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a399::KERNEL)),

                                    // #[cfg(feature = "c004_a400")]
                                    // "c004_a400" => Some((c004::c004_a400::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a400::KERNEL)),

                                    // #[cfg(feature = "c004_a401")]
                                    // "c004_a401" => Some((c004::c004_a401::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a401::KERNEL)),

                                    // #[cfg(feature = "c004_a402")]
                                    // "c004_a402" => Some((c004::c004_a402::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a402::KERNEL)),

                                    // #[cfg(feature = "c004_a403")]
                                    // "c004_a403" => Some((c004::c004_a403::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a403::KERNEL)),

                                    // #[cfg(feature = "c004_a404")]
                                    // "c004_a404" => Some((c004::c004_a404::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a404::KERNEL)),

                                    // #[cfg(feature = "c004_a405")]
                                    // "c004_a405" => Some((c004::c004_a405::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a405::KERNEL)),

                                    // #[cfg(feature = "c004_a406")]
                                    // "c004_a406" => Some((c004::c004_a406::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a406::KERNEL)),

                                    // #[cfg(feature = "c004_a407")]
                                    // "c004_a407" => Some((c004::c004_a407::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a407::KERNEL)),

                                    // #[cfg(feature = "c004_a408")]
                                    // "c004_a408" => Some((c004::c004_a408::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a408::KERNEL)),

                                    // #[cfg(feature = "c004_a409")]
                                    // "c004_a409" => Some((c004::c004_a409::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a409::KERNEL)),

                                    // #[cfg(feature = "c004_a410")]
                                    // "c004_a410" => Some((c004::c004_a410::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a410::KERNEL)),

                                    // #[cfg(feature = "c004_a411")]
                                    // "c004_a411" => Some((c004::c004_a411::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a411::KERNEL)),

                                    // #[cfg(feature = "c004_a412")]
                                    // "c004_a412" => Some((c004::c004_a412::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a412::KERNEL)),

                                    // #[cfg(feature = "c004_a413")]
                                    // "c004_a413" => Some((c004::c004_a413::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a413::KERNEL)),

                                    // #[cfg(feature = "c004_a414")]
                                    // "c004_a414" => Some((c004::c004_a414::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a414::KERNEL)),

                                    // #[cfg(feature = "c004_a415")]
                                    // "c004_a415" => Some((c004::c004_a415::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a415::KERNEL)),

                                    // #[cfg(feature = "c004_a416")]
                                    // "c004_a416" => Some((c004::c004_a416::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a416::KERNEL)),

                                    // #[cfg(feature = "c004_a417")]
                                    // "c004_a417" => Some((c004::c004_a417::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a417::KERNEL)),

                                    // #[cfg(feature = "c004_a418")]
                                    // "c004_a418" => Some((c004::c004_a418::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a418::KERNEL)),

                                    // #[cfg(feature = "c004_a419")]
                                    // "c004_a419" => Some((c004::c004_a419::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a419::KERNEL)),

                                    // #[cfg(feature = "c004_a420")]
                                    // "c004_a420" => Some((c004::c004_a420::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a420::KERNEL)),

                                    // #[cfg(feature = "c004_a421")]
                                    // "c004_a421" => Some((c004::c004_a421::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a421::KERNEL)),

                                    // #[cfg(feature = "c004_a422")]
                                    // "c004_a422" => Some((c004::c004_a422::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a422::KERNEL)),

                                    // #[cfg(feature = "c004_a423")]
                                    // "c004_a423" => Some((c004::c004_a423::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a423::KERNEL)),

                                    // #[cfg(feature = "c004_a424")]
                                    // "c004_a424" => Some((c004::c004_a424::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a424::KERNEL)),

                                    // #[cfg(feature = "c004_a425")]
                                    // "c004_a425" => Some((c004::c004_a425::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a425::KERNEL)),

                                    // #[cfg(feature = "c004_a426")]
                                    // "c004_a426" => Some((c004::c004_a426::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a426::KERNEL)),

                                    // #[cfg(feature = "c004_a427")]
                                    // "c004_a427" => Some((c004::c004_a427::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a427::KERNEL)),

                                    // #[cfg(feature = "c004_a428")]
                                    // "c004_a428" => Some((c004::c004_a428::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a428::KERNEL)),

                                    // #[cfg(feature = "c004_a429")]
                                    // "c004_a429" => Some((c004::c004_a429::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a429::KERNEL)),

                                    // #[cfg(feature = "c004_a430")]
                                    // "c004_a430" => Some((c004::c004_a430::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a430::KERNEL)),

                                    // #[cfg(feature = "c004_a431")]
                                    // "c004_a431" => Some((c004::c004_a431::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a431::KERNEL)),

                                    // #[cfg(feature = "c004_a432")]
                                    // "c004_a432" => Some((c004::c004_a432::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a432::KERNEL)),

                                    // #[cfg(feature = "c004_a433")]
                                    // "c004_a433" => Some((c004::c004_a433::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a433::KERNEL)),

                                    // #[cfg(feature = "c004_a434")]
                                    // "c004_a434" => Some((c004::c004_a434::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a434::KERNEL)),

                                    // #[cfg(feature = "c004_a435")]
                                    // "c004_a435" => Some((c004::c004_a435::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a435::KERNEL)),

                                    // #[cfg(feature = "c004_a436")]
                                    // "c004_a436" => Some((c004::c004_a436::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a436::KERNEL)),

                                    // #[cfg(feature = "c004_a437")]
                                    // "c004_a437" => Some((c004::c004_a437::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a437::KERNEL)),

                                    // #[cfg(feature = "c004_a438")]
                                    // "c004_a438" => Some((c004::c004_a438::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a438::KERNEL)),

                                    // #[cfg(feature = "c004_a439")]
                                    // "c004_a439" => Some((c004::c004_a439::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a439::KERNEL)),

                                    // #[cfg(feature = "c004_a440")]
                                    // "c004_a440" => Some((c004::c004_a440::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a440::KERNEL)),

                                    // #[cfg(feature = "c004_a441")]
                                    // "c004_a441" => Some((c004::c004_a441::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a441::KERNEL)),

                                    // #[cfg(feature = "c004_a442")]
                                    // "c004_a442" => Some((c004::c004_a442::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a442::KERNEL)),

                                    // #[cfg(feature = "c004_a443")]
                                    // "c004_a443" => Some((c004::c004_a443::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a443::KERNEL)),

                                    // #[cfg(feature = "c004_a444")]
                                    // "c004_a444" => Some((c004::c004_a444::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a444::KERNEL)),

                                    // #[cfg(feature = "c004_a445")]
                                    // "c004_a445" => Some((c004::c004_a445::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a445::KERNEL)),

                                    // #[cfg(feature = "c004_a446")]
                                    // "c004_a446" => Some((c004::c004_a446::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a446::KERNEL)),

                                    // #[cfg(feature = "c004_a447")]
                                    // "c004_a447" => Some((c004::c004_a447::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a447::KERNEL)),

                                    // #[cfg(feature = "c004_a448")]
                                    // "c004_a448" => Some((c004::c004_a448::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a448::KERNEL)),

                                    // #[cfg(feature = "c004_a449")]
                                    // "c004_a449" => Some((c004::c004_a449::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a449::KERNEL)),

                                    // #[cfg(feature = "c004_a450")]
                                    // "c004_a450" => Some((c004::c004_a450::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a450::KERNEL)),

                                    // #[cfg(feature = "c004_a451")]
                                    // "c004_a451" => Some((c004::c004_a451::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a451::KERNEL)),

                                    // #[cfg(feature = "c004_a452")]
                                    // "c004_a452" => Some((c004::c004_a452::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a452::KERNEL)),

                                    // #[cfg(feature = "c004_a453")]
                                    // "c004_a453" => Some((c004::c004_a453::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a453::KERNEL)),

                                    // #[cfg(feature = "c004_a454")]
                                    // "c004_a454" => Some((c004::c004_a454::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a454::KERNEL)),

                                    // #[cfg(feature = "c004_a455")]
                                    // "c004_a455" => Some((c004::c004_a455::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a455::KERNEL)),

                                    // #[cfg(feature = "c004_a456")]
                                    // "c004_a456" => Some((c004::c004_a456::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a456::KERNEL)),

                                    // #[cfg(feature = "c004_a457")]
                                    // "c004_a457" => Some((c004::c004_a457::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a457::KERNEL)),

                                    // #[cfg(feature = "c004_a458")]
                                    // "c004_a458" => Some((c004::c004_a458::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a458::KERNEL)),

                                    // #[cfg(feature = "c004_a459")]
                                    // "c004_a459" => Some((c004::c004_a459::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a459::KERNEL)),

                                    // #[cfg(feature = "c004_a460")]
                                    // "c004_a460" => Some((c004::c004_a460::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a460::KERNEL)),

                                    // #[cfg(feature = "c004_a461")]
                                    // "c004_a461" => Some((c004::c004_a461::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a461::KERNEL)),

                                    // #[cfg(feature = "c004_a462")]
                                    // "c004_a462" => Some((c004::c004_a462::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a462::KERNEL)),

                                    // #[cfg(feature = "c004_a463")]
                                    // "c004_a463" => Some((c004::c004_a463::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a463::KERNEL)),

                                    // #[cfg(feature = "c004_a464")]
                                    // "c004_a464" => Some((c004::c004_a464::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a464::KERNEL)),

                                    // #[cfg(feature = "c004_a465")]
                                    // "c004_a465" => Some((c004::c004_a465::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a465::KERNEL)),

                                    // #[cfg(feature = "c004_a466")]
                                    // "c004_a466" => Some((c004::c004_a466::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a466::KERNEL)),

                                    // #[cfg(feature = "c004_a467")]
                                    // "c004_a467" => Some((c004::c004_a467::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a467::KERNEL)),

                                    // #[cfg(feature = "c004_a468")]
                                    // "c004_a468" => Some((c004::c004_a468::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a468::KERNEL)),

                                    // #[cfg(feature = "c004_a469")]
                                    // "c004_a469" => Some((c004::c004_a469::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a469::KERNEL)),

                                    // #[cfg(feature = "c004_a470")]
                                    // "c004_a470" => Some((c004::c004_a470::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a470::KERNEL)),

                                    // #[cfg(feature = "c004_a471")]
                                    // "c004_a471" => Some((c004::c004_a471::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a471::KERNEL)),

                                    // #[cfg(feature = "c004_a472")]
                                    // "c004_a472" => Some((c004::c004_a472::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a472::KERNEL)),

                                    // #[cfg(feature = "c004_a473")]
                                    // "c004_a473" => Some((c004::c004_a473::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a473::KERNEL)),

                                    // #[cfg(feature = "c004_a474")]
                                    // "c004_a474" => Some((c004::c004_a474::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a474::KERNEL)),

                                    // #[cfg(feature = "c004_a475")]
                                    // "c004_a475" => Some((c004::c004_a475::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a475::KERNEL)),

                                    // #[cfg(feature = "c004_a476")]
                                    // "c004_a476" => Some((c004::c004_a476::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a476::KERNEL)),

                                    // #[cfg(feature = "c004_a477")]
                                    // "c004_a477" => Some((c004::c004_a477::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a477::KERNEL)),

                                    // #[cfg(feature = "c004_a478")]
                                    // "c004_a478" => Some((c004::c004_a478::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a478::KERNEL)),

                                    // #[cfg(feature = "c004_a479")]
                                    // "c004_a479" => Some((c004::c004_a479::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a479::KERNEL)),

                                    // #[cfg(feature = "c004_a480")]
                                    // "c004_a480" => Some((c004::c004_a480::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a480::KERNEL)),

                                    // #[cfg(feature = "c004_a481")]
                                    // "c004_a481" => Some((c004::c004_a481::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a481::KERNEL)),

                                    // #[cfg(feature = "c004_a482")]
                                    // "c004_a482" => Some((c004::c004_a482::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a482::KERNEL)),

                                    // #[cfg(feature = "c004_a483")]
                                    // "c004_a483" => Some((c004::c004_a483::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a483::KERNEL)),

                                    // #[cfg(feature = "c004_a484")]
                                    // "c004_a484" => Some((c004::c004_a484::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a484::KERNEL)),

                                    // #[cfg(feature = "c004_a485")]
                                    // "c004_a485" => Some((c004::c004_a485::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a485::KERNEL)),

                                    // #[cfg(feature = "c004_a486")]
                                    // "c004_a486" => Some((c004::c004_a486::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a486::KERNEL)),

                                    // #[cfg(feature = "c004_a487")]
                                    // "c004_a487" => Some((c004::c004_a487::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a487::KERNEL)),

                                    // #[cfg(feature = "c004_a488")]
                                    // "c004_a488" => Some((c004::c004_a488::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a488::KERNEL)),

                                    // #[cfg(feature = "c004_a489")]
                                    // "c004_a489" => Some((c004::c004_a489::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a489::KERNEL)),

                                    // #[cfg(feature = "c004_a490")]
                                    // "c004_a490" => Some((c004::c004_a490::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a490::KERNEL)),

                                    // #[cfg(feature = "c004_a491")]
                                    // "c004_a491" => Some((c004::c004_a491::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a491::KERNEL)),

                                    // #[cfg(feature = "c004_a492")]
                                    // "c004_a492" => Some((c004::c004_a492::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a492::KERNEL)),

                                    // #[cfg(feature = "c004_a493")]
                                    // "c004_a493" => Some((c004::c004_a493::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a493::KERNEL)),

                                    // #[cfg(feature = "c004_a494")]
                                    // "c004_a494" => Some((c004::c004_a494::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a494::KERNEL)),

                                    // #[cfg(feature = "c004_a495")]
                                    // "c004_a495" => Some((c004::c004_a495::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a495::KERNEL)),

                                    // #[cfg(feature = "c004_a496")]
                                    // "c004_a496" => Some((c004::c004_a496::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a496::KERNEL)),

                                    // #[cfg(feature = "c004_a497")]
                                    // "c004_a497" => Some((c004::c004_a497::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a497::KERNEL)),

                                    // #[cfg(feature = "c004_a498")]
                                    // "c004_a498" => Some((c004::c004_a498::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a498::KERNEL)),

                                    // #[cfg(feature = "c004_a499")]
                                    // "c004_a499" => Some((c004::c004_a499::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a499::KERNEL)),

                                    // #[cfg(feature = "c004_a500")]
                                    // "c004_a500" => Some((c004::c004_a500::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a500::KERNEL)),

                                    // #[cfg(feature = "c004_a501")]
                                    // "c004_a501" => Some((c004::c004_a501::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a501::KERNEL)),

                                    // #[cfg(feature = "c004_a502")]
                                    // "c004_a502" => Some((c004::c004_a502::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a502::KERNEL)),

                                    // #[cfg(feature = "c004_a503")]
                                    // "c004_a503" => Some((c004::c004_a503::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a503::KERNEL)),

                                    // #[cfg(feature = "c004_a504")]
                                    // "c004_a504" => Some((c004::c004_a504::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a504::KERNEL)),

                                    // #[cfg(feature = "c004_a505")]
                                    // "c004_a505" => Some((c004::c004_a505::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a505::KERNEL)),

                                    // #[cfg(feature = "c004_a506")]
                                    // "c004_a506" => Some((c004::c004_a506::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a506::KERNEL)),

                                    // #[cfg(feature = "c004_a507")]
                                    // "c004_a507" => Some((c004::c004_a507::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a507::KERNEL)),

                                    // #[cfg(feature = "c004_a508")]
                                    // "c004_a508" => Some((c004::c004_a508::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a508::KERNEL)),

                                    // #[cfg(feature = "c004_a509")]
                                    // "c004_a509" => Some((c004::c004_a509::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a509::KERNEL)),

                                    // #[cfg(feature = "c004_a510")]
                                    // "c004_a510" => Some((c004::c004_a510::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a510::KERNEL)),

                                    // #[cfg(feature = "c004_a511")]
                                    // "c004_a511" => Some((c004::c004_a511::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a511::KERNEL)),

                                    // #[cfg(feature = "c004_a512")]
                                    // "c004_a512" => Some((c004::c004_a512::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a512::KERNEL)),

                                    // #[cfg(feature = "c004_a513")]
                                    // "c004_a513" => Some((c004::c004_a513::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a513::KERNEL)),

                                    // #[cfg(feature = "c004_a514")]
                                    // "c004_a514" => Some((c004::c004_a514::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a514::KERNEL)),

                                    // #[cfg(feature = "c004_a515")]
                                    // "c004_a515" => Some((c004::c004_a515::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a515::KERNEL)),

                                    // #[cfg(feature = "c004_a516")]
                                    // "c004_a516" => Some((c004::c004_a516::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a516::KERNEL)),

                                    // #[cfg(feature = "c004_a517")]
                                    // "c004_a517" => Some((c004::c004_a517::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a517::KERNEL)),

                                    // #[cfg(feature = "c004_a518")]
                                    // "c004_a518" => Some((c004::c004_a518::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a518::KERNEL)),

                                    // #[cfg(feature = "c004_a519")]
                                    // "c004_a519" => Some((c004::c004_a519::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a519::KERNEL)),

                                    // #[cfg(feature = "c004_a520")]
                                    // "c004_a520" => Some((c004::c004_a520::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a520::KERNEL)),

                                    // #[cfg(feature = "c004_a521")]
                                    // "c004_a521" => Some((c004::c004_a521::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a521::KERNEL)),

                                    // #[cfg(feature = "c004_a522")]
                                    // "c004_a522" => Some((c004::c004_a522::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a522::KERNEL)),

                                    // #[cfg(feature = "c004_a523")]
                                    // "c004_a523" => Some((c004::c004_a523::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a523::KERNEL)),

                                    // #[cfg(feature = "c004_a524")]
                                    // "c004_a524" => Some((c004::c004_a524::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a524::KERNEL)),

                                    // #[cfg(feature = "c004_a525")]
                                    // "c004_a525" => Some((c004::c004_a525::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a525::KERNEL)),

                                    // #[cfg(feature = "c004_a526")]
                                    // "c004_a526" => Some((c004::c004_a526::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a526::KERNEL)),

                                    // #[cfg(feature = "c004_a527")]
                                    // "c004_a527" => Some((c004::c004_a527::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a527::KERNEL)),

                                    // #[cfg(feature = "c004_a528")]
                                    // "c004_a528" => Some((c004::c004_a528::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a528::KERNEL)),

                                    // #[cfg(feature = "c004_a529")]
                                    // "c004_a529" => Some((c004::c004_a529::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a529::KERNEL)),

                                    // #[cfg(feature = "c004_a530")]
                                    // "c004_a530" => Some((c004::c004_a530::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a530::KERNEL)),

                                    // #[cfg(feature = "c004_a531")]
                                    // "c004_a531" => Some((c004::c004_a531::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a531::KERNEL)),

                                    // #[cfg(feature = "c004_a532")]
                                    // "c004_a532" => Some((c004::c004_a532::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a532::KERNEL)),

                                    // #[cfg(feature = "c004_a533")]
                                    // "c004_a533" => Some((c004::c004_a533::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a533::KERNEL)),

                                    // #[cfg(feature = "c004_a534")]
                                    // "c004_a534" => Some((c004::c004_a534::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a534::KERNEL)),

                                    // #[cfg(feature = "c004_a535")]
                                    // "c004_a535" => Some((c004::c004_a535::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a535::KERNEL)),

                                    // #[cfg(feature = "c004_a536")]
                                    // "c004_a536" => Some((c004::c004_a536::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a536::KERNEL)),

                                    // #[cfg(feature = "c004_a537")]
                                    // "c004_a537" => Some((c004::c004_a537::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a537::KERNEL)),

                                    // #[cfg(feature = "c004_a538")]
                                    // "c004_a538" => Some((c004::c004_a538::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a538::KERNEL)),

                                    // #[cfg(feature = "c004_a539")]
                                    // "c004_a539" => Some((c004::c004_a539::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a539::KERNEL)),

                                    // #[cfg(feature = "c004_a540")]
                                    // "c004_a540" => Some((c004::c004_a540::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a540::KERNEL)),

                                    // #[cfg(feature = "c004_a541")]
                                    // "c004_a541" => Some((c004::c004_a541::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a541::KERNEL)),

                                    // #[cfg(feature = "c004_a542")]
                                    // "c004_a542" => Some((c004::c004_a542::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a542::KERNEL)),

                                    // #[cfg(feature = "c004_a543")]
                                    // "c004_a543" => Some((c004::c004_a543::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a543::KERNEL)),

                                    // #[cfg(feature = "c004_a544")]
                                    // "c004_a544" => Some((c004::c004_a544::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a544::KERNEL)),

                                    // #[cfg(feature = "c004_a545")]
                                    // "c004_a545" => Some((c004::c004_a545::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a545::KERNEL)),

                                    // #[cfg(feature = "c004_a546")]
                                    // "c004_a546" => Some((c004::c004_a546::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a546::KERNEL)),

                                    // #[cfg(feature = "c004_a547")]
                                    // "c004_a547" => Some((c004::c004_a547::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a547::KERNEL)),

                                    // #[cfg(feature = "c004_a548")]
                                    // "c004_a548" => Some((c004::c004_a548::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a548::KERNEL)),

                                    // #[cfg(feature = "c004_a549")]
                                    // "c004_a549" => Some((c004::c004_a549::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a549::KERNEL)),

                                    // #[cfg(feature = "c004_a550")]
                                    // "c004_a550" => Some((c004::c004_a550::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a550::KERNEL)),

                                    // #[cfg(feature = "c004_a551")]
                                    // "c004_a551" => Some((c004::c004_a551::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a551::KERNEL)),

                                    // #[cfg(feature = "c004_a552")]
                                    // "c004_a552" => Some((c004::c004_a552::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a552::KERNEL)),

                                    // #[cfg(feature = "c004_a553")]
                                    // "c004_a553" => Some((c004::c004_a553::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a553::KERNEL)),

                                    // #[cfg(feature = "c004_a554")]
                                    // "c004_a554" => Some((c004::c004_a554::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a554::KERNEL)),

                                    // #[cfg(feature = "c004_a555")]
                                    // "c004_a555" => Some((c004::c004_a555::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a555::KERNEL)),

                                    // #[cfg(feature = "c004_a556")]
                                    // "c004_a556" => Some((c004::c004_a556::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a556::KERNEL)),

                                    // #[cfg(feature = "c004_a557")]
                                    // "c004_a557" => Some((c004::c004_a557::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a557::KERNEL)),

                                    // #[cfg(feature = "c004_a558")]
                                    // "c004_a558" => Some((c004::c004_a558::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a558::KERNEL)),

                                    // #[cfg(feature = "c004_a559")]
                                    // "c004_a559" => Some((c004::c004_a559::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a559::KERNEL)),

                                    // #[cfg(feature = "c004_a560")]
                                    // "c004_a560" => Some((c004::c004_a560::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a560::KERNEL)),

                                    // #[cfg(feature = "c004_a561")]
                                    // "c004_a561" => Some((c004::c004_a561::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a561::KERNEL)),

                                    // #[cfg(feature = "c004_a562")]
                                    // "c004_a562" => Some((c004::c004_a562::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a562::KERNEL)),

                                    // #[cfg(feature = "c004_a563")]
                                    // "c004_a563" => Some((c004::c004_a563::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a563::KERNEL)),

                                    // #[cfg(feature = "c004_a564")]
                                    // "c004_a564" => Some((c004::c004_a564::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a564::KERNEL)),

                                    // #[cfg(feature = "c004_a565")]
                                    // "c004_a565" => Some((c004::c004_a565::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a565::KERNEL)),

                                    // #[cfg(feature = "c004_a566")]
                                    // "c004_a566" => Some((c004::c004_a566::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a566::KERNEL)),

                                    // #[cfg(feature = "c004_a567")]
                                    // "c004_a567" => Some((c004::c004_a567::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a567::KERNEL)),

                                    // #[cfg(feature = "c004_a568")]
                                    // "c004_a568" => Some((c004::c004_a568::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a568::KERNEL)),

                                    // #[cfg(feature = "c004_a569")]
                                    // "c004_a569" => Some((c004::c004_a569::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a569::KERNEL)),

                                    // #[cfg(feature = "c004_a570")]
                                    // "c004_a570" => Some((c004::c004_a570::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a570::KERNEL)),

                                    // #[cfg(feature = "c004_a571")]
                                    // "c004_a571" => Some((c004::c004_a571::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a571::KERNEL)),

                                    // #[cfg(feature = "c004_a572")]
                                    // "c004_a572" => Some((c004::c004_a572::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a572::KERNEL)),

                                    // #[cfg(feature = "c004_a573")]
                                    // "c004_a573" => Some((c004::c004_a573::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a573::KERNEL)),

                                    // #[cfg(feature = "c004_a574")]
                                    // "c004_a574" => Some((c004::c004_a574::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a574::KERNEL)),

                                    // #[cfg(feature = "c004_a575")]
                                    // "c004_a575" => Some((c004::c004_a575::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a575::KERNEL)),

                                    // #[cfg(feature = "c004_a576")]
                                    // "c004_a576" => Some((c004::c004_a576::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a576::KERNEL)),

                                    // #[cfg(feature = "c004_a577")]
                                    // "c004_a577" => Some((c004::c004_a577::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a577::KERNEL)),

                                    // #[cfg(feature = "c004_a578")]
                                    // "c004_a578" => Some((c004::c004_a578::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a578::KERNEL)),

                                    // #[cfg(feature = "c004_a579")]
                                    // "c004_a579" => Some((c004::c004_a579::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a579::KERNEL)),

                                    // #[cfg(feature = "c004_a580")]
                                    // "c004_a580" => Some((c004::c004_a580::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a580::KERNEL)),

                                    // #[cfg(feature = "c004_a581")]
                                    // "c004_a581" => Some((c004::c004_a581::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a581::KERNEL)),

                                    // #[cfg(feature = "c004_a582")]
                                    // "c004_a582" => Some((c004::c004_a582::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a582::KERNEL)),

                                    // #[cfg(feature = "c004_a583")]
                                    // "c004_a583" => Some((c004::c004_a583::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a583::KERNEL)),

                                    // #[cfg(feature = "c004_a584")]
                                    // "c004_a584" => Some((c004::c004_a584::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a584::KERNEL)),

                                    // #[cfg(feature = "c004_a585")]
                                    // "c004_a585" => Some((c004::c004_a585::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a585::KERNEL)),

                                    // #[cfg(feature = "c004_a586")]
                                    // "c004_a586" => Some((c004::c004_a586::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a586::KERNEL)),

                                    // #[cfg(feature = "c004_a587")]
                                    // "c004_a587" => Some((c004::c004_a587::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a587::KERNEL)),

                                    // #[cfg(feature = "c004_a588")]
                                    // "c004_a588" => Some((c004::c004_a588::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a588::KERNEL)),

                                    // #[cfg(feature = "c004_a589")]
                                    // "c004_a589" => Some((c004::c004_a589::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a589::KERNEL)),

                                    // #[cfg(feature = "c004_a590")]
                                    // "c004_a590" => Some((c004::c004_a590::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a590::KERNEL)),

                                    // #[cfg(feature = "c004_a591")]
                                    // "c004_a591" => Some((c004::c004_a591::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a591::KERNEL)),

                                    // #[cfg(feature = "c004_a592")]
                                    // "c004_a592" => Some((c004::c004_a592::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a592::KERNEL)),

                                    // #[cfg(feature = "c004_a593")]
                                    // "c004_a593" => Some((c004::c004_a593::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a593::KERNEL)),

                                    // #[cfg(feature = "c004_a594")]
                                    // "c004_a594" => Some((c004::c004_a594::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a594::KERNEL)),

                                    // #[cfg(feature = "c004_a595")]
                                    // "c004_a595" => Some((c004::c004_a595::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a595::KERNEL)),

                                    // #[cfg(feature = "c004_a596")]
                                    // "c004_a596" => Some((c004::c004_a596::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a596::KERNEL)),

                                    // #[cfg(feature = "c004_a597")]
                                    // "c004_a597" => Some((c004::c004_a597::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a597::KERNEL)),

                                    // #[cfg(feature = "c004_a598")]
                                    // "c004_a598" => Some((c004::c004_a598::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a598::KERNEL)),

                                    // #[cfg(feature = "c004_a599")]
                                    // "c004_a599" => Some((c004::c004_a599::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a599::KERNEL)),

                                    // #[cfg(feature = "c004_a600")]
                                    // "c004_a600" => Some((c004::c004_a600::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a600::KERNEL)),

                                    // #[cfg(feature = "c004_a601")]
                                    // "c004_a601" => Some((c004::c004_a601::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a601::KERNEL)),

                                    // #[cfg(feature = "c004_a602")]
                                    // "c004_a602" => Some((c004::c004_a602::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a602::KERNEL)),

                                    // #[cfg(feature = "c004_a603")]
                                    // "c004_a603" => Some((c004::c004_a603::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a603::KERNEL)),

                                    // #[cfg(feature = "c004_a604")]
                                    // "c004_a604" => Some((c004::c004_a604::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a604::KERNEL)),

                                    // #[cfg(feature = "c004_a605")]
                                    // "c004_a605" => Some((c004::c004_a605::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a605::KERNEL)),

                                    // #[cfg(feature = "c004_a606")]
                                    // "c004_a606" => Some((c004::c004_a606::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a606::KERNEL)),

                                    // #[cfg(feature = "c004_a607")]
                                    // "c004_a607" => Some((c004::c004_a607::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a607::KERNEL)),

                                    // #[cfg(feature = "c004_a608")]
                                    // "c004_a608" => Some((c004::c004_a608::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a608::KERNEL)),

                                    // #[cfg(feature = "c004_a609")]
                                    // "c004_a609" => Some((c004::c004_a609::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a609::KERNEL)),

                                    // #[cfg(feature = "c004_a610")]
                                    // "c004_a610" => Some((c004::c004_a610::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a610::KERNEL)),

                                    // #[cfg(feature = "c004_a611")]
                                    // "c004_a611" => Some((c004::c004_a611::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a611::KERNEL)),

                                    // #[cfg(feature = "c004_a612")]
                                    // "c004_a612" => Some((c004::c004_a612::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a612::KERNEL)),

                                    // #[cfg(feature = "c004_a613")]
                                    // "c004_a613" => Some((c004::c004_a613::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a613::KERNEL)),

                                    // #[cfg(feature = "c004_a614")]
                                    // "c004_a614" => Some((c004::c004_a614::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a614::KERNEL)),

                                    // #[cfg(feature = "c004_a615")]
                                    // "c004_a615" => Some((c004::c004_a615::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a615::KERNEL)),

                                    // #[cfg(feature = "c004_a616")]
                                    // "c004_a616" => Some((c004::c004_a616::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a616::KERNEL)),

                                    // #[cfg(feature = "c004_a617")]
                                    // "c004_a617" => Some((c004::c004_a617::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a617::KERNEL)),

                                    // #[cfg(feature = "c004_a618")]
                                    // "c004_a618" => Some((c004::c004_a618::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a618::KERNEL)),

                                    // #[cfg(feature = "c004_a619")]
                                    // "c004_a619" => Some((c004::c004_a619::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a619::KERNEL)),

                                    // #[cfg(feature = "c004_a620")]
                                    // "c004_a620" => Some((c004::c004_a620::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a620::KERNEL)),

                                    // #[cfg(feature = "c004_a621")]
                                    // "c004_a621" => Some((c004::c004_a621::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a621::KERNEL)),

                                    // #[cfg(feature = "c004_a622")]
                                    // "c004_a622" => Some((c004::c004_a622::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a622::KERNEL)),

                                    // #[cfg(feature = "c004_a623")]
                                    // "c004_a623" => Some((c004::c004_a623::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a623::KERNEL)),

                                    // #[cfg(feature = "c004_a624")]
                                    // "c004_a624" => Some((c004::c004_a624::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a624::KERNEL)),

                                    // #[cfg(feature = "c004_a625")]
                                    // "c004_a625" => Some((c004::c004_a625::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a625::KERNEL)),

                                    // #[cfg(feature = "c004_a626")]
                                    // "c004_a626" => Some((c004::c004_a626::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a626::KERNEL)),

                                    // #[cfg(feature = "c004_a627")]
                                    // "c004_a627" => Some((c004::c004_a627::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a627::KERNEL)),

                                    // #[cfg(feature = "c004_a628")]
                                    // "c004_a628" => Some((c004::c004_a628::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a628::KERNEL)),

                                    // #[cfg(feature = "c004_a629")]
                                    // "c004_a629" => Some((c004::c004_a629::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a629::KERNEL)),

                                    // #[cfg(feature = "c004_a630")]
                                    // "c004_a630" => Some((c004::c004_a630::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a630::KERNEL)),

                                    // #[cfg(feature = "c004_a631")]
                                    // "c004_a631" => Some((c004::c004_a631::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a631::KERNEL)),

                                    // #[cfg(feature = "c004_a632")]
                                    // "c004_a632" => Some((c004::c004_a632::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a632::KERNEL)),

                                    // #[cfg(feature = "c004_a633")]
                                    // "c004_a633" => Some((c004::c004_a633::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a633::KERNEL)),

                                    // #[cfg(feature = "c004_a634")]
                                    // "c004_a634" => Some((c004::c004_a634::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a634::KERNEL)),

                                    // #[cfg(feature = "c004_a635")]
                                    // "c004_a635" => Some((c004::c004_a635::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a635::KERNEL)),

                                    // #[cfg(feature = "c004_a636")]
                                    // "c004_a636" => Some((c004::c004_a636::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a636::KERNEL)),

                                    // #[cfg(feature = "c004_a637")]
                                    // "c004_a637" => Some((c004::c004_a637::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a637::KERNEL)),

                                    // #[cfg(feature = "c004_a638")]
                                    // "c004_a638" => Some((c004::c004_a638::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a638::KERNEL)),

                                    // #[cfg(feature = "c004_a639")]
                                    // "c004_a639" => Some((c004::c004_a639::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a639::KERNEL)),

                                    // #[cfg(feature = "c004_a640")]
                                    // "c004_a640" => Some((c004::c004_a640::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a640::KERNEL)),

                                    // #[cfg(feature = "c004_a641")]
                                    // "c004_a641" => Some((c004::c004_a641::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a641::KERNEL)),

                                    // #[cfg(feature = "c004_a642")]
                                    // "c004_a642" => Some((c004::c004_a642::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a642::KERNEL)),

                                    // #[cfg(feature = "c004_a643")]
                                    // "c004_a643" => Some((c004::c004_a643::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a643::KERNEL)),

                                    // #[cfg(feature = "c004_a644")]
                                    // "c004_a644" => Some((c004::c004_a644::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a644::KERNEL)),

                                    // #[cfg(feature = "c004_a645")]
                                    // "c004_a645" => Some((c004::c004_a645::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a645::KERNEL)),

                                    // #[cfg(feature = "c004_a646")]
                                    // "c004_a646" => Some((c004::c004_a646::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a646::KERNEL)),

                                    // #[cfg(feature = "c004_a647")]
                                    // "c004_a647" => Some((c004::c004_a647::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a647::KERNEL)),

                                    // #[cfg(feature = "c004_a648")]
                                    // "c004_a648" => Some((c004::c004_a648::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a648::KERNEL)),

                                    // #[cfg(feature = "c004_a649")]
                                    // "c004_a649" => Some((c004::c004_a649::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a649::KERNEL)),

                                    // #[cfg(feature = "c004_a650")]
                                    // "c004_a650" => Some((c004::c004_a650::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a650::KERNEL)),

                                    // #[cfg(feature = "c004_a651")]
                                    // "c004_a651" => Some((c004::c004_a651::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a651::KERNEL)),

                                    // #[cfg(feature = "c004_a652")]
                                    // "c004_a652" => Some((c004::c004_a652::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a652::KERNEL)),

                                    // #[cfg(feature = "c004_a653")]
                                    // "c004_a653" => Some((c004::c004_a653::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a653::KERNEL)),

                                    // #[cfg(feature = "c004_a654")]
                                    // "c004_a654" => Some((c004::c004_a654::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a654::KERNEL)),

                                    // #[cfg(feature = "c004_a655")]
                                    // "c004_a655" => Some((c004::c004_a655::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a655::KERNEL)),

                                    // #[cfg(feature = "c004_a656")]
                                    // "c004_a656" => Some((c004::c004_a656::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a656::KERNEL)),

                                    // #[cfg(feature = "c004_a657")]
                                    // "c004_a657" => Some((c004::c004_a657::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a657::KERNEL)),

                                    // #[cfg(feature = "c004_a658")]
                                    // "c004_a658" => Some((c004::c004_a658::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a658::KERNEL)),

                                    // #[cfg(feature = "c004_a659")]
                                    // "c004_a659" => Some((c004::c004_a659::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a659::KERNEL)),

                                    // #[cfg(feature = "c004_a660")]
                                    // "c004_a660" => Some((c004::c004_a660::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a660::KERNEL)),

                                    // #[cfg(feature = "c004_a661")]
                                    // "c004_a661" => Some((c004::c004_a661::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a661::KERNEL)),

                                    // #[cfg(feature = "c004_a662")]
                                    // "c004_a662" => Some((c004::c004_a662::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a662::KERNEL)),

                                    // #[cfg(feature = "c004_a663")]
                                    // "c004_a663" => Some((c004::c004_a663::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a663::KERNEL)),

                                    // #[cfg(feature = "c004_a664")]
                                    // "c004_a664" => Some((c004::c004_a664::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a664::KERNEL)),

                                    // #[cfg(feature = "c004_a665")]
                                    // "c004_a665" => Some((c004::c004_a665::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a665::KERNEL)),

                                    // #[cfg(feature = "c004_a666")]
                                    // "c004_a666" => Some((c004::c004_a666::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a666::KERNEL)),

                                    // #[cfg(feature = "c004_a667")]
                                    // "c004_a667" => Some((c004::c004_a667::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a667::KERNEL)),

                                    // #[cfg(feature = "c004_a668")]
                                    // "c004_a668" => Some((c004::c004_a668::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a668::KERNEL)),

                                    // #[cfg(feature = "c004_a669")]
                                    // "c004_a669" => Some((c004::c004_a669::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a669::KERNEL)),

                                    // #[cfg(feature = "c004_a670")]
                                    // "c004_a670" => Some((c004::c004_a670::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a670::KERNEL)),

                                    // #[cfg(feature = "c004_a671")]
                                    // "c004_a671" => Some((c004::c004_a671::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a671::KERNEL)),

                                    // #[cfg(feature = "c004_a672")]
                                    // "c004_a672" => Some((c004::c004_a672::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a672::KERNEL)),

                                    // #[cfg(feature = "c004_a673")]
                                    // "c004_a673" => Some((c004::c004_a673::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a673::KERNEL)),

                                    // #[cfg(feature = "c004_a674")]
                                    // "c004_a674" => Some((c004::c004_a674::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a674::KERNEL)),

                                    // #[cfg(feature = "c004_a675")]
                                    // "c004_a675" => Some((c004::c004_a675::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a675::KERNEL)),

                                    // #[cfg(feature = "c004_a676")]
                                    // "c004_a676" => Some((c004::c004_a676::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a676::KERNEL)),

                                    // #[cfg(feature = "c004_a677")]
                                    // "c004_a677" => Some((c004::c004_a677::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a677::KERNEL)),

                                    // #[cfg(feature = "c004_a678")]
                                    // "c004_a678" => Some((c004::c004_a678::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a678::KERNEL)),

                                    // #[cfg(feature = "c004_a679")]
                                    // "c004_a679" => Some((c004::c004_a679::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a679::KERNEL)),

                                    // #[cfg(feature = "c004_a680")]
                                    // "c004_a680" => Some((c004::c004_a680::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a680::KERNEL)),

                                    // #[cfg(feature = "c004_a681")]
                                    // "c004_a681" => Some((c004::c004_a681::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a681::KERNEL)),

                                    // #[cfg(feature = "c004_a682")]
                                    // "c004_a682" => Some((c004::c004_a682::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a682::KERNEL)),

                                    // #[cfg(feature = "c004_a683")]
                                    // "c004_a683" => Some((c004::c004_a683::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a683::KERNEL)),

                                    // #[cfg(feature = "c004_a684")]
                                    // "c004_a684" => Some((c004::c004_a684::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a684::KERNEL)),

                                    // #[cfg(feature = "c004_a685")]
                                    // "c004_a685" => Some((c004::c004_a685::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a685::KERNEL)),

                                    // #[cfg(feature = "c004_a686")]
                                    // "c004_a686" => Some((c004::c004_a686::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a686::KERNEL)),

                                    // #[cfg(feature = "c004_a687")]
                                    // "c004_a687" => Some((c004::c004_a687::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a687::KERNEL)),

                                    // #[cfg(feature = "c004_a688")]
                                    // "c004_a688" => Some((c004::c004_a688::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a688::KERNEL)),

                                    // #[cfg(feature = "c004_a689")]
                                    // "c004_a689" => Some((c004::c004_a689::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a689::KERNEL)),

                                    // #[cfg(feature = "c004_a690")]
                                    // "c004_a690" => Some((c004::c004_a690::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a690::KERNEL)),

                                    // #[cfg(feature = "c004_a691")]
                                    // "c004_a691" => Some((c004::c004_a691::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a691::KERNEL)),

                                    // #[cfg(feature = "c004_a692")]
                                    // "c004_a692" => Some((c004::c004_a692::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a692::KERNEL)),

                                    // #[cfg(feature = "c004_a693")]
                                    // "c004_a693" => Some((c004::c004_a693::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a693::KERNEL)),

                                    // #[cfg(feature = "c004_a694")]
                                    // "c004_a694" => Some((c004::c004_a694::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a694::KERNEL)),

                                    // #[cfg(feature = "c004_a695")]
                                    // "c004_a695" => Some((c004::c004_a695::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a695::KERNEL)),

                                    // #[cfg(feature = "c004_a696")]
                                    // "c004_a696" => Some((c004::c004_a696::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a696::KERNEL)),

                                    // #[cfg(feature = "c004_a697")]
                                    // "c004_a697" => Some((c004::c004_a697::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a697::KERNEL)),

                                    // #[cfg(feature = "c004_a698")]
                                    // "c004_a698" => Some((c004::c004_a698::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a698::KERNEL)),

                                    // #[cfg(feature = "c004_a699")]
                                    // "c004_a699" => Some((c004::c004_a699::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a699::KERNEL)),

                                    // #[cfg(feature = "c004_a700")]
                                    // "c004_a700" => Some((c004::c004_a700::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a700::KERNEL)),

                                    // #[cfg(feature = "c004_a701")]
                                    // "c004_a701" => Some((c004::c004_a701::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a701::KERNEL)),

                                    // #[cfg(feature = "c004_a702")]
                                    // "c004_a702" => Some((c004::c004_a702::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a702::KERNEL)),

                                    // #[cfg(feature = "c004_a703")]
                                    // "c004_a703" => Some((c004::c004_a703::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a703::KERNEL)),

                                    // #[cfg(feature = "c004_a704")]
                                    // "c004_a704" => Some((c004::c004_a704::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a704::KERNEL)),

                                    // #[cfg(feature = "c004_a705")]
                                    // "c004_a705" => Some((c004::c004_a705::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a705::KERNEL)),

                                    // #[cfg(feature = "c004_a706")]
                                    // "c004_a706" => Some((c004::c004_a706::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a706::KERNEL)),

                                    // #[cfg(feature = "c004_a707")]
                                    // "c004_a707" => Some((c004::c004_a707::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a707::KERNEL)),

                                    // #[cfg(feature = "c004_a708")]
                                    // "c004_a708" => Some((c004::c004_a708::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a708::KERNEL)),

                                    // #[cfg(feature = "c004_a709")]
                                    // "c004_a709" => Some((c004::c004_a709::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a709::KERNEL)),

                                    // #[cfg(feature = "c004_a710")]
                                    // "c004_a710" => Some((c004::c004_a710::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a710::KERNEL)),

                                    // #[cfg(feature = "c004_a711")]
                                    // "c004_a711" => Some((c004::c004_a711::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a711::KERNEL)),

                                    // #[cfg(feature = "c004_a712")]
                                    // "c004_a712" => Some((c004::c004_a712::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a712::KERNEL)),

                                    // #[cfg(feature = "c004_a713")]
                                    // "c004_a713" => Some((c004::c004_a713::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a713::KERNEL)),

                                    // #[cfg(feature = "c004_a714")]
                                    // "c004_a714" => Some((c004::c004_a714::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a714::KERNEL)),

                                    // #[cfg(feature = "c004_a715")]
                                    // "c004_a715" => Some((c004::c004_a715::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a715::KERNEL)),

                                    // #[cfg(feature = "c004_a716")]
                                    // "c004_a716" => Some((c004::c004_a716::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a716::KERNEL)),

                                    // #[cfg(feature = "c004_a717")]
                                    // "c004_a717" => Some((c004::c004_a717::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a717::KERNEL)),

                                    // #[cfg(feature = "c004_a718")]
                                    // "c004_a718" => Some((c004::c004_a718::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a718::KERNEL)),

                                    // #[cfg(feature = "c004_a719")]
                                    // "c004_a719" => Some((c004::c004_a719::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a719::KERNEL)),

                                    // #[cfg(feature = "c004_a720")]
                                    // "c004_a720" => Some((c004::c004_a720::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a720::KERNEL)),

                                    // #[cfg(feature = "c004_a721")]
                                    // "c004_a721" => Some((c004::c004_a721::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a721::KERNEL)),

                                    // #[cfg(feature = "c004_a722")]
                                    // "c004_a722" => Some((c004::c004_a722::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a722::KERNEL)),

                                    // #[cfg(feature = "c004_a723")]
                                    // "c004_a723" => Some((c004::c004_a723::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a723::KERNEL)),

                                    // #[cfg(feature = "c004_a724")]
                                    // "c004_a724" => Some((c004::c004_a724::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a724::KERNEL)),

                                    // #[cfg(feature = "c004_a725")]
                                    // "c004_a725" => Some((c004::c004_a725::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a725::KERNEL)),

                                    // #[cfg(feature = "c004_a726")]
                                    // "c004_a726" => Some((c004::c004_a726::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a726::KERNEL)),

                                    // #[cfg(feature = "c004_a727")]
                                    // "c004_a727" => Some((c004::c004_a727::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a727::KERNEL)),

                                    // #[cfg(feature = "c004_a728")]
                                    // "c004_a728" => Some((c004::c004_a728::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a728::KERNEL)),

                                    // #[cfg(feature = "c004_a729")]
                                    // "c004_a729" => Some((c004::c004_a729::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a729::KERNEL)),

                                    // #[cfg(feature = "c004_a730")]
                                    // "c004_a730" => Some((c004::c004_a730::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a730::KERNEL)),

                                    // #[cfg(feature = "c004_a731")]
                                    // "c004_a731" => Some((c004::c004_a731::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a731::KERNEL)),

                                    // #[cfg(feature = "c004_a732")]
                                    // "c004_a732" => Some((c004::c004_a732::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a732::KERNEL)),

                                    // #[cfg(feature = "c004_a733")]
                                    // "c004_a733" => Some((c004::c004_a733::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a733::KERNEL)),

                                    // #[cfg(feature = "c004_a734")]
                                    // "c004_a734" => Some((c004::c004_a734::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a734::KERNEL)),

                                    // #[cfg(feature = "c004_a735")]
                                    // "c004_a735" => Some((c004::c004_a735::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a735::KERNEL)),

                                    // #[cfg(feature = "c004_a736")]
                                    // "c004_a736" => Some((c004::c004_a736::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a736::KERNEL)),

                                    // #[cfg(feature = "c004_a737")]
                                    // "c004_a737" => Some((c004::c004_a737::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a737::KERNEL)),

                                    // #[cfg(feature = "c004_a738")]
                                    // "c004_a738" => Some((c004::c004_a738::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a738::KERNEL)),

                                    // #[cfg(feature = "c004_a739")]
                                    // "c004_a739" => Some((c004::c004_a739::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a739::KERNEL)),

                                    // #[cfg(feature = "c004_a740")]
                                    // "c004_a740" => Some((c004::c004_a740::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a740::KERNEL)),

                                    // #[cfg(feature = "c004_a741")]
                                    // "c004_a741" => Some((c004::c004_a741::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a741::KERNEL)),

                                    // #[cfg(feature = "c004_a742")]
                                    // "c004_a742" => Some((c004::c004_a742::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a742::KERNEL)),

                                    // #[cfg(feature = "c004_a743")]
                                    // "c004_a743" => Some((c004::c004_a743::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a743::KERNEL)),

                                    // #[cfg(feature = "c004_a744")]
                                    // "c004_a744" => Some((c004::c004_a744::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a744::KERNEL)),

                                    // #[cfg(feature = "c004_a745")]
                                    // "c004_a745" => Some((c004::c004_a745::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a745::KERNEL)),

                                    // #[cfg(feature = "c004_a746")]
                                    // "c004_a746" => Some((c004::c004_a746::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a746::KERNEL)),

                                    // #[cfg(feature = "c004_a747")]
                                    // "c004_a747" => Some((c004::c004_a747::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a747::KERNEL)),

                                    // #[cfg(feature = "c004_a748")]
                                    // "c004_a748" => Some((c004::c004_a748::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a748::KERNEL)),

                                    // #[cfg(feature = "c004_a749")]
                                    // "c004_a749" => Some((c004::c004_a749::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a749::KERNEL)),

                                    // #[cfg(feature = "c004_a750")]
                                    // "c004_a750" => Some((c004::c004_a750::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a750::KERNEL)),

                                    // #[cfg(feature = "c004_a751")]
                                    // "c004_a751" => Some((c004::c004_a751::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a751::KERNEL)),

                                    // #[cfg(feature = "c004_a752")]
                                    // "c004_a752" => Some((c004::c004_a752::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a752::KERNEL)),

                                    // #[cfg(feature = "c004_a753")]
                                    // "c004_a753" => Some((c004::c004_a753::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a753::KERNEL)),

                                    // #[cfg(feature = "c004_a754")]
                                    // "c004_a754" => Some((c004::c004_a754::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a754::KERNEL)),

                                    // #[cfg(feature = "c004_a755")]
                                    // "c004_a755" => Some((c004::c004_a755::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a755::KERNEL)),

                                    // #[cfg(feature = "c004_a756")]
                                    // "c004_a756" => Some((c004::c004_a756::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a756::KERNEL)),

                                    // #[cfg(feature = "c004_a757")]
                                    // "c004_a757" => Some((c004::c004_a757::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a757::KERNEL)),

                                    // #[cfg(feature = "c004_a758")]
                                    // "c004_a758" => Some((c004::c004_a758::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a758::KERNEL)),

                                    // #[cfg(feature = "c004_a759")]
                                    // "c004_a759" => Some((c004::c004_a759::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a759::KERNEL)),

                                    // #[cfg(feature = "c004_a760")]
                                    // "c004_a760" => Some((c004::c004_a760::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a760::KERNEL)),

                                    // #[cfg(feature = "c004_a761")]
                                    // "c004_a761" => Some((c004::c004_a761::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a761::KERNEL)),

                                    // #[cfg(feature = "c004_a762")]
                                    // "c004_a762" => Some((c004::c004_a762::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a762::KERNEL)),

                                    // #[cfg(feature = "c004_a763")]
                                    // "c004_a763" => Some((c004::c004_a763::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a763::KERNEL)),

                                    // #[cfg(feature = "c004_a764")]
                                    // "c004_a764" => Some((c004::c004_a764::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a764::KERNEL)),

                                    // #[cfg(feature = "c004_a765")]
                                    // "c004_a765" => Some((c004::c004_a765::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a765::KERNEL)),

                                    // #[cfg(feature = "c004_a766")]
                                    // "c004_a766" => Some((c004::c004_a766::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a766::KERNEL)),

                                    // #[cfg(feature = "c004_a767")]
                                    // "c004_a767" => Some((c004::c004_a767::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a767::KERNEL)),

                                    // #[cfg(feature = "c004_a768")]
                                    // "c004_a768" => Some((c004::c004_a768::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a768::KERNEL)),

                                    // #[cfg(feature = "c004_a769")]
                                    // "c004_a769" => Some((c004::c004_a769::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a769::KERNEL)),

                                    // #[cfg(feature = "c004_a770")]
                                    // "c004_a770" => Some((c004::c004_a770::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a770::KERNEL)),

                                    // #[cfg(feature = "c004_a771")]
                                    // "c004_a771" => Some((c004::c004_a771::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a771::KERNEL)),

                                    // #[cfg(feature = "c004_a772")]
                                    // "c004_a772" => Some((c004::c004_a772::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a772::KERNEL)),

                                    // #[cfg(feature = "c004_a773")]
                                    // "c004_a773" => Some((c004::c004_a773::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a773::KERNEL)),

                                    // #[cfg(feature = "c004_a774")]
                                    // "c004_a774" => Some((c004::c004_a774::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a774::KERNEL)),

                                    // #[cfg(feature = "c004_a775")]
                                    // "c004_a775" => Some((c004::c004_a775::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a775::KERNEL)),

                                    // #[cfg(feature = "c004_a776")]
                                    // "c004_a776" => Some((c004::c004_a776::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a776::KERNEL)),

                                    // #[cfg(feature = "c004_a777")]
                                    // "c004_a777" => Some((c004::c004_a777::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a777::KERNEL)),

                                    // #[cfg(feature = "c004_a778")]
                                    // "c004_a778" => Some((c004::c004_a778::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a778::KERNEL)),

                                    // #[cfg(feature = "c004_a779")]
                                    // "c004_a779" => Some((c004::c004_a779::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a779::KERNEL)),

                                    // #[cfg(feature = "c004_a780")]
                                    // "c004_a780" => Some((c004::c004_a780::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a780::KERNEL)),

                                    // #[cfg(feature = "c004_a781")]
                                    // "c004_a781" => Some((c004::c004_a781::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a781::KERNEL)),

                                    // #[cfg(feature = "c004_a782")]
                                    // "c004_a782" => Some((c004::c004_a782::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a782::KERNEL)),

                                    // #[cfg(feature = "c004_a783")]
                                    // "c004_a783" => Some((c004::c004_a783::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a783::KERNEL)),

                                    // #[cfg(feature = "c004_a784")]
                                    // "c004_a784" => Some((c004::c004_a784::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a784::KERNEL)),

                                    // #[cfg(feature = "c004_a785")]
                                    // "c004_a785" => Some((c004::c004_a785::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a785::KERNEL)),

                                    // #[cfg(feature = "c004_a786")]
                                    // "c004_a786" => Some((c004::c004_a786::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a786::KERNEL)),

                                    // #[cfg(feature = "c004_a787")]
                                    // "c004_a787" => Some((c004::c004_a787::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a787::KERNEL)),

                                    // #[cfg(feature = "c004_a788")]
                                    // "c004_a788" => Some((c004::c004_a788::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a788::KERNEL)),

                                    // #[cfg(feature = "c004_a789")]
                                    // "c004_a789" => Some((c004::c004_a789::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a789::KERNEL)),

                                    // #[cfg(feature = "c004_a790")]
                                    // "c004_a790" => Some((c004::c004_a790::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a790::KERNEL)),

                                    // #[cfg(feature = "c004_a791")]
                                    // "c004_a791" => Some((c004::c004_a791::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a791::KERNEL)),

                                    // #[cfg(feature = "c004_a792")]
                                    // "c004_a792" => Some((c004::c004_a792::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a792::KERNEL)),

                                    // #[cfg(feature = "c004_a793")]
                                    // "c004_a793" => Some((c004::c004_a793::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a793::KERNEL)),

                                    // #[cfg(feature = "c004_a794")]
                                    // "c004_a794" => Some((c004::c004_a794::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a794::KERNEL)),

                                    // #[cfg(feature = "c004_a795")]
                                    // "c004_a795" => Some((c004::c004_a795::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a795::KERNEL)),

                                    // #[cfg(feature = "c004_a796")]
                                    // "c004_a796" => Some((c004::c004_a796::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a796::KERNEL)),

                                    // #[cfg(feature = "c004_a797")]
                                    // "c004_a797" => Some((c004::c004_a797::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a797::KERNEL)),

                                    // #[cfg(feature = "c004_a798")]
                                    // "c004_a798" => Some((c004::c004_a798::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a798::KERNEL)),

                                    // #[cfg(feature = "c004_a799")]
                                    // "c004_a799" => Some((c004::c004_a799::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a799::KERNEL)),

                                    // #[cfg(feature = "c004_a800")]
                                    // "c004_a800" => Some((c004::c004_a800::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a800::KERNEL)),

                                    // #[cfg(feature = "c004_a801")]
                                    // "c004_a801" => Some((c004::c004_a801::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a801::KERNEL)),

                                    // #[cfg(feature = "c004_a802")]
                                    // "c004_a802" => Some((c004::c004_a802::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a802::KERNEL)),

                                    // #[cfg(feature = "c004_a803")]
                                    // "c004_a803" => Some((c004::c004_a803::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a803::KERNEL)),

                                    // #[cfg(feature = "c004_a804")]
                                    // "c004_a804" => Some((c004::c004_a804::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a804::KERNEL)),

                                    // #[cfg(feature = "c004_a805")]
                                    // "c004_a805" => Some((c004::c004_a805::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a805::KERNEL)),

                                    // #[cfg(feature = "c004_a806")]
                                    // "c004_a806" => Some((c004::c004_a806::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a806::KERNEL)),

                                    // #[cfg(feature = "c004_a807")]
                                    // "c004_a807" => Some((c004::c004_a807::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a807::KERNEL)),

                                    // #[cfg(feature = "c004_a808")]
                                    // "c004_a808" => Some((c004::c004_a808::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a808::KERNEL)),

                                    // #[cfg(feature = "c004_a809")]
                                    // "c004_a809" => Some((c004::c004_a809::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a809::KERNEL)),

                                    // #[cfg(feature = "c004_a810")]
                                    // "c004_a810" => Some((c004::c004_a810::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a810::KERNEL)),

                                    // #[cfg(feature = "c004_a811")]
                                    // "c004_a811" => Some((c004::c004_a811::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a811::KERNEL)),

                                    // #[cfg(feature = "c004_a812")]
                                    // "c004_a812" => Some((c004::c004_a812::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a812::KERNEL)),

                                    // #[cfg(feature = "c004_a813")]
                                    // "c004_a813" => Some((c004::c004_a813::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a813::KERNEL)),

                                    // #[cfg(feature = "c004_a814")]
                                    // "c004_a814" => Some((c004::c004_a814::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a814::KERNEL)),

                                    // #[cfg(feature = "c004_a815")]
                                    // "c004_a815" => Some((c004::c004_a815::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a815::KERNEL)),

                                    // #[cfg(feature = "c004_a816")]
                                    // "c004_a816" => Some((c004::c004_a816::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a816::KERNEL)),

                                    // #[cfg(feature = "c004_a817")]
                                    // "c004_a817" => Some((c004::c004_a817::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a817::KERNEL)),

                                    // #[cfg(feature = "c004_a818")]
                                    // "c004_a818" => Some((c004::c004_a818::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a818::KERNEL)),

                                    // #[cfg(feature = "c004_a819")]
                                    // "c004_a819" => Some((c004::c004_a819::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a819::KERNEL)),

                                    // #[cfg(feature = "c004_a820")]
                                    // "c004_a820" => Some((c004::c004_a820::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a820::KERNEL)),

                                    // #[cfg(feature = "c004_a821")]
                                    // "c004_a821" => Some((c004::c004_a821::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a821::KERNEL)),

                                    // #[cfg(feature = "c004_a822")]
                                    // "c004_a822" => Some((c004::c004_a822::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a822::KERNEL)),

                                    // #[cfg(feature = "c004_a823")]
                                    // "c004_a823" => Some((c004::c004_a823::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a823::KERNEL)),

                                    // #[cfg(feature = "c004_a824")]
                                    // "c004_a824" => Some((c004::c004_a824::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a824::KERNEL)),

                                    // #[cfg(feature = "c004_a825")]
                                    // "c004_a825" => Some((c004::c004_a825::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a825::KERNEL)),

                                    // #[cfg(feature = "c004_a826")]
                                    // "c004_a826" => Some((c004::c004_a826::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a826::KERNEL)),

                                    // #[cfg(feature = "c004_a827")]
                                    // "c004_a827" => Some((c004::c004_a827::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a827::KERNEL)),

                                    // #[cfg(feature = "c004_a828")]
                                    // "c004_a828" => Some((c004::c004_a828::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a828::KERNEL)),

                                    // #[cfg(feature = "c004_a829")]
                                    // "c004_a829" => Some((c004::c004_a829::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a829::KERNEL)),

                                    // #[cfg(feature = "c004_a830")]
                                    // "c004_a830" => Some((c004::c004_a830::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a830::KERNEL)),

                                    // #[cfg(feature = "c004_a831")]
                                    // "c004_a831" => Some((c004::c004_a831::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a831::KERNEL)),

                                    // #[cfg(feature = "c004_a832")]
                                    // "c004_a832" => Some((c004::c004_a832::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a832::KERNEL)),

                                    // #[cfg(feature = "c004_a833")]
                                    // "c004_a833" => Some((c004::c004_a833::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a833::KERNEL)),

                                    // #[cfg(feature = "c004_a834")]
                                    // "c004_a834" => Some((c004::c004_a834::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a834::KERNEL)),

                                    // #[cfg(feature = "c004_a835")]
                                    // "c004_a835" => Some((c004::c004_a835::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a835::KERNEL)),

                                    // #[cfg(feature = "c004_a836")]
                                    // "c004_a836" => Some((c004::c004_a836::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a836::KERNEL)),

                                    // #[cfg(feature = "c004_a837")]
                                    // "c004_a837" => Some((c004::c004_a837::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a837::KERNEL)),

                                    // #[cfg(feature = "c004_a838")]
                                    // "c004_a838" => Some((c004::c004_a838::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a838::KERNEL)),

                                    // #[cfg(feature = "c004_a839")]
                                    // "c004_a839" => Some((c004::c004_a839::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a839::KERNEL)),

                                    // #[cfg(feature = "c004_a840")]
                                    // "c004_a840" => Some((c004::c004_a840::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a840::KERNEL)),

                                    // #[cfg(feature = "c004_a841")]
                                    // "c004_a841" => Some((c004::c004_a841::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a841::KERNEL)),

                                    // #[cfg(feature = "c004_a842")]
                                    // "c004_a842" => Some((c004::c004_a842::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a842::KERNEL)),

                                    // #[cfg(feature = "c004_a843")]
                                    // "c004_a843" => Some((c004::c004_a843::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a843::KERNEL)),

                                    // #[cfg(feature = "c004_a844")]
                                    // "c004_a844" => Some((c004::c004_a844::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a844::KERNEL)),

                                    // #[cfg(feature = "c004_a845")]
                                    // "c004_a845" => Some((c004::c004_a845::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a845::KERNEL)),

                                    // #[cfg(feature = "c004_a846")]
                                    // "c004_a846" => Some((c004::c004_a846::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a846::KERNEL)),

                                    // #[cfg(feature = "c004_a847")]
                                    // "c004_a847" => Some((c004::c004_a847::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a847::KERNEL)),

                                    // #[cfg(feature = "c004_a848")]
                                    // "c004_a848" => Some((c004::c004_a848::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a848::KERNEL)),

                                    // #[cfg(feature = "c004_a849")]
                                    // "c004_a849" => Some((c004::c004_a849::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a849::KERNEL)),

                                    // #[cfg(feature = "c004_a850")]
                                    // "c004_a850" => Some((c004::c004_a850::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a850::KERNEL)),

                                    // #[cfg(feature = "c004_a851")]
                                    // "c004_a851" => Some((c004::c004_a851::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a851::KERNEL)),

                                    // #[cfg(feature = "c004_a852")]
                                    // "c004_a852" => Some((c004::c004_a852::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a852::KERNEL)),

                                    // #[cfg(feature = "c004_a853")]
                                    // "c004_a853" => Some((c004::c004_a853::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a853::KERNEL)),

                                    // #[cfg(feature = "c004_a854")]
                                    // "c004_a854" => Some((c004::c004_a854::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a854::KERNEL)),

                                    // #[cfg(feature = "c004_a855")]
                                    // "c004_a855" => Some((c004::c004_a855::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a855::KERNEL)),

                                    // #[cfg(feature = "c004_a856")]
                                    // "c004_a856" => Some((c004::c004_a856::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a856::KERNEL)),

                                    // #[cfg(feature = "c004_a857")]
                                    // "c004_a857" => Some((c004::c004_a857::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a857::KERNEL)),

                                    // #[cfg(feature = "c004_a858")]
                                    // "c004_a858" => Some((c004::c004_a858::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a858::KERNEL)),

                                    // #[cfg(feature = "c004_a859")]
                                    // "c004_a859" => Some((c004::c004_a859::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a859::KERNEL)),

                                    // #[cfg(feature = "c004_a860")]
                                    // "c004_a860" => Some((c004::c004_a860::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a860::KERNEL)),

                                    // #[cfg(feature = "c004_a861")]
                                    // "c004_a861" => Some((c004::c004_a861::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a861::KERNEL)),

                                    // #[cfg(feature = "c004_a862")]
                                    // "c004_a862" => Some((c004::c004_a862::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a862::KERNEL)),

                                    // #[cfg(feature = "c004_a863")]
                                    // "c004_a863" => Some((c004::c004_a863::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a863::KERNEL)),

                                    // #[cfg(feature = "c004_a864")]
                                    // "c004_a864" => Some((c004::c004_a864::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a864::KERNEL)),

                                    // #[cfg(feature = "c004_a865")]
                                    // "c004_a865" => Some((c004::c004_a865::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a865::KERNEL)),

                                    // #[cfg(feature = "c004_a866")]
                                    // "c004_a866" => Some((c004::c004_a866::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a866::KERNEL)),

                                    // #[cfg(feature = "c004_a867")]
                                    // "c004_a867" => Some((c004::c004_a867::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a867::KERNEL)),

                                    // #[cfg(feature = "c004_a868")]
                                    // "c004_a868" => Some((c004::c004_a868::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a868::KERNEL)),

                                    // #[cfg(feature = "c004_a869")]
                                    // "c004_a869" => Some((c004::c004_a869::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a869::KERNEL)),

                                    // #[cfg(feature = "c004_a870")]
                                    // "c004_a870" => Some((c004::c004_a870::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a870::KERNEL)),

                                    // #[cfg(feature = "c004_a871")]
                                    // "c004_a871" => Some((c004::c004_a871::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a871::KERNEL)),

                                    // #[cfg(feature = "c004_a872")]
                                    // "c004_a872" => Some((c004::c004_a872::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a872::KERNEL)),

                                    // #[cfg(feature = "c004_a873")]
                                    // "c004_a873" => Some((c004::c004_a873::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a873::KERNEL)),

                                    // #[cfg(feature = "c004_a874")]
                                    // "c004_a874" => Some((c004::c004_a874::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a874::KERNEL)),

                                    // #[cfg(feature = "c004_a875")]
                                    // "c004_a875" => Some((c004::c004_a875::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a875::KERNEL)),

                                    // #[cfg(feature = "c004_a876")]
                                    // "c004_a876" => Some((c004::c004_a876::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a876::KERNEL)),

                                    // #[cfg(feature = "c004_a877")]
                                    // "c004_a877" => Some((c004::c004_a877::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a877::KERNEL)),

                                    // #[cfg(feature = "c004_a878")]
                                    // "c004_a878" => Some((c004::c004_a878::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a878::KERNEL)),

                                    // #[cfg(feature = "c004_a879")]
                                    // "c004_a879" => Some((c004::c004_a879::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a879::KERNEL)),

                                    // #[cfg(feature = "c004_a880")]
                                    // "c004_a880" => Some((c004::c004_a880::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a880::KERNEL)),

                                    // #[cfg(feature = "c004_a881")]
                                    // "c004_a881" => Some((c004::c004_a881::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a881::KERNEL)),

                                    // #[cfg(feature = "c004_a882")]
                                    // "c004_a882" => Some((c004::c004_a882::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a882::KERNEL)),

                                    // #[cfg(feature = "c004_a883")]
                                    // "c004_a883" => Some((c004::c004_a883::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a883::KERNEL)),

                                    // #[cfg(feature = "c004_a884")]
                                    // "c004_a884" => Some((c004::c004_a884::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a884::KERNEL)),

                                    // #[cfg(feature = "c004_a885")]
                                    // "c004_a885" => Some((c004::c004_a885::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a885::KERNEL)),

                                    // #[cfg(feature = "c004_a886")]
                                    // "c004_a886" => Some((c004::c004_a886::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a886::KERNEL)),

                                    // #[cfg(feature = "c004_a887")]
                                    // "c004_a887" => Some((c004::c004_a887::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a887::KERNEL)),

                                    // #[cfg(feature = "c004_a888")]
                                    // "c004_a888" => Some((c004::c004_a888::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a888::KERNEL)),

                                    // #[cfg(feature = "c004_a889")]
                                    // "c004_a889" => Some((c004::c004_a889::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a889::KERNEL)),

                                    // #[cfg(feature = "c004_a890")]
                                    // "c004_a890" => Some((c004::c004_a890::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a890::KERNEL)),

                                    // #[cfg(feature = "c004_a891")]
                                    // "c004_a891" => Some((c004::c004_a891::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a891::KERNEL)),

                                    // #[cfg(feature = "c004_a892")]
                                    // "c004_a892" => Some((c004::c004_a892::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a892::KERNEL)),

                                    // #[cfg(feature = "c004_a893")]
                                    // "c004_a893" => Some((c004::c004_a893::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a893::KERNEL)),

                                    // #[cfg(feature = "c004_a894")]
                                    // "c004_a894" => Some((c004::c004_a894::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a894::KERNEL)),

                                    // #[cfg(feature = "c004_a895")]
                                    // "c004_a895" => Some((c004::c004_a895::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a895::KERNEL)),

                                    // #[cfg(feature = "c004_a896")]
                                    // "c004_a896" => Some((c004::c004_a896::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a896::KERNEL)),

                                    // #[cfg(feature = "c004_a897")]
                                    // "c004_a897" => Some((c004::c004_a897::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a897::KERNEL)),

                                    // #[cfg(feature = "c004_a898")]
                                    // "c004_a898" => Some((c004::c004_a898::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a898::KERNEL)),

                                    // #[cfg(feature = "c004_a899")]
                                    // "c004_a899" => Some((c004::c004_a899::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a899::KERNEL)),

                                    // #[cfg(feature = "c004_a900")]
                                    // "c004_a900" => Some((c004::c004_a900::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a900::KERNEL)),

                                    // #[cfg(feature = "c004_a901")]
                                    // "c004_a901" => Some((c004::c004_a901::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a901::KERNEL)),

                                    // #[cfg(feature = "c004_a902")]
                                    // "c004_a902" => Some((c004::c004_a902::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a902::KERNEL)),

                                    // #[cfg(feature = "c004_a903")]
                                    // "c004_a903" => Some((c004::c004_a903::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a903::KERNEL)),

                                    // #[cfg(feature = "c004_a904")]
                                    // "c004_a904" => Some((c004::c004_a904::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a904::KERNEL)),

                                    // #[cfg(feature = "c004_a905")]
                                    // "c004_a905" => Some((c004::c004_a905::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a905::KERNEL)),

                                    // #[cfg(feature = "c004_a906")]
                                    // "c004_a906" => Some((c004::c004_a906::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a906::KERNEL)),

                                    // #[cfg(feature = "c004_a907")]
                                    // "c004_a907" => Some((c004::c004_a907::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a907::KERNEL)),

                                    // #[cfg(feature = "c004_a908")]
                                    // "c004_a908" => Some((c004::c004_a908::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a908::KERNEL)),

                                    // #[cfg(feature = "c004_a909")]
                                    // "c004_a909" => Some((c004::c004_a909::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a909::KERNEL)),

                                    // #[cfg(feature = "c004_a910")]
                                    // "c004_a910" => Some((c004::c004_a910::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a910::KERNEL)),

                                    // #[cfg(feature = "c004_a911")]
                                    // "c004_a911" => Some((c004::c004_a911::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a911::KERNEL)),

                                    // #[cfg(feature = "c004_a912")]
                                    // "c004_a912" => Some((c004::c004_a912::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a912::KERNEL)),

                                    // #[cfg(feature = "c004_a913")]
                                    // "c004_a913" => Some((c004::c004_a913::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a913::KERNEL)),

                                    // #[cfg(feature = "c004_a914")]
                                    // "c004_a914" => Some((c004::c004_a914::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a914::KERNEL)),

                                    // #[cfg(feature = "c004_a915")]
                                    // "c004_a915" => Some((c004::c004_a915::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a915::KERNEL)),

                                    // #[cfg(feature = "c004_a916")]
                                    // "c004_a916" => Some((c004::c004_a916::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a916::KERNEL)),

                                    // #[cfg(feature = "c004_a917")]
                                    // "c004_a917" => Some((c004::c004_a917::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a917::KERNEL)),

                                    // #[cfg(feature = "c004_a918")]
                                    // "c004_a918" => Some((c004::c004_a918::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a918::KERNEL)),

                                    // #[cfg(feature = "c004_a919")]
                                    // "c004_a919" => Some((c004::c004_a919::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a919::KERNEL)),

                                    // #[cfg(feature = "c004_a920")]
                                    // "c004_a920" => Some((c004::c004_a920::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a920::KERNEL)),

                                    // #[cfg(feature = "c004_a921")]
                                    // "c004_a921" => Some((c004::c004_a921::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a921::KERNEL)),

                                    // #[cfg(feature = "c004_a922")]
                                    // "c004_a922" => Some((c004::c004_a922::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a922::KERNEL)),

                                    // #[cfg(feature = "c004_a923")]
                                    // "c004_a923" => Some((c004::c004_a923::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a923::KERNEL)),

                                    // #[cfg(feature = "c004_a924")]
                                    // "c004_a924" => Some((c004::c004_a924::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a924::KERNEL)),

                                    // #[cfg(feature = "c004_a925")]
                                    // "c004_a925" => Some((c004::c004_a925::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a925::KERNEL)),

                                    // #[cfg(feature = "c004_a926")]
                                    // "c004_a926" => Some((c004::c004_a926::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a926::KERNEL)),

                                    // #[cfg(feature = "c004_a927")]
                                    // "c004_a927" => Some((c004::c004_a927::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a927::KERNEL)),

                                    // #[cfg(feature = "c004_a928")]
                                    // "c004_a928" => Some((c004::c004_a928::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a928::KERNEL)),

                                    // #[cfg(feature = "c004_a929")]
                                    // "c004_a929" => Some((c004::c004_a929::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a929::KERNEL)),

                                    // #[cfg(feature = "c004_a930")]
                                    // "c004_a930" => Some((c004::c004_a930::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a930::KERNEL)),

                                    // #[cfg(feature = "c004_a931")]
                                    // "c004_a931" => Some((c004::c004_a931::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a931::KERNEL)),

                                    // #[cfg(feature = "c004_a932")]
                                    // "c004_a932" => Some((c004::c004_a932::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a932::KERNEL)),

                                    // #[cfg(feature = "c004_a933")]
                                    // "c004_a933" => Some((c004::c004_a933::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a933::KERNEL)),

                                    // #[cfg(feature = "c004_a934")]
                                    // "c004_a934" => Some((c004::c004_a934::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a934::KERNEL)),

                                    // #[cfg(feature = "c004_a935")]
                                    // "c004_a935" => Some((c004::c004_a935::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a935::KERNEL)),

                                    // #[cfg(feature = "c004_a936")]
                                    // "c004_a936" => Some((c004::c004_a936::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a936::KERNEL)),

                                    // #[cfg(feature = "c004_a937")]
                                    // "c004_a937" => Some((c004::c004_a937::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a937::KERNEL)),

                                    // #[cfg(feature = "c004_a938")]
                                    // "c004_a938" => Some((c004::c004_a938::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a938::KERNEL)),

                                    // #[cfg(feature = "c004_a939")]
                                    // "c004_a939" => Some((c004::c004_a939::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a939::KERNEL)),

                                    // #[cfg(feature = "c004_a940")]
                                    // "c004_a940" => Some((c004::c004_a940::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a940::KERNEL)),

                                    // #[cfg(feature = "c004_a941")]
                                    // "c004_a941" => Some((c004::c004_a941::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a941::KERNEL)),

                                    // #[cfg(feature = "c004_a942")]
                                    // "c004_a942" => Some((c004::c004_a942::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a942::KERNEL)),

                                    // #[cfg(feature = "c004_a943")]
                                    // "c004_a943" => Some((c004::c004_a943::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a943::KERNEL)),

                                    // #[cfg(feature = "c004_a944")]
                                    // "c004_a944" => Some((c004::c004_a944::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a944::KERNEL)),

                                    // #[cfg(feature = "c004_a945")]
                                    // "c004_a945" => Some((c004::c004_a945::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a945::KERNEL)),

                                    // #[cfg(feature = "c004_a946")]
                                    // "c004_a946" => Some((c004::c004_a946::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a946::KERNEL)),

                                    // #[cfg(feature = "c004_a947")]
                                    // "c004_a947" => Some((c004::c004_a947::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a947::KERNEL)),

                                    // #[cfg(feature = "c004_a948")]
                                    // "c004_a948" => Some((c004::c004_a948::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a948::KERNEL)),

                                    // #[cfg(feature = "c004_a949")]
                                    // "c004_a949" => Some((c004::c004_a949::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a949::KERNEL)),

                                    // #[cfg(feature = "c004_a950")]
                                    // "c004_a950" => Some((c004::c004_a950::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a950::KERNEL)),

                                    // #[cfg(feature = "c004_a951")]
                                    // "c004_a951" => Some((c004::c004_a951::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a951::KERNEL)),

                                    // #[cfg(feature = "c004_a952")]
                                    // "c004_a952" => Some((c004::c004_a952::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a952::KERNEL)),

                                    // #[cfg(feature = "c004_a953")]
                                    // "c004_a953" => Some((c004::c004_a953::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a953::KERNEL)),

                                    // #[cfg(feature = "c004_a954")]
                                    // "c004_a954" => Some((c004::c004_a954::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a954::KERNEL)),

                                    // #[cfg(feature = "c004_a955")]
                                    // "c004_a955" => Some((c004::c004_a955::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a955::KERNEL)),

                                    // #[cfg(feature = "c004_a956")]
                                    // "c004_a956" => Some((c004::c004_a956::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a956::KERNEL)),

                                    // #[cfg(feature = "c004_a957")]
                                    // "c004_a957" => Some((c004::c004_a957::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a957::KERNEL)),

                                    // #[cfg(feature = "c004_a958")]
                                    // "c004_a958" => Some((c004::c004_a958::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a958::KERNEL)),

                                    // #[cfg(feature = "c004_a959")]
                                    // "c004_a959" => Some((c004::c004_a959::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a959::KERNEL)),

                                    // #[cfg(feature = "c004_a960")]
                                    // "c004_a960" => Some((c004::c004_a960::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a960::KERNEL)),

                                    // #[cfg(feature = "c004_a961")]
                                    // "c004_a961" => Some((c004::c004_a961::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a961::KERNEL)),

                                    // #[cfg(feature = "c004_a962")]
                                    // "c004_a962" => Some((c004::c004_a962::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a962::KERNEL)),

                                    // #[cfg(feature = "c004_a963")]
                                    // "c004_a963" => Some((c004::c004_a963::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a963::KERNEL)),

                                    // #[cfg(feature = "c004_a964")]
                                    // "c004_a964" => Some((c004::c004_a964::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a964::KERNEL)),

                                    // #[cfg(feature = "c004_a965")]
                                    // "c004_a965" => Some((c004::c004_a965::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a965::KERNEL)),

                                    // #[cfg(feature = "c004_a966")]
                                    // "c004_a966" => Some((c004::c004_a966::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a966::KERNEL)),

                                    // #[cfg(feature = "c004_a967")]
                                    // "c004_a967" => Some((c004::c004_a967::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a967::KERNEL)),

                                    // #[cfg(feature = "c004_a968")]
                                    // "c004_a968" => Some((c004::c004_a968::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a968::KERNEL)),

                                    // #[cfg(feature = "c004_a969")]
                                    // "c004_a969" => Some((c004::c004_a969::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a969::KERNEL)),

                                    // #[cfg(feature = "c004_a970")]
                                    // "c004_a970" => Some((c004::c004_a970::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a970::KERNEL)),

                                    // #[cfg(feature = "c004_a971")]
                                    // "c004_a971" => Some((c004::c004_a971::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a971::KERNEL)),

                                    // #[cfg(feature = "c004_a972")]
                                    // "c004_a972" => Some((c004::c004_a972::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a972::KERNEL)),

                                    // #[cfg(feature = "c004_a973")]
                                    // "c004_a973" => Some((c004::c004_a973::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a973::KERNEL)),

                                    // #[cfg(feature = "c004_a974")]
                                    // "c004_a974" => Some((c004::c004_a974::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a974::KERNEL)),

                                    // #[cfg(feature = "c004_a975")]
                                    // "c004_a975" => Some((c004::c004_a975::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a975::KERNEL)),

                                    // #[cfg(feature = "c004_a976")]
                                    // "c004_a976" => Some((c004::c004_a976::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a976::KERNEL)),

                                    // #[cfg(feature = "c004_a977")]
                                    // "c004_a977" => Some((c004::c004_a977::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a977::KERNEL)),

                                    // #[cfg(feature = "c004_a978")]
                                    // "c004_a978" => Some((c004::c004_a978::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a978::KERNEL)),

                                    // #[cfg(feature = "c004_a979")]
                                    // "c004_a979" => Some((c004::c004_a979::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a979::KERNEL)),

                                    // #[cfg(feature = "c004_a980")]
                                    // "c004_a980" => Some((c004::c004_a980::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a980::KERNEL)),

                                    // #[cfg(feature = "c004_a981")]
                                    // "c004_a981" => Some((c004::c004_a981::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a981::KERNEL)),

                                    // #[cfg(feature = "c004_a982")]
                                    // "c004_a982" => Some((c004::c004_a982::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a982::KERNEL)),

                                    // #[cfg(feature = "c004_a983")]
                                    // "c004_a983" => Some((c004::c004_a983::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a983::KERNEL)),

                                    // #[cfg(feature = "c004_a984")]
                                    // "c004_a984" => Some((c004::c004_a984::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a984::KERNEL)),

                                    // #[cfg(feature = "c004_a985")]
                                    // "c004_a985" => Some((c004::c004_a985::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a985::KERNEL)),

                                    // #[cfg(feature = "c004_a986")]
                                    // "c004_a986" => Some((c004::c004_a986::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a986::KERNEL)),

                                    // #[cfg(feature = "c004_a987")]
                                    // "c004_a987" => Some((c004::c004_a987::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a987::KERNEL)),

                                    // #[cfg(feature = "c004_a988")]
                                    // "c004_a988" => Some((c004::c004_a988::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a988::KERNEL)),

                                    // #[cfg(feature = "c004_a989")]
                                    // "c004_a989" => Some((c004::c004_a989::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a989::KERNEL)),

                                    // #[cfg(feature = "c004_a990")]
                                    // "c004_a990" => Some((c004::c004_a990::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a990::KERNEL)),

                                    // #[cfg(feature = "c004_a991")]
                                    // "c004_a991" => Some((c004::c004_a991::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a991::KERNEL)),

                                    // #[cfg(feature = "c004_a992")]
                                    // "c004_a992" => Some((c004::c004_a992::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a992::KERNEL)),

                                    // #[cfg(feature = "c004_a993")]
                                    // "c004_a993" => Some((c004::c004_a993::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a993::KERNEL)),

                                    // #[cfg(feature = "c004_a994")]
                                    // "c004_a994" => Some((c004::c004_a994::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a994::KERNEL)),

                                    // #[cfg(feature = "c004_a995")]
                                    // "c004_a995" => Some((c004::c004_a995::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a995::KERNEL)),

                                    // #[cfg(feature = "c004_a996")]
                                    // "c004_a996" => Some((c004::c004_a996::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a996::KERNEL)),

                                    // #[cfg(feature = "c004_a997")]
                                    // "c004_a997" => Some((c004::c004_a997::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a997::KERNEL)),

                                    // #[cfg(feature = "c004_a998")]
                                    // "c004_a998" => Some((c004::c004_a998::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a998::KERNEL)),

                                    // #[cfg(feature = "c004_a999")]
                                    // "c004_a999" => Some((c004::c004_a999::cuda_solve_challenge as CudaSolveChallengeFn, &c004::c004_a999::KERNEL)),
                                    _ => {
                                        Option::<(CudaSolveChallengeFn, &Option<CudaKernel>)>::None
                                    }
                                };

                                match filter_step {
                                    Some((cuda_solve_challenge, algo_kernel)) => {
                                        let challenge_cuda_funcs = challenge_cuda_funcs
                                            .get_or_insert(
                                                get_or_compile_cuda(
                                                    &job.settings.challenge_id,
                                                    &tig_challenges::c004::KERNEL,
                                                    &dev,
                                                )
                                                .await,
                                            )
                                            .clone();
                                        let algorithm_cuda_funcs = algorithm_cuda_funcs
                                            .get_or_insert(
                                                get_or_compile_cuda(
                                                    &job.settings.algorithm_id,
                                                    algo_kernel,
                                                    &dev,
                                                )
                                                .await,
                                            )
                                            .clone();
                                        let challenge =
                                            tig_challenges::c004::Challenge::cuda_generate_instance_from_vec(
                                                seeds,
                                                &job.settings.difficulty,
                                                &dev,
                                                challenge_cuda_funcs,
                                            )
                                            .unwrap();
                                        match cuda_solve_challenge(
                                            &challenge,
                                            &dev,
                                            algorithm_cuda_funcs,
                                        ) {
                                            Ok(Some(solution)) => {
                                                challenge.verify_solution(&solution).is_err()
                                            }
                                            _ => true,
                                        }
                                    }
                                    None => false,
                                }
                            }
                            _ => panic!("Unknown challenge id: {}", job.settings.challenge_id),
                        };
                        if skip {
                            continue;
                        }
                        if let Ok(Some(solution_data)) = compute_solution(
                            &job.settings,
                            nonce,
                            wasm.as_slice(),
                            job.wasm_vm_config.max_memory,
                            job.wasm_vm_config.max_fuel,
                        ) {
                            if verify_solution(&job.settings, nonce, &solution_data.solution)
                                .is_ok()
                            {
                                {
                                    let mut solutions_count = (*solutions_count).lock().await;
                                    *solutions_count += 1;
                                }
                                if solution_data.calc_solution_signature()
                                    <= job.solution_signature_threshold
                                {
                                    let mut solutions_data = (*solutions_data).lock().await;
                                    (*solutions_data).push(solution_data);
                                }
                            }
                        }
                    }
                }
            }
        });
    }
}
