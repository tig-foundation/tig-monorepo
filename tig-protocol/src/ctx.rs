pub use anyhow::{Error as ContextError, Result as ContextResult};
use {
    std::sync::Arc,
    tig_structs::{config::*, core::*, *},
    std::sync::RwLock,
    crate::
    {
        store::*,
    },
};

// TODO Vote
// TODO Deposit
// TODO Breakthrough
// TODO Binary

pub struct Context<B, A, P, BE, PR, F, C, T, W>
where
    B: BlocksStore,
    A: AlgorithmsStore,
    P: PrecommitsStore,
    BE: BenchmarksStore,
    PR: ProofsStore,
    F: FraudsStore,
    C: ChallengesStore,
    T: TopUpsStore,
    W: WasmsStore,
{
    pub blocks:        Arc<RwLock<B>>,
    pub algorithms:    Arc<RwLock<A>>,
    pub precommits:    Arc<RwLock<P>>,
    pub benchmarks:    Arc<RwLock<BE>>,
    pub proofs:        Arc<RwLock<PR>>,
    pub frauds:        Arc<RwLock<F>>,
    pub challenges:    Arc<RwLock<C>>,
    pub topups:        Arc<RwLock<T>>,
    pub wasms:         Arc<RwLock<W>>,
}

impl<B, A, P, BE, PR, F, C, T, W> Context<B, A, P, BE, PR, F, C, T, W>
where
    B: BlocksStore,
    A: AlgorithmsStore,
    P: PrecommitsStore,
    BE: BenchmarksStore,
    PR: ProofsStore,
    F: FraudsStore,
    C: ChallengesStore,
    T: TopUpsStore,
    W: WasmsStore,
{
    pub fn new(blocks: B, algorithms: A, precommits: P, benchmarks: BE, proofs: PR, frauds: F, challenges: C, topups: T, wasms: W) -> Self
    {
        return Self 
        {
            blocks:        Arc::new(RwLock::new(blocks)),
            algorithms:    Arc::new(RwLock::new(algorithms)),
            precommits:    Arc::new(RwLock::new(precommits)),
            benchmarks:    Arc::new(RwLock::new(benchmarks)),
            proofs:        Arc::new(RwLock::new(proofs)),
            frauds:        Arc::new(RwLock::new(frauds)),
            challenges:    Arc::new(RwLock::new(challenges)),
            topups:        Arc::new(RwLock::new(topups)),
            wasms:         Arc::new(RwLock::new(wasms)),
        };
    }
}

// pub trait Context {
//     fn get_block_by_height(&self, block_height: i64) -> Option<Block>;

//     fn get_block_by_id(&self, block_id: &String) -> Option<&Block>;

//     fn get_next_block(&self) -> &Block;

//     fn get_algorithm_by_id(&self, algorithm_id: &String) -> Option<Algorithm>;

//     fn get_algorithm_by_tx_hash(&self, tx_hash: &String) -> Option<Algorithm>;

//     fn get_challenges_by_id(&self, challenge_id: &String) -> Vec<Challenge>;

//     fn get_challenge_by_id_and_height(
//         &self,
//         challenge_id: &String,
//         block_height: u64,
//     ) -> Option<Challenge>;

//     fn get_challenge_by_id_and_block_id(
//         &self,
//         challenge_id: &String,
//         block_id: &String,
//     ) -> Option<Challenge>;

//     fn get_precommits_by_settings(&self, settings: &BenchmarkSettings) -> Vec<Precommit>;

//     fn get_precommits_by_benchmark_id(&self, benchmark_id: &String) -> Vec<Precommit>;

//     fn get_transaction(&self, tx_hash: &String) -> Option<Transaction>;

//     fn get_topups_by_tx_hash(&self, tx_hash: &String) -> Vec<TopUp>;

//     fn get_benchmarks_by_id(&self, benchmark_id: &String) -> Vec<Benchmark>;

//     fn get_proofs_by_benchmark_id(&self, benchmark_id: &String) -> Vec<Proof>;

//     fn get_player_deposit(
//         &self,
//         eth_block_num: &String,
//         player_id: &String,
//     ) -> Option<PreciseNumber>;

//     fn verify_solution(
//         &self,
//         settings: &BenchmarkSettings,
//         nonce: u64,
//         solution: &Solution,
//     ) -> ContextResult<anyhow::Result<()>>;

//     fn compute_solution(
//         &self,
//         settings: &BenchmarkSettings,
//         nonce: u64,
//         wasm_vm_config: &WasmVMConfig,
//     ) -> ContextResult<anyhow::Result<OutputData>>;

//     fn add_precommit_to_mempool(
//         &self,
//         settings: &BenchmarkSettings,
//         details: &PrecommitDetails,
//     ) -> ContextResult<String>;

//     fn add_benchmark_to_mempool(
//         &self,
//         benchmark_id: &String,
//         merkle_root: &MerkleHash,
//         solution_nonces: &HashSet<u64>,
//     ) -> ContextResult<()>;

//     fn add_proof_to_mempool(
//         &self,
//         benchmark_id: &String,
//         merkle_proofs: &Vec<MerkleProof>,
//     ) -> ContextResult<()>;

//     fn add_fraud_to_mempool(
//         &self,
//         benchmark_id: &String,
//         allegations: &String,
//     ) -> ContextResult<()>;

//     fn update_precommit_state(
//         &self,
//         benchmark_id: &String,
//         state: &PrecommitState,
//     ) -> ContextResult<()>;

//     fn update_algorithm_state(
//         &self,
//         algorithm_id: &String,
//         state: &AlgorithmState,
//     ) -> ContextResult<()>;

//     fn notify_new_block(&self);

//     fn block_assembled(&self, block: &Block);

//     fn data_committed(&self, block: &Block);
// }
