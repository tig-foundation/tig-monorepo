use {
    crate::{
        contracts::Contracts,
        ctx::{Context, ContextError, ContextResult},
        err::{ProtocolError, ProtocolResult},
    },
    std::{
        collections::HashSet,
        sync::{Arc, RwLock},
    },
    tig_structs::{core::*, *},
    crate::store::*,
};
pub struct Protocol<B, A, P, BE, PR, F, C, T, W>
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
    ctx:        Context<B, A, P, BE, PR, F, C, T, W>,
    contracts:  Arc<Contracts>,
}

impl<B: BlocksStore, A: AlgorithmsStore, P: PrecommitsStore, BE: BenchmarksStore, PR: ProofsStore, 
    F: FraudsStore, C: ChallengesStore, T: TopUpsStore, W: WasmsStore
> Protocol<B, A, P, BE, PR, F, C, T, W>
{
    pub fn new(blocks: B, algorithms: A, precommits: P, benchmarks: BE, proofs: PR, frauds: F, challenges: C, topups: T, wasms: W) -> Self 
    {
        return Self 
        {
            contracts:  Arc::new(Contracts::new()),
            ctx:        Context::new(blocks, algorithms, precommits, benchmarks, proofs, frauds, challenges, topups, wasms),
        };
    }

    pub async fn submit_precommit(
        &self,
        player: &Player,
        settings: &BenchmarkSettings,
        num_nonces: u32,
    ) -> ProtocolResult<String> {
        /*return self
            .contracts
            .benchmark
            .submit_precommit(
                &Arc::into_inner(self.ctx.clone()).unwrap(),
                player,
                settings,
                num_nonces,
            )
            .await;*/

        return Ok(String::new());
    }

    async fn submit_benchmark(
        &self,
        player: &Player,
        benchmark_id: &String,
        merkle_root: &MerkleHash,
        solution_nonces: &HashSet<u64>,
    ) -> ProtocolResult<()> {
        /*return self
            .contracts
            .benchmark
            .submit_benchmark(
                &Arc::into_inner(self.ctx.clone()).unwrap(),
                player,
                benchmark_id,
                merkle_root,
                solution_nonces,
            )
            .await;*/

        return Ok(());
    }

    pub async fn submit_proof(
        &self,
        player: &Player,
        benchmark_id: &String,
        merkle_proofs: &Vec<MerkleProof>,
    ) -> ProtocolResult<Result<(), String>> {
        /*return self
            .contracts
            .benchmark
            .submit_proof(
                &Arc::into_inner(self.ctx.clone()).unwrap(),
                player,
                benchmark_id,
                &merkle_proofs,
            )
            .await;*/

        return Ok(Ok(()));
    }

    pub async fn add_block(&self) -> String 
    {
        //return block::add_block(self.ctx.clone(), self.contracts.clone()).await;

        return String::new();
    }
}
