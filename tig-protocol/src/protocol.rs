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
};

pub struct Protocol<T: Context + Send + Sync>
{
    ctx:        Arc<T>,
    contracts:  Arc<Contracts<T>>,
}

impl<T: Context + std::marker::Sync + std::marker::Send> Protocol<T>
{
    pub fn new(ctx: T) -> Self 
    {
        return Self 
        {
            contracts:  Arc::new(Contracts::new()),
            ctx:        Arc::new(ctx),
        };
    }

    pub async fn submit_precommit(
        &self,
        player: &Player,
        settings: &BenchmarkSettings,
        num_nonces: u32,
    ) -> ProtocolResult<String> {
        return self
            .contracts
            .benchmark
            .submit_precommit(
                &Arc::into_inner(self.ctx.clone()).unwrap(),
                player,
                settings,
                num_nonces,
            )
            .await;
    }

    async fn submit_benchmark(
        &self,
        player: &Player,
        benchmark_id: &String,
        merkle_root: &MerkleHash,
        solution_nonces: &HashSet<u64>,
    ) -> ProtocolResult<()> {
        return self
            .contracts
            .benchmark
            .submit_benchmark(
                &Arc::into_inner(self.ctx.clone()).unwrap(),
                player,
                benchmark_id,
                merkle_root,
                solution_nonces,
            )
            .await;
    }

    pub async fn submit_proof(
        &self,
        player: &Player,
        benchmark_id: &String,
        merkle_proofs: &Vec<MerkleProof>,
    ) -> ProtocolResult<Result<(), String>> {
        return self
            .contracts
            .benchmark
            .submit_proof(
                &Arc::into_inner(self.ctx.clone()).unwrap(),
                player,
                benchmark_id,
                &merkle_proofs,
            )
            .await;
    }

    pub async fn add_block(&self) -> String 
    {
        let curr_block_id           = self.ctx.notify_add_new_block().unwrap();
        let prev_block_id           = self.ctx.get_prev_block_id();

        let (mut block, cache)      = crate::block::create_block(&Arc::into_inner(self.ctx.clone()).unwrap(), &self.ctx.get_config(), &curr_block_id).await;
        block.id                    = curr_block_id;

        self.contracts.opow.update(&Arc::into_inner(self.ctx.clone()).unwrap(), &cache, &block);
        rayon::scope(|s| 
        {
            s.spawn(|_| self.contracts.algorithm.update(&Arc::into_inner(self.ctx.clone()).unwrap(), &cache, &block, &prev_block_id));
            s.spawn(|_| self.contracts.challenge.update(&Arc::into_inner(self.ctx.clone()).unwrap(), &cache, &block, &prev_block_id));
            s.spawn(|_| self.contracts.player.update(&Arc::into_inner(self.ctx.clone()).unwrap(), &cache, &block, &prev_block_id));
        });
        self.contracts.rewards.update(&Arc::into_inner(self.ctx.clone()).unwrap(), &cache, &block);
        
        // commit data
        let _ = self.contracts.opow.commit_updates(&Arc::into_inner(self.ctx.clone()).unwrap(), &cache, &block).unwrap();
        let _ = self.contracts.challenge.commit_updates(&Arc::into_inner(self.ctx.clone()).unwrap(), &cache, &block).unwrap();
        let _ = self.contracts.algorithm.commit_update(&Arc::into_inner(self.ctx.clone()).unwrap(), &cache, &block).unwrap();

        return block.id;
    }
}
