use
{
    crate::
    {
        ctx::
        {
            Context,
            ContextResult,
            ContextError,
        },
        err::
        {
            ProtocolError,
            ProtocolResult,
        },
        contracts::
        {
            Contracts,
        },
    },
    std::
    {
        collections::
        {
            HashSet,
        },
        sync::
        {
            RwLock,
            Arc
        }
    },
    tig_structs::
    {
        core::
        {
            *
        },
        *,
    },
};

pub struct Protocol<T: Context>
{
    ctx:                        Arc<RwLock<T>>,
    contracts:                  Contracts<T>,
}

impl<T: Context> Protocol<T>
{
    pub fn new(ctx: T)          -> Self
    {
        let mut new                 = Self 
        { 
            ctx                     : Arc::new(RwLock::new(ctx)),
            contracts               : Contracts::new(),
        };

        return new;
    }

    pub async fn submit_precommit(
        &self,
        player:                 &Player,
        settings:               &BenchmarkSettings,
        num_nonces:             u32,
    )                                   -> ProtocolResult<String>
    {
        return self.contracts.benchmark.submit_precommit(&Arc::into_inner(self.ctx.clone()).unwrap(), player, settings, num_nonces).await;
    }

    async fn submit_benchmark(
        &self,
        player:                 &Player,
        benchmark_id:           &String,
        merkle_root:            &MerkleHash,
        solution_nonces:        &HashSet<u64>,
    )                                   -> ProtocolResult<()>
    {
        return self.contracts.benchmark.submit_benchmark(&Arc::into_inner(self.ctx.clone()).unwrap(), player, benchmark_id, merkle_root, solution_nonces).await;
    }

    pub async fn submit_proof(
        &self,
        player:                 &Player,
        benchmark_id:           &String,
        merkle_proofs:          &Vec<MerkleProof>,
    )                                   -> ProtocolResult<Result<(), String>>
    {
        return self.contracts.benchmark.submit_proof(&Arc::into_inner(self.ctx.clone()).unwrap(), player, benchmark_id, &merkle_proofs).await;
    }

    pub async fn add_block(&self)       -> String
    {
        return "".to_string();
    }
}
