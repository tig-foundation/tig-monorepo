use 
{
    crate::
    {
        ctx::
        {
            Context,
        },
        err::
        {
            ContractResult,
        },
    },
    std::
    {
        marker::
        {
            PhantomData,
        },
        collections::
        {
            HashMap,
        },
    },
    tig_structs::
    {
        *,
        core::
        {
            *
        }
    }
};

pub struct AlgorithmCache<T: Context>
{
    algorithms:                 HashMap<String, Algorithm>,
    ctx:                        PhantomData<T>,
}

impl<T: Context> AlgorithmCache<T>
{
    pub fn new() -> Self
    {
        return Self
        { 
            algorithms:                     HashMap::new(),
            ctx:                            PhantomData
        };
    }

    pub async fn fetch_algorithm<'a>(
        &mut self,
        ctx:                    &T,
        id:                     &'a String,
        include_data:           bool
    )                                   -> ContractResult<'a, &Algorithm>
    {
        if !self.algorithms.contains_key(id)
        {
            let algorithm                   = ctx.get_algorithm_by_id(id, include_data).await;
            self.algorithms.insert(id.clone(), algorithm.unwrap().expect("Algorithm not found"));
        }

        return Ok(self.algorithms.get(id).unwrap());
    }

    pub fn get_algorithm(
        &self,
        ctx:                        &T,
        id:                         &String
        
    )                                   -> Option<&Algorithm>
    {
        return self.algorithms.get(id);
    }

    pub fn has_algorithm(
        &self, 
        id:                     &String
    )                                   -> bool
    {
        return self.algorithms.contains_key(id);
    }
}
