use
{
    crate::
    {
        Block
    },
    tig_structs::
    {
        core::
        {
            Player,
            BenchmarkSettings
        }
    }
};

pub trait IBenchmarksContract
{
    fn verify_num_nonces(
        &self, 
        num_nonces:                     u32
    )                                           -> ();

    fn verify_player_owns_benchmark(
        &self, 
        player:                         &Player, 
        settings:                       &BenchmarkSettings
    )                                           -> ();

    fn verify_sufficient_lifespan(
        &self, 
        block:                          &Block
    )                                           -> ();

    fn get_challenge_by_id(
        &self, 
        challenge_id:                   &String,
        block:                          &Block
    )                                           -> ();

    fn verify_algorithm(
        &self, 
        algorithm_id:                   &String,
        block:                          &Block
    )                                           -> ();

    fn submit_precommit(
        &self
    )                                           -> ();

    fn submit_benchmark(&self)  -> ();
    fn submit_proof(&self)      -> ();
}

impl dyn IBenchmarksContract 
{
    pub fn submit_precommit(
        &self
    )                                           -> ()
    {

    }

    pub fn submit_benchmark(
        &self
    )                                           -> ()
    {

    }

    pub fn submit_proof(
        &self
    )                                           -> ()
    {

    }
}