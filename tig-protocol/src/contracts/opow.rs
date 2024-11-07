use {
    crate::{
        ctx::Context,
        err::{ContractResult, ProtocolError},
    },
    logging_timer::time,
    std::{
        marker::PhantomData,
        sync::{Arc, RwLock},
    },
    tig_structs::core::*,
};

pub struct PlayerContract<T: Context> {
    phantom: PhantomData<T>,
}

impl<T: Context> PlayerContract<T> {
    pub fn new() -> Self {
        return Self {
            phantom: PhantomData,
        };
    }

    async fn submit_delegation_share() {
        // FIXME
    }

    // update (order of ops)
    // 1. update_cutoffs
    // 2. update_qualifiers
    // 3. update_frontiers
    // 4. update_influence
}
