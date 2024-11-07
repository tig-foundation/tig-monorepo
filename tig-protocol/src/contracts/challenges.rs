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

pub struct ChallengeContract<T: Context> {
    phantom: PhantomData<T>,
}

impl<T: Context> ChallengeContract<T> {
    pub fn new() -> Self {
        return Self {
            phantom: PhantomData,
        };
    }
}
