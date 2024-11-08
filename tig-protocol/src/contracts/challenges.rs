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

pub struct ChallengeContract
{
}

impl ChallengeContract {
    pub fn new() -> Self {
        return Self {};
    }
}
