use {
    crate::{
        ctx::Context,
        err::{ContractResult, ProtocolError},
    },
    logging_timer::time,
    std::{marker::PhantomData, sync::RwLock},
    tig_structs::core::*,
};

pub struct AlgorithmContract
{
}

impl AlgorithmContract {
    pub fn new() -> Self {
        return Self {};
    }

    async fn submit_algorithm() {
        // FIXME
    }

    async fn submit_wasm() {
        // FIXME
    }

    // update (call after opow.update)
    // update_adoption
    // update_merge_points
    // update_merges

    // FUTURE submit_brekthrough
    // FUTURE rename wasm -> binary
    // FUTURE update breakthrough adoption
    // FUTURE update breakthrough merge points
    // FUTURE update breakthrough merges
}
