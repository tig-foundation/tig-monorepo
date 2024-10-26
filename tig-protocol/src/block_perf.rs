use criterion::{black_box, criterion_group, criterion_main, Criterion};
pub mod context;
mod add_block;
mod error;
mod submit_algorithm;
mod submit_benchmark;
mod submit_precommit;
mod submit_proof;
mod submit_topup;
mod verify_proof;
use context::*;
pub use error::*;
use std::collections::HashSet;
use tig_structs::core::*;
use crate::context::Context;
use crate::add_block::AddBlockCache;

struct BenchmarkContext
{
}

impl Context for BenchmarkContext
{
}

#[inline]
fn bench_update_qualifiers<T: Context>(ctx: &T, block: &Block, cache: &mut AddBlockCache)
{
    add_block::update_qualifiers(block, cache);
}

pub fn criterion_benchmark(c: &mut Criterion) 
{
    let ctx: BenchmarkContext = BenchmarkContext {};
    c.bench_function("update_qualifiers", |b|
    {
        let (block, cache)  = add_block::create_block(&ctx).await;

        b.iter(|| bench_update_qualifiers(&ctx, &block, &mut cache))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);