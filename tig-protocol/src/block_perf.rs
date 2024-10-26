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

pub struct Protocol<T: Context> 
{
    pub ctx: T,
}

impl<'a, T: Context> Protocol<T> 
{
    pub fn new(ctx: T) -> Self 
    {
        Self { ctx }
    }
}

struct BenchmarkContext
{
}

impl Context for BenchmarkContext
{
}

#[inline]
fn bench_update_qualifiers<T: Context>(ctx: &T)
{
    let (block, cache)  = add_block::create_block(ctx).await;
}

pub fn criterion_benchmark(c: &mut Criterion) 
{
    let ctx: BenchmarkContext = BenchmarkContext {};
    c.bench_function("update_qualifiers", |b|
    {
        b.iter(|| bench_update_qualifiers(&ctx))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);