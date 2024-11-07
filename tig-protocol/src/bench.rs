#![allow(unused)]

mod benchmarks;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

pub fn criterion_benchmark(
    c:                          &mut Criterion
) 
{
    benchmarks::block::perform_benchmarks(c);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);