use criterion::Criterion;

pub(crate) fn benchmark_block()
{
    for _ in 0..1000
    {
        let v = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        v.iter().sum::<i32>();
    }
}

pub(crate) fn perform_benchmarks(
    c: &mut Criterion
)
{
    c.bench_function("block", |b| b.iter(|| benchmark_block()));
}