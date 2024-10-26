use criterion::{black_box, criterion_group, criterion_main, Criterion};

#[inline]
fn bench_update_qualifiers()
{

}

pub fn criterion_benchmark(c: &mut Criterion) 
{
    c.bench_function("update_qualifiers", |b|
    {
        b.iter(|| bench_update_qualifiers())
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);