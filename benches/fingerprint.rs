use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use superinstance_hdc_core::fingerprint;

fn bench_fingerprint(c: &mut Criterion) {
    let mut group = c.benchmark_group("fingerprint");
    
    let inputs = [
        ("short", "hello"),
        ("medium", "the quick brown fox jumps over the lazy dog"),
        ("long", &"a".repeat(1000)),
    ];
    
    for (name, text) in &inputs {
        group.bench_with_input(BenchmarkId::new("murmur3", name), text, |b, text| {
            b.iter(|| fingerprint(black_box(text), black_box(0xDEADBEEF)));
        });
    }
    
    group.finish();
}

criterion_group!(benches, bench_fingerprint);
criterion_main!(benches);
