use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use superinstance_hdc_core::{SramImageBuilder, judge};
use superinstance_hdc_core::fingerprint::fingerprint;

fn create_test_sram(num_records: usize) -> superinstance_hdc_core::SramImage {
    let mut builder = SramImageBuilder::new();
    for i in 0..num_records {
        let fp = fingerprint(&format!("record_{}", i), 0xDEAD);
        builder = builder.canary(fp).add_record(fp, i as u32);
    }
    builder.build().unwrap()
}

fn bench_judge(c: &mut Criterion) {
    let mut group = c.benchmark_group("judge");
    
    for size in [100, 1000, 10000].iter() {
        let sram = create_test_sram(*size);
        let query = "record_50";
        
        group.bench_with_input(
            BenchmarkId::new("single", size),
            size,
            |b, _| {
                b.iter(|| {
                    let _ = judge(
                        black_box(&sram), 
                        black_box(query), 
                        black_box(0xDEAD), 
                        black_box(10)
                    );
                });
            }
        );
    }
    
    group.finish();
}

criterion_group!(benches, bench_judge);
criterion_main!(benches);
