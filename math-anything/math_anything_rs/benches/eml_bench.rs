use criterion::{black_box, Criterion, criterion_group, criterion_main};
use math_anything_rs::*;

fn bench_eml(c: &mut Criterion) {
    c.bench_function("eml_single", |b| {
        b.iter(|| eml(black_box(1.0), black_box(2.0)))
    });
}

fn bench_eml_closure(c: &mut Criterion) {
    let base = vec![0.0_f64, 1.0, std::f64::consts::E, std::f64::consts::PI];
    c.bench_function("eml_closure_depth3_max100", |b| {
        b.iter(|| eml_closure(black_box(base.clone()), black_box(3), black_box(100)))
    });
}

fn bench_buckingham(c: &mut Criterion) {
    let data = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
    c.bench_function("buckingham_pi_5x4", |b| {
        b.iter(|| buckingham_pi(black_box(4), black_box(5), black_box(data.clone())))
    });
}

fn bench_shortest_path(c: &mut Criterion) {
    let edges: Vec<(usize, usize)> = (0..999).map(|i| (i, i + 1)).collect();
    c.bench_function("shortest_path_1000_nodes", |b| {
        b.iter(|| shortest_path(black_box(1000), black_box(edges.clone()), black_box(0), black_box(999)))
    });
}

criterion_group!(benches, bench_eml, bench_eml_closure, bench_buckingham, bench_shortest_path);
criterion_main!(benches);
