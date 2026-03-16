use std::path::{Path, PathBuf};
use std::thread;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ndarray::ArrayD;
use netcdf_reader::{NcFile, NcGroup};

#[derive(Clone, Copy)]
enum NumericKind {
    F32,
    F64,
}

struct BenchCase {
    id: &'static str,
    subdir: &'static str,
    file: &'static str,
    variable: &'static str,
    kind: NumericKind,
}

const CASES: &[BenchCase] = &[
    BenchCase {
        id: "cdf1_simple",
        subdir: "netcdf3",
        file: "cdf1_simple.nc",
        variable: "temp",
        kind: NumericKind::F32,
    },
    BenchCase {
        id: "nc4_basic",
        subdir: "netcdf4",
        file: "nc4_basic.nc",
        variable: "data",
        kind: NumericKind::F64,
    },
    BenchCase {
        id: "nc4_compressed",
        subdir: "netcdf4",
        file: "nc4_compressed.nc",
        variable: "compressed",
        kind: NumericKind::F32,
    },
    BenchCase {
        id: "nc4_groups_nested",
        subdir: "netcdf4",
        file: "nc4_groups.nc",
        variable: "obs/surface/pressure",
        kind: NumericKind::F64,
    },
];

fn fixture_path(subdir: &str, file: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("testdata")
        .join(subdir)
        .join(file)
}

fn bench_threads() -> usize {
    std::env::var("BENCH_THREADS")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|threads| *threads > 1)
        .unwrap_or_else(|| {
            thread::available_parallelism()
                .map(|parallelism| parallelism.get().min(4))
                .unwrap_or(2)
                .max(2)
        })
}

fn checksum_f32(array: &ArrayD<f32>) -> u64 {
    let sum = array.iter().fold(0.0_f64, |acc, value| acc + f64::from(*value));
    sum.to_bits() ^ array.len() as u64
}

fn checksum_f64(array: &ArrayD<f64>) -> u64 {
    let sum = array.iter().copied().sum::<f64>();
    sum.to_bits() ^ array.len() as u64
}

fn read_with_cairn(path: &Path, case: &BenchCase) -> u64 {
    let file = NcFile::open(path).unwrap();
    match case.kind {
        NumericKind::F32 => checksum_f32(&file.read_variable::<f32>(case.variable).unwrap()),
        NumericKind::F64 => checksum_f64(&file.read_variable::<f64>(case.variable).unwrap()),
    }
}

fn read_with_georust(path: &Path, case: &BenchCase) -> u64 {
    let file = netcdf::open(path).unwrap();
    let variable = file.variable(case.variable).unwrap();
    match case.kind {
        NumericKind::F32 => checksum_f32(&variable.get::<f32, _>(..).unwrap()),
        NumericKind::F64 => checksum_f64(&variable.get::<f64, _>(..).unwrap()),
    }
}

fn walk_cairn_group(group: &NcGroup) -> usize {
    let mut total = group.name.len();
    total += group.dimensions.len();
    total += group.attributes.len();

    for dimension in &group.dimensions {
        total += dimension.name.len();
        total += dimension.size as usize;
        total += usize::from(dimension.is_unlimited);
    }

    for variable in &group.variables {
        total += variable.name().len();
        total += variable.ndim();
        total += variable.num_elements() as usize;
        total += variable.attributes().len();
        for dimension in variable.dimensions() {
            total += dimension.name.len();
            total += dimension.size as usize;
        }
    }

    for attribute in &group.attributes {
        total += attribute.name.len();
    }

    for child in &group.groups {
        total += walk_cairn_group(child);
    }

    total
}

fn metadata_with_cairn(path: &Path) -> usize {
    let file = NcFile::open(path).unwrap();
    walk_cairn_group(file.root_group())
}

fn walk_georust_group(group: &netcdf::Group<'_>) -> usize {
    let mut total = group.name().len();

    for dimension in group.dimensions() {
        total += dimension.name().len();
        total += dimension.len();
        total += usize::from(dimension.is_unlimited());
    }

    for attribute in group.attributes() {
        total += attribute.name().len();
    }

    for variable in group.variables() {
        total += variable.name().len();
        total += variable.dimensions().len();
        total += variable.len();
        total += variable.attributes().count();
        for dimension in variable.dimensions() {
            total += dimension.name().len();
            total += dimension.len();
        }
    }

    for child in group.groups() {
        total += walk_georust_group(&child);
    }

    total
}

fn metadata_with_georust(path: &Path) -> usize {
    let file = netcdf::open(path).unwrap();
    let mut total = 0usize;

    for dimension in file.dimensions() {
        total += dimension.name().len();
        total += dimension.len();
        total += usize::from(dimension.is_unlimited());
    }

    for attribute in file.attributes() {
        total += attribute.name().len();
    }

    for variable in file.variables() {
        total += variable.name().len();
        total += variable.dimensions().len();
        total += variable.len();
        total += variable.attributes().count();
        for dimension in variable.dimensions() {
            total += dimension.name().len();
            total += dimension.len();
        }
    }

    if let Ok(groups) = file.groups() {
        for group in groups {
            total += walk_georust_group(&group);
        }
    }

    total
}

fn parallel_read_with_cairn(path: &Path, case: &BenchCase, threads: usize) -> u64 {
    thread::scope(|scope| {
        let mut handles = Vec::with_capacity(threads);
        for _ in 0..threads {
            handles.push(scope.spawn(|| read_with_cairn(path, case)));
        }
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .fold(0_u64, |acc, value| acc ^ value)
    })
}

fn parallel_read_with_georust(path: &Path, case: &BenchCase, threads: usize) -> u64 {
    thread::scope(|scope| {
        let mut handles = Vec::with_capacity(threads);
        for _ in 0..threads {
            handles.push(scope.spawn(|| read_with_georust(path, case)));
        }
        handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .fold(0_u64, |acc, value| acc ^ value)
    })
}

fn bench_metadata(c: &mut Criterion) {
    let mut group = c.benchmark_group("metadata_walk");

    for case in CASES {
        let path = fixture_path(case.subdir, case.file);
        assert!(path.exists(), "missing fixture {}", path.display());

        group.bench_with_input(BenchmarkId::new("cairn", case.id), &path, |b, path| {
            b.iter(|| black_box(metadata_with_cairn(path)));
        });

        group.bench_with_input(BenchmarkId::new("georust", case.id), &path, |b, path| {
            b.iter(|| black_box(metadata_with_georust(path)));
        });
    }

    group.finish();
}

fn bench_single_read(c: &mut Criterion) {
    let mut group = c.benchmark_group("read_full_single");

    for case in CASES {
        let path = fixture_path(case.subdir, case.file);
        assert!(path.exists(), "missing fixture {}", path.display());

        group.bench_with_input(BenchmarkId::new("cairn", case.id), &(path.clone(), case), |b, input| {
            b.iter(|| black_box(read_with_cairn(&input.0, input.1)));
        });

        group.bench_with_input(BenchmarkId::new("georust", case.id), &(path.clone(), case), |b, input| {
            b.iter(|| black_box(read_with_georust(&input.0, input.1)));
        });
    }

    group.finish();
}

fn bench_parallel_read(c: &mut Criterion) {
    let threads = bench_threads();
    let mut group = c.benchmark_group("read_full_parallel");
    group.throughput(Throughput::Elements(threads as u64));

    for case in CASES.iter().filter(|case| case.subdir == "netcdf4") {
        let path = fixture_path(case.subdir, case.file);
        assert!(path.exists(), "missing fixture {}", path.display());

        group.bench_with_input(
            BenchmarkId::new(format!("cairn_x{threads}"), case.id),
            &(path.clone(), case),
            |b, input| {
                b.iter(|| black_box(parallel_read_with_cairn(&input.0, input.1, threads)));
            },
        );

        group.bench_with_input(
            BenchmarkId::new(format!("georust_x{threads}"), case.id),
            &(path.clone(), case),
            |b, input| {
                b.iter(|| black_box(parallel_read_with_georust(&input.0, input.1, threads)));
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_metadata, bench_single_read, bench_parallel_read);
criterion_main!(benches);
