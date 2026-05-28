# Benchmark Report

Date: 2026-03-17
Targeted update: 2026-05-28

This report summarizes the current benchmark suite for `netcdf-rust` and a
representative local run against the C-backed `georust/netcdf` baseline. It
captures the current performance shape across metadata, full reads, slicing,
and threaded throughput.

## System Under Test

- Machine: Apple M1
- CPU topology: 8 logical CPUs, 4 performance cores
- Memory: 16 GiB
- OS: macOS 13.0
- Architecture: `arm64`
- Rust toolchain: `rustc 1.92.0`

These measurements reflect this machine and should not be read as universal
throughput claims. The suite is designed to be representative across workload
types, but absolute numbers will move with CPU, memory bandwidth, storage, and
the local HDF5/netCDF stack.

## Scope

The suite covers open cost, metadata walks, warm full reads, end-to-end reads,
slices, shared-handle throughput, and threaded scaling. The comparison target
is `georust/netcdf` backed by `netcdf-c`.

## Methodology

Commands used for the current report:

```sh
cargo test --workspace

cargo bench -p netcdf-reader --bench compare_georust -- --noplot

python3 scripts/criterion_summary.py --speedup \
  --group open_only \
  --group metadata_reuse_handle \
  --group read_full_reuse_handle \
  --group open_and_read_full \
  --group slice_reuse_handle_hdf5_backend \
  --group parallel_metadata_batch \
  --group parallel_slice_batch \
  --group parallel_open_and_read \
  --group parallel_read_shared_netcdf_rust \
  --group read_full_internal_parallel \
  --group read_full_internal_parallel_nocache

# Follow-up reruns used for the updated warm full-read numbers:
cargo bench -p netcdf-reader --bench compare_georust -- --noplot 'read_full_reuse_handle'

# Targeted classic internal-parallel follow-up, run in Docker:
BENCH_THREAD_LIST=1,2,4,8,16 \
  cargo bench -p netcdf-reader --bench compare_georust \
  'read_full_internal_parallel_classic|slice_internal_parallel_classic' \
  -- --noplot --sample-size 12 --measurement-time 0.4 --warm-up-time 0.2
```

Notes:

- Criterion is used for all timing.
- Both implementations read the same files and variable paths.
- Benchmarks validate checksum and shape parity before timing.
- Contention-oriented groups open one handle per worker, synchronize start, and
  then repeat small operations to make cross-thread serialization visible.

## Current Results

### Single-Thread Summary

| workload | netcdf-rust | georust/netcdf | result |
| --- | ---: | ---: | --- |
| `cdf1_simple` metadata | 8.8 us | 62.2 us | `netcdf-rust` 7.1x faster |
| `cdf1_simple` full read | 25.5 us | 79.1 us | `netcdf-rust` 3.1x faster |
| `nc4_basic` metadata | 35.0 us | 195.0 us | `netcdf-rust` 5.6x faster |
| `nc4_basic` full read | 57.5 us | 402.2 us | `netcdf-rust` 7.0x faster |
| `nc4_compressed` metadata | 34.3 us | 206.1 us | `netcdf-rust` 6.0x faster |
| `nc4_compressed` full read | 393.8 us | 795.2 us | `netcdf-rust` 2.0x faster |
| `nested_nc4_groups` full read | 90.1 us | 340.5 us | `netcdf-rust` 3.8x faster |
| `large_nc4_compressed` full read | 20.8 ms | 25.9 ms | `netcdf-rust` 1.25x faster |

### Threaded Highlights

| workload | netcdf-rust | georust/netcdf | takeaway |
| --- | --- | --- | --- |
| `parallel_metadata_batch` `nc4_basic` | `2.02 -> 3.79 Mops/s` (`x1 -> x8`) | `203.73 -> 176.70 Kops/s` | `netcdf-rust` scales; baseline does not |
| `parallel_slice_batch` `large_nc4_compressed` | `78.45 -> 252.23 Kops/s` (`x1 -> x8`) | `166.56 -> 62.52 Kops/s` | repeated tiny-slice throughput scales better |
| `parallel_open_and_read` `large_nc4_compressed` | `497 MiB/s -> 1.60 GiB/s` (`x1 -> x4`) | `383 MiB/s -> 1.22 GiB/s` | `netcdf-rust` leads through `x4` |
| `read_full_internal_parallel_nocache` | `362 MiB/s -> 1.23 GiB/s` (`x1 -> x8`) | n/a | cold chunk work still parallelizes |

### Classic Internal Parallel Follow-Up

A targeted Docker run on 2026-05-28 compared the previous classic local-read
policy against the storage-aware policy and final-buffer decode path. The
classic fixtures in this benchmark are still only a few MiB, so these numbers
mainly exercise Rayon overhead control rather than PnetCDF-scale throughput.

| group | current/base median geomean | result |
| --- | ---: | --- |
| `read_full_internal_parallel_classic` | `0.886x` | 11.4% faster |
| `slice_internal_parallel_classic` | `0.553x` | 44.7% faster |

Selected median deltas from that run:

| workload | base | current | result |
| --- | ---: | ---: | --- |
| full `large_cdf5` `x1` | 2.307 ms | 2.366 ms | 2.6% slower |
| full `large_cdf5` `x4` | 5.030 ms | 2.443 ms | 51.4% faster |
| full `large_cdf5` `x8` | 2.887 ms | 2.453 ms | 15.0% faster |
| full `large_record_cdf5` `x1` | 1.531 ms | 1.282 ms | 16.2% faster |
| slice `large_cdf5` `x8` | 629.2 us | 535.0 us | 15.0% faster |
| slice `large_cdf5` `x16` | 3.831 ms | 536.9 us | 86.0% faster |
| slice `large_record_cdf5` `x1` | 1.617 ms | 741.0 us | 54.2% faster |

## Interpretation

- `netcdf-rust` is faster than `georust/netcdf` on the measured classic-format
  reads, including large CDF-5.
- `netcdf-rust` is also ahead on NetCDF-4 metadata, nested groups, and
  small-to-medium reads in this run.
- `netcdf-rust` scales materially better on repeated concurrent metadata and
  tiny-slice workloads in this environment.
- The `georust/netcdf` baseline uses a shared process-global mutex around FFI
  calls into `netcdf-c`, and it does not show the same aggregate throughput scaling.
- The previously remaining paired loss in
  `read_full_reuse_handle/large_nc4_compressed` is now ahead in the latest
  focused reruns for that group.
- The targeted `slice_reuse_handle_hdf5_backend/large_nc4_compressed` case is
  now slightly ahead rather than behind.
- Warm single-read scaling now appears more constrained by this host than by
  obvious decoder serialization.
- The 2026-05-28 classic internal-parallel follow-up shows that storage-aware
  thresholds avoid most local Rayon overhead on small CDF-5 fixtures, while
  decoding directly into final buffers materially improves planned record
  slices.

## Limits

- This report reflects one local system.
- GitHub Actions is useful for benchmark regression checks, but not for authoritative
  performance claims on bandwidth-sensitive workloads like these.
- The large warm cached single-read results should be interpreted as
  machine-specific latency behavior, not as a general statement about internal
  parallelism.
