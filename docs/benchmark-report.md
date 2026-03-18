# Benchmark Report

Date: 2026-03-17

This document summarizes the current benchmark suite for `netcdf-rust` and a
representative local run against the C-backed `georust/netcdf` baseline.
The goal is to capture the current performance shape across metadata, full
reads, slicing, and threaded throughput.

## System Under Test

- Machine: Apple M1
- CPU topology: 8 logical CPUs, 4 performance cores
- Memory: 16 GiB
- OS: macOS 13.0
- Architecture: `arm64`
- Rust toolchain: `cargo 1.92.0`

These numbers should be read as results for this machine, not as universal
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
```

Notes:

- Criterion is used for all timing.
- Both implementations read the same files and variable paths.
- Benchmarks validate checksum and shape parity before timing.
- Contention-oriented groups open one handle per worker, synchronize start, and
  then repeat small operations to make cross-thread serialization visible.

## Current Results

### Single-thread summary

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

### Threaded highlights

| workload | netcdf-rust | georust/netcdf | takeaway |
| --- | --- | --- | --- |
| `parallel_metadata_batch` `nc4_basic` | `2.02 -> 3.79 Mops/s` (`x1 -> x8`) | `203.73 -> 176.70 Kops/s` | `netcdf-rust` scales; baseline does not |
| `parallel_slice_batch` `large_nc4_compressed` | `78.45 -> 252.23 Kops/s` (`x1 -> x8`) | `166.56 -> 62.52 Kops/s` | repeated tiny-slice throughput scales better |
| `parallel_open_and_read` `large_nc4_compressed` | `497 MiB/s -> 1.60 GiB/s` (`x1 -> x4`) | `383 MiB/s -> 1.22 GiB/s` | `netcdf-rust` leads through `x4` |
| `read_full_internal_parallel_nocache` | `362 MiB/s -> 1.23 GiB/s` (`x1 -> x8`) | n/a | cold chunk work still parallelizes |

## Interpretation

- `netcdf-rust` is faster than georust/netcdf-C on classic format reads at all
  sizes, including large CDF-5 (previously a loss, now 1.6x faster due to bulk
  big-endian decode optimization)
- `netcdf-rust` is faster on NetCDF-4 metadata, nested groups, and small-to-medium
  reads
- `netcdf-rust` scales materially better on repeated concurrent metadata and
  tiny-slice workloads in this environment
- the `georust/netcdf` baseline uses a shared process-global mutex around FFI
  calls into `netcdf-c`, and it does not show the same aggregate throughput
  scaling in those workloads
- the previously remaining paired loss,
  `read_full_reuse_handle/large_nc4_compressed`, is now ahead in the latest
  focused reruns for that group
- the targeted `slice_reuse_handle_hdf5_backend/large_nc4_compressed` case is
  now slightly ahead rather than behind
- `netcdf-rust` benefits from concurrency in independent-read and cold-read paths
- warm single-read scaling is now constrained more by this host than by obvious
  decoder serialization

## Limits

- This report reflects one local system.
- GitHub Actions is useful for benchmark smoke tests, but not for authoritative
  performance claims on bandwidth-sensitive workloads like these.
- The large warm cached single-read results should be interpreted as
  machine-specific latency behavior, not as a statement that internal
  parallelism is ineffective in general.
