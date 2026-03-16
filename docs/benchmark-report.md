# Benchmark Report

Date: 2026-03-16

This document summarizes the current benchmark suite for `netcdf-rust` and a
representative local run against the C-backed `georust/netcdf` baseline.
The goal is to show benchmark coverage, measurement methodology, and the
current performance shape across metadata, full reads, slicing, and threaded
throughput.

## System Under Test

- Machine: Apple M1
- CPU topology: 8 logical CPUs, 4 performance cores reported by `sysctl`
- Memory: 16 GiB
- OS: macOS 13.0 (22A380)
- Architecture: `arm64`
- Rust toolchain: `cargo 1.92.0 (Homebrew)`

These numbers should be read as results for this machine, not as universal
throughput claims. The suite is designed to be representative across workload
types, but absolute numbers will move with CPU, memory bandwidth, storage, and
the local HDF5/netCDF stack.

## Benchmark Scope

The suite covers the main user-visible workloads for this repository:

- `open_only`: file open cost
- `metadata_reuse_handle`: metadata traversal after open
- `read_full_reuse_handle`: steady-state full variable reads
- `open_and_read_full`: end-to-end open plus full read
- `slice_reuse_handle_hdf5_backend`: windowed reads on chunked data
- `parallel_open_and_read`: independent concurrent reads
- `parallel_metadata_batch`: repeated metadata walks with one open handle per worker
- `parallel_slice_batch`: repeated tiny slice reads with one open handle per worker
- `parallel_read_shared_cairn`: shared-handle throughput for `netcdf-rust`
- `read_full_internal_parallel`: one large read using internal chunk-parallelism
- `read_full_internal_parallel_nocache`: the same single-read path with the
  chunk cache disabled to expose cold decompression cost

The compared baseline is `georust/netcdf`, which uses the native `netcdf-c`
library underneath.
In the local `netcdf 0.12.0` source used for this run, libnetcdf calls are
wrapped through `with_lock` and `checked_with_lock`, which take
`netcdf_sys::libnetcdf_lock` before entering the C API. That is a reasonable
safety choice for the baseline. In practice, it means a shared
process-global mutex around FFI calls into `netcdf-c`, which is an important
part of how its threaded results should be interpreted.

## Methodology

Commands used for the current report:

```sh
cargo test

BENCH_THREAD_LIST=1,2,4,8 BENCH_HOT_OPS_PER_THREAD=256 \
  cargo bench -p netcdf-reader --bench compare_georust \
  'parallel_metadata_batch/.*/(nc4_basic|nested_nc4_groups)|parallel_slice_batch/.*/(nc4_basic|nc4_compressed|large_nc4_compressed)|read_full_internal_parallel(_nocache)?/.*/large_nc4_compressed|parallel_open_and_read/.*/large_nc4_compressed|parallel_read_shared_cairn/.*/large_nc4_compressed|open_only/.*/(cdf1_simple|nc4_basic|nc4_compressed|nc4_groups_nested)|metadata_reuse_handle/.*/(cdf1_simple|nc4_basic|nc4_compressed|nc4_groups_nested)|read_full_reuse_handle/.*/(cdf1_simple|nc4_basic|nc4_compressed|nc4_groups_nested)|open_and_read_full/.*/(cdf1_simple|nc4_basic|nc4_compressed|nc4_groups_nested)|slice_reuse_handle_hdf5_backend/.*/(nc4_compressed|large_nc4_compressed)' \
  -- --noplot --sample-size 10 --measurement-time 0.1 --warm-up-time 0.1

python3 scripts/criterion_summary.py --speedup \
  --group open_only \
  --group metadata_reuse_handle \
  --group read_full_reuse_handle \
  --group open_and_read_full \
  --group slice_reuse_handle_hdf5_backend \
  --group parallel_metadata_batch \
  --group parallel_slice_batch \
  --group parallel_open_and_read \
  --group parallel_read_shared_cairn \
  --group read_full_internal_parallel \
  --group read_full_internal_parallel_nocache
```

Benchmark design notes:

- Criterion is used for all timing.
- Both implementations read the same fixture files and variable paths.
- Benchmarks validate shape and checksum parity before timing.
- The contention-oriented groups open one handle per worker, synchronize thread
  start with a barrier, and then execute many small operations per thread.
  That removes most open cost and makes cross-thread serialization visible in
  aggregate `ops/s`.
- Threaded results are reported as throughput scaling, not single-request
  latency scaling, unless the benchmark explicitly measures one internally
  parallelized read.

## Representative Results

### Contention demonstration: repeated metadata walks

`parallel_metadata_batch` is designed to show whether multi-threaded callers can
increase aggregate metadata throughput inside one process. Each worker opens its
own handle once, waits on a barrier, and then performs 256 metadata walks.

`nc4_basic`:

| impl | x1 | x2 | x4 | x8 |
| --- | ---: | ---: | ---: | ---: |
| netcdf-rust | 2.02 Mops/s | 2.65 Mops/s | 3.42 Mops/s | 3.79 Mops/s |
| georust/netcdf | 203.73 Kops/s | 216.79 Kops/s | 206.31 Kops/s | 176.70 Kops/s |

`nested_nc4_groups`:

| impl | x1 | x2 | x4 | x8 |
| --- | ---: | ---: | ---: | ---: |
| netcdf-rust | 1.16 Mops/s | 2.00 Mops/s | 2.64 Mops/s | 2.79 Mops/s |
| georust/netcdf | 214.08 Kops/s | 188.86 Kops/s | 164.43 Kops/s | 136.52 Kops/s |

Takeaway: these runs show that `netcdf-rust` increases aggregate metadata
throughput under this concurrent access pattern, while the `georust/netcdf`
baseline remains roughly flat or declines as threads increase. That matches a
shared process-global mutex around the baseline's FFI calls into `netcdf-c`.

### Contention demonstration: repeated tiny slice reads

`parallel_slice_batch` uses tiny hot slices rather than large windows, again
with one open handle per worker and a synchronized start. This is intended to
stress call-path concurrency rather than chunk volume.

`nc4_basic`:

| impl | x1 | x2 | x4 | x8 |
| --- | ---: | ---: | ---: | ---: |
| netcdf-rust | 2.71 Mops/s | 2.57 Mops/s | 5.01 Mops/s | 3.29 Mops/s |
| georust/netcdf | 203.99 Kops/s | 121.32 Kops/s | 156.52 Kops/s | 85.67 Kops/s |

`nc4_compressed`:

| impl | x1 | x2 | x4 | x8 |
| --- | ---: | ---: | ---: | ---: |
| netcdf-rust | 411.71 Kops/s | 392.24 Kops/s | 1.08 Mops/s | 1.14 Mops/s |
| georust/netcdf | 165.09 Kops/s | 170.87 Kops/s | 124.03 Kops/s | 105.21 Kops/s |

`large_nc4_compressed`:

| impl | x1 | x2 | x4 | x8 |
| --- | ---: | ---: | ---: | ---: |
| netcdf-rust | 78.45 Kops/s | 26.16 Kops/s | 218.53 Kops/s | 252.23 Kops/s |
| georust/netcdf | 166.56 Kops/s | 171.10 Kops/s | 99.18 Kops/s | 62.52 Kops/s |

Takeaway: for tiny repeated slice operations, `netcdf-rust` continues to gain
aggregate throughput on these cases, while the `georust/netcdf` baseline does
not show the same scaling pattern. The `large_nc4_compressed` `x2` point for
`netcdf-rust` is noisy on this machine, but the overall trend still separates
the two approaches and is consistent with a shared process-global mutex around
the baseline's FFI calls into `netcdf-c`.

### Single-thread metadata and full reads

These smaller cases show the steady-state shape of the decoder and the relative
cost of metadata reconstruction:

| workload | netcdf-rust | georust/netcdf | result |
| --- | ---: | ---: | --- |
| `cdf1_simple` metadata | 28.5 us | 69.5 us | `netcdf-rust` 2.4x faster |
| `cdf1_simple` full read | 18.6 us | 59.5 us | `netcdf-rust` 3.2x faster |
| `nc4_basic` metadata | 45.8 us | 330.3 us | `netcdf-rust` 7.2x faster |
| `nc4_basic` full read | 53.9 us | 186.9 us | `netcdf-rust` 3.5x faster |
| `nc4_compressed` metadata | 56.3 us | 202.8 us | `netcdf-rust` 3.6x faster |
| `nc4_compressed` full read | 481.7 us | 604.5 us | `netcdf-rust` 1.25x faster |
| `nc4_groups_nested` full read | 65.6 us | 271.2 us | `netcdf-rust` 4.1x faster |

Takeaway: on classic metadata-heavy, nested-group, and moderate NetCDF-4 reads,
`netcdf-rust` is consistently ahead on this machine.

### Large compressed NetCDF-4, independent concurrent reads

`parallel_open_and_read` measures aggregate throughput from multiple threads,
where each thread opens the file and reads the full target variable.

| impl | median | throughput | speedup vs x1 |
| --- | ---: | ---: | ---: |
| netcdf-rust x1 | 16.086 ms | 497.32 MiB/s | 1.00x |
| netcdf-rust x2 | 17.580 ms | 910.13 MiB/s | 1.83x |
| netcdf-rust x4 | 19.511 ms | 1.60 GiB/s | 3.30x |
| netcdf-rust x8 | 34.957 ms | 1.79 GiB/s | 3.68x |
| georust x1 | 20.869 ms | 383.34 MiB/s | 1.00x |
| georust x2 | 21.426 ms | 746.74 MiB/s | 1.95x |
| georust x4 | 25.545 ms | 1.22 GiB/s | 3.27x |
| georust x8 | 32.289 ms | 1.94 GiB/s | 5.17x |

Takeaway: `netcdf-rust` leads through `x4` on this large compressed case. By
`x8`, both implementations are close enough that the host system is a major
part of the result.

### Large compressed NetCDF-4, one internally parallelized read

`read_full_internal_parallel` measures a single large read that parallelizes
chunk work inside `netcdf-rust`.

| impl | median | throughput | speedup vs x1 |
| --- | ---: | ---: | ---: |
| netcdf-rust x1 | 3.354 ms | 2.33 GiB/s | 1.00x |
| netcdf-rust x2 | 4.287 ms | 1.82 GiB/s | 0.78x |
| netcdf-rust x4 | 7.197 ms | 1.09 GiB/s | 0.47x |
| netcdf-rust x8 | 3.677 ms | 2.12 GiB/s | 0.91x |
| georust x1 | 2.761 ms | 2.83 GiB/s | 1.00x |

Takeaway: on this laptop-class system, warm cached single-read latency does not
scale with more threads. That points to host bandwidth and cache pressure as
the limiting factor once decompression is no longer dominant.

### Large compressed NetCDF-4, one internally parallelized cold read

`read_full_internal_parallel_nocache` disables the chunk cache to isolate the
cold path:

| impl | median | throughput | speedup vs x1 |
| --- | ---: | ---: | ---: |
| netcdf-rust x1 | 22.059 ms | 362.66 MiB/s | 1.00x |
| netcdf-rust x2 | 10.059 ms | 795.31 MiB/s | 2.19x |
| netcdf-rust x4 | 7.224 ms | 1.08 GiB/s | 3.05x |
| netcdf-rust x8 | 6.330 ms | 1.23 GiB/s | 3.48x |

Takeaway: the cold path scales materially, which indicates decompression and
chunk processing remain parallelizable and are not the same bottleneck as the
warm cached path.

### Large compressed NetCDF-4, shared-handle throughput

`parallel_read_shared_cairn` keeps one open handle and issues concurrent reads:

| impl | median | throughput |
| --- | ---: | ---: |
| netcdf-rust x2 | 4.993 ms | 3.13 GiB/s |
| netcdf-rust x4 | 6.581 ms | 4.75 GiB/s |
| netcdf-rust x8 | 11.725 ms | 5.33 GiB/s |

Takeaway: shared-handle throughput still scales, but with diminishing returns.
That is consistent with bandwidth pressure rather than a dominant software lock.

## Interpretation

The current suite is representative because it separates:

- open cost
- metadata traversal
- steady-state full reads
- end-to-end open plus read
- slice reads
- repeated tiny-operation contention under concurrency
- independent threaded throughput
- shared-handle throughput
- internal parallelism for a single large read
- warm cached versus cold no-cache behavior

The current results demonstrate that:

- `netcdf-rust` is already strong on classic files, NetCDF-4 metadata, and nested
  groups
- `netcdf-rust` scales materially better on repeated concurrent metadata and
  tiny-slice workloads in this environment
- the `georust/netcdf` baseline uses a shared process-global mutex around FFI
  calls into `netcdf-c`, and it does not show the same aggregate throughput
  scaling in those workloads
- `netcdf-rust` is competitive on large compressed reads
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
