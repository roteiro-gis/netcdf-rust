# netcdf-rust

Pure-Rust, read-only decoders for HDF5 and NetCDF files. No C libraries, no build scripts, no unsafe beyond `memmap2`.

## Crates

| Crate | Description |
|---|---|
| `hdf5-reader` | Low-level HDF5 decoder (superblock, object headers, B-trees, chunked I/O, filters) |
| `netcdf-reader` | NetCDF reader supporting CDF-1/2/5 classic and NetCDF-4 (HDF5-backed) formats |

## Usage

```rust
use netcdf_reader::NcFile;

let file = NcFile::open("era5.nc")?;
println!("format: {:?}", file.format());

for dim in file.dimensions() {
    println!("  dim: {} ({})", dim.name, dim.size);
}

for var in file.variables() {
    println!("  var: {} {:?}", var.name(), var.shape());
}

// Read data (works for both classic and NetCDF-4)
let temp: ndarray::ArrayD<f32> = file.read_variable("temperature")?;
```

Using `hdf5-reader` directly:

```rust
use hdf5_reader::Hdf5File;

let file = Hdf5File::open("data.h5")?;
let ds = file.dataset("/group1/temperature")?;
let data: ndarray::ArrayD<f64> = ds.read_array()?;

// Hyperslab selection
use hdf5_reader::{SliceInfo, SliceInfoElem};
let sel = SliceInfo {
    selections: vec![
        SliceInfoElem::Slice { start: 0, end: 10, step: 1 },
        SliceInfoElem::Index(5),
    ],
};
let slice: ndarray::ArrayD<f64> = ds.read_slice(&sel)?;
```

## Features

**HDF5 reader:**
- Superblock v0-v3, object header v1/v2 with checksum verification
- Compact, contiguous, and chunked data layouts
- All chunk index types: V1/V2 B-tree, single-chunk, implicit, Fixed Array, Extensible Array
- Layout v4 and v5 (HDF5 2.0)
- Filter pipeline: deflate (zlib), shuffle, Fletcher-32
- Pluggable `FilterRegistry` for custom filters (Blosc, LZ4, etc.)
- Fractal heap + B-tree v2 dense link resolution
- Shared (committed) datatype resolution
- Variable-length string reading via global heap
- Object reference resolution
- LRU chunk cache with configurable size
- Object header cache for repeated access
- Memory-mapped I/O or owned-byte (`from_vec`) access

**NetCDF reader:**
- CDF-1 classic, CDF-2 64-bit offset, CDF-5 64-bit data
- NetCDF-4 via `hdf5-reader` (feature-gated, on by default)
- Automatic format detection from magic bytes
- Dimension reconstruction via `DIMENSION_LIST` object references (with size-based fallback)
- Unified `NcFile::read_variable::<T>()` across all formats
- Attribute filtering (hides internal `_NCProperties`, `DIMENSION_LIST`, etc.)
- Record (unlimited) dimension support

## Feature flags

```toml
[dependencies]
netcdf-reader = "0.1"           # CDF-1/2/5 + NetCDF-4 (default)
netcdf-reader = { version = "0.1", default-features = false }  # CDF-1/2/5 only
```

| Flag | Default | Description |
|---|---|---|
| `netcdf4` | yes | NetCDF-4 support via `hdf5-reader` |
| `cf` | no | CF Conventions helpers (time decoding, CRS, axes) |

## Custom filters

Register filters before opening files:

```rust
use hdf5_reader::{Hdf5File, OpenOptions};
use hdf5_reader::filters::FilterRegistry;

let mut registry = FilterRegistry::new();
registry.register(32001, Box::new(|data, _elem_size| {
    // Custom decompression logic
    Ok(data.to_vec())
}));

let file = Hdf5File::open_with_options("data.h5", OpenOptions {
    filter_registry: Some(registry),
    ..Default::default()
})?;
```

## Testing

```sh
# Unit tests (no external dependencies)
cargo test

# Integration tests with generated fixtures
pip install h5py netCDF4 numpy
python testdata/generate_fixtures.py
cargo test
```

## Benchmarks

`netcdf-reader` includes a Criterion benchmark that compares this implementation
against the C-backed [`netcdf`](https://github.com/georust/netcdf) crate on the
checked-in fixtures plus larger generated benchmark fixtures.

```sh
# Full benchmark matrix
cargo bench -p netcdf-reader --bench compare_georust

# Contention-focused scaling run that makes cross-thread serialization visible
BENCH_THREAD_LIST=1,2,4,8 BENCH_HOT_OPS_PER_THREAD=256 \
  cargo bench -p netcdf-reader --bench compare_georust \
  'parallel_metadata_batch/.*/(nc4_basic|nested_nc4_groups)|parallel_slice_batch/.*/(nc4_basic|nc4_compressed|large_nc4_compressed)'

# Restrict thread scaling cases
BENCH_THREAD_LIST=1,2,4 cargo bench -p netcdf-reader --bench compare_georust

# Summarize the latest Criterion results as a markdown table
python3 scripts/criterion_summary.py

# Include x1-relative speedup for threaded workloads
python3 scripts/criterion_summary.py --speedup \
  --group parallel_metadata_batch \
  --group parallel_slice_batch \
  --group read_full_internal_parallel \
  --group parallel_open_and_read
```

Notes:
- The benchmark uses `netcdf` with its `static` feature, so it builds a bundled
  `netcdf-c` stack instead of depending on a specific system HDF5 install.
- The suite separates `open_only`, `metadata_reuse_handle`,
  `read_full_reuse_handle`, `open_and_read_full`,
  `read_full_internal_parallel`, `slice_reuse_handle_hdf5_backend`, and
  parallel throughput workloads so setup costs and steady-state read costs are
  visible independently.
- The suite also includes `parallel_metadata_batch` and
  `parallel_slice_batch`, which keep one open handle per worker, synchronize
  start with a barrier, and then execute many small operations. Those runs are
  intended to show how `netcdf-rust` and the C-backed `georust/netcdf`
  baseline behave under the same concurrent call pattern.
- Larger benchmark-only fixtures are generated at runtime, so the suite covers
  both small checked-in files and more realistic compressed datasets.
- Parallel cases include both independent `open + read` scaling against the
  C-backed baseline and a `parallel_read_shared_cairn` case that measures our
  shared-open-handle throughput directly.
- `read_full_internal_parallel` measures one large chunked read with internal
  chunk-level Rayon parallelism, which is distinct from the independent-read
  throughput cases.
- In the local `netcdf 0.12.0` baseline source used for these runs, libnetcdf
  entry points are wrapped through `with_lock` / `checked_with_lock`, which
  take `netcdf_sys::libnetcdf_lock` before entering the C API. In practice,
  that means a shared process-global mutex around FFI calls into `netcdf-c`.
  The contention benchmarks are intended to make the effect of that design
  visible.
- The slice workload is NetCDF-4 only and uses the native HDF5 dataset API on
  our side, because `netcdf-reader` does not yet expose high-level sliced reads.

## Known limitations

- Soft and external HDF5 links are skipped
- SZIP, N-Bit, and ScaleOffset filters are not built in (register via `FilterRegistry`)
- SOHM (shared object header message table) resolution is deferred

## License

MIT OR Apache-2.0
