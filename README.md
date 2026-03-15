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

// Read data (classic format)
if let Some(classic) = file.as_classic() {
    let temp: ndarray::ArrayD<f32> = classic.read_variable("temperature")?;
}
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
- V1 B-tree, V2 B-tree, single-chunk, and implicit chunk indexing
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
- Dimension reconstruction from HDF5 dimension scales
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

## Known limitations

- Fixed array and extensible array chunk indexing are not yet supported (returns an error)
- Soft and external HDF5 links are skipped
- SZIP, N-Bit, and ScaleOffset filters are not built in (register via `FilterRegistry`)
- SOHM (shared object header message table) resolution is deferred
- NetCDF-4 dimension matching uses size-based heuristics rather than parsing `DIMENSION_LIST` object references

## License

MIT OR Apache-2.0
