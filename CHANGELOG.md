# Changelog

## Unreleased

- Added coordinate-variable metadata helpers and CF discovery APIs for coordinate axes and time coordinates, including NetCDF-4 dimension-scale coordinate variables.

## 0.3.0 - 2026-04-16

- Added range-backed open APIs in `hdf5-reader` and `netcdf-reader`, including `from_storage`/`from_storage_with_options`, plus public `Storage`/`DynStorage` reexports for custom backends.
- Added `NcMetadataMode` and `NcOpenOptions::metadata_mode` so NetCDF-4 callers can choose strict metadata reconstruction or lossy fallback heuristics for malformed files.
- Changed NetCDF-4 metadata handling to stay lazy and path-local, reusing immutable cached metadata trees instead of the old pointer-based metadata arena.
- Changed `NcFile::{root_group, dimensions, variables, global_attributes}` to return `Result<_>` so NetCDF-4 metadata errors can surface at access time instead of forcing eager reconstruction during open.
- Improved NetCDF classic contiguous slice reuse and NetCDF-4 metadata lookup locality to cut repeated setup work and unnecessary full-file metadata fallbacks.
- Fixed storage-backed format detection to handle short backing stores consistently with the byte-slice and file-based open paths.
- Clarified the top-level safety wording in `README.md` so release docs no longer imply that all internal `unsafe` is limited to `memmap2`.
- Expanded `RELEASING.md` with explicit version-bump steps and the staged `hdf5-reader`/`netcdf-reader` packaging flow required by the publish order.

## 0.2.0 - 2026-04-01

- Added built-in HDF5 `N-Bit` and `ScaleOffset` decoding in `hdf5-reader`, closing a zero-config compatibility gap for NetCDF-4 and direct HDF5 reads. `SZIP` remains opt-in via `FilterRegistry`, and ScaleOffset floating-point E-scale remains unsupported.
- Changed `hdf5_reader::filters::FilterRegistry::register` callbacks to receive the full `FilterDescription` alongside the raw bytes and element size, so custom filters can use HDF5 filter metadata when decoding.
- Fixed the `ScaleOffset` full-precision reverse path so chunks that store the raw payload after the filter header bypass postprocessing correctly.

## 0.1.4 - 2026-03-25

- Added `Hdf5File::{from_bytes_with_options, from_vec_with_options}` and `NcFile::from_bytes_with_options` so custom cache and filter settings also work for in-memory reads.
- Added string dataset/variable APIs:
  `hdf5_reader::Dataset::{read_string, read_strings}` and
  `netcdf_reader::NcFile::{read_variable_as_string, read_variable_as_strings}`.
- Fixed NetCDF-4 string typing so `NC_STRING` variables backed by HDF5
  variable-length byte storage are exposed as `NcType::String`.
- Removed extra open/remap work in the NetCDF open path and restored classic
  open performance after the detection refactor.

## 0.1.2 - 2026-03-22

Patch release to upgrade the HDF5 caching layer from `lru 0.12` to the fixed `lru 0.16.3` line.

## 0.1.1 - 2026-03-19

Patch release to move the public `ndarray` dependency to `0.17` for better compatibility with zarr-oriented downstream code.

## 0.1.0 - 2026-03-18

Initial public release.

- Added a pure-Rust, read-only HDF5 decoder in `hdf5-reader`
- Added a pure-Rust NetCDF reader in `netcdf-reader` covering CDF-1/2/5 and NetCDF-4
- Added chunked I/O, filter support, parallel read paths, and cache configuration
- Added Criterion benchmarks against the C-backed `netcdf` crate plus CI benchmark regression checks
