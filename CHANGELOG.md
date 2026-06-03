# Changelog

## Unreleased

## 0.6.0 - 2026-06-03

- require an explicit HDF5 external raw-data resolver, and make filesystem external raw/link resolvers reject absolute paths, parent components, and canonical paths outside the resolver base directory
- stop filtered HDF5 chunk and fractal-heap decompression at the expected decoded size plus one byte before final size validation
- return fallible results from public HDF5/NetCDF element-count and type-size helpers, and reject shape, array-type, and implicit chunk size overflows
- read HDF5 extensible-array chunk index blocks with computed block lengths instead of reading metadata blocks through end-of-file
- add internal Rayon parallelism for classic NetCDF CDF-1/2/5 full-variable reads and hyperslab slices
- add storage-aware classic parallel read policy so local byte and mmap reads use larger thresholds and chunks, while range-backed storage can still use more aggressive request parallelism
- optimize classic parallel reads and planned slice spans to decode directly into final result buffers instead of allocating per-chunk vectors and flattening them
- keep classic `from_storage` opens range-backed after header parsing so variable reads can use the caller's storage backend instead of copying the whole file into memory
- add committed compatibility fixtures and docs for PnetCDF-style CDF-1/2/5 and parallel NetCDF-C/HDF5-produced files
- add CDF-5 huge-dimension overflow coverage and explicit unsupported-feature errors for classic subfiling markers, including lowercase marker attributes
- fix classic range-header helper builds when `netcdf-reader` is compiled without default NetCDF-4 support
- fix `hdf5-reader` local file storage linkability on `wasm32` targets

## 0.5.0 - 2026-05-17

- add generic range-request storage helpers and block-cache storage wrappers for HTTP, S3, object-store, and other range-capable backends
- add streaming and performance APIs for HDF5 and NetCDF-4 reads, including `read_into`, raw/native-endian byte reads, decoded chunk iteration, and cache stats
- expose HDF5 external raw-file and external-link resolvers through `netcdf_reader::NcOpenOptions`
- add NetCDF-4 user-defined type read APIs for enum, opaque, compound, fixed-size array, and non-string vlen variables, with both dynamic `NcValue` decoding and custom borrowed `NcValueView` decoders
- add support for filtered HDF5 fractal-heap managed and huge objects used by dense links and dense attributes
- add an MSRV CI job and change the minimum supported Rust version from 1.77 to 1.81 to match the current default/all-features dependency graph
- optimize classic NetCDF `read_variable_into` to decode non-record and record variables directly into caller-provided buffers
- fix dense-group fractal heap handling for generated NetCDF-4 files that store committed user-defined types alongside variables
- fix HDF5 compact and contiguous raw-data reads so corrupt or truncated allocated storage errors instead of returning zero-padded values
- fix group APIs so unknown or unsupported HDF5 object classes are not exposed as groups
- fix dense HDF5 attribute loading so B-tree, fractal-heap object, and attribute-message parse failures are reported instead of silently omitting attributes
- remove legacy internal in-memory HDF5 metadata paths now superseded by the storage-backed implementation

## 0.4.0 - 2026-04-28

- add coordinate-variable metadata helpers and CF discovery APIs for coordinate axes and time coordinates, including NetCDF-4 dimension-scale coordinate variables
- add exact CF calendar decoding via `CfDateTime`, including noleap, all_leap, 360_day, Julian, and standard Gregorian reform calendar arithmetic
- expand HDF5 compatibility with SOHM shared-message lookup, fractal heap tiny and unfiltered huge objects, external raw data file reads, and optional external-link resolution
- add HDF5 external file/link resolver APIs, including filesystem resolvers and path-backed external raw data resolution by default for direct HDF5 opens
- improve HDF5 contiguous and chunked read validation, full-chunk caching behavior, corruption-test coverage, and fuzz/CI release checks

## 0.3.0 - 2026-04-16

- add range-backed open APIs in `hdf5-reader` and `netcdf-reader`, including `from_storage`/`from_storage_with_options`, plus public `Storage`/`DynStorage` reexports for custom backends
- add `NcMetadataMode` and `NcOpenOptions::metadata_mode` so NetCDF-4 callers can choose strict metadata reconstruction or lossy fallback heuristics for malformed files
- change NetCDF-4 metadata handling to stay lazy and path-local, reusing immutable cached metadata trees instead of the old pointer-based metadata arena
- change `NcFile::{root_group, dimensions, variables, global_attributes}` to return `Result<_>` so NetCDF-4 metadata errors can surface at access time instead of forcing eager reconstruction during open
- improve NetCDF classic contiguous slice reuse and NetCDF-4 metadata lookup locality to cut repeated setup work and unnecessary full-file metadata fallbacks
- fix storage-backed format detection to handle short backing stores consistently with the byte-slice and file-based open paths
- clarify the top-level safety wording in `README.md` so release docs no longer imply that all internal `unsafe` is limited to `memmap2`
- expand `RELEASING.md` with explicit version-bump steps and the staged `hdf5-reader`/`netcdf-reader` packaging flow required by the publish order

## 0.2.0 - 2026-04-01

- add built-in HDF5 `N-Bit` and `ScaleOffset` decoding in `hdf5-reader`, closing a zero-config compatibility gap for NetCDF-4 and direct HDF5 reads; `SZIP` remains opt-in via `FilterRegistry`, and ScaleOffset floating-point E-scale remains unsupported
- change `hdf5_reader::filters::FilterRegistry::register` callbacks to receive the full `FilterDescription` alongside the raw bytes and element size, so custom filters can use HDF5 filter metadata when decoding
- fix the `ScaleOffset` full-precision reverse path so chunks that store the raw payload after the filter header bypass postprocessing correctly

## 0.1.4 - 2026-03-25

- add `Hdf5File::{from_bytes_with_options, from_vec_with_options}` and `NcFile::from_bytes_with_options` so custom cache and filter settings also work for in-memory reads
- add string dataset/variable APIs: `hdf5_reader::Dataset::{read_string, read_strings}` and `netcdf_reader::NcFile::{read_variable_as_string, read_variable_as_strings}`
- fix NetCDF-4 string typing so `NC_STRING` variables backed by HDF5 variable-length byte storage are exposed as `NcType::String`
- remove extra open/remap work in the NetCDF open path and restore classic open performance after the detection refactor

## 0.1.2 - 2026-03-22

- upgrade the HDF5 caching layer from `lru 0.12` to the fixed `lru 0.16.3` line

## 0.1.1 - 2026-03-19

- move the public `ndarray` dependency to `0.17` for better compatibility with zarr-oriented downstream code

## 0.1.0 - 2026-03-18

- initial public release
- add a pure-Rust, read-only HDF5 decoder in `hdf5-reader`
- add a pure-Rust NetCDF reader in `netcdf-reader` covering CDF-1/2/5 and NetCDF-4
- add chunked I/O, filter support, parallel read paths, and cache configuration
- add Criterion benchmarks against the C-backed `netcdf` crate plus CI benchmark regression checks
