# Changelog

## Unreleased

## 0.9.0 - 2026-07-23

### Writers (first release of `hdf5-writer` and `netcdf-writer`)

- validate all writer output against the reference C libraries (libhdf5 via
  h5py, netcdf-c via netCDF4-python) with a new harness, in addition to
  round-tripping through the reader crates
- emit Link Info and Group Info messages on HDF5 groups so libhdf5 accepts the
  files (previously every writer file was rejected)
- fix HDF5 encoding to match libhdf5: floating-point sign-bit position,
  variable-length string datatype, the v4 chunk-layout element-size dimension,
  fixed-array page bits and variable chunk-size field width, the global-heap
  free-space object, and empty variable-length references
- write classic `vsize` padded to 4 bytes and implement the single-record-
  variable no-padding exception, matching netcdf-c byte-for-byte; the reader
  now recomputes record strides from dimensions so it reads conformant files
  with a lone odd-sized record variable
- emit a version-2 B-tree chunk index for datasets with unlimited maximum
  dimensions (the implicit and fixed-array indices are only valid for fixed
  max dims)
- emit `REFERENCE_LIST` back-references on dimension scales so HDF5
  dimension-scale tooling (`h5py` H5DS) sees data variables as attached
- fill native-endian slice growth gaps with the variable's fill value instead
  of zeros
- eliminate redundant full-file copies in the HDF5 write path
- fix `netcdf-writer` without default features so classic-only builds retain
  byte conversion, fill-value, and output-writing support
- reject zero-sized HDF5 datatypes and zero-length array dimensions before
  producing invalid files

### Reader hardening

- guard object-header continuation following with a visited-set and depth
  limit, and check continuation offset arithmetic, to stop infinite loops on
  crafted files
- limit datatype-parsing recursion depth to prevent stack overflow on deeply
  nested compound/array/enum/vlen datatypes
- bound classic attribute and record-count allocations by the available file
  size so a tiny crafted header cannot force a multi-gigabyte allocation
- bound the variable dimension-count and the parallel non-record read against
  the file size before allocating (found by the new mutation property test)
- parse libhdf5's filtered v2 B-tree chunk records, which use a minimal
  chunk-size field width rather than the file's global length size
- resolve shared object header message (SOHM) attributes on the group path,
  matching the dataset path
- reject chunk dimensions that exceed `u32` and bound the implicit chunk-index
  entry count by the file size

### Tests and infrastructure

- add a classic-NetCDF parser fuzz target and seed both fuzz corpora
- add a writer→reader roundtrip property test (a true classic-format oracle)
  and a reader byte-mutation property test with bounded-allocation assertions
- add unit tests for the chunk-grid coverage gate that guards the
  uninitialized-output fast path
- run core-crate unit tests on the MSRV toolchain and test classic-only
  parallel reads plus classic-only writes without the HDF5 stack

### Other

- allow parallel classic reads without enabling the HDF5 backend
  (`rayon` no longer implies `netcdf4`)
- bound the unlimited LZ4 decompression path's declared output size by the
  input length
- remove the unreachable `Hdf5Writer` finalized flag
- make zero-copy initialization boundaries explicitly unsafe, document every
  unsafe operation's invariants, and enforce both checks with workspace lints
- fix rustdoc links and adopt macro fragment specifiers that retain their
  intended meaning in future Rust editions
- update the audited development dependency graph to `anyhow` 1.0.103 and
  `rand` 0.9.3

## 0.8.0 - 2026-07-10

- breaking: move shared public HDF5 format types and NetCDF data-model types
  into the new `hdf5-core` and `netcdf-core` crates, while re-exporting them
  from the reader crates at their existing paths
- breaking: add `decoded_strings` to the public `hdf5_reader::Attribute` type
  so byte-backed and range-backed opens can expose decoded variable-length
  string attributes

- add independently publishable `hdf5-core` and `netcdf-core` crates used by
  the reader crates
- decode variable-length HDF5 string attributes for both byte-backed and
  storage-backed opens
- add NetCDF-4 `NC_CHAR` variable reading through the unified string API
- classify HDF5 child objects containing attributes but no dataset messages as
  groups instead of dropping them from group traversal
- correct implicit HDF5 chunk enumeration for empty unlimited dimensions
- optimize native 2D unit-stride HDF5 slices contained within one chunk
- fix dense HDF5 attribute name and creation-order B-tree records to decode
  fields in their on-disk order, preventing invalid fractal-heap offsets when
  NetCDF-4 objects cross the compact-to-dense attribute threshold

- keep the in-development `hdf5-writer` and `netcdf-writer` crates unpublished
  in this release
- add the initial `hdf5-writer` crate and HDF5 file-emission path used by the
  NetCDF-4 writer bridge
- add compact datasets, nested groups, and chunked datasets using implicit and
  fixed-array chunk indexes
- add HDF5 fill values plus Deflate and shuffle filter pipelines
- add fixed and variable-length string data, variable-length sequence datasets
  and attributes, and enhanced enum, opaque, compound, and array datatypes
- support filtered variable-length datasets and empty HDF5 variable-length
  references

- add the initial classic CDF writer and NetCDF-4 writer bridge
- emit NetCDF-4 dimension-list metadata and support fixed, unlimited, and
  multiple-unlimited dimensions
- add nested group authoring and NetCDF-4 string attributes
- add `NC_CHAR`, `NC_STRING`, enum, opaque, compound, fixed-size array, and
  variable-length sequence variables, including typed enhanced-value writes
- add variable storage controls for layout, chunking, and filters
- add explicit fill values and default-fill prewrites for classic and NetCDF-4
  variables
- add hyperslab writes and unlimited-dimension appends for numeric, character,
  string, variable-length, and enhanced variables, including character-string
  convenience writes
- harden multi-unlimited slice planning and writer size/address accounting

- add package checks and release-order documentation for the new core crates

## 0.7.0 - 2026-06-23

- breaking: change `netcdf_reader::classic::data::compute_record_stride` to return
  `Result<u64>` and reject padded record-size and record-stride overflows instead
  of wrapping
- on Unix, open filesystem external raw-data and external-link resolver paths
  with directory-relative `openat` traversal so path validation and file opening
  share the same trusted root, closing symlink-swap races in attacker-writable
  trees
- infer classic CDF-1/2 streaming record counts from storage length, update the
  unlimited dimension size accordingly, and ignore trailing partial records
- fix chunked HDF5 variable-length string/sequence slices and fill values to use
  the file's actual vlen reference width instead of the fixed 16-byte datatype
  API size
- reject HDF5 integer reads when the Rust signedness does not match the file
  datatype, including bulk decode and native-copy paths
- bound HDF5 object-header storage reads by the parsed prefix/chunk sizes and
  reject oversized continuation or chunk lengths before platform casts
- check HDF5 superblock magic-search offset arithmetic for overflow on huge
  range-backed storage
- add a real B-tree v2 chunk-index fixture and parse chunk offsets as scaled
  dataset element coordinates
- reject malformed B-tree v2 records that consume more bytes than the declared
  record size or have record sizes too small for fixed fields
- preserve B-tree v2 internal-node records in traversal order so records from
  depth > 0 trees are not skipped
- reject repeated B-tree v1/v2 nodes and repeated fractal-heap indirect blocks,
  and bound recursive traversal depth and indirect row counts
- use B-tree v2 link-name hash ranges to prune dense group lookup instead of
  enumerating every dense link before matching a requested name
- update `memmap2` to 0.9.11
- add CodeQL analysis and run the HDF5 open fuzz target in CI
- clarify README dependency-scope wording for the published pure-Rust crates

## 0.6.1 - 2026-06-11

- fix version 1 B-tree raw-data chunk keys to read each per-dimension chunk
  offset as a fixed 8-byte value instead of `size_of_offsets` bytes, so chunked
  datasets in 32-bit-superblock files (`size_of_offsets = 4`) locate their chunk
  addresses correctly instead of failing to decompress
- require benchmark regression CI failures to exceed both the percentage
  threshold and a minimum absolute slowdown, reducing false positives for
  low-microsecond workloads on shared runners
- stop hard-gating CI on the end-to-end parallel open-and-read benchmark, which
  is dominated by thread scheduling and filesystem/cache noise on shared runners

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
