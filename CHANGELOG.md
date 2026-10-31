# Changelog

## Unreleased

### Improved

- reuse parsed dataset metadata and attribute storage across hard-link aliases,
  so listing a group with many aliases does not reread its dense attribute
  tables for every link

## 0.12.0 - 2026-10-31

### Breaking changes

- replace `MaskParams::missing_value: Option<f64>` with
  `MaskParams::missing_values: Vec<f64>` to support every sentinel in a CF
  `missing_value` attribute. For manual construction, use `vec![value]` for
  one sentinel or `Vec::new()` for none (#122)
- add `NcOpenOptions::{max_chunk_bytes, max_read_bytes,
  max_object_header_bytes}` when the `netcdf4` feature is enabled. Update
  exhaustive struct literals or use `..NcOpenOptions::default()` (#124)
- add a `dim_id_map: &HashMap<i64, NcDimension>` parameter before
  `metadata_mode` to `nc4::variables::{extract_variable, extract_variables,
  extract_variables_from_datasets}`. It maps `_Netcdf4Dimid` IDs to visible
  dimensions so multi-dimensional coordinate variables can resolve their
  `_Netcdf4Coordinates` metadata; include inherited dimensions when calling
  these functions directly (#125)
- add `NcAttrValue::UserDefined { dtype, values }` for decoded compound,
  opaque, and non-string vlen attributes. Update exhaustive matches on
  `NcAttrValue`; values use the existing `NcValue` representation (#133)

### Added

- decode SZIP-compressed HDF5 and NetCDF-4 datasets, including nearest-neighbor
  and entropy-coding modes (#130)
- open NetCDF files from owned or shared bytes without a second full-file
  copy with `NcFile::{from_vec, from_shared_bytes}` or
  `NcFile::from_owner(owner, options)`; these entry points require the
  `netcdf4` feature and accept both NetCDF-4 and classic files. Classic
  `from_storage` opens also remain range-backed instead of reading the
  whole file into memory (#126)
- expose HDF5 chunk, selection, and object-header allocation limits through
  `NcOpenOptions`. Defaults are 256 MiB per chunk, unlimited per selection,
  and 64 MiB per object header (#124)

### Improved

- replace quadratic duplicate checks when listing dense HDF5 groups with
  a hash set, preserving aliases, external-file identities, and listing order (#132)
- cache parsed global heap collections per file so repeated variable-length
  string and dimension-reference reads reuse them (#117)

### Fixed

- split writer variable-length data across global heap collections after
  65,535 objects instead of overflowing the 16-bit object index (#118)
- read HDF5 and NetCDF-4 files behind a user block. `Superblock::base_address`
  now reports the byte position where the superblock was found, which is the
  effective base for file addresses, rather than the value recorded in the
  superblock (#119)
- return errors for non-null variable-length references to missing global
  heap objects instead of substituting empty strings (#117)
- report a NetCDF-4 unlimited dimension at the longest variable extent
  across its defining group and descendant groups, including when its
  dimension scale is shorter (#120). Shorter variables expose this logical
  shape and pad unwritten records with fill values in full, sliced, buffer,
  string, user-defined, and parallel reads; child dimensions that shadow
  ancestor names remain independent (#133)
- honor CF `_Unsigned` in masked and unpacked reads, mask every value in a
  vector `missing_value` attribute, and compare masking attributes in the
  variable's stored type before unpacking (#122)
- decode compound, opaque, and non-string vlen attributes on NetCDF-4
  groups and variables, including nested fields and heap-backed sequences.
  Expose enum attributes as their underlying integer values (#123, #133)
- preserve every dimension of multi-dimensional coordinate variables using
  `_Netcdf4Coordinates`, instead of reporting them as one-dimensional (#125)
- accept absolute and slash-normalized variable paths in classic NetCDF
  reads, matching the variable lookup API (#127)
- bound fuzz-target reads before allocating so mutated sparse dataset
  extents cannot trigger enormous allocations under ASan (#124, #128)
- read v2 B-trees with nested internal nodes, including the dense link
  indexes of groups with thousands of members (#129)

## 0.11.0 - 2026-10-30

### Added

- open HDF5 files from shared or borrowed-owner bytes without copying them:
  `Hdf5File::from_shared_bytes` (`Arc<[u8]>`) and `Hdf5File::from_owner` (any
  `AsRef<[u8]> + Send + Sync` owner), `BytesStorage::from_shared` and
  `BytesStorage::from_owner`, and `StorageBuffer::from_shared` and
  `StorageBuffer::from_shared_owner` for custom storage backends (#114)
- add `OpenOptions::max_read_bytes`, a limit on the decoded bytes one dataset
  read may allocate, which is unlimited by default, and
  `OpenOptions::max_object_header_bytes`, a limit on the bytes of one object
  header across its continuation chunks, which defaults to 64 MiB. Both refuse
  with `Error::ResourceLimit` before reading or allocating (#113)

### Improved

- `Hdf5File::from_vec`, `BytesStorage::new` and `StorageBuffer::from_vec` no
  longer copy their bytes into a new `Arc<[u8]>`, so an in-memory open needs no
  second buffer and each positional `FileStorage` read is copied once, not
  twice (#114)

### Fixed

- reject datatypes with a zero element size (fixed-point, floating-point,
  time, bitfield, opaque, compound, and reference) and array datatypes with a
  zero-sized dimension, as libhdf5 does. A zero-width element made every
  byte-length check pass, so a large dataset read back as no bytes, or a typed
  read aborted trying to allocate terabytes for its output (#113)
- report numeric and reference datatypes wider than 255 bytes as unsupported
  instead of truncating their size (256 bytes read as 0) (#113)

## 0.10.1 - 2026-10-28

### Improved

- look up only selected row and column chunks for chunked HDF5 gathers,
  including v1 and v2 B-tree indexes. Sparse selections avoid walking
  unrelated B-tree branches and intervening row chunks (#111)

## 0.10.0 - 2026-10-26

### Added

- expose exact on-disk datatype encodings for HDF5 datasets and attributes,
  including attributes stored in dense groups (#108)
- add chunked row/column gathers, including parallel variants, and uncached
  full reads; sparse column selections skip unrelated chunk-index entries
  (#106)
- add `Group::member_names` and cache old-style group listings so repeated
  lookups avoid reading and parsing the same group metadata (#107)
- add builder methods for `OpenOptions` and a configurable per-chunk allocation
  limit, which defaults to 256 MiB (#104)

### Fixed

- read extensible-array chunk indexes through secondary blocks and paged data
  blocks, and use maximum dimension sizes when numbering their chunks (#102,
  #105)
- read paged fixed-array chunk indexes with the correct page bitmap bit order,
  checksum offset, and reserved page spacing; sparse files no longer return
  fill values for initialized chunks
- handle shared message locations, old-style group soft links, and link
  resolution across soft and external links (#103)
- reject malformed HDF5 size, offset, filter, attribute, and chunk metadata
  with errors instead of overflowing, hanging, or aborting; release in-flight
  chunk loads if the loader panics (#104, #109)
- avoid holding the old-style group cache mutex during storage reads (#107)

## 0.9.3 - 2026-10-20

### Added

- open plain HDF5 files (written by h5py or libhdf5, with no `_NCProperties`
  attribute or dimension scales) through `NcFile::open` as netcdf-c does:
  datasets without `DIMENSION_LIST` get phony dimensions named by size instead
  of failing strict metadata validation, and a second axis of the same size is
  named `dim_<size>_2` rather than reusing the first axis's name (#99)

### Fixed

- fail reads with an error instead of aborting the process on corrupt size
  fields: classic selections larger than the file are refused before
  allocating, HDF5 selection and fill buffers are allocated fallibly, and the
  DEFLATE and LZ4 filters no longer reserve their whole output limit up front
  (#100)

## 0.9.2 - 2026-10-18

### Fixed

- write every chunked HDF5/NetCDF-4 variable with a version-3 layout and a v1
  B-tree chunk index, as netcdf-c does, instead of the v4 layout's implicit,
  single-chunk, fixed-array, and v2 B-tree indexes, which netcdf-java (and so
  Panoply) fails to read (#88)
- read version 1/2 compound datatypes whose member names start off the 8-byte grid:
  a name is padded to a multiple of eight bytes from its own start, not from the
  start of the buffer, so a member after a float (20-byte description) no longer
  derails the parse (`unsupported datatype class`, `unexpected end of data`).
  HEC-RAS geometry and plan files hit this in every attribute table
- read the old fill value message (type 0x0004) as a size followed by the value;
  the size was taken as part of the value, so unwritten chunks read back wrong
- list a group's children in link creation order when the group tracks it, as
  libhdf5's creation-order index and netcdf-c do, instead of on-disk storage
  order; NetCDF-4 variables now come back in netcdf-c's order
- read "huge" fractal-heap objects addressed through an indirect heap ID: the
  v2 B-tree key is `min(heap ID length - 1, 8)` bytes, as in libhdf5, rather than
  a full-width length field, so large attributes in dense attribute storage
  (8-byte heap IDs) no longer fail with "huge fractal heap ID is too short"
- read variable-length strings whose base type is a one-byte fixed string (as
  rust-hdf5 writes them) as well as a one-byte integer (as libhdf5 does); netcdf-c
  treats both as NC_STRING, but attributes written by rust-hdf5 failed with
  "unsupported NetCDF-4 datatype"

## 0.9.1 - 2026-07-28

### Fixed

- read NetCDF-4 files containing scalar (rank-0) variables: such variables carry
  no `DIMENSION_LIST` attribute, which previously failed the entire open under
  `NcMetadataMode::Strict` rather than resolving to an empty dimension list

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
