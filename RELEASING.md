# Releasing

This repository publishes crates that must be released in dependency order:

1. `hdf5-core`
2. `netcdf-core`
3. `hdf5-reader`
4. `hdf5-writer`
5. `netcdf-reader`
6. `netcdf-writer`

Publish verification for crates with newly versioned workspace dependencies will
fail until those dependencies have been published and the crates.io index has
updated. In particular, `hdf5-reader` and `hdf5-writer` depend on
`hdf5-core`; `netcdf-reader` depends on `netcdf-core` and, by default,
`hdf5-reader`; `netcdf-writer` depends on `netcdf-core` and, by default,
`hdf5-writer`.

## Version prep

Before running the release checks:

1. Update `[workspace.package].version` in `Cargo.toml`
2. Update intra-workspace dependency versions in crate manifests
3. Update versioned dependency snippets in `README.md`
4. Move release notes from `CHANGELOG.md` `Unreleased` into the new version heading
5. Commit those versioning changes before packaging or publishing

## Pre-release checks

```sh
cargo fmt --all -- --check
cargo clippy --all -- -D warnings
cargo test --workspace
cargo test -p hdf5-reader --no-default-features
cargo test -p netcdf-reader --no-default-features
cargo package -p hdf5-core --offline --locked
cargo package -p netcdf-core --offline --locked
```

Package each dependent crate only after its new dependencies are visible in the
crates.io index:

```sh
cargo package -p hdf5-reader
cargo package -p hdf5-writer
cargo package -p netcdf-reader
cargo package -p netcdf-writer
```

Optional but recommended:

```sh
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
```

## Publish order

```sh
cargo publish -p hdf5-core
cargo publish -p netcdf-core
# wait for crates.io index to update
cargo publish -p hdf5-reader
cargo publish -p hdf5-writer
# wait for crates.io index to update
cargo package -p netcdf-reader
cargo publish -p netcdf-reader
cargo package -p netcdf-writer
cargo publish -p netcdf-writer
```

After all publishes succeed:

1. Update `CHANGELOG.md` if needed
2. Create the git tag, for example `v<version>`
3. Push the tag
4. Create the GitHub release notes from the tag
