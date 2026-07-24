# Releasing

This repository currently publishes these crates in dependency order:

1. `hdf5-core`
2. `netcdf-core`
3. `hdf5-reader`
4. `hdf5-writer`
5. `netcdf-reader`
6. `netcdf-writer`

Publish verification for crates with newly versioned workspace dependencies will
fail until those dependencies have been published and the crates.io index has
updated. The reader and writer packages also carry sibling crates as
development dependencies, so keep the order above and wait for each published
version to become visible before packaging the next dependent crate.

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
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps --locked
cargo test --workspace --locked
cargo test --workspace --all-features --locked
cargo test -p hdf5-reader --no-default-features --locked
cargo test -p netcdf-reader --no-default-features --locked
cargo test -p netcdf-writer --no-default-features --locked
# MSRV: compile-check the whole workspace, then unit-test the crates that have
# no dev-dependencies. The other crates are only compile-checked on MSRV
# because their test-only dependencies (proptest, tempfile, criterion, the C
# libnetcdf) pull in transitive crates that require a newer Rust edition.
rustup run 1.81.0 cargo check --workspace --all-features --locked
rustup run 1.81.0 cargo test -p hdf5-core -p netcdf-core --lib --locked
# Writer output conformance against the reference C libraries (needs Python
# with netCDF4 + h5py).
scripts/validate-writer-output.sh
cargo audit
# Before publishing workspace dependencies, verify every archive can be
# assembled. Full package verification follows the staged publish order below.
cargo package --workspace --offline --locked --no-verify
cargo package -p hdf5-core --offline --locked
cargo package -p netcdf-core --offline --locked
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
cargo package -p hdf5-reader
cargo publish -p hdf5-reader
# wait for crates.io index to update
cargo package -p hdf5-writer
cargo publish -p hdf5-writer
# wait for crates.io index to update
cargo package -p netcdf-reader
cargo publish -p netcdf-reader
# wait for crates.io index to update
cargo package -p netcdf-writer
cargo publish -p netcdf-writer
```

After all publishes succeed:

1. Update `CHANGELOG.md` if needed
2. Create the git tag, for example `v<version>`
3. Push the tag
4. Create the GitHub release notes from the tag
