# Releasing

This repository publishes two crates that must be released in order:

1. `hdf5-reader`
2. `netcdf-reader`

`netcdf-reader` depends on the published `hdf5-reader` crate, so a dry-run or
publish verification for `netcdf-reader` will fail until `hdf5-reader` has been
published and the crates.io index has updated.

## Pre-release checks

```sh
cargo fmt --all -- --check
cargo clippy --all -- -D warnings
cargo test --workspace
cargo test -p hdf5-reader --no-default-features
cargo test -p netcdf-reader --no-default-features
cargo package -p hdf5-reader
```

`netcdf-reader` packaging still depends on the new `hdf5-reader` version being
visible in the crates.io index, so run its packaging check only after
`hdf5-reader` has been published and the index has updated:

```sh
cargo package -p netcdf-reader --no-verify
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
cargo publish -p hdf5-reader
# wait for crates.io index to update
cargo publish -p netcdf-reader
```

After both publishes succeed:

1. Update `CHANGELOG.md` if needed
2. Create the git tag, for example `v<version>`
3. Push the tag
4. Create the GitHub release notes from the tag
