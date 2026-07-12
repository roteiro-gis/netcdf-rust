#!/usr/bin/env bash
# Validate netcdf-writer/hdf5-writer output against the reference C libraries
# (netcdf-c via netCDF4-python, libhdf5 via h5py).
#
# Environment:
#   PYTHON / NETCDF_RUST_VALIDATOR_PYTHON  python interpreter (default python3)
#   INSTALL_VALIDATOR_DEPS=1               pip-install netCDF4/h5py/numpy first
set -euo pipefail

repo_root="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
python_bin="${NETCDF_RUST_VALIDATOR_PYTHON:-${PYTHON:-python3}}"

if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "python executable not found: $python_bin" >&2
  exit 1
fi

# cargo test runs with each crate as cwd; make the interpreter path absolute.
python_bin="$(command -v "$python_bin")"
case "$python_bin" in
  /*) ;;
  *) python_bin="$(cd "$(dirname "$python_bin")" && pwd)/$(basename "$python_bin")" ;;
esac

if [[ "${INSTALL_VALIDATOR_DEPS:-0}" == "1" ]]; then
  "$python_bin" -m pip install --disable-pip-version-check h5py netCDF4 numpy
fi

if ! "$python_bin" -c "import netCDF4, h5py, numpy" >/dev/null 2>&1; then
  echo "validator python ($python_bin) is missing netCDF4/h5py/numpy;" \
    "rerun with INSTALL_VALIDATOR_DEPS=1 or point PYTHON at an environment that has them" >&2
  exit 1
fi

cd "$repo_root"
NETCDF_RUST_EXTERNAL_VALIDATION=1 \
NETCDF_RUST_VALIDATOR_PYTHON="$python_bin" \
  cargo test -p hdf5-writer -p netcdf-writer --test external_validation -- --ignored --nocapture
