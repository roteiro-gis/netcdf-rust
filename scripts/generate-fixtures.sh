#!/usr/bin/env bash
set -euo pipefail

repo_root="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
python_bin="${PYTHON:-python3}"

if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "python executable not found: $python_bin" >&2
  exit 1
fi

if [[ ! -f "$repo_root/testdata/generate_fixtures.py" ]]; then
  echo "fixture generator not found under $repo_root/testdata" >&2
  exit 1
fi

if [[ "${INSTALL_FIXTURE_DEPS:-1}" == "1" ]]; then
  "$python_bin" -m pip install --disable-pip-version-check 'h5py<3.16' netCDF4 numpy
fi

"$python_bin" "$repo_root/testdata/generate_fixtures.py"
