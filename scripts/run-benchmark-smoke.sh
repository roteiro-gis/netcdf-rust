#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON:-python3}"

base_ref="${BENCH_BASE_REF:-${GITHUB_BASE_REF:-}}"
if [[ -z "$base_ref" ]]; then
  echo "BENCH_BASE_REF or GITHUB_BASE_REF must be set" >&2
  exit 1
fi

if [[ "$base_ref" != origin/* ]]; then
  base_ref="origin/$base_ref"
fi

export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$repo_root/target/bench-regression}"
export BENCH_THREAD_LIST="${BENCH_THREAD_LIST:-1,4}"
export BENCH_HOT_OPS_PER_THREAD="${BENCH_HOT_OPS_PER_THREAD:-128}"
export BENCH_SAMPLE_SIZE="${BENCH_SAMPLE_SIZE:-15}"
export BENCH_MEASUREMENT_TIME="${BENCH_MEASUREMENT_TIME:-0.2}"
export BENCH_WARMUP_TIME="${BENCH_WARMUP_TIME:-0.2}"
export BENCH_REGRESSION_THRESHOLD_PERCENT="${BENCH_REGRESSION_THRESHOLD_PERCENT:-10}"
export BENCH_SMOKE_FILTER="${BENCH_SMOKE_FILTER:-open_only/netcdf_rust/(cdf1_simple|large_nc4_compressed)|metadata_reuse_handle/netcdf_rust/(cdf1_simple|nc4_compressed)|read_full_reuse_handle/netcdf_rust/(nc4_compressed|large_nc4_compressed)|slice_reuse_handle_hdf5_backend/netcdf_rust/(nc4_compressed|large_nc4_compressed)|parallel_open_and_read/netcdf_rust_x(1|4)/large_nc4_compressed|parallel_metadata_batch/netcdf_rust_x(1|4)/nc4_basic|parallel_slice_batch/netcdf_rust_x(1|4)/nc4_compressed|read_full_internal_parallel/netcdf_rust_x(1|4)/(large_nc4_compressed|nc4_compressed)|read_full_internal_parallel_nocache/netcdf_rust_x(1|4)/large_nc4_compressed}"

if ! command -v critcmp >/dev/null 2>&1; then
  env -u RUSTFLAGS cargo install --locked critcmp
fi

"$repo_root/scripts/generate-fixtures.sh"

git -C "$repo_root" fetch --no-tags origin "${base_ref#origin/}"

base_worktree="$(mktemp -d "${TMPDIR:-/tmp}/netcdf-rust-bench-base.XXXXXX")"
cleanup() {
  git -C "$repo_root" worktree remove --force "$base_worktree" >/dev/null 2>&1 || true
  rm -rf "$base_worktree"
}
trap cleanup EXIT

git -C "$repo_root" worktree add --detach "$base_worktree" "$base_ref"

run_compare_bench() {
  local repo_dir="$1"
  local baseline_name="$2"

  (
    cd "$repo_dir"
    cargo bench -p netcdf-reader --bench compare_georust \
      "$BENCH_SMOKE_FILTER" \
      -- --noplot \
      --sample-size "$BENCH_SAMPLE_SIZE" \
      --measurement-time "$BENCH_MEASUREMENT_TIME" \
      --warm-up-time "$BENCH_WARMUP_TIME" \
      --save-baseline "$baseline_name"
  )
}

REPO_ROOT="$base_worktree" INSTALL_FIXTURE_DEPS=0 \
  "$repo_root/scripts/generate-fixtures.sh"
run_compare_bench "$base_worktree" base
run_compare_bench "$repo_root" pr

echo "Benchmark deltas >=${BENCH_REGRESSION_THRESHOLD_PERCENT}%:"
critcmp base pr --target-dir "$CARGO_TARGET_DIR" --threshold "$BENCH_REGRESSION_THRESHOLD_PERCENT" || true

"$python_bin" - <<'PY'
import json
import os
import sys
from pathlib import Path

threshold_percent = float(os.environ["BENCH_REGRESSION_THRESHOLD_PERCENT"])
threshold = 1.0 + threshold_percent / 100.0
target = Path(os.environ["CARGO_TARGET_DIR"]) / "criterion"
base_files = sorted(target.glob("**/base/estimates.json"))
pr_files = sorted(target.glob("**/pr/estimates.json"))
if not base_files:
    print(f"No base benchmark estimates found under {target}", file=sys.stderr)
    sys.exit(1)
if not pr_files:
    print(f"No PR benchmark estimates found under {target}", file=sys.stderr)
    sys.exit(1)

pr_index = {
    pr_file.parent.parent.relative_to(target).as_posix(): pr_file for pr_file in pr_files
}

regressions = []
missing_pr = []

for base_file in base_files:
    bench_dir = base_file.parent.parent
    bench_name = bench_dir.relative_to(target).as_posix()
    pr_file = pr_index.get(bench_name)
    if pr_file is None:
        missing_pr.append(bench_name)
        continue

    with base_file.open() as fh:
        base_estimates = json.load(fh)
    with pr_file.open() as fh:
        pr_estimates = json.load(fh)

    # Median is more robust than mean for short smoke-style CI runs on noisy
    # shared runners.
    base_mean = base_estimates["median"]["point_estimate"]
    pr_mean = pr_estimates["median"]["point_estimate"]

    if base_mean <= 0 or pr_mean <= 0:
        continue

    ratio = pr_mean / base_mean
    if ratio > threshold:
        regressions.append((ratio, bench_name, base_mean, pr_mean))

if missing_pr:
    print("Missing PR benchmark estimates for:", file=sys.stderr)
    for name in missing_pr:
        print(f"  {name}", file=sys.stderr)
    sys.exit(1)

def fmt_ns(value: float) -> str:
    for unit, scale in (("s", 1e9), ("ms", 1e6), ("us", 1e3)):
        if value >= scale:
            return f"{value / scale:.1f}{unit}"
    return f"{value:.1f}ns"

if regressions:
    print(
        f"Benchmark regressions >{threshold_percent:.0f}% "
        "(PR median / base median):",
        file=sys.stderr,
    )
    for ratio, name, base_mean, pr_mean in sorted(regressions, reverse=True):
        print(
            f"  {name:70} {ratio:5.2f}x  "
            f"base={fmt_ns(base_mean):>8}  pr={fmt_ns(pr_mean):>8}",
            file=sys.stderr,
        )
    sys.exit(1)

print(f"No benchmark regressions >{threshold_percent:.0f}% detected.")
PY
