#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize Criterion benchmark results as markdown or CSV."
    )
    parser.add_argument(
        "--criterion-root",
        default="netcdf-reader/target/criterion",
        help="Path to the Criterion output directory",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "csv"),
        default="markdown",
        help="Output format",
    )
    parser.add_argument(
        "--group",
        action="append",
        default=[],
        help="Benchmark group(s) to include; repeat to include multiple",
    )
    return parser.parse_args()


def iter_results(root: Path):
    for benchmark_path in sorted(root.glob("**/new/benchmark.json")):
        estimates_path = benchmark_path.with_name("estimates.json")
        if not estimates_path.exists():
            continue

        benchmark = json.loads(benchmark_path.read_text())
        estimates = json.loads(estimates_path.read_text())
        median_ns = float(estimates["median"]["point_estimate"])
        throughput = benchmark.get("throughput") or {}
        bytes_count = throughput.get("Bytes")
        yield {
            "group": benchmark["group_id"],
            "implementation": benchmark.get("function_id") or "",
            "case": benchmark.get("value_str") or "",
            "median_ns": median_ns,
            "bytes": int(bytes_count) if bytes_count is not None else None,
        }


def format_time(ns: float) -> str:
    if ns >= 1_000_000_000:
        return f"{ns / 1_000_000_000:.3f} s"
    if ns >= 1_000_000:
        return f"{ns / 1_000_000:.3f} ms"
    if ns >= 1_000:
        return f"{ns / 1_000:.3f} us"
    return f"{ns:.1f} ns"


def format_bytes_per_second(bytes_per_second: float) -> str:
    units = ["B/s", "KiB/s", "MiB/s", "GiB/s", "TiB/s"]
    value = bytes_per_second
    unit = units[0]
    for candidate in units[1:]:
        if value < 1024.0:
            break
        value /= 1024.0
        unit = candidate
    return f"{value:.2f} {unit}"


def render_markdown(rows):
    print("| Workload | Implementation | Case | Median | Throughput |")
    print("| --- | --- | --- | ---: | ---: |")
    for row in rows:
        print(
            "| {group} | {implementation} | {case} | {median} | {throughput} |".format(
                group=row["group"],
                implementation=row["implementation"],
                case=row["case"],
                median=format_time(row["median_ns"]),
                throughput=row["throughput_display"],
            )
        )


def render_csv(rows):
    print("group,implementation,case,median_ns,throughput_bytes_per_second")
    for row in rows:
        print(
            "{group},{implementation},{case},{median_ns:.1f},{throughput}".format(
                group=row["group"],
                implementation=row["implementation"],
                case=row["case"],
                median_ns=row["median_ns"],
                throughput="" if row["throughput_bps"] is None else f"{row['throughput_bps']:.3f}",
            )
        )


def main() -> int:
    args = parse_args()
    root = Path(args.criterion_root)
    if not root.exists():
        print(f"criterion root not found: {root}", file=sys.stderr)
        return 1

    rows = []
    for result in iter_results(root):
        if args.group and result["group"] not in args.group:
            continue

        throughput_bps = None
        throughput_display = "-"
        if result["bytes"] is not None and result["median_ns"] > 0.0:
            throughput_bps = result["bytes"] / (result["median_ns"] / 1_000_000_000.0)
            throughput_display = format_bytes_per_second(throughput_bps)

        result["throughput_bps"] = throughput_bps
        result["throughput_display"] = throughput_display
        rows.append(result)

    rows.sort(key=lambda row: (row["group"], row["case"], row["implementation"]))

    if args.format == "markdown":
        render_markdown(rows)
    else:
        render_csv(rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
