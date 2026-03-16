#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
import re
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
    parser.add_argument(
        "--speedup",
        action="store_true",
        help="Include speedup vs the x1 baseline for threaded benchmark names like impl_x4",
    )
    return parser.parse_args()


THREAD_SUFFIX_RE = re.compile(r"^(?P<family>.+)_x(?P<threads>\d+)$")


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
        elements_count = throughput.get("Elements")
        yield {
            "group": benchmark["group_id"],
            "implementation": benchmark.get("function_id") or "",
            "case": benchmark.get("value_str") or "",
            "median_ns": median_ns,
            "bytes": int(bytes_count) if bytes_count is not None else None,
            "elements": int(elements_count) if elements_count is not None else None,
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


def format_elements_per_second(elements_per_second: float) -> str:
    units = ["ops/s", "Kops/s", "Mops/s", "Gops/s", "Tops/s"]
    value = elements_per_second
    unit = units[0]
    for candidate in units[1:]:
        if value < 1000.0:
            break
        value /= 1000.0
        unit = candidate
    return f"{value:.2f} {unit}"


def render_markdown(rows, include_speedup: bool):
    if include_speedup:
        print("| Workload | Implementation | Case | Median | Throughput | Speedup vs x1 |")
        print("| --- | --- | --- | ---: | ---: | ---: |")
    else:
        print("| Workload | Implementation | Case | Median | Throughput |")
        print("| --- | --- | --- | ---: | ---: |")
    for row in rows:
        base = (
            "| {group} | {implementation} | {case} | {median} | {throughput} |".format(
                group=row["group"],
                implementation=row["implementation"],
                case=row["case"],
                median=format_time(row["median_ns"]),
                throughput=row["throughput_display"],
            )
        )
        if include_speedup:
            print(f"{base} {row['speedup_display']} |")
        else:
            print(base)


def render_csv(rows, include_speedup: bool):
    columns = [
        "group",
        "implementation",
        "case",
        "median_ns",
        "throughput_value_per_second",
        "throughput_kind",
    ]
    if include_speedup:
        columns.append("speedup_vs_x1")
    print(",".join(columns))
    for row in rows:
        values = [
            row["group"],
            row["implementation"],
            row["case"],
            f"{row['median_ns']:.1f}",
            "" if row["throughput_value"] is None else f"{row['throughput_value']:.3f}",
            row["throughput_kind"] or "",
        ]
        if include_speedup:
            values.append("" if row["speedup"] is None else f"{row['speedup']:.3f}")
        print(",".join(values))


def annotate_speedups(rows):
    baselines = {}
    for row in rows:
        match = THREAD_SUFFIX_RE.match(row["implementation"])
        if not match:
            continue
        family = match.group("family")
        threads = int(match.group("threads"))
        if threads == 1:
            baselines[(row["group"], row["case"], family)] = row

    for row in rows:
        row["speedup"] = None
        row["speedup_display"] = "-"
        match = THREAD_SUFFIX_RE.match(row["implementation"])
        if not match:
            continue
        family = match.group("family")
        baseline = baselines.get((row["group"], row["case"], family))
        if baseline is None:
            continue

        if (
            row["throughput_value"] is not None
            and baseline["throughput_value"] is not None
            and row["throughput_kind"] == baseline["throughput_kind"]
        ):
            speedup = row["throughput_value"] / baseline["throughput_value"]
        else:
            speedup = baseline["median_ns"] / row["median_ns"]

        row["speedup"] = speedup
        row["speedup_display"] = f"{speedup:.2f}x"


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

        throughput_value = None
        throughput_kind = None
        throughput_display = "-"
        if result["bytes"] is not None and result["median_ns"] > 0.0:
            throughput_value = result["bytes"] / (result["median_ns"] / 1_000_000_000.0)
            throughput_kind = "bytes"
            throughput_display = format_bytes_per_second(throughput_value)
        elif result["elements"] is not None and result["median_ns"] > 0.0:
            throughput_value = result["elements"] / (result["median_ns"] / 1_000_000_000.0)
            throughput_kind = "elements"
            throughput_display = format_elements_per_second(throughput_value)

        result["throughput_value"] = throughput_value
        result["throughput_kind"] = throughput_kind
        result["throughput_display"] = throughput_display
        rows.append(result)

    rows.sort(key=lambda row: (row["group"], row["case"], row["implementation"]))
    if args.speedup:
        annotate_speedups(rows)

    if args.format == "markdown":
        render_markdown(rows, args.speedup)
    else:
        render_csv(rows, args.speedup)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
