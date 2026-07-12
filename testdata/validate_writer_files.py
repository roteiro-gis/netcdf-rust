#!/usr/bin/env python3
"""Validate netcdf-writer/hdf5-writer output against the reference C libraries.

Usage: validate_writer_files.py <dir>

<dir> must contain a manifest.json describing every file the Rust writers
produced, plus the files themselves. Each file is opened with netCDF4-python
(kind "netcdf") or h5py (kind "hdf5") -- both backed by the canonical C
libraries -- and every dimension, variable/dataset, attribute, filter, layout,
and dimension-scale attachment listed in the manifest is verified. Exits
nonzero with a readable diff on any mismatch.

Manifest schema (informal):
{
  "files": [
    {
      "path": "cdf1_single_record.nc",
      "kind": "netcdf",
      "data_model": "NETCDF3_CLASSIC",
      "dimensions": [{"name": "time", "size": 3, "unlimited": true}, ...],
      "groups": ["forecast"],                       # optional
      "attributes": {"title": "..."},               # optional, global
      "variables": [
        {
          "name": "temp",                           # may be "group/temp"
          "dtype": "f4",                            # numpy dtype string
          "shape": [2, 3],
          "dimensions": ["y", "x"],
          "values": [280.0, ...],                   # flattened, exactly one of
          "strings": ["a", "b"],                    #   values/strings/
          "char_strings": ["ab", "cd"],             #   char_strings/vlen
          "vlen": [[1, 2], [3]],
          "attributes": {"units": "K"}              # optional
        }
      ],
      "scales": [                                   # optional, NC4 only
        {"variable": "temp", "dim_index": 0, "scale": "time"}
      ]
    },
    {
      "path": "layouts.h5",
      "kind": "hdf5",
      "groups": ["outer/inner"],                    # optional
      "attributes": {...},                          # optional, root group
      "datasets": [
        {
          "name": "grid",
          "dtype": "<i4",                           # or "str" for vlen str
          "shape": [4, 5],
          "maxshape": [null, 5],                    # optional, null=unlimited
          "layout": "chunked",                      # optional
          "chunks": [2, 5],                         # optional
          "filters": {"deflate": 6, "shuffle": true, "fletcher32": true},
          "fillvalue": -1,                          # optional
          "values": [...],                          # as for variables
          "attributes": {...}
        }
      ]
    }
  ]
}
"""

import json
import sys
import traceback
from pathlib import Path

import numpy as np

failures = []


def fail(path, what, expected, actual):
    failures.append(f"{path}: {what}\n    expected: {expected!r}\n    actual:   {actual!r}")


def normalize_attr(value):
    """Normalize an attribute value read by netCDF4/h5py for comparison."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.dtype.kind in ("S", "U", "O"):
            return [normalize_attr(v) for v in value.tolist()]
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def attrs_equal(expected, actual):
    actual = normalize_attr(actual)
    if isinstance(expected, list):
        if not isinstance(actual, list):
            actual = [actual]
        if len(expected) != len(actual):
            return False
        return all(attrs_equal(e, a) for e, a in zip(expected, actual))
    if isinstance(expected, float) or isinstance(actual, float):
        if not isinstance(actual, (int, float)):
            return False
        e = np.float64(expected)
        a = np.float64(actual)
        if np.isnan(e) and np.isnan(a):
            return True
        # Attribute floats may be stored at f32 precision; compare at the
        # coarser precision of the two.
        return e == a or np.float32(e) == np.float32(a)
    return expected == actual


def check_attributes(path, owner, expected_attrs, read_attr):
    for name, expected in expected_attrs.items():
        try:
            actual = read_attr(name)
        except (KeyError, AttributeError):
            fail(path, f"{owner}: missing attribute {name!r}", expected, "<absent>")
            continue
        if not attrs_equal(expected, actual):
            fail(path, f"{owner}: attribute {name!r} mismatch", expected, normalize_attr(actual))


def check_values(path, owner, spec, data):
    """Compare dataset/variable payload against the manifest spec."""
    if "values" in spec:
        expected = np.asarray(spec["values"])
        actual = np.asarray(data)
        if actual.dtype.kind == "S":
            fail(path, f"{owner}: expected numeric data, got bytes", spec["values"], actual)
            return
        expected = expected.astype(actual.dtype).ravel()
        actual = actual.ravel()
        if expected.shape != actual.shape or not np.array_equal(expected, actual, equal_nan=actual.dtype.kind == "f"):
            fail(path, f"{owner}: data mismatch", expected.tolist(), actual.tolist())
    elif "strings" in spec:
        actual = [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in np.asarray(data, dtype=object).ravel()]
        if actual != spec["strings"]:
            fail(path, f"{owner}: string data mismatch", spec["strings"], actual)
    elif "char_strings" in spec:
        import netCDF4

        joined = netCDF4.chartostring(np.asarray(data))
        actual = [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in np.atleast_1d(joined)]
        expected = spec["char_strings"]
        if [a.rstrip("\x00") for a in actual] != [e.rstrip("\x00") for e in expected]:
            fail(path, f"{owner}: char data mismatch", expected, actual)
    elif "vlen" in spec:
        actual = [np.asarray(v).tolist() for v in np.asarray(data, dtype=object).ravel()]
        if actual != spec["vlen"]:
            fail(path, f"{owner}: vlen data mismatch", spec["vlen"], actual)


def resolve_nc_group(ds, name):
    """Split "a/b/var" into the owning netCDF4 group and the leaf name."""
    parts = name.split("/")
    group = ds
    for part in parts[:-1]:
        group = group.groups[part]
    return group, parts[-1]


def validate_netcdf(path, entry):
    import netCDF4

    ds = netCDF4.Dataset(path, "r")
    ds.set_auto_maskandscale(False)
    try:
        if "data_model" in entry and ds.data_model != entry["data_model"]:
            fail(path, "data_model mismatch", entry["data_model"], ds.data_model)

        for dim in entry.get("dimensions", []):
            actual = ds.dimensions.get(dim["name"])
            if actual is None:
                fail(path, f"missing dimension {dim['name']!r}", dim, "<absent>")
                continue
            if len(actual) != dim["size"]:
                fail(path, f"dimension {dim['name']!r} size", dim["size"], len(actual))
            if actual.isunlimited() != dim.get("unlimited", False):
                fail(path, f"dimension {dim['name']!r} unlimited", dim.get("unlimited", False), actual.isunlimited())

        for group in entry.get("groups", []):
            node = ds
            try:
                for part in group.split("/"):
                    node = node.groups[part]
            except KeyError:
                fail(path, f"missing group {group!r}", group, "<absent>")

        check_attributes(path, "global", entry.get("attributes", {}), lambda n: ds.getncattr(n))

        for spec in entry.get("variables", []):
            try:
                group, leaf = resolve_nc_group(ds, spec["name"])
                var = group.variables[leaf]
            except KeyError:
                fail(path, f"missing variable {spec['name']!r}", spec["name"], "<absent>")
                continue
            var.set_auto_maskandscale(False)
            if "dtype" in spec:
                expected_dtype = spec["dtype"]
                actual_dtype = "str" if var.dtype is str else np.dtype(var.dtype).str.lstrip("<>=|")
                if expected_dtype not in (actual_dtype, np.dtype(var.dtype).str if var.dtype is not str else "str"):
                    fail(path, f"variable {spec['name']!r} dtype", expected_dtype, actual_dtype)
            if "shape" in spec and list(var.shape) != spec["shape"]:
                fail(path, f"variable {spec['name']!r} shape", spec["shape"], list(var.shape))
            if "dimensions" in spec and list(var.dimensions) != spec["dimensions"]:
                fail(path, f"variable {spec['name']!r} dimensions", spec["dimensions"], list(var.dimensions))
            check_values(path, f"variable {spec['name']!r}", spec, var[...])
            check_attributes(
                path,
                f"variable {spec['name']!r}",
                spec.get("attributes", {}),
                lambda n, v=var: v.getncattr(n),
            )
    finally:
        ds.close()

    if entry.get("scales"):
        validate_scales(path, entry["scales"])


def validate_scales(path, checks):
    import h5py

    with h5py.File(path, "r") as f:
        for check in checks:
            var = f[check["variable"]]
            scale = f[check["scale"]]
            idx = check["dim_index"]
            if not h5py.h5ds.is_scale(scale.id):
                fail(path, f"{check['scale']!r} is not a dimension scale", True, False)
                continue
            attached = h5py.h5ds.is_attached(var.id, scale.id, idx)
            if not attached:
                fail(
                    path,
                    f"scale {check['scale']!r} not attached to {check['variable']!r} dim {idx}",
                    True,
                    False,
                )


def validate_hdf5(path, entry):
    import h5py

    with h5py.File(path, "r") as f:
        for group in entry.get("groups", []):
            if group not in f or not isinstance(f[group], h5py.Group):
                fail(path, f"missing group {group!r}", group, "<absent>")

        check_attributes(path, "root", entry.get("attributes", {}), lambda n: f.attrs[n])

        for group, attrs in entry.get("group_attributes", {}).items():
            if group not in f:
                fail(path, f"missing group {group!r} for attributes", group, "<absent>")
                continue
            check_attributes(path, f"group {group!r}", attrs, lambda n, g=f[group]: g.attrs[n])

        for spec in entry.get("datasets", []):
            name = spec["name"]
            if name not in f:
                fail(path, f"missing dataset {name!r}", name, "<absent>")
                continue
            ds = f[name]
            if "dtype" in spec:
                if spec["dtype"] == "str":
                    ok = h5py.check_string_dtype(ds.dtype) is not None
                    if not ok:
                        fail(path, f"dataset {name!r} dtype", "str", str(ds.dtype))
                elif str(ds.dtype) != str(np.dtype(spec["dtype"])):
                    fail(path, f"dataset {name!r} dtype", str(np.dtype(spec["dtype"])), str(ds.dtype))
            if "shape" in spec and list(ds.shape) != spec["shape"]:
                fail(path, f"dataset {name!r} shape", spec["shape"], list(ds.shape))
            if "maxshape" in spec:
                expected = [None if m is None else m for m in spec["maxshape"]]
                actual = [None if m is None else m for m in ds.maxshape]
                if expected != actual:
                    fail(path, f"dataset {name!r} maxshape", expected, actual)
            if "layout" in spec:
                plist = ds.id.get_create_plist()
                layout = {
                    h5py.h5d.COMPACT: "compact",
                    h5py.h5d.CONTIGUOUS: "contiguous",
                    h5py.h5d.CHUNKED: "chunked",
                }.get(plist.get_layout(), "unknown")
                if layout != spec["layout"]:
                    fail(path, f"dataset {name!r} layout", spec["layout"], layout)
            if "chunks" in spec and (ds.chunks is None or list(ds.chunks) != spec["chunks"]):
                fail(path, f"dataset {name!r} chunks", spec["chunks"], None if ds.chunks is None else list(ds.chunks))
            filters = spec.get("filters")
            if filters is not None:
                if "deflate" in filters:
                    expected = filters["deflate"]
                    actual = ds.compression_opts if ds.compression == "gzip" else None
                    if expected is None:
                        if ds.compression == "gzip":
                            fail(path, f"dataset {name!r} unexpectedly deflated", None, actual)
                    elif ds.compression != "gzip" or actual != expected:
                        fail(path, f"dataset {name!r} deflate level", expected, actual)
                if "shuffle" in filters and ds.shuffle != filters["shuffle"]:
                    fail(path, f"dataset {name!r} shuffle", filters["shuffle"], ds.shuffle)
                if "fletcher32" in filters and ds.fletcher32 != filters["fletcher32"]:
                    fail(path, f"dataset {name!r} fletcher32", filters["fletcher32"], ds.fletcher32)
            if "fillvalue" in spec and spec["fillvalue"] is not None:
                actual = ds.fillvalue
                if not attrs_equal(spec["fillvalue"], actual):
                    fail(path, f"dataset {name!r} fillvalue", spec["fillvalue"], normalize_attr(actual))
            check_values(path, f"dataset {name!r}", spec, ds[...] if ds.shape != () else ds[()])
            check_attributes(path, f"dataset {name!r}", spec.get("attributes", {}), lambda n, d=ds: d.attrs[n])


def main():
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    root = Path(sys.argv[1])
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        print(f"manifest not found: {manifest_path}", file=sys.stderr)
        return 2
    manifest = json.loads(manifest_path.read_text())

    checked = 0
    for entry in manifest["files"]:
        path = root / entry["path"]
        if not path.is_file():
            fail(entry["path"], "file missing", entry["path"], "<absent>")
            continue
        try:
            if entry["kind"] == "netcdf":
                validate_netcdf(path, entry)
            elif entry["kind"] == "hdf5":
                validate_hdf5(path, entry)
            else:
                fail(entry["path"], "unknown kind", "netcdf|hdf5", entry["kind"])
        except Exception as exc:  # noqa: BLE001 - a crash while opening IS the finding
            detail = f"{exc}\n{traceback.format_exc()}"
            fail(entry["path"], f"reference library raised {type(exc).__name__}", "clean open/read", detail)
        checked += 1

    if failures:
        print(f"FAILED: {len(failures)} mismatch(es) across {checked} file(s)\n", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print(f"OK: {checked} file(s) validated against reference libraries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
