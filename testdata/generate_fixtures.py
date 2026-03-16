#!/usr/bin/env python3
"""
Generate HDF5 and NetCDF test fixture files for the netcdf-rust crate.

Requirements:
    pip install h5py netCDF4 numpy

Creates fixtures in hdf5/, netcdf3/, and netcdf4/ subdirectories
relative to this script's location.
"""

import os
import subprocess
import sys
import numpy as np


def generate_hdf5_fixtures(base_dir):
    """Generate all HDF5 test fixtures."""
    import h5py

    hdf5_dir = os.path.join(base_dir, "hdf5")
    os.makedirs(hdf5_dir, exist_ok=True)

    # ---- 1. simple_contiguous.h5 ----
    # Tests: basic contiguous storage, simple 2D f64 dataset, dataset and global attributes.
    path = os.path.join(hdf5_dir, "simple_contiguous.h5")
    print(f"  Generating {path}")
    with h5py.File(path, "w") as f:
        data = np.arange(20, dtype=np.float64).reshape(4, 5)
        ds = f.create_dataset("data", data=data)
        ds.attrs["description"] = "test data"
        f.attrs["creator"] = "test"

    # ---- 2. simple_chunked_deflate.h5 ----
    # Tests: chunked storage with gzip (deflate) compression, f32 type.
    path = os.path.join(hdf5_dir, "simple_chunked_deflate.h5")
    print(f"  Generating {path}")
    with h5py.File(path, "w") as f:
        data = np.zeros((10, 20), dtype=np.float32)
        for r in range(10):
            for c in range(20):
                data[r, c] = r * 20 + c
        f.create_dataset(
            "temperature", data=data, chunks=(5, 10), compression="gzip", compression_opts=4
        )

    # ---- 3. chunked_shuffle_deflate.h5 ----
    # Tests: chunked storage with shuffle filter + gzip compression pipeline.
    path = os.path.join(hdf5_dir, "chunked_shuffle_deflate.h5")
    print(f"  Generating {path}")
    with h5py.File(path, "w") as f:
        np.random.seed(42)
        data = np.random.randn(100, 100).astype(np.float64)
        f.create_dataset(
            "values", data=data, chunks=(10, 10), shuffle=True, compression="gzip", compression_opts=6
        )

    # ---- 4. nested_groups.h5 ----
    # Tests: HDF5 group hierarchy, multiple groups at different nesting levels.
    path = os.path.join(hdf5_dir, "nested_groups.h5")
    print(f"  Generating {path}")
    with h5py.File(path, "w") as f:
        g1 = f.create_group("group1")
        g1.create_dataset("data", data=np.array([1, 2, 3], dtype=np.int32))
        sg = g1.create_group("subgroup")
        sg.create_dataset("data", data=np.array([4, 5, 6], dtype=np.int32))
        g2 = f.create_group("group2")
        g2.create_dataset("data", data=np.array([7, 8, 9], dtype=np.int32))

    # ---- 5. string_attrs.h5 ----
    # Tests: various attribute types — string, f64 scalar, i32 array.
    path = os.path.join(hdf5_dir, "string_attrs.h5")
    print(f"  Generating {path}")
    with h5py.File(path, "w") as f:
        ds = f.create_dataset("data", data=np.array([10, 20], dtype=np.int32))
        ds.attrs["name"] = "test_dataset"
        ds.attrs["units"] = "meters"
        ds.attrs["scale_factor"] = np.float64(1.5)
        ds.attrs["valid_range"] = np.array([0, 100], dtype=np.int32)

    # ---- 6. compound_type.h5 ----
    # Tests: compound (struct) datatypes with mixed numeric and fixed-length string fields.
    path = os.path.join(hdf5_dir, "compound_type.h5")
    print(f"  Generating {path}")
    with h5py.File(path, "w") as f:
        dt = np.dtype([("x", np.float64), ("y", np.float64), ("label", "S10")])
        data = np.array(
            [(1.0, 2.0, b"alpha"), (3.0, 4.0, b"beta"), (5.0, 6.0, b"gamma")],
            dtype=dt,
        )
        f.create_dataset("records", data=data)

    # ---- 7. scalar_dataset.h5 ----
    # Tests: scalar (0-dimensional) datasets.
    path = os.path.join(hdf5_dir, "scalar_dataset.h5")
    print(f"  Generating {path}")
    with h5py.File(path, "w") as f:
        f.create_dataset("value", data=np.float64(42.0))

    # ---- 8. large_chunked.h5 ----
    # Tests: larger chunked dataset with gzip, multiple chunks in both dimensions.
    path = os.path.join(hdf5_dir, "large_chunked.h5")
    print(f"  Generating {path}")
    with h5py.File(path, "w") as f:
        data = np.arange(60000, dtype=np.int32).reshape(200, 300)
        f.create_dataset("big", data=data, chunks=(50, 50), compression="gzip")

    # ---- 9. multi_dim_4d.h5 ----
    # Tests: 4-dimensional dataset layout and indexing.
    path = os.path.join(hdf5_dir, "multi_dim_4d.h5")
    print(f"  Generating {path}")
    with h5py.File(path, "w") as f:
        data = np.arange(120, dtype=np.float32).reshape(2, 3, 4, 5)
        f.create_dataset("data4d", data=data)

    # ---- 10. old_format_v1.h5 ----
    # Tests: HDF5 v1 object headers and v1 B-tree group structures (earliest library version).
    path = os.path.join(hdf5_dir, "old_format_v1.h5")
    print(f"  Generating {path}")
    with h5py.File(path, "w", libver="earliest", track_order=False) as f:
        data = np.arange(20, dtype=np.float64).reshape(4, 5)
        f.create_dataset("data", data=data)

    # ---- 11. fill_value.h5 ----
    # Tests: explicit fill values and sparse writes where unwritten elements keep the fill value.
    path = os.path.join(hdf5_dir, "fill_value.h5")
    print(f"  Generating {path}")
    with h5py.File(path, "w") as f:
        ds = f.create_dataset("sparse", shape=(10,), dtype=np.int32, fillvalue=999)
        ds[0] = 0
        ds[3] = 3
        ds[7] = 7

        # Dataset with no explicit fill value (HDF5 library default for i32 is 0)
        f.create_dataset("nofill", shape=(10,), dtype=np.int32)

    # ---- 12. fletcher32.h5 ----
    # Tests: Fletcher32 checksum filter on dataset chunks.
    path = os.path.join(hdf5_dir, "fletcher32.h5")
    print(f"  Generating {path}")
    with h5py.File(path, "w") as f:
        data = np.eye(4, dtype=np.float32)
        f.create_dataset("checked", data=data, fletcher32=True)

    # ---- 13. dense_groups.h5 ----
    # Tests: dense link storage (fractal heap + B-tree v2), triggered by 20+ children.
    path = os.path.join(hdf5_dir, "dense_groups.h5")
    print(f"  Generating {path}")
    with h5py.File(path, "w") as f:
        for i in range(25):
            f.create_dataset(f"ds_{i:03d}", data=np.array([i], dtype=np.int32))

    # ---- 14. vlen_strings.h5 ----
    # Tests: variable-length string attributes and datasets.
    path = os.path.join(hdf5_dir, "vlen_strings.h5")
    print(f"  Generating {path}")
    with h5py.File(path, "w") as f:
        dt = h5py.string_dtype()
        ds = f.create_dataset("labels", data=np.array(["alpha", "beta", "gamma"], dtype=object), dtype=dt)
        ds.attrs.create("title", "vlen test", dtype=dt)

    # ---- 15. v4_layout.h5 ----
    # Tests: v4 data layout (libver='latest' forces new-style object headers and chunk indexing).
    path = os.path.join(hdf5_dir, "v4_layout.h5")
    print(f"  Generating {path}")
    with h5py.File(path, "w", libver="latest") as f:
        data = np.arange(100, dtype=np.float32).reshape(10, 10)
        f.create_dataset("data", data=data, chunks=(5, 5), compression="gzip")

    # ---- 16. single_chunk.h5 ----
    # Tests: single-chunk layout (entire dataset is one chunk).
    path = os.path.join(hdf5_dir, "single_chunk.h5")
    print(f"  Generating {path}")
    with h5py.File(path, "w") as f:
        data = np.arange(20, dtype=np.float64).reshape(4, 5)
        f.create_dataset("data", data=data, chunks=(4, 5))

    # ---- 17. committed_dtype.h5 ----
    # Tests: committed (shared) datatype — a named datatype that is shared across datasets.
    path = os.path.join(hdf5_dir, "committed_dtype.h5")
    print(f"  Generating {path}")
    with h5py.File(path, "w") as f:
        dt = np.dtype(np.float64)
        f["my_type"] = dt  # commit the type
        # Reference the committed type
        ds = f.create_dataset("data", data=np.array([1.0, 2.0, 3.0]), dtype=f["my_type"].dtype)

    # ---- 18. fixed_array_chunked.h5 ----
    # Tests: Fixed Array chunk indexing (libver='latest', fixed-size chunked).
    path = os.path.join(hdf5_dir, "fixed_array_chunked.h5")
    print(f"  Generating {path}")
    with h5py.File(path, "w", libver="latest") as f:
        data = np.arange(60, dtype=np.float64).reshape(6, 10)
        f.create_dataset("data", data=data, chunks=(3, 5), compression="gzip")

    # ---- 19. extensible_array_chunked.h5 ----
    # Tests: Extensible Array chunk indexing (libver='latest', one unlimited dim).
    path = os.path.join(hdf5_dir, "extensible_array_chunked.h5")
    print(f"  Generating {path}")
    with h5py.File(path, "w", libver="latest") as f:
        ds = f.create_dataset(
            "data", shape=(5, 8), maxshape=(None, 8), dtype=np.float64,
            chunks=(2, 4), compression="gzip"
        )
        data = np.arange(40, dtype=np.float64).reshape(5, 8)
        ds[:] = data


def generate_netcdf3_fixtures(base_dir):
    """Generate all NetCDF-3 (classic, 64-bit offset, 64-bit data) test fixtures."""
    import netCDF4

    nc3_dir = os.path.join(base_dir, "netcdf3")
    os.makedirs(nc3_dir, exist_ok=True)

    # ---- 1. cdf1_simple.nc ----
    # Tests: CDF-1 classic format, basic dimensions/variables/attributes.
    path = os.path.join(nc3_dir, "cdf1_simple.nc")
    print(f"  Generating {path}")
    ds = netCDF4.Dataset(path, "w", format="NETCDF3_CLASSIC")
    ds.title = "CDF-1 test"
    ds.createDimension("x", 5)
    ds.createDimension("y", 10)
    temp = ds.createVariable("temp", "f4", ("x", "y"))
    temp[:] = np.arange(50, dtype=np.float32).reshape(5, 10)
    ds.close()

    # ---- 2. cdf2_large_offset.nc ----
    # Tests: CDF-2 (64-bit offset) format, same structure as CDF-1 to compare header differences.
    path = os.path.join(nc3_dir, "cdf2_large_offset.nc")
    print(f"  Generating {path}")
    ds = netCDF4.Dataset(path, "w", format="NETCDF3_64BIT_OFFSET")
    ds.title = "CDF-2 test"
    ds.createDimension("x", 5)
    ds.createDimension("y", 10)
    temp = ds.createVariable("temp", "f4", ("x", "y"))
    temp[:] = np.arange(50, dtype=np.float32).reshape(5, 10)
    ds.close()

    # ---- 3. cdf5_new_types.nc ----
    # Tests: CDF-5 format with unsigned and 64-bit integer types not available in CDF-1/2.
    path = os.path.join(nc3_dir, "cdf5_new_types.nc")
    print(f"  Generating {path}")
    ds = netCDF4.Dataset(path, "w", format="NETCDF3_64BIT_DATA")
    ds.createDimension("n", 4)
    vals = [1, 2, 3, 4]

    ubyte_var = ds.createVariable("ubyte_var", "u1", ("n",))
    ubyte_var[:] = np.array(vals, dtype=np.uint8)

    ushort_var = ds.createVariable("ushort_var", "u2", ("n",))
    ushort_var[:] = np.array(vals, dtype=np.uint16)

    uint_var = ds.createVariable("uint_var", "u4", ("n",))
    uint_var[:] = np.array(vals, dtype=np.uint32)

    int64_var = ds.createVariable("int64_var", "i8", ("n",))
    int64_var[:] = np.array(vals, dtype=np.int64)

    uint64_var = ds.createVariable("uint64_var", "u8", ("n",))
    uint64_var[:] = np.array(vals, dtype=np.uint64)

    ds.close()

    # ---- 4. record_vars.nc ----
    # Tests: unlimited (record) dimension with multiple time-step writes.
    path = os.path.join(nc3_dir, "record_vars.nc")
    print(f"  Generating {path}")
    ds = netCDF4.Dataset(path, "w", format="NETCDF3_CLASSIC")
    ds.createDimension("time", None)  # unlimited
    ds.createDimension("x", 5)
    series = ds.createVariable("series", "f8", ("time", "x"))
    series[0, :] = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    series[1, :] = np.array([6.0, 7.0, 8.0, 9.0, 10.0])
    series[2, :] = np.array([11.0, 12.0, 13.0, 14.0, 15.0])
    ds.close()

    # ---- 5. multi_var.nc ----
    # Tests: multiple variables of different types sharing the same dimensions.
    path = os.path.join(nc3_dir, "multi_var.nc")
    print(f"  Generating {path}")
    ds = netCDF4.Dataset(path, "w", format="NETCDF3_CLASSIC")
    ds.createDimension("x", 3)
    ds.createDimension("y", 4)

    pressure = ds.createVariable("pressure", "f8", ("x", "y"))
    pressure[:] = np.arange(12, dtype=np.float64).reshape(3, 4)

    humidity = ds.createVariable("humidity", "f4", ("x", "y"))
    humidity[:] = np.arange(12, dtype=np.float32).reshape(3, 4)

    flag = ds.createVariable("flag", "i4", ("x", "y"))
    flag[:] = np.arange(12, dtype=np.int32).reshape(3, 4)

    ds.close()

    # ---- 6. global_attrs.nc ----
    # Tests: global attributes of various types (string, i32, f64).
    path = os.path.join(nc3_dir, "global_attrs.nc")
    print(f"  Generating {path}")
    ds = netCDF4.Dataset(path, "w", format="NETCDF3_CLASSIC")
    ds.setncattr("title", "Global Attributes Test")
    ds.setncattr("version", np.int32(2))
    ds.setncattr("scale", np.float64(1.5))
    ds.setncattr("history", "created for testing")
    ds.createDimension("x", 1)
    dummy = ds.createVariable("dummy", "i4", ("x",))
    dummy[:] = np.array([0], dtype=np.int32)
    ds.close()


def generate_netcdf4_fixtures(base_dir):
    """Generate all NetCDF-4 (HDF5-backed) test fixtures."""
    import netCDF4

    nc4_dir = os.path.join(base_dir, "netcdf4")
    os.makedirs(nc4_dir, exist_ok=True)

    # ---- 1. nc4_basic.nc ----
    # Tests: basic NetCDF-4 format, dimensions, variables, global attributes.
    path = os.path.join(nc4_dir, "nc4_basic.nc")
    print(f"  Generating {path}")
    ds = netCDF4.Dataset(path, "w", format="NETCDF4")
    ds.createDimension("x", 5)
    ds.createDimension("y", 10)
    data = ds.createVariable("data", "f8", ("x", "y"))
    data[:] = np.arange(50, dtype=np.float64).reshape(5, 10)
    ds.close()

    # ---- 2. nc4_groups.nc ----
    # Tests: NetCDF-4 group hierarchy with variables and dimensions at different levels.
    path = os.path.join(nc4_dir, "nc4_groups.nc")
    print(f"  Generating {path}")
    ds = netCDF4.Dataset(path, "w", format="NETCDF4")
    ds.createDimension("x", 5)
    root_var = ds.createVariable("root_var", "f8", ("x",))
    root_var[:] = np.arange(5, dtype=np.float64)

    obs = ds.createGroup("obs")
    obs.createDimension("time", 3)
    temp = obs.createVariable("temperature", "f4", ("time",))
    temp[:] = np.array([20.5, 21.0, 19.8], dtype=np.float32)

    surface = obs.createGroup("surface")
    # surface inherits 'time' dimension from parent group 'obs'
    pressure = surface.createVariable("pressure", "f8", ("time",))
    pressure[:] = np.array([1013.25, 1012.0, 1014.5], dtype=np.float64)
    ds.close()

    # ---- 3. nc4_compressed.nc ----
    # Tests: NetCDF-4 chunked + zlib compressed variable.
    path = os.path.join(nc4_dir, "nc4_compressed.nc")
    print(f"  Generating {path}")
    ds = netCDF4.Dataset(path, "w", format="NETCDF4")
    ds.createDimension("x", 100)
    ds.createDimension("y", 100)
    compressed = ds.createVariable(
        "compressed", "f4", ("x", "y"), chunksizes=(25, 25), zlib=True, complevel=4
    )
    np.random.seed(123)
    compressed[:] = np.random.randn(100, 100).astype(np.float32)
    ds.close()

    # ---- 4. nc4_string_var.nc ----
    # Tests: NetCDF-4 variable-length string variable type.
    path = os.path.join(nc4_dir, "nc4_string_var.nc")
    print(f"  Generating {path}")
    ds = netCDF4.Dataset(path, "w", format="NETCDF4")
    ds.createDimension("n", 4)
    names = ds.createVariable("names", str, ("n",))
    names[0] = "alpha"
    names[1] = "beta"
    names[2] = "gamma"
    names[3] = "delta"
    ds.close()

    # ---- 5. nc4_unlimited.nc ----
    # Tests: NetCDF-4 unlimited dimension with multiple record writes.
    path = os.path.join(nc4_dir, "nc4_unlimited.nc")
    print(f"  Generating {path}")
    ds = netCDF4.Dataset(path, "w", format="NETCDF4")
    ds.createDimension("time", None)  # unlimited
    ds.createDimension("x", 4)
    data = ds.createVariable("data", "f8", ("time", "x"))
    for t in range(5):
        data[t, :] = np.array([t * 4 + i for i in range(4)], dtype=np.float64)
    ds.close()

    # ---- 6. nc4_classic_model.nc ----
    # Tests: NETCDF4_CLASSIC format (HDF5 backend but classic data model only).
    path = os.path.join(nc4_dir, "nc4_classic_model.nc")
    print(f"  Generating {path}")
    ds = netCDF4.Dataset(path, "w", format="NETCDF4_CLASSIC")
    ds.createDimension("x", 5)
    ds.createDimension("y", 10)
    data = ds.createVariable("data", "f8", ("x", "y"))
    data[:] = np.arange(50, dtype=np.float64).reshape(5, 10)
    ds.close()

    # ---- 7. same_size_dims.nc ----
    # Tests: Two dimensions with identical sizes — verifies DIMENSION_LIST
    # resolves by reference, not by size matching.
    path = os.path.join(nc4_dir, "same_size_dims.nc")
    print(f"  Generating {path}")
    ds = netCDF4.Dataset(path, "w", format="NETCDF4")
    ds.createDimension("lat", 10)
    ds.createDimension("lon", 10)
    temp = ds.createVariable("temperature", "f4", ("lat", "lon"))
    temp[:] = np.arange(100, dtype=np.float32).reshape(10, 10)
    ds.close()


SECTION_GENERATORS = {
    "hdf5": ("HDF5 fixtures", "h5py", generate_hdf5_fixtures),
    "netcdf3": ("NetCDF-3 fixtures", "netCDF4", generate_netcdf3_fixtures),
    "netcdf4": ("NetCDF-4 fixtures", "netCDF4", generate_netcdf4_fixtures),
}


def run_section(section, base_dir):
    """Run a single fixture family inside the current process."""
    if section not in SECTION_GENERATORS:
        print(f"unknown fixture section: {section}", file=sys.stderr)
        return 2

    label, dependency, generator = SECTION_GENERATORS[section]
    print(f"[{label}]", flush=True)
    try:
        generator(base_dir)
        print("  Done.\n", flush=True)
        return 0
    except ImportError as e:
        print(f"  SKIPPED: {dependency} not available ({e})\n", flush=True)
        return 0


def run_all_sections(base_dir):
    """Run fixture families in isolated subprocesses.

    Isolating sections avoids HDF5 library conflicts between h5py and netCDF4
    in CI environments where the two extensions may link against different
    HDF5 builds.
    """
    print("Generating test fixture files...", flush=True)
    print(f"Base directory: {base_dir}\n", flush=True)
    script_path = os.path.abspath(__file__)

    for section in SECTION_GENERATORS:
        subprocess.run(
            [sys.executable, script_path, "--section", section],
            check=True,
        )

    print("Fixture generation complete.", flush=True)
    return 0


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if len(sys.argv) == 3 and sys.argv[1] == "--section":
        sys.exit(run_section(sys.argv[2], base_dir))

    sys.exit(run_all_sections(base_dir))
