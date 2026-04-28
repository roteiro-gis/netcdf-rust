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

    # ---- 11b. external_raw.h5 ----
    # Tests: contiguous dataset raw data stored in an external binary file.
    path = os.path.join(hdf5_dir, "external_raw.h5")
    external_name = "external_raw.bin"
    external_path = os.path.join(hdf5_dir, external_name)
    print(f"  Generating {path}")
    external_data = np.arange(12, dtype=np.int32)
    external_data.tofile(external_path)
    with h5py.File(path, "w") as f:
        f.create_dataset(
            "data",
            shape=external_data.shape,
            dtype=external_data.dtype,
            external=[(external_name, 0, external_data.nbytes)],
        )

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

    # ---- 17b. external_links.h5 ----
    # Tests: optional external-link resolver across HDF5 files.
    target_path = os.path.join(hdf5_dir, "external_link_target.h5")
    path = os.path.join(hdf5_dir, "external_links.h5")
    print(f"  Generating {target_path}")
    with h5py.File(target_path, "w") as f:
        f.create_dataset("target_data", data=np.array([4, 5, 6], dtype=np.int32))
    print(f"  Generating {path}")
    with h5py.File(path, "w") as f:
        f["linked_data"] = h5py.ExternalLink("external_link_target.h5", "/target_data")

    # ---- 18. fixed_array_chunked.h5 ----
    # Tests: Fixed Array chunk indexing (libver='latest', fixed-size chunked).
    path = os.path.join(hdf5_dir, "fixed_array_chunked.h5")
    print(f"  Generating {path}")
    with h5py.File(path, "w", libver="latest") as f:
        data = np.arange(60, dtype=np.float64).reshape(6, 10)
        f.create_dataset("data", data=data, chunks=(3, 5), compression="gzip")

    # ---- 19. chunked_lz4.h5 ----
    # Tests: LZ4 compression filter (requires hdf5plugin for writing).
    path = os.path.join(hdf5_dir, "chunked_lz4.h5")
    print(f"  Generating {path}")
    try:
        import hdf5plugin

        with h5py.File(path, "w") as f:
            data = np.arange(200, dtype=np.float32).reshape(10, 20)
            f.create_dataset(
                "data",
                data=data,
                chunks=(5, 10),
                **hdf5plugin.LZ4(),
            )
        # ---- 19b. chunked_lz4_zeros.h5 ----
        # Tests: LZ4 with data that actually compresses (exercises the compressed block path).
        path = os.path.join(hdf5_dir, "chunked_lz4_zeros.h5")
        print(f"  Generating {path}")
        with h5py.File(path, "w") as f:
            data = np.zeros((10, 20), dtype=np.float32)
            f.create_dataset(
                "data",
                data=data,
                chunks=(5, 10),
                **hdf5plugin.LZ4(),
            )
    except ImportError:
        print("    SKIPPED: hdf5plugin not available")

    # ---- 20. extensible_array_chunked.h5 ----
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

    # ---- 21. climate_4d.h5 ----
    # Tests: 4D dataset (time x level x lat x lon) with coordinate variables,
    # chunked + deflate. Critical for climate/netcdf-rust integration testing.
    path = os.path.join(hdf5_dir, "climate_4d.h5")
    print(f"  Generating {path}")
    with h5py.File(path, "w") as f:
        ntime, nlevel, nlat, nlon = 4, 3, 6, 12
        # Coordinate variables
        time_ds = f.create_dataset("time", data=np.arange(ntime, dtype=np.float64))
        time_ds.attrs["units"] = "hours since 2000-01-01"
        time_ds.attrs["calendar"] = "standard"

        level_ds = f.create_dataset("level", data=np.array([1000, 850, 500], dtype=np.float64))
        level_ds.attrs["units"] = "hPa"
        level_ds.attrs["positive"] = "down"

        lat_vals = np.linspace(-75.0, 75.0, nlat, dtype=np.float64)
        lat_ds = f.create_dataset("lat", data=lat_vals)
        lat_ds.attrs["units"] = "degrees_north"

        lon_vals = np.linspace(0.0, 330.0, nlon, dtype=np.float64)
        lon_ds = f.create_dataset("lon", data=lon_vals)
        lon_ds.attrs["units"] = "degrees_east"

        # Main 4D variable with chunked + deflate compression
        np.random.seed(100)
        temp_data = (280.0 + 20.0 * np.random.rand(ntime, nlevel, nlat, nlon)).astype(np.float32)
        temp = f.create_dataset(
            "temperature", data=temp_data,
            chunks=(1, 1, nlat, nlon), compression="gzip", compression_opts=4,
        )
        temp.attrs["units"] = "K"
        temp.attrs["long_name"] = "air temperature"

        f.attrs["Conventions"] = "CF-1.8"
        f.attrs["title"] = "4D climate test fixture"

    # ---- 22. big_endian.h5 ----
    # Tests: big-endian (BE) floating-point and integer data in HDF5.
    path = os.path.join(hdf5_dir, "big_endian.h5")
    print(f"  Generating {path}")
    with h5py.File(path, "w") as f:
        be_f32 = np.dtype(">f4")
        be_f64 = np.dtype(">f8")
        be_i32 = np.dtype(">i4")

        data_f32 = np.arange(20, dtype=np.float32).reshape(4, 5)
        f.create_dataset("float32_be", data=data_f32, dtype=be_f32)

        data_f64 = np.arange(12, dtype=np.float64).reshape(3, 4)
        f.create_dataset("float64_be", data=data_f64, dtype=be_f64)

        data_i32 = np.arange(15, dtype=np.int32).reshape(3, 5)
        f.create_dataset("int32_be", data=data_i32, dtype=be_i32)


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

    # ---- 7. climate_4d.nc ----
    # Tests: 4D dataset (time x level x lat x lon) in CDF-1 classic format
    # with coordinate variables. CDF-1 does not support chunking/compression,
    # so data is contiguous. Critical for climate/netcdf-rust integration testing.
    path = os.path.join(nc3_dir, "climate_4d.nc")
    print(f"  Generating {path}")
    ds = netCDF4.Dataset(path, "w", format="NETCDF3_CLASSIC")
    ds.Conventions = "CF-1.8"
    ds.title = "4D climate test fixture (CDF-1)"

    ntime, nlevel, nlat, nlon = 4, 3, 6, 12
    ds.createDimension("time", ntime)
    ds.createDimension("level", nlevel)
    ds.createDimension("lat", nlat)
    ds.createDimension("lon", nlon)

    time_var = ds.createVariable("time", "f8", ("time",))
    time_var.units = "hours since 2000-01-01"
    time_var.calendar = "standard"
    time_var[:] = np.arange(ntime, dtype=np.float64)

    level_var = ds.createVariable("level", "f8", ("level",))
    level_var.units = "hPa"
    level_var.positive = "down"
    level_var[:] = np.array([1000, 850, 500], dtype=np.float64)

    lat_var = ds.createVariable("lat", "f8", ("lat",))
    lat_var.units = "degrees_north"
    lat_var[:] = np.linspace(-75.0, 75.0, nlat, dtype=np.float64)

    lon_var = ds.createVariable("lon", "f8", ("lon",))
    lon_var.units = "degrees_east"
    lon_var[:] = np.linspace(0.0, 330.0, nlon, dtype=np.float64)

    np.random.seed(100)
    temp = ds.createVariable("temperature", "f4", ("time", "level", "lat", "lon"))
    temp.units = "K"
    temp.long_name = "air temperature"
    temp[:] = (280.0 + 20.0 * np.random.rand(ntime, nlevel, nlat, nlon)).astype(np.float32)
    ds.close()

    # ---- 8. packed_cf.nc ----
    # Tests: CF-convention packed integer data with scale_factor, add_offset,
    # and _FillValue. Typical for temperature stored as i16.
    path = os.path.join(nc3_dir, "packed_cf.nc")
    print(f"  Generating {path}")
    ds = netCDF4.Dataset(path, "w", format="NETCDF3_CLASSIC")
    ds.Conventions = "CF-1.8"
    ds.createDimension("x", 10)
    ds.createDimension("y", 10)

    temp = ds.createVariable("temperature", "i2", ("x", "y"), fill_value=np.int16(-9999))
    temp.scale_factor = np.float64(0.01)
    temp.add_offset = np.float64(273.15)
    temp.units = "K"
    temp.long_name = "temperature"
    # Disable auto-scaling so we write raw packed integers directly
    temp.set_auto_maskandscale(False)
    # Unpacked values: 273.15 + raw * 0.01
    # Raw range: 0..99 -> unpacked: 273.15..274.14
    raw_data = np.arange(100, dtype=np.int16).reshape(10, 10)
    # Set a few fill values to test masking
    raw_data[0, 0] = -9999
    raw_data[5, 5] = -9999
    raw_data[9, 9] = -9999
    temp[:] = raw_data
    ds.close()

    # ---- 9. shared_dims.nc ----
    # Tests: multiple variables (temperature, humidity, pressure) sharing
    # lat/lon dimensions with proper CF metadata.
    path = os.path.join(nc3_dir, "shared_dims.nc")
    print(f"  Generating {path}")
    ds = netCDF4.Dataset(path, "w", format="NETCDF3_CLASSIC")
    ds.Conventions = "CF-1.8"
    ds.title = "Multi-variable shared dimensions test"

    nlat, nlon = 8, 16
    ds.createDimension("lat", nlat)
    ds.createDimension("lon", nlon)

    lat_var = ds.createVariable("lat", "f8", ("lat",))
    lat_var.units = "degrees_north"
    lat_var.long_name = "latitude"
    lat_var[:] = np.linspace(-87.5, 87.5, nlat, dtype=np.float64)

    lon_var = ds.createVariable("lon", "f8", ("lon",))
    lon_var.units = "degrees_east"
    lon_var.long_name = "longitude"
    lon_var[:] = np.linspace(0.0, 337.5, nlon, dtype=np.float64)

    np.random.seed(200)
    temp = ds.createVariable("temperature", "f4", ("lat", "lon"))
    temp.units = "K"
    temp.long_name = "air temperature"
    temp[:] = (280.0 + 20.0 * np.random.rand(nlat, nlon)).astype(np.float32)

    humidity = ds.createVariable("humidity", "f4", ("lat", "lon"))
    humidity.units = "percent"
    humidity.long_name = "relative humidity"
    humidity[:] = (50.0 + 40.0 * np.random.rand(nlat, nlon)).astype(np.float32)

    pressure = ds.createVariable("pressure", "f4", ("lat", "lon"))
    pressure.units = "Pa"
    pressure.long_name = "surface pressure"
    pressure[:] = (101000.0 + 2000.0 * np.random.rand(nlat, nlon)).astype(np.float32)
    ds.close()

    # ---- 10. zero_record.nc ----
    # Tests: unlimited dimension with zero records written. Edge case that
    # exercises empty record handling.
    path = os.path.join(nc3_dir, "zero_record.nc")
    print(f"  Generating {path}")
    ds = netCDF4.Dataset(path, "w", format="NETCDF3_CLASSIC")
    ds.createDimension("time", None)  # unlimited, 0 records
    ds.createDimension("x", 5)
    series = ds.createVariable("series", "f4", ("time", "x"))
    series.units = "m/s"
    # Intentionally write zero records
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

    # ---- 8. nc4_climate_4d.nc ----
    # Tests: 4D dataset (time x level x lat x lon) in NetCDF-4/HDF5 format
    # with coordinate variables, chunked + zlib compression.
    # Critical for climate/netcdf-rust integration testing.
    path = os.path.join(nc4_dir, "nc4_climate_4d.nc")
    print(f"  Generating {path}")
    ds = netCDF4.Dataset(path, "w", format="NETCDF4")
    ds.Conventions = "CF-1.8"
    ds.title = "4D climate test fixture (NC4)"

    ntime, nlevel, nlat, nlon = 4, 3, 6, 12
    ds.createDimension("time", ntime)
    ds.createDimension("level", nlevel)
    ds.createDimension("lat", nlat)
    ds.createDimension("lon", nlon)

    time_var = ds.createVariable("time", "f8", ("time",))
    time_var.units = "hours since 2000-01-01"
    time_var.calendar = "standard"
    time_var[:] = np.arange(ntime, dtype=np.float64)

    level_var = ds.createVariable("level", "f8", ("level",))
    level_var.units = "hPa"
    level_var.positive = "down"
    level_var[:] = np.array([1000, 850, 500], dtype=np.float64)

    lat_var = ds.createVariable("lat", "f8", ("lat",))
    lat_var.units = "degrees_north"
    lat_var[:] = np.linspace(-75.0, 75.0, nlat, dtype=np.float64)

    lon_var = ds.createVariable("lon", "f8", ("lon",))
    lon_var.units = "degrees_east"
    lon_var[:] = np.linspace(0.0, 330.0, nlon, dtype=np.float64)

    np.random.seed(100)
    temp = ds.createVariable(
        "temperature", "f4", ("time", "level", "lat", "lon"),
        chunksizes=(1, 1, nlat, nlon), zlib=True, complevel=4,
    )
    temp.units = "K"
    temp.long_name = "air temperature"
    temp[:] = (280.0 + 20.0 * np.random.rand(ntime, nlevel, nlat, nlon)).astype(np.float32)
    ds.close()

    # ---- 9. nc4_packed_cf.nc ----
    # Tests: CF-convention packed integer data with scale_factor, add_offset,
    # and _FillValue in NetCDF-4 format. Typical for temperature stored as i16.
    path = os.path.join(nc4_dir, "nc4_packed_cf.nc")
    print(f"  Generating {path}")
    ds = netCDF4.Dataset(path, "w", format="NETCDF4")
    ds.Conventions = "CF-1.8"
    ds.createDimension("x", 10)
    ds.createDimension("y", 10)

    temp = ds.createVariable("temperature", "i2", ("x", "y"), fill_value=np.int16(-9999))
    temp.scale_factor = np.float64(0.01)
    temp.add_offset = np.float64(273.15)
    temp.units = "K"
    temp.long_name = "temperature"
    # Disable auto-scaling so we write raw packed integers directly
    temp.set_auto_maskandscale(False)
    # Unpacked values: 273.15 + raw * 0.01
    # Raw range: 0..99 -> unpacked: 273.15..274.14
    raw_data = np.arange(100, dtype=np.int16).reshape(10, 10)
    # Set a few fill values to test masking
    raw_data[0, 0] = -9999
    raw_data[5, 5] = -9999
    raw_data[9, 9] = -9999
    temp[:] = raw_data
    ds.close()

    # ---- 10. nc4_shared_dims.nc ----
    # Tests: multiple variables (temperature, humidity, pressure) sharing
    # lat/lon dimensions with proper CF metadata in NetCDF-4 format.
    path = os.path.join(nc4_dir, "nc4_shared_dims.nc")
    print(f"  Generating {path}")
    ds = netCDF4.Dataset(path, "w", format="NETCDF4")
    ds.Conventions = "CF-1.8"
    ds.title = "Multi-variable shared dimensions test (NC4)"

    nlat, nlon = 8, 16
    ds.createDimension("lat", nlat)
    ds.createDimension("lon", nlon)

    lat_var = ds.createVariable("lat", "f8", ("lat",))
    lat_var.units = "degrees_north"
    lat_var.long_name = "latitude"
    lat_var[:] = np.linspace(-87.5, 87.5, nlat, dtype=np.float64)

    lon_var = ds.createVariable("lon", "f8", ("lon",))
    lon_var.units = "degrees_east"
    lon_var.long_name = "longitude"
    lon_var[:] = np.linspace(0.0, 337.5, nlon, dtype=np.float64)

    np.random.seed(200)
    temp = ds.createVariable(
        "temperature", "f4", ("lat", "lon"),
        chunksizes=(4, 8), zlib=True, complevel=4,
    )
    temp.units = "K"
    temp.long_name = "air temperature"
    temp[:] = (280.0 + 20.0 * np.random.rand(nlat, nlon)).astype(np.float32)

    humidity = ds.createVariable(
        "humidity", "f4", ("lat", "lon"),
        chunksizes=(4, 8), zlib=True, complevel=4,
    )
    humidity.units = "percent"
    humidity.long_name = "relative humidity"
    humidity[:] = (50.0 + 40.0 * np.random.rand(nlat, nlon)).astype(np.float32)

    pressure = ds.createVariable(
        "pressure", "f4", ("lat", "lon"),
        chunksizes=(4, 8), zlib=True, complevel=4,
    )
    pressure.units = "Pa"
    pressure.long_name = "surface pressure"
    pressure[:] = (101000.0 + 2000.0 * np.random.rand(nlat, nlon)).astype(np.float32)
    ds.close()

    # ---- 11. nc4_zero_record.nc ----
    # Tests: unlimited dimension with zero records written in NetCDF-4 format.
    # Edge case that exercises empty record handling in HDF5-backed storage.
    path = os.path.join(nc4_dir, "nc4_zero_record.nc")
    print(f"  Generating {path}")
    ds = netCDF4.Dataset(path, "w", format="NETCDF4")
    ds.createDimension("time", None)  # unlimited, 0 records
    ds.createDimension("x", 5)
    series = ds.createVariable("series", "f4", ("time", "x"), chunksizes=(1, 5))
    series.units = "m/s"
    # Intentionally write zero records
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
