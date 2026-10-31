#!/usr/bin/env python3
"""
Generate HDF5 and NetCDF test fixture files for the netcdf-rust crate.

Requirements:
    pip install h5py netCDF4 numpy

Creates fixtures in hdf5/, netcdf3/, and netcdf4/ subdirectories
relative to this script's location.
"""

import os
import struct
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

    # ---- 6b. sohm_shared_attrs.h5 ----
    # Tests: shared object header messages (SOHM). Identical attributes on a
    # group and a dataset are stored once in the shared-message table and
    # referenced from both object headers, exercising the SOHM resolver on both
    # the dataset and group attribute paths. Only generated on h5py builds that
    # expose the low-level shared-message-table configuration.
    if hasattr(h5py.h5p.create(h5py.h5p.FILE_CREATE), "set_shared_mesg_nindexes"):
        path = os.path.join(hdf5_dir, "sohm_shared_attrs.h5")
        print(f"  Generating {path}")
        fcpl = h5py.h5p.create(h5py.h5p.FILE_CREATE)
        fcpl.set_shared_mesg_nindexes(1)
        fcpl.set_shared_mesg_index(0, h5py.h5f.MESG_ATTR_FLAG, 1)
        fapl = h5py.h5p.create(h5py.h5p.FILE_ACCESS)
        fapl.set_libver_bounds(h5py.h5f.LIBVER_LATEST, h5py.h5f.LIBVER_LATEST)
        file_id = h5py.h5f.create(path.encode(), h5py.h5f.ACC_TRUNC, fcpl=fcpl, fapl=fapl)
        with h5py.File(file_id) as f:
            shared_value = np.arange(4, dtype=np.int32)
            grp = f.create_group("shared_group")
            grp.attrs["shared_meta"] = shared_value
            ds = f.create_dataset("shared_data", data=np.arange(6, dtype=np.int32))
            ds.attrs["shared_meta"] = shared_value
    else:
        print("  Skipping sohm_shared_attrs.h5 (h5py lacks shared-message API)")

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

    # ---- 10d. old_format_soft_links.h5 ----
    # Tests: soft links in v1 (symbol table) groups. Their symbol table entries
    # hold no object header address; the scratch pad holds the local heap offset
    # of the link value instead.
    path = os.path.join(hdf5_dir, "old_format_soft_links.h5")
    print(f"  Generating {path}")
    with h5py.File(path, "w", libver="earliest", track_order=False) as f:
        f.create_dataset("data", data=np.arange(5, dtype=np.int32))
        g = f.create_group("group")
        g.create_dataset("inner", data=np.array([7, 8], dtype=np.int32))
        f["link"] = h5py.SoftLink("/data")
        f["group_link"] = h5py.SoftLink("/group")
        g["relative"] = h5py.SoftLink("inner")
        g["dangling"] = h5py.SoftLink("/missing")

    # ---- 10e. old_format_large_group.h5 ----
    # Tests: an old-style (symbol table) root group with 40 members, 20 groups
    # g00-g19 and 20 int32 datasets d00-d19 holding their index.
    path = os.path.join(hdf5_dir, "old_format_large_group.h5")
    print(f"  Generating {path}")
    with h5py.File(path, "w", libver="earliest", track_order=False) as f:
        for i in range(20):
            f.create_group(f"g{i:02d}")
            f.create_dataset(f"d{i:02d}", data=np.array([i], dtype=np.int32))

    # ---- 10f. old_format_empty_heap_past_eof.h5 ----
    # Tests: an empty old-style root group whose local heap data segment
    # address points past the end of the file. Its heap holds no member name,
    # so it must not be read. Written by h5py, then the heap's data segment
    # address is edited.
    path = os.path.join(hdf5_dir, "old_format_empty_heap_past_eof.h5")
    print(f"  Generating {path}")
    with h5py.File(path, "w", libver="earliest", track_order=False) as f:
        pass
    with open(path, "r+b") as fh:
        raw = fh.read()
        assert raw.count(b"HEAP") == 1
        # signature, version, 3 reserved bytes, data segment size and free
        # list offset (8 bytes each), then the data segment address
        fh.seek(raw.index(b"HEAP") + 24)
        fh.write(struct.pack("<Q", len(raw) + 4096))

    # ---- 10b. compound_v1_floats.h5 ----
    # Tests: version 1 compound datatype (earliest format) whose member names start off
    # the 8-byte grid: a float member (20-byte description) followed by a fixed-length
    # string member. HEC-RAS geometry and plan files store their attribute tables so.
    path = os.path.join(hdf5_dir, "compound_v1_floats.h5")
    print(f"  Generating {path}")
    with h5py.File(path, "w", libver="earliest", track_order=False) as f:
        dt = np.dtype([("x", np.float32), ("name", "S16"), ("count", np.int32)])
        data = np.array([(1.5, b"alpha", 1), (2.5, b"beta", 2), (3.5, b"gamma", 3)], dtype=dt)
        f.create_dataset("table", data=data)

    # ---- 10c. fill_value_old_message.h5 ----
    # Tests: the old fill value message (size, then value) on a chunked dataset with an
    # unwritten chunk (earliest format), which must read back as the fill value.
    path = os.path.join(hdf5_dir, "fill_value_old_message.h5")
    print(f"  Generating {path}")
    with h5py.File(path, "w", libver="earliest", track_order=False) as f:
        ds = f.create_dataset("sparse", shape=(8, 2), dtype=np.int32, chunks=(4, 2), fillvalue=-2147483648)
        ds[0:4] = np.arange(8, dtype=np.int32).reshape(4, 2)

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

    # ---- 12b. chained_filters.h5 ----
    # Tests: filter pipelines whose intermediate output is larger than the
    # chunk, because it is still encoded by an earlier filter. "deflate_twice"
    # applies DEFLATE twice to 8-byte chunks (one float64 each);
    # "scaleoffset_gzip" is h5py's scaleoffset=0 + gzip on 64 int32 values
    # that need 31 bits, so scale-offset's output (21-byte header and 248
    # bytes of packed values) is larger than the 256-byte chunk.
    path = os.path.join(hdf5_dir, "chained_filters.h5")
    print(f"  Generating {path}")
    with h5py.File(path, "w") as f:
        dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
        dcpl.set_chunk((1,))
        dcpl.set_deflate(1)
        dcpl.set_deflate(1)
        space = h5py.h5s.create_simple((4,))
        dsid = h5py.h5d.create(f.id, b"deflate_twice", h5py.h5t.IEEE_F64LE, space, dcpl=dcpl)
        h5py.Dataset(dsid)[:] = np.array([-0.0, 1.5, 2.25, -3.0])
        values = np.arange(64, dtype=np.int32) * np.int32(2**25 - 1)
        f.create_dataset(
            "scaleoffset_gzip", data=values, chunks=(64,), scaleoffset=0, compression="gzip"
        )

    # ---- 13. dense_groups.h5 ----
    # Tests: dense link storage (fractal heap + B-tree v2), triggered by 20+ children.
    path = os.path.join(hdf5_dir, "dense_groups.h5")
    print(f"  Generating {path}")
    with h5py.File(path, "w", libver="latest") as f:
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

    # ---- 16b. stale_chunks.h5 ----
    # Tests: chunk index entries wholly outside the dataset's extent, as some
    # writers leave them when a dataset shrinks. libhdf5 removes such chunks,
    # so this writes 4x4 int32 values in 2x2 chunks (v1 B-tree index) and then
    # edits the dataspace to 2x4: the two chunks of rows 2-3 become stale.
    path = os.path.join(hdf5_dir, "stale_chunks.h5")
    print(f"  Generating {path}")
    with h5py.File(path, "w", libver="earliest") as f:
        f.create_dataset(
            "data", data=np.arange(16, dtype=np.int32).reshape(4, 4), chunks=(2, 2),
            maxshape=(None, 4)
        )
    with open(path, "r+b") as fh:
        raw = fh.read()
        dims = struct.pack("<4Q", 4, 4, 0xFFFFFFFFFFFFFFFF, 4)
        assert raw.count(dims) == 1
        fh.seek(raw.index(dims))
        fh.write(struct.pack("<Q", 2))

    # ---- 17. committed_dtype.h5 ----
    # Tests: committed (shared) datatype — a named datatype that is shared across datasets.
    path = os.path.join(hdf5_dir, "committed_dtype.h5")
    print(f"  Generating {path}")
    with h5py.File(path, "w") as f:
        dt = np.dtype(np.float64)
        f["my_type"] = dt  # commit the type
        # Reference the committed type
        ds = f.create_dataset("data", data=np.array([1.0, 2.0, 3.0]), dtype=f["my_type"].dtype)

    # ---- 17c. committed_dtype_dataset.h5, committed_dtype_dataset_latest.h5 ----
    # Tests: datasets whose datatype message is shared: it refers to a committed
    # (named) datatype, in the earliest and the latest file format.
    for name, libver in [
        ("committed_dtype_dataset.h5", "earliest"),
        ("committed_dtype_dataset_latest.h5", "latest"),
    ]:
        path = os.path.join(hdf5_dir, name)
        print(f"  Generating {path}")
        with h5py.File(path, "w", libver=libver) as f:
            f["f8"] = np.dtype("<f8")
            f.create_dataset("data", data=np.arange(5, dtype=np.float64), dtype=f["f8"])
            f["i4"] = np.dtype("<i4")
            f.create_dataset("ints", data=np.array([-1, 0, 1], dtype=np.int32), dtype=f["i4"])

    # ---- 17e. raw_encodings.h5, raw_encodings_latest.h5 ----
    # Tests: the exact stored bytes of datatype messages (Dataset::raw_datatype)
    # and attribute datatypes (raw_attributes), in the earliest format (v1
    # object headers, messages padded to 8 bytes) and the latest (v2 headers;
    # the 20 attributes of /dense are in dense storage). "spacepad" is a
    # space-padded fixed-length string, "padded_compound" a compound type with
    # a gap between its members.
    for name, libver in [
        ("raw_encodings.h5", "earliest"),
        ("raw_encodings_latest.h5", "latest"),
    ]:
        path = os.path.join(hdf5_dir, name)
        print(f"  Generating {path}")
        with h5py.File(path, "w", libver=libver) as f:
            ds = f.create_dataset("f8", data=np.arange(3, dtype="<f8"))
            ds.attrs["units"] = np.bytes_(b"m")
            ds.attrs["scale"] = np.float32(0.5)
            ds.attrs["labels"] = np.array([b"ab", b"cde"], dtype="S4")
            ds.attrs["note"] = "variable-length text"
            spacepad = h5py.h5t.C_S1.copy()
            spacepad.set_size(6)
            spacepad.set_strpad(h5py.h5t.STR_SPACEPAD)
            space = h5py.h5s.create_simple((2,))
            dsid = h5py.h5d.create(f.id, b"spacepad", spacepad, space)
            dsid.write(h5py.h5s.ALL, h5py.h5s.ALL, np.array([b"ab    ", b"cdef  "], dtype="S6"))
            padded = np.dtype({"names": ["a", "b"], "formats": ["<i2", "<f8"],
                               "offsets": [0, 8], "itemsize": 16})
            f.create_dataset("padded_compound", data=np.zeros(2, dtype=padded))
            f.create_dataset("i2", data=np.arange(2, dtype="<i2"))
            g = f.create_group("dense")
            for i in range(20):
                g.attrs[f"a{i:02d}"] = np.int32(i)
            g.attrs["text"] = np.bytes_(b"fixed")

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

    # ---- 17d. soft_link_cycles.h5 ----
    # Tests: soft links that cannot be resolved because they lead back to
    # themselves, through an intermediate component of their own target path
    # ("s" -> "/s/x") or through each other ("a" -> "/b", "b" -> "/a").
    path = os.path.join(hdf5_dir, "soft_link_cycles.h5")
    print(f"  Generating {path}")
    with h5py.File(path, "w", libver="latest") as f:
        f.create_dataset("data", data=np.arange(3, dtype=np.int32))
        f["ok"] = h5py.SoftLink("/data")
        f["s"] = h5py.SoftLink("/s/x")
        f["a"] = h5py.SoftLink("/b")
        f["b"] = h5py.SoftLink("/a")

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

    # ---- 20b. filtered_chunk_size_widths.h5 ----
    # Tests: the chunk-size field of filtered chunk-index entries is sized to the
    # chunk (1 + (log2(chunk bytes) + 8) / 8 bytes): 256 B - 64 KiB chunks store
    # it in 3 bytes, 16 MiB - 4 GiB chunks in 5. Fixed array, extensible array and
    # v2 B-tree indexes, gzip: 16x16 float64 chunks (2 KiB, 3 bytes) and
    # 2048x1024 float64 chunks (16 MiB, 5 bytes; zeros with sparse markers
    # r * 2048 + c at r % 1021 == 0 and c % 509 == 0, to keep the file small).
    path = os.path.join(hdf5_dir, "filtered_chunk_size_widths.h5")
    print(f"  Generating {path}")
    with h5py.File(path, "w", libver="latest") as f:
        data = np.arange(32 * 48, dtype=np.float64).reshape(32, 48)
        large = np.zeros((4096, 2048), dtype=np.float64)
        rows, cols = np.meshgrid(np.arange(0, 4096, 1021), np.arange(0, 2048, 509), indexing="ij")
        large[rows, cols] = rows * 2048 + cols
        for name, maxshape, large_maxshape in [
            ("fixed", None, None),
            ("extensible", (None, 48), (None, 2048)),
            ("btree_v2", (None, None), (None, None)),
        ]:
            ds = f.create_dataset(
                name, shape=data.shape, maxshape=maxshape, dtype=np.float64,
                chunks=(16, 16), compression="gzip"
            )
            ds[:] = data
            ds = f.create_dataset(
                name + "_16mib", shape=large.shape, maxshape=large_maxshape,
                dtype=np.float64, chunks=(2048, 1024), compression="gzip",
                compression_opts=9
            )
            ds[:] = large

    # ---- 20c. extensible_array_super_blocks.h5 ----
    # Tests: extensible array chunk indexes that grow past the index block.
    # libhdf5 creates them with 4 index block elements, 16 elements in the
    # smallest data block, 4 data block pointers in the smallest secondary
    # block and 1024-element data block pages. Super block u then holds
    # 2^(u/2) data blocks of 2^((u+1)/2) * 16 elements: chunks 0-3 live in the
    # index block, 4-243 in the six data blocks it points to directly (super
    # blocks 0-3), later chunks behind secondary blocks, and from chunk 131060
    # (super block 13: 64 data blocks of 2048 elements) the data blocks are
    # paged, with each page's "initialized" bit in the secondary block.
    # "data" / "data_gzip": 600 one-row chunks, all written, so every chunk up
    # to super block 5 is set. "sparse" / "sparse_gzip": 133200 one-element
    # chunks with fill value -1 and only a few written, among them chunks in
    # the second page of the first paged data block (132084, 133107) and the
    # first page of the second one (133108), whose first pages stay unwritten.
    path = os.path.join(hdf5_dir, "extensible_array_super_blocks.h5")
    print(f"  Generating {path}")
    with h5py.File(path, "w", libver="latest") as f:
        data = np.arange(600 * 4, dtype=np.int32).reshape(600, 4)
        for name, compression in [("data", None), ("data_gzip", "gzip")]:
            ds = f.create_dataset(
                name, shape=data.shape, maxshape=(None, 4), dtype=np.int32,
                chunks=(1, 4), compression=compression
            )
            ds[:] = data
        written = [0, 3, 4, 19, 20, 35, 36, 51, 52, 243, 244, 499, 500, 599,
                   65524, 132084, 133107, 133108]
        for name, compression in [("sparse", None), ("sparse_gzip", "gzip")]:
            ds = f.create_dataset(
                name, shape=(133200,), maxshape=(None,), dtype=np.int32,
                chunks=(1,), fillvalue=-1, compression=compression
            )
            for i in written:
                ds[i] = i

    # ---- 20d. extensible_array_max_dims.h5 ----
    # Tests: libhdf5 numbers extensible array chunks row-major over the chunk
    # counts at the maximum dimensions, with the unlimited dimension moved to
    # the front. Each dataset holds int32 np.arange values (fill value -1):
    # "wider_max" has a fixed dimension whose maximum holds 4 chunks per row
    # but whose extent holds 2; "wider_max_same_chunks" one whose maximum adds
    # no chunk; "unlimited_last" and "unlimited_middle" an unlimited dimension
    # that is not the first.
    path = os.path.join(hdf5_dir, "extensible_array_max_dims.h5")
    print(f"  Generating {path}")
    with h5py.File(path, "w", libver="latest") as f:
        for name, shape, maxshape, chunks in [
            ("wider_max", (25, 8), (None, 16), (1, 4)),
            ("wider_max_same_chunks", (50, 7), (None, 8), (1, 4)),
            ("unlimited_last", (8, 25), (8, None), (4, 1)),
            ("unlimited_middle", (2, 25, 3), (2, None, 3), (1, 1, 3)),
        ]:
            ds = f.create_dataset(
                name, shape=shape, maxshape=maxshape, chunks=chunks, dtype=np.int32,
                fillvalue=-1
            )
            ds[...] = np.arange(np.prod(shape), dtype=np.int32).reshape(shape)

    # ---- 20e. row_column_gather.h5 ----
    # Tests: reading a row range and a column list of 2-D chunked datasets.
    # 17x19 values v[r, c] = r * 19 + c - 123 in 5x7 chunks (partial chunks at
    # the right and bottom edges), as little-endian float32 and float64 and
    # big-endian int32, each unfiltered ("plain") and with shuffle + gzip
    # ("filtered"); "fill" is int32 with fill value -7 where only rows 0-4
    # (the first row of chunks) were written, with 42.
    path = os.path.join(hdf5_dir, "row_column_gather.h5")
    print(f"  Generating {path}")
    with h5py.File(path, "w") as f:
        values = np.arange(17 * 19).reshape(17, 19) - 123
        for prefix, dtype in [("lef4", "<f4"), ("lef8", "<f8"), ("bei4", ">i4")]:
            f.create_dataset(prefix + "plain", data=values.astype(dtype), chunks=(5, 7))
            f.create_dataset(
                prefix + "filtered", data=values.astype(dtype), chunks=(5, 7), shuffle=True,
                compression="gzip"
            )
        ds = f.create_dataset("fill", shape=(17, 19), dtype="<i4", chunks=(5, 7), fillvalue=-7)
        ds[0:5, :] = 42

    # ---- 20f. row_column_gather_wide_sparse.h5 ----
    # A fixed-array chunk index with 1024 column chunks but only two
    # allocated ones. Gathering the endpoints should not walk the gap.
    path = os.path.join(hdf5_dir, "row_column_gather_wide_sparse.h5")
    print(f"  Generating {path}")
    with h5py.File(path, "w", libver="latest") as f:
        ds = f.create_dataset(
            "data", shape=(1, 1024), dtype="<i4", chunks=(1, 1), fillvalue=-7
        )
        ds[0, 0] = 11
        ds[0, 1023] = 22

    # ---- 20g. fixed_array_paged_sparse.h5 ----
    # Nine 1024-entry pages with only the first and ninth initialized. The
    # fixed-array page bitmap uses the high bit for the first page in each byte.
    path = os.path.join(hdf5_dir, "fixed_array_paged_sparse.h5")
    print(f"  Generating {path}")
    with h5py.File(path, "w", libver="latest") as f:
        ds = f.create_dataset(
            "data", shape=(1, 8193), dtype="<i4", chunks=(1, 1), fillvalue=-7
        )
        ds[0, 0] = 11
        ds[0, 8192] = 22

    # ---- 20h. row_column_gather_wide.h5, row_column_gather_wide_v1.h5 ----
    # Tests: gathers of widely separated columns look up only the selected
    # chunks. 4x1000 int32 values v[r, c] = r * 1000 + c in 2x1 chunks (2000
    # one-column chunks), indexed by a fixed array ("fixed"; paged), an
    # extensible array ("earray"), a v2 B-tree ("btree2") and an implicit
    # index ("implicit", allocated early), and in the _v1 file (earliest
    # format) by a v1 B-tree ("btree1").
    values = (np.arange(4)[:, None] * 1000 + np.arange(1000)).astype(np.int32)
    path = os.path.join(hdf5_dir, "row_column_gather_wide.h5")
    print(f"  Generating {path}")
    with h5py.File(path, "w", libver="latest") as f:
        for name, maxshape in [("fixed", None), ("earray", (4, None)), ("btree2", (None, None))]:
            f.create_dataset(name, data=values, chunks=(2, 1), maxshape=maxshape)
        dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
        dcpl.set_alloc_time(h5py.h5d.ALLOC_TIME_EARLY)
        f.create_dataset("implicit", data=values, chunks=(2, 1), dcpl=dcpl)
    path = os.path.join(hdf5_dir, "row_column_gather_wide_v1.h5")
    print(f"  Generating {path}")
    with h5py.File(path, "w", libver="earliest") as f:
        f.create_dataset("btree1", data=values, chunks=(2, 1))


    # ---- 21. btree_v2_chunked.h5 ----
    # Tests: v2 B-tree chunk indexing. HDF5 uses this for chunked datasets
    # with more than one unlimited dimension when libver='latest'.
    path = os.path.join(hdf5_dir, "btree_v2_chunked.h5")
    print(f"  Generating {path}")
    with h5py.File(path, "w", libver="latest") as f:
        data = np.zeros((4, 6), dtype=np.int32)
        for r in range(data.shape[0]):
            for c in range(data.shape[1]):
                data[r, c] = r * 10 + c
        ds = f.create_dataset(
            "data", shape=data.shape, maxshape=(None, None), dtype=np.int32,
            chunks=(2, 3)
        )
        ds[:] = data

    # ---- 22. climate_4d.h5 ----
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


    # ---- 23. dense_huge_attribute.h5 ----
    # Tests: a "huge" fractal-heap object in dense attribute storage. Attribute heaps use
    # 8-byte heap IDs, too short to hold the address and length directly, so libhdf5
    # stores a 7-byte B-tree key (min(id_len - 1, sizeof(hsize_t))) after the flag byte.
    path = os.path.join(hdf5_dir, "dense_huge_attribute.h5")
    print(f"  Generating {path}")
    with h5py.File(path, "w", libver="latest") as f:
        ds = f.create_dataset("data", data=np.arange(4, dtype=np.int32))
        for i in range(12):
            ds.attrs[f"small_{i:02d}"] = np.int32(i)
        ds.attrs["huge"] = np.arange(50_000, dtype=np.float32)

    # ---- user_block.h5 ----
    # Tests: addresses relative to a superblock moved 1024 bytes in by a user block.
    path = os.path.join(hdf5_dir, "user_block.h5")
    print(f"  Generating {path}")
    with h5py.File(path, "w", userblock_size=1024, libver="earliest") as f:
        f.attrs["title"] = "behind a user block"
        grp = f.create_group("group")
        grp.create_dataset("data", data=np.arange(6, dtype="f8").reshape(2, 3))
        grp.create_dataset(
            "chunked", data=np.arange(100, dtype="i4"), chunks=(10,), compression="gzip"
        )
    with open(path, "r+b") as f:
        f.write(b"application data, not HDF5")


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

    # ---- 12. dense_empty_attributes.nc ----
    # Tests: the eighth user attribute moves NetCDF-4 root attributes from
    # compact object-header storage into a fractal heap with v2 B-tree indexes.
    path = os.path.join(nc4_dir, "dense_empty_attributes.nc")
    print(f"  Generating {path}")
    ds = netCDF4.Dataset(path, "w", format="NETCDF4")
    for index in range(8):
        ds.setncattr(f"att_{index}", "")
    ds.close()

    # ---- 13. nc4_scalar_variable.nc ----
    # Tests: rank-0 variables, which NetCDF-4 writes without a DIMENSION_LIST
    # attribute because they have no dimensions to reference. Shaped after a
    # Copernicus CDS ERA5 download, whose scalar `number` (ensemble member) and
    # `expver` variables sit alongside the gridded fields.
    path = os.path.join(nc4_dir, "nc4_scalar_variable.nc")
    print(f"  Generating {path}")
    ds = netCDF4.Dataset(path, "w", format="NETCDF4")
    ds.createDimension("valid_time", 2)
    ds.createDimension("latitude", 3)
    ds.createDimension("longitude", 4)

    valid_time = ds.createVariable("valid_time", "i8", ("valid_time",))
    valid_time.units = "seconds since 1970-01-01"
    valid_time.calendar = "proleptic_gregorian"
    valid_time[:] = [0, 3600]

    latitude = ds.createVariable("latitude", "f4", ("latitude",))
    latitude.units = "degrees_north"
    latitude[:] = [10.0, 11.0, 12.0]

    longitude = ds.createVariable("longitude", "f4", ("longitude",))
    longitude.units = "degrees_east"
    longitude[:] = [0.0, 1.0, 2.0, 3.0]

    # Scalar: no dimensions, therefore no DIMENSION_LIST on disk.
    number = ds.createVariable("number", "i8", ())
    number.long_name = "ensemble member numerical id"
    number[...] = 0

    t = ds.createVariable("t", "f4", ("valid_time", "latitude", "longitude"))
    t.units = "K"
    t.long_name = "temperature"
    t[:] = 280.0
    ds.close()

    # ---- 14. creation_order_variables.nc ----
    # Tests: variables listed in creation order, as netcdf-c numbers them. libhdf5
    # stores compact link messages wherever the root header has room, so on disk
    # their order differs from creation order; a reader that lists storage order
    # reorders the variables. The generator asserts that the two orders differ.
    path = os.path.join(nc4_dir, "creation_order_variables.nc")
    print(f"  Generating {path}")
    ds = netCDF4.Dataset(path, "w", format="NETCDF4")
    ds.createDimension("y", 3)
    ds.createDimension("x", 4)
    ds.createDimension("time", 2)
    ds.createVariable("y", "f4", ("y",))[:] = [40.0, 41.0, 42.0]
    ds.createVariable("x", "f4", ("x",))[:] = [26.0, 27.0, 28.0, 29.0]
    ds.createVariable("time", "f8", ("time",))[:] = [0.0, 3600.0]
    for name in ["Hs", "Tp", "Dir", "Te"]:
        ds.createVariable(name, "f4", ("y", "x", "time"))[:] = 1.0
    ds.close()
    import h5py

    def _link_order(index, order):
        names = []
        with h5py.File(path, "r") as f:
            f.id.links.iterate(lambda n: names.append(n.decode()), idx_type=index, order=order)
        return names

    created = _link_order(h5py.h5.INDEX_CRT_ORDER, h5py.h5.ITER_INC)
    stored = _link_order(h5py.h5.INDEX_NAME, h5py.h5.ITER_NATIVE)
    assert created == ["y", "x", "time", "Hs", "Tp", "Dir", "Te"], created
    assert stored != created, "fixture no longer exercises storage != creation order"

    # ---- nc4_uneven_unlimited.nc ----
    # Tests: variables that share an unlimited dimension but hold different record counts.
    # Only the datasets written to are extended, so the coordinate is shorter than `longest`.
    path = os.path.join(nc4_dir, "nc4_uneven_unlimited.nc")
    print(f"  Generating {path}")
    with netCDF4.Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("time", None)
        ds.createDimension("x", 2)
        time = ds.createVariable("time", "f8", ("time",))
        longest = ds.createVariable("longest", "f4", ("time", "x"))
        shortest = ds.createVariable("shortest", "f4", ("time", "x"))
        time[:3] = [0.0, 1.0, 2.0]
        longest[:6] = np.arange(12, dtype="f4").reshape(6, 2)
        shortest[:2] = np.arange(4, dtype="f4").reshape(2, 2)


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
