"""Generate the NetCDF-4 release compatibility regression fixtures."""

from pathlib import Path
import ctypes

import h5py
import netCDF4
import numpy as np


def generate(nc4_dir):
    nc4_dir = Path(nc4_dir)
    nc4_dir.mkdir(parents=True, exist_ok=True)
    path = nc4_dir / "nc4_descendant_unlimited.nc"
    with netCDF4.Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("time", None)
        ds.createDimension("other", None)
        ds.createDimension("x", 2)
        ds.createVariable("time", "i4", ("time",))[:2] = [10, 20]
        short = ds.createVariable("short", "i2", ("time", "x"),
                                  fill_value=-999, chunksizes=(4, 2), zlib=True)
        short[:1] = [[1, 2]]
        ds.createVariable("empty", "f4", ("time",))
        ds.createVariable("labels", str, ("time",))[:2] = np.array(["a", "b"], dtype=object)
        ds.createVariable("other", "i4", ("other",))[:4] = [0, 1, 2, 3]
        ds.createVariable("grid", "i4", ("time", "other"), fill_value=-77,
                          chunksizes=(4, 3))[:2, :1] = [[5], [6]]
        sequences = ds.createVLType(np.int32, "sequence_t")
        ds.createVariable("sequences", sequences, ("time",))[0] = np.array([7, 8], dtype="i4")
        child = ds.createGroup("child")
        child.createVariable("long", "i4", ("time", "x"))[:5] = np.arange(10).reshape(5, 2)
        grandchild = child.createGroup("grandchild")
        grandchild.createVariable("longest", "i4", ("time",))[:7] = np.arange(7)
        shadow = ds.createGroup("shadow")
        shadow.createDimension("time", None)
        shadow.createVariable("own", "i4", ("time",))[:3] = [31, 32, 33]
    # Confirm the reference library's logical shape and padding, independent of
    # each dataset's physical HDF5 extent.
    with netCDF4.Dataset(path) as ds:
        ds.set_auto_mask(False)
        assert len(ds.dimensions["time"]) == 7
        assert ds["short"].shape == (7, 2)
        assert ds["short"][:].tolist() == [[1, 2]] + [[-999, -999]] * 6
        assert ds["grid"][:2, :1].tolist() == [[5], [6]]
        assert len(ds["shadow"].dimensions["time"]) == 3
    # Put stale data in a physical edge chunk, then shrink its dataset. A
    # logical padding view must hide it even if a reader's chunk cache is warm.
    with h5py.File(path, "a") as f:
        f["short"].resize((3, 2))
        f["short"][1:3] = [[111, 112], [121, 122]]
        f["short"].resize((1, 2))

    generate_attribute_fixture(nc4_dir / "nc4_user_attribute_values.nc")


def generate_attribute_fixture(path):
    # Use the same netcdf-c library as the Python extension: a second library
    # instance would have its own file IDs. ctypes covers types the Python
    # attribute API cannot write, while still creating conformant named types.
    library = ctypes.CDLL(netCDF4._netCDF4.__file__)
    integer = ctypes.c_int
    size = ctypes.c_size_t
    pointer = ctypes.c_void_p
    name = ctypes.c_char_p
    signatures = {
        "nc_def_opaque": [integer, size, name, ctypes.POINTER(integer)],
        "nc_def_vlen": [integer, name, integer, ctypes.POINTER(integer)],
        "nc_def_compound": [integer, size, name, ctypes.POINTER(integer)],
        "nc_insert_compound": [integer, integer, name, size, integer],
        "nc_put_att": [integer, integer, name, integer, size, pointer],
        "nc_get_att": [integer, integer, name, pointer],
        "nc_free_vlens": [size, pointer],
        "nc_inq_typeid": [integer, name, ctypes.POINTER(integer)],
    }
    for function, arguments in signatures.items():
        getattr(library, function).argtypes = arguments
        getattr(library, function).restype = integer

    def call(function, *arguments):
        status = getattr(library, function)(*arguments)
        if status:
            raise RuntimeError(f"{function} failed with netcdf-c status {status}")

    class Sequence(ctypes.Structure):
        _fields_ = [("length", size), ("data", pointer)]

    class Nested(ctypes.Structure):
        _fields_ = [("tag", ctypes.c_short), ("samples", Sequence)]

    with netCDF4.Dataset(path, "w", format="NETCDF4") as ds:
        ds.createDimension("x", 1)
        variable = ds.createVariable("data", "i4", ("x",))
        variable[:] = [42]
        child = ds.createGroup("child")
        compound = ds.createCompoundType(np.dtype([
            ("a", "i4"), ("b", "f8"), ("weights", "f4", (2,))
        ], align=True), "record_t")
        records = np.array([(3, 2.5, [1.0, 2.0]), (-4, 9.0, [3.0, 4.0])], dtype=compound.dtype)
        opaque = integer()
        sequences = integer()
        shorts = integer()
        nested = integer()
        call("nc_def_opaque", ds._grpid, 4, b"opaque_t", ctypes.byref(opaque))
        call("nc_def_vlen", ds._grpid, b"sequence_t", 4, ctypes.byref(sequences))  # NC_INT
        call("nc_def_vlen", ds._grpid, b"short_sequence_t", 3, ctypes.byref(shorts))  # NC_SHORT
        call("nc_def_compound", ds._grpid, ctypes.sizeof(Nested), b"nested_t", ctypes.byref(nested))
        call("nc_insert_compound", ds._grpid, nested.value, b"tag", Nested.tag.offset, 3)
        call("nc_insert_compound", ds._grpid, nested.value, b"samples", Nested.samples.offset, shorts.value)
        payloads = [(integer * 2)(1, -2), (integer * 0)(), (integer * 1)(9)]
        values = (Sequence * 3)(*(Sequence(len(v), ctypes.cast(v, pointer)) for v in payloads))
        samples = (ctypes.c_short * 2)(10, 20)
        record = Nested(5, Sequence(2, ctypes.cast(samples, pointer)))
        opaque_value = (ctypes.c_ubyte * 4)(1, 0x23, 0x45, 0x67)
        for obj, group, variable_id in [(ds, ds._grpid, -1), (variable, ds._grpid, variable._varid), (child, child._grpid, -1)]:
            obj.setncattr("compound", records)
            call("nc_put_att", group, variable_id, b"opaque", opaque.value, 1, opaque_value)
            call("nc_put_att", group, variable_id, b"sequences", sequences.value, 3, values)
            call("nc_put_att", group, variable_id, b"nested", nested.value, 1, ctypes.byref(record))
    with netCDF4.Dataset(path) as ds:
        assert set(ds.ncattrs()) == {"compound", "opaque", "sequences", "nested"}
        np.testing.assert_array_equal(ds.getncattr("compound"), records)
        decoded = (Sequence * 3)()
        call("nc_get_att", ds._grpid, -1, b"sequences", decoded)
        try:
            actual = [list(ctypes.cast(v.data, ctypes.POINTER(integer))[:v.length]) for v in decoded]
            assert actual == [[1, -2], [], [9]]
        finally:
            call("nc_free_vlens", 3, decoded)


if __name__ == "__main__":
    generate(Path(__file__).resolve().parent / "netcdf4")
