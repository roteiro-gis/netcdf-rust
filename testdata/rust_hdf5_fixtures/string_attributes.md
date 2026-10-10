# `netcdf4/rust_hdf5_string_attributes.nc`

A NetCDF-4 file whose text attributes were added by **rust-hdf5**, which records the
base type of a variable-length string as a one-byte *fixed string*. libhdf5 (and so
h5py / netCDF4-python) records a one-byte *integer* there. netcdf-c accepts both: it
classifies `NC_STRING` from `H5Tget_class == H5T_STRING` and `H5Tis_variable_str`
only (`libhdf5/hdf5open.c`), never from the base type.

Contents (verified with netCDF4-python 1.7.2 / netcdf-c 4.9.3):

- global `title = "string attribute fixture"` (netCDF4-python, NC_CHAR)
- global `history = "edited by rust-hdf5"` (rust-hdf5, variable-length string)
- `temp(y=2, x=3)` float, `units = "K"` (NC_CHAR), `comment = "written by rust-hdf5"`
  (rust-hdf5, variable-length string)

How it was made — base file with netCDF4-python:

```python
import netCDF4
ds = netCDF4.Dataset("rust_hdf5_string_attributes.nc", "w", format="NETCDF4")
ds.createDimension("y", 2); ds.createDimension("x", 3)
ds.title = "string attribute fixture"
t = ds.createVariable("temp", "f4", ("y", "x")); t.units = "K"; t[:] = 280.0
ds.close()
```

then rust-hdf5 0.7.2 (a binary depending on `rust-hdf5`, path given as argument):

```rust
use rust_hdf5::types::VarLenUnicode;
let file = rust_hdf5::H5File::open_rw(&path)?;
file.set_attr_string("history", "edited by rust-hdf5")?;
let ds = file.dataset_writer("temp")?;
ds.new_attr::<VarLenUnicode>().shape(()).create("comment")?.write_string("written by rust-hdf5")?;
file.close()?;
```
