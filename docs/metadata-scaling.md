# Large-group metadata scaling

Measured on macOS arm64 with Rust 1.92.0, using release builds and the median
of seven repetitions on the same files. Each file contains one fixed dimension
of length two and the given number of unwritten `i16` variables in its root
group. The baseline includes the nested v2 B-tree fix in #129.

| Variables | Before: open + metadata | After: open + metadata |
| ---: | ---: | ---: |
| 1,600 | 13.5 ms | 13.5 ms |
| 3,200 | 33.0 ms | 27.8 ms |
| 6,720 | 107.1 ms | 58.6 ms |
| 20,000 | 945.7 ms | 182.6 ms |

`NcFile::open` alone took 0.07–0.15 ms. The expensive operation was loading
variable metadata, which lists the HDF5 group's children. Dense-link merging
previously scanned all preceding children for every link, giving quadratic
duplicate checks. A hash set preserves the same name, address, and file-context
identity checks while avoiding that repeated scan. Listing the 20,000-variable
HDF5 group fell from 994 ms to 120 ms.

The final NetCDF reader also scans dimension extents across groups so shorter
variables can be padded correctly. With all compatibility fixes included,
increasing the variable count by 2.98 times (6,720 to 20,000) increased metadata
time by 3.12 times, compared with 8.83 times before. Absolute timings depend on
the machine and file layout.

To reproduce, install `netCDF4` in a Python environment and generate the files:

```python
from pathlib import Path
import netCDF4

directory = Path("/tmp/netcdf-metadata-scaling")
directory.mkdir(exist_ok=True)
for count in [1600, 3200, 6720, 20000]:
    with netCDF4.Dataset(directory / f"n{count}.nc", "w") as file:
        file.createDimension("x", 2)
        for index in range(count):
            file.createVariable(f"v{index}", "i2", ("x",))
```

Then run:

```sh
cargo run -p netcdf-reader --release --example metadata_scaling -- \
  /tmp/netcdf-metadata-scaling/n{1600,3200,6720,20000}.nc
```

The example reports median HDF5 listing time, NetCDF open time, and NetCDF open
plus variable-metadata time separately. It warms each file before sampling.
