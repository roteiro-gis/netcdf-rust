//! External validation of netcdf-writer output against the reference C
//! libraries (netcdf-c via netCDF4-python, libhdf5 via h5py).
//!
//! These tests are `#[ignore]`d because they need a Python interpreter with
//! `netCDF4`, `h5py`, and `numpy` installed. Run them via
//! `scripts/validate-writer-output.sh`, or directly with:
//!
//! ```sh
//! NETCDF_RUST_EXTERNAL_VALIDATION=1 \
//! NETCDF_RUST_VALIDATOR_PYTHON=.venv312/bin/python \
//!   cargo test -p netcdf-writer --test external_validation -- --ignored
//! ```
//!
//! Each test writes a matrix of files plus a `manifest.json` into a temp dir
//! and shells out to `testdata/validate_writer_files.py`, which re-reads every
//! file with the reference libraries and diffs it against the manifest.

use std::path::{Path, PathBuf};
use std::process::Command;

use netcdf_writer::{NcAttrValue, NcFileBuilder, NcWriteFormat, NcWriteOptions};
use serde_json::{json, Value};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root")
        .to_path_buf()
}

fn validator_python() -> String {
    std::env::var("NETCDF_RUST_VALIDATOR_PYTHON").unwrap_or_else(|_| "python3".to_string())
}

fn require_validation_env() {
    assert_eq!(
        std::env::var("NETCDF_RUST_EXTERNAL_VALIDATION").as_deref(),
        Ok("1"),
        "external validation tests need NETCDF_RUST_EXTERNAL_VALIDATION=1 and a Python \
         interpreter with netCDF4/h5py; run scripts/validate-writer-output.sh"
    );
}

fn write_file(dir: &Path, name: &str, builder: &NcFileBuilder, options: NcWriteOptions) {
    let (_, bytes) = builder.to_vec(options).expect("write should succeed");
    std::fs::write(dir.join(name), bytes).expect("write file");
}

fn run_validator(dir: &Path, manifest: Value) {
    std::fs::write(
        dir.join("manifest.json"),
        serde_json::to_string_pretty(&manifest).unwrap(),
    )
    .expect("write manifest");
    if let Ok(export) = std::env::var("NETCDF_RUST_VALIDATION_EXPORT") {
        let export = PathBuf::from(export);
        std::fs::create_dir_all(&export).expect("create export dir");
        for entry in std::fs::read_dir(dir).expect("read validation dir") {
            let path = entry.expect("dir entry").path();
            if path.is_file() {
                std::fs::copy(&path, export.join(path.file_name().unwrap())).expect("export file");
            }
        }
    }
    let script = repo_root().join("testdata/validate_writer_files.py");
    let output = Command::new(validator_python())
        .arg(&script)
        .arg(dir)
        .output()
        .expect("failed to launch validator python; set NETCDF_RUST_VALIDATOR_PYTHON");
    assert!(
        output.status.success(),
        "reference-library validation failed:\n--- stdout ---\n{}\n--- stderr ---\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

fn dim(name: &str, size: u64, unlimited: bool) -> Value {
    json!({"name": name, "size": size, "unlimited": unlimited})
}

/// CDF-1 with fixed variables of odd/even sizes, a char variable, and every
/// classic attribute type.
fn cdf1_fixed(dir: &Path) -> Value {
    let mut b = NcFileBuilder::new();
    let y = b.add_dimension("y", 2).unwrap();
    let x = b.add_dimension("x", 3).unwrap();
    let station = b.add_dimension("station", 2).unwrap();
    let strlen = b.add_dimension("strlen", 4).unwrap();

    b.add_attribute("title", NcAttrValue::Chars("external validation".into()))
        .unwrap();
    b.add_attribute("gb", NcAttrValue::Bytes(vec![-1, 2]))
        .unwrap();
    b.add_attribute("gs", NcAttrValue::Shorts(vec![-3, 4]))
        .unwrap();
    b.add_attribute("gi", NcAttrValue::Ints(vec![-5, 6]))
        .unwrap();
    b.add_attribute("gf", NcAttrValue::Floats(vec![1.5, -2.5]))
        .unwrap();
    b.add_attribute("gd", NcAttrValue::Doubles(vec![3.25, -4.75]))
        .unwrap();

    let temp = b.add_variable::<f32>("temp", &[y, x]).unwrap();
    b.add_variable_attribute(temp, "units", NcAttrValue::Chars("K".into()))
        .unwrap();
    b.write_variable(temp, &[280.0_f32, 281.0, 282.0, 283.0, 284.0, 285.0])
        .unwrap();

    // 3 bytes: odd-sized fixed variable exercises tail padding + next begin.
    let mask = b.add_variable::<i8>("mask", &[x]).unwrap();
    b.write_variable(mask, &[1_i8, 0, 1]).unwrap();

    // 6 bytes: even but not a multiple of 4.
    let counts = b.add_variable::<i16>("counts", &[x]).unwrap();
    b.write_variable(counts, &[10_i16, 20, 30]).unwrap();

    let names = b
        .add_char_variable("station_name", &[station, strlen])
        .unwrap();
    b.write_char_variable_strings(names, &["abcd", "ef"])
        .unwrap();

    write_file(dir, "cdf1_fixed.nc", &b, NcWriteOptions::classic());
    json!({
        "path": "cdf1_fixed.nc",
        "kind": "netcdf",
        "data_model": "NETCDF3_CLASSIC",
        "dimensions": [dim("y", 2, false), dim("x", 3, false), dim("station", 2, false), dim("strlen", 4, false)],
        "attributes": {
            "title": "external validation",
            "gb": [-1, 2],
            "gs": [-3, 4],
            "gi": [-5, 6],
            "gf": [1.5, -2.5],
            "gd": [3.25, -4.75]
        },
        "variables": [
            {
                "name": "temp", "dtype": "f4", "shape": [2, 3], "dimensions": ["y", "x"],
                "values": [280.0, 281.0, 282.0, 283.0, 284.0, 285.0],
                "attributes": {"units": "K"}
            },
            {"name": "mask", "dtype": "i1", "shape": [3], "dimensions": ["x"], "values": [1, 0, 1]},
            {"name": "counts", "dtype": "i2", "shape": [3], "dimensions": ["x"], "values": [10, 20, 30]},
            {
                "name": "station_name", "dtype": "S1", "shape": [2, 4],
                "dimensions": ["station", "strlen"], "char_strings": ["abcd", "ef"]
            }
        ]
    })
}

/// The lone odd-sized record variable: the netCDF classic spec omits
/// inter-record padding when there is exactly one record variable, so each
/// 6-byte record is packed back-to-back.
fn cdf1_single_record(dir: &Path) -> Value {
    let mut b = NcFileBuilder::new();
    let time = b.add_unlimited_dimension("time").unwrap();
    let x = b.add_dimension("x", 3).unwrap();
    let s = b.add_variable::<i16>("s", &[time, x]).unwrap();
    b.write_variable(s, &[1_i16, 2, 3, 4, 5, 6, 7, 8, 9])
        .unwrap();

    write_file(dir, "cdf1_single_record.nc", &b, NcWriteOptions::classic());
    json!({
        "path": "cdf1_single_record.nc",
        "kind": "netcdf",
        "data_model": "NETCDF3_CLASSIC",
        "dimensions": [dim("time", 3, true), dim("x", 3, false)],
        "variables": [{
            "name": "s", "dtype": "i2", "shape": [3, 3], "dimensions": ["time", "x"],
            "values": [1, 2, 3, 4, 5, 6, 7, 8, 9]
        }]
    })
}

/// Multiple record variables, including odd-sized ones: every record slab is
/// padded to 4 bytes and vsize reflects the padded size.
fn cdf1_multi_record(dir: &Path) -> Value {
    let mut b = NcFileBuilder::new();
    let time = b.add_unlimited_dimension("time").unwrap();
    let bv = b.add_variable::<i8>("b", &[time]).unwrap();
    let sv = b.add_variable::<i16>("s", &[time]).unwrap();
    let iv = b.add_variable::<i32>("i", &[time]).unwrap();
    b.write_variable(bv, &[1_i8, 2]).unwrap();
    b.write_variable(sv, &[10_i16, 20]).unwrap();
    b.write_variable(iv, &[100_i32, 200]).unwrap();

    write_file(dir, "cdf1_multi_record.nc", &b, NcWriteOptions::classic());
    json!({
        "path": "cdf1_multi_record.nc",
        "kind": "netcdf",
        "data_model": "NETCDF3_CLASSIC",
        "dimensions": [dim("time", 2, true)],
        "variables": [
            {"name": "b", "dtype": "i1", "shape": [2], "dimensions": ["time"], "values": [1, 2]},
            {"name": "s", "dtype": "i2", "shape": [2], "dimensions": ["time"], "values": [10, 20]},
            {"name": "i", "dtype": "i4", "shape": [2], "dimensions": ["time"], "values": [100, 200]}
        ]
    })
}

fn cdf2_offset64(dir: &Path) -> Value {
    let mut b = NcFileBuilder::new();
    let y = b.add_dimension("y", 2).unwrap();
    let x = b.add_dimension("x", 2).unwrap();
    let v = b.add_variable::<f64>("pressure", &[y, x]).unwrap();
    b.add_variable_attribute(v, "units", NcAttrValue::Chars("hPa".into()))
        .unwrap();
    b.write_variable(v, &[1013.25_f64, 1000.0, 985.5, 970.25])
        .unwrap();

    write_file(dir, "cdf2_offset64.nc", &b, NcWriteOptions::offset64());
    json!({
        "path": "cdf2_offset64.nc",
        "kind": "netcdf",
        "data_model": "NETCDF3_64BIT_OFFSET",
        "dimensions": [dim("y", 2, false), dim("x", 2, false)],
        "variables": [{
            "name": "pressure", "dtype": "f8", "shape": [2, 2], "dimensions": ["y", "x"],
            "values": [1013.25, 1000.0, 985.5, 970.25],
            "attributes": {"units": "hPa"}
        }]
    })
}

/// CDF-5 exercises the extended (unsigned + 64-bit) type set in both
/// variables and attributes.
fn cdf5_types(dir: &Path) -> Value {
    let mut b = NcFileBuilder::new();
    let x = b.add_dimension("x", 3).unwrap();
    b.add_attribute("gub", NcAttrValue::UBytes(vec![250, 251]))
        .unwrap();
    b.add_attribute("gus", NcAttrValue::UShorts(vec![65000]))
        .unwrap();
    b.add_attribute("gui", NcAttrValue::UInts(vec![4000000000]))
        .unwrap();
    b.add_attribute("gll", NcAttrValue::Int64s(vec![-9000000000]))
        .unwrap();
    b.add_attribute("gull", NcAttrValue::UInt64s(vec![18000000000000000000]))
        .unwrap();

    let ub = b.add_variable::<u8>("ub", &[x]).unwrap();
    let us = b.add_variable::<u16>("us", &[x]).unwrap();
    let ui = b.add_variable::<u32>("ui", &[x]).unwrap();
    let ll = b.add_variable::<i64>("ll", &[x]).unwrap();
    let ull = b.add_variable::<u64>("ull", &[x]).unwrap();
    b.write_variable(ub, &[1_u8, 2, 255]).unwrap();
    b.write_variable(us, &[1_u16, 2, 65535]).unwrap();
    b.write_variable(ui, &[1_u32, 2, 4294967295]).unwrap();
    b.write_variable(ll, &[-1_i64, 0, 9007199254740993])
        .unwrap();
    b.write_variable(ull, &[1_u64, 2, 18446744073709551615])
        .unwrap();

    write_file(dir, "cdf5_types.nc", &b, NcWriteOptions::cdf5());
    json!({
        "path": "cdf5_types.nc",
        "kind": "netcdf",
        "data_model": "NETCDF3_64BIT_DATA",
        "dimensions": [dim("x", 3, false)],
        "attributes": {
            "gub": [250_u8, 251],
            "gus": [65000_u16],
            "gui": [4000000000_u32],
            "gll": [-9000000000_i64],
            "gull": [18000000000000000000_u64]
        },
        "variables": [
            {"name": "ub", "dtype": "u1", "shape": [3], "dimensions": ["x"], "values": [1, 2, 255]},
            {"name": "us", "dtype": "u2", "shape": [3], "dimensions": ["x"], "values": [1, 2, 65535]},
            {"name": "ui", "dtype": "u4", "shape": [3], "dimensions": ["x"], "values": [1, 2, 4294967295_u32]},
            {"name": "ll", "dtype": "i8", "shape": [3], "dimensions": ["x"], "values": [-1, 0, 9007199254740993_i64]},
            {"name": "ull", "dtype": "u8", "shape": [3], "dimensions": ["x"], "values": [1, 2, 18446744073709551615_u64]}
        ]
    })
}

#[test]
#[ignore = "requires python with netCDF4/h5py; run scripts/validate-writer-output.sh"]
fn classic_formats_validate_against_netcdf_c() {
    require_validation_env();
    let dir = tempfile::tempdir().unwrap();
    let files = vec![
        cdf1_fixed(dir.path()),
        cdf1_single_record(dir.path()),
        cdf1_multi_record(dir.path()),
        cdf2_offset64(dir.path()),
        cdf5_types(dir.path()),
    ];
    run_validator(dir.path(), json!({"files": files}));
}

fn nc4_options() -> NcWriteOptions {
    NcWriteOptions {
        format: NcWriteFormat::Nc4,
    }
}

/// Groups, scalar variables, and global/variable attributes through the HDF5
/// bridge.
fn nc4_basic(dir: &Path) -> Value {
    let mut b = NcFileBuilder::new();
    let y = b.add_dimension("y", 2).unwrap();
    let x = b.add_dimension("x", 3).unwrap();
    let z = b.add_dimension_path("science/z", 2).unwrap();

    b.add_attribute("title", NcAttrValue::Chars("nc4 basic".into()))
        .unwrap();
    b.add_attribute("gi", NcAttrValue::Ints(vec![7])).unwrap();

    let temp = b.add_variable::<f64>("temp", &[y, x]).unwrap();
    b.add_variable_attribute(temp, "units", NcAttrValue::Chars("K".into()))
        .unwrap();
    b.write_variable(temp, &[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap();

    let depth = b.add_variable_path::<i32>("science/depth", &[z]).unwrap();
    b.write_variable(depth, &[100_i32, 200]).unwrap();

    let scalar = b.add_variable::<f32>("scale", &[]).unwrap();
    b.write_variable(scalar, &[2.5_f32]).unwrap();

    write_file(dir, "nc4_basic.nc", &b, nc4_options());
    json!({
        "path": "nc4_basic.nc",
        "kind": "netcdf",
        "data_model": "NETCDF4",
        "dimensions": [dim("y", 2, false), dim("x", 3, false)],
        "groups": ["science"],
        "attributes": {"title": "nc4 basic", "gi": [7]},
        "variables": [
            {
                "name": "temp", "dtype": "f8", "shape": [2, 3], "dimensions": ["y", "x"],
                "values": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "attributes": {"units": "K"}
            },
            {"name": "science/depth", "dtype": "i4", "shape": [2], "dimensions": ["z"], "values": [100, 200]},
            {"name": "scale", "dtype": "f4", "shape": [], "dimensions": [], "values": [2.5]}
        ]
    })
}

/// Chunked + deflate/shuffle/fletcher32 pipelines.
fn nc4_filters(dir: &Path) -> Value {
    let mut b = NcFileBuilder::new();
    let y = b.add_dimension("y", 4).unwrap();
    let x = b.add_dimension("x", 6).unwrap();
    let values: Vec<i32> = (0..24).collect();

    let deflated = b.add_variable::<i32>("deflated", &[y, x]).unwrap();
    b.set_variable_chunking(deflated, [2, 6]).unwrap();
    b.set_variable_deflate(deflated, Some(5), false).unwrap();
    b.write_variable(deflated, &values).unwrap();

    let shuffled = b.add_variable::<i32>("shuffled", &[y, x]).unwrap();
    b.set_variable_chunking(shuffled, [2, 6]).unwrap();
    b.set_variable_deflate(shuffled, Some(4), true).unwrap();
    b.write_variable(shuffled, &values).unwrap();

    let checked = b.add_variable::<i32>("checked", &[y, x]).unwrap();
    b.set_variable_chunking(checked, [4, 3]).unwrap();
    b.set_variable_fletcher32(checked, true).unwrap();
    b.write_variable(checked, &values).unwrap();

    write_file(dir, "nc4_filters.nc", &b, nc4_options());
    let expected: Vec<Value> = values.iter().map(|v| json!(v)).collect();
    json!({
        "path": "nc4_filters.nc",
        "kind": "netcdf",
        "data_model": "NETCDF4",
        "dimensions": [dim("y", 4, false), dim("x", 6, false)],
        "variables": [
            {"name": "deflated", "dtype": "i4", "shape": [4, 6], "values": expected},
            {"name": "shuffled", "dtype": "i4", "shape": [4, 6], "values": expected},
            {"name": "checked", "dtype": "i4", "shape": [4, 6], "values": expected}
        ]
    })
}

/// Unlimited dimension with appended records (exercises the chunk index the
/// writer emits for resizable datasets).
fn nc4_unlimited(dir: &Path) -> Value {
    let mut b = NcFileBuilder::new();
    let time = b.add_unlimited_dimension("time").unwrap();
    let x = b.add_dimension("x", 3).unwrap();
    let v = b.add_variable::<i32>("obs", &[time, x]).unwrap();
    b.write_variable(v, &[1_i32, 2, 3, 4, 5, 6]).unwrap();

    write_file(dir, "nc4_unlimited.nc", &b, nc4_options());
    json!({
        "path": "nc4_unlimited.nc",
        "kind": "netcdf",
        "data_model": "NETCDF4",
        "dimensions": [dim("time", 2, true), dim("x", 3, false)],
        "variables": [{
            "name": "obs", "dtype": "i4", "shape": [2, 3], "dimensions": ["time", "x"],
            "values": [1, 2, 3, 4, 5, 6]
        }]
    })
}

/// NC_STRING variables and fill values.
fn nc4_strings(dir: &Path) -> Value {
    let mut b = NcFileBuilder::new();
    let station = b.add_dimension("station", 3).unwrap();
    let names = b.add_string_variable("name", &[station]).unwrap();
    b.write_string_variable(names, &["alpha", "bravo", "charlie"])
        .unwrap();

    let filled = b.add_variable::<i16>("filled", &[station]).unwrap();
    b.set_variable_fill_value(filled, -999_i16).unwrap();
    b.write_variable(filled, &[5_i16, -999, 7]).unwrap();

    write_file(dir, "nc4_strings.nc", &b, nc4_options());
    json!({
        "path": "nc4_strings.nc",
        "kind": "netcdf",
        "data_model": "NETCDF4",
        "dimensions": [dim("station", 3, false)],
        "variables": [
            {"name": "name", "dtype": "str", "shape": [3], "strings": ["alpha", "bravo", "charlie"]},
            {
                "name": "filled", "dtype": "i2", "shape": [3], "values": [5, -999, 7],
                "attributes": {"_FillValue": -999}
            }
        ]
    })
}

fn nc4_classic_model(dir: &Path) -> Value {
    let mut b = NcFileBuilder::new();
    let x = b.add_dimension("x", 4).unwrap();
    let v = b.add_variable::<f32>("wind", &[x]).unwrap();
    b.write_variable(v, &[1.0_f32, 2.0, 3.0, 4.0]).unwrap();

    write_file(
        dir,
        "nc4_classic.nc",
        &b,
        NcWriteOptions {
            format: NcWriteFormat::Nc4Classic,
        },
    );
    json!({
        "path": "nc4_classic.nc",
        "kind": "netcdf",
        "data_model": "NETCDF4_CLASSIC",
        "dimensions": [dim("x", 4, false)],
        "variables": [{"name": "wind", "dtype": "f4", "shape": [4], "values": [1.0, 2.0, 3.0, 4.0]}]
    })
}

/// Coordinate variables become dimension scales that must be attached to the
/// data variables referencing them (h5py h5ds checks REFERENCE_LIST wiring).
fn nc4_scales(dir: &Path) -> Value {
    let mut b = NcFileBuilder::new();
    let y = b.add_dimension("y", 2).unwrap();
    let x = b.add_dimension("x", 3).unwrap();
    let yv = b.add_variable::<f32>("y", &[y]).unwrap();
    b.write_variable(yv, &[10.0_f32, 20.0]).unwrap();
    let xv = b.add_variable::<f32>("x", &[x]).unwrap();
    b.write_variable(xv, &[1.0_f32, 2.0, 3.0]).unwrap();
    let temp = b.add_variable::<f64>("temp", &[y, x]).unwrap();
    b.write_variable(temp, &[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0])
        .unwrap();

    write_file(dir, "nc4_scales.nc", &b, nc4_options());
    json!({
        "path": "nc4_scales.nc",
        "kind": "netcdf",
        "data_model": "NETCDF4",
        "dimensions": [dim("y", 2, false), dim("x", 3, false)],
        "variables": [
            {"name": "y", "dtype": "f4", "shape": [2], "values": [10.0, 20.0]},
            {"name": "x", "dtype": "f4", "shape": [3], "values": [1.0, 2.0, 3.0]},
            {"name": "temp", "dtype": "f8", "shape": [2, 3], "values": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}
        ],
        "scales": [
            {"variable": "temp", "dim_index": 0, "scale": "y"},
            {"variable": "temp", "dim_index": 1, "scale": "x"}
        ]
    })
}

#[test]
#[ignore = "requires python with netCDF4/h5py; run scripts/validate-writer-output.sh"]
fn nc4_files_validate_against_netcdf_c() {
    require_validation_env();
    let dir = tempfile::tempdir().unwrap();
    let files = vec![
        nc4_basic(dir.path()),
        nc4_filters(dir.path()),
        nc4_unlimited(dir.path()),
        nc4_strings(dir.path()),
        nc4_classic_model(dir.path()),
        nc4_scales(dir.path()),
    ];
    run_validator(dir.path(), json!({"files": files}));
}
