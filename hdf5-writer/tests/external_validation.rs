//! External validation of hdf5-writer output against libhdf5 (via h5py).
//!
//! These tests are `#[ignore]`d because they need a Python interpreter with
//! `h5py` and `numpy` installed. Run them via
//! `scripts/validate-writer-output.sh`, or directly with:
//!
//! ```sh
//! NETCDF_RUST_EXTERNAL_VALIDATION=1 \
//! NETCDF_RUST_VALIDATOR_PYTHON=.venv312/bin/python \
//!   cargo test -p hdf5-writer --test external_validation -- --ignored
//! ```

use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::process::Command;

use hdf5_writer::{
    AttributeBuilder, ByteOrder, DatasetBuilder, Datatype, FilterDescription, Hdf5Builder,
    Hdf5Writer, WriteOptions, FILTER_DEFLATE, FILTER_FLETCHER32, FILTER_SHUFFLE, UNLIMITED,
};
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
         interpreter with h5py; run scripts/validate-writer-output.sh"
    );
}

fn write_file(dir: &Path, name: &str, builder: Hdf5Builder) {
    let plan = builder.into_plan().expect("plan should validate");
    let cursor = Hdf5Writer::new(Cursor::new(Vec::new()), WriteOptions::default())
        .finish(plan)
        .expect("write should succeed");
    std::fs::write(dir.join(name), cursor.into_inner()).expect("write file");
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

fn deflate(level: u32) -> FilterDescription {
    FilterDescription {
        id: FILTER_DEFLATE,
        name: None,
        client_data: vec![level],
    }
}

/// Contiguous/compact/scalar layouts, nested groups, and every attribute
/// flavor the writer supports.
fn layouts(dir: &Path) -> Value {
    let grid: Vec<i32> = (0..6).collect();
    let series = [1.5_f64, 2.5, 3.5, 4.5];
    let builder = Hdf5Builder::new()
        .dataset(
            DatasetBuilder::typed_data("outer/inner/grid", vec![2, 3], &grid)
                .unwrap()
                .attribute(AttributeBuilder::scalar("valid_max", 100_i32).unwrap()),
        )
        .dataset(
            DatasetBuilder::typed_data("series", vec![4], &series)
                .unwrap()
                .compact(),
        )
        .dataset(DatasetBuilder::typed_data("origin", Vec::<u64>::new(), &[42_i64]).unwrap())
        .attribute(AttributeBuilder::scalar("version", 3_i32).unwrap())
        .attribute(AttributeBuilder::vector("bounds", &[0.0_f64, 10.0]).unwrap())
        .attribute(AttributeBuilder::fixed_string("institution", "cairn"))
        .attribute(AttributeBuilder::vlen_strings("sources", &["a", "bc"]).unwrap())
        .group_attribute("outer", AttributeBuilder::fixed_string("kind", "container"));

    write_file(dir, "layouts.h5", builder);
    json!({
        "path": "layouts.h5",
        "kind": "hdf5",
        "groups": ["outer", "outer/inner"],
        "attributes": {
            "version": 3,
            "bounds": [0.0, 10.0],
            "institution": "cairn",
            "sources": ["a", "bc"]
        },
        "group_attributes": {"outer": {"kind": "container"}},
        "datasets": [
            {
                "name": "outer/inner/grid", "dtype": "i4", "shape": [2, 3],
                "layout": "contiguous", "values": [0, 1, 2, 3, 4, 5],
                "attributes": {"valid_max": 100}
            },
            {
                "name": "series", "dtype": "f8", "shape": [4],
                "layout": "compact", "values": [1.5, 2.5, 3.5, 4.5]
            },
            {"name": "origin", "dtype": "i8", "shape": [], "values": [42]}
        ]
    })
}

/// Chunked layouts with the deflate/shuffle/fletcher32 pipelines.
fn filters(dir: &Path) -> Value {
    let values: Vec<i32> = (0..24).collect();
    let small: Vec<i16> = (0..12).collect();
    let builder = Hdf5Builder::new()
        .dataset(
            DatasetBuilder::typed_data("deflated", vec![4, 6], &values)
                .unwrap()
                .chunked(vec![2, 3])
                .filter(deflate(6)),
        )
        .dataset(
            DatasetBuilder::typed_data("shuffled", vec![4, 6], &values)
                .unwrap()
                .chunked(vec![2, 6])
                .filter(FilterDescription {
                    id: FILTER_SHUFFLE,
                    name: None,
                    client_data: vec![4],
                })
                .filter(deflate(4)),
        )
        .dataset(
            DatasetBuilder::typed_data("checked", vec![3, 4], &small)
                .unwrap()
                .chunked(vec![3, 2])
                .filter(FilterDescription {
                    id: FILTER_FLETCHER32,
                    name: None,
                    client_data: Vec::new(),
                }),
        )
        .dataset(
            // Odd chunk byte counts exercise the fletcher32 trailing-byte rule.
            DatasetBuilder::typed_data("checked_odd", vec![5], &[1_i8, 2, 3, 4, 5])
                .unwrap()
                .chunked(vec![5])
                .filter(FilterDescription {
                    id: FILTER_FLETCHER32,
                    name: None,
                    client_data: Vec::new(),
                }),
        );

    write_file(dir, "filters.h5", builder);
    let expected: Vec<Value> = values.iter().map(|v| json!(v)).collect();
    json!({
        "path": "filters.h5",
        "kind": "hdf5",
        "datasets": [
            {
                "name": "deflated", "dtype": "i4", "shape": [4, 6], "layout": "chunked",
                "chunks": [2, 3], "filters": {"deflate": 6, "shuffle": false, "fletcher32": false},
                "values": expected
            },
            {
                "name": "shuffled", "dtype": "i4", "shape": [4, 6], "layout": "chunked",
                "chunks": [2, 6], "filters": {"deflate": 4, "shuffle": true, "fletcher32": false},
                "values": expected
            },
            {
                "name": "checked", "dtype": "i2", "shape": [3, 4], "layout": "chunked",
                "chunks": [3, 2], "filters": {"deflate": null, "shuffle": false, "fletcher32": true},
                "values": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
            },
            {
                "name": "checked_odd", "dtype": "i1", "shape": [5], "layout": "chunked",
                "chunks": [5], "filters": {"deflate": null, "shuffle": false, "fletcher32": true},
                "values": [1, 2, 3, 4, 5]
            }
        ]
    })
}

/// Fixed strings, vlen strings, and vlen integer sequences (global heap).
fn strings(dir: &Path) -> Value {
    let base = Datatype::FixedPoint {
        size: 4,
        signed: true,
        byte_order: ByteOrder::LittleEndian,
    };
    let sequences = vec![
        [1_i32.to_le_bytes(), 2_i32.to_le_bytes()].concat(),
        3_i32.to_le_bytes().to_vec(),
        Vec::new(),
        [
            10_i32.to_le_bytes(),
            11_i32.to_le_bytes(),
            12_i32.to_le_bytes(),
        ]
        .concat(),
    ];
    let builder = Hdf5Builder::new()
        .dataset(
            DatasetBuilder::fixed_string_data("fixed", vec![3], &["north", "south", "up"]).unwrap(),
        )
        .dataset(
            DatasetBuilder::vlen_string_data("vlen_str", vec![2, 2], &["a", "bb", "ccc", ""])
                .unwrap(),
        )
        .dataset(
            DatasetBuilder::vlen_sequence_data("ragged", base, vec![2, 2], sequences).unwrap(),
        );

    write_file(dir, "strings.h5", builder);
    json!({
        "path": "strings.h5",
        "kind": "hdf5",
        "datasets": [
            {"name": "fixed", "shape": [3], "strings": ["north", "south", "up"]},
            {"name": "vlen_str", "dtype": "str", "shape": [2, 2], "strings": ["a", "bb", "ccc", ""]},
            {"name": "ragged", "shape": [2, 2], "vlen": [[1, 2], [3], [], [10, 11, 12]]}
        ]
    })
}

/// Resizable (unlimited max dims) chunked datasets, unfiltered and filtered.
/// libhdf5 requires a real chunk index (not implicit) for these.
fn resizable(dir: &Path) -> Value {
    let values: Vec<i32> = (0..12).collect();
    let builder = Hdf5Builder::new()
        .dataset(
            DatasetBuilder::typed_data("growing", vec![2, 6], &values)
                .unwrap()
                .chunked(vec![1, 6])
                .max_shape(vec![UNLIMITED, 6]),
        )
        .dataset(
            DatasetBuilder::typed_data("growing_deflated", vec![2, 6], &values)
                .unwrap()
                .chunked(vec![1, 6])
                .max_shape(vec![UNLIMITED, 6])
                .filter(deflate(5)),
        );

    write_file(dir, "resizable.h5", builder);
    let expected: Vec<Value> = values.iter().map(|v| json!(v)).collect();
    json!({
        "path": "resizable.h5",
        "kind": "hdf5",
        "datasets": [
            {
                "name": "growing", "dtype": "i4", "shape": [2, 6], "maxshape": [null, 6],
                "layout": "chunked", "chunks": [1, 6], "values": expected
            },
            {
                "name": "growing_deflated", "dtype": "i4", "shape": [2, 6], "maxshape": [null, 6],
                "layout": "chunked", "chunks": [1, 6],
                "filters": {"deflate": 5, "shuffle": false, "fletcher32": false},
                "values": expected
            }
        ]
    })
}

/// A fill value on an unallocated dataset: readers must synthesize the fill.
fn fillvalue(dir: &Path) -> Value {
    let builder = Hdf5Builder::new().dataset(
        DatasetBuilder::typed::<i32>("filled", vec![2, 2])
            .fill_value((-7_i32).to_le_bytes().to_vec()),
    );

    write_file(dir, "fillvalue.h5", builder);
    json!({
        "path": "fillvalue.h5",
        "kind": "hdf5",
        "datasets": [{
            "name": "filled", "dtype": "i4", "shape": [2, 2],
            "fillvalue": -7, "values": [-7, -7, -7, -7]
        }]
    })
}

#[test]
#[ignore = "requires python with h5py; run scripts/validate-writer-output.sh"]
fn hdf5_files_validate_against_libhdf5() {
    require_validation_env();
    let dir = tempfile::tempdir().unwrap();
    let files = vec![
        layouts(dir.path()),
        filters(dir.path()),
        strings(dir.path()),
        resizable(dir.path()),
        fillvalue(dir.path()),
    ];
    run_validator(dir.path(), json!({"files": files}));
}
