#![cfg(feature = "netcdf4")]

use netcdf_reader::{
    NcAttrValue, NcFile, NcOpenOptions, NcSliceInfo, NcSliceInfoElem, NcType, NcValue,
};

fn fixture(name: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../testdata/netcdf4")
        .join(name)
}

fn open() -> NcFile {
    NcFile::open(fixture("nc4_descendant_unlimited.nc")).unwrap()
}

#[test]
fn descendant_extents_grow_ancestor_dimensions_and_preserve_shadowing() {
    let file = open();
    // Exercise local metadata before building the full tree.
    assert_eq!(file.dimensions().unwrap()[0].size, 7);
    assert_eq!(file.variable("short").unwrap().shape(), [7, 2]);
    assert_eq!(file.variable("child/long").unwrap().shape(), [7, 2]);
    assert_eq!(
        file.variable("child/grandchild/longest").unwrap().shape(),
        [7]
    );
    assert_eq!(file.variable("shadow/own").unwrap().shape(), [3]);
    assert_eq!(file.dimension("time").unwrap().size, 7);
    assert_eq!(file.dimension("child/time").unwrap().size, 7);
    assert_eq!(file.dimension("shadow/time").unwrap().size, 3);
    let hdf5 = hdf5_reader::Hdf5File::open(fixture("nc4_descendant_unlimited.nc")).unwrap();
    let child = netcdf_reader::nc4::groups::build_group_at_path(
        &hdf5,
        "child",
        netcdf_reader::NcMetadataMode::Strict,
        false,
    )
    .unwrap();
    assert_eq!(child.dimension("time").unwrap().size, 7);
}

#[test]
fn full_reads_and_buffers_match_netcdf_c_padding() {
    let file = open();
    let reference = netcdf::open(fixture("nc4_descendant_unlimited.nc")).unwrap();
    let expected = reference
        .variable("short")
        .unwrap()
        .get_values::<i16, _>(..)
        .unwrap();
    assert_eq!(expected, [vec![1, 2], vec![-999; 12]].concat());
    let array = file.read_variable::<i16>("short").unwrap();
    assert_eq!(array.shape(), [7, 2]);
    assert_eq!(array.as_slice().unwrap(), expected);
    let mut values = vec![0i16; 14];
    file.read_variable_into("short", &mut values).unwrap();
    assert_eq!(values, expected);
    let raw: Vec<_> = expected.iter().flat_map(|v| v.to_le_bytes()).collect();
    let native: Vec<_> = expected.iter().flat_map(|v| v.to_ne_bytes()).collect();
    assert_eq!(file.read_variable_raw_bytes("short").unwrap(), raw);
    assert_eq!(file.read_variable_native_bytes("short").unwrap(), native);
    let mut bytes = vec![0; raw.len()];
    file.read_variable_raw_bytes_into("short", &mut bytes)
        .unwrap();
    assert_eq!(bytes, raw);
    file.read_variable_native_bytes_into("short", &mut bytes)
        .unwrap();
    assert_eq!(bytes, native);
    let expected = reference
        .variable("empty")
        .unwrap()
        .get_values::<f32, _>(..)
        .unwrap();
    assert_eq!(
        file.read_variable::<f32>("empty")
            .unwrap()
            .as_slice()
            .unwrap(),
        expected
    );
    assert!(file
        .read_variable_masked("short")
        .unwrap()
        .iter()
        .skip(2)
        .all(|v| v.is_nan()));
}

#[test]
fn padded_slices_include_indices_and_strides() {
    let file = open();
    let selection = NcSliceInfo {
        selections: vec![
            NcSliceInfoElem::Slice {
                start: 0,
                end: 7,
                step: 2,
            },
            NcSliceInfoElem::Index(1),
        ],
    };
    let values = file
        .read_variable_slice::<i16>("short", &selection)
        .unwrap();
    assert_eq!(values.shape(), [4]);
    assert_eq!(values.as_slice().unwrap(), [2, -999, -999, -999]);
    let records: Vec<_> = file
        .iter_slices::<i16>("short", 0)
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(records.len(), 7);
    assert_eq!(records[6].as_slice().unwrap(), [-999, -999]);
    #[cfg(feature = "rayon")]
    {
        assert_eq!(
            file.read_variable_slice_parallel::<i16>("short", &selection)
                .unwrap(),
            values
        );
        assert_eq!(
            file.read_variable_parallel::<i16>("short").unwrap(),
            file.read_variable::<i16>("short").unwrap()
        );
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        assert_eq!(
            file.read_variable_in_pool::<i16>("short", &pool).unwrap(),
            file.read_variable::<i16>("short").unwrap()
        );
    }
}

#[test]
fn two_unlimited_axes_pad_each_axis_in_row_major_order() {
    let file = open();
    let grid = file.read_variable::<i32>("grid").unwrap();
    assert_eq!(grid.shape(), [7, 4]);
    let mut expected = vec![-77; 28];
    expected[0] = 5;
    expected[4] = 6;
    assert_eq!(grid.as_slice().unwrap(), expected);
    #[cfg(feature = "rayon")]
    assert_eq!(file.read_variable_parallel::<i32>("grid").unwrap(), grid);
}

#[test]
fn strings_and_sequences_pad_with_empty_values() {
    let file = open();
    assert_eq!(
        file.read_variable_as_strings("labels").unwrap(),
        ["a", "b", "", "", "", "", ""]
    );
    let sequences = file.read_variable_user_defined("sequences").unwrap();
    assert_eq!(sequences.shape(), [7]);
    assert_eq!(
        sequences[0],
        NcValue::VLen(vec![NcValue::Int(7), NcValue::Int(8)])
    );
    assert!(sequences
        .iter()
        .skip(1)
        .all(|v| *v == NcValue::VLen(vec![])));
}

#[test]
fn padded_reads_check_the_logical_selection_budget() {
    let file = NcFile::open_with_options(
        fixture("nc4_descendant_unlimited.nc"),
        NcOpenOptions {
            max_read_bytes: 8,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(file.read_variable::<i16>("short").is_err());
    assert!(file.read_variable_raw_bytes("short").is_err());
    let selection = NcSliceInfo {
        selections: vec![
            NcSliceInfoElem::Index(6),
            NcSliceInfoElem::Slice {
                start: 0,
                end: 2,
                step: 1,
            },
        ],
    };
    assert_eq!(
        file.read_variable_slice::<i16>("short", &selection)
            .unwrap()
            .as_slice()
            .unwrap(),
        [-999, -999]
    );
    let mut dst = vec![0i16; 14];
    file.read_variable_into("short", &mut dst).unwrap();
    assert_eq!(&dst[2..], [-999; 12]);
}

#[test]
fn compound_opaque_and_vlen_attributes_are_decoded_on_groups_and_variables() {
    let file = NcFile::open(fixture("nc4_user_attribute_values.nc")).unwrap();
    for attributes in [
        file.global_attributes().unwrap(),
        file.variable("data").unwrap().attributes(),
        &file.group("child").unwrap().attributes,
    ] {
        let value = |name: &str| {
            &attributes
                .iter()
                .find(|attr| attr.name == name)
                .unwrap()
                .value
        };
        let NcAttrValue::UserDefined { dtype, values } = value("opaque") else {
            panic!("opaque attribute omitted");
        };
        assert!(matches!(dtype, NcType::Opaque { size: 4, .. }));
        assert_eq!(values, &[NcValue::Opaque(vec![1, 0x23, 0x45, 0x67])]);
        let NcAttrValue::UserDefined { values, .. } = value("compound") else {
            panic!("compound attribute omitted");
        };
        let NcValue::Compound(fields) = &values[0] else {
            panic!("wrong compound value");
        };
        assert_eq!(fields[0].value, NcValue::Int(3));
        assert_eq!(fields[1].value, NcValue::Double(2.5));
        let NcValue::Array(weights) = &fields[2].value else {
            panic!("array field lost");
        };
        assert_eq!(weights.values, [NcValue::Float(1.0), NcValue::Float(2.0)]);
        let NcAttrValue::UserDefined { values, .. } = value("sequences") else {
            panic!("vlen attribute omitted");
        };
        assert_eq!(
            values,
            &[
                NcValue::VLen(vec![NcValue::Int(1), NcValue::Int(-2)]),
                NcValue::VLen(vec![]),
                NcValue::VLen(vec![NcValue::Int(9)])
            ]
        );
        let NcAttrValue::UserDefined { values, .. } = value("nested") else {
            panic!("nested attribute omitted");
        };
        let NcValue::Compound(fields) = &values[0] else {
            panic!("wrong nested value");
        };
        assert_eq!(
            fields[1].value,
            NcValue::VLen(vec![NcValue::Short(10), NcValue::Short(20)])
        );
    }
}
