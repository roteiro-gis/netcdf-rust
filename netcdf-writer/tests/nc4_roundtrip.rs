#![cfg(feature = "netcdf4")]

use netcdf_reader::{NcFile, NcFormat};
use netcdf_writer::{
    NcAttrValue, NcCompoundField, NcEnumMember, NcFileBuilder, NcIntegerValue, NcSliceInfo,
    NcSliceInfoElem, NcType, NcWriteFormat, NcWriteOptions, NC_FILL_INT,
};

#[test]
fn writes_nc4_coordinate_variable() {
    let mut builder = NcFileBuilder::new();
    let x = builder.add_dimension("x", 4).unwrap();
    builder
        .add_attribute("title", NcAttrValue::Chars("nc4 roundtrip".to_string()))
        .unwrap();
    let coordinate = builder.add_variable::<f32>("x", &[x]).unwrap();
    builder
        .add_variable_attribute(coordinate, "units", NcAttrValue::Chars("m".to_string()))
        .unwrap();
    builder
        .write_variable(coordinate, &[0.0_f32, 1.5, 3.0, 4.5])
        .unwrap();

    let (format, bytes) = builder
        .to_vec(NcWriteOptions {
            format: NcWriteFormat::Nc4,
        })
        .unwrap();
    assert_eq!(format, NcFormat::Nc4);

    let file = NcFile::from_bytes(&bytes).unwrap();
    assert_eq!(file.format(), NcFormat::Nc4);
    assert_eq!(file.dimension("x").unwrap().size, 4);
    assert_eq!(
        file.global_attribute("title")
            .unwrap()
            .value
            .as_string()
            .unwrap(),
        "nc4 roundtrip"
    );

    let variable = file.variable("x").unwrap();
    assert!(variable.is_coordinate_variable_for("x"));
    assert_eq!(
        variable
            .attribute("units")
            .unwrap()
            .value
            .as_string()
            .unwrap(),
        "m"
    );

    let values = file.read_variable::<f32>("x").unwrap();
    assert_eq!(values.shape(), &[4]);
    assert_eq!(
        values.as_slice_memory_order().unwrap(),
        &[0.0, 1.5, 3.0, 4.5]
    );
}

#[test]
fn writes_nc4_group_dimension_and_variable_paths() {
    let mut builder = NcFileBuilder::new();
    builder
        .add_group_attribute(
            "/science/",
            "title",
            NcAttrValue::Chars("science profile".to_string()),
        )
        .unwrap();
    builder
        .add_group_attribute(
            "metadata",
            "Conventions",
            NcAttrValue::Chars("CF-1.11".to_string()),
        )
        .unwrap();
    let z = builder.add_dimension_path("science/z", 3).unwrap();
    let temp = builder
        .add_variable_path::<f32>("science/temp", &[z])
        .unwrap();
    builder
        .write_variable(temp, &[280.0_f32, 281.0, 282.0])
        .unwrap();

    let x = builder.add_dimension_path("/science/x/", 3).unwrap();
    let coordinate = builder.add_variable_path::<f32>("science/x", &[x]).unwrap();
    builder
        .write_variable(coordinate, &[0.0_f32, 1.0, 2.0])
        .unwrap();

    let (_format, bytes) = builder
        .to_vec(NcWriteOptions {
            format: NcWriteFormat::Nc4,
        })
        .unwrap();

    let file = NcFile::from_bytes(&bytes).unwrap();
    let group = file.group("science").unwrap();
    assert_eq!(group.name, "science");
    assert_eq!(
        group.attribute("title").unwrap().value.as_string().unwrap(),
        "science profile"
    );
    assert_eq!(
        file.global_attribute("science/title")
            .unwrap()
            .value
            .as_string()
            .unwrap(),
        "science profile"
    );
    assert_eq!(
        file.group("metadata")
            .unwrap()
            .attribute("Conventions")
            .unwrap()
            .value
            .as_string()
            .unwrap(),
        "CF-1.11"
    );
    assert_eq!(group.dimension("z").unwrap().size, 3);
    assert_eq!(group.dimension("x").unwrap().size, 3);
    assert_eq!(file.dimension("science/z").unwrap().size, 3);
    assert_eq!(file.dimension("science/x").unwrap().size, 3);

    let temp_variable = file.variable("science/temp").unwrap();
    assert_eq!(temp_variable.shape(), vec![3]);
    assert!(!temp_variable.is_coordinate_variable());
    let temp_values = file.read_variable::<f32>("science/temp").unwrap();
    assert_eq!(
        temp_values.as_slice_memory_order().unwrap(),
        &[280.0, 281.0, 282.0]
    );

    let coordinate_variable = file.variable("science/x").unwrap();
    assert!(coordinate_variable.is_coordinate_variable_for("x"));
    let coordinate_values = file.read_variable::<f32>("science/x").unwrap();
    assert_eq!(
        coordinate_values.as_slice_memory_order().unwrap(),
        &[0.0, 1.0, 2.0]
    );
}

#[test]
fn writes_nc4_scalar_variable() {
    let mut builder = NcFileBuilder::new();
    let variable = builder.add_variable::<i32>("answer", &[]).unwrap();
    builder.write_variable(variable, &[42_i32]).unwrap();

    let (_format, bytes) = builder
        .to_vec(NcWriteOptions {
            format: NcWriteFormat::Nc4,
        })
        .unwrap();

    let file = NcFile::from_bytes(&bytes).unwrap();
    let variable = file.variable("answer").unwrap();
    assert_eq!(variable.shape(), Vec::<u64>::new());

    let values = file.read_variable::<i32>("answer").unwrap();
    assert_eq!(values.shape(), &[] as &[usize]);
    assert_eq!(values.as_slice_memory_order().unwrap(), &[42]);
}

#[test]
fn writes_nc4_fill_value_for_unwritten_variable() {
    let mut builder = NcFileBuilder::new();
    let y = builder.add_dimension("y", 2).unwrap();
    let x = builder.add_dimension("x", 3).unwrap();
    let temp = builder.add_variable::<i16>("temp", &[y, x]).unwrap();
    builder.set_variable_fill_value(temp, -999_i16).unwrap();

    let (_format, bytes) = builder
        .to_vec(NcWriteOptions {
            format: NcWriteFormat::Nc4,
        })
        .unwrap();

    let file = NcFile::from_bytes(&bytes).unwrap();
    let variable = file.variable("temp").unwrap();
    assert_eq!(variable.shape(), vec![2, 3]);
    assert_eq!(
        variable.attribute("_FillValue").unwrap().value,
        NcAttrValue::Shorts(vec![-999])
    );

    let values = file.read_variable::<i16>("temp").unwrap();
    assert_eq!(values.shape(), &[2, 3]);
    assert_eq!(
        values.as_slice_memory_order().unwrap(),
        &[-999_i16, -999, -999, -999, -999, -999]
    );
}

#[test]
fn writes_nc4_strided_variable_slice() {
    let mut builder = NcFileBuilder::new();
    let y = builder.add_dimension("y", 3).unwrap();
    let x = builder.add_dimension("x", 4).unwrap();
    let temp = builder.add_variable::<i32>("temp", &[y, x]).unwrap();
    builder
        .write_variable_slice(
            temp,
            &NcSliceInfo {
                selections: vec![
                    NcSliceInfoElem::Slice {
                        start: 0,
                        end: 3,
                        step: 2,
                    },
                    NcSliceInfoElem::Slice {
                        start: 0,
                        end: 4,
                        step: 2,
                    },
                ],
            },
            &[1_i32, 2, 3, 4],
        )
        .unwrap();

    let (_format, bytes) = builder
        .to_vec(NcWriteOptions {
            format: NcWriteFormat::Nc4,
        })
        .unwrap();

    let file = NcFile::from_bytes(&bytes).unwrap();
    let values = file.read_variable::<i32>("temp").unwrap();
    assert_eq!(values.shape(), &[3, 4]);
    assert_eq!(
        values.as_slice_memory_order().unwrap(),
        &[
            1,
            NC_FILL_INT,
            2,
            NC_FILL_INT,
            NC_FILL_INT,
            NC_FILL_INT,
            NC_FILL_INT,
            NC_FILL_INT,
            3,
            NC_FILL_INT,
            4,
            NC_FILL_INT
        ]
    );
}

#[test]
fn writes_nc4_char_variable() {
    let mut builder = NcFileBuilder::new();
    let name = builder.add_dimension("name", 2).unwrap();
    let strlen = builder.add_dimension("strlen", 5).unwrap();
    let variable = builder
        .add_char_variable("station_name", &[name, strlen])
        .unwrap();
    builder
        .write_char_variable(variable, b"alphabeta\0")
        .unwrap();

    let (_format, bytes) = builder
        .to_vec(NcWriteOptions {
            format: NcWriteFormat::Nc4,
        })
        .unwrap();

    let file = NcFile::from_bytes(&bytes).unwrap();
    let variable = file.variable("station_name").unwrap();
    assert_eq!(variable.dtype(), &netcdf_writer::NcType::Char);
    assert_eq!(variable.shape(), vec![2, 5]);
    assert_eq!(
        file.read_variable_raw_bytes("station_name").unwrap(),
        b"alphabeta\0"
    );
    assert_eq!(
        file.read_variable_as_strings("station_name").unwrap(),
        vec!["alpha".to_string(), "beta".to_string()]
    );
}

#[test]
fn writes_nc4_char_variable_slice() {
    let mut builder = NcFileBuilder::new();
    let station = builder.add_dimension("station", 2).unwrap();
    let strlen = builder.add_dimension("strlen", 5).unwrap();
    let variable = builder
        .add_char_variable("station_name", &[station, strlen])
        .unwrap();
    builder
        .write_char_variable_slice(
            variable,
            &NcSliceInfo {
                selections: vec![
                    NcSliceInfoElem::Index(0),
                    NcSliceInfoElem::Slice {
                        start: 0,
                        end: 5,
                        step: 1,
                    },
                ],
            },
            b"alpha",
        )
        .unwrap();
    builder
        .write_char_variable_slice(
            variable,
            &NcSliceInfo {
                selections: vec![
                    NcSliceInfoElem::Index(1),
                    NcSliceInfoElem::Slice {
                        start: 0,
                        end: 4,
                        step: 1,
                    },
                ],
            },
            b"beta",
        )
        .unwrap();

    let (_format, bytes) = builder
        .to_vec(NcWriteOptions {
            format: NcWriteFormat::Nc4,
        })
        .unwrap();

    let file = NcFile::from_bytes(&bytes).unwrap();
    assert_eq!(
        file.read_variable_raw_bytes("station_name").unwrap(),
        b"alphabeta\0"
    );
    assert_eq!(
        file.read_variable_as_strings("station_name").unwrap(),
        vec!["alpha".to_string(), "beta".to_string()]
    );
}

#[test]
fn writes_nc4_char_variable_strings() {
    let mut builder = NcFileBuilder::new();
    let station = builder.add_dimension("station", 2).unwrap();
    let strlen = builder.add_dimension("strlen", 5).unwrap();
    let variable = builder
        .add_char_variable("station_name", &[station, strlen])
        .unwrap();
    builder
        .write_char_variable_strings(variable, &["alpha", "beta"])
        .unwrap();

    let (_format, bytes) = builder
        .to_vec(NcWriteOptions {
            format: NcWriteFormat::Nc4,
        })
        .unwrap();

    let file = NcFile::from_bytes(&bytes).unwrap();
    assert_eq!(
        file.read_variable_raw_bytes("station_name").unwrap(),
        b"alphabeta\0"
    );
    assert_eq!(
        file.read_variable_as_strings("station_name").unwrap(),
        vec!["alpha".to_string(), "beta".to_string()]
    );
}

#[test]
fn writes_nc4_multiple_unlimited_char_variable_string_slices() {
    let mut builder = NcFileBuilder::new();
    let time = builder.add_unlimited_dimension("time").unwrap();
    let member = builder.add_unlimited_dimension("member").unwrap();
    let strlen = builder.add_dimension("strlen", 5).unwrap();
    let variable = builder
        .add_char_variable("station_name", &[time, member, strlen])
        .unwrap();

    for (time_index, member_index, value) in [
        (0, 0, "t0m0"),
        (1, 0, "t1m0"),
        (0, 1, "t0m1"),
        (2, 1, "t2m1"),
    ] {
        builder
            .write_char_variable_strings_slice(
                variable,
                &NcSliceInfo {
                    selections: vec![
                        NcSliceInfoElem::Index(time_index),
                        NcSliceInfoElem::Index(member_index),
                    ],
                },
                &[value],
            )
            .unwrap();
    }

    let (_format, bytes) = builder
        .to_vec(NcWriteOptions {
            format: NcWriteFormat::Nc4,
        })
        .unwrap();

    let file = NcFile::from_bytes(&bytes).unwrap();
    assert_eq!(file.dimension("time").unwrap().size, 3);
    assert_eq!(file.dimension("member").unwrap().size, 2);
    assert_eq!(
        file.read_variable_as_strings("station_name").unwrap(),
        vec![
            "t0m0".to_string(),
            "t0m1".to_string(),
            "t1m0".to_string(),
            "".to_string(),
            "".to_string(),
            "t2m1".to_string()
        ]
    );
}

#[test]
fn writes_nc4_string_vector_attributes() {
    let mut builder = NcFileBuilder::new();
    builder
        .add_attribute(
            "history",
            NcAttrValue::Strings(vec!["created".to_string(), "updated".to_string()]),
        )
        .unwrap();
    builder
        .add_attribute(
            "single_string_array",
            NcAttrValue::Strings(vec!["kept as array".to_string()]),
        )
        .unwrap();
    let variable = builder.add_variable::<i32>("value", &[]).unwrap();
    builder
        .add_variable_attribute(
            variable,
            "labels",
            NcAttrValue::Strings(vec!["minimum".to_string(), "maximum".to_string()]),
        )
        .unwrap();
    builder.write_variable(variable, &[7_i32]).unwrap();

    let (_format, bytes) = builder
        .to_vec(NcWriteOptions {
            format: NcWriteFormat::Nc4,
        })
        .unwrap();

    let file = NcFile::from_bytes(&bytes).unwrap();
    assert_eq!(
        file.global_attribute("history").unwrap().value,
        NcAttrValue::Strings(vec!["created".to_string(), "updated".to_string()])
    );
    assert_eq!(
        file.global_attribute("single_string_array").unwrap().value,
        NcAttrValue::Strings(vec!["kept as array".to_string()])
    );
    assert_eq!(
        file.variable("value")
            .unwrap()
            .attribute("labels")
            .unwrap()
            .value,
        NcAttrValue::Strings(vec!["minimum".to_string(), "maximum".to_string()])
    );
}

#[test]
fn writes_nc4_string_variable() {
    let mut builder = NcFileBuilder::new();
    let station = builder.add_dimension("station", 3).unwrap();
    let variable = builder
        .add_string_variable("station_name", &[station])
        .unwrap();
    builder
        .write_string_variable(variable, &["alpha", "bravo", "charlie"])
        .unwrap();

    let (_format, bytes) = builder
        .to_vec(NcWriteOptions {
            format: NcWriteFormat::Nc4,
        })
        .unwrap();

    let file = NcFile::from_bytes(&bytes).unwrap();
    let variable = file.variable("station_name").unwrap();
    assert_eq!(variable.dtype(), &netcdf_writer::NcType::String);
    assert_eq!(variable.shape(), vec![3]);
    assert_eq!(
        file.read_variable_as_strings("station_name").unwrap(),
        vec![
            "alpha".to_string(),
            "bravo".to_string(),
            "charlie".to_string()
        ]
    );
}

#[test]
fn writes_nc4_string_variable_slices() {
    let mut builder = NcFileBuilder::new();
    let station = builder.add_dimension("station", 3).unwrap();
    let variable = builder
        .add_string_variable("station_name", &[station])
        .unwrap();
    builder
        .write_string_variable_slice(
            variable,
            &NcSliceInfo {
                selections: vec![NcSliceInfoElem::Index(1)],
            },
            &["bravo"],
        )
        .unwrap();
    builder
        .write_string_variable_slice(
            variable,
            &NcSliceInfo {
                selections: vec![NcSliceInfoElem::Slice {
                    start: 0,
                    end: 3,
                    step: 2,
                }],
            },
            &["alpha", "charlie"],
        )
        .unwrap();

    let (_format, bytes) = builder
        .to_vec(NcWriteOptions {
            format: NcWriteFormat::Nc4,
        })
        .unwrap();

    let file = NcFile::from_bytes(&bytes).unwrap();
    let variable = file.variable("station_name").unwrap();
    assert_eq!(variable.dtype(), &netcdf_writer::NcType::String);
    assert_eq!(variable.shape(), vec![3]);
    assert_eq!(
        file.read_variable_as_strings("station_name").unwrap(),
        vec![
            "alpha".to_string(),
            "bravo".to_string(),
            "charlie".to_string()
        ]
    );
}

#[test]
fn writes_nc4_unlimited_string_variable() {
    let mut builder = NcFileBuilder::new();
    let obs = builder.add_unlimited_dimension("obs").unwrap();
    let variable = builder.add_string_variable("quality", &[obs]).unwrap();
    builder
        .write_string_variable(variable, &["good", "suspect", "bad"])
        .unwrap();

    let (_format, bytes) = builder
        .to_vec(NcWriteOptions {
            format: NcWriteFormat::Nc4,
        })
        .unwrap();

    let file = NcFile::from_bytes(&bytes).unwrap();
    let obs_dim = file.dimension("obs").unwrap();
    assert_eq!(obs_dim.size, 3);
    assert!(obs_dim.is_unlimited);
    assert_eq!(
        file.read_variable_as_strings("quality").unwrap(),
        vec!["good".to_string(), "suspect".to_string(), "bad".to_string()]
    );
}

#[test]
fn writes_nc4_unlimited_string_variable_slice_append() {
    let mut builder = NcFileBuilder::new();
    let obs = builder.add_unlimited_dimension("obs").unwrap();
    let variable = builder.add_string_variable("quality", &[obs]).unwrap();
    builder
        .write_string_variable_slice(
            variable,
            &NcSliceInfo {
                selections: vec![NcSliceInfoElem::Index(1)],
            },
            &["suspect"],
        )
        .unwrap();
    builder
        .write_string_variable_slice(
            variable,
            &NcSliceInfo {
                selections: vec![NcSliceInfoElem::Index(3)],
            },
            &["bad"],
        )
        .unwrap();

    let (_format, bytes) = builder
        .to_vec(NcWriteOptions {
            format: NcWriteFormat::Nc4,
        })
        .unwrap();

    let file = NcFile::from_bytes(&bytes).unwrap();
    let obs_dim = file.dimension("obs").unwrap();
    assert_eq!(obs_dim.size, 4);
    assert!(obs_dim.is_unlimited);
    assert_eq!(
        file.read_variable_as_strings("quality").unwrap(),
        vec![
            "".to_string(),
            "suspect".to_string(),
            "".to_string(),
            "bad".to_string()
        ]
    );
}

#[test]
fn writes_nc4_multiple_unlimited_string_variable_slice_append() {
    let mut builder = NcFileBuilder::new();
    let time = builder.add_unlimited_dimension("time").unwrap();
    let member = builder.add_unlimited_dimension("member").unwrap();
    let variable = builder
        .add_string_variable("quality", &[time, member])
        .unwrap();

    for (time_index, member_index, value) in [
        (0, 0, "t0m0"),
        (1, 0, "t1m0"),
        (0, 1, "t0m1"),
        (2, 1, "t2m1"),
    ] {
        builder
            .write_string_variable_slice(
                variable,
                &NcSliceInfo {
                    selections: vec![
                        NcSliceInfoElem::Index(time_index),
                        NcSliceInfoElem::Index(member_index),
                    ],
                },
                &[value],
            )
            .unwrap();
    }

    let (_format, bytes) = builder
        .to_vec(NcWriteOptions {
            format: NcWriteFormat::Nc4,
        })
        .unwrap();

    let file = NcFile::from_bytes(&bytes).unwrap();
    assert_eq!(file.dimension("time").unwrap().size, 3);
    assert_eq!(file.dimension("member").unwrap().size, 2);
    assert_eq!(
        file.read_variable_as_strings("quality").unwrap(),
        vec![
            "t0m0".to_string(),
            "t0m1".to_string(),
            "t1m0".to_string(),
            "".to_string(),
            "".to_string(),
            "t2m1".to_string()
        ]
    );
}

#[test]
fn writes_nc4_user_defined_variables() {
    let mut builder = NcFileBuilder::new();
    let n = builder.add_dimension("n", 2).unwrap();
    let quality_type = NcType::Enum {
        base: Box::new(NcType::UByte),
        members: vec![
            NcEnumMember {
                name: "bad".to_string(),
                value: NcIntegerValue::U8(1),
            },
            NcEnumMember {
                name: "good".to_string(),
                value: NcIntegerValue::U8(2),
            },
        ],
    };
    let samples_type = NcType::Array {
        base: Box::new(NcType::Short),
        dims: vec![3],
    };
    let obs_type = NcType::Compound {
        size: 12,
        fields: vec![
            NcCompoundField {
                name: "temp".to_string(),
                offset: 0,
                dtype: NcType::Float,
            },
            NcCompoundField {
                name: "quality".to_string(),
                offset: 4,
                dtype: quality_type.clone(),
            },
            NcCompoundField {
                name: "samples".to_string(),
                offset: 6,
                dtype: samples_type.clone(),
            },
        ],
    };

    let quality = builder
        .add_user_defined_variable("quality", &[n], quality_type.clone())
        .unwrap();
    let blob = builder
        .add_user_defined_variable(
            "blob",
            &[n],
            NcType::Opaque {
                size: 4,
                tag: "blob".to_string(),
            },
        )
        .unwrap();
    let obs = builder
        .add_user_defined_variable("obs", &[], obs_type.clone())
        .unwrap();
    let samples = builder
        .add_user_defined_variable("samples", &[n], samples_type.clone())
        .unwrap();

    builder
        .write_enum_variable(quality, &[NcIntegerValue::U8(2), NcIntegerValue::U8(1)])
        .unwrap();
    builder
        .write_opaque_variable(blob, &[&[1_u8, 2, 3, 4][..], &[5_u8, 6, 7, 8][..]])
        .unwrap();
    builder
        .write_array_variable(samples, &[1_i16, 2, 3, 10, 11, 12])
        .unwrap();
    let mut obs_bytes = Vec::new();
    obs_bytes.extend_from_slice(&12.5_f32.to_le_bytes());
    obs_bytes.push(2);
    obs_bytes.push(0);
    obs_bytes.extend_from_slice(&10_i16.to_le_bytes());
    obs_bytes.extend_from_slice(&11_i16.to_le_bytes());
    obs_bytes.extend_from_slice(&12_i16.to_le_bytes());
    builder
        .write_user_defined_variable_bytes(obs, &obs_bytes)
        .unwrap();

    let (_format, bytes) = builder
        .to_vec(NcWriteOptions {
            format: NcWriteFormat::Nc4,
        })
        .unwrap();

    let file = NcFile::from_bytes(&bytes).unwrap();
    assert_eq!(file.variable("quality").unwrap().dtype(), &quality_type);
    assert_eq!(file.read_variable_raw_bytes("quality").unwrap(), vec![2, 1]);

    let quality_values = file.read_variable_user_defined("quality").unwrap();
    match &quality_values.as_slice().unwrap()[0] {
        netcdf_reader::NcValue::Enum(value) => {
            assert_eq!(value.value, NcIntegerValue::U8(2));
            assert_eq!(value.member.as_deref(), Some("good"));
        }
        other => panic!("expected enum value, got {other:?}"),
    }

    let blob_values = file.read_variable_user_defined("blob").unwrap();
    assert_eq!(
        blob_values.as_slice().unwrap()[1],
        netcdf_reader::NcValue::Opaque(vec![5, 6, 7, 8])
    );

    let sample_values = file.read_variable_user_defined("samples").unwrap();
    match &sample_values.as_slice().unwrap()[1] {
        netcdf_reader::NcValue::Array(array) => {
            assert_eq!(array.dims, vec![3]);
            assert_eq!(
                array.values,
                vec![
                    netcdf_reader::NcValue::Short(10),
                    netcdf_reader::NcValue::Short(11),
                    netcdf_reader::NcValue::Short(12),
                ]
            );
        }
        other => panic!("expected array value, got {other:?}"),
    }

    assert_eq!(file.variable("obs").unwrap().dtype(), &obs_type);
    let obs_values = file.read_variable_user_defined("obs").unwrap();
    match &obs_values.as_slice().unwrap()[0] {
        netcdf_reader::NcValue::Compound(fields) => {
            assert_eq!(fields[0].name, "temp");
            assert_eq!(fields[0].value, netcdf_reader::NcValue::Float(12.5));
            match &fields[1].value {
                netcdf_reader::NcValue::Enum(value) => {
                    assert_eq!(value.member.as_deref(), Some("good"));
                }
                other => panic!("expected enum field, got {other:?}"),
            }
            match &fields[2].value {
                netcdf_reader::NcValue::Array(array) => {
                    assert_eq!(array.dims, vec![3]);
                    assert_eq!(
                        array.values,
                        vec![
                            netcdf_reader::NcValue::Short(10),
                            netcdf_reader::NcValue::Short(11),
                            netcdf_reader::NcValue::Short(12),
                        ]
                    );
                }
                other => panic!("expected array field, got {other:?}"),
            }
        }
        other => panic!("expected compound value, got {other:?}"),
    }
}

#[test]
fn writes_nc4_user_defined_variable_slices() {
    let mut builder = NcFileBuilder::new();
    let n = builder.add_dimension("n", 3).unwrap();
    let quality_type = NcType::Enum {
        base: Box::new(NcType::UByte),
        members: vec![
            NcEnumMember {
                name: "bad".to_string(),
                value: NcIntegerValue::U8(1),
            },
            NcEnumMember {
                name: "good".to_string(),
                value: NcIntegerValue::U8(2),
            },
        ],
    };
    let samples_type = NcType::Array {
        base: Box::new(NcType::Short),
        dims: vec![3],
    };
    let obs_type = NcType::Compound {
        size: 12,
        fields: vec![
            NcCompoundField {
                name: "temp".to_string(),
                offset: 0,
                dtype: NcType::Float,
            },
            NcCompoundField {
                name: "quality".to_string(),
                offset: 4,
                dtype: NcType::UByte,
            },
            NcCompoundField {
                name: "samples".to_string(),
                offset: 6,
                dtype: NcType::Array {
                    base: Box::new(NcType::Short),
                    dims: vec![3],
                },
            },
        ],
    };
    let quality = builder
        .add_user_defined_variable("quality", &[n], quality_type)
        .unwrap();
    let blob = builder
        .add_user_defined_variable(
            "blob",
            &[n],
            NcType::Opaque {
                size: 4,
                tag: "blob".to_string(),
            },
        )
        .unwrap();
    let samples = builder
        .add_user_defined_variable("samples", &[n], samples_type)
        .unwrap();
    let obs = builder
        .add_user_defined_variable("obs", &[n], obs_type)
        .unwrap();

    builder
        .write_enum_variable_slice(
            quality,
            &NcSliceInfo {
                selections: vec![NcSliceInfoElem::Slice {
                    start: 1,
                    end: 3,
                    step: 1,
                }],
            },
            &[NcIntegerValue::U8(2), NcIntegerValue::U8(1)],
        )
        .unwrap();
    builder
        .write_opaque_variable_slice(
            blob,
            &NcSliceInfo {
                selections: vec![NcSliceInfoElem::Index(2)],
            },
            &[&[5_u8, 6, 7, 8][..]],
        )
        .unwrap();
    builder
        .write_array_variable_slice(
            samples,
            &NcSliceInfo {
                selections: vec![NcSliceInfoElem::Index(1)],
            },
            &[10_i16, 11, 12],
        )
        .unwrap();
    let mut obs_bytes = Vec::new();
    obs_bytes.extend_from_slice(&12.5_f32.to_le_bytes());
    obs_bytes.push(2);
    obs_bytes.push(0);
    obs_bytes.extend_from_slice(&10_i16.to_le_bytes());
    obs_bytes.extend_from_slice(&11_i16.to_le_bytes());
    obs_bytes.extend_from_slice(&12_i16.to_le_bytes());
    builder
        .write_user_defined_variable_slice_bytes(
            obs,
            &NcSliceInfo {
                selections: vec![NcSliceInfoElem::Index(2)],
            },
            &obs_bytes,
        )
        .unwrap();

    let (_format, bytes) = builder
        .to_vec(NcWriteOptions {
            format: NcWriteFormat::Nc4,
        })
        .unwrap();

    let file = NcFile::from_bytes(&bytes).unwrap();
    assert_eq!(
        file.read_variable_raw_bytes("quality").unwrap(),
        vec![0, 2, 1]
    );
    assert_eq!(
        file.read_variable_raw_bytes("blob").unwrap(),
        vec![0, 0, 0, 0, 0, 0, 0, 0, 5, 6, 7, 8]
    );
    let mut expected_samples = vec![0_u8; 18];
    expected_samples[6..8].copy_from_slice(&10_i16.to_le_bytes());
    expected_samples[8..10].copy_from_slice(&11_i16.to_le_bytes());
    expected_samples[10..12].copy_from_slice(&12_i16.to_le_bytes());
    assert_eq!(
        file.read_variable_raw_bytes("samples").unwrap(),
        expected_samples
    );
    let mut expected_obs = vec![0_u8; 36];
    expected_obs[24..36].copy_from_slice(&obs_bytes);
    assert_eq!(file.read_variable_raw_bytes("obs").unwrap(), expected_obs);
}

#[test]
fn writes_nc4_user_defined_unlimited_slice_append() {
    let mut builder = NcFileBuilder::new();
    let obs = builder.add_unlimited_dimension("obs").unwrap();
    let blob = builder
        .add_user_defined_variable(
            "blob",
            &[obs],
            NcType::Opaque {
                size: 4,
                tag: "blob".to_string(),
            },
        )
        .unwrap();
    builder
        .write_opaque_variable_slice(
            blob,
            &NcSliceInfo {
                selections: vec![NcSliceInfoElem::Index(1)],
            },
            &[&[1_u8, 2, 3, 4][..]],
        )
        .unwrap();
    builder
        .write_opaque_variable_slice(
            blob,
            &NcSliceInfo {
                selections: vec![NcSliceInfoElem::Index(3)],
            },
            &[&[9_u8, 8, 7, 6][..]],
        )
        .unwrap();

    let (_format, bytes) = builder
        .to_vec(NcWriteOptions {
            format: NcWriteFormat::Nc4,
        })
        .unwrap();

    let file = NcFile::from_bytes(&bytes).unwrap();
    let obs_dim = file.dimension("obs").unwrap();
    assert_eq!(obs_dim.size, 4);
    assert!(obs_dim.is_unlimited);
    assert_eq!(
        file.read_variable_raw_bytes("blob").unwrap(),
        vec![0, 0, 0, 0, 1, 2, 3, 4, 0, 0, 0, 0, 9, 8, 7, 6]
    );
}

#[test]
fn writes_nc4_user_defined_multiple_unlimited_slice_append() {
    let mut builder = NcFileBuilder::new();
    let time = builder.add_unlimited_dimension("time").unwrap();
    let member = builder.add_unlimited_dimension("member").unwrap();
    let blob = builder
        .add_user_defined_variable(
            "blob",
            &[time, member],
            NcType::Opaque {
                size: 2,
                tag: "blob".to_string(),
            },
        )
        .unwrap();

    for (time_index, member_index, value) in [
        (0, 0, [1_u8, 0_u8]),
        (1, 0, [2_u8, 0_u8]),
        (0, 1, [1_u8, 1_u8]),
        (2, 1, [3_u8, 1_u8]),
    ] {
        builder
            .write_opaque_variable_slice(
                blob,
                &NcSliceInfo {
                    selections: vec![
                        NcSliceInfoElem::Index(time_index),
                        NcSliceInfoElem::Index(member_index),
                    ],
                },
                &[&value[..]],
            )
            .unwrap();
    }

    let (_format, bytes) = builder
        .to_vec(NcWriteOptions {
            format: NcWriteFormat::Nc4,
        })
        .unwrap();

    let file = NcFile::from_bytes(&bytes).unwrap();
    assert_eq!(file.dimension("time").unwrap().size, 3);
    assert_eq!(file.dimension("member").unwrap().size, 2);
    assert_eq!(
        file.read_variable_raw_bytes("blob").unwrap(),
        vec![1, 0, 1, 1, 2, 0, 0, 0, 0, 0, 3, 1]
    );
}

#[test]
fn writes_nc4_vlen_sequence_variable() {
    let mut builder = NcFileBuilder::new();
    let obs = builder.add_unlimited_dimension("obs").unwrap();
    let dtype = NcType::VLen {
        base: Box::new(NcType::Short),
    };
    let ragged = builder
        .add_user_defined_variable("ragged", &[obs], dtype.clone())
        .unwrap();
    let sequences = vec![vec![1_i16, 2], Vec::<i16>::new(), vec![10_i16, 11, 12]];
    builder.write_vlen_variable(ragged, &sequences).unwrap();

    let (_format, bytes) = builder
        .to_vec(NcWriteOptions {
            format: NcWriteFormat::Nc4,
        })
        .unwrap();

    let file = NcFile::from_bytes(&bytes).unwrap();
    let obs_dim = file.dimension("obs").unwrap();
    assert_eq!(obs_dim.size, 3);
    assert!(obs_dim.is_unlimited);
    assert_eq!(file.variable("ragged").unwrap().dtype(), &dtype);

    let decoded = file.read_variable_user_defined("ragged").unwrap();
    let values = decoded.as_slice().unwrap();
    assert_eq!(
        values[0],
        netcdf_reader::NcValue::VLen(vec![
            netcdf_reader::NcValue::Short(1),
            netcdf_reader::NcValue::Short(2)
        ])
    );
    assert_eq!(values[1], netcdf_reader::NcValue::VLen(Vec::new()));
    assert_eq!(
        values[2],
        netcdf_reader::NcValue::VLen(vec![
            netcdf_reader::NcValue::Short(10),
            netcdf_reader::NcValue::Short(11),
            netcdf_reader::NcValue::Short(12)
        ])
    );
}

#[test]
fn writes_nc4_vlen_sequence_variable_slices() {
    let mut builder = NcFileBuilder::new();
    let obs = builder.add_dimension("obs", 3).unwrap();
    let dtype = NcType::VLen {
        base: Box::new(NcType::Short),
    };
    let ragged = builder
        .add_user_defined_variable("ragged", &[obs], dtype.clone())
        .unwrap();
    builder
        .write_vlen_variable_slice(
            ragged,
            &NcSliceInfo {
                selections: vec![NcSliceInfoElem::Index(1)],
            },
            &[vec![3_i16, 4]],
        )
        .unwrap();
    builder
        .write_vlen_variable_slice(
            ragged,
            &NcSliceInfo {
                selections: vec![NcSliceInfoElem::Slice {
                    start: 0,
                    end: 3,
                    step: 2,
                }],
            },
            &[vec![1_i16], vec![10_i16, 11, 12]],
        )
        .unwrap();

    let (_format, bytes) = builder
        .to_vec(NcWriteOptions {
            format: NcWriteFormat::Nc4,
        })
        .unwrap();

    let file = NcFile::from_bytes(&bytes).unwrap();
    let variable = file.variable("ragged").unwrap();
    assert_eq!(variable.dtype(), &dtype);
    assert_eq!(variable.shape(), vec![3]);

    let decoded = file.read_variable_user_defined("ragged").unwrap();
    let values = decoded.as_slice().unwrap();
    assert_eq!(
        values[0],
        netcdf_reader::NcValue::VLen(vec![netcdf_reader::NcValue::Short(1)])
    );
    assert_eq!(
        values[1],
        netcdf_reader::NcValue::VLen(vec![
            netcdf_reader::NcValue::Short(3),
            netcdf_reader::NcValue::Short(4)
        ])
    );
    assert_eq!(
        values[2],
        netcdf_reader::NcValue::VLen(vec![
            netcdf_reader::NcValue::Short(10),
            netcdf_reader::NcValue::Short(11),
            netcdf_reader::NcValue::Short(12)
        ])
    );
}

#[test]
fn writes_nc4_unlimited_vlen_sequence_variable_slice_append() {
    let mut builder = NcFileBuilder::new();
    let obs = builder.add_unlimited_dimension("obs").unwrap();
    let dtype = NcType::VLen {
        base: Box::new(NcType::Short),
    };
    let ragged = builder
        .add_user_defined_variable("ragged", &[obs], dtype.clone())
        .unwrap();
    builder
        .write_vlen_variable_slice_bytes(
            ragged,
            &NcSliceInfo {
                selections: vec![NcSliceInfoElem::Index(1)],
            },
            &[2_i16.to_le_bytes().to_vec()],
        )
        .unwrap();
    let mut sequence = Vec::new();
    sequence.extend_from_slice(&4_i16.to_le_bytes());
    sequence.extend_from_slice(&5_i16.to_le_bytes());
    builder
        .write_vlen_variable_slice_bytes(
            ragged,
            &NcSliceInfo {
                selections: vec![NcSliceInfoElem::Index(3)],
            },
            &[sequence],
        )
        .unwrap();

    let (_format, bytes) = builder
        .to_vec(NcWriteOptions {
            format: NcWriteFormat::Nc4,
        })
        .unwrap();

    let file = NcFile::from_bytes(&bytes).unwrap();
    let obs_dim = file.dimension("obs").unwrap();
    assert_eq!(obs_dim.size, 4);
    assert!(obs_dim.is_unlimited);
    assert_eq!(file.variable("ragged").unwrap().dtype(), &dtype);

    let decoded = file.read_variable_user_defined("ragged").unwrap();
    let values = decoded.as_slice().unwrap();
    assert_eq!(values[0], netcdf_reader::NcValue::VLen(Vec::new()));
    assert_eq!(
        values[1],
        netcdf_reader::NcValue::VLen(vec![netcdf_reader::NcValue::Short(2)])
    );
    assert_eq!(values[2], netcdf_reader::NcValue::VLen(Vec::new()));
    assert_eq!(
        values[3],
        netcdf_reader::NcValue::VLen(vec![
            netcdf_reader::NcValue::Short(4),
            netcdf_reader::NcValue::Short(5)
        ])
    );
}

#[test]
fn writes_nc4_multiple_unlimited_vlen_sequence_variable_slice_append() {
    let mut builder = NcFileBuilder::new();
    let time = builder.add_unlimited_dimension("time").unwrap();
    let member = builder.add_unlimited_dimension("member").unwrap();
    let dtype = NcType::VLen {
        base: Box::new(NcType::Short),
    };
    let ragged = builder
        .add_user_defined_variable("ragged", &[time, member], dtype)
        .unwrap();

    for (time_index, member_index, value) in [
        (0, 0, vec![1_i16]),
        (1, 0, vec![2_i16]),
        (0, 1, vec![3_i16]),
        (2, 1, vec![4_i16, 5]),
    ] {
        builder
            .write_vlen_variable_slice(
                ragged,
                &NcSliceInfo {
                    selections: vec![
                        NcSliceInfoElem::Index(time_index),
                        NcSliceInfoElem::Index(member_index),
                    ],
                },
                &[value],
            )
            .unwrap();
    }

    let (_format, bytes) = builder
        .to_vec(NcWriteOptions {
            format: NcWriteFormat::Nc4,
        })
        .unwrap();

    let file = NcFile::from_bytes(&bytes).unwrap();
    assert_eq!(file.dimension("time").unwrap().size, 3);
    assert_eq!(file.dimension("member").unwrap().size, 2);

    let decoded = file.read_variable_user_defined("ragged").unwrap();
    let values = decoded.as_slice().unwrap();
    assert_eq!(
        values,
        &[
            netcdf_reader::NcValue::VLen(vec![netcdf_reader::NcValue::Short(1)]),
            netcdf_reader::NcValue::VLen(vec![netcdf_reader::NcValue::Short(3)]),
            netcdf_reader::NcValue::VLen(vec![netcdf_reader::NcValue::Short(2)]),
            netcdf_reader::NcValue::VLen(Vec::new()),
            netcdf_reader::NcValue::VLen(Vec::new()),
            netcdf_reader::NcValue::VLen(vec![
                netcdf_reader::NcValue::Short(4),
                netcdf_reader::NcValue::Short(5)
            ])
        ]
    );
}

#[test]
fn writes_nc4_classic_marker() {
    let mut builder = NcFileBuilder::new();
    let variable = builder.add_variable::<f64>("value", &[]).unwrap();
    builder.write_variable(variable, &[6.25_f64]).unwrap();

    let (format, bytes) = builder
        .to_vec(NcWriteOptions {
            format: NcWriteFormat::Nc4Classic,
        })
        .unwrap();
    assert_eq!(format, NcFormat::Nc4Classic);

    let file = NcFile::from_bytes(&bytes).unwrap();
    assert_eq!(file.format(), NcFormat::Nc4Classic);
    let values = file.read_variable::<f64>("value").unwrap();
    assert_eq!(values.as_slice_memory_order().unwrap(), &[6.25]);
}

#[test]
fn rejects_nc4_classic_string_attributes() {
    let mut builder = NcFileBuilder::new();
    builder
        .add_attribute(
            "history",
            NcAttrValue::Strings(vec!["requires enhanced model".to_string()]),
        )
        .unwrap();
    let variable = builder.add_variable::<f64>("value", &[]).unwrap();
    builder.write_variable(variable, &[6.25_f64]).unwrap();

    let err = builder
        .to_vec(NcWriteOptions {
            format: NcWriteFormat::Nc4Classic,
        })
        .unwrap_err();

    assert!(err.to_string().contains("NC_STRING attributes"));
}

#[test]
fn rejects_nc4_classic_string_variable() {
    let mut builder = NcFileBuilder::new();
    let variable = builder.add_string_variable("name", &[]).unwrap();
    builder
        .write_string_variable(variable, &["enhanced"])
        .unwrap();

    let err = builder
        .to_vec(NcWriteOptions {
            format: NcWriteFormat::Nc4Classic,
        })
        .unwrap_err();

    assert!(err.to_string().contains("requires NetCDF-4"));
}

#[test]
fn rejects_nc4_classic_group_paths() {
    let mut builder = NcFileBuilder::new();
    let dim = builder.add_dimension_path("science/x", 2).unwrap();
    let variable = builder
        .add_variable_path::<i32>("science/value", &[dim])
        .unwrap();
    builder.write_variable(variable, &[1_i32, 2]).unwrap();

    let err = builder
        .to_vec(NcWriteOptions {
            format: NcWriteFormat::Nc4Classic,
        })
        .unwrap_err();

    assert!(err.to_string().contains("requires NetCDF-4"));
}

#[test]
fn rejects_nc4_classic_group_attributes() {
    let mut builder = NcFileBuilder::new();
    builder
        .add_group_attribute(
            "science",
            "title",
            NcAttrValue::Chars("enhanced model".to_string()),
        )
        .unwrap();

    let err = builder
        .to_vec(NcWriteOptions {
            format: NcWriteFormat::Nc4Classic,
        })
        .unwrap_err();

    assert!(err.to_string().contains("requires NetCDF-4"));
}

#[test]
fn writes_nc4_one_dimensional_non_coordinate_variable() {
    let mut builder = NcFileBuilder::new();
    let x = builder.add_dimension("x", 4).unwrap();
    let coordinate = builder.add_variable::<f32>("x", &[x]).unwrap();
    builder
        .write_variable(coordinate, &[0.0_f32, 1.0, 2.0, 3.0])
        .unwrap();
    let temp = builder.add_variable::<f32>("temp", &[x]).unwrap();
    builder
        .write_variable(temp, &[10.0_f32, 11.0, 12.0, 13.0])
        .unwrap();

    let (_format, bytes) = builder
        .to_vec(NcWriteOptions {
            format: NcWriteFormat::Nc4,
        })
        .unwrap();

    let file = NcFile::from_bytes(&bytes).unwrap();
    assert_eq!(file.dimension("x").unwrap().size, 4);

    let variable = file.variable("temp").unwrap();
    assert_eq!(variable.shape(), vec![4]);
    assert!(!variable.is_coordinate_variable());

    let values = file.read_variable::<f32>("temp").unwrap();
    assert_eq!(
        values.as_slice_memory_order().unwrap(),
        &[10.0, 11.0, 12.0, 13.0]
    );
}

#[test]
fn writes_nc4_multidimensional_variable_with_hidden_dimension_scales() {
    let mut builder = NcFileBuilder::new();
    let y = builder.add_dimension("y", 2).unwrap();
    let x = builder.add_dimension("x", 3).unwrap();
    let temp = builder.add_variable::<f32>("temp", &[y, x]).unwrap();
    builder
        .add_variable_attribute(temp, "units", NcAttrValue::Chars("K".to_string()))
        .unwrap();
    builder
        .write_variable(temp, &[280.0_f32, 281.0, 282.0, 283.0, 284.0, 285.0])
        .unwrap();

    let (_format, bytes) = builder
        .to_vec(NcWriteOptions {
            format: NcWriteFormat::Nc4,
        })
        .unwrap();

    let file = NcFile::from_bytes(&bytes).unwrap();
    assert_eq!(file.dimensions().unwrap().len(), 2);
    assert_eq!(file.dimension("y").unwrap().size, 2);
    assert_eq!(file.dimension("x").unwrap().size, 3);
    assert!(file.variable("y").is_err());
    assert!(file.variable("x").is_err());

    let variable = file.variable("temp").unwrap();
    assert_eq!(variable.shape(), vec![2, 3]);
    assert_eq!(
        variable
            .attribute("units")
            .unwrap()
            .value
            .as_string()
            .unwrap(),
        "K"
    );

    let values = file.read_variable::<f32>("temp").unwrap();
    assert_eq!(values.shape(), &[2, 3]);
    assert_eq!(
        values.as_slice_memory_order().unwrap(),
        &[280.0, 281.0, 282.0, 283.0, 284.0, 285.0]
    );
}

#[test]
fn writes_nc4_unlimited_dimension_variables() {
    let mut builder = NcFileBuilder::new();
    let time = builder.add_unlimited_dimension("time").unwrap();
    let station = builder.add_dimension("station", 2).unwrap();
    let time_var = builder.add_variable::<i32>("time", &[time]).unwrap();
    let temp = builder
        .add_variable::<f32>("temp", &[time, station])
        .unwrap();

    builder.write_variable(time_var, &[0_i32, 1, 2]).unwrap();
    builder
        .write_variable(temp, &[10.0_f32, 11.0, 20.0, 21.0, 30.0, 31.0])
        .unwrap();

    let (_format, bytes) = builder
        .to_vec(NcWriteOptions {
            format: NcWriteFormat::Nc4,
        })
        .unwrap();

    let file = NcFile::from_bytes(&bytes).unwrap();
    let time_dim = file.dimension("time").unwrap();
    assert_eq!(time_dim.size, 3);
    assert!(time_dim.is_unlimited);
    assert!(!file.dimension("station").unwrap().is_unlimited);

    let time_variable = file.variable("time").unwrap();
    assert!(time_variable.is_coordinate_variable_for("time"));
    let time_values = file.read_variable::<i32>("time").unwrap();
    assert_eq!(time_values.as_slice_memory_order().unwrap(), &[0, 1, 2]);

    let temp_values = file.read_variable::<f32>("temp").unwrap();
    assert_eq!(temp_values.shape(), &[3, 2]);
    assert_eq!(
        temp_values.as_slice_memory_order().unwrap(),
        &[10.0, 11.0, 20.0, 21.0, 30.0, 31.0]
    );
}

#[test]
fn writes_nc4_unlimited_variable_slice_append() {
    let mut builder = NcFileBuilder::new();
    let time = builder.add_unlimited_dimension("time").unwrap();
    let station = builder.add_dimension("station", 2).unwrap();
    let temp = builder
        .add_variable::<i32>("temp", &[time, station])
        .unwrap();
    builder
        .write_variable_slice(
            temp,
            &NcSliceInfo {
                selections: vec![
                    NcSliceInfoElem::Index(0),
                    NcSliceInfoElem::Slice {
                        start: 0,
                        end: 2,
                        step: 1,
                    },
                ],
            },
            &[10_i32, 11],
        )
        .unwrap();
    builder
        .write_variable_slice(
            temp,
            &NcSliceInfo {
                selections: vec![
                    NcSliceInfoElem::Index(2),
                    NcSliceInfoElem::Slice {
                        start: 0,
                        end: 2,
                        step: 1,
                    },
                ],
            },
            &[30_i32, 31],
        )
        .unwrap();

    let (_format, bytes) = builder
        .to_vec(NcWriteOptions {
            format: NcWriteFormat::Nc4,
        })
        .unwrap();

    let file = NcFile::from_bytes(&bytes).unwrap();
    let time_dim = file.dimension("time").unwrap();
    assert_eq!(time_dim.size, 3);
    assert!(time_dim.is_unlimited);
    let values = file.read_variable::<i32>("temp").unwrap();
    assert_eq!(values.shape(), &[3, 2]);
    assert_eq!(
        values.as_slice_memory_order().unwrap(),
        &[10, 11, NC_FILL_INT, NC_FILL_INT, 30, 31]
    );
}

#[test]
fn writes_nc4_multiple_unlimited_variable_slice_append() {
    let mut builder = NcFileBuilder::new();
    let time = builder.add_unlimited_dimension("time").unwrap();
    let member = builder.add_unlimited_dimension("member").unwrap();
    let temp = builder
        .add_variable::<i32>("temp", &[time, member])
        .unwrap();

    for (time_index, member_index, value) in [(0, 0, 10), (1, 0, 20), (0, 1, 11), (2, 1, 31)] {
        builder
            .write_variable_slice(
                temp,
                &NcSliceInfo {
                    selections: vec![
                        NcSliceInfoElem::Index(time_index),
                        NcSliceInfoElem::Index(member_index),
                    ],
                },
                &[value],
            )
            .unwrap();
    }

    let (_format, bytes) = builder
        .to_vec(NcWriteOptions {
            format: NcWriteFormat::Nc4,
        })
        .unwrap();

    let file = NcFile::from_bytes(&bytes).unwrap();
    let time_dim = file.dimension("time").unwrap();
    let member_dim = file.dimension("member").unwrap();
    assert_eq!(time_dim.size, 3);
    assert_eq!(member_dim.size, 2);
    assert!(time_dim.is_unlimited);
    assert!(member_dim.is_unlimited);

    let values = file.read_variable::<i32>("temp").unwrap();
    assert_eq!(values.shape(), &[3, 2]);
    assert_eq!(
        values.as_slice_memory_order().unwrap(),
        &[10, 11, 20, NC_FILL_INT, NC_FILL_INT, 31]
    );
}

#[test]
fn writes_nc4_zero_record_unlimited_dimension_variables() {
    let mut builder = NcFileBuilder::new();
    let time = builder.add_unlimited_dimension("time").unwrap();
    let station = builder.add_dimension("station", 2).unwrap();
    let time_var = builder.add_variable::<i32>("time", &[time]).unwrap();
    let temp = builder
        .add_variable::<f32>("temp", &[time, station])
        .unwrap();

    builder.write_variable(time_var, &[] as &[i32]).unwrap();
    builder.write_variable(temp, &[] as &[f32]).unwrap();

    let (_format, bytes) = builder
        .to_vec(NcWriteOptions {
            format: NcWriteFormat::Nc4,
        })
        .unwrap();

    let file = NcFile::from_bytes(&bytes).unwrap();
    let time_dim = file.dimension("time").unwrap();
    assert_eq!(time_dim.size, 0);
    assert!(time_dim.is_unlimited);
    assert!(!file.dimension("station").unwrap().is_unlimited);

    let time_values = file.read_variable::<i32>("time").unwrap();
    assert_eq!(time_values.shape(), &[0]);
    assert!(time_values.as_slice_memory_order().unwrap().is_empty());

    let temp_values = file.read_variable::<f32>("temp").unwrap();
    assert_eq!(temp_values.shape(), &[0, 2]);
    assert!(temp_values.as_slice_memory_order().unwrap().is_empty());
}

#[test]
fn writes_nc4_multiple_unlimited_dimensions() {
    let mut builder = NcFileBuilder::new();
    let time = builder.add_unlimited_dimension("time").unwrap();
    let member = builder.add_unlimited_dimension("member").unwrap();
    let time_var = builder.add_variable::<i32>("time", &[time]).unwrap();
    let member_var = builder.add_variable::<i32>("member", &[member]).unwrap();
    let temp = builder
        .add_variable::<f32>("temp", &[time, member])
        .unwrap();

    builder.write_variable(time_var, &[0_i32, 1, 2]).unwrap();
    builder.write_variable(member_var, &[10_i32, 20]).unwrap();
    builder
        .write_variable(temp, &[10.0_f32, 11.0, 20.0, 21.0, 30.0, 31.0])
        .unwrap();

    let (_format, bytes) = builder
        .to_vec(NcWriteOptions {
            format: NcWriteFormat::Nc4,
        })
        .unwrap();

    let file = NcFile::from_bytes(&bytes).unwrap();
    let time_dim = file.dimension("time").unwrap();
    let member_dim = file.dimension("member").unwrap();
    assert_eq!(time_dim.size, 3);
    assert_eq!(member_dim.size, 2);
    assert!(time_dim.is_unlimited);
    assert!(member_dim.is_unlimited);

    let temp_values = file.read_variable::<f32>("temp").unwrap();
    assert_eq!(temp_values.shape(), &[3, 2]);
    assert_eq!(
        temp_values.as_slice_memory_order().unwrap(),
        &[10.0, 11.0, 20.0, 21.0, 30.0, 31.0]
    );
}

#[test]
fn writes_nc4_filtered_chunked_variable() {
    let mut builder = NcFileBuilder::new();
    let y = builder.add_dimension("y", 5).unwrap();
    let x = builder.add_dimension("x", 6).unwrap();
    let temp = builder.add_variable::<i32>("temp", &[y, x]).unwrap();
    builder.set_variable_chunking(temp, vec![2, 3]).unwrap();
    builder.set_variable_deflate(temp, Some(6), true).unwrap();
    builder.set_variable_fletcher32(temp, true).unwrap();
    let values = (0_i32..30).map(|value| value * 257).collect::<Vec<_>>();
    builder.write_variable(temp, &values).unwrap();

    let (_format, bytes) = builder
        .to_vec(NcWriteOptions {
            format: NcWriteFormat::Nc4,
        })
        .unwrap();

    let file = NcFile::from_bytes(&bytes).unwrap();
    let decoded = file.read_variable::<i32>("temp").unwrap();
    assert_eq!(decoded.shape(), &[5, 6]);
    assert_eq!(decoded.as_slice_memory_order().unwrap(), values.as_slice());
}

#[test]
fn writes_nc4_filtered_chunked_string_variable() {
    let mut builder = NcFileBuilder::new();
    let y = builder.add_dimension("y", 2).unwrap();
    let x = builder.add_dimension("x", 3).unwrap();
    let name = builder.add_string_variable("name", &[y, x]).unwrap();
    builder.set_variable_chunking(name, vec![1, 2]).unwrap();
    builder.set_variable_deflate(name, Some(6), false).unwrap();
    builder
        .write_string_variable(
            name,
            &["alpha", "beta", "gamma", "delta", "epsilon", "zeta"],
        )
        .unwrap();

    let (_format, bytes) = builder
        .to_vec(NcWriteOptions {
            format: NcWriteFormat::Nc4,
        })
        .unwrap();

    let file = NcFile::from_bytes(&bytes).unwrap();
    assert_eq!(
        file.read_variable_as_strings("name").unwrap(),
        vec![
            "alpha".to_string(),
            "beta".to_string(),
            "gamma".to_string(),
            "delta".to_string(),
            "epsilon".to_string(),
            "zeta".to_string()
        ]
    );
}

#[test]
fn writes_nc4_filtered_chunked_vlen_sequence_variable() {
    let mut builder = NcFileBuilder::new();
    let y = builder.add_dimension("y", 2).unwrap();
    let x = builder.add_dimension("x", 2).unwrap();
    let dtype = NcType::VLen {
        base: Box::new(NcType::Short),
    };
    let ragged = builder
        .add_user_defined_variable("ragged", &[y, x], dtype)
        .unwrap();
    builder.set_variable_chunking(ragged, vec![1, 1]).unwrap();
    builder
        .set_variable_deflate(ragged, Some(6), false)
        .unwrap();
    let values = vec![
        vec![1_i16, 2],
        vec![3_i16],
        Vec::new(),
        vec![10_i16, 11, 12],
    ];
    builder.write_vlen_variable(ragged, &values).unwrap();

    let (_format, bytes) = builder
        .to_vec(NcWriteOptions {
            format: NcWriteFormat::Nc4,
        })
        .unwrap();

    let file = NcFile::from_bytes(&bytes).unwrap();
    let decoded = file.read_variable_user_defined("ragged").unwrap();
    assert_eq!(
        decoded.as_slice().unwrap(),
        &[
            netcdf_reader::NcValue::VLen(vec![
                netcdf_reader::NcValue::Short(1),
                netcdf_reader::NcValue::Short(2)
            ]),
            netcdf_reader::NcValue::VLen(vec![netcdf_reader::NcValue::Short(3)]),
            netcdf_reader::NcValue::VLen(Vec::new()),
            netcdf_reader::NcValue::VLen(vec![
                netcdf_reader::NcValue::Short(10),
                netcdf_reader::NcValue::Short(11),
                netcdf_reader::NcValue::Short(12)
            ])
        ]
    );
}

#[test]
fn rejects_classic_variable_storage_options() {
    let mut builder = NcFileBuilder::new();
    let x = builder.add_dimension("x", 4).unwrap();
    let value = builder.add_variable::<i32>("value", &[x]).unwrap();
    builder.set_variable_chunking(value, vec![2]).unwrap();
    builder.write_variable(value, &[1_i32, 2, 3, 4]).unwrap();

    let err = builder
        .to_vec(NcWriteOptions {
            format: NcWriteFormat::Classic,
        })
        .unwrap_err();
    assert!(err.to_string().contains("NetCDF-4 storage options"));
}

#[test]
fn rejects_nc4_classic_multiple_unlimited_dimensions() {
    let mut builder = NcFileBuilder::new();
    let time = builder.add_unlimited_dimension("time").unwrap();
    let member = builder.add_unlimited_dimension("member").unwrap();
    let time_var = builder.add_variable::<i32>("time", &[time]).unwrap();
    let member_var = builder.add_variable::<i32>("member", &[member]).unwrap();

    builder.write_variable(time_var, &[0_i32, 1]).unwrap();
    builder.write_variable(member_var, &[10_i32, 20]).unwrap();

    let err = builder
        .to_vec(NcWriteOptions {
            format: NcWriteFormat::Nc4Classic,
        })
        .unwrap_err();
    assert!(err.to_string().contains("at most one unlimited"));
}

#[test]
fn rejects_nc4_dimension_scale_name_conflict() {
    let mut builder = NcFileBuilder::new();
    let x = builder.add_dimension("x", 2).unwrap();
    let temp = builder.add_variable::<f32>("temp", &[x]).unwrap();
    builder.write_variable(temp, &[10.0_f32, 11.0]).unwrap();
    let x_scalar = builder.add_variable::<i32>("x", &[]).unwrap();
    builder.write_variable(x_scalar, &[5]).unwrap();

    let err = builder
        .to_vec(NcWriteOptions {
            format: NcWriteFormat::Nc4,
        })
        .unwrap_err();

    assert!(err.to_string().contains("hidden dimension-scale"));
}
