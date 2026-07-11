use netcdf_reader::NcFile;
use netcdf_writer::{
    NcAttrValue, NcFileBuilder, NcSliceInfo, NcSliceInfoElem, NcWriteOptions, NC_FILL_INT,
};

#[test]
fn writes_cdf1_fixed_variables() {
    let mut builder = NcFileBuilder::new();
    let y = builder.add_dimension("y", 2).unwrap();
    let x = builder.add_dimension("x", 3).unwrap();
    builder
        .add_attribute(
            "title",
            netcdf_writer::NcAttrValue::Chars("roundtrip".to_string()),
        )
        .unwrap();
    let temp = builder.add_variable::<f32>("temp", &[y, x]).unwrap();
    builder
        .add_variable_attribute(
            temp,
            "units",
            netcdf_writer::NcAttrValue::Chars("K".to_string()),
        )
        .unwrap();
    builder
        .write_variable(temp, &[280.0_f32, 281.0, 282.0, 283.0, 284.0, 285.0])
        .unwrap();

    let (format, bytes) = builder.to_vec(NcWriteOptions::default()).unwrap();
    assert_eq!(format, netcdf_reader::NcFormat::Classic);

    let file = NcFile::from_bytes(&bytes).unwrap();
    assert_eq!(file.dimension("x").unwrap().size, 3);
    assert_eq!(file.dimension("y").unwrap().size, 2);
    assert_eq!(
        file.global_attribute("title")
            .unwrap()
            .value
            .as_string()
            .unwrap(),
        "roundtrip"
    );
    let data = file.read_variable::<f32>("temp").unwrap();
    assert_eq!(data.shape(), &[2, 3]);
    assert_eq!(
        data.as_slice_memory_order().unwrap(),
        &[280.0, 281.0, 282.0, 283.0, 284.0, 285.0]
    );
}

#[test]
fn writes_classic_fill_value_attribute() {
    let mut builder = NcFileBuilder::new();
    let y = builder.add_dimension("y", 2).unwrap();
    let x = builder.add_dimension("x", 3).unwrap();
    let temp = builder.add_variable::<i16>("temp", &[y, x]).unwrap();
    builder.set_variable_fill_value(temp, -999_i16).unwrap();
    builder
        .write_variable(temp, &[1_i16, 2, -999, 4, 5, 6])
        .unwrap();

    let (format, bytes) = builder.to_vec(NcWriteOptions::default()).unwrap();
    assert_eq!(format, netcdf_reader::NcFormat::Classic);

    let file = NcFile::from_bytes(&bytes).unwrap();
    assert_eq!(
        file.variable("temp")
            .unwrap()
            .attribute("_FillValue")
            .unwrap()
            .value,
        NcAttrValue::Shorts(vec![-999])
    );
    let data = file.read_variable::<i16>("temp").unwrap();
    assert_eq!(
        data.as_slice_memory_order().unwrap(),
        &[1, 2, -999, 4, 5, 6]
    );
}

#[test]
fn writes_classic_variable_slice_with_fill_initialization() {
    let mut builder = NcFileBuilder::new();
    let y = builder.add_dimension("y", 3).unwrap();
    let x = builder.add_dimension("x", 4).unwrap();
    let temp = builder.add_variable::<i16>("temp", &[y, x]).unwrap();
    builder.set_variable_fill_value(temp, -999_i16).unwrap();
    builder
        .write_variable_slice(
            temp,
            &NcSliceInfo {
                selections: vec![
                    NcSliceInfoElem::Index(1),
                    NcSliceInfoElem::Slice {
                        start: 1,
                        end: 4,
                        step: 1,
                    },
                ],
            },
            &[10_i16, 11, 12],
        )
        .unwrap();

    let (format, bytes) = builder.to_vec(NcWriteOptions::default()).unwrap();
    assert_eq!(format, netcdf_reader::NcFormat::Classic);

    let file = NcFile::from_bytes(&bytes).unwrap();
    let data = file.read_variable::<i16>("temp").unwrap();
    assert_eq!(data.shape(), &[3, 4]);
    assert_eq!(
        data.as_slice_memory_order().unwrap(),
        &[-999, -999, -999, -999, -999, 10, 11, 12, -999, -999, -999, -999]
    );
}

#[test]
fn writes_classic_variable_slice_with_default_fill_initialization() {
    let mut builder = NcFileBuilder::new();
    let y = builder.add_dimension("y", 2).unwrap();
    let x = builder.add_dimension("x", 3).unwrap();
    let values = builder.add_variable::<i32>("values", &[y, x]).unwrap();
    builder
        .write_variable_slice(
            values,
            &NcSliceInfo {
                selections: vec![
                    NcSliceInfoElem::Index(0),
                    NcSliceInfoElem::Slice {
                        start: 1,
                        end: 3,
                        step: 1,
                    },
                ],
            },
            &[5_i32, 6],
        )
        .unwrap();

    let (format, bytes) = builder.to_vec(NcWriteOptions::default()).unwrap();
    assert_eq!(format, netcdf_reader::NcFormat::Classic);

    let file = NcFile::from_bytes(&bytes).unwrap();
    let data = file.read_variable::<i32>("values").unwrap();
    assert_eq!(
        data.as_slice_memory_order().unwrap(),
        &[NC_FILL_INT, 5, 6, NC_FILL_INT, NC_FILL_INT, NC_FILL_INT]
    );
}

#[test]
fn writes_classic_char_variable_slice() {
    let mut builder = NcFileBuilder::new();
    let station = builder.add_dimension("station", 2).unwrap();
    let strlen = builder.add_dimension("strlen", 5).unwrap();
    let names = builder
        .add_char_variable("station_name", &[station, strlen])
        .unwrap();
    builder
        .write_char_variable_slice(
            names,
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
            names,
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

    let (format, bytes) = builder.to_vec(NcWriteOptions::default()).unwrap();
    assert_eq!(format, netcdf_reader::NcFormat::Classic);

    let file = NcFile::from_bytes(&bytes).unwrap();
    assert_eq!(
        file.read_variable_as_strings("station_name").unwrap(),
        vec!["alpha".to_string(), "beta".to_string()]
    );
}

#[test]
fn writes_classic_char_variable_strings() {
    let mut builder = NcFileBuilder::new();
    let station = builder.add_dimension("station", 2).unwrap();
    let strlen = builder.add_dimension("strlen", 5).unwrap();
    let names = builder
        .add_char_variable("station_name", &[station, strlen])
        .unwrap();
    builder
        .write_char_variable_strings(names, &["alpha", "beta"])
        .unwrap();

    let (format, bytes) = builder.to_vec(NcWriteOptions::default()).unwrap();
    assert_eq!(format, netcdf_reader::NcFormat::Classic);

    let file = NcFile::from_bytes(&bytes).unwrap();
    assert_eq!(
        file.read_variable_as_strings("station_name").unwrap(),
        vec!["alpha".to_string(), "beta".to_string()]
    );
}

#[test]
fn writes_classic_char_variable_string_slices() {
    let mut builder = NcFileBuilder::new();
    let station = builder.add_dimension("station", 2).unwrap();
    let strlen = builder.add_dimension("strlen", 5).unwrap();
    let names = builder
        .add_char_variable("station_name", &[station, strlen])
        .unwrap();
    builder
        .write_char_variable_strings_slice(
            names,
            &NcSliceInfo {
                selections: vec![NcSliceInfoElem::Index(1)],
            },
            &["beta"],
        )
        .unwrap();
    builder
        .write_char_variable_strings_slice(
            names,
            &NcSliceInfo {
                selections: vec![NcSliceInfoElem::Index(0)],
            },
            &["alpha"],
        )
        .unwrap();

    let (format, bytes) = builder.to_vec(NcWriteOptions::default()).unwrap();
    assert_eq!(format, netcdf_reader::NcFormat::Classic);

    let file = NcFile::from_bytes(&bytes).unwrap();
    assert_eq!(
        file.read_variable_as_strings("station_name").unwrap(),
        vec!["alpha".to_string(), "beta".to_string()]
    );
}

#[test]
fn writes_classic_unlimited_variable_slice_append() {
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

    let (format, bytes) = builder.to_vec(NcWriteOptions::default()).unwrap();
    assert_eq!(format, netcdf_reader::NcFormat::Classic);

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
fn auto_promotes_unsigned_and_u64_to_cdf5() {
    let mut builder = NcFileBuilder::new();
    let n = builder.add_dimension("n", 3).unwrap();
    let values = builder.add_variable::<u64>("values", &[n]).unwrap();
    builder
        .add_attribute(
            "flag_masks",
            netcdf_writer::NcAttrValue::UShorts(vec![1, 2]),
        )
        .unwrap();
    builder.write_variable(values, &[1, 2, u64::MAX]).unwrap();

    let (format, bytes) = builder.to_vec(NcWriteOptions::default()).unwrap();
    assert_eq!(format, netcdf_reader::NcFormat::Cdf5);

    let file = NcFile::from_bytes(&bytes).unwrap();
    assert_eq!(file.format(), netcdf_reader::NcFormat::Cdf5);
    let data = file.read_variable::<u64>("values").unwrap();
    assert_eq!(data.as_slice_memory_order().unwrap(), &[1, 2, u64::MAX]);
}

#[test]
fn writes_interleaved_record_variables() {
    let mut builder = NcFileBuilder::new();
    let time = builder.add_unlimited_dimension("time").unwrap();
    let station = builder.add_dimension("station", 2).unwrap();
    let temp = builder
        .add_variable::<f32>("temp", &[time, station])
        .unwrap();
    let qc = builder.add_variable::<i16>("qc", &[time]).unwrap();

    builder
        .write_variable(temp, &[10.0_f32, 11.0, 20.0, 21.0, 30.0, 31.0])
        .unwrap();
    builder.write_variable(qc, &[1_i16, 2, 3]).unwrap();

    let (format, bytes) = builder.to_vec(NcWriteOptions::default()).unwrap();
    assert_eq!(format, netcdf_reader::NcFormat::Classic);

    let file = NcFile::from_bytes(&bytes).unwrap();
    assert_eq!(file.dimension("time").unwrap().size, 3);
    let temp_data = file.read_variable::<f32>("temp").unwrap();
    assert_eq!(temp_data.shape(), &[3, 2]);
    assert_eq!(
        temp_data.as_slice_memory_order().unwrap(),
        &[10.0, 11.0, 20.0, 21.0, 30.0, 31.0]
    );
    let qc_data = file.read_variable::<i16>("qc").unwrap();
    assert_eq!(qc_data.as_slice_memory_order().unwrap(), &[1, 2, 3]);
}

#[test]
fn rejects_multiple_unlimited_dimensions_for_classic_writes() {
    let mut builder = NcFileBuilder::new();
    let time = builder.add_unlimited_dimension("time").unwrap();
    let member = builder.add_unlimited_dimension("member").unwrap();
    let time_var = builder.add_variable::<i32>("time", &[time]).unwrap();
    let member_var = builder.add_variable::<i32>("member", &[member]).unwrap();

    builder.write_variable(time_var, &[0_i32, 1]).unwrap();
    builder.write_variable(member_var, &[10_i32, 20]).unwrap();

    let err = builder.to_vec(NcWriteOptions::classic()).unwrap_err();
    assert!(err.to_string().contains("at most one unlimited"));
}

#[test]
fn rejects_group_paths_for_classic_writes() {
    let mut builder = NcFileBuilder::new();
    let dim = builder.add_dimension_path("science/x", 2).unwrap();
    let variable = builder
        .add_variable_path::<i32>("science/value", &[dim])
        .unwrap();
    builder.write_variable(variable, &[1_i32, 2]).unwrap();

    let err = builder.to_vec(NcWriteOptions::classic()).unwrap_err();
    assert!(matches!(err, netcdf_writer::Error::RequiresNetcdf4 { .. }));
}

#[test]
fn rejects_group_attributes_for_classic_writes() {
    let mut builder = NcFileBuilder::new();
    builder
        .add_group_attribute(
            "science",
            "title",
            NcAttrValue::Chars("enhanced model".to_string()),
        )
        .unwrap();

    let err = builder.to_vec(NcWriteOptions::classic()).unwrap_err();
    assert!(matches!(err, netcdf_writer::Error::RequiresNetcdf4 { .. }));
}

/// `/tmp` reference bytes produced by netcdf-c (via netCDF4-python) for a
/// CDF-1 file with exactly one odd-sized record variable:
/// `time` unlimited (3 records), `x = 3`, `short s(time, x)`. The classic
/// spec packs the 6-byte records without padding in this special case while
/// the header vsize stays padded to 8.
const NETCDF_C_SINGLE_RECORD: &[u8] = &[
    0x43, 0x44, 0x46, 0x01, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x0a, 0x00, 0x00, 0x00, 0x02,
    0x00, 0x00, 0x00, 0x04, 0x74, 0x69, 0x6d, 0x65, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
    0x78, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x0b, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x73, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x60,
    0x00, 0x01, 0x00, 0x02, 0x00, 0x03, 0x00, 0x04, 0x00, 0x05, 0x00, 0x06, 0x00, 0x07, 0x00, 0x08,
    0x00, 0x09,
];

/// netcdf-c reference bytes for a CDF-1 file with three record variables
/// (`byte b(time)`, `short s(time)`, `int i(time)`, 2 records): every record
/// slab is padded to 4 bytes with the type's default fill bytes.
const NETCDF_C_MULTI_RECORD: &[u8] = &[
    0x43, 0x44, 0x46, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x0a, 0x00, 0x00, 0x00, 0x01,
    0x00, 0x00, 0x00, 0x04, 0x74, 0x69, 0x6d, 0x65, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0b, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x01,
    0x62, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x98,
    0x00, 0x00, 0x00, 0x01, 0x73, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x04,
    0x00, 0x00, 0x00, 0x9c, 0x00, 0x00, 0x00, 0x01, 0x69, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04,
    0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0xa0, 0x01, 0x81, 0x81, 0x81, 0x00, 0x0a, 0x80, 0x01,
    0x00, 0x00, 0x00, 0x64, 0x02, 0x81, 0x81, 0x81, 0x00, 0x14, 0x80, 0x01, 0x00, 0x00, 0x00, 0xc8,
];

#[test]
fn single_odd_sized_record_variable_matches_netcdf_c_bytes() {
    let mut builder = NcFileBuilder::new();
    let time = builder.add_unlimited_dimension("time").unwrap();
    let x = builder.add_dimension("x", 3).unwrap();
    let s = builder.add_variable::<i16>("s", &[time, x]).unwrap();
    builder
        .write_variable(s, &[1_i16, 2, 3, 4, 5, 6, 7, 8, 9])
        .unwrap();

    let (_, bytes) = builder.to_vec(NcWriteOptions::classic()).unwrap();
    assert_eq!(bytes, NETCDF_C_SINGLE_RECORD);
}

#[test]
fn multiple_record_variables_match_netcdf_c_bytes() {
    let mut builder = NcFileBuilder::new();
    let time = builder.add_unlimited_dimension("time").unwrap();
    let b = builder.add_variable::<i8>("b", &[time]).unwrap();
    let s = builder.add_variable::<i16>("s", &[time]).unwrap();
    let i = builder.add_variable::<i32>("i", &[time]).unwrap();
    builder.write_variable(b, &[1_i8, 2]).unwrap();
    builder.write_variable(s, &[10_i16, 20]).unwrap();
    builder.write_variable(i, &[100_i32, 200]).unwrap();

    let (_, bytes) = builder.to_vec(NcWriteOptions::classic()).unwrap();
    assert_eq!(bytes, NETCDF_C_MULTI_RECORD);
}

#[test]
fn reads_netcdf_c_single_odd_sized_record_variable() {
    let file = NcFile::from_bytes(NETCDF_C_SINGLE_RECORD).unwrap();
    let data = file.read_variable::<i16>("s").unwrap();
    assert_eq!(data.shape(), &[3, 3]);
    assert_eq!(
        data.as_slice_memory_order().unwrap(),
        &[1, 2, 3, 4, 5, 6, 7, 8, 9]
    );
}

#[test]
fn reads_netcdf_c_multiple_record_variables() {
    let file = NcFile::from_bytes(NETCDF_C_MULTI_RECORD).unwrap();
    assert_eq!(
        file.read_variable::<i8>("b")
            .unwrap()
            .as_slice_memory_order()
            .unwrap(),
        &[1, 2]
    );
    assert_eq!(
        file.read_variable::<i16>("s")
            .unwrap()
            .as_slice_memory_order()
            .unwrap(),
        &[10, 20]
    );
    assert_eq!(
        file.read_variable::<i32>("i")
            .unwrap()
            .as_slice_memory_order()
            .unwrap(),
        &[100, 200]
    );
}

#[test]
fn cdf5_only_data_reports_format_capacity_exceeded() {
    // Unsigned 64-bit data cannot be represented in CDF-1; requesting it
    // yields a matchable FormatCapacityExceeded so callers can retry as CDF-5.
    let mut builder = NcFileBuilder::new();
    let x = builder.add_dimension("x", 2).unwrap();
    let v = builder.add_variable::<u64>("v", &[x]).unwrap();
    builder.write_variable(v, &[1_u64, 2]).unwrap();

    let err = builder.to_vec(NcWriteOptions::classic()).unwrap_err();
    assert!(matches!(
        err,
        netcdf_writer::Error::FormatCapacityExceeded { .. }
    ));

    // The same schema succeeds as CDF-5.
    assert!(builder.to_vec(NcWriteOptions::cdf5()).is_ok());
}
