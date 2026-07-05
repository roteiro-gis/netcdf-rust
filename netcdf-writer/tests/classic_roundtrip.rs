use netcdf_reader::NcFile;
use netcdf_writer::{NcFileBuilder, NcWriteOptions};

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
