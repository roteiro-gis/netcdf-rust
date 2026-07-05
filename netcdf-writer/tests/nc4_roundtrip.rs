#![cfg(feature = "netcdf4")]

use netcdf_reader::{NcFile, NcFormat};
use netcdf_writer::{NcAttrValue, NcFileBuilder, NcWriteFormat, NcWriteOptions};

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
