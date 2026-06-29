use std::io::Cursor;

use hdf5_reader::Hdf5File;
use hdf5_writer::{AttributeBuilder, DatasetBuilder, Hdf5Builder, Hdf5Writer, WriteOptions};

#[test]
fn writes_contiguous_f32_dataset_readable_by_hdf5_reader() {
    let values = [1.25_f32, 2.5, 3.75, 4.5, 5.25, 6.5];
    let dataset = DatasetBuilder::typed_data("data", vec![2, 3], &values)
        .unwrap()
        .attribute(AttributeBuilder::fixed_string("units", "kelvin"))
        .attribute(AttributeBuilder::scalar("_FillValue", -9999.0_f32).unwrap());
    let plan = Hdf5Builder::new()
        .attribute(AttributeBuilder::fixed_string("title", "roundtrip"))
        .dataset(dataset)
        .into_plan()
        .unwrap();

    let cursor = Hdf5Writer::new(Cursor::new(Vec::new()), WriteOptions::default())
        .finish(plan)
        .unwrap();
    let bytes = cursor.into_inner();

    let file = Hdf5File::from_bytes(&bytes).unwrap();
    assert_eq!(
        file.root_group()
            .unwrap()
            .attribute("title")
            .unwrap()
            .read_string()
            .unwrap(),
        "roundtrip"
    );
    let dataset = file.dataset("/data").unwrap();
    assert_eq!(dataset.shape(), &[2, 3]);
    assert_eq!(
        dataset.attribute("units").unwrap().read_string().unwrap(),
        "kelvin"
    );
    assert_eq!(
        dataset
            .attribute("_FillValue")
            .unwrap()
            .read_scalar::<f32>()
            .unwrap(),
        -9999.0
    );

    let array = dataset.read_array::<f32>().unwrap();
    assert_eq!(array.as_slice_memory_order().unwrap(), values);
}

#[test]
fn writes_multiple_root_datasets() {
    let floats = [1.0_f64, 2.0, 4.0, 8.0];
    let ints = [7_i32, 8, 9];
    let plan = Hdf5Builder::new()
        .dataset(DatasetBuilder::typed_data("float_values", vec![4], &floats).unwrap())
        .dataset(DatasetBuilder::typed_data("int_values", vec![3], &ints).unwrap())
        .into_plan()
        .unwrap();

    let cursor = Hdf5Writer::new(Cursor::new(Vec::new()), WriteOptions::default())
        .finish(plan)
        .unwrap();
    let bytes = cursor.into_inner();

    let file = Hdf5File::from_bytes(&bytes).unwrap();
    let read_floats = file
        .dataset("/float_values")
        .unwrap()
        .read_array::<f64>()
        .unwrap();
    let read_ints = file
        .dataset("/int_values")
        .unwrap()
        .read_array::<i32>()
        .unwrap();

    assert_eq!(read_floats.as_slice_memory_order().unwrap(), floats);
    assert_eq!(read_ints.as_slice_memory_order().unwrap(), ints);
}

#[test]
fn binary_emission_requires_data() {
    let plan = Hdf5Builder::new()
        .dataset(DatasetBuilder::typed::<f32>("data", vec![2, 3]))
        .into_plan()
        .unwrap();

    let err = Hdf5Writer::new(Cursor::new(Vec::new()), WriteOptions::default())
        .finish(plan)
        .unwrap_err();

    assert!(err.to_string().contains("has no raw data"));
}
