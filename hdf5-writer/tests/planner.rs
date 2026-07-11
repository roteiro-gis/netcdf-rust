use hdf5_writer::{AttributeBuilder, DatasetBuilder, Hdf5Builder};

#[test]
fn validates_basic_contiguous_dataset_plan() {
    let plan = Hdf5Builder::new()
        .dataset(DatasetBuilder::typed::<f32>("temperature", vec![2, 3]))
        .into_plan()
        .unwrap();

    assert_eq!(plan.datasets().len(), 1);
    assert_eq!(plan.datasets()[0].shape(), &[2, 3]);
}

#[test]
fn validates_group_attribute_plan() {
    let plan = Hdf5Builder::new()
        .group_attribute(
            "science",
            AttributeBuilder::fixed_string("title", "profile"),
        )
        .into_plan()
        .unwrap();

    assert_eq!(plan.group_attributes().len(), 1);
    assert_eq!(plan.group_attributes()[0].path(), "science");
    assert_eq!(plan.group_attributes()[0].attribute().name(), "title");
}

#[test]
fn rejects_duplicate_group_attributes() {
    let err = Hdf5Builder::new()
        .group_attribute("science", AttributeBuilder::fixed_string("title", "one"))
        .group_attribute("science", AttributeBuilder::fixed_string("title", "two"))
        .into_plan()
        .unwrap_err();

    assert!(err.to_string().contains("duplicate attribute"));
}

#[test]
fn rejects_filters_on_non_chunked_dataset() {
    let err = Hdf5Builder::new()
        .dataset(
            DatasetBuilder::typed::<f32>("temperature", vec![2, 3]).filter(
                hdf5_writer::FilterDescription {
                    id: hdf5_writer::FILTER_DEFLATE,
                    name: None,
                    client_data: vec![6],
                },
            ),
        )
        .into_plan()
        .unwrap_err();

    assert!(err.to_string().contains("filtered HDF5 datasets"));
}

#[test]
fn rejects_resizable_non_chunked_dataset() {
    let err = Hdf5Builder::new()
        .dataset(DatasetBuilder::typed::<f32>("temperature", vec![2, 3]).max_shape(vec![4, 3]))
        .into_plan()
        .unwrap_err();

    assert!(err.to_string().contains("must use chunked layout"));
}

#[test]
fn rejects_fill_value_with_wrong_element_size() {
    let err = Hdf5Builder::new()
        .dataset(DatasetBuilder::typed::<i32>("temperature", vec![2, 3]).fill_value(vec![0]))
        .into_plan()
        .unwrap_err();

    assert!(err.to_string().contains("fill value byte length"));
}

#[test]
fn wrong_length_data_returns_structured_mismatch() {
    // Writing fewer values than the shape requires yields a matchable
    // DataLengthMismatch, not an opaque string error. The numeric data path
    // reports byte counts (6 elements * 4 bytes expected, 3 * 4 supplied).
    let err = DatasetBuilder::typed_data("grid", vec![2, 3], &[0_i32, 1, 2]).unwrap_err();
    assert!(matches!(
        err,
        hdf5_writer::Error::DataLengthMismatch {
            expected: 24,
            actual: 12,
            ..
        }
    ));
}
