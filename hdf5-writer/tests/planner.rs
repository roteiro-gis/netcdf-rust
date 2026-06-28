use hdf5_writer::{DatasetBuilder, Hdf5Builder};

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
