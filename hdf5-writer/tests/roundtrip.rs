use std::io::Cursor;

use hdf5_reader::{Hdf5File, SliceInfo, SliceInfoElem};
use hdf5_writer::{
    AttributeBuilder, ByteOrder, CompoundField, DatasetBuilder, Datatype, EnumMember,
    FilterDescription, Hdf5Builder, Hdf5Writer, WriteOptions, FILTER_DEFLATE, FILTER_FLETCHER32,
    FILTER_SHUFFLE, UNLIMITED,
};

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
fn writes_nested_group_paths() {
    let x_values = [0_i32, 1];
    let temperatures = [280.0_f32, 281.5];
    let quality = [1_i16, 2];
    let plan = Hdf5Builder::new()
        .dataset(
            DatasetBuilder::typed_data("science/x", vec![2], &x_values)
                .unwrap()
                .attribute(AttributeBuilder::fixed_string("CLASS", "DIMENSION_SCALE")),
        )
        .dataset(
            DatasetBuilder::typed_data("science/temperature", vec![2], &temperatures)
                .unwrap()
                .attribute(AttributeBuilder::vlen_object_references(
                    "DIMENSION_LIST",
                    vec![vec!["science/x".to_string()]],
                )),
        )
        .dataset(DatasetBuilder::typed_data("science/quality/qc", vec![2], &quality).unwrap())
        .into_plan()
        .unwrap();

    let cursor = Hdf5Writer::new(Cursor::new(Vec::new()), WriteOptions::default())
        .finish(plan)
        .unwrap();
    let bytes = cursor.into_inner();

    let file = Hdf5File::from_bytes(&bytes).unwrap();
    let science = file.group("/science").unwrap();
    let quality_group = file.group("/science/quality").unwrap();
    assert_eq!(science.name(), "science");
    assert_eq!(quality_group.name(), "quality");
    assert!(file.dataset("/temperature").is_err());

    let x = file.dataset("/science/x").unwrap();
    let temperature = file.dataset("/science/temperature").unwrap();
    let qc = file.dataset("/science/quality/qc").unwrap();
    assert_eq!(
        x.read_array::<i32>()
            .unwrap()
            .as_slice_memory_order()
            .unwrap(),
        x_values
    );
    assert_eq!(
        temperature
            .read_array::<f32>()
            .unwrap()
            .as_slice_memory_order()
            .unwrap(),
        temperatures
    );
    assert_eq!(
        qc.read_array::<i16>()
            .unwrap()
            .as_slice_memory_order()
            .unwrap(),
        quality
    );

    let attr = temperature.attribute("DIMENSION_LIST").unwrap();
    let heap_addr = u64::from_le_bytes(attr.raw_data[4..12].try_into().unwrap());
    let heap_index = u32::from_le_bytes(attr.raw_data[12..16].try_into().unwrap()) as u16;
    let heap = hdf5_reader::global_heap::GlobalHeapCollection::parse_at_storage(
        file.storage(),
        heap_addr,
        file.superblock().offset_size,
        file.superblock().length_size,
    )
    .unwrap();
    let heap_object = heap.get_object(heap_index).unwrap();
    let refs = hdf5_reader::reference::read_object_references(
        &heap_object.data,
        file.superblock().offset_size,
    )
    .unwrap();
    assert_eq!(refs, vec![x.address()]);
}

#[test]
fn rejects_dataset_path_that_is_also_group() {
    let err = Hdf5Builder::new()
        .dataset(DatasetBuilder::typed_data("science", vec![1], &[1_i32]).unwrap())
        .dataset(DatasetBuilder::typed_data("science/temp", vec![1], &[2_i32]).unwrap())
        .into_plan()
        .unwrap_err();

    assert!(err.to_string().contains("implicit HDF5 group"));
}

#[test]
fn writes_fixed_string_arrays() {
    let plan = Hdf5Builder::new()
        .attribute(
            AttributeBuilder::fixed_string_vector("history", &["created", "updated"]).unwrap(),
        )
        .dataset(
            DatasetBuilder::fixed_string_data("labels", vec![3], &["red", "green", "blue"])
                .unwrap(),
        )
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
            .attribute("history")
            .unwrap()
            .read_strings()
            .unwrap(),
        vec!["created".to_string(), "updated".to_string()]
    );
    assert_eq!(
        file.dataset("/labels").unwrap().read_strings().unwrap(),
        vec!["red".to_string(), "green".to_string(), "blue".to_string()]
    );
}

#[test]
fn writes_enhanced_datatype_messages() {
    let bitfield_type = Datatype::Bitfield {
        size: 2,
        byte_order: ByteOrder::LittleEndian,
    };
    let opaque_type = Datatype::Opaque {
        size: 3,
        tag: "rgb".to_string(),
    };
    let array_type = Datatype::Array {
        base: Box::new(Datatype::FixedPoint {
            size: 2,
            signed: true,
            byte_order: ByteOrder::LittleEndian,
        }),
        dims: vec![2],
    };
    let enum_type = Datatype::Enum {
        base: Box::new(Datatype::FixedPoint {
            size: 1,
            signed: false,
            byte_order: ByteOrder::LittleEndian,
        }),
        members: vec![
            EnumMember {
                name: "OFF".to_string(),
                value: vec![0],
            },
            EnumMember {
                name: "ON".to_string(),
                value: vec![1],
            },
        ],
    };
    let compound_type = Datatype::Compound {
        size: 12,
        fields: vec![
            CompoundField {
                name: "id".to_string(),
                byte_offset: 0,
                datatype: Datatype::FixedPoint {
                    size: 4,
                    signed: true,
                    byte_order: ByteOrder::LittleEndian,
                },
            },
            CompoundField {
                name: "value".to_string(),
                byte_offset: 4,
                datatype: Datatype::FloatingPoint {
                    size: 8,
                    byte_order: ByteOrder::LittleEndian,
                },
            },
        ],
    };
    let compound_data = {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&7_i32.to_le_bytes());
        bytes.extend_from_slice(&3.5_f64.to_le_bytes());
        bytes
    };

    let plan = Hdf5Builder::new()
        .dataset(
            DatasetBuilder::new("flags", bitfield_type.clone(), vec![2])
                .raw_data([0x34, 0x12, 0xcd, 0xab]),
        )
        .dataset(
            DatasetBuilder::new("pixels", opaque_type.clone(), vec![2])
                .raw_data([0xff, 0x00, 0x00, 0x00, 0xff, 0x00]),
        )
        .dataset(
            DatasetBuilder::new("pairs", array_type.clone(), vec![2])
                .raw_data([1, 0, 2, 0, 3, 0, 4, 0]),
        )
        .dataset(DatasetBuilder::new("states", enum_type.clone(), vec![3]).raw_data([0, 1, 1]))
        .dataset(
            DatasetBuilder::new("records", compound_type.clone(), vec![1])
                .raw_data(compound_data.clone()),
        )
        .into_plan()
        .unwrap();

    let cursor = Hdf5Writer::new(Cursor::new(Vec::new()), WriteOptions::default())
        .finish(plan)
        .unwrap();
    let bytes = cursor.into_inner();

    let file = Hdf5File::from_bytes(&bytes).unwrap();
    assert_eq!(file.dataset("/flags").unwrap().dtype(), &bitfield_type);
    assert_eq!(
        file.dataset("/flags").unwrap().read_raw_bytes().unwrap(),
        vec![0x34, 0x12, 0xcd, 0xab]
    );
    assert_eq!(file.dataset("/pixels").unwrap().dtype(), &opaque_type);
    assert_eq!(
        file.dataset("/pixels").unwrap().read_raw_bytes().unwrap(),
        vec![0xff, 0x00, 0x00, 0x00, 0xff, 0x00]
    );
    assert_eq!(file.dataset("/pairs").unwrap().dtype(), &array_type);
    assert_eq!(
        file.dataset("/pairs").unwrap().read_raw_bytes().unwrap(),
        vec![1, 0, 2, 0, 3, 0, 4, 0]
    );
    assert_eq!(file.dataset("/states").unwrap().dtype(), &enum_type);
    assert_eq!(
        file.dataset("/states").unwrap().read_raw_bytes().unwrap(),
        vec![0, 1, 1]
    );
    assert_eq!(file.dataset("/records").unwrap().dtype(), &compound_type);
    assert_eq!(
        file.dataset("/records").unwrap().read_raw_bytes().unwrap(),
        compound_data
    );
}

#[test]
fn writes_vlen_string_dataset_backed_by_global_heap() {
    let plan = Hdf5Builder::new()
        .dataset(
            DatasetBuilder::vlen_string_data("names", vec![3], &["alpha", "beta", "gamma"])
                .unwrap(),
        )
        .into_plan()
        .unwrap();

    let cursor = Hdf5Writer::new(Cursor::new(Vec::new()), WriteOptions::default())
        .finish(plan)
        .unwrap();
    let bytes = cursor.into_inner();

    let file = Hdf5File::from_bytes(&bytes).unwrap();
    assert_eq!(
        file.dataset("/names").unwrap().read_strings().unwrap(),
        vec!["alpha".to_string(), "beta".to_string(), "gamma".to_string()]
    );
}

#[test]
fn writes_chunked_vlen_string_dataset_backed_by_global_heap() {
    let plan = Hdf5Builder::new()
        .dataset(
            DatasetBuilder::vlen_string_data(
                "names",
                vec![2, 2],
                &["alpha", "beta", "gamma", "delta"],
            )
            .unwrap()
            .chunked(vec![1, 2]),
        )
        .into_plan()
        .unwrap();

    let cursor = Hdf5Writer::new(Cursor::new(Vec::new()), WriteOptions::default())
        .finish(plan)
        .unwrap();
    let bytes = cursor.into_inner();

    let file = Hdf5File::from_bytes(&bytes).unwrap();
    let dataset = file.dataset("/names").unwrap();
    assert_eq!(dataset.chunks().unwrap(), vec![1, 2]);
    assert_eq!(
        dataset.read_strings().unwrap(),
        vec![
            "alpha".to_string(),
            "beta".to_string(),
            "gamma".to_string(),
            "delta".to_string()
        ]
    );
}

#[test]
fn writes_implicit_chunked_dataset() {
    let values = (0_i32..15).collect::<Vec<_>>();
    let plan = Hdf5Builder::new()
        .dataset(
            DatasetBuilder::typed_data("chunked", vec![3, 5], &values)
                .unwrap()
                .chunked(vec![2, 3]),
        )
        .into_plan()
        .unwrap();

    let cursor = Hdf5Writer::new(Cursor::new(Vec::new()), WriteOptions::default())
        .finish(plan)
        .unwrap();
    let bytes = cursor.into_inner();

    let file = Hdf5File::from_bytes(&bytes).unwrap();
    let dataset = file.dataset("/chunked").unwrap();
    assert_eq!(dataset.chunks().unwrap(), vec![2, 3]);
    let array = dataset.read_array::<i32>().unwrap();
    assert_eq!(array.shape(), &[3, 5]);
    assert_eq!(array.as_slice_memory_order().unwrap(), values.as_slice());
}

#[test]
fn writes_resizable_chunked_dataspace() {
    let values = (0_i32..15).collect::<Vec<_>>();
    let plan = Hdf5Builder::new()
        .dataset(
            DatasetBuilder::typed_data("stream", vec![3, 5], &values)
                .unwrap()
                .chunked(vec![2, 3])
                .max_shape(vec![UNLIMITED, 5]),
        )
        .into_plan()
        .unwrap();

    let cursor = Hdf5Writer::new(Cursor::new(Vec::new()), WriteOptions::default())
        .finish(plan)
        .unwrap();
    let bytes = cursor.into_inner();

    let file = Hdf5File::from_bytes(&bytes).unwrap();
    let dataset = file.dataset("/stream").unwrap();
    assert_eq!(dataset.shape(), &[3, 5]);
    assert_eq!(dataset.max_dims().unwrap(), &[UNLIMITED, 5]);
    assert_eq!(dataset.chunks().unwrap(), vec![2, 3]);
    let array = dataset.read_array::<i32>().unwrap();
    assert_eq!(array.as_slice_memory_order().unwrap(), values.as_slice());
}

#[test]
fn writes_empty_resizable_chunked_dataspace() {
    let values: [i32; 0] = [];
    let plan = Hdf5Builder::new()
        .dataset(
            DatasetBuilder::typed_data("stream", vec![0, 5], &values)
                .unwrap()
                .chunked(vec![2, 3])
                .max_shape(vec![UNLIMITED, 5]),
        )
        .into_plan()
        .unwrap();

    let cursor = Hdf5Writer::new(Cursor::new(Vec::new()), WriteOptions::default())
        .finish(plan)
        .unwrap();
    let bytes = cursor.into_inner();

    let file = Hdf5File::from_bytes(&bytes).unwrap();
    let dataset = file.dataset("/stream").unwrap();
    assert_eq!(dataset.shape(), &[0, 5]);
    assert_eq!(dataset.max_dims().unwrap(), &[UNLIMITED, 5]);
    assert_eq!(dataset.chunks().unwrap(), vec![2, 3]);
    let array = dataset.read_array::<i32>().unwrap();
    assert_eq!(array.shape(), &[0, 5]);
    assert!(array.as_slice_memory_order().unwrap().is_empty());
}

#[test]
fn writes_fill_value_for_unallocated_dataset() {
    let fill = (-9999_i32).to_le_bytes();
    let plan = Hdf5Builder::new()
        .dataset(
            DatasetBuilder::typed_with_order::<i32>("filled", vec![2, 3], ByteOrder::LittleEndian)
                .fill_value(fill),
        )
        .into_plan()
        .unwrap();

    let cursor = Hdf5Writer::new(Cursor::new(Vec::new()), WriteOptions::default())
        .finish(plan)
        .unwrap();
    let bytes = cursor.into_inner();

    let file = Hdf5File::from_bytes(&bytes).unwrap();
    let dataset = file.dataset("/filled").unwrap();
    assert_eq!(dataset.shape(), &[2, 3]);
    assert_eq!(
        dataset.fill_value().unwrap().value.as_deref(),
        Some(&fill[..])
    );

    let array = dataset.read_array::<i32>().unwrap();
    assert_eq!(array.shape(), &[2, 3]);
    assert_eq!(
        array.as_slice_memory_order().unwrap(),
        &[-9999, -9999, -9999, -9999, -9999, -9999]
    );
}

#[test]
fn writes_compact_dataset_inline() {
    let values = [10_i16, 20, 30, 40];
    let plan = Hdf5Builder::new()
        .dataset(
            DatasetBuilder::typed_data("small", vec![2, 2], &values)
                .unwrap()
                .compact(),
        )
        .into_plan()
        .unwrap();

    let cursor = Hdf5Writer::new(Cursor::new(Vec::new()), WriteOptions::default())
        .finish(plan)
        .unwrap();
    let bytes = cursor.into_inner();

    let file = Hdf5File::from_bytes(&bytes).unwrap();
    let dataset = file.dataset("/small").unwrap();
    assert_eq!(dataset.shape(), &[2, 2]);
    let array = dataset.read_array::<i16>().unwrap();
    assert_eq!(array.as_slice_memory_order().unwrap(), values);
}

#[test]
fn rejects_oversized_compact_dataset() {
    let values = vec![0_u8; usize::from(u16::MAX) + 1];
    let plan = Hdf5Builder::new()
        .dataset(
            DatasetBuilder::typed_data("too_large", vec![values.len() as u64], &values)
                .unwrap()
                .compact(),
        )
        .into_plan()
        .unwrap();

    let err = Hdf5Writer::new(Cursor::new(Vec::new()), WriteOptions::default())
        .finish(plan)
        .unwrap_err();

    assert!(err.to_string().contains("compact HDF5 dataset data"));
}

#[test]
fn writes_single_chunk_deflate_dataset() {
    let values = (0_i32..12).collect::<Vec<_>>();
    let plan = Hdf5Builder::new()
        .dataset(
            DatasetBuilder::typed_data("compressed", vec![3, 4], &values)
                .unwrap()
                .chunked(vec![3, 4])
                .filter(FilterDescription {
                    id: FILTER_DEFLATE,
                    name: None,
                    client_data: vec![6],
                }),
        )
        .into_plan()
        .unwrap();

    let cursor = Hdf5Writer::new(Cursor::new(Vec::new()), WriteOptions::default())
        .finish(plan)
        .unwrap();
    let bytes = cursor.into_inner();

    let file = Hdf5File::from_bytes(&bytes).unwrap();
    let dataset = file.dataset("/compressed").unwrap();
    assert_eq!(dataset.chunks().unwrap(), vec![3, 4]);
    let array = dataset.read_array::<i32>().unwrap();
    assert_eq!(array.shape(), &[3, 4]);
    assert_eq!(array.as_slice_memory_order().unwrap(), values.as_slice());
}

#[test]
fn writes_multi_chunk_deflate_dataset_with_fixed_array_index() {
    let values = (0_i32..30).collect::<Vec<_>>();
    let plan = Hdf5Builder::new()
        .dataset(
            DatasetBuilder::typed_data("compressed", vec![5, 6], &values)
                .unwrap()
                .chunked(vec![2, 3])
                .filter(FilterDescription {
                    id: FILTER_DEFLATE,
                    name: None,
                    client_data: vec![6],
                }),
        )
        .into_plan()
        .unwrap();

    let cursor = Hdf5Writer::new(Cursor::new(Vec::new()), WriteOptions::default())
        .finish(plan)
        .unwrap();
    let bytes = cursor.into_inner();

    let file = Hdf5File::from_bytes(&bytes).unwrap();
    let dataset = file.dataset("/compressed").unwrap();
    assert_eq!(dataset.chunks().unwrap(), vec![2, 3]);
    let array = dataset.read_array::<i32>().unwrap();
    assert_eq!(array.shape(), &[5, 6]);
    assert_eq!(array.as_slice_memory_order().unwrap(), values.as_slice());

    let selection = SliceInfo {
        selections: vec![
            SliceInfoElem::Slice {
                start: 1,
                end: 5,
                step: 1,
            },
            SliceInfoElem::Slice {
                start: 2,
                end: 6,
                step: 1,
            },
        ],
    };
    let slice = dataset.read_slice::<i32>(&selection).unwrap();
    assert_eq!(slice.shape(), &[4, 4]);
    assert_eq!(
        slice.as_slice_memory_order().unwrap(),
        &[8, 9, 10, 11, 14, 15, 16, 17, 20, 21, 22, 23, 26, 27, 28, 29]
    );
}

#[test]
fn writes_shuffle_deflate_dataset() {
    let values = (0_i32..30).map(|value| value * 257).collect::<Vec<_>>();
    let plan = Hdf5Builder::new()
        .dataset(
            DatasetBuilder::typed_data("compressed", vec![5, 6], &values)
                .unwrap()
                .chunked(vec![2, 3])
                .filter(FilterDescription {
                    id: FILTER_SHUFFLE,
                    name: None,
                    client_data: Vec::new(),
                })
                .filter(FilterDescription {
                    id: FILTER_DEFLATE,
                    name: None,
                    client_data: vec![6],
                }),
        )
        .into_plan()
        .unwrap();

    let cursor = Hdf5Writer::new(Cursor::new(Vec::new()), WriteOptions::default())
        .finish(plan)
        .unwrap();
    let bytes = cursor.into_inner();

    let file = Hdf5File::from_bytes(&bytes).unwrap();
    let dataset = file.dataset("/compressed").unwrap();
    assert_eq!(dataset.chunks().unwrap(), vec![2, 3]);
    let array = dataset.read_array::<i32>().unwrap();
    assert_eq!(array.shape(), &[5, 6]);
    assert_eq!(array.as_slice_memory_order().unwrap(), values.as_slice());
}

#[test]
fn writes_fletcher32_dataset() {
    let values = (0_i16..35).map(|value| value * 11).collect::<Vec<_>>();
    let plan = Hdf5Builder::new()
        .dataset(
            DatasetBuilder::typed_data("checked", vec![5, 7], &values)
                .unwrap()
                .chunked(vec![2, 3])
                .filter(FilterDescription {
                    id: FILTER_FLETCHER32,
                    name: None,
                    client_data: Vec::new(),
                }),
        )
        .into_plan()
        .unwrap();

    let cursor = Hdf5Writer::new(Cursor::new(Vec::new()), WriteOptions::default())
        .finish(plan)
        .unwrap();
    let bytes = cursor.into_inner();

    let file = Hdf5File::from_bytes(&bytes).unwrap();
    let dataset = file.dataset("/checked").unwrap();
    assert_eq!(dataset.chunks().unwrap(), vec![2, 3]);
    let array = dataset.read_array::<i16>().unwrap();
    assert_eq!(array.shape(), &[5, 7]);
    assert_eq!(array.as_slice_memory_order().unwrap(), values.as_slice());
}

#[test]
fn writes_shuffle_deflate_fletcher32_pipeline() {
    let values = (0_i32..42).map(|value| value * 4099).collect::<Vec<_>>();
    let plan = Hdf5Builder::new()
        .dataset(
            DatasetBuilder::typed_data("filtered", vec![6, 7], &values)
                .unwrap()
                .chunked(vec![3, 4])
                .filter(FilterDescription {
                    id: FILTER_SHUFFLE,
                    name: None,
                    client_data: Vec::new(),
                })
                .filter(FilterDescription {
                    id: FILTER_DEFLATE,
                    name: None,
                    client_data: vec![4],
                })
                .filter(FilterDescription {
                    id: FILTER_FLETCHER32,
                    name: None,
                    client_data: Vec::new(),
                }),
        )
        .into_plan()
        .unwrap();

    let cursor = Hdf5Writer::new(Cursor::new(Vec::new()), WriteOptions::default())
        .finish(plan)
        .unwrap();
    let bytes = cursor.into_inner();

    let file = Hdf5File::from_bytes(&bytes).unwrap();
    let dataset = file.dataset("/filtered").unwrap();
    assert_eq!(dataset.chunks().unwrap(), vec![3, 4]);
    let array = dataset.read_array::<i32>().unwrap();
    assert_eq!(array.shape(), &[6, 7]);
    assert_eq!(array.as_slice_memory_order().unwrap(), values.as_slice());
}

#[test]
fn rejects_fletcher32_client_data() {
    let values = (0_i16..12).collect::<Vec<_>>();
    let plan = Hdf5Builder::new()
        .dataset(
            DatasetBuilder::typed_data("bad_checksum", vec![3, 4], &values)
                .unwrap()
                .chunked(vec![3, 4])
                .filter(FilterDescription {
                    id: FILTER_FLETCHER32,
                    name: None,
                    client_data: vec![1],
                }),
        )
        .into_plan()
        .unwrap();

    let err = Hdf5Writer::new(Cursor::new(Vec::new()), WriteOptions::default())
        .finish(plan)
        .unwrap_err();
    assert!(err.to_string().contains("Fletcher32 filter"));
}

#[test]
fn rejects_shuffle_with_wrong_element_size() {
    let values = (0_i32..12).collect::<Vec<_>>();
    let plan = Hdf5Builder::new()
        .dataset(
            DatasetBuilder::typed_data("bad_shuffle", vec![3, 4], &values)
                .unwrap()
                .chunked(vec![3, 4])
                .filter(FilterDescription {
                    id: FILTER_SHUFFLE,
                    name: None,
                    client_data: vec![8],
                }),
        )
        .into_plan()
        .unwrap();

    let err = Hdf5Writer::new(Cursor::new(Vec::new()), WriteOptions::default())
        .finish(plan)
        .unwrap_err();
    assert!(err.to_string().contains("shuffle filter element size"));
}

#[test]
fn writes_vlen_string_attribute_backed_by_global_heap() {
    let plan = Hdf5Builder::new()
        .attribute(AttributeBuilder::vlen_strings("history", &["created", "updated"]).unwrap())
        .dataset(DatasetBuilder::typed_data("data", vec![1], &[1_i32]).unwrap())
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
            .attribute("history")
            .unwrap()
            .read_strings()
            .unwrap(),
        vec!["created".to_string(), "updated".to_string()]
    );
}

#[test]
fn writes_vlen_object_reference_attribute_backed_by_global_heap() {
    let scale = DatasetBuilder::typed_data("x", vec![3], &[0_i32, 0, 0])
        .unwrap()
        .attribute(AttributeBuilder::fixed_string("CLASS", "DIMENSION_SCALE"))
        .attribute(AttributeBuilder::fixed_string("NAME", "x"));
    let values = [10.0_f32, 11.0, 12.0];
    let data = DatasetBuilder::typed_data("temp", vec![3], &values)
        .unwrap()
        .attribute(AttributeBuilder::vlen_object_references(
            "DIMENSION_LIST",
            vec![vec!["x".to_string()]],
        ));
    let plan = Hdf5Builder::new()
        .dataset(scale)
        .dataset(data)
        .into_plan()
        .unwrap();

    let cursor = Hdf5Writer::new(Cursor::new(Vec::new()), WriteOptions::default())
        .finish(plan)
        .unwrap();
    let bytes = cursor.into_inner();

    let file = Hdf5File::from_bytes(&bytes).unwrap();
    let scale = file.dataset("/x").unwrap();
    let data = file.dataset("/temp").unwrap();
    let attr = data.attribute("DIMENSION_LIST").unwrap();
    assert_eq!(attr.shape, vec![1]);
    assert_eq!(attr.raw_data.len(), 16);

    let seq_len = u32::from_le_bytes(attr.raw_data[0..4].try_into().unwrap());
    let heap_addr = u64::from_le_bytes(attr.raw_data[4..12].try_into().unwrap());
    let heap_index = u32::from_le_bytes(attr.raw_data[12..16].try_into().unwrap()) as u16;
    assert_eq!(seq_len, 1);

    let heap = hdf5_reader::global_heap::GlobalHeapCollection::parse_at_storage(
        file.storage(),
        heap_addr,
        file.superblock().offset_size,
        file.superblock().length_size,
    )
    .unwrap();
    let heap_object = heap.get_object(heap_index).unwrap();
    let refs = hdf5_reader::reference::read_object_references(
        &heap_object.data,
        file.superblock().offset_size,
    )
    .unwrap();
    assert_eq!(refs, vec![scale.address()]);
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
