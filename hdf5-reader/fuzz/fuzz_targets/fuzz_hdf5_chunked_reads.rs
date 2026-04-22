#![no_main]

use hdf5_reader::{Dataset, Hdf5File, SliceInfo, SliceInfoElem};
use libfuzzer_sys::fuzz_target;

const SIMPLE_CHUNKED: &[u8] = include_bytes!("../../../testdata/hdf5/simple_chunked_deflate.h5");
const FIXED_ARRAY_CHUNKED: &[u8] = include_bytes!("../../../testdata/hdf5/fixed_array_chunked.h5");

fn mutated_fixture(data: &[u8]) -> Vec<u8> {
    let base = if data.first().copied().unwrap_or(0) & 1 == 0 {
        SIMPLE_CHUNKED
    } else {
        FIXED_ARRAY_CHUNKED
    };
    let mut bytes = base.to_vec();

    for mutation in data.get(1..).unwrap_or(&[]).chunks_exact(5).take(64) {
        let offset = u32::from_le_bytes([mutation[0], mutation[1], mutation[2], mutation[3]])
            as usize
            % bytes.len();
        bytes[offset] ^= mutation[4];
    }

    bytes
}

fn exercise_dataset(dataset: &Dataset) {
    let _ = dataset.read_array::<f32>();
    let _ = dataset.read_array::<f64>();
    let _ = dataset.read_array::<i32>();

    let shape = dataset.shape();
    if shape.is_empty() {
        return;
    }

    let all = SliceInfo::all(shape.len());
    let _ = dataset.read_slice::<f32>(&all);
    let _ = dataset.read_slice::<f64>(&all);

    let bounded = SliceInfo {
        selections: shape
            .iter()
            .map(|&dim| SliceInfoElem::Slice {
                start: 0,
                end: dim.min(2),
                step: 1,
            })
            .collect(),
    };
    let _ = dataset.read_slice::<f32>(&bounded);
    let _ = dataset.read_slice::<f64>(&bounded);

    if shape.iter().all(|&dim| dim > 1) {
        let strided = SliceInfo {
            selections: shape
                .iter()
                .map(|&dim| SliceInfoElem::Slice {
                    start: 0,
                    end: dim,
                    step: 2,
                })
                .collect(),
        };
        let _ = dataset.read_slice::<f32>(&strided);
        let _ = dataset.read_slice::<f64>(&strided);
    }
}

fuzz_target!(|data: &[u8]| {
    let bytes = mutated_fixture(data);
    if let Ok(file) = Hdf5File::from_bytes(&bytes) {
        for path in ["/temperature", "/data"] {
            if let Ok(dataset) = file.dataset(path) {
                exercise_dataset(&dataset);
            }
        }
    }
});
