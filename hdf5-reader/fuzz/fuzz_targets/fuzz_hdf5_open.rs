#![no_main]

use hdf5_reader::{Hdf5File, OpenOptions};
use libfuzzer_sys::fuzz_target;

/// A few bytes can declare a sparse dataset of any size, whose read is a legitimate allocation.
/// ASan aborts on a request that large rather than letting it fail, so refuse it first.
const MAX_READ_BYTES: usize = 256 * 1024 * 1024;

fuzz_target!(|data: &[u8]| {
    // Try to open arbitrary bytes as an HDF5 file.
    // This should never panic — only return errors.
    let options = OpenOptions::default().with_max_read_bytes(MAX_READ_BYTES);
    if let Ok(file) = Hdf5File::from_bytes_with_options(data, options) {
        // If parsing succeeds, try reading the root group and its children.
        if let Ok(root) = file.root_group() {
            let _ = root.attributes();
            let _ = root.groups();
            if let Ok(datasets) = root.datasets() {
                for dataset in &datasets {
                    let _ = dataset.shape();
                    let _ = dataset.dtype();
                    let _ = dataset.attributes();
                    // Try reading as different types — should not panic.
                    let _ = dataset.read_array::<f64>();
                    let _ = dataset.read_array::<f32>();
                    let _ = dataset.read_array::<i32>();
                }
            }
        }
    }
});
