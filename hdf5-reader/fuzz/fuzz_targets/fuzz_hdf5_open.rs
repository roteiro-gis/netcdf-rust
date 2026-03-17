#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Try to open arbitrary bytes as an HDF5 file.
    // This should never panic — only return errors.
    if let Ok(file) = hdf5_reader::Hdf5File::from_bytes(data) {
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
