#![no_main]

//! Fuzz the classic (and NetCDF-4) parser entry point. Opening arbitrary bytes
//! must only ever return an error, never panic, hang, or allocate unbounded
//! memory. After a successful open, enumerating metadata and reading each
//! variable with a small bounded slice must also stay panic-free.

use libfuzzer_sys::fuzz_target;
use netcdf_reader::{NcFile, NcSliceInfo, NcSliceInfoElem};

/// Cap per-read work so a valid-but-huge declared shape cannot make the fuzzer
/// time out on legitimate (non-buggy) large allocations.
const MAX_SLICE_LEN: u64 = 64;

fuzz_target!(|data: &[u8]| {
    let Ok(file) = NcFile::from_bytes(data) else {
        return;
    };

    let _ = file.global_attributes();
    let _ = file.dimensions();

    let Ok(variables) = file.variables() else {
        return;
    };

    // Snapshot names/ranks first so we are not iterating a borrow of `file`
    // while calling read methods on it.
    let plans: Vec<(String, usize)> = variables
        .iter()
        .map(|var| (var.name.clone(), var.dimensions.len()))
        .collect();

    for (name, rank) in plans {
        // Whole-variable read of a bounded-but-arbitrary type.
        let _ = file.read_variable::<f64>(&name);
        let _ = file.read_variable_as_f64(&name);
        let _ = file.read_variable_as_string(&name);

        // A small hyperslab from the origin exercises the strided slice path
        // without depending on the (attacker-controlled) declared shape.
        let selection = NcSliceInfo {
            selections: (0..rank)
                .map(|_| NcSliceInfoElem::Slice {
                    start: 0,
                    end: MAX_SLICE_LEN,
                    step: 1,
                })
                .collect(),
        };
        let _: Result<ndarray::ArrayD<f64>, _> = file.read_variable_slice(&name, &selection);
    }
});
