//! Property test: any classic schema the builder accepts must round-trip
//! through the reader with identical shapes and values.
//!
//! Before the vsize/record-stride fix, the writer and reader shared the same
//! (incorrect) size derivation, so a roundtrip could not detect it. They now
//! derive record layout independently — the reader recomputes strides from
//! dimensions while the writer follows the spec's padding rules — which makes
//! this roundtrip a genuine oracle for the classic on-disk format.

use netcdf_reader::NcFile;
use netcdf_writer::{NcFileBuilder, NcWriteFormat, NcWriteOptions};
use proptest::prelude::*;

/// One variable in a generated schema: a numeric type, whether it is a record
/// variable (spans the unlimited dimension), and which fixed dimensions it
/// spans.
#[derive(Debug, Clone)]
struct VarSpec {
    type_idx: u8,
    is_record: bool,
    fixed_dims: Vec<bool>,
}

#[derive(Debug, Clone)]
struct SchemaSpec {
    fixed_sizes: Vec<u64>,
    has_unlimited: bool,
    num_records: u64,
    vars: Vec<VarSpec>,
}

fn schema_strategy() -> impl Strategy<Value = SchemaSpec> {
    // 0..=2 fixed dimensions of size 1..=4, an optional unlimited dimension
    // with 1..=4 records, and 1..=4 variables over arbitrary dim subsets.
    (
        prop::collection::vec(1u64..=4, 0..=2),
        any::<bool>(),
        1u64..=4,
    )
        .prop_flat_map(|(fixed_sizes, has_unlimited, num_records)| {
            let n_fixed = fixed_sizes.len();
            let var = (
                0u8..4,
                any::<bool>(),
                prop::collection::vec(any::<bool>(), n_fixed),
            )
                .prop_map(|(type_idx, wants_record, fixed_dims)| VarSpec {
                    type_idx,
                    is_record: wants_record,
                    fixed_dims,
                });
            prop::collection::vec(var, 1..=4).prop_map(move |mut vars| {
                if !has_unlimited {
                    for v in &mut vars {
                        v.is_record = false;
                    }
                }
                SchemaSpec {
                    fixed_sizes: fixed_sizes.clone(),
                    has_unlimited,
                    num_records,
                    vars,
                }
            })
        })
}

fn element_count(spec: &SchemaSpec, var: &VarSpec) -> u64 {
    let mut count = if var.is_record { spec.num_records } else { 1 };
    for (dim, &included) in var.fixed_dims.iter().enumerate() {
        if included {
            count *= spec.fixed_sizes[dim];
        }
    }
    count
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    #[test]
    fn classic_schema_roundtrips_through_reader(spec in schema_strategy()) {
        let mut builder = NcFileBuilder::new();

        let record_dim = spec
            .has_unlimited
            .then(|| builder.add_unlimited_dimension("time").unwrap());
        let fixed_dims: Vec<_> = spec
            .fixed_sizes
            .iter()
            .enumerate()
            .map(|(i, &size)| builder.add_dimension(format!("d{i}"), size).unwrap())
            .collect();

        // Track expected data so we can compare after the roundtrip.
        let mut expected: Vec<(String, u8, Vec<f64>)> = Vec::new();

        for (vi, var) in spec.vars.iter().enumerate() {
            let name = format!("v{vi}");
            let mut dims = Vec::new();
            if var.is_record {
                dims.push(record_dim.unwrap());
            }
            for (dim, &included) in var.fixed_dims.iter().enumerate() {
                if included {
                    dims.push(fixed_dims[dim]);
                }
            }

            let count = element_count(&spec, var) as usize;
            // Deterministic ascending values keep the assertion simple while
            // still exercising every byte position.
            let values: Vec<f64> = (0..count).map(|k| (vi * 100 + k) as f64).collect();

            match var.type_idx {
                0 => {
                    let handle = builder.add_variable::<i16>(&name, &dims).unwrap();
                    let data: Vec<i16> = values.iter().map(|&v| v as i16).collect();
                    builder.write_variable(handle, &data).unwrap();
                    expected.push((name, 0, data.iter().map(|&v| v as f64).collect()));
                }
                1 => {
                    let handle = builder.add_variable::<i32>(&name, &dims).unwrap();
                    let data: Vec<i32> = values.iter().map(|&v| v as i32).collect();
                    builder.write_variable(handle, &data).unwrap();
                    expected.push((name, 1, data.iter().map(|&v| v as f64).collect()));
                }
                2 => {
                    let handle = builder.add_variable::<f32>(&name, &dims).unwrap();
                    let data: Vec<f32> = values.iter().map(|&v| v as f32).collect();
                    builder.write_variable(handle, &data).unwrap();
                    expected.push((name, 2, data.iter().map(|&v| v as f64).collect()));
                }
                _ => {
                    let handle = builder.add_variable::<f64>(&name, &dims).unwrap();
                    builder.write_variable(handle, &values).unwrap();
                    expected.push((name, 3, values));
                }
            }
        }

        let (_format, bytes) = builder.to_vec(NcWriteOptions {
            format: NcWriteFormat::AutoClassic,
        })?;
        let file = NcFile::from_bytes(&bytes)?;

        for (name, type_idx, want) in &expected {
            let got: Vec<f64> = match type_idx {
                0 => file.read_variable::<i16>(name)?.iter().map(|&v| v as f64).collect(),
                1 => file.read_variable::<i32>(name)?.iter().map(|&v| v as f64).collect(),
                2 => file.read_variable::<f32>(name)?.iter().map(|&v| v as f64).collect(),
                _ => file.read_variable::<f64>(name)?.iter().copied().collect(),
            };
            prop_assert_eq!(&got, want, "variable {} mismatch", name);
        }
    }
}
