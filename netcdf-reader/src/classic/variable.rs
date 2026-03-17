//! Variable-level read methods for classic NetCDF files.
//!
//! Provides convenience methods for reading variable data from a `ClassicFile`,
//! with type-checked access and support for both record and non-record variables.

use ndarray::ArrayD;

use crate::error::{Error, Result};
use crate::types::{NcType, NcVariable};

use super::data::{self, compute_record_stride, NcReadType};
use super::ClassicFile;

impl ClassicFile {
    /// Read a variable's data as an ndarray of the specified type.
    ///
    /// The type parameter `T` must match the variable's NetCDF type. For example,
    /// use `f32` for NC_FLOAT variables and `f64` for NC_DOUBLE variables.
    pub fn read_variable<T: NcReadType>(&self, name: &str) -> Result<ArrayD<T>> {
        let var = self.find_variable(name)?;

        // Check type compatibility.
        let expected = T::nc_type();
        if var.dtype != expected {
            return Err(Error::TypeMismatch {
                expected: format!("{:?}", expected),
                actual: format!("{:?}", var.dtype),
            });
        }

        let file_data = self.data.as_slice();

        if var.is_record_var {
            let record_stride = compute_record_stride(&self.root_group.variables);
            data::read_record_variable(file_data, var, self.numrecs, record_stride)
        } else {
            data::read_non_record_variable(file_data, var)
        }
    }

    /// Read a variable's data with automatic type promotion to f64.
    ///
    /// This reads any numeric variable and converts all values to f64,
    /// which is convenient for analysis but may lose precision for i64/u64.
    pub fn read_variable_as_f64(&self, name: &str) -> Result<ArrayD<f64>> {
        let var = self.find_variable(name)?;
        let file_data = self.data.as_slice();

        match var.dtype {
            NcType::Byte => {
                let arr = self.read_typed_variable::<i8>(var, file_data)?;
                Ok(arr.mapv(|v| v as f64))
            }
            NcType::Short => {
                let arr = self.read_typed_variable::<i16>(var, file_data)?;
                Ok(arr.mapv(|v| v as f64))
            }
            NcType::Int => {
                let arr = self.read_typed_variable::<i32>(var, file_data)?;
                Ok(arr.mapv(|v| v as f64))
            }
            NcType::Float => {
                let arr = self.read_typed_variable::<f32>(var, file_data)?;
                Ok(arr.mapv(|v| v as f64))
            }
            NcType::Double => self.read_typed_variable::<f64>(var, file_data),
            NcType::UByte => {
                let arr = self.read_typed_variable::<u8>(var, file_data)?;
                Ok(arr.mapv(|v| v as f64))
            }
            NcType::UShort => {
                let arr = self.read_typed_variable::<u16>(var, file_data)?;
                Ok(arr.mapv(|v| v as f64))
            }
            NcType::UInt => {
                let arr = self.read_typed_variable::<u32>(var, file_data)?;
                Ok(arr.mapv(|v| v as f64))
            }
            NcType::Int64 => {
                let arr = self.read_typed_variable::<i64>(var, file_data)?;
                Ok(arr.mapv(|v| v as f64))
            }
            NcType::UInt64 => {
                let arr = self.read_typed_variable::<u64>(var, file_data)?;
                Ok(arr.mapv(|v| v as f64))
            }
            NcType::Char => Err(Error::TypeMismatch {
                expected: "numeric type".to_string(),
                actual: "Char".to_string(),
            }),
            NcType::String => Err(Error::TypeMismatch {
                expected: "numeric type".to_string(),
                actual: "String".to_string(),
            }),
            _ => Err(Error::TypeMismatch {
                expected: "numeric type".to_string(),
                actual: format!("{:?}", var.dtype),
            }),
        }
    }

    /// Read a char variable as a String (or Vec<String> for multi-dimensional).
    pub fn read_variable_as_string(&self, name: &str) -> Result<String> {
        let var = self.find_variable(name)?;
        if var.dtype != NcType::Char {
            return Err(Error::TypeMismatch {
                expected: "Char".to_string(),
                actual: format!("{:?}", var.dtype),
            });
        }

        let file_data = self.data.as_slice();
        let arr = self.read_typed_variable::<u8>(var, file_data)?;
        let bytes: Vec<u8> = arr.iter().copied().collect();
        let s = String::from_utf8_lossy(&bytes)
            .trim_end_matches('\0')
            .to_string();
        Ok(s)
    }

    /// Read a slice (hyperslab) of a variable.
    ///
    /// For non-record variables where the selection only restricts the outermost
    /// dimension (inner dims select full range, step=1), byte offsets are computed
    /// directly to avoid reading the entire variable. Otherwise falls back to
    /// full-read-then-slice.
    pub fn read_variable_slice<T: NcReadType>(
        &self,
        name: &str,
        selection: &crate::types::NcSliceInfo,
    ) -> Result<ArrayD<T>> {
        use crate::types::NcSliceInfoElem;
        use ndarray::IxDyn;

        let var = self.find_variable(name)?;
        let expected = T::nc_type();
        if var.dtype != expected {
            return Err(Error::TypeMismatch {
                expected: format!("{:?}", expected),
                actual: format!("{:?}", var.dtype),
            });
        }
        let file_data = self.data.as_slice();

        // For non-record variables, try direct byte-offset extraction.
        if !var.is_record_var && var.ndim() > 0 {
            let shape: Vec<u64> = var.shape();
            let ndim = shape.len();

            // Check if inner dims are all full-range with step=1.
            let can_direct = selection.selections.len() == ndim
                && selection
                    .selections
                    .iter()
                    .enumerate()
                    .skip(1)
                    .all(|(d, sel)| {
                        matches!(sel, NcSliceInfoElem::Slice { start, end, step }
                        if *start == 0 && *step == 1 && (*end == u64::MAX || *end >= shape[d]))
                    });

            if can_direct {
                let elem_size = T::element_size();
                let row_elements: u64 = shape[1..].iter().product::<u64>().max(1);
                let row_bytes = row_elements as usize * elem_size;

                let (first_row, num_rows, result_shape) = match &selection.selections[0] {
                    NcSliceInfoElem::Index(idx) => {
                        let rs: Vec<usize> = shape[1..].iter().map(|&d| d as usize).collect();
                        (*idx, 1u64, rs)
                    }
                    NcSliceInfoElem::Slice { start, end, step } => {
                        let actual_end = if *end == u64::MAX {
                            shape[0]
                        } else {
                            (*end).min(shape[0])
                        };
                        let count = (actual_end - start).div_ceil(*step);
                        if *step == 1 {
                            let mut rs = vec![count as usize];
                            rs.extend(shape[1..].iter().map(|&d| d as usize));
                            (*start, count, rs)
                        } else {
                            // Step > 1: can't do contiguous read, fall through.
                            return {
                                let full = data::read_non_record_variable(file_data, var)?;
                                slice_classic_array(&full, var, selection, 0)
                            };
                        }
                    }
                };

                let byte_offset = var.data_offset as usize + first_row as usize * row_bytes;
                let total_bytes = num_rows as usize * row_bytes;
                let total_elements = num_rows as usize * row_elements as usize;

                if byte_offset + total_bytes > file_data.len() {
                    return Err(Error::InvalidData(format!(
                        "variable '{}' slice data extends beyond file",
                        var.name
                    )));
                }

                let data_slice = &file_data[byte_offset..byte_offset + total_bytes];
                let values = T::decode_bulk_be(data_slice, total_elements)?;

                return ndarray::ArrayD::from_shape_vec(IxDyn(&result_shape), values)
                    .map_err(|e| Error::InvalidData(format!("failed to create array: {}", e)));
            }
        }

        // Fallback: read full variable then slice.
        if var.is_record_var {
            let record_stride = compute_record_stride(&self.root_group.variables);
            let full = data::read_record_variable(file_data, var, self.numrecs, record_stride)?;
            slice_classic_array(&full, var, selection, self.numrecs)
        } else {
            let full = data::read_non_record_variable(file_data, var)?;
            slice_classic_array(&full, var, selection, 0)
        }
    }

    /// Read a slice with automatic type promotion to f64.
    pub fn read_variable_slice_as_f64(
        &self,
        name: &str,
        selection: &crate::types::NcSliceInfo,
    ) -> Result<ArrayD<f64>> {
        let var = self.find_variable(name)?;
        let file_data = self.data.as_slice();

        macro_rules! slice_promoted {
            ($ty:ty) => {{
                let full = self.read_typed_variable::<$ty>(var, file_data)?;
                let full_f64 = full.mapv(|v| v as f64);
                slice_classic_array(
                    &full_f64,
                    var,
                    selection,
                    if var.is_record_var { self.numrecs } else { 0 },
                )
            }};
        }

        match var.dtype {
            NcType::Byte => slice_promoted!(i8),
            NcType::Short => slice_promoted!(i16),
            NcType::Int => slice_promoted!(i32),
            NcType::Float => slice_promoted!(f32),
            NcType::Double => slice_promoted!(f64),
            NcType::UByte => slice_promoted!(u8),
            NcType::UShort => slice_promoted!(u16),
            NcType::UInt => slice_promoted!(u32),
            NcType::Int64 => slice_promoted!(i64),
            NcType::UInt64 => slice_promoted!(u64),
            NcType::Char => Err(Error::TypeMismatch {
                expected: "numeric type".to_string(),
                actual: "Char".to_string(),
            }),
            _ => Err(Error::TypeMismatch {
                expected: "numeric type".to_string(),
                actual: format!("{:?}", var.dtype),
            }),
        }
    }

    /// Internal: find a variable by name.
    fn find_variable(&self, name: &str) -> Result<&NcVariable> {
        self.root_group
            .variables
            .iter()
            .find(|v| v.name == name)
            .ok_or_else(|| Error::VariableNotFound(name.to_string()))
    }

    /// Internal: read a variable with the correct record handling.
    fn read_typed_variable<T: NcReadType>(
        &self,
        var: &NcVariable,
        file_data: &[u8],
    ) -> Result<ArrayD<T>> {
        if var.is_record_var {
            let record_stride = compute_record_stride(&self.root_group.variables);
            data::read_record_variable(file_data, var, self.numrecs, record_stride)
        } else {
            data::read_non_record_variable(file_data, var)
        }
    }
}

/// Apply a NcSliceInfo selection to an already-read classic array.
///
/// For classic format, we read the full variable first then extract the
/// hyperslab, since the data is contiguous (non-record) or interleaved (record).
fn slice_classic_array<T: Clone + Default + 'static>(
    full: &ArrayD<T>,
    var: &NcVariable,
    selection: &crate::types::NcSliceInfo,
    numrecs: u64,
) -> Result<ArrayD<T>> {
    use crate::types::NcSliceInfoElem;
    use ndarray::Slice;

    let ndim = var.ndim();
    if selection.selections.len() != ndim {
        return Err(Error::InvalidData(format!(
            "selection has {} dimensions but variable '{}' has {}",
            selection.selections.len(),
            var.name,
            ndim
        )));
    }

    // Build the shape of the full array, resolving unlimited dims.
    let mut shape: Vec<usize> = var.shape().iter().map(|&s| s as usize).collect();
    if var.is_record_var && !shape.is_empty() {
        shape[0] = numrecs as usize;
    }

    // First, apply range slicing on all dimensions.
    let mut view = full.view();
    for (d, sel) in selection.selections.iter().enumerate() {
        let dim_size = shape[d] as u64;
        match sel {
            NcSliceInfoElem::Index(idx) => {
                if *idx >= dim_size {
                    return Err(Error::InvalidData(format!(
                        "index {} out of bounds for dimension {} (size {})",
                        idx, d, dim_size
                    )));
                }
                // Slice to a single element (will collapse later).
                view.slice_axis_inplace(
                    ndarray::Axis(d),
                    Slice::new(*idx as isize, Some(*idx as isize + 1), 1),
                );
            }
            NcSliceInfoElem::Slice { start, end, step } => {
                let actual_end = if *end == u64::MAX {
                    dim_size as isize
                } else {
                    (*end).min(dim_size) as isize
                };
                view.slice_axis_inplace(
                    ndarray::Axis(d),
                    Slice::new(*start as isize, Some(actual_end), *step as isize),
                );
            }
        }
    }

    // Now collapse Index dimensions (remove axes of size 1 from Index selections).
    let mut result = view.to_owned();
    let mut removed = 0;
    for (d, sel) in selection.selections.iter().enumerate() {
        if matches!(sel, NcSliceInfoElem::Index(_)) {
            let axis = d - removed;
            result = result.index_axis_move(ndarray::Axis(axis), 0);
            removed += 1;
        }
    }

    Ok(result)
}
