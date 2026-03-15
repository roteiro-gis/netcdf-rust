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
