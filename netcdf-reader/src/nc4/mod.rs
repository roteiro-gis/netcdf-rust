//! NetCDF-4 (HDF5-backed) format support.
//!
//! This module maps the HDF5 data model to the NetCDF data model:
//! - HDF5 groups become NetCDF groups
//! - HDF5 datasets become NetCDF variables
//! - HDF5 attributes become NetCDF attributes
//! - Dimensions are reconstructed from `DIMENSION_LIST` and `_Netcdf4Dimid` attributes
//!
//! Requires the `netcdf4` feature (enabled by default).

pub mod attributes;
pub mod dimensions;
pub mod groups;
pub mod types;
pub mod variables;

use std::path::Path;

use hdf5_reader::datatype_api::H5Type;
use hdf5_reader::Hdf5File;
use ndarray::ArrayD;
#[cfg(feature = "rayon")]
use rayon::ThreadPool;

use crate::error::{Error, Result};
use crate::types::{NcGroup, NcType};

/// Dispatch on `NcType` to read data and promote to `f64`.
///
/// `$dtype` must be an expression of type `&NcType`.
/// `$read_expr` is a macro-like callback: for each numeric type `$T`,
/// the macro evaluates `$read_expr` with `$T` substituted in.
///
/// Usage:
/// ```ignore
/// dispatch_read_as_f64!(&var.dtype, |$T| dataset.read_array::<$T>())
/// ```
macro_rules! dispatch_read_as_f64 {
    ($dtype:expr, |$T:ident| $read_expr:expr) => {{
        use crate::types::NcType;
        match $dtype {
            NcType::Byte => {
                type $T = i8;
                let arr = $read_expr?;
                Ok(arr.mapv(|v| v as f64))
            }
            NcType::Short => {
                type $T = i16;
                let arr = $read_expr?;
                Ok(arr.mapv(|v| v as f64))
            }
            NcType::Int => {
                type $T = i32;
                let arr = $read_expr?;
                Ok(arr.mapv(|v| v as f64))
            }
            NcType::Float => {
                type $T = f32;
                let arr = $read_expr?;
                Ok(arr.mapv(|v| v as f64))
            }
            NcType::Double => {
                type $T = f64;
                Ok($read_expr?)
            }
            NcType::UByte => {
                type $T = u8;
                let arr = $read_expr?;
                Ok(arr.mapv(|v| v as f64))
            }
            NcType::UShort => {
                type $T = u16;
                let arr = $read_expr?;
                Ok(arr.mapv(|v| v as f64))
            }
            NcType::UInt => {
                type $T = u32;
                let arr = $read_expr?;
                Ok(arr.mapv(|v| v as f64))
            }
            NcType::Int64 => {
                type $T = i64;
                let arr = $read_expr?;
                Ok(arr.mapv(|v| v as f64))
            }
            NcType::UInt64 => {
                type $T = u64;
                let arr = $read_expr?;
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
            other => Err(Error::TypeMismatch {
                expected: "numeric type".to_string(),
                actual: format!("{:?}", other),
            }),
        }
    }};
}

/// An opened NetCDF-4 file (backed by HDF5).
pub struct Nc4File {
    hdf5: Hdf5File,
    root_group: NcGroup,
}

impl Nc4File {
    pub(crate) fn from_hdf5(hdf5: Hdf5File, root_group: NcGroup) -> Self {
        Nc4File { hdf5, root_group }
    }

    /// Open a NetCDF-4 file from disk.
    pub fn open(path: &Path) -> Result<Self> {
        Self::open_with_options(path, crate::NcOpenOptions::default())
    }

    /// Open a NetCDF-4 file from disk with custom options.
    pub fn open_with_options(path: &Path, options: crate::NcOpenOptions) -> Result<Self> {
        let hdf5 = Hdf5File::open_with_options(
            path,
            hdf5_reader::OpenOptions {
                chunk_cache_bytes: options.chunk_cache_bytes,
                chunk_cache_slots: options.chunk_cache_slots,
                filter_registry: options.filter_registry,
            },
        )?;
        let root_group = groups::build_root_group(&hdf5)?;
        Ok(Nc4File { hdf5, root_group })
    }

    /// Open a NetCDF-4 file from in-memory bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        Self::from_bytes_with_options(data, crate::NcOpenOptions::default())
    }

    /// Open a NetCDF-4 file from in-memory bytes with custom options.
    pub fn from_bytes_with_options(data: &[u8], options: crate::NcOpenOptions) -> Result<Self> {
        let hdf5 = Hdf5File::from_bytes_with_options(
            data,
            hdf5_reader::OpenOptions {
                chunk_cache_bytes: options.chunk_cache_bytes,
                chunk_cache_slots: options.chunk_cache_slots,
                filter_registry: options.filter_registry,
            },
        )?;
        let root_group = groups::build_root_group(&hdf5)?;
        Ok(Nc4File { hdf5, root_group })
    }

    /// The root group.
    pub fn root_group(&self) -> &NcGroup {
        &self.root_group
    }

    /// Check if this file uses the classic data model (`_nc3_strict`).
    ///
    /// This checks the raw HDF5 root group attributes (before the internal
    /// attribute filter removes `_nc3_strict`).
    pub fn is_classic_model(&self) -> bool {
        self.hdf5
            .root_group()
            .ok()
            .and_then(|g| g.attribute("_nc3_strict").ok())
            .is_some()
    }

    /// Read a variable's data as a typed array.
    ///
    /// Looks up the variable by path relative to the root group, then opens the
    /// matching HDF5 dataset and reads the data.
    pub fn read_variable<T: H5Type>(&self, path: &str) -> Result<ArrayD<T>> {
        let normalized = normalize_dataset_path(path)?;
        let dataset = self.hdf5.dataset(normalized)?;
        Ok(dataset.read_array::<T>()?)
    }

    /// Read a string variable as a single string.
    pub fn read_variable_as_string(&self, path: &str) -> Result<String> {
        let mut strings = self.read_variable_as_strings(path)?;
        match strings.len() {
            1 => Ok(strings.swap_remove(0)),
            0 => Err(Error::InvalidData(format!(
                "variable '{}' contains no string elements",
                path
            ))),
            count => Err(Error::InvalidData(format!(
                "variable '{}' contains {count} string elements; use read_variable_as_strings()",
                path
            ))),
        }
    }

    /// Read a string variable as a flat vector of strings.
    pub fn read_variable_as_strings(&self, path: &str) -> Result<Vec<String>> {
        let normalized = normalize_dataset_path(path)?;
        let var = self
            .root_group
            .variable(normalized)
            .ok_or_else(|| Error::VariableNotFound(path.to_string()))?;
        if var.dtype != NcType::String {
            return Err(Error::TypeMismatch {
                expected: "String".to_string(),
                actual: format!("{:?}", var.dtype),
            });
        }

        let dataset = self.hdf5.dataset(normalized)?;
        Ok(dataset.read_strings()?)
    }

    #[cfg(feature = "rayon")]
    pub fn read_variable_parallel<T: H5Type>(&self, path: &str) -> Result<ArrayD<T>> {
        let normalized = normalize_dataset_path(path)?;
        let dataset = self.hdf5.dataset(normalized)?;
        Ok(dataset.read_array_parallel::<T>()?)
    }

    #[cfg(feature = "rayon")]
    pub fn read_variable_in_pool<T: H5Type>(
        &self,
        path: &str,
        pool: &ThreadPool,
    ) -> Result<ArrayD<T>> {
        let normalized = normalize_dataset_path(path)?;
        let dataset = self.hdf5.dataset(normalized)?;
        Ok(dataset.read_array_in_pool::<T>(pool)?)
    }
}

impl Nc4File {
    /// Read a variable with automatic type promotion to f64.
    ///
    /// Reads in the native HDF5 type and promotes to f64 via `mapv`.
    pub fn read_variable_as_f64(&self, path: &str) -> Result<ArrayD<f64>> {
        let normalized = normalize_dataset_path(path)?;
        let var = self
            .root_group
            .variable(normalized)
            .ok_or_else(|| Error::VariableNotFound(path.to_string()))?;
        let dataset = self.hdf5.dataset(normalized)?;

        debug_assert_eq!(dataset.shape(), &var.shape()[..]);

        dispatch_read_as_f64!(&var.dtype, |T| dataset.read_array::<T>())
    }

    /// Read a slice of a variable with automatic type promotion to f64.
    pub fn read_variable_slice_as_f64(
        &self,
        path: &str,
        selection: &crate::types::NcSliceInfo,
    ) -> Result<ArrayD<f64>> {
        let normalized = normalize_dataset_path(path)?;
        let var = self
            .root_group
            .variable(normalized)
            .ok_or_else(|| Error::VariableNotFound(path.to_string()))?;
        let dataset = self.hdf5.dataset(normalized)?;
        let hdf5_sel = selection.to_hdf5_slice_info();

        dispatch_read_as_f64!(&var.dtype, |T| dataset.read_slice::<T>(&hdf5_sel))
    }

    /// Read a typed slice of a variable (NC4 delegation).
    pub fn read_variable_slice<T: H5Type>(
        &self,
        path: &str,
        selection: &crate::types::NcSliceInfo,
    ) -> Result<ArrayD<T>> {
        let normalized = normalize_dataset_path(path)?;
        let dataset = self.hdf5.dataset(normalized)?;
        let hdf5_sel = selection.to_hdf5_slice_info();
        Ok(dataset.read_slice::<T>(&hdf5_sel)?)
    }

    /// Read a typed slice of a variable using chunk-level parallelism.
    ///
    /// Chunked datasets decompress overlapping chunks in parallel via Rayon.
    /// Non-chunked layouts fall back to `read_variable_slice`.
    #[cfg(feature = "rayon")]
    pub fn read_variable_slice_parallel<T: H5Type>(
        &self,
        path: &str,
        selection: &crate::types::NcSliceInfo,
    ) -> Result<ArrayD<T>> {
        let normalized = normalize_dataset_path(path)?;
        let dataset = self.hdf5.dataset(normalized)?;
        let hdf5_sel = selection.to_hdf5_slice_info();
        Ok(dataset.read_slice_parallel::<T>(&hdf5_sel)?)
    }
}

fn normalize_dataset_path(path: &str) -> Result<&str> {
    let trimmed = path.trim_matches('/');
    if trimmed.is_empty() {
        return Err(Error::VariableNotFound(path.to_string()));
    }
    Ok(trimmed)
}
