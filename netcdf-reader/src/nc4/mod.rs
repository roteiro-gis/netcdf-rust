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

use crate::error::{Error, Result};
use crate::types::NcGroup;

/// An opened NetCDF-4 file (backed by HDF5).
pub struct Nc4File {
    hdf5: Hdf5File,
    root_group: NcGroup,
}

impl Nc4File {
    /// Open a NetCDF-4 file from disk.
    pub fn open(path: &Path) -> Result<Self> {
        let hdf5 = Hdf5File::open(path)?;
        let root_group = groups::build_root_group(&hdf5)?;
        Ok(Nc4File { hdf5, root_group })
    }

    /// Open a NetCDF-4 file from in-memory bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        let hdf5 = Hdf5File::from_bytes(data)?;
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
        let var = self
            .root_group
            .variable(normalized)
            .ok_or_else(|| Error::VariableNotFound(path.to_string()))?;
        let dataset = self.hdf5.dataset(normalized)?;

        // Verify the shape matches what we expect
        debug_assert_eq!(dataset.shape(), &var.shape()[..]);

        Ok(dataset.read_array::<T>()?)
    }
}

fn normalize_dataset_path(path: &str) -> Result<&str> {
    let trimmed = path.trim_matches('/');
    if trimmed.is_empty() {
        return Err(Error::VariableNotFound(path.to_string()));
    }
    Ok(trimmed)
}
