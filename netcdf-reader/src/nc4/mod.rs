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

    /// Read a variable's data as a typed array.
    ///
    /// The variable is identified by name in the root group. The HDF5 dataset
    /// at the stored `data_offset` address is opened and read.
    pub fn read_variable<T: H5Type>(&self, name: &str) -> Result<ArrayD<T>> {
        let _var = self
            .root_group
            .variable(name)
            .ok_or_else(|| Error::VariableNotFound(name.to_string()))?;

        // The data_offset stores the HDF5 dataset object header address
        let dataset = self.hdf5.dataset(&find_dataset_path(&self.root_group, name))?;
        Ok(dataset.read_array::<T>()?)
    }
}

/// Find the HDF5 path for a variable by searching the group hierarchy.
fn find_dataset_path(group: &NcGroup, name: &str) -> String {
    // Check this group's variables
    if group.variable(name).is_some() {
        return if group.name == "/" {
            format!("/{}", name)
        } else {
            format!("{}/{}", group.name, name)
        };
    }

    // Recurse into child groups
    for child in &group.groups {
        let path = find_dataset_path(child, name);
        if !path.is_empty() {
            return path;
        }
    }

    String::new()
}
