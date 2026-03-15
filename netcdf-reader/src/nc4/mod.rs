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

use hdf5_reader::Hdf5File;

use crate::error::Result;
use crate::types::NcGroup;

/// An opened NetCDF-4 file (backed by HDF5).
pub struct Nc4File {
    #[allow(dead_code)]
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
}
