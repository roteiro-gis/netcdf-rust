//! Reconstruct NetCDF-4 dimensions from HDF5 metadata.
//!
//! NetCDF-4 stores dimension information as:
//! - Scale datasets (one per dimension) with `CLASS=DIMENSION_SCALE` attribute
//! - `DIMENSION_LIST` attribute on each variable dataset referencing the scales
//! - `REFERENCE_LIST` attribute on scale datasets (back-references)
//! - `_Netcdf4Dimid` attribute to assign stable dimension IDs
//! - `NAME` attribute on scale datasets gives the dimension name
//!
//! Unlimited dimensions are represented by chunked datasets whose maximum
//! dimension in the dataspace is `H5S_UNLIMITED`.
//!
//! TODO: Phase 5 — Full implementation of dimension reconstruction.

use hdf5_reader::group::Group;

use crate::error::Result;
use crate::types::NcDimension;

/// Extract dimensions from an HDF5 group.
///
/// This scans the group's datasets for those that are dimension scales
/// (have `CLASS=DIMENSION_SCALE` attribute) and constructs `NcDimension`
/// entries from them.
pub fn extract_dimensions(_group: &Group<'_>) -> Result<Vec<NcDimension>> {
    // TODO: Phase 5
    // 1. Iterate over datasets in the group.
    // 2. For each dataset with CLASS=DIMENSION_SCALE, read NAME for dim name.
    // 3. Read the dataset's dataspace to get current size.
    // 4. Check max dims for H5S_UNLIMITED to mark is_unlimited.
    // 5. Use _Netcdf4Dimid to assign stable ordering.
    Ok(Vec::new())
}
