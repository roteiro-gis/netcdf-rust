//! Filter and convert HDF5 attributes to NetCDF-4 attributes.
//!
//! NetCDF-4 uses several internal HDF5 attributes that should not be exposed
//! to users:
//! - `_NCProperties`: file provenance metadata
//! - `_Netcdf4Dimid`: dimension ID assignment
//! - `_Netcdf4Coordinates`: coordinate variable tracking
//! - `DIMENSION_LIST`: dimension-to-variable references
//! - `REFERENCE_LIST`: variable-to-dimension back-references
//! - `CLASS`: dimension scale marker
//! - `NAME`: dimension scale name
//!
//! TODO: Phase 5 — Full attribute conversion.

use hdf5_reader::group::Group;

use crate::error::Result;
use crate::types::NcAttribute;

/// Internal attribute names that should be hidden from the NetCDF user API.
const INTERNAL_ATTRIBUTES: &[&str] = &[
    "_NCProperties",
    "_Netcdf4Dimid",
    "_Netcdf4Coordinates",
    "_nc3_strict",
    "DIMENSION_LIST",
    "REFERENCE_LIST",
    "CLASS",
    "NAME",
];

/// Returns true if the given attribute name is an internal NetCDF-4/HDF5 attribute
/// that should not be exposed to users.
pub fn is_internal_attribute(name: &str) -> bool {
    INTERNAL_ATTRIBUTES.contains(&name)
}

/// Extract user-visible attributes from an HDF5 group, filtering out
/// internal NetCDF-4 attributes.
pub fn extract_group_attributes(_group: &Group<'_>) -> Result<Vec<NcAttribute>> {
    // TODO: Phase 5
    // 1. Read all HDF5 attributes from the group.
    // 2. Filter out internal attributes (using is_internal_attribute).
    // 3. Convert remaining HDF5 Attributes to NcAttributes using convert_attribute.
    Ok(Vec::new())
}

/// Extract user-visible attributes from an HDF5 dataset, filtering out
/// internal attributes.
pub fn extract_variable_attributes(
    _dataset: &hdf5_reader::Dataset<'_>,
) -> Result<Vec<NcAttribute>> {
    // TODO: Phase 5
    Ok(Vec::new())
}
