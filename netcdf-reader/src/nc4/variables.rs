//! Map HDF5 datasets to NetCDF-4 variables.
//!
//! Each HDF5 dataset that is NOT a dimension scale becomes an NcVariable.
//! The variable's dimensions are determined from the `DIMENSION_LIST` attribute,
//! which contains object references to the corresponding dimension-scale datasets.
//!
//! TODO: Phase 5 — Full variable extraction from HDF5 datasets.

use hdf5_reader::group::Group;

use crate::error::Result;
use crate::types::{NcDimension, NcVariable};

/// Extract variables from an HDF5 group.
///
/// Datasets with `CLASS=DIMENSION_SCALE` are dimensions, not variables.
/// All other datasets become NcVariables.
pub fn extract_variables(
    _group: &Group<'_>,
    _dimensions: &[NcDimension],
) -> Result<Vec<NcVariable>> {
    // TODO: Phase 5
    // 1. List all datasets in the group.
    // 2. Skip those with CLASS=DIMENSION_SCALE attribute.
    // 3. For each remaining dataset:
    //    a. Read DIMENSION_LIST to resolve dimension references.
    //    b. Map HDF5 datatype to NcType.
    //    c. Extract variable-level attributes (filtering internal ones).
    //    d. Record the dataset's object header address as data_offset.
    //    e. Determine if it uses an unlimited dimension.
    Ok(Vec::new())
}
