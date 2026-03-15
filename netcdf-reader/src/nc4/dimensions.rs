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

use hdf5_reader::group::Group;

use crate::error::Result;
use crate::types::NcDimension;

/// Extract dimensions from an HDF5 group.
///
/// This scans the group's datasets for those that are dimension scales
/// (have `CLASS=DIMENSION_SCALE` attribute) and constructs `NcDimension`
/// entries from them.
pub fn extract_dimensions(group: &Group<'_>) -> Result<Vec<NcDimension>> {
    let datasets = match group.datasets() {
        Ok(ds) => ds,
        Err(_) => return Ok(Vec::new()),
    };

    let mut dims: Vec<(Option<i64>, NcDimension)> = Vec::new();

    for ds in &datasets {
        // Check for CLASS=DIMENSION_SCALE attribute
        let is_dim_scale = ds
            .attribute("CLASS")
            .ok()
            .and_then(|attr| attr.read_string().ok())
            .map(|s| s == "DIMENSION_SCALE")
            .unwrap_or(false);

        if !is_dim_scale {
            continue;
        }

        // Get dimension name from NAME attribute, falling back to dataset name
        let name = ds
            .attribute("NAME")
            .ok()
            .and_then(|attr| attr.read_string().ok())
            .map(|s| {
                // NetCDF-4 uses "This is a netCDF dimension but not a netCDF variable."
                // as a sentinel for anonymous dimensions. In that case, use the dataset name.
                if s.starts_with("This is a netCDF dimension but not a netCDF variable") {
                    ds.name().to_string()
                } else {
                    s
                }
            })
            .unwrap_or_else(|| ds.name().to_string());

        // Get current size from dataspace
        let shape = ds.shape();
        let size = if shape.is_empty() { 0 } else { shape[0] };

        // Check max dims for unlimited
        let is_unlimited = ds
            .max_dims()
            .is_some_and(|md| !md.is_empty() && md[0] == u64::MAX);

        // Get stable ordering from _Netcdf4Dimid
        let dimid = ds
            .attribute("_Netcdf4Dimid")
            .ok()
            .and_then(|attr| attr.read_scalar::<i32>().ok())
            .map(|id| id as i64);

        dims.push((
            dimid,
            NcDimension {
                name,
                size,
                is_unlimited,
            },
        ));
    }

    // Sort by _Netcdf4Dimid if available, otherwise preserve order
    dims.sort_by_key(|(id, _)| id.unwrap_or(i64::MAX));

    Ok(dims.into_iter().map(|(_, d)| d).collect())
}
