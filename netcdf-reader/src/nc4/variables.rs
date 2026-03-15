//! Map HDF5 datasets to NetCDF-4 variables.
//!
//! Each HDF5 dataset that is NOT a dimension scale becomes an NcVariable.
//! The variable's dimensions are determined from the `DIMENSION_LIST` attribute,
//! which contains object references to the corresponding dimension-scale datasets.

use hdf5_reader::group::Group;

use crate::error::Result;
use crate::types::{NcDimension, NcVariable};

use super::attributes;
use super::types::hdf5_to_nc_type;

/// Extract variables from an HDF5 group.
///
/// Datasets with `CLASS=DIMENSION_SCALE` are dimensions, not variables.
/// All other datasets become NcVariables.
pub fn extract_variables(
    group: &Group<'_>,
    dimensions: &[NcDimension],
) -> Result<Vec<NcVariable>> {
    let datasets = match group.datasets() {
        Ok(ds) => ds,
        Err(_) => return Ok(Vec::new()),
    };

    let mut variables = Vec::new();

    for ds in &datasets {
        // Skip dimension scale datasets
        let is_dim_scale = ds
            .attribute("CLASS")
            .ok()
            .and_then(|attr| attr.read_string().ok())
            .map(|s| s == "DIMENSION_SCALE")
            .unwrap_or(false);

        if is_dim_scale {
            continue;
        }

        // Map the HDF5 datatype to a NetCDF type
        let nc_type = match hdf5_to_nc_type(ds.dtype()) {
            Ok(t) => t,
            Err(_) => continue, // Skip datasets with unsupported types
        };

        // Resolve dimensions from DIMENSION_LIST attribute or shape
        let var_dims = resolve_variable_dimensions(ds, dimensions);

        // Detect if this variable uses an unlimited dimension
        let is_unlimited = var_dims.iter().any(|d| d.is_unlimited);

        // Compute data size
        let shape = ds.shape();
        let elem_size = nc_type.size() as u64;
        let total_elements: u64 = if shape.is_empty() { 1 } else { shape.iter().product() };
        let data_size = total_elements * elem_size;

        // Compute record size (size per element along the first dim)
        let record_size = if is_unlimited && shape.len() > 1 {
            shape[1..].iter().product::<u64>() * elem_size
        } else {
            elem_size
        };

        // Extract variable-level attributes
        let var_attrs = attributes::extract_variable_attributes(ds)?;

        variables.push(NcVariable {
            name: ds.name().to_string(),
            dimensions: var_dims,
            dtype: nc_type,
            attributes: var_attrs,
            data_offset: ds.address(),
            _data_size: data_size,
            is_record_var: is_unlimited,
            record_size,
        });
    }

    Ok(variables)
}

/// Resolve dimensions for a variable by matching its shape against the
/// group's dimensions. Falls back to anonymous dimensions from the shape.
fn resolve_variable_dimensions(
    ds: &hdf5_reader::Dataset<'_>,
    dimensions: &[NcDimension],
) -> Vec<NcDimension> {
    let shape = ds.shape();

    // Try to match dimensions by size (simple heuristic when DIMENSION_LIST
    // isn't available or parseable). This matches dims in order.
    let mut var_dims = Vec::with_capacity(shape.len());
    let mut used = vec![false; dimensions.len()];

    for &dim_size in shape {
        let mut matched = false;
        for (i, dim) in dimensions.iter().enumerate() {
            if !used[i] && dim.size == dim_size {
                var_dims.push(dim.clone());
                used[i] = true;
                matched = true;
                break;
            }
        }
        if !matched {
            // Create an anonymous dimension
            var_dims.push(NcDimension {
                name: format!("dim_{}", dim_size),
                size: dim_size,
                is_unlimited: false,
            });
        }
    }

    // Check if any matched dimension is unlimited and update dataspace info
    if let Some(max_dims) = ds.max_dims() {
        for (i, md) in max_dims.iter().enumerate() {
            if *md == u64::MAX && i < var_dims.len() {
                var_dims[i].is_unlimited = true;
            }
        }
    }

    var_dims
}
