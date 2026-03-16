//! Map HDF5 datasets to NetCDF-4 variables.
//!
//! Each HDF5 dataset that is NOT a dimension scale becomes an NcVariable.
//! The variable's dimensions are determined from the `DIMENSION_LIST` attribute,
//! which contains object references to the corresponding dimension-scale datasets.

use std::collections::HashMap;

use hdf5_reader::group::Group;

use crate::error::Result;
use crate::types::{NcDimension, NcVariable};

use super::attributes;
use super::types::hdf5_to_nc_type;

fn leaf_name(name: &str) -> &str {
    name.rsplit('/').next().unwrap_or(name)
}

/// Extract variables from an HDF5 group.
///
/// Datasets with `CLASS=DIMENSION_SCALE` are dimensions, not variables.
/// All other datasets become NcVariables.
///
/// `dim_addr_map` maps dimension-scale dataset addresses to their `NcDimension`,
/// used to resolve `DIMENSION_LIST` object references.
pub fn extract_variables(
    group: &Group<'_>,
    dimensions: &[NcDimension],
    dim_addr_map: &HashMap<u64, NcDimension>,
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

        // Resolve dimensions from DIMENSION_LIST attribute, falling back to size heuristic
        let var_dims = resolve_variable_dimensions_from_dimlist(ds, group, dim_addr_map)
            .unwrap_or_else(|| resolve_variable_dimensions_by_size(ds, dimensions));

        // Detect if this variable uses an unlimited dimension
        let is_unlimited = var_dims.iter().any(|d| d.is_unlimited);

        // Compute data size
        let shape = ds.shape();
        let elem_size = nc_type.size() as u64;
        let total_elements: u64 = if shape.is_empty() {
            1
        } else {
            shape.iter().product()
        };
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
            name: leaf_name(ds.name()).to_string(),
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

/// Resolve variable dimensions via the `DIMENSION_LIST` attribute.
///
/// `DIMENSION_LIST` is a VLen-of-object-reference attribute. Each entry is a
/// variable-length sequence of object references pointing to dimension-scale
/// datasets. We parse the raw attribute data to extract these references.
///
/// Returns `None` if the attribute is missing or unparseable.
fn resolve_variable_dimensions_from_dimlist(
    ds: &hdf5_reader::Dataset<'_>,
    group: &Group<'_>,
    dim_addr_map: &HashMap<u64, NcDimension>,
) -> Option<Vec<NcDimension>> {
    let attr = ds.attribute("DIMENSION_LIST").ok()?;
    let raw_data = &attr.raw_data;
    let ndim = ds.ndim();
    let offset_size = group.offset_size();
    let file_data = group.file_data();

    if raw_data.is_empty() || ndim == 0 {
        return None;
    }

    // Each vlen entry in the attribute data is:
    //   seq_len: u32 (number of references in this vlen)
    //   heap_addr: offset_size bytes (global heap collection address)
    //   heap_idx: u32 (object index within the global heap collection)
    let entry_size = 4 + offset_size as usize + 4;
    if raw_data.len() < ndim * entry_size {
        return None;
    }

    let mut var_dims = Vec::with_capacity(ndim);
    let mut cursor = hdf5_reader::io::Cursor::new(raw_data);

    for _ in 0..ndim {
        let seq_len = cursor.read_u32_le().ok()? as usize;
        let heap_addr = cursor.read_offset(offset_size).ok()?;
        let heap_idx = cursor.read_u32_le().ok()? as u16;

        if seq_len == 0 || hdf5_reader::io::Cursor::is_undefined_offset(heap_addr, offset_size) {
            // No reference for this dimension — can't resolve.
            return None;
        }

        // Parse the global heap collection at heap_addr.
        let mut heap_cursor = hdf5_reader::io::Cursor::new(file_data);
        heap_cursor.set_position(heap_addr);
        let collection = hdf5_reader::global_heap::GlobalHeapCollection::parse(
            &mut heap_cursor,
            offset_size,
            group.length_size(),
        )
        .ok()?;

        let heap_obj = collection.get_object(heap_idx)?;

        // The heap object data contains `seq_len` object references,
        // each `offset_size` bytes.
        let refs =
            hdf5_reader::reference::read_object_references(&heap_obj.data, offset_size).ok()?;

        if refs.is_empty() {
            return None;
        }

        // Use the first reference (there's usually only one per dimension).
        let dim_addr = refs[0];
        if let Some(dim) = dim_addr_map.get(&dim_addr) {
            var_dims.push(dim.clone());
        } else {
            // Reference points to unknown address — can't resolve.
            return None;
        }
    }

    // Apply unlimited status from the dataset's max_dims.
    if let Some(max_dims) = ds.max_dims() {
        for (i, md) in max_dims.iter().enumerate() {
            if *md == u64::MAX && i < var_dims.len() {
                var_dims[i].is_unlimited = true;
            }
        }
    }

    Some(var_dims)
}

/// Resolve dimensions for a variable by matching its shape against the
/// group's dimensions. Falls back to anonymous dimensions from the shape.
fn resolve_variable_dimensions_by_size(
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
