//! Map HDF5 groups to NetCDF-4 groups.
//!
//! Each HDF5 group becomes an `NcGroup`. The root group is special: it may
//! contain `_NCProperties` and other internal attributes that should be filtered.
//! Sub-groups are traversed recursively.

use hdf5_reader::Hdf5File;

use crate::error::Result;
use crate::types::NcGroup;

use super::attributes;
use super::dimensions;
use super::variables;

/// Build the root NcGroup from an HDF5 file.
pub fn build_root_group(hdf5: &Hdf5File) -> Result<NcGroup> {
    let root = hdf5.root_group()?;
    build_group_recursive(&root, "/")
}

/// Recursively build an NcGroup from an HDF5 Group.
fn build_group_recursive(
    hdf5_group: &hdf5_reader::group::Group<'_>,
    name: &str,
) -> Result<NcGroup> {
    // Extract dimensions from dimension-scale datasets in this group.
    let dimensions = dimensions::extract_dimensions(hdf5_group)?;

    // Extract variables (non-dimension-scale datasets).
    let variables = variables::extract_variables(hdf5_group, &dimensions)?;

    // Extract group-level attributes, filtering internal NetCDF-4 attributes.
    let nc_attributes = attributes::extract_group_attributes(hdf5_group)?;

    // Recurse into child groups.
    let mut child_groups = Vec::new();
    if let Ok(hdf5_children) = hdf5_group.groups() {
        for child in &hdf5_children {
            let child_name = child.name().to_string();
            let nc_child = build_group_recursive(child, &child_name)?;
            child_groups.push(nc_child);
        }
    }

    Ok(NcGroup {
        name: name.to_string(),
        dimensions,
        variables,
        attributes: nc_attributes,
        groups: child_groups,
    })
}
