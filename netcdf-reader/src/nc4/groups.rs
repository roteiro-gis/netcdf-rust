//! Map HDF5 groups to NetCDF-4 groups.
//!
//! Each HDF5 group becomes an `NcGroup`. The root group is special: it may
//! contain `_NCProperties` and other internal attributes that should be filtered.
//! Sub-groups are traversed recursively.

use std::collections::{HashMap, HashSet};

use hdf5_reader::Hdf5File;

use crate::error::{Error, Result};
use crate::types::{NcDimension, NcGroup};

use super::attributes;
use super::dimensions;
use super::variables;
use super::{leaf_name, normalize_group_path};

fn visible_dimensions(
    local_dimensions: &[NcDimension],
    inherited_dimensions: &[NcDimension],
) -> Vec<NcDimension> {
    let mut visible_dimensions = local_dimensions.to_vec();
    visible_dimensions.extend(
        inherited_dimensions
            .iter()
            .filter(|dim| {
                !local_dimensions
                    .iter()
                    .any(|local_dim| local_dim.name == dim.name)
            })
            .cloned(),
    );
    visible_dimensions
}

fn visible_dim_addr_map(
    local_dim_addr_map: HashMap<u64, NcDimension>,
    inherited_dim_addr_map: &HashMap<u64, NcDimension>,
) -> HashMap<u64, NcDimension> {
    let mut visible_dim_addr_map = inherited_dim_addr_map.clone();
    visible_dim_addr_map.extend(local_dim_addr_map);
    visible_dim_addr_map
}

pub(crate) type DimensionExtents = HashMap<usize, HashMap<u64, u64>>;

fn storage_id(group: &hdf5_reader::group::Group) -> usize {
    std::ptr::from_ref(group.storage()) as *const () as usize
}

/// Extents are indexed by scale address, so a child dimension that shadows an
/// ancestor's name remains independent. Descendant variables can extend an
/// ancestor dimension before any group's metadata is exposed.
pub(crate) fn collect_dimension_extents(hdf5: &Hdf5File) -> Result<DimensionExtents> {
    let mut extents = HashMap::new();
    collect_group_extents(
        &hdf5.root_group()?,
        &HashMap::new(),
        &mut extents,
        &mut HashSet::new(),
        0,
    )?;
    Ok(extents)
}

fn collect_group_extents(
    group: &hdf5_reader::group::Group,
    inherited_ids: &HashMap<i64, u64>,
    extents: &mut DimensionExtents,
    visited: &mut HashSet<(usize, u64)>,
    depth: usize,
) -> Result<()> {
    if depth > 64 {
        return Err(Error::InvalidData("NetCDF group nesting exceeds 64".into()));
    }
    if !visited.insert((storage_id(group), group.address())) {
        return Ok(());
    }
    let (children, datasets) = group.members()?;
    if children.is_empty()
        && !datasets.iter().any(|dataset| {
            dataset
                .max_dims()
                .is_some_and(|dims| dims.contains(&u64::MAX))
        })
    {
        return Ok(());
    }
    let mut ids = inherited_ids.clone();
    let is_scale = |dataset: &hdf5_reader::Dataset| {
        dataset
            .attribute("CLASS")
            .ok()
            .and_then(|attr| attr.read_string().ok())
            .is_some_and(|class| class == "DIMENSION_SCALE")
    };
    for dataset in &datasets {
        if is_scale(dataset) {
            if let Ok(id) = dataset
                .attribute("_Netcdf4Dimid")
                .and_then(|attr| attr.read_scalar::<i32>())
            {
                ids.insert(i64::from(id), dataset.address());
            }
        }
    }
    for dataset in &datasets {
        let Some(max_dims) = dataset.max_dims() else {
            continue;
        };
        if !max_dims.contains(&u64::MAX) {
            continue;
        }
        let addresses = if is_scale(dataset) {
            if dataset.ndim() == 1 {
                vec![dataset.address()]
            } else {
                let Ok(coordinates) = dataset
                    .attribute("_Netcdf4Coordinates")
                    .and_then(|attr| attr.read_1d::<i32>())
                else {
                    continue;
                };
                let Some(addresses) = coordinates
                    .iter()
                    .map(|id| ids.get(&i64::from(*id)).copied())
                    .collect::<Option<Vec<_>>>()
                else {
                    continue;
                };
                addresses
            }
        } else {
            let Ok(addresses) = variables::resolve_dimension_scale_addresses(dataset, group) else {
                continue;
            };
            addresses
        };
        for ((address, &extent), &max_dim) in addresses.iter().zip(dataset.shape()).zip(max_dims) {
            if max_dim == u64::MAX {
                let current = extents
                    .entry(storage_id(group))
                    .or_default()
                    .entry(*address)
                    .or_default();
                *current = (*current).max(extent);
            }
        }
    }
    for child in children {
        let inherited = if storage_id(&child) == storage_id(group) {
            &ids
        } else {
            &HashMap::new()
        };
        collect_group_extents(&child, inherited, extents, visited, depth + 1)?;
    }
    Ok(())
}

/// Build the root NcGroup from an HDF5 file.
pub fn build_root_group(hdf5: &Hdf5File, metadata_mode: crate::NcMetadataMode) -> Result<NcGroup> {
    build_group_at_path(hdf5, "/", metadata_mode, true)
}

/// Build only the root group's local metadata.
pub fn build_root_group_metadata(
    hdf5: &Hdf5File,
    metadata_mode: crate::NcMetadataMode,
) -> Result<NcGroup> {
    build_group_at_path(hdf5, "/", metadata_mode, false)
}

/// Build metadata for a group path.
pub fn build_group_at_path(
    hdf5: &Hdf5File,
    path: &str,
    metadata_mode: crate::NcMetadataMode,
    recursive: bool,
) -> Result<NcGroup> {
    let extents = collect_dimension_extents(hdf5)?;
    build_group_at_path_with_extents(hdf5, path, metadata_mode, recursive, &extents)
}

pub(crate) fn build_group_at_path_with_extents(
    hdf5: &Hdf5File,
    path: &str,
    metadata_mode: crate::NcMetadataMode,
    recursive: bool,
    extents: &DimensionExtents,
) -> Result<NcGroup> {
    let normalized = normalize_group_path(path);
    let root = hdf5.root_group()?;
    let mut group = root;
    let mut inherited_dimensions = Vec::new();
    let mut inherited_dim_addr_map = HashMap::new();
    let mut inherited_dim_id_map = HashMap::new();

    for component in normalized.split('/').filter(|part| !part.is_empty()) {
        let datasets = group.datasets()?;
        let index = dimensions::index_dimensions_with_extents(
            &datasets,
            metadata_mode,
            extents.get(&storage_id(&group)).unwrap_or(&HashMap::new()),
        )?;
        inherited_dimensions = visible_dimensions(&index.dimensions, &inherited_dimensions);
        inherited_dim_addr_map = visible_dim_addr_map(index.by_address, &inherited_dim_addr_map);
        inherited_dim_id_map.extend(index.by_id);
        group = group
            .group(component)
            .map_err(|_| Error::GroupNotFound(path.to_string()))?;
    }

    let group_name = if normalized.is_empty() {
        "/".to_string()
    } else {
        leaf_name(group.name()).to_string()
    };

    if recursive {
        build_group_recursive(
            &group,
            &group_name,
            &inherited_dimensions,
            &inherited_dim_addr_map,
            &inherited_dim_id_map,
            metadata_mode,
            extents,
        )
    } else {
        build_group_metadata(
            &group,
            &group_name,
            &inherited_dimensions,
            &inherited_dim_addr_map,
            &inherited_dim_id_map,
            metadata_mode,
            extents,
        )
    }
}

/// Recursively build an NcGroup from an HDF5 Group.
fn build_group_recursive(
    hdf5_group: &hdf5_reader::group::Group,
    name: &str,
    inherited_dimensions: &[NcDimension],
    inherited_dim_addr_map: &HashMap<u64, NcDimension>,
    inherited_dim_id_map: &HashMap<i64, NcDimension>,
    metadata_mode: crate::NcMetadataMode,
    extents: &DimensionExtents,
) -> Result<NcGroup> {
    let (hdf5_children, datasets) = hdf5_group.members()?;

    // Extract dimensions declared locally in this group, then combine them
    // with dimensions inherited from ancestor groups for lookups and variable
    // reconstruction.
    let index = dimensions::index_dimensions_with_extents(
        &datasets,
        metadata_mode,
        extents
            .get(&storage_id(hdf5_group))
            .unwrap_or(&HashMap::new()),
    )?;
    let visible_dimensions = visible_dimensions(&index.dimensions, inherited_dimensions);
    let visible_dim_addr_map = visible_dim_addr_map(index.by_address, inherited_dim_addr_map);
    let mut visible_dim_id_map = inherited_dim_id_map.clone();
    visible_dim_id_map.extend(index.by_id);

    // Extract variables (non-dimension-scale datasets).
    let variables = variables::extract_variables_from_datasets(
        &datasets,
        hdf5_group,
        &visible_dimensions,
        &visible_dim_addr_map,
        &visible_dim_id_map,
        metadata_mode,
    )?;

    // Extract group-level attributes, filtering internal NetCDF-4 attributes.
    let nc_attributes = attributes::extract_group_attributes(hdf5_group, metadata_mode)?;

    // Recurse into child groups.
    let mut child_groups = Vec::new();
    for child in &hdf5_children {
        let child_name = leaf_name(child.name()).to_string();
        let nc_child = build_group_recursive(
            child,
            &child_name,
            &visible_dimensions,
            &visible_dim_addr_map,
            &visible_dim_id_map,
            metadata_mode,
            extents,
        )?;
        child_groups.push(nc_child);
    }

    Ok(NcGroup {
        name: name.to_string(),
        dimensions: visible_dimensions,
        variables,
        attributes: nc_attributes,
        groups: child_groups,
    })
}

fn build_group_metadata(
    hdf5_group: &hdf5_reader::group::Group,
    name: &str,
    inherited_dimensions: &[NcDimension],
    inherited_dim_addr_map: &HashMap<u64, NcDimension>,
    inherited_dim_id_map: &HashMap<i64, NcDimension>,
    metadata_mode: crate::NcMetadataMode,
    extents: &DimensionExtents,
) -> Result<NcGroup> {
    let datasets = hdf5_group.datasets()?;
    let index = dimensions::index_dimensions_with_extents(
        &datasets,
        metadata_mode,
        extents
            .get(&storage_id(hdf5_group))
            .unwrap_or(&HashMap::new()),
    )?;
    let visible_dimensions = visible_dimensions(&index.dimensions, inherited_dimensions);
    let visible_dim_addr_map = visible_dim_addr_map(index.by_address, inherited_dim_addr_map);
    let mut visible_dim_id_map = inherited_dim_id_map.clone();
    visible_dim_id_map.extend(index.by_id);
    let variables = variables::extract_variables_from_datasets(
        &datasets,
        hdf5_group,
        &visible_dimensions,
        &visible_dim_addr_map,
        &visible_dim_id_map,
        metadata_mode,
    )?;
    let attributes = attributes::extract_group_attributes(hdf5_group, metadata_mode)?;

    Ok(NcGroup {
        name: name.to_string(),
        dimensions: visible_dimensions,
        variables,
        attributes,
        groups: Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn visible_dimensions_include_inherited_without_duplicates() {
        let local = vec![NcDimension {
            name: "y".to_string(),
            size: 4,
            is_unlimited: false,
        }];
        let inherited = vec![
            NcDimension {
                name: "x".to_string(),
                size: 3,
                is_unlimited: false,
            },
            NcDimension {
                name: "y".to_string(),
                size: 99,
                is_unlimited: true,
            },
        ];

        let merged = visible_dimensions(&local, &inherited);
        let names: Vec<&str> = merged.iter().map(|dim| dim.name.as_str()).collect();
        assert_eq!(names, vec!["y", "x"]);
        assert_eq!(merged[0].size, 4);
        assert!(!merged[0].is_unlimited);
    }

    #[test]
    fn visible_dim_addr_map_prefers_local_dimensions() {
        let mut inherited = HashMap::new();
        inherited.insert(
            10,
            NcDimension {
                name: "x".to_string(),
                size: 3,
                is_unlimited: false,
            },
        );
        inherited.insert(
            20,
            NcDimension {
                name: "shared".to_string(),
                size: 1,
                is_unlimited: false,
            },
        );

        let mut local = HashMap::new();
        local.insert(
            20,
            NcDimension {
                name: "shared".to_string(),
                size: 2,
                is_unlimited: true,
            },
        );

        let merged = visible_dim_addr_map(local, &inherited);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged.get(&10).unwrap().name, "x");
        assert_eq!(merged.get(&20).unwrap().size, 2);
        assert!(merged.get(&20).unwrap().is_unlimited);
    }
}
