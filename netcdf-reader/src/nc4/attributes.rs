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

use hdf5_reader::group::Group;
use hdf5_reader::messages::datatype::Datatype;

use crate::error::Result;
use crate::types::{NcAttrValue, NcAttribute};

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
pub fn extract_group_attributes(group: &Group<'_>) -> Result<Vec<NcAttribute>> {
    let hdf5_attrs = match group.attributes() {
        Ok(a) => a,
        Err(_) => return Ok(Vec::new()),
    };

    let mut nc_attrs = Vec::new();
    for attr in &hdf5_attrs {
        if is_internal_attribute(&attr.name) {
            continue;
        }
        if let Some(nc_attr) = convert_attribute(attr) {
            nc_attrs.push(nc_attr);
        }
    }
    Ok(nc_attrs)
}

/// Extract user-visible attributes from an HDF5 dataset, filtering out
/// internal attributes.
pub fn extract_variable_attributes(
    dataset: &hdf5_reader::Dataset<'_>,
) -> Result<Vec<NcAttribute>> {
    let hdf5_attrs = dataset.attributes();
    let mut nc_attrs = Vec::new();
    for attr in &hdf5_attrs {
        if is_internal_attribute(&attr.name) {
            continue;
        }
        if let Some(nc_attr) = convert_attribute(attr) {
            nc_attrs.push(nc_attr);
        }
    }
    Ok(nc_attrs)
}

/// Convert an HDF5 attribute to a NetCDF attribute.
fn convert_attribute(attr: &hdf5_reader::Attribute) -> Option<NcAttribute> {
    let value = convert_attribute_value(attr)?;
    Some(NcAttribute {
        name: attr.name.clone(),
        value,
    })
}

/// Convert an HDF5 attribute's value to an NcAttrValue.
fn convert_attribute_value(attr: &hdf5_reader::Attribute) -> Option<NcAttrValue> {
    match &attr.datatype {
        Datatype::FixedPoint {
            size, signed, ..
        } => match (size, signed) {
            (1, true) => attr.read_1d::<i8>().ok().map(NcAttrValue::Bytes),
            (1, false) => attr.read_1d::<u8>().ok().map(NcAttrValue::UBytes),
            (2, true) => attr.read_1d::<i16>().ok().map(NcAttrValue::Shorts),
            (2, false) => attr.read_1d::<u16>().ok().map(NcAttrValue::UShorts),
            (4, true) => attr.read_1d::<i32>().ok().map(NcAttrValue::Ints),
            (4, false) => attr.read_1d::<u32>().ok().map(NcAttrValue::UInts),
            (8, true) => attr.read_1d::<i64>().ok().map(NcAttrValue::Int64s),
            (8, false) => attr.read_1d::<u64>().ok().map(NcAttrValue::UInt64s),
            _ => None,
        },
        Datatype::FloatingPoint { size, .. } => match size {
            4 => attr.read_1d::<f32>().ok().map(NcAttrValue::Floats),
            8 => attr.read_1d::<f64>().ok().map(NcAttrValue::Doubles),
            _ => None,
        },
        Datatype::String { .. } => {
            // Try reading as a single string first, then as array
            if attr.num_elements() == 1 {
                attr.read_string()
                    .ok()
                    .map(NcAttrValue::Chars)
            } else {
                attr.read_strings()
                    .ok()
                    .map(NcAttrValue::Strings)
            }
        }
        _ => None,
    }
}
