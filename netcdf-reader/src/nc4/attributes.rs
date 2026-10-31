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
use hdf5_reader::messages::datatype::{Datatype, VarLenKind};

use super::types::hdf5_to_nc_type;
use crate::error::{Error, Result};
use crate::types::{NcAttrValue, NcAttribute};
use crate::user_defined::{read_attribute_values, ValueSource};

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
pub fn extract_group_attributes(
    group: &Group,
    metadata_mode: crate::NcMetadataMode,
) -> Result<Vec<NcAttribute>> {
    collect_visible_attributes(
        &group.attributes()?,
        metadata_mode,
        ValueSource::Group(group),
    )
}

/// Extract user-visible attributes from an HDF5 dataset, filtering out
/// internal attributes.
pub fn extract_variable_attributes(
    dataset: &hdf5_reader::Dataset,
    metadata_mode: crate::NcMetadataMode,
) -> Result<Vec<NcAttribute>> {
    collect_visible_attributes(
        &dataset.attributes(),
        metadata_mode,
        ValueSource::Dataset(dataset),
    )
}

fn collect_visible_attributes(
    attrs: &[hdf5_reader::Attribute],
    metadata_mode: crate::NcMetadataMode,
    source: ValueSource<'_>,
) -> Result<Vec<NcAttribute>> {
    let mut nc_attrs = Vec::new();
    for attr in attrs {
        if is_internal_attribute(&attr.name) {
            continue;
        }
        if let Some(nc_attr) = convert_attribute(attr, metadata_mode, source)? {
            nc_attrs.push(nc_attr);
        }
    }
    Ok(nc_attrs)
}

pub fn convert_visible_attribute(
    attr: &hdf5_reader::Attribute,
    metadata_mode: crate::NcMetadataMode,
) -> Result<Option<NcAttribute>> {
    if is_internal_attribute(&attr.name) {
        return Ok(None);
    }
    convert_attribute(attr, metadata_mode, ValueSource::None)
}

/// Convert an HDF5 attribute to a NetCDF attribute.
fn convert_attribute(
    attr: &hdf5_reader::Attribute,
    metadata_mode: crate::NcMetadataMode,
    source: ValueSource<'_>,
) -> Result<Option<NcAttribute>> {
    let Some(value) = convert_attribute_value_with_source(attr, metadata_mode, source)? else {
        return Ok(None);
    };

    Ok(Some(NcAttribute {
        name: attr.name.clone(),
        value,
    }))
}

/// Convert an HDF5 attribute's value to an NcAttrValue.
///
/// Enum attributes become their integer values; other user-defined types
/// retain their type and decoded values. Strict mode rejects invalid payloads.
#[cfg(test)]
fn convert_attribute_value(
    attr: &hdf5_reader::Attribute,
    metadata_mode: crate::NcMetadataMode,
) -> Result<Option<NcAttrValue>> {
    convert_attribute_value_with_source(attr, metadata_mode, ValueSource::None)
}

fn convert_attribute_value_with_source(
    attr: &hdf5_reader::Attribute,
    metadata_mode: crate::NcMetadataMode,
    source: ValueSource<'_>,
) -> Result<Option<NcAttrValue>> {
    let strict = metadata_mode == crate::NcMetadataMode::Strict;

    let unsupported = || {
        Error::InvalidData(format!(
            "attribute '{}' uses unsupported NetCDF-4 datatype {:?}",
            attr.name, attr.datatype
        ))
    };

    let read_attr = |result: hdf5_reader::error::Result<NcAttrValue>| match result {
        Ok(value) => Ok(Some(value)),
        Err(err) if strict => Err(Error::InvalidData(format!(
            "attribute '{}' could not be decoded: {err}",
            attr.name
        ))),
        Err(_) => Ok(None),
    };

    match &attr.datatype {
        Datatype::FixedPoint { size, signed, .. } => match (size, signed) {
            (1, true) => read_attr(attr.read_1d::<i8>().map(NcAttrValue::Bytes)),
            (1, false) => read_attr(attr.read_1d::<u8>().map(NcAttrValue::UBytes)),
            (2, true) => read_attr(attr.read_1d::<i16>().map(NcAttrValue::Shorts)),
            (2, false) => read_attr(attr.read_1d::<u16>().map(NcAttrValue::UShorts)),
            (4, true) => read_attr(attr.read_1d::<i32>().map(NcAttrValue::Ints)),
            (4, false) => read_attr(attr.read_1d::<u32>().map(NcAttrValue::UInts)),
            (8, true) => read_attr(attr.read_1d::<i64>().map(NcAttrValue::Int64s)),
            (8, false) => read_attr(attr.read_1d::<u64>().map(NcAttrValue::UInt64s)),
            _ if strict => Err(unsupported()),
            _ => Ok(None),
        },
        Datatype::FloatingPoint { size, .. } => match size {
            4 => read_attr(attr.read_1d::<f32>().map(NcAttrValue::Floats)),
            8 => read_attr(attr.read_1d::<f64>().map(NcAttrValue::Doubles)),
            _ if strict => Err(unsupported()),
            _ => Ok(None),
        },
        Datatype::String { .. } => {
            if attr.shape.is_empty() {
                read_attr(attr.read_string().map(NcAttrValue::Chars))
            } else {
                read_attr(attr.read_strings().map(NcAttrValue::Strings))
            }
        }
        Datatype::VarLen {
            base,
            kind: VarLenKind::String,
            ..
        } if base.is_vlen_string_base() => {
            if attr.shape.is_empty() {
                read_attr(attr.read_string().map(NcAttrValue::Chars))
            } else {
                read_attr(attr.read_strings().map(NcAttrValue::Strings))
            }
        }
        Datatype::Enum { base, .. } => {
            let integers = hdf5_reader::Attribute {
                datatype: (**base).clone(),
                ..attr.clone()
            };
            convert_attribute_value_with_source(&integers, metadata_mode, source)
        }
        datatype if hdf5_to_nc_type(datatype).is_ok() => {
            let dtype = hdf5_to_nc_type(datatype)?;
            read_attribute_values(source, attr)
                .map(|values| Some(NcAttrValue::UserDefined { dtype, values }))
                .or_else(|err| if strict { Err(err) } else { Ok(None) })
        }
        _ if strict => Err(unsupported()),
        _ => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use hdf5_reader::Attribute;
    use hdf5_reader::ByteOrder;

    use super::*;

    #[test]
    fn strict_mode_rejects_attribute_types_netcdf4_does_not_have() {
        let attr = Attribute {
            name: "bitfield_attr".to_string(),
            datatype: Datatype::Bitfield {
                size: 4,
                byte_order: ByteOrder::LittleEndian,
            },
            shape: vec![1],
            raw_data: vec![0, 0, 0, 0],
            decoded_strings: None,
        };

        let err = convert_attribute_value(&attr, crate::NcMetadataMode::Strict).unwrap_err();
        assert!(matches!(err, Error::InvalidData(_)));
    }

    #[test]
    fn user_defined_attribute_types_are_preserved_in_both_modes() {
        let attr = Attribute {
            name: "opaque_attr".to_string(),
            datatype: Datatype::Opaque {
                size: 4,
                tag: "test".to_string(),
            },
            shape: vec![1],
            raw_data: vec![0, 0, 0, 0],
            decoded_strings: None,
        };

        for mode in [crate::NcMetadataMode::Strict, crate::NcMetadataMode::Lossy] {
            assert!(matches!(convert_attribute_value(&attr, mode).unwrap(),
                Some(NcAttrValue::UserDefined { values, .. })
                    if values == vec![crate::NcValue::Opaque(vec![0; 4])]
            ));
        }
    }

    #[test]
    fn enum_attributes_read_as_their_integers() {
        let attr = Attribute {
            name: "_FillValue".to_string(),
            datatype: Datatype::Enum {
                base: Box::new(Datatype::FixedPoint {
                    size: 2,
                    signed: true,
                    byte_order: ByteOrder::LittleEndian,
                }),
                members: Vec::new(),
            },
            shape: vec![2],
            raw_data: vec![1, 0, 0xff, 0xff],
            decoded_strings: None,
        };

        assert_eq!(
            convert_attribute_value(&attr, crate::NcMetadataMode::Strict).unwrap(),
            Some(NcAttrValue::Shorts(vec![1, -1]))
        );
    }

    #[test]
    fn dangling_vlen_attribute_references_error_in_strict_mode() {
        let file = hdf5_reader::Hdf5File::from_bytes(include_bytes!(
            "../../../testdata/netcdf4/nc4_user_attribute_values.nc"
        ))
        .unwrap();
        let group = file.root_group().unwrap();
        let mut attr = group.attribute("sequences").unwrap();
        let index = 4 + usize::from(group.offset_size());
        attr.raw_data[index..index + 4].copy_from_slice(&65535u32.to_le_bytes());
        assert!(convert_attribute_value_with_source(
            &attr,
            crate::NcMetadataMode::Strict,
            ValueSource::Group(&group)
        )
        .is_err());
        assert!(convert_attribute_value_with_source(
            &attr,
            crate::NcMetadataMode::Lossy,
            ValueSource::Group(&group)
        )
        .unwrap()
        .is_none());
    }

    #[test]
    fn lossy_mode_skips_unsupported_attribute_types() {
        let attr = Attribute {
            name: "bad_int".to_string(),
            datatype: Datatype::FixedPoint {
                size: 16,
                signed: true,
                byte_order: ByteOrder::LittleEndian,
            },
            shape: vec![1],
            raw_data: vec![0; 16],
            decoded_strings: None,
        };

        assert!(convert_attribute_value(&attr, crate::NcMetadataMode::Lossy)
            .unwrap()
            .is_none());
    }
}
