//! Shared NetCDF data model used by reader and writer crates.

/// Errors produced by shared NetCDF model helpers.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid data: {0}")]
    InvalidData(String),
}

pub type Result<T> = std::result::Result<T, Error>;

/// NetCDF file format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NcFormat {
    Classic,
    Offset64,
    Cdf5,
    Nc4,
    Nc4Classic,
}

/// NetCDF-4 metadata reconstruction policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NcMetadataMode {
    #[default]
    Strict,
    Lossy,
}

/// NetCDF classic type codes.
pub const NC_BYTE: u32 = 1;
pub const NC_CHAR: u32 = 2;
pub const NC_SHORT: u32 = 3;
pub const NC_INT: u32 = 4;
pub const NC_FLOAT: u32 = 5;
pub const NC_DOUBLE: u32 = 6;
pub const NC_UBYTE: u32 = 7;
pub const NC_USHORT: u32 = 8;
pub const NC_UINT: u32 = 9;
pub const NC_INT64: u32 = 10;
pub const NC_UINT64: u32 = 11;

/// A NetCDF dimension.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NcDimension {
    pub name: String,
    pub size: u64,
    pub is_unlimited: bool,
}

/// A field within a compound type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NcCompoundField {
    pub name: String,
    pub offset: u64,
    pub dtype: NcType,
}

/// A typed integer value used by NetCDF-4 enum definitions and values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NcIntegerValue {
    I8(i8),
    U8(u8),
    I16(i16),
    U16(u16),
    I32(i32),
    U32(u32),
    I64(i64),
    U64(u64),
}

impl NcIntegerValue {
    pub fn as_i128(self) -> Option<i128> {
        match self {
            NcIntegerValue::I8(value) => Some(value as i128),
            NcIntegerValue::U8(value) => Some(value as i128),
            NcIntegerValue::I16(value) => Some(value as i128),
            NcIntegerValue::U16(value) => Some(value as i128),
            NcIntegerValue::I32(value) => Some(value as i128),
            NcIntegerValue::U32(value) => Some(value as i128),
            NcIntegerValue::I64(value) => Some(value as i128),
            NcIntegerValue::U64(value) => Some(i128::from(value)),
        }
    }

    pub fn as_u128(self) -> Option<u128> {
        match self {
            NcIntegerValue::I8(value) => u128::try_from(value).ok(),
            NcIntegerValue::U8(value) => Some(value as u128),
            NcIntegerValue::I16(value) => u128::try_from(value).ok(),
            NcIntegerValue::U16(value) => Some(value as u128),
            NcIntegerValue::I32(value) => u128::try_from(value).ok(),
            NcIntegerValue::U32(value) => Some(value as u128),
            NcIntegerValue::I64(value) => u128::try_from(value).ok(),
            NcIntegerValue::U64(value) => Some(value as u128),
        }
    }
}

/// A named member of a NetCDF-4 enum type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NcEnumMember {
    pub name: String,
    pub value: NcIntegerValue,
}

/// NetCDF data types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NcType {
    Byte,
    Char,
    Short,
    Int,
    Float,
    Double,
    UByte,
    UShort,
    UInt,
    Int64,
    UInt64,
    String,
    Enum {
        base: Box<NcType>,
        members: Vec<NcEnumMember>,
    },
    Compound {
        size: u32,
        fields: Vec<NcCompoundField>,
    },
    Opaque {
        size: u32,
        tag: String,
    },
    Array {
        base: Box<NcType>,
        dims: Vec<u64>,
    },
    VLen {
        base: Box<NcType>,
    },
}

impl NcType {
    pub fn size(&self) -> Result<usize> {
        match self {
            NcType::Byte | NcType::Char | NcType::UByte => Ok(1),
            NcType::Short | NcType::UShort => Ok(2),
            NcType::Int | NcType::UInt | NcType::Float => Ok(4),
            NcType::Int64 | NcType::UInt64 | NcType::Double => Ok(8),
            NcType::String => Ok(std::mem::size_of::<usize>()),
            NcType::Enum { base, .. } => base.size(),
            NcType::Compound { size, .. } => Ok(*size as usize),
            NcType::Opaque { size, .. } => Ok(*size as usize),
            NcType::Array { base, dims } => {
                let base_size = base.size()?;
                let count = dims.iter().try_fold(1usize, |acc, &dim| {
                    let dim = usize::try_from(dim).map_err(|_| {
                        Error::InvalidData(
                            "NetCDF array type dimension exceeds platform usize capacity"
                                .to_string(),
                        )
                    })?;
                    acc.checked_mul(dim).ok_or_else(|| {
                        Error::InvalidData(
                            "NetCDF array type element count exceeds platform usize capacity"
                                .to_string(),
                        )
                    })
                })?;
                base_size.checked_mul(count).ok_or_else(|| {
                    Error::InvalidData(
                        "NetCDF array type byte size exceeds platform usize capacity".to_string(),
                    )
                })
            }
            NcType::VLen { .. } => Ok(std::mem::size_of::<usize>()),
        }
    }

    pub fn classic_type_code(&self) -> Option<u32> {
        match self {
            NcType::Byte => Some(NC_BYTE),
            NcType::Char => Some(NC_CHAR),
            NcType::Short => Some(NC_SHORT),
            NcType::Int => Some(NC_INT),
            NcType::Float => Some(NC_FLOAT),
            NcType::Double => Some(NC_DOUBLE),
            NcType::UByte => Some(NC_UBYTE),
            NcType::UShort => Some(NC_USHORT),
            NcType::UInt => Some(NC_UINT),
            NcType::Int64 => Some(NC_INT64),
            NcType::UInt64 => Some(NC_UINT64),
            NcType::String
            | NcType::Enum { .. }
            | NcType::Compound { .. }
            | NcType::Opaque { .. }
            | NcType::Array { .. }
            | NcType::VLen { .. } => None,
        }
    }

    pub fn is_primitive(&self) -> bool {
        matches!(
            self,
            NcType::Byte
                | NcType::Char
                | NcType::Short
                | NcType::Int
                | NcType::Float
                | NcType::Double
                | NcType::UByte
                | NcType::UShort
                | NcType::UInt
                | NcType::Int64
                | NcType::UInt64
                | NcType::String
        )
    }
}

/// A NetCDF attribute value.
#[derive(Debug, Clone, PartialEq)]
pub enum NcAttrValue {
    Bytes(Vec<i8>),
    Chars(String),
    Shorts(Vec<i16>),
    Ints(Vec<i32>),
    Floats(Vec<f32>),
    Doubles(Vec<f64>),
    UBytes(Vec<u8>),
    UShorts(Vec<u16>),
    UInts(Vec<u32>),
    Int64s(Vec<i64>),
    UInt64s(Vec<u64>),
    Strings(Vec<String>),
}

impl NcAttrValue {
    pub fn as_string(&self) -> Option<String> {
        match self {
            NcAttrValue::Chars(s) => Some(s.clone()),
            NcAttrValue::Strings(v) if v.len() == 1 => Some(v[0].clone()),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            NcAttrValue::Bytes(v) => v.first().map(|&x| x as f64),
            NcAttrValue::Shorts(v) => v.first().map(|&x| x as f64),
            NcAttrValue::Ints(v) => v.first().map(|&x| x as f64),
            NcAttrValue::Floats(v) => v.first().map(|&x| x as f64),
            NcAttrValue::Doubles(v) => v.first().copied(),
            NcAttrValue::UBytes(v) => v.first().map(|&x| x as f64),
            NcAttrValue::UShorts(v) => v.first().map(|&x| x as f64),
            NcAttrValue::UInts(v) => v.first().map(|&x| x as f64),
            NcAttrValue::Int64s(v) => v.first().map(|&x| x as f64),
            NcAttrValue::UInt64s(v) => v.first().map(|&x| x as f64),
            NcAttrValue::Chars(_) | NcAttrValue::Strings(_) => None,
        }
    }

    pub fn as_f64_vec(&self) -> Option<Vec<f64>> {
        match self {
            NcAttrValue::Bytes(v) => Some(v.iter().map(|&x| x as f64).collect()),
            NcAttrValue::Shorts(v) => Some(v.iter().map(|&x| x as f64).collect()),
            NcAttrValue::Ints(v) => Some(v.iter().map(|&x| x as f64).collect()),
            NcAttrValue::Floats(v) => Some(v.iter().map(|&x| x as f64).collect()),
            NcAttrValue::Doubles(v) => Some(v.clone()),
            NcAttrValue::UBytes(v) => Some(v.iter().map(|&x| x as f64).collect()),
            NcAttrValue::UShorts(v) => Some(v.iter().map(|&x| x as f64).collect()),
            NcAttrValue::UInts(v) => Some(v.iter().map(|&x| x as f64).collect()),
            NcAttrValue::Int64s(v) => Some(v.iter().map(|&x| x as f64).collect()),
            NcAttrValue::UInt64s(v) => Some(v.iter().map(|&x| x as f64).collect()),
            NcAttrValue::Chars(_) | NcAttrValue::Strings(_) => None,
        }
    }
}

/// A NetCDF attribute.
#[derive(Debug, Clone, PartialEq)]
pub struct NcAttribute {
    pub name: String,
    pub value: NcAttrValue,
}

/// A NetCDF variable. The offset fields are public for reader/writer crate
/// interoperability; they are not portable semantic metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct NcVariable {
    pub name: String,
    pub dimensions: Vec<NcDimension>,
    pub dtype: NcType,
    pub attributes: Vec<NcAttribute>,
    #[doc(hidden)]
    pub data_offset: u64,
    #[doc(hidden)]
    pub _data_size: u64,
    #[doc(hidden)]
    pub is_record_var: bool,
    #[doc(hidden)]
    pub record_size: u64,
}

impl NcVariable {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn dimensions(&self) -> &[NcDimension] {
        &self.dimensions
    }

    pub fn coordinate_dimension(&self) -> Option<&NcDimension> {
        match self.dimensions.as_slice() {
            [dim] if dim.name == self.name => Some(dim),
            _ => None,
        }
    }

    pub fn is_coordinate_variable(&self) -> bool {
        self.coordinate_dimension().is_some()
    }

    pub fn is_coordinate_variable_for(&self, dimension_name: &str) -> bool {
        self.coordinate_dimension()
            .is_some_and(|dim| dim.name == dimension_name)
    }

    pub fn dtype(&self) -> &NcType {
        &self.dtype
    }

    pub fn shape(&self) -> Vec<u64> {
        self.dimensions.iter().map(|d| d.size).collect()
    }

    pub fn attributes(&self) -> &[NcAttribute] {
        &self.attributes
    }

    pub fn attribute(&self, name: &str) -> Option<&NcAttribute> {
        self.attributes.iter().find(|a| a.name == name)
    }

    pub fn ndim(&self) -> usize {
        self.dimensions.len()
    }

    pub fn num_elements(&self) -> Result<u64> {
        match self.dimensions.as_slice() {
            [] => Ok(1),
            [dim] => Ok(dim.size),
            dimensions => {
                let mut total = 1u64;
                for dim in dimensions {
                    total = total.checked_mul(dim.size).ok_or_else(|| {
                        Error::InvalidData(
                            "NetCDF variable element count overflows u64".to_string(),
                        )
                    })?;
                }
                Ok(total)
            }
        }
    }
}

/// A NetCDF group.
#[derive(Debug, Clone, PartialEq)]
pub struct NcGroup {
    pub name: String,
    pub dimensions: Vec<NcDimension>,
    pub variables: Vec<NcVariable>,
    pub attributes: Vec<NcAttribute>,
    pub groups: Vec<NcGroup>,
}

impl NcGroup {
    pub fn variable(&self, name: &str) -> Option<&NcVariable> {
        let (group_path, variable_name) = split_parent_path(name)?;
        let group = self.group(group_path)?;
        group.variables.iter().find(|v| v.name == variable_name)
    }

    pub fn dimension(&self, name: &str) -> Option<&NcDimension> {
        let (group_path, dimension_name) = split_parent_path(name)?;
        let group = self.group(group_path)?;
        group.dimensions.iter().find(|d| d.name == dimension_name)
    }

    pub fn coordinate_variable(&self, name: &str) -> Option<&NcVariable> {
        let (group_path, dimension_name) = split_parent_path(name)?;
        let group = self.group(group_path)?;
        group
            .variables
            .iter()
            .find(|var| var.is_coordinate_variable_for(dimension_name))
    }

    pub fn coordinate_variables(&self) -> impl Iterator<Item = &NcVariable> {
        self.variables
            .iter()
            .filter(|var| var.is_coordinate_variable())
    }

    pub fn attribute(&self, name: &str) -> Option<&NcAttribute> {
        let (group_path, attribute_name) = split_parent_path(name)?;
        let group = self.group(group_path)?;
        group.attributes.iter().find(|a| a.name == attribute_name)
    }

    pub fn group(&self, name: &str) -> Option<&NcGroup> {
        let trimmed = name.trim_matches('/');
        if trimmed.is_empty() {
            return Some(self);
        }

        let mut group = self;
        for component in trimmed.split('/').filter(|part| !part.is_empty()) {
            group = group.groups.iter().find(|child| child.name == component)?;
        }

        Some(group)
    }
}

fn split_parent_path(path: &str) -> Option<(&str, &str)> {
    let trimmed = path.trim_matches('/');
    if trimmed.is_empty() {
        return None;
    }

    match trimmed.rsplit_once('/') {
        Some((group_path, leaf_name)) if !leaf_name.is_empty() => Some((group_path, leaf_name)),
        Some(_) => None,
        None => Some(("", trimmed)),
    }
}

pub fn checked_usize_from_u64(value: u64, context: &str) -> Result<usize> {
    usize::try_from(value)
        .map_err(|_| Error::InvalidData(format!("{context} exceeds platform usize")))
}

pub fn checked_mul_u64(lhs: u64, rhs: u64, context: &str) -> Result<u64> {
    lhs.checked_mul(rhs)
        .ok_or_else(|| Error::InvalidData(format!("{context} exceeds u64 capacity")))
}

pub fn checked_shape_elements(shape: &[u64], context: &str) -> Result<u64> {
    shape
        .iter()
        .try_fold(1u64, |acc, &dim| checked_mul_u64(acc, dim, context))
}

/// Compute the amount of padding needed to reach a 4-byte boundary.
pub fn padding_to_4(len: usize) -> usize {
    let rem = len % 4;
    if rem == 0 {
        0
    } else {
        4 - rem
    }
}

/// Round up to the next 4-byte boundary.
pub fn pad_to_4(len: usize) -> usize {
    len + padding_to_4(len)
}

/// Hyperslab selection for reading/writing slices of NetCDF variables.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NcSliceInfo {
    pub selections: Vec<NcSliceInfoElem>,
}

/// A single dimension selection within a hyperslab.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NcSliceInfoElem {
    Index(u64),
    Slice { start: u64, end: u64, step: u64 },
}

impl NcSliceInfo {
    pub fn all(ndim: usize) -> Self {
        NcSliceInfo {
            selections: vec![
                NcSliceInfoElem::Slice {
                    start: 0,
                    end: u64::MAX,
                    step: 1,
                };
                ndim
            ],
        }
    }
}
