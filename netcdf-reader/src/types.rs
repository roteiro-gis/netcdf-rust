/// A NetCDF dimension.
#[derive(Debug, Clone)]
pub struct NcDimension {
    pub name: String,
    pub size: u64,
    pub is_unlimited: bool,
}

/// NetCDF data types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NcType {
    /// NC_BYTE (i8)
    Byte,
    /// NC_CHAR (u8/char)
    Char,
    /// NC_SHORT (i16)
    Short,
    /// NC_INT (i32)
    Int,
    /// NC_FLOAT (f32)
    Float,
    /// NC_DOUBLE (f64)
    Double,
    /// NC_UBYTE (u8, CDF-5)
    UByte,
    /// NC_USHORT (u16, CDF-5)
    UShort,
    /// NC_UINT (u32, CDF-5)
    UInt,
    /// NC_INT64 (i64, CDF-5)
    Int64,
    /// NC_UINT64 (u64, CDF-5)
    UInt64,
    /// NetCDF-4 only (variable-length string)
    String,
}

impl NcType {
    /// Size of a single element in bytes.
    pub fn size(&self) -> usize {
        match self {
            NcType::Byte | NcType::Char | NcType::UByte => 1,
            NcType::Short | NcType::UShort => 2,
            NcType::Int | NcType::UInt | NcType::Float => 4,
            NcType::Int64 | NcType::UInt64 | NcType::Double => 8,
            // Variable-length string; no fixed element size, but pointer-sized in memory.
            NcType::String => std::mem::size_of::<usize>(),
        }
    }

    /// The numeric type code used in CDF-1/2/5 headers.
    pub fn classic_type_code(&self) -> Option<u32> {
        match self {
            NcType::Byte => Some(1),
            NcType::Char => Some(2),
            NcType::Short => Some(3),
            NcType::Int => Some(4),
            NcType::Float => Some(5),
            NcType::Double => Some(6),
            NcType::UByte => Some(7),
            NcType::UShort => Some(8),
            NcType::UInt => Some(9),
            NcType::Int64 => Some(10),
            NcType::UInt64 => Some(11),
            NcType::String => None, // Not valid in classic format
        }
    }
}

/// A NetCDF attribute value.
#[derive(Debug, Clone)]
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
    /// Get the value as a string (for Chars or single-element Strings).
    pub fn as_string(&self) -> Option<String> {
        match self {
            NcAttrValue::Chars(s) => Some(s.clone()),
            NcAttrValue::Strings(v) if v.len() == 1 => Some(v[0].clone()),
            _ => None,
        }
    }

    /// Get the value as f64 (with numeric promotion from the first element).
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

    /// Get the value as a vector of f64 (with numeric promotion).
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
#[derive(Debug, Clone)]
pub struct NcAttribute {
    pub name: String,
    pub value: NcAttrValue,
}

/// A NetCDF variable (metadata only -- data is read on demand).
#[derive(Debug, Clone)]
pub struct NcVariable {
    pub name: String,
    pub dimensions: Vec<NcDimension>,
    pub dtype: NcType,
    pub attributes: Vec<NcAttribute>,
    /// For classic: file byte offset to the start of this variable's data.
    /// For nc4: HDF5 dataset object header address.
    pub(crate) data_offset: u64,
    /// Total data size in bytes (for non-record variables).
    pub(crate) _data_size: u64,
    /// Whether this variable uses the unlimited (record) dimension.
    pub(crate) is_record_var: bool,
    /// Size of one record slice in bytes (only meaningful for record variables).
    pub(crate) record_size: u64,
}

impl NcVariable {
    /// Variable name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Variable dimensions.
    pub fn dimensions(&self) -> &[NcDimension] {
        &self.dimensions
    }

    /// Variable data type.
    pub fn dtype(&self) -> NcType {
        self.dtype
    }

    /// Shape of the variable as a vector of dimension sizes.
    pub fn shape(&self) -> Vec<u64> {
        self.dimensions.iter().map(|d| d.size).collect()
    }

    /// Variable attributes.
    pub fn attributes(&self) -> &[NcAttribute] {
        &self.attributes
    }

    /// Find an attribute by name.
    pub fn attribute(&self, name: &str) -> Option<&NcAttribute> {
        self.attributes.iter().find(|a| a.name == name)
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.dimensions.len()
    }

    /// Total number of elements.
    pub fn num_elements(&self) -> u64 {
        if self.dimensions.is_empty() {
            return 1; // scalar
        }
        self.dimensions.iter().map(|d| d.size).product()
    }
}

/// A NetCDF group (NetCDF-4 only; classic files have one implicit root group).
#[derive(Debug, Clone)]
pub struct NcGroup {
    pub name: String,
    pub dimensions: Vec<NcDimension>,
    pub variables: Vec<NcVariable>,
    pub attributes: Vec<NcAttribute>,
    pub groups: Vec<NcGroup>,
}

impl NcGroup {
    /// Find a variable by name in this group.
    pub fn variable(&self, name: &str) -> Option<&NcVariable> {
        self.variables.iter().find(|v| v.name == name)
    }

    /// Find a dimension by name in this group.
    pub fn dimension(&self, name: &str) -> Option<&NcDimension> {
        self.dimensions.iter().find(|d| d.name == name)
    }

    /// Find an attribute by name in this group.
    pub fn attribute(&self, name: &str) -> Option<&NcAttribute> {
        self.attributes.iter().find(|a| a.name == name)
    }

    /// Find a child group by name.
    pub fn group(&self, name: &str) -> Option<&NcGroup> {
        self.groups.iter().find(|g| g.name == name)
    }
}
