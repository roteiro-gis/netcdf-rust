//! Pure-Rust NetCDF writer.
//!
//! The current implementation writes CDF-1, CDF-2, and CDF-5 classic-family
//! files, plus a conservative NetCDF-4/HDF5 subset backed by `hdf5-writer`.
//! NetCDF-4 emission is intentionally strict about metadata it cannot yet
//! represent losslessly.

#[cfg(feature = "netcdf4")]
use std::io::Cursor as IoCursor;
use std::io::Write;

#[cfg(feature = "netcdf4")]
use hdf5_writer::{
    AttributeBuilder as H5AttributeBuilder, ByteOrder as H5ByteOrder,
    CompoundField as H5CompoundField, DatasetBuilder as H5DatasetBuilder, Datatype as H5Datatype,
    EnumMember as H5EnumMember, FilterDescription as H5FilterDescription, Hdf5Builder, Hdf5Writer,
    ReferenceType as H5ReferenceType, StringEncoding as H5StringEncoding,
    StringPadding as H5StringPadding, StringSize as H5StringSize, VarLenKind as H5VarLenKind,
    WriteOptions as H5WriteOptions, FILTER_DEFLATE as H5_FILTER_DEFLATE,
    FILTER_FLETCHER32 as H5_FILTER_FLETCHER32, FILTER_SHUFFLE as H5_FILTER_SHUFFLE,
    UNLIMITED as H5_UNLIMITED,
};

pub use netcdf_core::{
    NcAttrValue, NcAttribute, NcCompoundField, NcDimension, NcEnumMember, NcFormat, NcGroup,
    NcIntegerValue, NcSliceInfo, NcSliceInfoElem, NcType, NcVariable, NC_FILL_BYTE, NC_FILL_CHAR,
    NC_FILL_DOUBLE, NC_FILL_FLOAT, NC_FILL_INT, NC_FILL_INT64, NC_FILL_SHORT, NC_FILL_UBYTE,
    NC_FILL_UINT, NC_FILL_UINT64, NC_FILL_USHORT,
};

const ABSENT: u32 = 0x0000_0000;
const NC_DIMENSION: u32 = 0x0000_000A;
const NC_VARIABLE: u32 = 0x0000_000B;
const NC_ATTRIBUTE: u32 = 0x0000_000C;

/// NetCDF writer errors.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("NetCDF core error: {0}")]
    Core(#[from] netcdf_core::Error),

    #[error("invalid definition: {0}")]
    InvalidDefinition(String),

    #[error("type mismatch: expected {expected}, got {actual}")]
    TypeMismatch { expected: String, actual: String },

    #[error("data length mismatch: expected {expected}, got {actual}")]
    DataLengthMismatch { expected: usize, actual: usize },

    #[error("unsupported write feature: {0}")]
    UnsupportedFeature(String),
}

pub type Result<T> = std::result::Result<T, Error>;

/// Requested NetCDF output format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NcWriteFormat {
    /// Pick CDF-1, CDF-2, or CDF-5 from the exact schema and layout.
    AutoClassic,
    Classic,
    Offset64,
    Cdf5,
    Nc4,
    Nc4Classic,
}

/// NetCDF write options.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NcWriteOptions {
    pub format: NcWriteFormat,
}

impl Default for NcWriteOptions {
    fn default() -> Self {
        Self {
            format: NcWriteFormat::AutoClassic,
        }
    }
}

impl NcWriteOptions {
    pub fn classic() -> Self {
        Self {
            format: NcWriteFormat::Classic,
        }
    }

    pub fn offset64() -> Self {
        Self {
            format: NcWriteFormat::Offset64,
        }
    }

    pub fn cdf5() -> Self {
        Self {
            format: NcWriteFormat::Cdf5,
        }
    }
}

/// Dimension handle returned by [`NcFileBuilder::add_dimension`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DimensionId(usize);

/// Variable handle returned by [`NcFileBuilder::add_variable`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct VariableId(usize);

/// Rust types that can be written as NetCDF classic-family variable data.
pub trait NcWriteType: Copy {
    fn nc_type() -> NcType;
    fn write_one_be(self, dst: &mut Vec<u8>);

    fn write_one_le(self, dst: &mut Vec<u8>) {
        let start = dst.len();
        self.write_one_be(dst);
        dst[start..].reverse();
    }
}

/// Rust primitive types that can be used as a NetCDF variable fill value.
///
/// Setting a fill value writes the canonical `_FillValue` variable attribute.
/// NetCDF-4 output also records the value as an HDF5 dataset fill value, which
/// allows fixed-size datasets to be represented without eagerly materializing
/// every element.
pub trait NcFillValueType: NcWriteType {
    fn fill_attr_value(self) -> NcAttrValue;
}

macro_rules! impl_write_type {
    ($ty:ty, $nc_type:expr, $write:expr) => {
        impl NcWriteType for $ty {
            fn nc_type() -> NcType {
                $nc_type
            }

            fn write_one_be(self, dst: &mut Vec<u8>) {
                $write(self, dst)
            }
        }
    };
}

macro_rules! impl_fill_value_type {
    ($ty:ty, $attr:ident) => {
        impl NcFillValueType for $ty {
            fn fill_attr_value(self) -> NcAttrValue {
                NcAttrValue::$attr(vec![self])
            }
        }
    };
}

impl_write_type!(i8, NcType::Byte, |value: i8, dst: &mut Vec<u8>| dst
    .push(value as u8));
impl_write_type!(u8, NcType::UByte, |value: u8, dst: &mut Vec<u8>| dst
    .push(value));
impl_write_type!(i16, NcType::Short, |value: i16, dst: &mut Vec<u8>| dst
    .extend_from_slice(&value.to_be_bytes()));
impl_write_type!(u16, NcType::UShort, |value: u16, dst: &mut Vec<u8>| dst
    .extend_from_slice(&value.to_be_bytes()));
impl_write_type!(i32, NcType::Int, |value: i32, dst: &mut Vec<u8>| dst
    .extend_from_slice(&value.to_be_bytes()));
impl_write_type!(u32, NcType::UInt, |value: u32, dst: &mut Vec<u8>| dst
    .extend_from_slice(&value.to_be_bytes()));
impl_write_type!(i64, NcType::Int64, |value: i64, dst: &mut Vec<u8>| dst
    .extend_from_slice(&value.to_be_bytes()));
impl_write_type!(u64, NcType::UInt64, |value: u64, dst: &mut Vec<u8>| dst
    .extend_from_slice(&value.to_be_bytes()));
impl_write_type!(f32, NcType::Float, |value: f32, dst: &mut Vec<u8>| dst
    .extend_from_slice(&value.to_be_bytes()));
impl_write_type!(f64, NcType::Double, |value: f64, dst: &mut Vec<u8>| dst
    .extend_from_slice(&value.to_be_bytes()));

impl_fill_value_type!(i8, Bytes);
impl_fill_value_type!(u8, UBytes);
impl_fill_value_type!(i16, Shorts);
impl_fill_value_type!(u16, UShorts);
impl_fill_value_type!(i32, Ints);
impl_fill_value_type!(u32, UInts);
impl_fill_value_type!(i64, Int64s);
impl_fill_value_type!(u64, UInt64s);
impl_fill_value_type!(f32, Floats);
impl_fill_value_type!(f64, Doubles);

const FILL_VALUE_ATTR_NAME: &str = "_FillValue";

#[derive(Debug, Clone)]
struct DimensionDef {
    name: String,
    size: u64,
    is_unlimited: bool,
}

#[derive(Debug, Clone)]
struct VariableDef {
    name: String,
    dim_ids: Vec<DimensionId>,
    dtype: NcType,
    attributes: Vec<NcAttribute>,
    data: Vec<u8>,
    data_encoding: VariableDataEncoding,
    string_values: Option<Vec<String>>,
    vlen_values: Option<Vec<Vec<u8>>>,
    storage: VariableStorageDef,
    fill_value: Option<VariableFillValue>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VariableDataEncoding {
    ClassicBigEndian,
    Hdf5Native,
}

#[derive(Debug, Clone)]
struct VariableFillValue {
    classic_bytes: Vec<u8>,
    #[cfg(feature = "netcdf4")]
    hdf5_bytes: Vec<u8>,
}

#[derive(Debug, Clone, Default)]
struct VariableStorageDef {
    chunk_shape: Option<Vec<u64>>,
    shuffle: bool,
    deflate_level: Option<u8>,
    fletcher32: bool,
}

impl VariableStorageDef {
    fn has_filters(&self) -> bool {
        self.shuffle || self.deflate_level.is_some() || self.fletcher32
    }

    fn has_nc4_options(&self) -> bool {
        self.chunk_shape.is_some() || self.has_filters()
    }
}

/// Builder for a single NetCDF file.
#[derive(Debug, Clone, Default)]
pub struct NcFileBuilder {
    dimensions: Vec<DimensionDef>,
    attributes: Vec<NcAttribute>,
    variables: Vec<VariableDef>,
}

impl NcFileBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_dimension(&mut self, name: impl Into<String>, size: u64) -> Result<DimensionId> {
        let name = name.into();
        validate_name(&name, "dimension")?;
        if size == 0 {
            return Err(Error::InvalidDefinition(
                "fixed dimensions must have non-zero size; use add_unlimited_dimension".into(),
            ));
        }
        self.ensure_unique_dimension(&name)?;
        let id = DimensionId(self.dimensions.len());
        self.dimensions.push(DimensionDef {
            name,
            size,
            is_unlimited: false,
        });
        Ok(id)
    }

    pub fn add_unlimited_dimension(&mut self, name: impl Into<String>) -> Result<DimensionId> {
        let name = name.into();
        validate_name(&name, "dimension")?;
        self.ensure_unique_dimension(&name)?;
        let id = DimensionId(self.dimensions.len());
        self.dimensions.push(DimensionDef {
            name,
            size: 0,
            is_unlimited: true,
        });
        Ok(id)
    }

    pub fn add_attribute(&mut self, name: impl Into<String>, value: NcAttrValue) -> Result<()> {
        let name = name.into();
        validate_name(&name, "attribute")?;
        ensure_unique_attr(&self.attributes, &name)?;
        self.attributes.push(NcAttribute { name, value });
        Ok(())
    }

    pub fn add_variable<T: NcWriteType>(
        &mut self,
        name: impl Into<String>,
        dimensions: &[DimensionId],
    ) -> Result<VariableId> {
        self.add_variable_with_type(name, dimensions, T::nc_type())
    }

    pub fn add_char_variable(
        &mut self,
        name: impl Into<String>,
        dimensions: &[DimensionId],
    ) -> Result<VariableId> {
        self.add_variable_with_type(name, dimensions, NcType::Char)
    }

    pub fn add_string_variable(
        &mut self,
        name: impl Into<String>,
        dimensions: &[DimensionId],
    ) -> Result<VariableId> {
        self.add_variable_with_type(name, dimensions, NcType::String)
    }

    pub fn add_user_defined_variable(
        &mut self,
        name: impl Into<String>,
        dimensions: &[DimensionId],
        dtype: NcType,
    ) -> Result<VariableId> {
        validate_supported_user_defined_type(&dtype)?;
        self.add_variable_with_type(name, dimensions, dtype)
    }

    pub fn add_variable_attribute(
        &mut self,
        variable: VariableId,
        name: impl Into<String>,
        value: NcAttrValue,
    ) -> Result<()> {
        let name = name.into();
        validate_name(&name, "attribute")?;
        let variable = self.variable_mut(variable)?;
        ensure_unique_attr(&variable.attributes, &name)?;
        variable.attributes.push(NcAttribute { name, value });
        Ok(())
    }

    pub fn set_variable_fill_value<T: NcFillValueType>(
        &mut self,
        variable: VariableId,
        value: T,
    ) -> Result<()> {
        let variable = self.variable_mut(variable)?;
        let expected = T::nc_type();
        if variable.dtype != expected {
            return Err(Error::TypeMismatch {
                expected: format!("{:?}", variable.dtype),
                actual: format!("{expected:?}"),
            });
        }

        let mut classic_bytes = Vec::with_capacity(expected.size()?);
        value.write_one_be(&mut classic_bytes);
        let mut hdf5_bytes = Vec::with_capacity(expected.size()?);
        value.write_one_le(&mut hdf5_bytes);
        set_variable_fill_value_metadata(
            variable,
            value.fill_attr_value(),
            classic_bytes,
            hdf5_bytes,
        )
    }

    pub fn set_variable_chunking(
        &mut self,
        variable: VariableId,
        chunk_shape: impl Into<Vec<u64>>,
    ) -> Result<()> {
        let variable = self.variable_mut(variable)?;
        let chunk_shape = chunk_shape.into();
        validate_chunk_shape_for_variable(variable, &chunk_shape)?;
        variable.storage.chunk_shape = Some(chunk_shape);
        Ok(())
    }

    pub fn set_variable_shuffle(&mut self, variable: VariableId, enabled: bool) -> Result<()> {
        let variable = self.variable_mut(variable)?;
        reject_scalar_filtered_variable(variable)?;
        variable.storage.shuffle = enabled;
        Ok(())
    }

    pub fn set_variable_deflate(
        &mut self,
        variable: VariableId,
        level: Option<u8>,
        shuffle: bool,
    ) -> Result<()> {
        if let Some(level) = level {
            if level > 9 {
                return Err(Error::InvalidDefinition(
                    "deflate level must be in 0..=9".into(),
                ));
            }
        }
        let variable = self.variable_mut(variable)?;
        reject_scalar_filtered_variable(variable)?;
        variable.storage.deflate_level = level;
        variable.storage.shuffle = shuffle;
        Ok(())
    }

    pub fn set_variable_fletcher32(&mut self, variable: VariableId, enabled: bool) -> Result<()> {
        let variable = self.variable_mut(variable)?;
        reject_scalar_filtered_variable(variable)?;
        variable.storage.fletcher32 = enabled;
        Ok(())
    }

    pub fn write_variable<T: NcWriteType>(
        &mut self,
        variable: VariableId,
        values: &[T],
    ) -> Result<()> {
        let variable = self.variable_mut(variable)?;
        let expected = T::nc_type();
        if variable.dtype != expected {
            return Err(Error::TypeMismatch {
                expected: format!("{:?}", variable.dtype),
                actual: format!("{:?}", expected),
            });
        }
        let mut data = Vec::with_capacity(std::mem::size_of_val(values));
        for &value in values {
            value.write_one_be(&mut data);
        }
        variable.data = data;
        variable.data_encoding = VariableDataEncoding::ClassicBigEndian;
        variable.string_values = None;
        variable.vlen_values = None;
        Ok(())
    }

    pub fn write_variable_slice<T: NcWriteType>(
        &mut self,
        variable: VariableId,
        selection: &NcSliceInfo,
        values: &[T],
    ) -> Result<()> {
        let variable_def = self.variable(variable)?;
        let expected = T::nc_type();
        if variable_def.dtype != expected {
            return Err(Error::TypeMismatch {
                expected: format!("{:?}", variable_def.dtype),
                actual: format!("{expected:?}"),
            });
        }
        let shape = self.fixed_variable_shape(variable_def)?;
        let resolved = resolve_write_selection(&variable_def.name, &shape, selection)?;
        if values.len() != resolved.elements {
            return Err(Error::DataLengthMismatch {
                expected: resolved.elements,
                actual: values.len(),
            });
        }

        let elem_size = expected.size()?;
        let mut encoded = Vec::with_capacity(checked_mul_usize(
            values.len(),
            elem_size,
            "variable slice byte size",
        )?);
        for &value in values {
            value.write_one_be(&mut encoded);
        }

        let strides = row_major_strides(&shape, "writer slice stride")?;
        let total_elements = checked_shape_elements(&shape, "writer variable element count")?;
        let variable_def = self.variable_mut(variable)?;
        ensure_variable_slice_buffer(variable_def, total_elements, elem_size)?;
        scatter_slice_bytes(
            &mut variable_def.data,
            elem_size,
            &resolved.dims,
            &strides,
            &encoded,
        )
    }

    pub fn write_char_variable(&mut self, variable: VariableId, bytes: &[u8]) -> Result<()> {
        let variable = self.variable_mut(variable)?;
        if variable.dtype != NcType::Char {
            return Err(Error::TypeMismatch {
                expected: "Char".into(),
                actual: format!("{:?}", variable.dtype),
            });
        }
        variable.data = bytes.to_vec();
        variable.data_encoding = VariableDataEncoding::ClassicBigEndian;
        variable.string_values = None;
        variable.vlen_values = None;
        Ok(())
    }

    pub fn write_char_variable_slice(
        &mut self,
        variable: VariableId,
        selection: &NcSliceInfo,
        bytes: &[u8],
    ) -> Result<()> {
        let variable_def = self.variable(variable)?;
        if variable_def.dtype != NcType::Char {
            return Err(Error::TypeMismatch {
                expected: "Char".into(),
                actual: format!("{:?}", variable_def.dtype),
            });
        }
        let shape = self.fixed_variable_shape(variable_def)?;
        let resolved = resolve_write_selection(&variable_def.name, &shape, selection)?;
        if bytes.len() != resolved.elements {
            return Err(Error::DataLengthMismatch {
                expected: resolved.elements,
                actual: bytes.len(),
            });
        }

        let strides = row_major_strides(&shape, "writer char slice stride")?;
        let total_elements = checked_shape_elements(&shape, "writer char variable element count")?;
        let variable_def = self.variable_mut(variable)?;
        ensure_variable_slice_buffer(variable_def, total_elements, 1)?;
        scatter_slice_bytes(&mut variable_def.data, 1, &resolved.dims, &strides, bytes)
    }

    pub fn write_user_defined_variable_bytes(
        &mut self,
        variable: VariableId,
        bytes: &[u8],
    ) -> Result<()> {
        let variable = self.variable_mut(variable)?;
        validate_fixed_width_user_defined_type(&variable.dtype)?;
        variable.data = bytes.to_vec();
        variable.data_encoding = VariableDataEncoding::Hdf5Native;
        variable.string_values = None;
        variable.vlen_values = None;
        Ok(())
    }

    pub fn write_enum_variable(
        &mut self,
        variable: VariableId,
        values: &[NcIntegerValue],
    ) -> Result<()> {
        let variable = self.variable_mut(variable)?;
        let NcType::Enum { base, .. } = &variable.dtype else {
            return Err(Error::TypeMismatch {
                expected: "Enum".into(),
                actual: format!("{:?}", variable.dtype),
            });
        };
        let mut data = Vec::with_capacity(values.len() * base.size()?);
        for &value in values {
            data.extend_from_slice(&nc_enum_value_to_le_bytes(base, value)?);
        }
        variable.data = data;
        variable.data_encoding = VariableDataEncoding::Hdf5Native;
        variable.string_values = None;
        variable.vlen_values = None;
        Ok(())
    }

    pub fn write_opaque_variable<S: AsRef<[u8]>>(
        &mut self,
        variable: VariableId,
        values: &[S],
    ) -> Result<()> {
        let variable = self.variable_mut(variable)?;
        let NcType::Opaque { size, .. } = &variable.dtype else {
            return Err(Error::TypeMismatch {
                expected: "Opaque".into(),
                actual: format!("{:?}", variable.dtype),
            });
        };
        let size = usize::try_from(*size).map_err(|_| {
            Error::InvalidDefinition("opaque element size exceeds platform usize".into())
        })?;
        let mut data = Vec::with_capacity(values.len() * size);
        for value in values {
            let value = value.as_ref();
            if value.len() != size {
                return Err(Error::DataLengthMismatch {
                    expected: size,
                    actual: value.len(),
                });
            }
            data.extend_from_slice(value);
        }
        variable.data = data;
        variable.data_encoding = VariableDataEncoding::Hdf5Native;
        variable.string_values = None;
        variable.vlen_values = None;
        Ok(())
    }

    pub fn write_array_variable<T: NcWriteType>(
        &mut self,
        variable: VariableId,
        values: &[T],
    ) -> Result<()> {
        let variable = self.variable_mut(variable)?;
        let NcType::Array { base, .. } = &variable.dtype else {
            return Err(Error::TypeMismatch {
                expected: "Array".into(),
                actual: format!("{:?}", variable.dtype),
            });
        };
        let expected = T::nc_type();
        if base.as_ref() != &expected {
            return Err(Error::TypeMismatch {
                expected: format!("{:?}", base),
                actual: format!("{expected:?}"),
            });
        }
        let mut data = Vec::with_capacity(std::mem::size_of_val(values));
        for &value in values {
            value.write_one_le(&mut data);
        }
        variable.data = data;
        variable.data_encoding = VariableDataEncoding::Hdf5Native;
        variable.string_values = None;
        variable.vlen_values = None;
        Ok(())
    }

    pub fn write_vlen_variable_bytes<S: AsRef<[u8]>>(
        &mut self,
        variable: VariableId,
        values: &[S],
    ) -> Result<()> {
        let variable = self.variable_mut(variable)?;
        let NcType::VLen { base } = &variable.dtype else {
            return Err(Error::TypeMismatch {
                expected: "VLen".into(),
                actual: format!("{:?}", variable.dtype),
            });
        };
        validate_vlen_base_nc4_type(base)?;
        let base_size = base.size()?;
        let mut sequences = Vec::with_capacity(values.len());
        for value in values {
            let value = value.as_ref();
            if value.len() % base_size != 0 {
                return Err(Error::InvalidDefinition(format!(
                    "vlen sequence byte length {} is not a multiple of base element size {base_size}",
                    value.len()
                )));
            }
            sequences.push(value.to_vec());
        }
        variable.data.clear();
        variable.data_encoding = VariableDataEncoding::Hdf5Native;
        variable.string_values = None;
        variable.vlen_values = Some(sequences);
        Ok(())
    }

    pub fn write_vlen_variable<T: NcWriteType>(
        &mut self,
        variable: VariableId,
        values: &[Vec<T>],
    ) -> Result<()> {
        let variable_def = self.variable_mut(variable)?;
        let NcType::VLen { base } = &variable_def.dtype else {
            return Err(Error::TypeMismatch {
                expected: "VLen".into(),
                actual: format!("{:?}", variable_def.dtype),
            });
        };
        let expected = T::nc_type();
        if base.as_ref() != &expected {
            return Err(Error::TypeMismatch {
                expected: format!("{:?}", base),
                actual: format!("{expected:?}"),
            });
        }

        let mut sequences = Vec::with_capacity(values.len());
        for sequence in values {
            let mut bytes = Vec::with_capacity(std::mem::size_of_val(sequence.as_slice()));
            for &value in sequence {
                value.write_one_le(&mut bytes);
            }
            sequences.push(bytes);
        }
        variable_def.data.clear();
        variable_def.data_encoding = VariableDataEncoding::Hdf5Native;
        variable_def.string_values = None;
        variable_def.vlen_values = Some(sequences);
        Ok(())
    }

    pub fn write_string_variable<S: AsRef<str>>(
        &mut self,
        variable: VariableId,
        values: &[S],
    ) -> Result<()> {
        let variable = self.variable_mut(variable)?;
        if variable.dtype != NcType::String {
            return Err(Error::TypeMismatch {
                expected: "String".into(),
                actual: format!("{:?}", variable.dtype),
            });
        }

        let mut strings = Vec::with_capacity(values.len());
        for value in values {
            let value = value.as_ref();
            if value.as_bytes().contains(&0) {
                return Err(Error::InvalidDefinition(
                    "NC_STRING variable values cannot contain NUL bytes".into(),
                ));
            }
            strings.push(value.to_string());
        }
        variable.data.clear();
        variable.data_encoding = VariableDataEncoding::Hdf5Native;
        variable.string_values = Some(strings);
        variable.vlen_values = None;
        Ok(())
    }

    pub fn write<W: Write>(&self, mut writer: W, options: NcWriteOptions) -> Result<NcFormat> {
        if options.format == NcWriteFormat::AutoClassic {
            return self.write_auto_classic(&mut writer);
        }

        let format = self.select_format(options)?;
        match format {
            NcFormat::Classic | NcFormat::Offset64 | NcFormat::Cdf5 => {
                let plan = ClassicWritePlan::build(self, format)?;
                plan.write(&mut writer)?;
                Ok(format)
            }
            NcFormat::Nc4 | NcFormat::Nc4Classic => self.write_nc4(&mut writer, format),
        }
    }

    pub fn to_vec(&self, options: NcWriteOptions) -> Result<(NcFormat, Vec<u8>)> {
        let mut data = Vec::new();
        let format = self.write(&mut data, options)?;
        Ok((format, data))
    }

    fn write_auto_classic(&self, writer: &mut impl Write) -> Result<NcFormat> {
        if self.requires_cdf5() {
            self.validate_for_format(NcFormat::Cdf5)?;
            let plan = ClassicWritePlan::build(self, NcFormat::Cdf5)?;
            plan.write(writer)?;
            return Ok(NcFormat::Cdf5);
        }

        match ClassicWritePlan::build(self, NcFormat::Classic) {
            Ok(plan) => {
                plan.write(writer)?;
                Ok(NcFormat::Classic)
            }
            Err(classic_err) => match ClassicWritePlan::build(self, NcFormat::Offset64) {
                Ok(plan) => {
                    plan.write(writer)?;
                    Ok(NcFormat::Offset64)
                }
                Err(_) => Err(classic_err),
            },
        }
    }

    #[cfg(feature = "netcdf4")]
    fn write_nc4(&self, writer: &mut impl Write, format: NcFormat) -> Result<NcFormat> {
        self.validate_for_nc4_bridge(format)?;
        let hdf5_plan = self.build_hdf5_plan(format)?;
        let cursor = Hdf5Writer::new(IoCursor::new(Vec::new()), H5WriteOptions::default())
            .finish(hdf5_plan)
            .map_err(hdf5_error_to_unsupported)?;
        writer.write_all(&cursor.into_inner())?;
        Ok(format)
    }

    #[cfg(not(feature = "netcdf4"))]
    fn write_nc4(&self, _writer: &mut impl Write, _format: NcFormat) -> Result<NcFormat> {
        Err(Error::UnsupportedFeature(
            "NetCDF-4 writing requires the netcdf4 feature".into(),
        ))
    }

    #[cfg(feature = "netcdf4")]
    fn validate_for_nc4_bridge(&self, format: NcFormat) -> Result<()> {
        if format == NcFormat::Nc4Classic {
            validate_nc4_classic_model(self)?;
            for variable in &self.variables {
                validate_classic_type(&variable.dtype)?;
                for attr in &variable.attributes {
                    validate_nc4_classic_attr_value(&attr.value)?;
                }
            }
            for attr in &self.attributes {
                validate_nc4_classic_attr_value(&attr.value)?;
            }
        }
        let dimension_sizes = self.nc4_dimension_sizes()?;
        for variable in &self.variables {
            validate_nc4_variable_data(variable, &dimension_sizes)?;
        }
        for (dim_id, dimension) in self.dimensions.iter().enumerate() {
            let dimension_id = DimensionId(dim_id);
            if self
                .coordinate_variable_for_dimension(dimension_id)
                .is_none()
                && self
                    .variables
                    .iter()
                    .any(|variable| variable.name == dimension.name)
            {
                return Err(Error::UnsupportedFeature(format!(
                    "NetCDF-4 dimension '{}' needs a hidden dimension-scale dataset, but a scalar variable already uses that name",
                    dimension.name
                )));
            }
        }
        Ok(())
    }

    #[cfg(feature = "netcdf4")]
    fn build_hdf5_plan(&self, format: NcFormat) -> Result<hdf5_writer::Hdf5WritePlan> {
        let mut builder = Hdf5Builder::new();
        let dimension_sizes = self.nc4_dimension_sizes()?;
        if format == NcFormat::Nc4Classic {
            builder = builder.attribute(
                H5AttributeBuilder::scalar("_nc3_strict", 1_i32)
                    .map_err(hdf5_error_to_unsupported)?,
            );
        }
        for attr in &self.attributes {
            builder = builder.attribute(nc_attr_to_hdf5(attr)?);
        }

        for (dim_id, dimension) in self.dimensions.iter().enumerate() {
            if self
                .coordinate_variable_for_dimension(DimensionId(dim_id))
                .is_some()
            {
                continue;
            }
            let size = dimension_sizes[dim_id];
            let zeros = vec![0_i32; checked_usize(size, "dimension scale size")?];
            let mut dataset = H5DatasetBuilder::typed_data(&dimension.name, vec![size], &zeros)
                .map_err(hdf5_error_to_unsupported)?
                .attribute(H5AttributeBuilder::fixed_string("CLASS", "DIMENSION_SCALE"))
                .attribute(H5AttributeBuilder::fixed_string(
                    "NAME",
                    format!(
                        "This is a netCDF dimension but not a netCDF variable. {}",
                        dimension.name
                    ),
                ))
                .attribute(
                    H5AttributeBuilder::scalar("_Netcdf4Dimid", dim_id as i32)
                        .map_err(hdf5_error_to_unsupported)?,
                );
            if dimension.is_unlimited {
                dataset = dataset
                    .max_shape(vec![H5_UNLIMITED])
                    .chunked(nc4_default_chunk_shape(
                        &[size],
                        std::mem::size_of::<i32>(),
                    )?);
            }
            builder = builder.dataset(dataset);
        }

        for variable in &self.variables {
            let shape = variable
                .dim_ids
                .iter()
                .map(|id| dimension_sizes[id.0])
                .collect::<Vec<_>>();
            let mut dataset = if variable.dtype == NcType::String {
                let values = variable.string_values.as_ref().ok_or_else(|| {
                    Error::InvalidDefinition(format!(
                        "NC_STRING variable '{}' has no string data",
                        variable.name
                    ))
                })?;
                H5DatasetBuilder::vlen_string_data(&variable.name, shape.clone(), values)
                    .map_err(hdf5_error_to_unsupported)?
            } else if let NcType::VLen { base } = &variable.dtype {
                let values = variable.vlen_values.as_ref().ok_or_else(|| {
                    Error::InvalidDefinition(format!(
                        "NC_VLEN variable '{}' has no sequence data",
                        variable.name
                    ))
                })?;
                H5DatasetBuilder::vlen_sequence_data(
                    &variable.name,
                    nc_type_to_hdf5(base)?,
                    shape.clone(),
                    values.clone(),
                )
                .map_err(hdf5_error_to_unsupported)?
            } else {
                let data = match variable.data_encoding {
                    VariableDataEncoding::ClassicBigEndian => {
                        convert_classic_be_data_to_hdf5_le(&variable.dtype, &variable.data)?
                    }
                    VariableDataEncoding::Hdf5Native => variable.data.clone(),
                };
                let dataset = H5DatasetBuilder::new(
                    &variable.name,
                    nc_type_to_hdf5(&variable.dtype)?,
                    shape.clone(),
                );
                if data.is_empty() && variable.fill_value.is_some() {
                    dataset
                } else {
                    dataset.raw_data(data)
                }
            };
            if let Some(fill_value) = &variable.fill_value {
                dataset = dataset.fill_value(fill_value.hdf5_bytes.clone());
            }
            let chunk_shape = nc4_variable_chunk_shape(
                variable,
                &shape,
                self.nc4_variable_max_shape(variable).is_some(),
            )?;
            if let Some(max_shape) = self.nc4_variable_max_shape(variable) {
                dataset = dataset.max_shape(max_shape);
            }
            if let Some(chunk_shape) = chunk_shape {
                dataset = dataset.chunked(chunk_shape);
            }
            for filter in nc4_variable_filters(variable) {
                dataset = dataset.filter(filter);
            }

            if self.variable_is_coordinate_scale(variable)? {
                let dim_id = variable.dim_ids[0];
                dataset = dataset
                    .attribute(H5AttributeBuilder::fixed_string("CLASS", "DIMENSION_SCALE"))
                    .attribute(H5AttributeBuilder::fixed_string(
                        "NAME",
                        self.dimensions[dim_id.0].name.as_str(),
                    ))
                    .attribute(
                        H5AttributeBuilder::scalar("_Netcdf4Dimid", dim_id.0 as i32)
                            .map_err(hdf5_error_to_unsupported)?,
                    );
            } else if variable.dim_ids.is_empty() {
                dataset = dataset.attribute(empty_dimension_list_attribute());
            } else {
                dataset = dataset.attribute(self.dimension_list_attribute(variable)?);
            }

            for attr in &variable.attributes {
                dataset = dataset.attribute(nc_attr_to_hdf5(attr)?);
            }

            builder = builder.dataset(dataset);
        }

        builder.into_plan().map_err(hdf5_error_to_unsupported)
    }

    #[cfg(feature = "netcdf4")]
    fn nc4_dimension_sizes(&self) -> Result<Vec<u64>> {
        infer_nc4_dimension_sizes(self)
    }

    #[cfg(feature = "netcdf4")]
    fn nc4_variable_max_shape(&self, variable: &VariableDef) -> Option<Vec<u64>> {
        variable
            .dim_ids
            .iter()
            .any(|id| self.dimensions[id.0].is_unlimited)
            .then(|| {
                variable
                    .dim_ids
                    .iter()
                    .map(|id| {
                        let dimension = &self.dimensions[id.0];
                        if dimension.is_unlimited {
                            H5_UNLIMITED
                        } else {
                            dimension.size
                        }
                    })
                    .collect()
            })
    }

    #[cfg(feature = "netcdf4")]
    fn variable_is_coordinate_scale(&self, variable: &VariableDef) -> Result<bool> {
        Ok(match variable.dim_ids.as_slice() {
            [dim_id] => self.dimension(*dim_id)?.name == variable.name,
            _ => false,
        })
    }

    #[cfg(feature = "netcdf4")]
    fn coordinate_variable_for_dimension(&self, dimension: DimensionId) -> Option<&VariableDef> {
        let dim = self.dimensions.get(dimension.0)?;
        self.variables.iter().find(|variable| {
            variable.name == dim.name && variable.dim_ids.as_slice() == [dimension]
        })
    }

    #[cfg(feature = "netcdf4")]
    fn dimension_list_attribute(&self, variable: &VariableDef) -> Result<H5AttributeBuilder> {
        let target_sequences = variable
            .dim_ids
            .iter()
            .map(|dimension| Ok(vec![self.dimension(*dimension)?.name.clone()]))
            .collect::<Result<Vec<_>>>()?;
        Ok(H5AttributeBuilder::vlen_object_references(
            "DIMENSION_LIST",
            target_sequences,
        ))
    }

    fn add_variable_with_type(
        &mut self,
        name: impl Into<String>,
        dimensions: &[DimensionId],
        dtype: NcType,
    ) -> Result<VariableId> {
        let name = name.into();
        validate_name(&name, "variable")?;
        if self.variables.iter().any(|v| v.name == name) {
            return Err(Error::InvalidDefinition(format!(
                "duplicate variable '{name}'"
            )));
        }
        for dim in dimensions {
            self.dimension(*dim)?;
        }
        let id = VariableId(self.variables.len());
        self.variables.push(VariableDef {
            name,
            dim_ids: dimensions.to_vec(),
            dtype,
            attributes: Vec::new(),
            data: Vec::new(),
            data_encoding: VariableDataEncoding::ClassicBigEndian,
            string_values: None,
            vlen_values: None,
            storage: VariableStorageDef::default(),
            fill_value: None,
        });
        Ok(id)
    }

    fn dimension(&self, id: DimensionId) -> Result<&DimensionDef> {
        self.dimensions
            .get(id.0)
            .ok_or_else(|| Error::InvalidDefinition("invalid dimension handle".into()))
    }

    fn variable(&self, id: VariableId) -> Result<&VariableDef> {
        self.variables
            .get(id.0)
            .ok_or_else(|| Error::InvalidDefinition("invalid variable handle".into()))
    }

    fn variable_mut(&mut self, id: VariableId) -> Result<&mut VariableDef> {
        self.variables
            .get_mut(id.0)
            .ok_or_else(|| Error::InvalidDefinition("invalid variable handle".into()))
    }

    fn fixed_variable_shape(&self, variable: &VariableDef) -> Result<Vec<u64>> {
        variable
            .dim_ids
            .iter()
            .map(|id| {
                let dimension = &self.dimensions[id.0];
                if dimension.is_unlimited {
                    return Err(Error::UnsupportedFeature(format!(
                        "slice writes to unlimited dimension '{}' are not supported yet",
                        dimension.name
                    )));
                }
                Ok(dimension.size)
            })
            .collect()
    }

    fn ensure_unique_dimension(&self, name: &str) -> Result<()> {
        if self.dimensions.iter().any(|d| d.name == name) {
            return Err(Error::InvalidDefinition(format!(
                "duplicate dimension '{name}'"
            )));
        }
        Ok(())
    }

    fn select_format(&self, options: NcWriteOptions) -> Result<NcFormat> {
        match options.format {
            NcWriteFormat::Classic => {
                self.validate_for_format(NcFormat::Classic)?;
                Ok(NcFormat::Classic)
            }
            NcWriteFormat::Offset64 => {
                self.validate_for_format(NcFormat::Offset64)?;
                Ok(NcFormat::Offset64)
            }
            NcWriteFormat::Cdf5 => {
                self.validate_for_format(NcFormat::Cdf5)?;
                Ok(NcFormat::Cdf5)
            }
            NcWriteFormat::Nc4 => Ok(NcFormat::Nc4),
            NcWriteFormat::Nc4Classic => Ok(NcFormat::Nc4Classic),
            NcWriteFormat::AutoClassic => {
                let preferred = if self.requires_cdf5() {
                    NcFormat::Cdf5
                } else {
                    NcFormat::Classic
                };
                match self.validate_for_format(preferred) {
                    Ok(()) => Ok(preferred),
                    Err(_) if preferred == NcFormat::Classic => {
                        self.validate_for_format(NcFormat::Offset64)?;
                        Ok(NcFormat::Offset64)
                    }
                    Err(err) => Err(err),
                }
            }
        }
    }

    fn requires_cdf5(&self) -> bool {
        self.dimensions.iter().any(|d| d.size > u32::MAX as u64)
            || self.variables.iter().any(|v| {
                matches!(
                    v.dtype,
                    NcType::UByte | NcType::UShort | NcType::UInt | NcType::Int64 | NcType::UInt64
                )
            })
            || self.attributes.iter().any(|a| attr_requires_cdf5(&a.value))
            || self
                .variables
                .iter()
                .flat_map(|v| &v.attributes)
                .any(|a| attr_requires_cdf5(&a.value))
    }

    fn validate_for_format(&self, format: NcFormat) -> Result<()> {
        if !matches!(
            format,
            NcFormat::Classic | NcFormat::Offset64 | NcFormat::Cdf5
        ) {
            return Err(Error::UnsupportedFeature(
                "only classic-family NetCDF formats are implemented".into(),
            ));
        }
        let unlimited_count = self.dimensions.iter().filter(|d| d.is_unlimited).count();
        if unlimited_count > 1 {
            return Err(Error::InvalidDefinition(
                "classic NetCDF supports at most one unlimited dimension".into(),
            ));
        }
        if format != NcFormat::Cdf5 && self.requires_cdf5() {
            return Err(Error::InvalidDefinition(
                "CDF-5 is required for unsigned integer, 64-bit integer, or 64-bit count data"
                    .into(),
            ));
        }
        for variable in &self.variables {
            if variable.storage.has_nc4_options() {
                return Err(Error::UnsupportedFeature(format!(
                    "variable '{}' uses NetCDF-4 storage options",
                    variable.name
                )));
            }
            validate_classic_type(&variable.dtype)?;
            let is_record = variable_is_record(self, variable)?;
            if variable
                .dim_ids
                .iter()
                .skip(1)
                .any(|id| self.dimensions[id.0].is_unlimited)
            {
                return Err(Error::InvalidDefinition(format!(
                    "record dimension must be first for variable '{}'",
                    variable.name
                )));
            }
            let expected = expected_variable_bytes(self, variable, is_record)?;
            if variable.data.len() != expected {
                return Err(Error::DataLengthMismatch {
                    expected,
                    actual: variable.data.len(),
                });
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct PlannedVar {
    def: VariableDef,
    is_record: bool,
    vsize: u64,
    padded_vsize: u64,
    begin: u64,
    fixed_data_len: u64,
    record_slab_len: u64,
}

#[derive(Debug, Clone)]
struct ClassicWritePlan {
    format: NcFormat,
    num_records: u64,
    header: Vec<u8>,
    fixed_vars: Vec<PlannedVar>,
    record_vars: Vec<PlannedVar>,
}

impl ClassicWritePlan {
    fn build(builder: &NcFileBuilder, format: NcFormat) -> Result<Self> {
        builder.validate_for_format(format)?;
        let num_records = infer_num_records(builder)?;
        let mut planned = plan_variables(builder, format, num_records, 0)?;
        let header_without_offsets = encode_header(builder, format, num_records, &planned)?;
        planned = plan_variables(
            builder,
            format,
            num_records,
            header_without_offsets.len() as u64,
        )?;
        let header = encode_header(builder, format, num_records, &planned)?;
        let planned = plan_variables(builder, format, num_records, header.len() as u64)?;
        let header = encode_header(builder, format, num_records, &planned)?;

        if format == NcFormat::Classic {
            for var in &planned {
                require_u32(var.begin, "CDF-1 variable offset")?;
                require_u32(var.vsize, "CDF-1 variable size")?;
            }
        }
        if matches!(format, NcFormat::Classic | NcFormat::Offset64) {
            for var in &planned {
                require_u32(var.vsize, "CDF variable size")?;
            }
            require_u32(num_records, "record count")?;
        }

        let fixed_vars = planned.iter().filter(|v| !v.is_record).cloned().collect();
        let record_vars = planned.iter().filter(|v| v.is_record).cloned().collect();
        Ok(Self {
            format,
            num_records,
            header,
            fixed_vars,
            record_vars,
        })
    }

    fn write(&self, writer: &mut impl Write) -> Result<()> {
        let _ = self.format;
        writer.write_all(&self.header)?;
        for var in &self.fixed_vars {
            writer.write_all(&var.def.data)?;
            write_zero_padding(writer, var.padded_vsize - var.fixed_data_len)?;
        }
        for record in 0..self.num_records {
            for var in &self.record_vars {
                let start = usize::try_from(record * var.record_slab_len).map_err(|_| {
                    Error::InvalidDefinition("record byte offset exceeds platform usize".into())
                })?;
                let slab_len = usize::try_from(var.record_slab_len).map_err(|_| {
                    Error::InvalidDefinition("record slab length exceeds platform usize".into())
                })?;
                writer.write_all(&var.def.data[start..start + slab_len])?;
                write_zero_padding(writer, var.padded_vsize - var.record_slab_len)?;
            }
        }
        Ok(())
    }
}

fn plan_variables(
    builder: &NcFileBuilder,
    format: NcFormat,
    num_records: u64,
    data_start: u64,
) -> Result<Vec<PlannedVar>> {
    let mut fixed_offset = data_start;
    let mut planned = Vec::with_capacity(builder.variables.len());
    for def in &builder.variables {
        let is_record = variable_is_record(builder, def)?;
        let elem_size = def.dtype.size()? as u64;
        let record_slab_len = if is_record {
            variable_shape_elements(builder, def, 1)? * elem_size
        } else {
            0
        };
        let fixed_data_len = if is_record {
            0
        } else {
            variable_shape_elements(builder, def, 0)? * elem_size
        };
        let vsize = if is_record {
            record_slab_len
        } else {
            fixed_data_len
        };
        let padded_vsize = pad4_u64(vsize)?;
        let begin = if is_record { 0 } else { fixed_offset };
        if !is_record {
            fixed_offset = checked_add(fixed_offset, padded_vsize, "fixed data offset")?;
        }
        planned.push(PlannedVar {
            def: def.clone(),
            is_record,
            vsize,
            padded_vsize,
            begin,
            fixed_data_len,
            record_slab_len,
        });
    }

    let record_data_start = fixed_offset;
    let mut record_offset = record_data_start;
    for var in &mut planned {
        if var.is_record {
            var.begin = record_offset;
            record_offset = checked_add(record_offset, var.padded_vsize, "record offset")?;
        }
    }
    let _ = num_records;
    let _ = format;
    Ok(planned)
}

fn encode_header(
    builder: &NcFileBuilder,
    format: NcFormat,
    num_records: u64,
    planned: &[PlannedVar],
) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    out.extend_from_slice(match format {
        NcFormat::Classic => b"CDF\x01",
        NcFormat::Offset64 => b"CDF\x02",
        NcFormat::Cdf5 => b"CDF\x05",
        _ => unreachable!("classic header only"),
    });
    write_count(&mut out, format, num_records)?;
    encode_dimensions(&mut out, builder, format)?;
    encode_attributes(&mut out, format, &builder.attributes)?;
    encode_variables(&mut out, builder, format, planned)?;
    Ok(out)
}

fn encode_dimensions(out: &mut Vec<u8>, builder: &NcFileBuilder, format: NcFormat) -> Result<()> {
    if builder.dimensions.is_empty() {
        out.extend_from_slice(&ABSENT.to_be_bytes());
        write_count(out, format, 0)?;
        return Ok(());
    }
    out.extend_from_slice(&NC_DIMENSION.to_be_bytes());
    write_count(out, format, builder.dimensions.len() as u64)?;
    for dim in &builder.dimensions {
        write_name(out, format, &dim.name)?;
        write_count(out, format, if dim.is_unlimited { 0 } else { dim.size })?;
    }
    Ok(())
}

fn encode_attributes(out: &mut Vec<u8>, format: NcFormat, attrs: &[NcAttribute]) -> Result<()> {
    if attrs.is_empty() {
        out.extend_from_slice(&ABSENT.to_be_bytes());
        write_count(out, format, 0)?;
        return Ok(());
    }
    out.extend_from_slice(&NC_ATTRIBUTE.to_be_bytes());
    write_count(out, format, attrs.len() as u64)?;
    for attr in attrs {
        write_name(out, format, &attr.name)?;
        let (dtype, count, bytes) = encode_attr_value(&attr.value)?;
        write_u32(
            out,
            dtype.classic_type_code().ok_or_else(|| {
                Error::InvalidDefinition(format!("{dtype:?} is not valid in classic attributes"))
            })?,
        );
        write_count(out, format, count)?;
        out.extend_from_slice(&bytes);
        pad_vec_to_4(out);
    }
    Ok(())
}

fn encode_variables(
    out: &mut Vec<u8>,
    builder: &NcFileBuilder,
    format: NcFormat,
    planned: &[PlannedVar],
) -> Result<()> {
    if planned.is_empty() {
        out.extend_from_slice(&ABSENT.to_be_bytes());
        write_count(out, format, 0)?;
        return Ok(());
    }
    out.extend_from_slice(&NC_VARIABLE.to_be_bytes());
    write_count(out, format, planned.len() as u64)?;
    for var in planned {
        write_name(out, format, &var.def.name)?;
        write_count(out, format, var.def.dim_ids.len() as u64)?;
        for dim_id in &var.def.dim_ids {
            write_count(out, format, dim_id.0 as u64)?;
        }
        encode_attributes(out, format, &var.def.attributes)?;
        write_u32(
            out,
            var.def.dtype.classic_type_code().ok_or_else(|| {
                Error::InvalidDefinition(format!(
                    "{:?} is not valid in classic variables",
                    var.def.dtype
                ))
            })?,
        );
        write_count(out, format, var.vsize)?;
        match format {
            NcFormat::Classic => write_u32(out, require_u32(var.begin, "CDF-1 begin")?),
            NcFormat::Offset64 | NcFormat::Cdf5 => write_u64(out, var.begin),
            _ => unreachable!("classic header only"),
        }
        let _ = builder;
    }
    Ok(())
}

fn encode_attr_value(value: &NcAttrValue) -> Result<(NcType, u64, Vec<u8>)> {
    let mut out = Vec::new();
    let (dtype, count) = match value {
        NcAttrValue::Bytes(values) => {
            out.extend(values.iter().map(|v| *v as u8));
            (NcType::Byte, values.len() as u64)
        }
        NcAttrValue::Chars(value) => {
            out.extend_from_slice(value.as_bytes());
            (NcType::Char, value.len() as u64)
        }
        NcAttrValue::Shorts(values) => {
            for value in values {
                out.extend_from_slice(&value.to_be_bytes());
            }
            (NcType::Short, values.len() as u64)
        }
        NcAttrValue::Ints(values) => {
            for value in values {
                out.extend_from_slice(&value.to_be_bytes());
            }
            (NcType::Int, values.len() as u64)
        }
        NcAttrValue::Floats(values) => {
            for value in values {
                out.extend_from_slice(&value.to_be_bytes());
            }
            (NcType::Float, values.len() as u64)
        }
        NcAttrValue::Doubles(values) => {
            for value in values {
                out.extend_from_slice(&value.to_be_bytes());
            }
            (NcType::Double, values.len() as u64)
        }
        NcAttrValue::UBytes(values) => {
            out.extend_from_slice(values);
            (NcType::UByte, values.len() as u64)
        }
        NcAttrValue::UShorts(values) => {
            for value in values {
                out.extend_from_slice(&value.to_be_bytes());
            }
            (NcType::UShort, values.len() as u64)
        }
        NcAttrValue::UInts(values) => {
            for value in values {
                out.extend_from_slice(&value.to_be_bytes());
            }
            (NcType::UInt, values.len() as u64)
        }
        NcAttrValue::Int64s(values) => {
            for value in values {
                out.extend_from_slice(&value.to_be_bytes());
            }
            (NcType::Int64, values.len() as u64)
        }
        NcAttrValue::UInt64s(values) => {
            for value in values {
                out.extend_from_slice(&value.to_be_bytes());
            }
            (NcType::UInt64, values.len() as u64)
        }
        NcAttrValue::Strings(_) => {
            return Err(Error::UnsupportedFeature(
                "NC_STRING attributes require NetCDF-4".into(),
            ));
        }
    };
    Ok((dtype, count, out))
}

#[cfg(feature = "netcdf4")]
fn nc_type_to_hdf5(dtype: &NcType) -> Result<H5Datatype> {
    let byte_order = H5ByteOrder::LittleEndian;
    match dtype {
        NcType::Byte => Ok(H5Datatype::FixedPoint {
            size: 1,
            signed: true,
            byte_order,
        }),
        NcType::UByte => Ok(H5Datatype::FixedPoint {
            size: 1,
            signed: false,
            byte_order,
        }),
        NcType::Short => Ok(H5Datatype::FixedPoint {
            size: 2,
            signed: true,
            byte_order,
        }),
        NcType::UShort => Ok(H5Datatype::FixedPoint {
            size: 2,
            signed: false,
            byte_order,
        }),
        NcType::Int => Ok(H5Datatype::FixedPoint {
            size: 4,
            signed: true,
            byte_order,
        }),
        NcType::UInt => Ok(H5Datatype::FixedPoint {
            size: 4,
            signed: false,
            byte_order,
        }),
        NcType::Int64 => Ok(H5Datatype::FixedPoint {
            size: 8,
            signed: true,
            byte_order,
        }),
        NcType::UInt64 => Ok(H5Datatype::FixedPoint {
            size: 8,
            signed: false,
            byte_order,
        }),
        NcType::Float => Ok(H5Datatype::FloatingPoint {
            size: 4,
            byte_order,
        }),
        NcType::Double => Ok(H5Datatype::FloatingPoint {
            size: 8,
            byte_order,
        }),
        NcType::Char => Ok(H5Datatype::String {
            size: H5StringSize::Fixed(1),
            encoding: H5StringEncoding::Ascii,
            padding: H5StringPadding::NullPad,
        }),
        NcType::String => Ok(H5Datatype::String {
            size: H5StringSize::Variable,
            encoding: H5StringEncoding::Utf8,
            padding: H5StringPadding::NullTerminate,
        }),
        NcType::Enum { base, members } => {
            let h5_base = nc_type_to_hdf5(base)?;
            let members = members
                .iter()
                .map(|member| {
                    Ok(H5EnumMember {
                        name: member.name.clone(),
                        value: nc_enum_value_to_hdf5(base, member.value)?,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(H5Datatype::Enum {
                base: Box::new(h5_base),
                members,
            })
        }
        NcType::Compound { size, fields } => {
            let fields = fields
                .iter()
                .map(|field| {
                    let byte_offset = u32::try_from(field.offset).map_err(|_| {
                        Error::UnsupportedFeature(format!(
                            "compound field '{}' offset exceeds HDF5 u32 capacity",
                            field.name
                        ))
                    })?;
                    Ok(H5CompoundField {
                        name: field.name.clone(),
                        byte_offset,
                        datatype: nc_type_to_hdf5(&field.dtype)?,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(H5Datatype::Compound {
                size: *size,
                fields,
            })
        }
        NcType::Opaque { size, tag } => Ok(H5Datatype::Opaque {
            size: *size,
            tag: tag.clone(),
        }),
        NcType::Array { base, dims } => Ok(H5Datatype::Array {
            base: Box::new(nc_type_to_hdf5(base)?),
            dims: dims.clone(),
        }),
        NcType::VLen { base } => {
            validate_vlen_base_nc4_type(base)?;
            Ok(H5Datatype::VarLen {
                base: Box::new(nc_type_to_hdf5(base)?),
                kind: H5VarLenKind::Sequence,
                encoding: H5StringEncoding::Ascii,
                padding: H5StringPadding::NullTerminate,
            })
        }
    }
}

#[cfg(feature = "netcdf4")]
fn nc_enum_value_to_hdf5(base: &NcType, value: NcIntegerValue) -> Result<Vec<u8>> {
    nc_enum_value_to_le_bytes(base, value)
}

fn nc_enum_value_to_le_bytes(base: &NcType, value: NcIntegerValue) -> Result<Vec<u8>> {
    match (base, value) {
        (NcType::Byte, NcIntegerValue::I8(value)) => Ok(vec![value as u8]),
        (NcType::UByte, NcIntegerValue::U8(value)) => Ok(vec![value]),
        (NcType::Short, NcIntegerValue::I16(value)) => Ok(value.to_le_bytes().to_vec()),
        (NcType::UShort, NcIntegerValue::U16(value)) => Ok(value.to_le_bytes().to_vec()),
        (NcType::Int, NcIntegerValue::I32(value)) => Ok(value.to_le_bytes().to_vec()),
        (NcType::UInt, NcIntegerValue::U32(value)) => Ok(value.to_le_bytes().to_vec()),
        (NcType::Int64, NcIntegerValue::I64(value)) => Ok(value.to_le_bytes().to_vec()),
        (NcType::UInt64, NcIntegerValue::U64(value)) => Ok(value.to_le_bytes().to_vec()),
        (base, value) => Err(Error::InvalidDefinition(format!(
            "enum value {value:?} is incompatible with base type {base:?}"
        ))),
    }
}

#[cfg(feature = "netcdf4")]
fn nc4_hdf5_element_size(dtype: &NcType) -> Result<usize> {
    match dtype {
        NcType::String | NcType::VLen { .. } => Ok(16),
        other => other.size().map_err(Error::Core),
    }
}

#[cfg(feature = "netcdf4")]
fn convert_classic_be_data_to_hdf5_le(dtype: &NcType, data: &[u8]) -> Result<Vec<u8>> {
    let width = dtype.size()?;
    if width == 1 {
        return Ok(data.to_vec());
    }
    if data.len() % width != 0 {
        return Err(Error::InvalidDefinition(format!(
            "variable data length {} is not a multiple of element size {width}",
            data.len()
        )));
    }

    match dtype {
        NcType::Short | NcType::UShort => Ok(data
            .chunks_exact(2)
            .flat_map(|chunk| [chunk[1], chunk[0]])
            .collect()),
        NcType::Int | NcType::UInt | NcType::Float => Ok(data
            .chunks_exact(4)
            .flat_map(|chunk| [chunk[3], chunk[2], chunk[1], chunk[0]])
            .collect()),
        NcType::Int64 | NcType::UInt64 | NcType::Double => Ok(data
            .chunks_exact(8)
            .flat_map(|chunk| {
                [
                    chunk[7], chunk[6], chunk[5], chunk[4], chunk[3], chunk[2], chunk[1], chunk[0],
                ]
            })
            .collect()),
        other => Err(Error::UnsupportedFeature(format!(
            "NetCDF-4 data conversion is not implemented for {other:?}"
        ))),
    }
}

#[cfg(feature = "netcdf4")]
fn nc_attr_to_hdf5(attribute: &NcAttribute) -> Result<H5AttributeBuilder> {
    match &attribute.value {
        NcAttrValue::Chars(value) => Ok(H5AttributeBuilder::fixed_string(&attribute.name, value)),
        NcAttrValue::Bytes(values) => hdf5_numeric_attr(&attribute.name, NcType::Byte, values),
        NcAttrValue::UBytes(values) => hdf5_numeric_attr(&attribute.name, NcType::UByte, values),
        NcAttrValue::Shorts(values) => hdf5_numeric_attr(&attribute.name, NcType::Short, values),
        NcAttrValue::UShorts(values) => hdf5_numeric_attr(&attribute.name, NcType::UShort, values),
        NcAttrValue::Ints(values) => hdf5_numeric_attr(&attribute.name, NcType::Int, values),
        NcAttrValue::UInts(values) => hdf5_numeric_attr(&attribute.name, NcType::UInt, values),
        NcAttrValue::Int64s(values) => hdf5_numeric_attr(&attribute.name, NcType::Int64, values),
        NcAttrValue::UInt64s(values) => hdf5_numeric_attr(&attribute.name, NcType::UInt64, values),
        NcAttrValue::Floats(values) => hdf5_numeric_attr(&attribute.name, NcType::Float, values),
        NcAttrValue::Doubles(values) => hdf5_numeric_attr(&attribute.name, NcType::Double, values),
        NcAttrValue::Strings(values) => H5AttributeBuilder::vlen_strings(&attribute.name, values)
            .map_err(hdf5_error_to_unsupported),
    }
}

#[cfg(feature = "netcdf4")]
trait NcAttrElement {
    fn write_le(&self, dst: &mut Vec<u8>);
}

#[cfg(feature = "netcdf4")]
macro_rules! impl_attr_element_bytes {
    ($ty:ty) => {
        impl NcAttrElement for $ty {
            fn write_le(&self, dst: &mut Vec<u8>) {
                dst.push(*self as u8);
            }
        }
    };
}

#[cfg(feature = "netcdf4")]
macro_rules! impl_attr_element_le {
    ($ty:ty) => {
        impl NcAttrElement for $ty {
            fn write_le(&self, dst: &mut Vec<u8>) {
                dst.extend_from_slice(&self.to_le_bytes());
            }
        }
    };
}

#[cfg(feature = "netcdf4")]
impl_attr_element_bytes!(i8);
#[cfg(feature = "netcdf4")]
impl_attr_element_bytes!(u8);
#[cfg(feature = "netcdf4")]
impl_attr_element_le!(i16);
#[cfg(feature = "netcdf4")]
impl_attr_element_le!(u16);
#[cfg(feature = "netcdf4")]
impl_attr_element_le!(i32);
#[cfg(feature = "netcdf4")]
impl_attr_element_le!(u32);
#[cfg(feature = "netcdf4")]
impl_attr_element_le!(i64);
#[cfg(feature = "netcdf4")]
impl_attr_element_le!(u64);
#[cfg(feature = "netcdf4")]
impl_attr_element_le!(f32);
#[cfg(feature = "netcdf4")]
impl_attr_element_le!(f64);

#[cfg(feature = "netcdf4")]
fn hdf5_numeric_attr<T: NcAttrElement>(
    name: &str,
    dtype: NcType,
    values: &[T],
) -> Result<H5AttributeBuilder> {
    let mut raw = Vec::with_capacity(values.len() * dtype.size()?);
    for value in values {
        value.write_le(&mut raw);
    }
    Ok(H5AttributeBuilder::new(
        name,
        nc_type_to_hdf5(&dtype)?,
        vec![values.len() as u64],
        raw,
    ))
}

#[cfg(feature = "netcdf4")]
fn empty_dimension_list_attribute() -> H5AttributeBuilder {
    H5AttributeBuilder::new(
        "DIMENSION_LIST",
        H5Datatype::VarLen {
            base: Box::new(H5Datatype::Reference {
                ref_type: H5ReferenceType::Object,
                size: 8,
            }),
            kind: H5VarLenKind::Sequence,
            encoding: H5StringEncoding::Ascii,
            padding: H5StringPadding::NullTerminate,
        },
        vec![0],
        Vec::new(),
    )
}

#[cfg(feature = "netcdf4")]
fn hdf5_error_to_unsupported(err: hdf5_writer::Error) -> Error {
    Error::UnsupportedFeature(format!("HDF5 writer error: {err}"))
}

#[cfg(feature = "netcdf4")]
fn checked_usize(value: u64, context: &str) -> Result<usize> {
    usize::try_from(value).map_err(|_| {
        Error::InvalidDefinition(format!(
            "{context} value {value} exceeds platform usize capacity"
        ))
    })
}

fn expected_variable_bytes(
    builder: &NcFileBuilder,
    variable: &VariableDef,
    is_record: bool,
) -> Result<usize> {
    let elements = if is_record {
        let records = infer_num_records_for_var(builder, variable)?;
        records
            .checked_mul(variable_shape_elements(builder, variable, 1)?)
            .ok_or_else(|| Error::InvalidDefinition("variable element count overflow".into()))?
    } else {
        variable_shape_elements(builder, variable, 0)?
    };
    let bytes = elements
        .checked_mul(variable.dtype.size()? as u64)
        .ok_or_else(|| Error::InvalidDefinition("variable byte size overflow".into()))?;
    usize::try_from(bytes)
        .map_err(|_| Error::InvalidDefinition("variable byte size exceeds platform usize".into()))
}

#[derive(Debug, Clone)]
struct ResolvedWriteSelection {
    dims: Vec<ResolvedWriteSelectionDim>,
    elements: usize,
}

#[derive(Debug, Clone)]
enum ResolvedWriteSelectionDim {
    Index(u64),
    Slice { start: u64, step: u64, count: usize },
}

fn resolve_write_selection(
    variable_name: &str,
    shape: &[u64],
    selection: &NcSliceInfo,
) -> Result<ResolvedWriteSelection> {
    if selection.selections.len() != shape.len() {
        return Err(Error::InvalidDefinition(format!(
            "selection has {} dimensions but variable '{}' has {}",
            selection.selections.len(),
            variable_name,
            shape.len()
        )));
    }

    let mut dims = Vec::with_capacity(shape.len());
    let mut elements = 1usize;
    for (dim, (selection, &dim_size)) in selection.selections.iter().zip(shape).enumerate() {
        match selection {
            NcSliceInfoElem::Index(index) => {
                if *index >= dim_size {
                    return Err(Error::InvalidDefinition(format!(
                        "index {index} out of bounds for dimension {dim} (size {dim_size})"
                    )));
                }
                dims.push(ResolvedWriteSelectionDim::Index(*index));
            }
            NcSliceInfoElem::Slice { start, end, step } => {
                if *step == 0 {
                    return Err(Error::InvalidDefinition("slice step cannot be 0".into()));
                }
                if *start > dim_size {
                    return Err(Error::InvalidDefinition(format!(
                        "slice start {start} out of bounds for dimension {dim} (size {dim_size})"
                    )));
                }
                let actual_end = if *end == u64::MAX {
                    dim_size
                } else {
                    (*end).min(dim_size)
                };
                let count_u64 = if *start >= actual_end {
                    0
                } else {
                    (actual_end - *start).div_ceil(*step)
                };
                let count = require_usize(count_u64, "slice result dimension")?;
                elements = elements.checked_mul(count).ok_or_else(|| {
                    Error::InvalidDefinition(
                        "slice result element count exceeds platform usize".into(),
                    )
                })?;
                dims.push(ResolvedWriteSelectionDim::Slice {
                    start: *start,
                    step: *step,
                    count,
                });
            }
        }
    }

    Ok(ResolvedWriteSelection { dims, elements })
}

fn ensure_variable_slice_buffer(
    variable: &mut VariableDef,
    total_elements: u64,
    elem_size: usize,
) -> Result<()> {
    if variable.data_encoding != VariableDataEncoding::ClassicBigEndian && !variable.data.is_empty()
    {
        return Err(Error::UnsupportedFeature(format!(
            "typed slice writes cannot update native-endian payload for variable '{}'",
            variable.name
        )));
    }

    let expected = checked_mul_usize(
        require_usize(total_elements, "variable element count")?,
        elem_size,
        "variable byte size",
    )?;
    if variable.data.is_empty() {
        let fill_pattern = match &variable.fill_value {
            Some(fill_value) => fill_value.classic_bytes.clone(),
            None => default_classic_fill_bytes(&variable.dtype)?,
        };
        variable.data = filled_data_buffer(expected, elem_size, &fill_pattern)?;
    } else if variable.data.len() != expected {
        return Err(Error::DataLengthMismatch {
            expected,
            actual: variable.data.len(),
        });
    }

    variable.data_encoding = VariableDataEncoding::ClassicBigEndian;
    variable.string_values = None;
    variable.vlen_values = None;
    Ok(())
}

fn filled_data_buffer(len: usize, elem_size: usize, fill_pattern: &[u8]) -> Result<Vec<u8>> {
    if elem_size == 0 {
        return if len == 0 {
            Ok(Vec::new())
        } else {
            Err(Error::InvalidDefinition(
                "non-empty variable data requires a non-zero element size".into(),
            ))
        };
    }
    if len % elem_size != 0 {
        return Err(Error::InvalidDefinition(format!(
            "variable byte length {len} is not a multiple of element size {elem_size}"
        )));
    }
    if fill_pattern.len() != elem_size {
        return Err(Error::InvalidDefinition(format!(
            "fill value byte length must match datatype element size: expected {elem_size}, got {}",
            fill_pattern.len()
        )));
    }
    let mut data = Vec::with_capacity(len);
    for _ in 0..len / elem_size {
        data.extend_from_slice(fill_pattern);
    }
    Ok(data)
}

fn default_classic_fill_bytes(dtype: &NcType) -> Result<Vec<u8>> {
    match dtype {
        NcType::Byte => Ok(vec![NC_FILL_BYTE as u8]),
        NcType::Char => Ok(vec![NC_FILL_CHAR]),
        NcType::Short => Ok(NC_FILL_SHORT.to_be_bytes().to_vec()),
        NcType::Int => Ok(NC_FILL_INT.to_be_bytes().to_vec()),
        NcType::Float => Ok(NC_FILL_FLOAT.to_be_bytes().to_vec()),
        NcType::Double => Ok(NC_FILL_DOUBLE.to_be_bytes().to_vec()),
        NcType::UByte => Ok(vec![NC_FILL_UBYTE]),
        NcType::UShort => Ok(NC_FILL_USHORT.to_be_bytes().to_vec()),
        NcType::UInt => Ok(NC_FILL_UINT.to_be_bytes().to_vec()),
        NcType::Int64 => Ok(NC_FILL_INT64.to_be_bytes().to_vec()),
        NcType::UInt64 => Ok(NC_FILL_UINT64.to_be_bytes().to_vec()),
        other => Err(Error::UnsupportedFeature(format!(
            "{other:?} does not have a primitive NetCDF default fill value"
        ))),
    }
}

fn scatter_slice_bytes(
    data: &mut [u8],
    elem_size: usize,
    dims: &[ResolvedWriteSelectionDim],
    strides: &[u64],
    encoded: &[u8],
) -> Result<()> {
    let mut src_elem = 0usize;
    scatter_slice_bytes_recursive(data, elem_size, dims, strides, encoded, 0, 0, &mut src_elem)?;
    let expected = encoded.len() / elem_size.max(1);
    if src_elem != expected {
        return Err(Error::InvalidDefinition(format!(
            "slice scatter wrote {src_elem} elements but expected {expected}"
        )));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn scatter_slice_bytes_recursive(
    data: &mut [u8],
    elem_size: usize,
    dims: &[ResolvedWriteSelectionDim],
    strides: &[u64],
    encoded: &[u8],
    dim: usize,
    dst_elem: u64,
    src_elem: &mut usize,
) -> Result<()> {
    if dim == dims.len() {
        let dst_start = checked_mul_usize(
            require_usize(dst_elem, "slice destination element")?,
            elem_size,
            "slice destination byte offset",
        )?;
        let dst_end = dst_start.checked_add(elem_size).ok_or_else(|| {
            Error::InvalidDefinition("slice destination byte range exceeds usize".into())
        })?;
        let src_start = checked_mul_usize(*src_elem, elem_size, "slice source byte offset")?;
        let src_end = src_start.checked_add(elem_size).ok_or_else(|| {
            Error::InvalidDefinition("slice source byte range exceeds usize".into())
        })?;
        if dst_end > data.len() || src_end > encoded.len() {
            return Err(Error::InvalidDefinition(
                "slice write exceeded planned data bounds".into(),
            ));
        }
        data[dst_start..dst_end].copy_from_slice(&encoded[src_start..src_end]);
        *src_elem += 1;
        return Ok(());
    }

    match dims[dim] {
        ResolvedWriteSelectionDim::Index(index) => {
            let offset = checked_add(
                dst_elem,
                checked_mul_u64(index, strides[dim], "slice index stride")?,
                "slice destination element",
            )?;
            scatter_slice_bytes_recursive(
                data,
                elem_size,
                dims,
                strides,
                encoded,
                dim + 1,
                offset,
                src_elem,
            )
        }
        ResolvedWriteSelectionDim::Slice { start, step, count } => {
            for i in 0..count {
                let coord = checked_add(
                    start,
                    checked_mul_u64(i as u64, step, "slice coordinate step")?,
                    "slice coordinate",
                )?;
                let offset = checked_add(
                    dst_elem,
                    checked_mul_u64(coord, strides[dim], "slice coordinate stride")?,
                    "slice destination element",
                )?;
                scatter_slice_bytes_recursive(
                    data,
                    elem_size,
                    dims,
                    strides,
                    encoded,
                    dim + 1,
                    offset,
                    src_elem,
                )?;
            }
            Ok(())
        }
    }
}

fn row_major_strides(shape: &[u64], context: &str) -> Result<Vec<u64>> {
    if shape.is_empty() {
        return Ok(Vec::new());
    }
    let mut strides = vec![1u64; shape.len()];
    for dim in (0..shape.len() - 1).rev() {
        strides[dim] = checked_mul_u64(strides[dim + 1], shape[dim + 1], context)?;
    }
    Ok(strides)
}

fn checked_shape_elements(shape: &[u64], context: &str) -> Result<u64> {
    shape
        .iter()
        .try_fold(1u64, |acc, &dim| checked_mul_u64(acc, dim, context))
}

#[cfg(feature = "netcdf4")]
fn validate_nc4_variable_data(variable: &VariableDef, dimension_sizes: &[u64]) -> Result<()> {
    let elements = nc4_variable_elements(variable, dimension_sizes)?;
    validate_nc4_variable_storage(variable)?;
    validate_nc4_variable_fill_value(variable)?;
    if variable.dtype == NcType::String {
        let values = variable.string_values.as_ref().ok_or_else(|| {
            Error::InvalidDefinition(format!(
                "NC_STRING variable '{}' has no string data",
                variable.name
            ))
        })?;
        let expected = checked_usize(elements, "NetCDF-4 string variable element count")?;
        if values.len() != expected {
            return Err(Error::DataLengthMismatch {
                expected,
                actual: values.len(),
            });
        }
        return Ok(());
    }
    if let NcType::VLen { base } = &variable.dtype {
        let values = variable.vlen_values.as_ref().ok_or_else(|| {
            Error::InvalidDefinition(format!(
                "NC_VLEN variable '{}' has no sequence data",
                variable.name
            ))
        })?;
        validate_vlen_base_nc4_type(base)?;
        let expected = checked_usize(elements, "NetCDF-4 vlen variable element count")?;
        if values.len() != expected {
            return Err(Error::DataLengthMismatch {
                expected,
                actual: values.len(),
            });
        }
        let base_size = base.size()?;
        for value in values {
            if value.len() % base_size != 0 {
                return Err(Error::InvalidDefinition(format!(
                    "NC_VLEN variable '{}' sequence byte length {} is not a multiple of base element size {base_size}",
                    variable.name,
                    value.len()
                )));
            }
        }
        return Ok(());
    }

    let bytes = checked_mul_u64(
        elements,
        variable.dtype.size()? as u64,
        "NetCDF-4 variable byte size",
    )?;
    let expected = usize::try_from(bytes).map_err(|_| {
        Error::InvalidDefinition("NetCDF-4 variable byte size exceeds platform usize".into())
    })?;
    if variable.data.is_empty() && variable.fill_value.is_some() {
        return Ok(());
    }
    if variable.data.len() != expected {
        return Err(Error::DataLengthMismatch {
            expected,
            actual: variable.data.len(),
        });
    }
    Ok(())
}

#[cfg(feature = "netcdf4")]
fn validate_nc4_variable_fill_value(variable: &VariableDef) -> Result<()> {
    let Some(fill_value) = &variable.fill_value else {
        return Ok(());
    };
    if matches!(&variable.dtype, NcType::String | NcType::VLen { .. }) {
        return Err(Error::UnsupportedFeature(format!(
            "NetCDF-4 variable-length fill values are not supported yet: '{}'",
            variable.name
        )));
    }
    let expected = nc4_hdf5_element_size(&variable.dtype)?;
    if fill_value.hdf5_bytes.len() != expected {
        return Err(Error::InvalidDefinition(format!(
            "variable '{}' fill value byte length must match datatype element size: expected {expected}, got {}",
            variable.name,
            fill_value.hdf5_bytes.len()
        )));
    }
    Ok(())
}

#[cfg(feature = "netcdf4")]
fn nc4_variable_elements(variable: &VariableDef, dimension_sizes: &[u64]) -> Result<u64> {
    let elements = variable.dim_ids.iter().try_fold(1u64, |acc, id| {
        checked_mul_u64(
            acc,
            dimension_sizes[id.0],
            "NetCDF-4 variable element count",
        )
    })?;
    Ok(elements)
}

#[cfg(feature = "netcdf4")]
fn infer_nc4_dimension_sizes(builder: &NcFileBuilder) -> Result<Vec<u64>> {
    let mut sizes = builder
        .dimensions
        .iter()
        .map(|dimension| {
            if dimension.is_unlimited {
                None
            } else {
                Some(dimension.size)
            }
        })
        .collect::<Vec<_>>();

    loop {
        let mut changed = false;
        for variable in &builder.variables {
            changed |= infer_nc4_dimension_size_from_variable(builder, variable, &mut sizes)?;
        }
        if !changed {
            break;
        }
    }

    for variable in &builder.variables {
        let unresolved = variable
            .dim_ids
            .iter()
            .copied()
            .filter(|id| sizes[id.0].is_none())
            .collect::<Vec<_>>();
        if !unresolved.is_empty() && variable_has_data(variable) {
            let names = unresolved
                .iter()
                .map(|id| builder.dimensions[id.0].name.as_str())
                .collect::<Vec<_>>()
                .join(", ");
            return Err(Error::InvalidDefinition(format!(
                "cannot infer current size for NetCDF-4 unlimited dimension(s) {names} from variable '{}'",
                variable.name
            )));
        }
    }

    Ok(sizes.into_iter().map(|size| size.unwrap_or(0)).collect())
}

#[cfg(feature = "netcdf4")]
fn infer_nc4_dimension_size_from_variable(
    builder: &NcFileBuilder,
    variable: &VariableDef,
    sizes: &mut [Option<u64>],
) -> Result<bool> {
    let (elements, elem_size, actual_bytes) = variable_payload_element_count(variable)?;
    let mut known_product = 1u64;
    let mut unknown = Vec::new();
    for dim_id in &variable.dim_ids {
        match sizes[dim_id.0] {
            Some(size) => {
                known_product =
                    checked_mul_u64(known_product, size, "NetCDF-4 known dimension product")?;
            }
            None if !unknown.contains(dim_id) => unknown.push(*dim_id),
            None => {
                return Err(Error::UnsupportedFeature(format!(
                    "NetCDF-4 variable '{}' repeats unlimited dimension '{}'",
                    variable.name, builder.dimensions[dim_id.0].name
                )));
            }
        }
    }

    if unknown.len() != 1 || known_product == 0 {
        return Ok(false);
    }
    if elements % known_product != 0 {
        return Err(Error::DataLengthMismatch {
            expected: usize::try_from((elements / known_product + 1) * known_product * elem_size)
                .unwrap_or(usize::MAX),
            actual: actual_bytes,
        });
    }
    let inferred = elements / known_product;
    sizes[unknown[0].0] = Some(inferred);
    Ok(true)
}

#[cfg(feature = "netcdf4")]
fn variable_has_data(variable: &VariableDef) -> bool {
    match &variable.dtype {
        NcType::String => variable
            .string_values
            .as_ref()
            .is_some_and(|values| !values.is_empty()),
        NcType::VLen { .. } => variable
            .vlen_values
            .as_ref()
            .is_some_and(|values| !values.is_empty()),
        _ => !variable.data.is_empty(),
    }
}

#[cfg(feature = "netcdf4")]
fn variable_payload_element_count(variable: &VariableDef) -> Result<(u64, u64, usize)> {
    if variable.dtype == NcType::String {
        let values = variable.string_values.as_ref().map_or(0usize, Vec::len);
        return Ok((
            u64::try_from(values).map_err(|_| {
                Error::InvalidDefinition(
                    "NC_STRING variable element count exceeds u64 capacity".into(),
                )
            })?,
            1,
            values,
        ));
    }
    if matches!(&variable.dtype, NcType::VLen { .. }) {
        let values = variable.vlen_values.as_ref().map_or(0usize, Vec::len);
        return Ok((
            u64::try_from(values).map_err(|_| {
                Error::InvalidDefinition(
                    "NC_VLEN variable element count exceeds u64 capacity".into(),
                )
            })?,
            1,
            values,
        ));
    }

    let elem_size = variable.dtype.size()? as u64;
    let data_len = variable.data.len() as u64;
    if data_len % elem_size != 0 {
        return Err(Error::DataLengthMismatch {
            expected: usize::try_from((data_len / elem_size + 1) * elem_size).unwrap_or(usize::MAX),
            actual: variable.data.len(),
        });
    }
    Ok((data_len / elem_size, elem_size, variable.data.len()))
}

fn variable_shape_elements(
    builder: &NcFileBuilder,
    variable: &VariableDef,
    skip_dims: usize,
) -> Result<u64> {
    variable
        .dim_ids
        .iter()
        .skip(skip_dims)
        .try_fold(1u64, |acc, id| {
            let dim = &builder.dimensions[id.0];
            acc.checked_mul(dim.size).ok_or_else(|| {
                Error::InvalidDefinition(format!(
                    "variable '{}' element count overflows u64",
                    variable.name
                ))
            })
        })
}

fn variable_is_record(builder: &NcFileBuilder, variable: &VariableDef) -> Result<bool> {
    Ok(variable
        .dim_ids
        .first()
        .is_some_and(|id| builder.dimensions[id.0].is_unlimited))
}

fn infer_num_records(builder: &NcFileBuilder) -> Result<u64> {
    let mut records: Option<u64> = None;
    for variable in &builder.variables {
        if !variable_is_record(builder, variable)? {
            continue;
        }
        let current = infer_num_records_for_var(builder, variable)?;
        match records {
            Some(existing) if existing != current => {
                return Err(Error::InvalidDefinition(
                    "all record variables must contain the same record count".into(),
                ));
            }
            Some(_) => {}
            None => records = Some(current),
        }
    }
    Ok(records.unwrap_or(0))
}

fn infer_num_records_for_var(builder: &NcFileBuilder, variable: &VariableDef) -> Result<u64> {
    let record_elems = variable_shape_elements(builder, variable, 1)?;
    let elem_size = variable.dtype.size()? as u64;
    let record_bytes = record_elems
        .checked_mul(elem_size)
        .ok_or_else(|| Error::InvalidDefinition("record byte size overflow".into()))?;
    if record_bytes == 0 {
        return Ok(0);
    }
    let data_len = variable.data.len() as u64;
    if data_len % record_bytes != 0 {
        return Err(Error::DataLengthMismatch {
            expected: usize::try_from((data_len / record_bytes + 1) * record_bytes)
                .unwrap_or(usize::MAX),
            actual: variable.data.len(),
        });
    }
    Ok(data_len / record_bytes)
}

fn validate_classic_type(dtype: &NcType) -> Result<()> {
    if dtype.classic_type_code().is_none() {
        return Err(Error::UnsupportedFeature(format!(
            "{dtype:?} requires NetCDF-4"
        )));
    }
    Ok(())
}

fn validate_chunk_shape_for_variable(variable: &VariableDef, chunk_shape: &[u64]) -> Result<()> {
    if variable.dim_ids.is_empty() {
        return Err(Error::InvalidDefinition(format!(
            "chunked NetCDF-4 scalar variables are not supported: '{}'",
            variable.name
        )));
    }
    if chunk_shape.len() != variable.dim_ids.len() {
        return Err(Error::InvalidDefinition(format!(
            "variable '{}' chunk rank must match variable rank: expected {}, got {}",
            variable.name,
            variable.dim_ids.len(),
            chunk_shape.len()
        )));
    }
    if chunk_shape.contains(&0) {
        return Err(Error::InvalidDefinition(format!(
            "variable '{}' chunk dimensions must be non-zero",
            variable.name
        )));
    }
    Ok(())
}

fn reject_scalar_filtered_variable(variable: &VariableDef) -> Result<()> {
    if variable.dim_ids.is_empty() {
        return Err(Error::InvalidDefinition(format!(
            "filtered NetCDF-4 scalar variables are not supported: '{}'",
            variable.name
        )));
    }
    Ok(())
}

#[cfg(feature = "netcdf4")]
fn validate_nc4_variable_storage(variable: &VariableDef) -> Result<()> {
    if let Some(chunk_shape) = &variable.storage.chunk_shape {
        validate_chunk_shape_for_variable(variable, chunk_shape)?;
    }
    if variable.storage.has_filters() {
        reject_scalar_filtered_variable(variable)?;
        if matches!(&variable.dtype, NcType::String | NcType::VLen { .. }) {
            return Err(Error::UnsupportedFeature(format!(
                "filtered NetCDF-4 variable-length variable '{}' is not supported yet",
                variable.name
            )));
        }
    }
    Ok(())
}

#[cfg(feature = "netcdf4")]
fn nc4_variable_chunk_shape(
    variable: &VariableDef,
    shape: &[u64],
    requires_chunking: bool,
) -> Result<Option<Vec<u64>>> {
    if let Some(chunk_shape) = &variable.storage.chunk_shape {
        validate_chunk_shape_for_variable(variable, chunk_shape)?;
        return Ok(Some(chunk_shape.clone()));
    }
    if requires_chunking || variable.storage.has_filters() {
        return Ok(Some(nc4_default_chunk_shape(
            shape,
            nc4_hdf5_element_size(&variable.dtype)?,
        )?));
    }
    Ok(None)
}

#[cfg(feature = "netcdf4")]
fn nc4_variable_filters(variable: &VariableDef) -> Vec<H5FilterDescription> {
    let mut filters = Vec::new();
    if variable.storage.shuffle {
        filters.push(H5FilterDescription {
            id: H5_FILTER_SHUFFLE,
            name: None,
            client_data: Vec::new(),
        });
    }
    if let Some(level) = variable.storage.deflate_level {
        filters.push(H5FilterDescription {
            id: H5_FILTER_DEFLATE,
            name: None,
            client_data: vec![u32::from(level)],
        });
    }
    if variable.storage.fletcher32 {
        filters.push(H5FilterDescription {
            id: H5_FILTER_FLETCHER32,
            name: None,
            client_data: Vec::new(),
        });
    }
    filters
}

fn validate_supported_user_defined_type(dtype: &NcType) -> Result<()> {
    if let NcType::VLen { base } = dtype {
        return validate_vlen_base_nc4_type(base);
    }
    if !matches!(
        dtype,
        NcType::Enum { .. }
            | NcType::Compound { .. }
            | NcType::Opaque { .. }
            | NcType::Array { .. }
    ) {
        return Err(Error::InvalidDefinition(format!(
            "user-defined variables require enum, compound, opaque, fixed-size array, or vlen type, got {dtype:?}"
        )));
    }
    validate_fixed_width_nc4_type(dtype)
}

fn validate_fixed_width_user_defined_type(dtype: &NcType) -> Result<()> {
    if !matches!(
        dtype,
        NcType::Enum { .. }
            | NcType::Compound { .. }
            | NcType::Opaque { .. }
            | NcType::Array { .. }
    ) {
        return Err(Error::InvalidDefinition(format!(
            "raw user-defined variable bytes require enum, compound, opaque, or fixed-size array type, got {dtype:?}"
        )));
    }
    validate_fixed_width_nc4_type(dtype)
}

fn validate_fixed_width_nc4_type(dtype: &NcType) -> Result<()> {
    match dtype {
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
        | NcType::Opaque { .. } => Ok(()),
        NcType::Enum { base, .. } => match base.as_ref() {
            NcType::Byte
            | NcType::UByte
            | NcType::Short
            | NcType::UShort
            | NcType::Int
            | NcType::UInt
            | NcType::Int64
            | NcType::UInt64 => Ok(()),
            other => Err(Error::InvalidDefinition(format!(
                "enum base type must be fixed-width integer, got {other:?}"
            ))),
        },
        NcType::Compound { fields, .. } => {
            for field in fields {
                validate_fixed_width_nc4_type(&field.dtype)?;
            }
            Ok(())
        }
        NcType::Array { base, .. } => validate_fixed_width_nc4_type(base),
        NcType::String | NcType::VLen { .. } => Err(Error::UnsupportedFeature(format!(
            "{dtype:?} user-defined variable payloads require heap-backed sequence writing"
        ))),
    }
}

fn validate_vlen_base_nc4_type(dtype: &NcType) -> Result<()> {
    if dtype.size()? == 0 {
        return Err(Error::InvalidDefinition(
            "vlen base type must have non-zero byte size".into(),
        ));
    }
    match dtype {
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
        | NcType::Opaque { .. } => Ok(()),
        NcType::Enum { .. } => validate_fixed_width_nc4_type(dtype),
        NcType::Compound { fields, .. } => {
            for field in fields {
                validate_vlen_base_nc4_type(&field.dtype)?;
            }
            Ok(())
        }
        NcType::Array { base, .. } => validate_vlen_base_nc4_type(base),
        NcType::String | NcType::VLen { .. } => Err(Error::UnsupportedFeature(format!(
            "{dtype:?} cannot be used as a vlen base until recursive heap-backed payload writing is implemented"
        ))),
    }
}

#[cfg(feature = "netcdf4")]
fn validate_nc4_classic_attr_value(value: &NcAttrValue) -> Result<()> {
    if matches!(value, NcAttrValue::Strings(_)) {
        return Err(Error::UnsupportedFeature(
            "NC_STRING attributes require full NetCDF-4".into(),
        ));
    }
    Ok(())
}

#[cfg(feature = "netcdf4")]
fn validate_nc4_classic_model(builder: &NcFileBuilder) -> Result<()> {
    let unlimited_count = builder
        .dimensions
        .iter()
        .filter(|dimension| dimension.is_unlimited)
        .count();
    if unlimited_count > 1 {
        return Err(Error::InvalidDefinition(
            "classic NetCDF supports at most one unlimited dimension".into(),
        ));
    }
    for variable in &builder.variables {
        if variable
            .dim_ids
            .iter()
            .skip(1)
            .any(|id| builder.dimensions[id.0].is_unlimited)
        {
            return Err(Error::InvalidDefinition(format!(
                "record dimension must be first for variable '{}'",
                variable.name
            )));
        }
    }
    Ok(())
}

#[cfg(feature = "netcdf4")]
fn nc4_default_chunk_shape(shape: &[u64], element_size: usize) -> Result<Vec<u64>> {
    if shape.is_empty() {
        return Err(Error::InvalidDefinition(
            "chunked NetCDF-4 scalar variables are not supported".into(),
        ));
    }
    let element_size = element_size.max(1);
    let target_elements = (1024usize * 1024usize / element_size).max(1) as u64;
    let mut remaining = target_elements;
    let mut chunk_shape = vec![1u64; shape.len()];
    for dim in (0..shape.len()).rev() {
        let extent = shape[dim].max(1);
        let chunk = extent.min(remaining.max(1));
        chunk_shape[dim] = chunk;
        remaining = (remaining / chunk).max(1);
    }
    Ok(chunk_shape)
}

fn attr_requires_cdf5(value: &NcAttrValue) -> bool {
    matches!(
        value,
        NcAttrValue::UBytes(_)
            | NcAttrValue::UShorts(_)
            | NcAttrValue::UInts(_)
            | NcAttrValue::Int64s(_)
            | NcAttrValue::UInt64s(_)
    )
}

fn validate_name(name: &str, kind: &str) -> Result<()> {
    if name.is_empty() {
        return Err(Error::InvalidDefinition(format!(
            "{kind} name must not be empty"
        )));
    }
    if name.contains('/') || name.bytes().any(|b| b == 0) {
        return Err(Error::InvalidDefinition(format!(
            "{kind} name '{name}' contains invalid characters"
        )));
    }
    Ok(())
}

fn ensure_unique_attr(attrs: &[NcAttribute], name: &str) -> Result<()> {
    if attrs.iter().any(|a| a.name == name) {
        return Err(Error::InvalidDefinition(format!(
            "duplicate attribute '{name}'"
        )));
    }
    Ok(())
}

fn set_variable_fill_value_metadata(
    variable: &mut VariableDef,
    value: NcAttrValue,
    classic_bytes: Vec<u8>,
    #[cfg_attr(not(feature = "netcdf4"), allow(unused_variables))] hdf5_bytes: Vec<u8>,
) -> Result<()> {
    if let Some(attribute) = variable
        .attributes
        .iter_mut()
        .find(|attribute| attribute.name == FILL_VALUE_ATTR_NAME)
    {
        if variable.fill_value.is_none() {
            return Err(Error::InvalidDefinition(format!(
                "variable '{}' already has a manual _FillValue attribute",
                variable.name
            )));
        }
        attribute.value = value;
    } else {
        variable.attributes.push(NcAttribute {
            name: FILL_VALUE_ATTR_NAME.to_string(),
            value,
        });
    }

    variable.fill_value = Some(VariableFillValue {
        classic_bytes,
        #[cfg(feature = "netcdf4")]
        hdf5_bytes,
    });
    Ok(())
}

fn write_name(out: &mut Vec<u8>, format: NcFormat, name: &str) -> Result<()> {
    write_count(out, format, name.len() as u64)?;
    out.extend_from_slice(name.as_bytes());
    pad_vec_to_4(out);
    Ok(())
}

fn write_count(out: &mut Vec<u8>, format: NcFormat, value: u64) -> Result<()> {
    match format {
        NcFormat::Cdf5 => write_u64(out, value),
        NcFormat::Classic | NcFormat::Offset64 => write_u32(out, require_u32(value, "count")?),
        _ => unreachable!("classic count only"),
    }
    Ok(())
}

fn write_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_be_bytes());
}

fn write_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_be_bytes());
}

fn pad_vec_to_4(out: &mut Vec<u8>) {
    let pad = netcdf_core::padding_to_4(out.len());
    out.resize(out.len() + pad, 0);
}

fn write_zero_padding(writer: &mut impl Write, len: u64) -> Result<()> {
    const ZEROS: [u8; 4] = [0; 4];
    if len > 0 {
        let len = usize::try_from(len)
            .map_err(|_| Error::InvalidDefinition("padding exceeds platform usize".into()))?;
        writer.write_all(&ZEROS[..len])?;
    }
    Ok(())
}

fn pad4_u64(value: u64) -> Result<u64> {
    let rem = value % 4;
    if rem == 0 {
        Ok(value)
    } else {
        checked_add(value, 4 - rem, "4-byte padding")
    }
}

fn checked_add(lhs: u64, rhs: u64, context: &str) -> Result<u64> {
    lhs.checked_add(rhs)
        .ok_or_else(|| Error::InvalidDefinition(format!("{context} overflow")))
}

fn checked_mul_u64(lhs: u64, rhs: u64, context: &str) -> Result<u64> {
    lhs.checked_mul(rhs)
        .ok_or_else(|| Error::InvalidDefinition(format!("{context} overflow")))
}

fn checked_mul_usize(lhs: usize, rhs: usize, context: &str) -> Result<usize> {
    lhs.checked_mul(rhs)
        .ok_or_else(|| Error::InvalidDefinition(format!("{context} overflow")))
}

fn require_u32(value: u64, context: &str) -> Result<u32> {
    u32::try_from(value)
        .map_err(|_| Error::InvalidDefinition(format!("{context} exceeds u32 capacity")))
}

fn require_usize(value: u64, context: &str) -> Result<usize> {
    usize::try_from(value)
        .map_err(|_| Error::InvalidDefinition(format!("{context} exceeds platform usize capacity")))
}
