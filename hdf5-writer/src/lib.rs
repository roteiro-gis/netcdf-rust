//! Pure-Rust HDF5 writer.
//!
//! Build a file from [`DatasetBuilder`]s and [`AttributeBuilder`]s with
//! [`Hdf5Builder`], turn it into an [`Hdf5WritePlan`], and serialize it with
//! [`Hdf5Writer`] (a seekable sink) or [`Hdf5WritePlan::encode`] (bytes):
//!
//! ```
//! use std::io::Cursor;
//! use hdf5_writer::{DatasetBuilder, Hdf5Builder, Hdf5Writer, WriteOptions};
//!
//! let plan = Hdf5Builder::new()
//!     .dataset(DatasetBuilder::typed_data("grid", vec![2, 3], &[0_i32, 1, 2, 3, 4, 5]).unwrap())
//!     .into_plan()
//!     .unwrap();
//! let bytes = Hdf5Writer::new(Cursor::new(Vec::new()), WriteOptions::default())
//!     .finish(plan)
//!     .unwrap()
//!     .into_inner();
//! assert_eq!(&bytes[..8], b"\x89HDF\r\n\x1a\n");
//! ```
//!
//! Supported: superblock v2; contiguous, compact, and chunked layouts (implicit,
//! single-chunk, fixed-array, and version-2 B-tree indices, the last for
//! unlimited maximum dimensions); groups; attributes; the deflate, shuffle, and
//! fletcher32 filters; and fixed-point, floating-point, string (fixed and
//! variable-length), reference, variable-length, bitfield, opaque, compound,
//! enum, and array datatypes. Output is validated against libhdf5.

use flate2::{write::ZlibEncoder, Compression};
use std::io::{Seek, SeekFrom, Write};

pub use hdf5_core::{
    fletcher32, jenkins_lookup3, ByteOrder, ChunkIndexing, CompoundField, DataLayout,
    DataspaceMessage, DataspaceType, Datatype, EnumMember, FillTime, FillValueMessage,
    FilterDescription, FilterPipelineMessage, ReferenceType, StringEncoding, StringPadding,
    StringSize, VarLenKind, FILTER_DEFLATE, FILTER_FLETCHER32, FILTER_LZ4, FILTER_NBIT,
    FILTER_SCALEOFFSET, FILTER_SHUFFLE, FILTER_SZIP, UNLIMITED,
};

const HDF5_MAGIC: [u8; 8] = [0x89, b'H', b'D', b'F', 0x0d, 0x0a, 0x1a, 0x0a];
const OFFSET_SIZE: u8 = 8;
const LENGTH_SIZE: u8 = 8;
const UNDEFINED_ADDRESS: u64 = u64::MAX;

const MSG_DATASPACE: u8 = 0x01;
const MSG_LINK_INFO: u8 = 0x02;
const MSG_DATATYPE: u8 = 0x03;
const MSG_FILL_VALUE: u8 = 0x05;
const MSG_LINK: u8 = 0x06;
const MSG_DATA_LAYOUT: u8 = 0x08;
const MSG_GROUP_INFO: u8 = 0x0a;
const MSG_FILTER_PIPELINE: u8 = 0x0b;
const MSG_ATTRIBUTE: u8 = 0x0c;

const SUPERBLOCK_V2_SIZE: usize = 48;

/// HDF5 writer errors.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("HDF5 core error: {0}")]
    Core(#[from] hdf5_core::Error),

    #[error("invalid definition: {0}")]
    InvalidDefinition(String),

    /// The supplied data does not match the declared shape/type. `context`
    /// names what was being written (e.g. `dataset 'grid'`).
    #[error("{context}: expected {expected} element(s), got {actual}")]
    DataLengthMismatch {
        context: String,
        expected: usize,
        actual: usize,
    },

    #[error("unsupported write feature: {0}")]
    UnsupportedFeature(String),
}

pub type Result<T> = std::result::Result<T, Error>;

/// HDF5 file variant emitted by the writer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hdf5Variant {
    /// Modern HDF5 superblock/object-header encoding.
    Modern,
}

/// Configuration for HDF5 writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WriteOptions {
    pub byte_order: ByteOrder,
    pub variant: Hdf5Variant,
}

impl Default for WriteOptions {
    fn default() -> Self {
        Self {
            byte_order: ByteOrder::LittleEndian,
            variant: Hdf5Variant::Modern,
        }
    }
}

/// Trait for Rust types that can describe their HDF5 on-disk datatype.
pub trait H5WriteType {
    fn hdf5_type() -> Datatype;
}

/// Primitive element types that can be encoded into contiguous HDF5 raw data.
pub trait H5WriteElement: H5WriteType + Copy {
    fn hdf5_type_with_order(byte_order: ByteOrder) -> Datatype;
    fn write_one(self, byte_order: ByteOrder, dst: &mut Vec<u8>);
}

macro_rules! impl_int_type {
    ($ty:ty, $size:expr, $signed:expr) => {
        impl H5WriteType for $ty {
            fn hdf5_type() -> Datatype {
                <$ty as H5WriteElement>::hdf5_type_with_order(native_order())
            }
        }

        impl H5WriteElement for $ty {
            fn hdf5_type_with_order(byte_order: ByteOrder) -> Datatype {
                Datatype::FixedPoint {
                    size: $size,
                    signed: $signed,
                    byte_order,
                }
            }

            fn write_one(self, byte_order: ByteOrder, dst: &mut Vec<u8>) {
                match byte_order {
                    ByteOrder::LittleEndian => dst.extend_from_slice(&self.to_le_bytes()),
                    ByteOrder::BigEndian => dst.extend_from_slice(&self.to_be_bytes()),
                }
            }
        }
    };
}

impl_int_type!(i16, 2, true);
impl_int_type!(u16, 2, false);
impl_int_type!(i32, 4, true);
impl_int_type!(u32, 4, false);
impl_int_type!(i64, 8, true);
impl_int_type!(u64, 8, false);

macro_rules! impl_byte_type {
    ($ty:ty, $signed:expr) => {
        impl H5WriteType for $ty {
            fn hdf5_type() -> Datatype {
                <$ty as H5WriteElement>::hdf5_type_with_order(native_order())
            }
        }

        impl H5WriteElement for $ty {
            fn hdf5_type_with_order(byte_order: ByteOrder) -> Datatype {
                Datatype::FixedPoint {
                    size: 1,
                    signed: $signed,
                    byte_order,
                }
            }

            fn write_one(self, _byte_order: ByteOrder, dst: &mut Vec<u8>) {
                dst.push(self as u8);
            }
        }
    };
}

impl_byte_type!(i8, true);
impl_byte_type!(u8, false);

impl H5WriteType for f32 {
    fn hdf5_type() -> Datatype {
        <f32 as H5WriteElement>::hdf5_type_with_order(native_order())
    }
}

impl H5WriteElement for f32 {
    fn hdf5_type_with_order(byte_order: ByteOrder) -> Datatype {
        Datatype::FloatingPoint {
            size: 4,
            byte_order,
        }
    }

    fn write_one(self, byte_order: ByteOrder, dst: &mut Vec<u8>) {
        let bits = self.to_bits();
        match byte_order {
            ByteOrder::LittleEndian => dst.extend_from_slice(&bits.to_le_bytes()),
            ByteOrder::BigEndian => dst.extend_from_slice(&bits.to_be_bytes()),
        }
    }
}

impl H5WriteType for f64 {
    fn hdf5_type() -> Datatype {
        <f64 as H5WriteElement>::hdf5_type_with_order(native_order())
    }
}

impl H5WriteElement for f64 {
    fn hdf5_type_with_order(byte_order: ByteOrder) -> Datatype {
        Datatype::FloatingPoint {
            size: 8,
            byte_order,
        }
    }

    fn write_one(self, byte_order: ByteOrder, dst: &mut Vec<u8>) {
        let bits = self.to_bits();
        match byte_order {
            ByteOrder::LittleEndian => dst.extend_from_slice(&bits.to_le_bytes()),
            ByteOrder::BigEndian => dst.extend_from_slice(&bits.to_be_bytes()),
        }
    }
}

fn native_order() -> ByteOrder {
    if cfg!(target_endian = "little") {
        ByteOrder::LittleEndian
    } else {
        ByteOrder::BigEndian
    }
}

/// Dataset definition used by the write planner.
/// Builds a single HDF5 dataset: its name/path, datatype, shape, layout
/// (contiguous, compact, or chunked), optional max shape, filters, fill value,
/// data, and attributes.
#[derive(Debug, Clone)]
pub struct DatasetBuilder {
    name: String,
    datatype: Datatype,
    shape: Vec<u64>,
    max_shape: Option<Vec<u64>>,
    layout: PlannedLayout,
    filters: Vec<FilterDescription>,
    fill_value: Option<Vec<u8>>,
    raw_data: Option<Vec<u8>>,
    vlen_string_values: Option<Vec<String>>,
    vlen_sequence_values: Option<Vec<Vec<u8>>>,
    attributes: Vec<AttributeBuilder>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlannedLayout {
    Compact,
    Contiguous,
    Chunked { chunk_shape: Vec<u64> },
}

impl DatasetBuilder {
    pub fn new(name: impl Into<String>, datatype: Datatype, shape: impl Into<Vec<u64>>) -> Self {
        Self {
            name: name.into(),
            datatype,
            shape: shape.into(),
            max_shape: None,
            layout: PlannedLayout::Contiguous,
            filters: Vec::new(),
            fill_value: None,
            raw_data: None,
            vlen_string_values: None,
            vlen_sequence_values: None,
            attributes: Vec::new(),
        }
    }

    pub fn typed<T: H5WriteType>(name: impl Into<String>, shape: impl Into<Vec<u64>>) -> Self {
        Self::new(name, T::hdf5_type(), shape)
    }

    pub fn typed_with_order<T: H5WriteElement>(
        name: impl Into<String>,
        shape: impl Into<Vec<u64>>,
        byte_order: ByteOrder,
    ) -> Self {
        Self::new(name, T::hdf5_type_with_order(byte_order), shape)
    }

    pub fn typed_data<T: H5WriteElement>(
        name: impl Into<String>,
        shape: impl Into<Vec<u64>>,
        values: &[T],
    ) -> Result<Self> {
        Self::typed::<T>(name, shape).data(values)
    }

    pub fn fixed_string_data<S: AsRef<str>>(
        name: impl Into<String>,
        shape: impl Into<Vec<u64>>,
        values: &[S],
    ) -> Result<Self> {
        let name = name.into();
        let shape = shape.into();
        let expected = expected_element_count(&shape)?;
        if values.len() != expected {
            return Err(Error::DataLengthMismatch {
                context: format!("dataset '{name}' string data"),
                expected,
                actual: values.len(),
            });
        }
        let (datatype, raw_data) = encode_fixed_string_values(values)?;
        Ok(Self::new(name, datatype, shape).raw_data(raw_data))
    }

    pub fn vlen_string_data<S: AsRef<str>>(
        name: impl Into<String>,
        shape: impl Into<Vec<u64>>,
        values: &[S],
    ) -> Result<Self> {
        let name = name.into();
        let shape = shape.into();
        let expected = expected_element_count(&shape)?;
        if values.len() != expected {
            return Err(Error::DataLengthMismatch {
                context: format!("dataset '{name}' string data"),
                expected,
                actual: values.len(),
            });
        }

        let mut strings = Vec::with_capacity(values.len());
        let mut encoding = StringEncoding::Ascii;
        for value in values {
            let value = value.as_ref();
            if value.as_bytes().contains(&0) {
                return Err(Error::InvalidDefinition(
                    "variable-length string values cannot contain NUL bytes".into(),
                ));
            }
            if !value.is_ascii() {
                encoding = StringEncoding::Utf8;
            }
            strings.push(value.to_string());
        }

        Ok(Self {
            name,
            datatype: Datatype::String {
                size: StringSize::Variable,
                encoding,
                padding: StringPadding::NullTerminate,
            },
            shape,
            max_shape: None,
            layout: PlannedLayout::Contiguous,
            filters: Vec::new(),
            fill_value: None,
            raw_data: None,
            vlen_string_values: Some(strings),
            vlen_sequence_values: None,
            attributes: Vec::new(),
        })
    }

    pub fn vlen_sequence_data(
        name: impl Into<String>,
        base: Datatype,
        shape: impl Into<Vec<u64>>,
        values: Vec<Vec<u8>>,
    ) -> Result<Self> {
        let name = name.into();
        let shape = shape.into();
        let expected = expected_element_count(&shape)?;
        if values.len() != expected {
            return Err(Error::DataLengthMismatch {
                context: format!("dataset '{name}' vlen sequence data"),
                expected,
                actual: values.len(),
            });
        }
        validate_vlen_sequence_base(&base)?;
        let base_size = datatype_element_size(&base)?;
        for value in &values {
            if value.len() % base_size != 0 {
                return Err(Error::InvalidDefinition(format!(
                    "dataset '{}' vlen sequence byte length {} is not a multiple of base element size {base_size}",
                    name,
                    value.len()
                )));
            }
        }

        Ok(Self {
            name,
            datatype: Datatype::VarLen {
                base: Box::new(base),
                kind: VarLenKind::Sequence,
                encoding: StringEncoding::Ascii,
                padding: StringPadding::NullTerminate,
            },
            shape,
            max_shape: None,
            layout: PlannedLayout::Contiguous,
            filters: Vec::new(),
            fill_value: None,
            raw_data: None,
            vlen_string_values: None,
            vlen_sequence_values: Some(values),
            attributes: Vec::new(),
        })
    }

    pub fn max_shape(mut self, max_shape: impl Into<Vec<u64>>) -> Self {
        self.max_shape = Some(max_shape.into());
        self
    }

    pub fn compact(mut self) -> Self {
        self.layout = PlannedLayout::Compact;
        self
    }

    pub fn chunked(mut self, chunk_shape: impl Into<Vec<u64>>) -> Self {
        self.layout = PlannedLayout::Chunked {
            chunk_shape: chunk_shape.into(),
        };
        self
    }

    pub fn filter(mut self, filter: FilterDescription) -> Self {
        self.filters.push(filter);
        self
    }

    pub fn fill_value(mut self, bytes: impl Into<Vec<u8>>) -> Self {
        self.fill_value = Some(bytes.into());
        self
    }

    pub fn raw_data(mut self, bytes: impl Into<Vec<u8>>) -> Self {
        self.raw_data = Some(bytes.into());
        self.vlen_string_values = None;
        self.vlen_sequence_values = None;
        self
    }

    pub fn data<T: H5WriteElement>(mut self, values: &[T]) -> Result<Self> {
        let byte_order = numeric_datatype_order(&self.datatype)?;
        ensure_datatype_matches_element::<T>(&self.datatype)?;
        let expected = expected_data_len(&self.shape, datatype_element_size(&self.datatype)?)?;
        let actual = std::mem::size_of_val(values);
        if actual != expected {
            return Err(Error::DataLengthMismatch {
                context: format!("dataset '{}' data (bytes)", self.name),
                expected,
                actual,
            });
        }

        let mut bytes = Vec::with_capacity(actual);
        for &value in values {
            value.write_one(byte_order, &mut bytes);
        }
        self.raw_data = Some(bytes);
        self.vlen_string_values = None;
        self.vlen_sequence_values = None;
        Ok(self)
    }

    pub fn attribute(mut self, attribute: AttributeBuilder) -> Self {
        self.attributes.push(attribute);
        self
    }

    pub fn datatype(&self) -> &Datatype {
        &self.datatype
    }

    pub fn shape(&self) -> &[u64] {
        &self.shape
    }

    pub fn raw_data_bytes(&self) -> Option<&[u8]> {
        self.raw_data.as_deref()
    }

    pub fn validate(&self) -> Result<()> {
        validate_name(&self.name)?;
        if self.shape.len() > u8::MAX as usize {
            return Err(Error::InvalidDefinition(
                "dataset rank exceeds HDF5 rank field capacity".into(),
            ));
        }
        if let Some(max_shape) = &self.max_shape {
            if max_shape.len() != self.shape.len() {
                return Err(Error::InvalidDefinition(
                    "dataset max_shape rank must match shape rank".into(),
                ));
            }
            if !matches!(self.layout, PlannedLayout::Chunked { .. }) {
                return Err(Error::InvalidDefinition(
                    "resizable HDF5 datasets must use chunked layout".into(),
                ));
            }
            for (&dim, &max_dim) in self.shape.iter().zip(max_shape) {
                if max_dim != UNLIMITED && max_dim < dim {
                    return Err(Error::InvalidDefinition(
                        "dataset max_shape cannot be smaller than current shape".into(),
                    ));
                }
            }
        }
        if !self.filters.is_empty() && !matches!(self.layout, PlannedLayout::Chunked { .. }) {
            return Err(Error::InvalidDefinition(
                "filtered HDF5 datasets must use chunked layout".into(),
            ));
        }
        if let PlannedLayout::Chunked { chunk_shape } = &self.layout {
            if chunk_shape.len() != self.shape.len() {
                return Err(Error::InvalidDefinition(
                    "chunk shape rank must match dataset rank".into(),
                ));
            }
            if chunk_shape.contains(&0) {
                return Err(Error::InvalidDefinition(
                    "chunk dimensions must be non-zero".into(),
                ));
            }
        }
        if let Some(fill_value) = &self.fill_value {
            let element_size = datatype_element_size(&self.datatype)?;
            if fill_value.len() != element_size {
                return Err(Error::InvalidDefinition(format!(
                    "dataset '{}' fill value byte length must match datatype element size: expected {element_size}, got {}",
                    self.name,
                    fill_value.len()
                )));
            }
        }
        if let Some(values) = &self.vlen_sequence_values {
            let Datatype::VarLen {
                base,
                kind: VarLenKind::Sequence,
                ..
            } = &self.datatype
            else {
                return Err(Error::InvalidDefinition(format!(
                    "dataset '{}' vlen sequence values require a sequence datatype",
                    self.name
                )));
            };
            let expected = expected_element_count(&self.shape)?;
            if values.len() != expected {
                return Err(Error::InvalidDefinition(format!(
                    "dataset '{}' vlen sequence shape must match value count",
                    self.name
                )));
            }
            validate_vlen_sequence_base(base)?;
            let base_size = datatype_element_size(base)?;
            for value in values {
                if value.len() % base_size != 0 {
                    return Err(Error::InvalidDefinition(format!(
                        "dataset '{}' vlen sequence byte length {} is not a multiple of base element size {base_size}",
                        self.name,
                        value.len()
                    )));
                }
            }
        }
        validate_attributes(&self.attributes)?;
        Ok(())
    }
}

/// Attribute definition used by HDF5 object headers.
/// Builds a single HDF5 attribute attached to a dataset, group, or the file
/// root. Supports scalars, vectors, fixed and variable-length strings, object
/// references, and reference lists.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttributeBuilder {
    name: String,
    datatype: Datatype,
    shape: Vec<u64>,
    raw_data: Vec<u8>,
    vlen_object_reference_targets: Option<Vec<Vec<String>>>,
    vlen_string_values: Option<Vec<String>>,
    vlen_sequence_values: Option<Vec<Vec<u8>>>,
    /// Inline compound `{object reference, dimension}` list, resolved to object
    /// addresses at layout time. Used for HDF5 dimension-scale `REFERENCE_LIST`
    /// back-references.
    object_reference_list: Option<Vec<(String, u32)>>,
}

impl AttributeBuilder {
    pub fn new(
        name: impl Into<String>,
        datatype: Datatype,
        shape: impl Into<Vec<u64>>,
        raw_data: impl Into<Vec<u8>>,
    ) -> Self {
        Self {
            name: name.into(),
            datatype,
            shape: shape.into(),
            raw_data: raw_data.into(),
            vlen_object_reference_targets: None,
            vlen_string_values: None,
            vlen_sequence_values: None,
            object_reference_list: None,
        }
    }

    pub fn scalar<T: H5WriteElement>(name: impl Into<String>, value: T) -> Result<Self> {
        let datatype = T::hdf5_type();
        let byte_order = numeric_datatype_order(&datatype)?;
        let mut raw_data = Vec::with_capacity(datatype_element_size(&datatype)?);
        value.write_one(byte_order, &mut raw_data);
        Ok(Self::new(name, datatype, Vec::new(), raw_data))
    }

    pub fn vector<T: H5WriteElement>(name: impl Into<String>, values: &[T]) -> Result<Self> {
        let datatype = T::hdf5_type();
        let byte_order = numeric_datatype_order(&datatype)?;
        let element_size = datatype_element_size(&datatype)?;
        let mut raw_data = Vec::with_capacity(values.len() * element_size);
        for &value in values {
            value.write_one(byte_order, &mut raw_data);
        }
        Ok(Self::new(
            name,
            datatype,
            vec![values.len() as u64],
            raw_data,
        ))
    }

    pub fn fixed_string(name: impl Into<String>, value: impl AsRef<str>) -> Self {
        let value = value.as_ref();
        let mut raw_data = Vec::with_capacity(value.len() + 1);
        raw_data.extend_from_slice(value.as_bytes());
        raw_data.push(0);
        Self::new(
            name,
            Datatype::String {
                size: StringSize::Fixed(raw_data.len() as u32),
                encoding: if value.is_ascii() {
                    StringEncoding::Ascii
                } else {
                    StringEncoding::Utf8
                },
                padding: StringPadding::NullTerminate,
            },
            Vec::new(),
            raw_data,
        )
    }

    pub fn fixed_string_vector<S: AsRef<str>>(
        name: impl Into<String>,
        values: &[S],
    ) -> Result<Self> {
        let element_count = u64::try_from(values.len()).map_err(|_| {
            Error::InvalidDefinition("string attribute element count exceeds u64 capacity".into())
        })?;
        let (datatype, raw_data) = encode_fixed_string_values(values)?;
        Ok(Self::new(name, datatype, vec![element_count], raw_data))
    }

    pub fn vlen_object_references(
        name: impl Into<String>,
        target_sequences: Vec<Vec<String>>,
    ) -> Self {
        Self {
            name: name.into(),
            datatype: Datatype::VarLen {
                base: Box::new(Datatype::Reference {
                    ref_type: ReferenceType::Object,
                    size: OFFSET_SIZE,
                }),
                kind: VarLenKind::Sequence,
                encoding: StringEncoding::Ascii,
                padding: StringPadding::NullTerminate,
            },
            shape: vec![target_sequences.len() as u64],
            raw_data: Vec::new(),
            vlen_object_reference_targets: Some(target_sequences),
            vlen_string_values: None,
            vlen_sequence_values: None,
            object_reference_list: None,
        }
    }

    /// An inline `{object reference, dimension}` compound list, resolved to
    /// object-header addresses at write time. This is the HDF5 dimension-scale
    /// `REFERENCE_LIST` attribute: `entries` pairs each referencing dataset
    /// (by object path) with the dimension index it attaches the scale to.
    pub fn object_reference_list(name: impl Into<String>, entries: Vec<(String, u32)>) -> Self {
        Self {
            name: name.into(),
            datatype: reference_list_datatype(),
            shape: vec![entries.len() as u64],
            raw_data: Vec::new(),
            vlen_object_reference_targets: None,
            vlen_string_values: None,
            vlen_sequence_values: None,
            object_reference_list: Some(entries),
        }
    }

    pub fn vlen_strings<S: AsRef<str>>(name: impl Into<String>, values: &[S]) -> Result<Self> {
        let element_count = u64::try_from(values.len()).map_err(|_| {
            Error::InvalidDefinition("string attribute element count exceeds u64 capacity".into())
        })?;
        let mut strings = Vec::with_capacity(values.len());
        let mut encoding = StringEncoding::Ascii;
        for value in values {
            let value = value.as_ref();
            if value.as_bytes().contains(&0) {
                return Err(Error::InvalidDefinition(
                    "variable-length string values cannot contain NUL bytes".into(),
                ));
            }
            if !value.is_ascii() {
                encoding = StringEncoding::Utf8;
            }
            strings.push(value.to_string());
        }

        Ok(Self {
            name: name.into(),
            datatype: Datatype::VarLen {
                base: Box::new(Datatype::FixedPoint {
                    size: 1,
                    signed: false,
                    byte_order: ByteOrder::LittleEndian,
                }),
                kind: VarLenKind::String,
                encoding,
                padding: StringPadding::NullTerminate,
            },
            shape: vec![element_count],
            raw_data: Vec::new(),
            vlen_object_reference_targets: None,
            vlen_string_values: Some(strings),
            vlen_sequence_values: None,
            object_reference_list: None,
        })
    }

    pub fn vlen_sequences(
        name: impl Into<String>,
        base: Datatype,
        values: Vec<Vec<u8>>,
    ) -> Result<Self> {
        let name = name.into();
        validate_vlen_sequence_base(&base)?;
        let base_size = datatype_element_size(&base)?;
        for value in &values {
            if value.len() % base_size != 0 {
                return Err(Error::InvalidDefinition(format!(
                    "attribute '{}' vlen sequence byte length {} is not a multiple of base element size {base_size}",
                    name,
                    value.len()
                )));
            }
        }
        let element_count = u64::try_from(values.len()).map_err(|_| {
            Error::InvalidDefinition(
                "vlen sequence attribute element count exceeds u64 capacity".into(),
            )
        })?;

        Ok(Self {
            name,
            datatype: Datatype::VarLen {
                base: Box::new(base),
                kind: VarLenKind::Sequence,
                encoding: StringEncoding::Ascii,
                padding: StringPadding::NullTerminate,
            },
            shape: vec![element_count],
            raw_data: Vec::new(),
            vlen_object_reference_targets: None,
            vlen_string_values: None,
            vlen_sequence_values: Some(values),
            object_reference_list: None,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn datatype(&self) -> &Datatype {
        &self.datatype
    }

    pub fn shape(&self) -> &[u64] {
        &self.shape
    }

    pub fn raw_data(&self) -> &[u8] {
        &self.raw_data
    }

    pub fn validate(&self) -> Result<()> {
        validate_name(&self.name)?;
        if let Some(values) = &self.vlen_string_values {
            if self.shape != [values.len() as u64] {
                return Err(Error::InvalidDefinition(format!(
                    "attribute '{}' vlen string shape must match value count",
                    self.name
                )));
            }
            for value in values {
                if value.as_bytes().contains(&0) {
                    return Err(Error::InvalidDefinition(format!(
                        "attribute '{}' vlen string values cannot contain NUL bytes",
                        self.name
                    )));
                }
            }
            return Ok(());
        }
        if let Some(values) = &self.vlen_sequence_values {
            let Datatype::VarLen {
                base,
                kind: VarLenKind::Sequence,
                ..
            } = &self.datatype
            else {
                return Err(Error::InvalidDefinition(format!(
                    "attribute '{}' vlen sequence values require a sequence datatype",
                    self.name
                )));
            };
            if self.shape != [values.len() as u64] {
                return Err(Error::InvalidDefinition(format!(
                    "attribute '{}' vlen sequence shape must match value count",
                    self.name
                )));
            }
            validate_vlen_sequence_base(base)?;
            let base_size = datatype_element_size(base)?;
            for value in values {
                if value.len() % base_size != 0 {
                    return Err(Error::InvalidDefinition(format!(
                        "attribute '{}' vlen sequence byte length {} is not a multiple of base element size {base_size}",
                        self.name,
                        value.len()
                    )));
                }
            }
            return Ok(());
        }
        if let Some(target_sequences) = &self.vlen_object_reference_targets {
            if self.shape != [target_sequences.len() as u64] {
                return Err(Error::InvalidDefinition(format!(
                    "attribute '{}' vlen reference shape must match sequence count",
                    self.name
                )));
            }
            for sequence in target_sequences {
                for target in sequence {
                    validate_name(target)?;
                }
            }
            return Ok(());
        }
        if let Some(entries) = &self.object_reference_list {
            if self.shape != [entries.len() as u64] {
                return Err(Error::InvalidDefinition(format!(
                    "attribute '{}' reference-list shape must match entry count",
                    self.name
                )));
            }
            for (target, _) in entries {
                validate_name(target)?;
            }
            return Ok(());
        }
        let expected = expected_data_len(&self.shape, datatype_element_size(&self.datatype)?)?;
        if self.raw_data.len() != expected {
            return Err(Error::DataLengthMismatch {
                context: format!("attribute '{}' data (bytes)", self.name),
                expected,
                actual: self.raw_data.len(),
            });
        }
        Ok(())
    }
}

/// Byte size of one `REFERENCE_LIST` compound entry, matching netcdf-c:
/// object reference (8) at offset 0, dimension `u32` at offset 8, padded to 16.
const REFERENCE_LIST_ENTRY_SIZE: u32 = 16;
const REFERENCE_LIST_DIMENSION_OFFSET: usize = 8;

/// The compound datatype netcdf-c uses for dimension-scale `REFERENCE_LIST`
/// attributes: `{ dataset: object reference, dimension: u32 }`.
fn reference_list_datatype() -> Datatype {
    Datatype::Compound {
        size: REFERENCE_LIST_ENTRY_SIZE,
        fields: vec![
            CompoundField {
                name: "dataset".to_string(),
                byte_offset: 0,
                datatype: Datatype::Reference {
                    ref_type: ReferenceType::Object,
                    size: OFFSET_SIZE,
                },
            },
            CompoundField {
                name: "dimension".to_string(),
                byte_offset: REFERENCE_LIST_DIMENSION_OFFSET as u32,
                datatype: Datatype::FixedPoint {
                    size: 4,
                    signed: false,
                    byte_order: ByteOrder::LittleEndian,
                },
            },
        ],
    }
}

/// Attribute attached to an HDF5 group addressed by relative path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupAttributeBuilder {
    path: String,
    attribute: AttributeBuilder,
}

impl GroupAttributeBuilder {
    pub fn new(group_path: impl Into<String>, attribute: AttributeBuilder) -> Self {
        Self {
            path: group_path.into(),
            attribute,
        }
    }

    pub fn path(&self) -> &str {
        &self.path
    }

    pub fn attribute(&self) -> &AttributeBuilder {
        &self.attribute
    }

    pub fn validate(&self) -> Result<()> {
        validate_name(&self.path)?;
        self.attribute.validate()
    }
}

/// Root HDF5 file builder.
/// Accumulates datasets, root attributes, and group attributes, then produces
/// a validated [`Hdf5WritePlan`] via [`Hdf5Builder::into_plan`].
#[derive(Debug, Clone, Default)]
pub struct Hdf5Builder {
    datasets: Vec<DatasetBuilder>,
    attributes: Vec<AttributeBuilder>,
    group_attributes: Vec<GroupAttributeBuilder>,
}

impl Hdf5Builder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn dataset(mut self, dataset: DatasetBuilder) -> Self {
        self.datasets.push(dataset);
        self
    }

    pub fn attribute(mut self, attribute: AttributeBuilder) -> Self {
        self.attributes.push(attribute);
        self
    }

    pub fn group_attribute(
        mut self,
        group_path: impl Into<String>,
        attribute: AttributeBuilder,
    ) -> Self {
        self.group_attributes
            .push(GroupAttributeBuilder::new(group_path, attribute));
        self
    }

    pub fn validate(&self) -> Result<()> {
        let mut names = std::collections::BTreeSet::new();
        let mut group_paths = std::collections::BTreeSet::new();
        for dataset in &self.datasets {
            dataset.validate()?;
            if !names.insert(dataset.name.as_str()) {
                return Err(Error::InvalidDefinition(format!(
                    "duplicate root dataset '{}'",
                    dataset.name
                )));
            }
            let mut parent = parent_path(&dataset.name);
            while let Some(path) = parent {
                group_paths.insert(path.to_string());
                parent = parent_path(path);
            }
        }
        for group_attribute in &self.group_attributes {
            group_attribute.validate()?;
            let mut path = Some(group_attribute.path.as_str());
            while let Some(group_path) = path {
                group_paths.insert(group_path.to_string());
                path = parent_path(group_path);
            }
        }
        for dataset in &self.datasets {
            if group_paths.contains(dataset.name.as_str()) {
                return Err(Error::InvalidDefinition(format!(
                    "dataset '{}' conflicts with an implicit HDF5 group at the same path",
                    dataset.name
                )));
            }
        }
        validate_attributes(&self.attributes)?;
        validate_group_attributes(&self.group_attributes)?;
        Ok(())
    }

    pub fn into_plan(self) -> Result<Hdf5WritePlan> {
        self.validate()?;
        Ok(Hdf5WritePlan {
            datasets: self.datasets,
            attributes: self.attributes,
            group_attributes: self.group_attributes,
        })
    }
}

/// Validated HDF5 write plan.
/// A validated set of datasets and attributes ready to be serialized by
/// [`Hdf5Writer`] or [`Hdf5WritePlan::encode`].
#[derive(Debug, Clone)]
pub struct Hdf5WritePlan {
    datasets: Vec<DatasetBuilder>,
    attributes: Vec<AttributeBuilder>,
    group_attributes: Vec<GroupAttributeBuilder>,
}

impl Hdf5WritePlan {
    pub fn datasets(&self) -> &[DatasetBuilder] {
        &self.datasets
    }

    pub fn attributes(&self) -> &[AttributeBuilder] {
        &self.attributes
    }

    pub fn group_attributes(&self) -> &[GroupAttributeBuilder] {
        &self.group_attributes
    }

    pub fn validate(&self) -> Result<()> {
        Hdf5Builder {
            datasets: self.datasets.clone(),
            attributes: self.attributes.clone(),
            group_attributes: self.group_attributes.clone(),
        }
        .validate()
    }

    /// Encode the plan to the complete HDF5 file bytes. Callers that already
    /// hold a plain [`Write`] sink (and cannot supply a [`Seek`] one) can write
    /// these bytes directly, avoiding an intermediate in-memory copy.
    pub fn encode(&self, options: WriteOptions) -> Result<Vec<u8>> {
        encode_hdf5_file(self, options)
    }
}

/// Writes an [`Hdf5WritePlan`] to a seekable sink. Use
/// [`Hdf5WritePlan::encode`] when the sink is a plain [`Write`].
pub struct Hdf5Writer<W: Write + Seek> {
    sink: W,
    options: WriteOptions,
}

impl<W: Write + Seek> Hdf5Writer<W> {
    pub fn new(sink: W, options: WriteOptions) -> Self {
        Self { sink, options }
    }

    /// Encode `plan` and write the file to the sink. Consumes the writer, so a
    /// writer can be finished at most once.
    pub fn finish(mut self, plan: Hdf5WritePlan) -> Result<W> {
        let bytes = encode_hdf5_file(&plan, self.options)?;
        self.sink.seek(SeekFrom::Start(0))?;
        self.sink.write_all(&bytes)?;
        Ok(self.sink)
    }

    pub fn into_inner(self) -> W {
        self.sink
    }
}

fn validate_name(name: &str) -> Result<()> {
    if name.is_empty() {
        return Err(Error::InvalidDefinition("name must not be empty".into()));
    }
    if name == "/" || name.split('/').any(str::is_empty) {
        return Err(Error::InvalidDefinition(format!(
            "invalid HDF5 relative path '{name}'"
        )));
    }
    Ok(())
}

fn validate_attributes(attributes: &[AttributeBuilder]) -> Result<()> {
    let mut names = std::collections::BTreeSet::new();
    for attribute in attributes {
        attribute.validate()?;
        if !names.insert(attribute.name.as_str()) {
            return Err(Error::InvalidDefinition(format!(
                "duplicate attribute '{}'",
                attribute.name
            )));
        }
    }
    Ok(())
}

fn validate_group_attributes(group_attributes: &[GroupAttributeBuilder]) -> Result<()> {
    let mut names_by_group =
        std::collections::BTreeMap::<&str, std::collections::BTreeSet<&str>>::new();
    for group_attribute in group_attributes {
        group_attribute.validate()?;
        let group_names = names_by_group
            .entry(group_attribute.path.as_str())
            .or_default();
        if !group_names.insert(group_attribute.attribute.name.as_str()) {
            return Err(Error::InvalidDefinition(format!(
                "duplicate attribute '{}' on HDF5 group '{}'",
                group_attribute.attribute.name, group_attribute.path
            )));
        }
    }
    Ok(())
}

#[derive(Debug)]
struct DatasetEmission<'a> {
    dataset: &'a DatasetBuilder,
    raw_data: Vec<u8>,
    header_address: u64,
    data_address: u64,
    header: Vec<u8>,
    chunk_index: Option<FixedArrayChunkIndexEmission>,
}

#[derive(Debug)]
struct GroupEmission {
    header_address: u64,
    header: Vec<u8>,
}

#[derive(Debug)]
struct FixedArrayChunkIndexEmission {
    header_address: u64,
    data_block_address: u64,
    header: Vec<u8>,
    data_block: Vec<u8>,
}

#[derive(Debug)]
struct HeaderMessage {
    type_id: u8,
    payload: Vec<u8>,
}

#[derive(Debug, Clone)]
struct PlannedAttribute {
    name: String,
    datatype: Datatype,
    shape: Vec<u64>,
    raw_data: PlannedAttributeRaw,
}

#[derive(Debug, Clone)]
enum PlannedAttributeRaw {
    Inline(Vec<u8>),
    GlobalHeapVLenReferences(Vec<VLenHeapReference>),
}

#[derive(Debug, Clone)]
struct VLenHeapReference {
    sequence_len: u32,
    heap_index: u16,
}

#[derive(Debug, Clone)]
struct GlobalHeapObjectPlan {
    index: u16,
    reference_count: u16,
    data: Vec<u8>,
}

#[derive(Debug, Clone)]
struct PlannedAttributes {
    root: Vec<PlannedAttribute>,
    groups: Vec<Vec<PlannedAttribute>>,
    datasets: Vec<Vec<PlannedAttribute>>,
    dataset_vlen_refs: Vec<Option<Vec<VLenHeapReference>>>,
    heap_objects: Vec<GlobalHeapObjectPlan>,
}

#[derive(Debug, Clone)]
struct GroupHierarchy {
    paths: Vec<String>,
    root_links: Vec<PlannedLink>,
    group_links: Vec<Vec<PlannedLink>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PlannedLink {
    name: String,
    target: PlannedLinkTarget,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlannedLinkTarget {
    Group(usize),
    Dataset(usize),
}

impl PlannedAttribute {
    fn raw_data(&self, heap_address: u64) -> Vec<u8> {
        match &self.raw_data {
            PlannedAttributeRaw::Inline(raw_data) => raw_data.clone(),
            PlannedAttributeRaw::GlobalHeapVLenReferences(refs) => {
                encode_vlen_heap_references(refs, heap_address)
            }
        }
    }
}

fn encode_vlen_heap_references(refs: &[VLenHeapReference], heap_address: u64) -> Vec<u8> {
    let mut raw_data = Vec::with_capacity(refs.len() * (4 + usize::from(OFFSET_SIZE) + 4));
    for reference in refs {
        if reference.heap_index == 0 {
            // Empty sequence: an all-zero reference, never dereferenced
            // (index 0 is the collection's free-space object).
            raw_data.resize(raw_data.len() + 4 + usize::from(OFFSET_SIZE) + 4, 0);
            continue;
        }
        raw_data.extend_from_slice(&reference.sequence_len.to_le_bytes());
        raw_data.extend_from_slice(&heap_address.to_le_bytes());
        raw_data.extend_from_slice(&(u32::from(reference.heap_index)).to_le_bytes());
    }
    raw_data
}

fn encode_hdf5_file(plan: &Hdf5WritePlan, options: WriteOptions) -> Result<Vec<u8>> {
    if options.variant != Hdf5Variant::Modern {
        return Err(Error::UnsupportedFeature(format!(
            "unsupported HDF5 variant {:?}",
            options.variant
        )));
    }

    let mut prepared = prepare_datasets(plan, options)?;
    let groups = plan_group_hierarchy(&prepared, &plan.group_attributes)?;
    let placeholder_group_addresses = vec![0; groups.paths.len()];
    let placeholder_dataset_addresses = vec![0; prepared.len()];
    let placeholder_targets =
        placeholder_target_addresses(&prepared, &groups, &placeholder_group_addresses);
    let placeholder_attributes = plan_attributes(plan, &groups, &placeholder_targets)?;
    let root_header_size = encode_group_header_from_plan(
        &groups.root_links,
        &placeholder_group_addresses,
        &placeholder_dataset_addresses,
        &placeholder_attributes.root,
        0,
    )?
    .len();
    let mut next_address = align_u64(SUPERBLOCK_V2_SIZE as u64, 8);
    let root_address = next_address;
    next_address = align_u64(
        checked_add_u64(
            next_address,
            root_header_size as u64,
            "root object header end",
        )?,
        8,
    );

    let mut group_addresses = Vec::with_capacity(groups.paths.len());
    for (group_links, attributes) in groups
        .group_links
        .iter()
        .zip(&placeholder_attributes.groups)
    {
        group_addresses.push(next_address);
        let placeholder = encode_group_header_from_plan(
            group_links,
            &placeholder_group_addresses,
            &placeholder_dataset_addresses,
            attributes,
            0,
        )?;
        next_address = align_u64(
            checked_add_u64(
                next_address,
                placeholder.len() as u64,
                "group object header end",
            )?,
            8,
        );
    }

    let mut header_addresses = Vec::with_capacity(prepared.len());
    for (prepared_dataset, attributes) in prepared.iter().zip(&placeholder_attributes.datasets) {
        header_addresses.push(next_address);
        let placeholder = encode_dataset_header(prepared_dataset, attributes, 0, 0)?;
        next_address = align_u64(
            checked_add_u64(
                next_address,
                placeholder.len() as u64,
                "dataset object header end",
            )?,
            8,
        );
    }

    let mut chunk_index_addresses = Vec::with_capacity(prepared.len());
    for prepared_dataset in &prepared {
        if let Some(chunk_index) = &prepared_dataset.chunk_index {
            // Block sizes are address-invariant, so a single sizing pass fixes
            // the header and secondary-block addresses.
            let (header_len, data_block_len) = chunk_index_block_sizes(chunk_index)?;
            let header_address = next_address;
            next_address = align_u64(
                checked_add_u64(next_address, header_len as u64, "chunk index header end")?,
                8,
            );
            let data_block_address = next_address;
            next_address = align_u64(
                checked_add_u64(
                    next_address,
                    data_block_len as u64,
                    "chunk index secondary block end",
                )?,
                8,
            );
            chunk_index_addresses.push(Some(FixedArrayChunkIndexAddresses {
                header: header_address,
                data_block: data_block_address,
            }));
        } else {
            chunk_index_addresses.push(None);
        }
    }

    let data_start_address = next_address;

    let target_addresses =
        target_addresses(&prepared, &header_addresses, &groups, &group_addresses);
    let planned_attributes = plan_attributes(plan, &groups, &target_addresses)?;

    let heap = if planned_attributes.heap_objects.is_empty() {
        Vec::new()
    } else {
        encode_global_heap_collection(&planned_attributes.heap_objects)?
    };
    let DataHeapLayout {
        data_addresses,
        heap_address,
        eof_address,
    } = stabilize_vlen_dataset_storage(
        &mut prepared,
        &planned_attributes,
        data_start_address,
        &heap,
    )?;

    let root_header = encode_group_header_from_plan(
        &groups.root_links,
        &group_addresses,
        &header_addresses,
        &planned_attributes.root,
        heap_address,
    )?;

    let mut group_emissions = Vec::with_capacity(groups.paths.len());
    for ((group_links, attributes), header_address) in groups
        .group_links
        .iter()
        .zip(&planned_attributes.groups)
        .zip(group_addresses.iter().copied())
    {
        let header = encode_group_header_from_plan(
            group_links,
            &group_addresses,
            &header_addresses,
            attributes,
            heap_address,
        )?;
        group_emissions.push(GroupEmission {
            header_address,
            header,
        });
    }

    let mut emissions = Vec::with_capacity(prepared.len());
    // Consume `prepared` so each dataset's storage bytes move into the emission
    // instead of being cloned.
    for ((((prepared_dataset, attributes), header_address), data_address), chunk_index_address) in
        prepared
            .into_iter()
            .zip(&planned_attributes.datasets)
            .zip(header_addresses.iter().copied())
            .zip(data_addresses.iter().copied())
            .zip(chunk_index_addresses.iter().copied())
    {
        let layout_address = chunk_index_address.map_or(data_address, |address| address.header);
        let header =
            encode_dataset_header(&prepared_dataset, attributes, layout_address, heap_address)?;
        let chunk_index = if let (Some(chunk_index), Some(addresses)) =
            (&prepared_dataset.chunk_index, chunk_index_address)
        {
            let (header, data_block) = encode_chunk_index_blocks(
                chunk_index,
                addresses.header,
                addresses.data_block,
                data_address,
            )?;
            Some(FixedArrayChunkIndexEmission {
                header_address: addresses.header,
                data_block_address: addresses.data_block,
                header,
                data_block,
            })
        } else {
            None
        };
        emissions.push(DatasetEmission {
            dataset: prepared_dataset.dataset,
            raw_data: prepared_dataset.raw_data,
            header_address,
            data_address,
            header,
            chunk_index,
        });
    }

    let mut file = Vec::with_capacity(checked_usize(eof_address, "HDF5 file size")?);
    file.extend_from_slice(&encode_superblock_v2(root_address, eof_address));
    pad_to_address(&mut file, root_address)?;
    file.extend_from_slice(&root_header);

    for emission in &group_emissions {
        pad_to_address(&mut file, emission.header_address)?;
        file.extend_from_slice(&emission.header);
    }

    for emission in &emissions {
        pad_to_address(&mut file, emission.header_address)?;
        file.extend_from_slice(&emission.header);
    }

    for emission in &emissions {
        if let Some(chunk_index) = &emission.chunk_index {
            pad_to_address(&mut file, chunk_index.header_address)?;
            file.extend_from_slice(&chunk_index.header);
            pad_to_address(&mut file, chunk_index.data_block_address)?;
            file.extend_from_slice(&chunk_index.data_block);
        }
    }

    for emission in &emissions {
        if !matches!(emission.dataset.layout, PlannedLayout::Compact) {
            pad_to_address(&mut file, emission.data_address)?;
            file.extend_from_slice(&emission.raw_data);
        }
    }

    if !heap.is_empty() {
        pad_to_address(&mut file, heap_address)?;
        file.extend_from_slice(&heap);
    }

    pad_to_address(&mut file, eof_address)?;
    Ok(file)
}

#[derive(Debug)]
struct PreparedDataset<'a> {
    dataset: &'a DatasetBuilder,
    filters: Vec<FilterDescription>,
    raw_data: Vec<u8>,
    data_size: u64,
    chunk_index: Option<PreparedChunkIndex>,
}

/// On-disk index structure used to locate a chunked dataset's chunks.
///
/// Datasets with unlimited maximum dimensions require a version-2 B-tree index
/// (the fixed array and implicit indices are only valid for fixed max dims);
/// everything else with a resolvable chunk index uses the fixed array.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChunkIndexKind {
    FixedArray,
    BtreeV2,
}

#[derive(Debug)]
struct PreparedChunkIndex {
    kind: ChunkIndexKind,
    chunks: Vec<PreparedChunkIndexEntry>,
    /// Byte width of the encoded chunk-size field in each entry.
    chunk_size_len: u8,
}

#[derive(Debug)]
struct PreparedChunkIndexEntry {
    relative_address: u64,
    size: u64,
    filter_mask: u32,
    /// Chunk grid coordinates (one per dataset dimension); the B-tree index
    /// records these as scaled offsets.
    scaled_offsets: Vec<u64>,
}

#[derive(Debug, Clone)]
struct DataHeapLayout {
    data_addresses: Vec<u64>,
    heap_address: u64,
    eof_address: u64,
}

#[derive(Debug, Clone, Copy)]
struct FixedArrayChunkIndexAddresses {
    header: u64,
    data_block: u64,
}

fn prepare_datasets(
    plan: &Hdf5WritePlan,
    options: WriteOptions,
) -> Result<Vec<PreparedDataset<'_>>> {
    plan.validate()?;
    let mut prepared = Vec::with_capacity(plan.datasets.len());
    for dataset in &plan.datasets {
        let element_size = datatype_element_size(&dataset.datatype)?;
        let filters = validate_writer_filters(dataset, element_size)?;
        if let Some(datatype_order) = datatype_numeric_order(&dataset.datatype) {
            if datatype_order != options.byte_order {
                return Err(Error::InvalidDefinition(format!(
                    "dataset '{}' datatype byte order {:?} does not match writer byte order {:?}",
                    dataset.name, datatype_order, options.byte_order
                )));
            }
        }
        let expected = expected_data_len(&dataset.shape, element_size)?;
        let (storage_data, chunk_index) = if let Some(values) = &dataset.vlen_string_values {
            if dataset.fill_value.is_some() {
                return Err(Error::InvalidDefinition(format!(
                    "dataset '{}' cannot combine variable-length strings with a fill value",
                    dataset.name
                )));
            }
            if matches!(dataset.layout, PlannedLayout::Compact) {
                return Err(Error::UnsupportedFeature(format!(
                    "compact variable-length string HDF5 datasets are not emitted yet: '{}'",
                    dataset.name
                )));
            }
            let refs = vlen_string_heap_references(&dataset.name, values, None)?;
            let raw_data = encode_vlen_heap_references(&refs, 0);
            if raw_data.len() != expected {
                return Err(Error::InvalidDefinition(format!(
                    "dataset '{}' expects {expected} vlen reference bytes, got {}",
                    dataset.name,
                    raw_data.len()
                )));
            }
            prepare_raw_dataset_storage(
                &raw_data,
                &dataset.shape,
                &dataset.layout,
                &filters,
                element_size,
                dataset.max_shape.as_deref(),
            )?
        } else if let Some(values) = &dataset.vlen_sequence_values {
            if dataset.fill_value.is_some() {
                return Err(Error::InvalidDefinition(format!(
                    "dataset '{}' cannot combine variable-length sequences with a fill value",
                    dataset.name
                )));
            }
            if matches!(dataset.layout, PlannedLayout::Compact) {
                return Err(Error::UnsupportedFeature(format!(
                    "compact variable-length sequence HDF5 datasets are not emitted yet: '{}'",
                    dataset.name
                )));
            }
            let Datatype::VarLen {
                base,
                kind: VarLenKind::Sequence,
                ..
            } = &dataset.datatype
            else {
                return Err(Error::InvalidDefinition(format!(
                    "dataset '{}' vlen sequence values require a sequence datatype",
                    dataset.name
                )));
            };
            validate_vlen_sequence_base(base)?;
            let base_size = datatype_element_size(base)?;
            let refs = vlen_sequence_heap_references(&dataset.name, values, base_size, None)?;
            let raw_data = encode_vlen_heap_references(&refs, 0);
            if raw_data.len() != expected {
                return Err(Error::InvalidDefinition(format!(
                    "dataset '{}' expects {expected} vlen reference bytes, got {}",
                    dataset.name,
                    raw_data.len()
                )));
            }
            prepare_raw_dataset_storage(
                &raw_data,
                &dataset.shape,
                &dataset.layout,
                &filters,
                element_size,
                dataset.max_shape.as_deref(),
            )?
        } else if let Some(raw_data) = dataset.raw_data.as_deref() {
            if raw_data.len() != expected {
                return Err(Error::InvalidDefinition(format!(
                    "dataset '{}' expects {expected} data bytes, got {}",
                    dataset.name,
                    raw_data.len()
                )));
            }
            prepare_raw_dataset_storage(
                raw_data,
                &dataset.shape,
                &dataset.layout,
                &filters,
                element_size,
                dataset.max_shape.as_deref(),
            )?
        } else if dataset.fill_value.is_some() {
            if matches!(dataset.layout, PlannedLayout::Compact) {
                return Err(Error::UnsupportedFeature(format!(
                    "compact HDF5 datasets are not emitted yet: '{}'",
                    dataset.name
                )));
            }
            (Vec::new(), None)
        } else {
            return Err(Error::InvalidDefinition(format!(
                "dataset '{}' has no raw data for binary emission",
                dataset.name
            )));
        };
        let data_size = u64::try_from(storage_data.len()).map_err(|_| {
            Error::InvalidDefinition(format!(
                "dataset '{}' storage byte length exceeds u64 capacity",
                dataset.name
            ))
        })?;

        prepared.push(PreparedDataset {
            dataset,
            filters,
            raw_data: storage_data,
            data_size,
            chunk_index,
        });
    }
    Ok(prepared)
}

fn prepare_raw_dataset_storage(
    raw_data: &[u8],
    shape: &[u64],
    layout: &PlannedLayout,
    filters: &[FilterDescription],
    element_size: usize,
    max_shape: Option<&[u64]>,
) -> Result<(Vec<u8>, Option<PreparedChunkIndex>)> {
    let unlimited = has_unlimited_max_shape(max_shape);
    match layout {
        PlannedLayout::Contiguous | PlannedLayout::Compact => Ok((raw_data.to_vec(), None)),
        PlannedLayout::Chunked { chunk_shape } => {
            if filters.is_empty() {
                let storage = chunked_storage_data(raw_data, shape, chunk_shape, element_size)?;
                // Fixed max dims use the compact implicit index; unlimited dims
                // require a real (B-tree v2) index.
                let index = if unlimited {
                    unfiltered_btree_chunk_index(shape, chunk_shape, element_size)?
                } else {
                    None
                };
                Ok((storage, index))
            } else if chunk_count(shape, chunk_shape)? <= 1 && !unlimited {
                let storage_data =
                    chunked_storage_data(raw_data, shape, chunk_shape, element_size)?;
                if storage_data.is_empty() {
                    Ok((storage_data, None))
                } else {
                    Ok((
                        apply_write_filters(&storage_data, filters, element_size)?,
                        None,
                    ))
                }
            } else {
                let kind = if unlimited {
                    ChunkIndexKind::BtreeV2
                } else {
                    ChunkIndexKind::FixedArray
                };
                filtered_chunk_storage_data(
                    raw_data,
                    shape,
                    chunk_shape,
                    element_size,
                    filters,
                    kind,
                )
            }
        }
    }
}

fn stabilize_vlen_dataset_storage(
    prepared: &mut [PreparedDataset<'_>],
    planned_attributes: &PlannedAttributes,
    data_start_address: u64,
    heap: &[u8],
) -> Result<DataHeapLayout> {
    let mut layout = compute_data_heap_layout(prepared, data_start_address, heap)?;
    for _ in 0..8 {
        let mut address_sensitive_size_changed = false;
        for (prepared_dataset, refs) in prepared
            .iter_mut()
            .zip(&planned_attributes.dataset_vlen_refs)
        {
            let Some(refs) = refs else {
                continue;
            };
            let old_data_size = prepared_dataset.data_size;
            let old_chunk_count = prepared_dataset
                .chunk_index
                .as_ref()
                .map(|chunk_index| chunk_index.chunks.len());
            let (raw_data, chunk_index) = dataset_vlen_storage_data(
                prepared_dataset.dataset,
                refs,
                layout.heap_address,
                &prepared_dataset.filters,
            )?;
            let data_size = u64::try_from(raw_data.len()).map_err(|_| {
                Error::InvalidDefinition(format!(
                    "dataset '{}' storage byte length exceeds u64 capacity",
                    prepared_dataset.dataset.name
                ))
            })?;
            let new_chunk_count = chunk_index
                .as_ref()
                .map(|chunk_index| chunk_index.chunks.len());
            if old_chunk_count != new_chunk_count {
                return Err(Error::InvalidDefinition(format!(
                    "dataset '{}' changed filtered chunk index cardinality during VLen address stabilization",
                    prepared_dataset.dataset.name
                )));
            }
            prepared_dataset.raw_data = raw_data;
            prepared_dataset.data_size = data_size;
            prepared_dataset.chunk_index = chunk_index;
            if data_size != old_data_size {
                address_sensitive_size_changed = true;
            }
        }
        if !address_sensitive_size_changed {
            return Ok(layout);
        }
        layout = compute_data_heap_layout(prepared, data_start_address, heap)?;
    }

    Err(Error::InvalidDefinition(
        "HDF5 VLen dataset storage did not stabilize after repeated address planning".into(),
    ))
}

fn compute_data_heap_layout(
    prepared: &[PreparedDataset<'_>],
    data_start_address: u64,
    heap: &[u8],
) -> Result<DataHeapLayout> {
    let mut next_address = data_start_address;
    let mut data_addresses = Vec::with_capacity(prepared.len());
    for prepared_dataset in prepared {
        data_addresses.push(next_address);
        if !matches!(prepared_dataset.dataset.layout, PlannedLayout::Compact) {
            next_address = align_u64(
                checked_add_u64(
                    next_address,
                    prepared_dataset.raw_data.len() as u64,
                    "dataset raw data end",
                )?,
                8,
            );
        }
    }

    let heap_address = if heap.is_empty() { 0 } else { next_address };
    let eof_address = align_u64(
        checked_add_u64(next_address, heap.len() as u64, "global heap end")?,
        8,
    );
    Ok(DataHeapLayout {
        data_addresses,
        heap_address,
        eof_address,
    })
}

fn plan_group_hierarchy(
    prepared: &[PreparedDataset<'_>],
    group_attributes: &[GroupAttributeBuilder],
) -> Result<GroupHierarchy> {
    let mut group_paths = std::collections::BTreeSet::new();
    for prepared_dataset in prepared {
        let mut parent = parent_path(&prepared_dataset.dataset.name);
        while let Some(path) = parent {
            group_paths.insert(path.to_string());
            parent = parent_path(path);
        }
    }
    for group_attribute in group_attributes {
        let mut path = Some(group_attribute.path.as_str());
        while let Some(group_path) = path {
            group_paths.insert(group_path.to_string());
            path = parent_path(group_path);
        }
    }

    for prepared_dataset in prepared {
        if group_paths.contains(&prepared_dataset.dataset.name) {
            return Err(Error::InvalidDefinition(format!(
                "dataset '{}' conflicts with an implicit HDF5 group at the same path",
                prepared_dataset.dataset.name
            )));
        }
    }

    let paths: Vec<_> = group_paths.into_iter().collect();
    let group_indices: std::collections::BTreeMap<_, _> = paths
        .iter()
        .enumerate()
        .map(|(index, path)| (path.as_str(), index))
        .collect();
    let mut root_names = std::collections::BTreeSet::new();
    let mut group_names = vec![std::collections::BTreeSet::new(); paths.len()];
    let mut root_links = Vec::new();
    let mut group_links = vec![Vec::new(); paths.len()];

    for (group_index, path) in paths.iter().enumerate() {
        let link = PlannedLink {
            name: path_basename(path).to_string(),
            target: PlannedLinkTarget::Group(group_index),
        };
        if let Some(parent) = parent_path(path) {
            let parent_index = group_indices[parent];
            push_group_link(
                &mut group_links[parent_index],
                &mut group_names[parent_index],
                link,
            )?;
        } else {
            push_group_link(&mut root_links, &mut root_names, link)?;
        }
    }

    for (dataset_index, prepared_dataset) in prepared.iter().enumerate() {
        let path = prepared_dataset.dataset.name.as_str();
        let link = PlannedLink {
            name: path_basename(path).to_string(),
            target: PlannedLinkTarget::Dataset(dataset_index),
        };
        if let Some(parent) = parent_path(path) {
            let parent_index = group_indices[parent];
            push_group_link(
                &mut group_links[parent_index],
                &mut group_names[parent_index],
                link,
            )?;
        } else {
            push_group_link(&mut root_links, &mut root_names, link)?;
        }
    }

    Ok(GroupHierarchy {
        paths,
        root_links,
        group_links,
    })
}

fn push_group_link(
    links: &mut Vec<PlannedLink>,
    names: &mut std::collections::BTreeSet<String>,
    link: PlannedLink,
) -> Result<()> {
    if !names.insert(link.name.clone()) {
        return Err(Error::InvalidDefinition(format!(
            "duplicate HDF5 link name '{}'",
            link.name
        )));
    }
    links.push(link);
    Ok(())
}

fn parent_path(path: &str) -> Option<&str> {
    path.rsplit_once('/').map(|(parent, _)| parent)
}

fn path_basename(path: &str) -> &str {
    path.rsplit_once('/').map_or(path, |(_, name)| name)
}

fn placeholder_target_addresses(
    prepared: &[PreparedDataset<'_>],
    groups: &GroupHierarchy,
    group_addresses: &[u64],
) -> std::collections::BTreeMap<String, u64> {
    object_target_addresses(prepared, &vec![0; prepared.len()], groups, group_addresses)
}

fn target_addresses(
    prepared: &[PreparedDataset<'_>],
    header_addresses: &[u64],
    groups: &GroupHierarchy,
    group_addresses: &[u64],
) -> std::collections::BTreeMap<String, u64> {
    object_target_addresses(prepared, header_addresses, groups, group_addresses)
}

fn object_target_addresses(
    prepared: &[PreparedDataset<'_>],
    header_addresses: &[u64],
    groups: &GroupHierarchy,
    group_addresses: &[u64],
) -> std::collections::BTreeMap<String, u64> {
    let mut target_addresses: std::collections::BTreeMap<String, u64> = groups
        .paths
        .iter()
        .zip(group_addresses.iter().copied())
        .map(|(path, address)| (path.clone(), address))
        .collect();
    target_addresses.extend(
        prepared
            .iter()
            .zip(header_addresses.iter().copied())
            .map(|(prepared_dataset, address)| (prepared_dataset.dataset.name.clone(), address)),
    );
    target_addresses
}

fn resolve_planned_links(
    links: &[PlannedLink],
    group_addresses: &[u64],
    dataset_addresses: &[u64],
) -> Result<Vec<(String, u64)>> {
    links
        .iter()
        .map(|link| {
            let address = match link.target {
                PlannedLinkTarget::Group(index) => {
                    *group_addresses.get(index).ok_or_else(|| {
                        Error::InvalidDefinition(format!(
                            "missing planned address for HDF5 group '{}'",
                            link.name
                        ))
                    })?
                }
                PlannedLinkTarget::Dataset(index) => {
                    *dataset_addresses.get(index).ok_or_else(|| {
                        Error::InvalidDefinition(format!(
                            "missing planned address for HDF5 dataset '{}'",
                            link.name
                        ))
                    })?
                }
            };
            Ok((link.name.clone(), address))
        })
        .collect()
}

#[derive(Debug)]
struct FixedArrayChunkEntry {
    address: u64,
    size: u64,
    filter_mask: u32,
    scaled_offsets: Vec<u64>,
}

fn fixed_array_entries(
    chunk_index: &PreparedChunkIndex,
    data_address: u64,
) -> Result<Vec<FixedArrayChunkEntry>> {
    chunk_index
        .chunks
        .iter()
        .map(|chunk| {
            Ok(FixedArrayChunkEntry {
                address: checked_add_u64(
                    data_address,
                    chunk.relative_address,
                    "fixed array chunk address",
                )?,
                size: chunk.size,
                filter_mask: chunk.filter_mask,
                scaled_offsets: chunk.scaled_offsets.clone(),
            })
        })
        .collect()
}

fn fixed_array_entry_count(chunk_index: &PreparedChunkIndex) -> Result<u64> {
    u64::try_from(chunk_index.chunks.len()).map_err(|_| {
        Error::InvalidDefinition("fixed array chunk entry count exceeds u64 capacity".into())
    })
}

/// Byte lengths of a chunk index's two on-disk blocks (header, secondary),
/// computed without needing final addresses — used to lay out the file.
fn chunk_index_block_sizes(chunk_index: &PreparedChunkIndex) -> Result<(usize, usize)> {
    let (header, data_block) = encode_chunk_index_blocks(chunk_index, 0, 0, 0)?;
    Ok((header.len(), data_block.len()))
}

/// Encode a chunk index's header and secondary block for the given addresses.
/// The header goes at `header_address`, the secondary block (fixed-array data
/// block or B-tree leaf) at `data_block_address`, and chunk data starts at
/// `data_address`.
fn encode_chunk_index_blocks(
    chunk_index: &PreparedChunkIndex,
    header_address: u64,
    data_block_address: u64,
    data_address: u64,
) -> Result<(Vec<u8>, Vec<u8>)> {
    let entries = fixed_array_entries(chunk_index, data_address)?;
    let count = fixed_array_entry_count(chunk_index)?;
    match chunk_index.kind {
        ChunkIndexKind::FixedArray => Ok((
            encode_fixed_array_chunk_index_header(
                count,
                chunk_index.chunk_size_len,
                data_block_address,
            )?,
            encode_fixed_array_chunk_index_data_block(
                header_address,
                &entries,
                chunk_index.chunk_size_len,
            )?,
        )),
        ChunkIndexKind::BtreeV2 => {
            let (record_type, record_size, node_size) = btree_v2_index_params(chunk_index)?;
            Ok((
                encode_btree_v2_header(
                    record_type,
                    record_size,
                    node_size,
                    data_block_address,
                    count,
                )?,
                encode_btree_v2_leaf(
                    record_type,
                    record_size,
                    node_size,
                    &entries,
                    chunk_index.chunk_size_len,
                )?,
            ))
        }
    }
}

/// Record type, record size, and single-leaf node size for a v2 B-tree chunk
/// index.
fn btree_v2_index_params(chunk_index: &PreparedChunkIndex) -> Result<(u8, u16, u32)> {
    let filtered = chunk_index.chunk_size_len > 0;
    let record_type = if filtered {
        BTREE_V2_RECORD_TYPE_FILTERED
    } else {
        BTREE_V2_RECORD_TYPE_UNFILTERED
    };
    let ndims = chunk_index
        .chunks
        .first()
        .map_or(0, |chunk| chunk.scaled_offsets.len());
    let record_size = btree_v2_record_size(ndims, chunk_index.chunk_size_len, filtered)?;
    let node_size = btree_v2_node_size(fixed_array_entry_count(chunk_index)?, record_size)?;
    Ok((record_type, record_size, node_size))
}

fn plan_attributes(
    plan: &Hdf5WritePlan,
    groups: &GroupHierarchy,
    target_addresses: &std::collections::BTreeMap<String, u64>,
) -> Result<PlannedAttributes> {
    let mut heap_objects = Vec::new();
    let root = plan_attribute_list(plan.attributes.iter(), target_addresses, &mut heap_objects)?;
    let groups = groups
        .paths
        .iter()
        .map(|path| {
            plan_attribute_list(
                plan.group_attributes.iter().filter_map(|group_attribute| {
                    (group_attribute.path == *path).then_some(&group_attribute.attribute)
                }),
                target_addresses,
                &mut heap_objects,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let datasets = plan
        .datasets
        .iter()
        .map(|dataset| {
            plan_attribute_list(
                dataset.attributes.iter(),
                target_addresses,
                &mut heap_objects,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let dataset_vlen_refs = plan
        .datasets
        .iter()
        .map(|dataset| plan_dataset_vlen_values(dataset, &mut heap_objects))
        .collect::<Result<Vec<_>>>()?;
    Ok(PlannedAttributes {
        root,
        groups,
        datasets,
        dataset_vlen_refs,
        heap_objects,
    })
}

fn plan_attribute_list<'a>(
    attributes: impl IntoIterator<Item = &'a AttributeBuilder>,
    target_addresses: &std::collections::BTreeMap<String, u64>,
    heap_objects: &mut Vec<GlobalHeapObjectPlan>,
) -> Result<Vec<PlannedAttribute>> {
    attributes
        .into_iter()
        .map(|attribute| plan_attribute(attribute, target_addresses, heap_objects))
        .collect()
}

fn plan_attribute(
    attribute: &AttributeBuilder,
    target_addresses: &std::collections::BTreeMap<String, u64>,
    heap_objects: &mut Vec<GlobalHeapObjectPlan>,
) -> Result<PlannedAttribute> {
    attribute.validate()?;
    if let Some(values) = &attribute.vlen_string_values {
        let refs = vlen_string_heap_references(&attribute.name, values, Some(heap_objects))?;
        Ok(PlannedAttribute {
            name: attribute.name.clone(),
            datatype: attribute.datatype.clone(),
            shape: attribute.shape.clone(),
            raw_data: PlannedAttributeRaw::GlobalHeapVLenReferences(refs),
        })
    } else if let Some(values) = &attribute.vlen_sequence_values {
        let Datatype::VarLen {
            base,
            kind: VarLenKind::Sequence,
            ..
        } = &attribute.datatype
        else {
            return Err(Error::InvalidDefinition(format!(
                "attribute '{}' vlen sequence values require a sequence datatype",
                attribute.name
            )));
        };
        validate_vlen_sequence_base(base)?;
        let base_size = datatype_element_size(base)?;
        let refs =
            vlen_sequence_heap_references(&attribute.name, values, base_size, Some(heap_objects))?;
        Ok(PlannedAttribute {
            name: attribute.name.clone(),
            datatype: attribute.datatype.clone(),
            shape: attribute.shape.clone(),
            raw_data: PlannedAttributeRaw::GlobalHeapVLenReferences(refs),
        })
    } else if let Some(target_sequences) = &attribute.vlen_object_reference_targets {
        let mut refs = Vec::with_capacity(target_sequences.len());
        for sequence in target_sequences {
            if sequence.is_empty() {
                refs.push(VLenHeapReference {
                    sequence_len: 0,
                    heap_index: 0,
                });
                continue;
            }
            let index = checked_heap_index(heap_objects.len() + 1)?;
            let mut data = Vec::with_capacity(sequence.len() * usize::from(OFFSET_SIZE));
            for target in sequence {
                let address = target_addresses.get(target).ok_or_else(|| {
                    Error::InvalidDefinition(format!(
                        "attribute '{}' references unknown HDF5 object '{}'",
                        attribute.name, target
                    ))
                })?;
                data.extend_from_slice(&address.to_le_bytes());
            }
            let sequence_len = u32::try_from(sequence.len()).map_err(|_| {
                Error::UnsupportedFeature(format!(
                    "attribute '{}' vlen object-reference sequence exceeds u32 capacity",
                    attribute.name
                ))
            })?;
            heap_objects.push(GlobalHeapObjectPlan {
                index,
                reference_count: 1,
                data,
            });
            refs.push(VLenHeapReference {
                sequence_len,
                heap_index: index,
            });
        }
        Ok(PlannedAttribute {
            name: attribute.name.clone(),
            datatype: attribute.datatype.clone(),
            shape: attribute.shape.clone(),
            raw_data: PlannedAttributeRaw::GlobalHeapVLenReferences(refs),
        })
    } else if let Some(entries) = &attribute.object_reference_list {
        // Inline compound of resolved object references + dimension index,
        // one 16-byte entry each (address, u32 dimension, padding).
        let mut data = Vec::with_capacity(entries.len() * REFERENCE_LIST_ENTRY_SIZE as usize);
        for (target, dimension) in entries {
            let address = target_addresses.get(target).ok_or_else(|| {
                Error::InvalidDefinition(format!(
                    "attribute '{}' references unknown HDF5 object '{}'",
                    attribute.name, target
                ))
            })?;
            let start = data.len();
            data.extend_from_slice(&address.to_le_bytes());
            data.extend_from_slice(&dimension.to_le_bytes());
            data.resize(start + REFERENCE_LIST_ENTRY_SIZE as usize, 0);
        }
        Ok(PlannedAttribute {
            name: attribute.name.clone(),
            datatype: attribute.datatype.clone(),
            shape: attribute.shape.clone(),
            raw_data: PlannedAttributeRaw::Inline(data),
        })
    } else {
        Ok(PlannedAttribute {
            name: attribute.name.clone(),
            datatype: attribute.datatype.clone(),
            shape: attribute.shape.clone(),
            raw_data: PlannedAttributeRaw::Inline(attribute.raw_data.clone()),
        })
    }
}

fn plan_dataset_vlen_values(
    dataset: &DatasetBuilder,
    heap_objects: &mut Vec<GlobalHeapObjectPlan>,
) -> Result<Option<Vec<VLenHeapReference>>> {
    if let Some(values) = dataset.vlen_string_values.as_ref() {
        return vlen_string_heap_references(&dataset.name, values, Some(heap_objects)).map(Some);
    }
    if let Some(values) = dataset.vlen_sequence_values.as_ref() {
        let Datatype::VarLen {
            base,
            kind: VarLenKind::Sequence,
            ..
        } = &dataset.datatype
        else {
            return Err(Error::InvalidDefinition(format!(
                "dataset '{}' vlen sequence values require a sequence datatype",
                dataset.name
            )));
        };
        validate_vlen_sequence_base(base)?;
        let base_size = datatype_element_size(base)?;
        return vlen_sequence_heap_references(&dataset.name, values, base_size, Some(heap_objects))
            .map(Some);
    }
    Ok(None)
}

/// Plan the global-heap references for variable-length string values.
///
/// With `heap_objects` present the strings are written to the heap and each
/// reference points at its real index; with `None` only the sequence lengths
/// are computed (the placeholder pass used for address sizing, heap index 0).
fn vlen_string_heap_references(
    context: &str,
    values: &[String],
    mut heap_objects: Option<&mut Vec<GlobalHeapObjectPlan>>,
) -> Result<Vec<VLenHeapReference>> {
    let mut refs = Vec::with_capacity(values.len());
    for value in values {
        // The stored object is the string plus a NUL terminator.
        let sequence_len = value
            .len()
            .checked_add(1)
            .and_then(|len| u32::try_from(len).ok())
            .ok_or_else(|| {
                Error::UnsupportedFeature(format!(
                    "'{context}' variable-length string exceeds u32 capacity"
                ))
            })?;
        let heap_index = match heap_objects.as_deref_mut() {
            Some(heap) => {
                let index = checked_heap_index(heap.len() + 1)?;
                let mut data = Vec::with_capacity(value.len() + 1);
                data.extend_from_slice(value.as_bytes());
                data.push(0);
                heap.push(GlobalHeapObjectPlan {
                    index,
                    reference_count: 1,
                    data,
                });
                index
            }
            None => 0,
        };
        refs.push(VLenHeapReference {
            sequence_len,
            heap_index,
        });
    }
    Ok(refs)
}

/// Plan the global-heap references for variable-length sequence values. As with
/// [`vlen_string_heap_references`], `None` computes only sequence lengths.
/// Empty sequences never occupy a heap object (index 0).
fn vlen_sequence_heap_references(
    context: &str,
    values: &[Vec<u8>],
    base_size: usize,
    mut heap_objects: Option<&mut Vec<GlobalHeapObjectPlan>>,
) -> Result<Vec<VLenHeapReference>> {
    let mut refs = Vec::with_capacity(values.len());
    for value in values {
        if value.len() % base_size != 0 {
            return Err(Error::InvalidDefinition(format!(
                "'{context}' variable-length sequence byte length {} is not a multiple of base element size {base_size}",
                value.len()
            )));
        }
        let sequence_len = u32::try_from(value.len() / base_size).map_err(|_| {
            Error::UnsupportedFeature(format!(
                "'{context}' variable-length sequence exceeds u32 element capacity"
            ))
        })?;
        let heap_index = match heap_objects.as_deref_mut() {
            Some(heap) if sequence_len != 0 => {
                let index = checked_heap_index(heap.len() + 1)?;
                heap.push(GlobalHeapObjectPlan {
                    index,
                    reference_count: 1,
                    data: value.clone(),
                });
                index
            }
            _ => 0,
        };
        refs.push(VLenHeapReference {
            sequence_len,
            heap_index,
        });
    }
    Ok(refs)
}

fn dataset_vlen_storage_data(
    dataset: &DatasetBuilder,
    refs: &[VLenHeapReference],
    heap_address: u64,
    filters: &[FilterDescription],
) -> Result<(Vec<u8>, Option<PreparedChunkIndex>)> {
    let element_size = datatype_element_size(&dataset.datatype)?;
    let raw_data = encode_vlen_heap_references(refs, heap_address);
    if matches!(&dataset.layout, PlannedLayout::Compact) {
        return Err(Error::UnsupportedFeature(format!(
            "compact variable-length HDF5 datasets are not emitted yet: '{}'",
            dataset.name
        )));
    }
    prepare_raw_dataset_storage(
        &raw_data,
        &dataset.shape,
        &dataset.layout,
        filters,
        element_size,
        dataset.max_shape.as_deref(),
    )
}

fn checked_heap_index(index: usize) -> Result<u16> {
    u16::try_from(index).map_err(|_| {
        Error::UnsupportedFeature("global heap object count exceeds u16 capacity".into())
    })
}

/// Smallest global heap collection libhdf5 accepts (H5HG_MINSIZE).
const GLOBAL_HEAP_MIN_COLLECTION_SIZE: u64 = 4096;

fn encode_global_heap_collection(objects: &[GlobalHeapObjectPlan]) -> Result<Vec<u8>> {
    let mut body = Vec::new();
    for object in objects {
        body.extend_from_slice(&object.index.to_le_bytes());
        body.extend_from_slice(&object.reference_count.to_le_bytes());
        body.extend_from_slice(&[0; 4]);
        body.extend_from_slice(&(object.data.len() as u64).to_le_bytes());
        body.extend_from_slice(&object.data);
        let padded_len = align_u64(object.data.len() as u64, 8) as usize;
        body.resize(body.len() + (padded_len - object.data.len()), 0);
    }

    // Close the collection with the free-space object (index 0) covering the
    // remaining bytes, its declared size including its own 16-byte header.
    // libhdf5 also rejects collections below its 4096-byte floor.
    let used = checked_add_u64(16, body.len() as u64, "global heap size")?;
    let collection_size = align_u64(checked_add_u64(used, 16, "global heap free object end")?, 8)
        .max(GLOBAL_HEAP_MIN_COLLECTION_SIZE);
    let free_space = collection_size - used;
    body.extend_from_slice(&0u16.to_le_bytes());
    body.extend_from_slice(&0u16.to_le_bytes());
    body.extend_from_slice(&[0; 4]);
    body.extend_from_slice(&free_space.to_le_bytes());

    let mut bytes = Vec::with_capacity(checked_usize(collection_size, "global heap size")?);
    bytes.extend_from_slice(b"GCOL");
    bytes.push(1);
    bytes.extend_from_slice(&[0, 0, 0]);
    bytes.extend_from_slice(&collection_size.to_le_bytes());
    bytes.extend_from_slice(&body);
    bytes.resize(checked_usize(collection_size, "global heap size")?, 0);
    Ok(bytes)
}

fn encode_superblock_v2(root_address: u64, eof_address: u64) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(SUPERBLOCK_V2_SIZE);
    bytes.extend_from_slice(&HDF5_MAGIC);
    bytes.push(2);
    bytes.push(OFFSET_SIZE);
    bytes.push(LENGTH_SIZE);
    bytes.push(0);
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&UNDEFINED_ADDRESS.to_le_bytes());
    bytes.extend_from_slice(&eof_address.to_le_bytes());
    bytes.extend_from_slice(&root_address.to_le_bytes());
    let checksum = jenkins_lookup3(&bytes);
    bytes.extend_from_slice(&checksum.to_le_bytes());
    bytes
}

fn encode_group_header_from_plan(
    links: &[PlannedLink],
    group_addresses: &[u64],
    dataset_addresses: &[u64],
    attributes: &[PlannedAttribute],
    heap_address: u64,
) -> Result<Vec<u8>> {
    let resolved_links = resolve_planned_links(links, group_addresses, dataset_addresses)?;
    encode_group_header_from_links(&resolved_links, attributes, heap_address)
}

fn encode_group_header_from_links(
    links: &[(String, u64)],
    attributes: &[PlannedAttribute],
    heap_address: u64,
) -> Result<Vec<u8>> {
    // New-style groups must carry a Link Info message so readers can resolve
    // the link storage style (libhdf5 rejects group headers without one).
    let mut messages = vec![
        HeaderMessage {
            type_id: MSG_LINK_INFO,
            payload: encode_link_info_message(),
        },
        HeaderMessage {
            type_id: MSG_GROUP_INFO,
            payload: encode_group_info_message(),
        },
    ];
    messages.extend(encode_attribute_header_messages(attributes, heap_address)?);
    let link_messages: Result<Vec<_>> = links
        .iter()
        .map(|(name, address)| {
            Ok(HeaderMessage {
                type_id: MSG_LINK,
                payload: encode_hard_link_message(name.as_str(), *address)?,
            })
        })
        .collect();
    messages.extend(link_messages?);
    encode_object_header_v2(&messages)
}

/// Link Info message: version 0, no creation-order tracking. Links are stored
/// compactly in the header, so the fractal heap and name-index B-tree
/// addresses are undefined.
fn encode_link_info_message() -> Vec<u8> {
    let mut bytes = Vec::with_capacity(18);
    bytes.push(0);
    bytes.push(0);
    bytes.extend_from_slice(&UNDEFINED_ADDRESS.to_le_bytes());
    bytes.extend_from_slice(&UNDEFINED_ADDRESS.to_le_bytes());
    bytes
}

/// Group Info message: version 0, default phase-change values and estimates.
fn encode_group_info_message() -> Vec<u8> {
    vec![0, 0]
}

fn encode_dataset_header(
    dataset: &PreparedDataset<'_>,
    attributes: &[PlannedAttribute],
    data_address: u64,
    heap_address: u64,
) -> Result<Vec<u8>> {
    let mut messages = Vec::new();
    messages.push(HeaderMessage {
        type_id: MSG_DATASPACE,
        payload: encode_dataspace_message(
            &dataset.dataset.shape,
            dataset.dataset.max_shape.as_deref(),
        )?,
    });
    messages.push(HeaderMessage {
        type_id: MSG_DATATYPE,
        payload: encode_datatype_message(&dataset.dataset.datatype)?,
    });
    let alloc_time = space_allocation_time(dataset);
    messages.push(HeaderMessage {
        type_id: MSG_FILL_VALUE,
        payload: match &dataset.dataset.fill_value {
            Some(fill_value) => encode_fill_value_message(fill_value, alloc_time)?,
            None => encode_default_fill_value_message(alloc_time),
        },
    });
    if !dataset.filters.is_empty() {
        messages.push(HeaderMessage {
            type_id: MSG_FILTER_PIPELINE,
            payload: encode_filter_pipeline_message(&dataset.filters)?,
        });
    }
    messages.push(HeaderMessage {
        type_id: MSG_DATA_LAYOUT,
        payload: encode_data_layout_message(dataset, data_address)?,
    });
    messages.extend(encode_attribute_header_messages(attributes, heap_address)?);
    encode_object_header_v2(&messages)
}

fn encode_data_layout_message(dataset: &PreparedDataset<'_>, data_address: u64) -> Result<Vec<u8>> {
    let storage_address = if dataset.data_size == 0 {
        UNDEFINED_ADDRESS
    } else {
        data_address
    };
    match &dataset.dataset.layout {
        PlannedLayout::Contiguous => Ok(encode_contiguous_layout_message(
            storage_address,
            dataset.data_size,
        )),
        PlannedLayout::Chunked { chunk_shape } => {
            let element_size = chunk_element_size(&dataset.dataset.datatype)?;
            if let Some(chunk_index) = &dataset.chunk_index {
                match chunk_index.kind {
                    ChunkIndexKind::FixedArray => {
                        let page_bits =
                            fixed_array_page_bits(fixed_array_entry_count(chunk_index)?)?;
                        encode_fixed_array_chunked_layout_message(
                            storage_address,
                            chunk_shape,
                            element_size,
                            page_bits,
                        )
                    }
                    ChunkIndexKind::BtreeV2 => {
                        let (_, _, node_size) = btree_v2_index_params(chunk_index)?;
                        encode_btree_v2_chunked_layout_message(
                            storage_address,
                            chunk_shape,
                            element_size,
                            node_size,
                        )
                    }
                }
            } else if dataset.filters.is_empty() {
                // Fixed max dims, unfiltered: the compact implicit index.
                encode_implicit_chunked_layout_message(storage_address, chunk_shape, element_size)
            } else {
                // Fixed max dims, filtered, single chunk.
                encode_single_chunk_layout_message(
                    storage_address,
                    chunk_shape,
                    dataset.data_size,
                    element_size,
                )
            }
        }
        PlannedLayout::Compact => encode_compact_layout_message(&dataset.raw_data),
    }
}

/// Byte width of one dataset element as stored in chunked data, appended as
/// the trailing chunk dimension in v3/v4 layout messages.
fn chunk_element_size(datatype: &Datatype) -> Result<u32> {
    let size = datatype_element_size(datatype)?;
    u32::try_from(size).map_err(|_| {
        Error::UnsupportedFeature("chunked element size exceeds layout message capacity".into())
    })
}

/// Fixed-array data-block page size (log2). Readers reject 0, and any entry
/// count above one page would require the paged data-block format the writer
/// does not emit, so size the page to hold every entry.
fn fixed_array_page_bits(num_entries: u64) -> Result<u8> {
    let mut bits = 10u8;
    while (1u64 << bits) < num_entries {
        bits += 1;
        if bits > 25 {
            return Err(Error::UnsupportedFeature(
                "fixed array chunk count exceeds the single-page data block limit".into(),
            ));
        }
    }
    Ok(bits)
}

fn encode_attribute_header_messages(
    attributes: &[PlannedAttribute],
    heap_address: u64,
) -> Result<Vec<HeaderMessage>> {
    attributes
        .iter()
        .map(|attribute| {
            Ok(HeaderMessage {
                type_id: MSG_ATTRIBUTE,
                payload: encode_attribute_message(attribute, heap_address)?,
            })
        })
        .collect()
}

fn encode_object_header_v2(messages: &[HeaderMessage]) -> Result<Vec<u8>> {
    let mut message_bytes = Vec::new();
    for message in messages {
        if message.payload.len() > u16::MAX as usize {
            return Err(Error::UnsupportedFeature(
                "object header continuation messages are not emitted yet".into(),
            ));
        }
        message_bytes.push(message.type_id);
        message_bytes.extend_from_slice(&(message.payload.len() as u16).to_le_bytes());
        message_bytes.push(0);
        message_bytes.extend_from_slice(&message.payload);
    }

    let (size_flag, size_width) = size_width_for(message_bytes.len() as u64)?;
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"OHDR");
    bytes.push(2);
    bytes.push(size_flag);
    write_uvar(message_bytes.len() as u64, size_width, &mut bytes);
    bytes.extend_from_slice(&message_bytes);
    let checksum = jenkins_lookup3(&bytes);
    bytes.extend_from_slice(&checksum.to_le_bytes());
    Ok(bytes)
}

fn encode_hard_link_message(name: &str, address: u64) -> Result<Vec<u8>> {
    let name_bytes = name.as_bytes();
    let (name_len_flag, name_len_width) = size_width_for(name_bytes.len() as u64)?;
    let utf8_flag = if name.is_ascii() { 0 } else { 0x10 };
    let mut bytes = Vec::new();
    bytes.push(1);
    bytes.push(name_len_flag | utf8_flag);
    if utf8_flag != 0 {
        bytes.push(1);
    }
    write_uvar(name_bytes.len() as u64, name_len_width, &mut bytes);
    bytes.extend_from_slice(name_bytes);
    bytes.extend_from_slice(&address.to_le_bytes());
    Ok(bytes)
}

fn encode_attribute_message(attribute: &PlannedAttribute, heap_address: u64) -> Result<Vec<u8>> {
    let name_bytes = attribute.name.as_bytes();
    let datatype = encode_datatype_message(&attribute.datatype)?;
    let dataspace = encode_dataspace_message(&attribute.shape, None)?;
    let raw_data = attribute.raw_data(heap_address);
    if name_bytes.len() + 1 > u16::MAX as usize
        || datatype.len() > u16::MAX as usize
        || dataspace.len() > u16::MAX as usize
    {
        return Err(Error::UnsupportedFeature(format!(
            "attribute '{}' is too large for compact attribute message emission",
            attribute.name
        )));
    }

    let mut bytes = Vec::new();
    bytes.push(3);
    bytes.push(0);
    bytes.extend_from_slice(&((name_bytes.len() + 1) as u16).to_le_bytes());
    bytes.extend_from_slice(&(datatype.len() as u16).to_le_bytes());
    bytes.extend_from_slice(&(dataspace.len() as u16).to_le_bytes());
    bytes.push(if attribute.name.is_ascii() { 0 } else { 1 });
    bytes.extend_from_slice(name_bytes);
    bytes.push(0);
    bytes.extend_from_slice(&datatype);
    bytes.extend_from_slice(&dataspace);
    bytes.extend_from_slice(&raw_data);
    Ok(bytes)
}

fn encode_dataspace_message(shape: &[u64], max_shape: Option<&[u64]>) -> Result<Vec<u8>> {
    if shape.len() > u8::MAX as usize {
        return Err(Error::InvalidDefinition(
            "dataset rank exceeds HDF5 rank field capacity".into(),
        ));
    }
    let mut bytes = Vec::new();
    bytes.push(2);
    bytes.push(shape.len() as u8);
    bytes.push(if max_shape.is_some() { 0x01 } else { 0x00 });
    bytes.push(if shape.is_empty() { 0 } else { 1 });
    for &dim in shape {
        bytes.extend_from_slice(&dim.to_le_bytes());
    }
    if let Some(max_shape) = max_shape {
        for &dim in max_shape {
            bytes.extend_from_slice(&dim.to_le_bytes());
        }
    }
    Ok(bytes)
}

fn encode_datatype_message(datatype: &Datatype) -> Result<Vec<u8>> {
    match datatype {
        Datatype::FixedPoint {
            size,
            signed,
            byte_order,
        } => {
            let mut flags = byte_order_flag(*byte_order);
            if *signed {
                flags |= 0x08;
            }
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&class_word(0, 1, flags).to_le_bytes());
            bytes.extend_from_slice(&u32::from(*size).to_le_bytes());
            bytes.extend_from_slice(&0u16.to_le_bytes());
            bytes.extend_from_slice(&(u16::from(*size) * 8).to_le_bytes());
            Ok(bytes)
        }
        Datatype::FloatingPoint { size, byte_order } => {
            let (exp_location, exp_size, mantissa_size, exp_bias, sign_location) = match *size {
                4 => (23u8, 8u8, 23u8, 127u32, 31u32),
                8 => (52u8, 11u8, 52u8, 1023u32, 63u32),
                other => {
                    return Err(Error::UnsupportedFeature(format!(
                        "unsupported floating-point byte width {other}"
                    )))
                }
            };
            // Class flags: byte order in bit 0, implied mantissa
            // normalization in bits 4-5 (IEEE), and the sign-bit position in
            // bits 8-15 (readers reject a sign bit overlapping the mantissa).
            let flags = byte_order_flag(*byte_order) | 0x20 | (sign_location << 8);
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&class_word(1, 1, flags).to_le_bytes());
            bytes.extend_from_slice(&u32::from(*size).to_le_bytes());
            bytes.extend_from_slice(&0u16.to_le_bytes());
            bytes.extend_from_slice(&(u16::from(*size) * 8).to_le_bytes());
            bytes.push(exp_location);
            bytes.push(exp_size);
            bytes.push(0);
            bytes.push(mantissa_size);
            bytes.extend_from_slice(&exp_bias.to_le_bytes());
            Ok(bytes)
        }
        Datatype::String {
            size,
            encoding,
            padding,
        } => match size {
            StringSize::Fixed(size) => {
                let flags = string_padding_bits(*padding) | (string_encoding_bits(*encoding) << 4);
                let mut bytes = Vec::new();
                bytes.extend_from_slice(&class_word(3, 1, flags).to_le_bytes());
                bytes.extend_from_slice(&size.to_le_bytes());
                Ok(bytes)
            }
            // A variable-length string is a class-9 (variable-length) datatype
            // over the 1-byte fixed-point base libhdf5 uses; class 3 has no
            // zero-size encoding.
            StringSize::Variable => encode_datatype_message(&Datatype::VarLen {
                base: Box::new(Datatype::FixedPoint {
                    size: 1,
                    signed: false,
                    byte_order: ByteOrder::LittleEndian,
                }),
                kind: VarLenKind::String,
                encoding: *encoding,
                padding: *padding,
            }),
        },
        Datatype::Reference { ref_type, size } => {
            let flags = match ref_type {
                hdf5_core::ReferenceType::Object => 0,
                hdf5_core::ReferenceType::DatasetRegion => 1,
            };
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&class_word(7, 1, flags).to_le_bytes());
            bytes.extend_from_slice(&u32::from(*size).to_le_bytes());
            Ok(bytes)
        }
        Datatype::VarLen {
            base,
            kind,
            encoding,
            padding,
        } => {
            let kind_bits = match kind {
                VarLenKind::Sequence => 0,
                VarLenKind::String => 1,
                VarLenKind::Unknown(value) => u32::from(*value),
            };
            let flags = kind_bits
                | (string_padding_bits(*padding) << 4)
                | (string_encoding_bits(*encoding) << 8);
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&class_word(9, 1, flags).to_le_bytes());
            bytes.extend_from_slice(&16u32.to_le_bytes());
            bytes.extend_from_slice(&encode_datatype_message(base)?);
            Ok(bytes)
        }
        Datatype::Bitfield { size, byte_order } => {
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&class_word(4, 1, byte_order_flag(*byte_order)).to_le_bytes());
            bytes.extend_from_slice(&u32::from(*size).to_le_bytes());
            bytes.extend_from_slice(&0u16.to_le_bytes());
            bytes.extend_from_slice(&(u16::from(*size) * 8).to_le_bytes());
            Ok(bytes)
        }
        Datatype::Opaque { size, tag } => {
            let tag_len = if tag.is_empty() {
                0
            } else {
                checked_add_usize(tag.len(), 1, "opaque datatype tag length")?
            };
            if tag.as_bytes().contains(&0) {
                return Err(Error::InvalidDefinition(
                    "opaque datatype tag cannot contain NUL bytes".into(),
                ));
            }
            let flags = u32::try_from(tag_len).map_err(|_| {
                Error::UnsupportedFeature("opaque datatype tag exceeds u32 capacity".into())
            })?;
            if flags > 0xff {
                return Err(Error::UnsupportedFeature(
                    "opaque datatype tag exceeds HDF5 class flag capacity".into(),
                ));
            }
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&class_word(5, 1, flags).to_le_bytes());
            bytes.extend_from_slice(&size.to_le_bytes());
            if tag_len > 0 {
                bytes.extend_from_slice(tag.as_bytes());
                bytes.push(0);
                align_vec(&mut bytes, 8);
            }
            Ok(bytes)
        }
        Datatype::Compound { size, fields } => {
            let n_fields = u32::try_from(fields.len()).map_err(|_| {
                Error::UnsupportedFeature(
                    "compound datatype member count exceeds u32 capacity".into(),
                )
            })?;
            if n_fields > 0xffff {
                return Err(Error::UnsupportedFeature(
                    "compound datatype member count exceeds HDF5 class flag capacity".into(),
                ));
            }
            let offset_size = compound_member_offset_size(*size);
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&class_word(6, 3, n_fields).to_le_bytes());
            bytes.extend_from_slice(&size.to_le_bytes());
            let mut names = std::collections::BTreeSet::new();
            for field in fields {
                encode_datatype_name(&field.name, "compound field")?;
                if !names.insert(field.name.as_str()) {
                    return Err(Error::InvalidDefinition(format!(
                        "duplicate compound field '{}'",
                        field.name
                    )));
                }
                let field_size =
                    u32::try_from(datatype_element_size(&field.datatype)?).map_err(|_| {
                        Error::UnsupportedFeature(format!(
                            "compound field '{}' size exceeds u32 capacity",
                            field.name
                        ))
                    })?;
                let field_end = field.byte_offset.checked_add(field_size).ok_or_else(|| {
                    Error::InvalidDefinition(format!(
                        "compound field '{}' offset overflows datatype size",
                        field.name
                    ))
                })?;
                if field_end > *size {
                    return Err(Error::InvalidDefinition(format!(
                        "compound field '{}' byte range {}..{} is outside datatype size {size}",
                        field.name, field.byte_offset, field_end
                    )));
                }
                bytes.extend_from_slice(field.name.as_bytes());
                bytes.push(0);
                write_uvar(u64::from(field.byte_offset), offset_size, &mut bytes);
                bytes.extend_from_slice(&encode_datatype_message(&field.datatype)?);
            }
            Ok(bytes)
        }
        Datatype::Enum { base, members } => {
            let n_members = u32::try_from(members.len()).map_err(|_| {
                Error::UnsupportedFeature("enum datatype member count exceeds u32 capacity".into())
            })?;
            if n_members > 0xffff {
                return Err(Error::UnsupportedFeature(
                    "enum datatype member count exceeds HDF5 class flag capacity".into(),
                ));
            }
            if !matches!(base.as_ref(), Datatype::FixedPoint { .. }) {
                return Err(Error::InvalidDefinition(
                    "enum datatype base must be fixed-point integer".into(),
                ));
            }
            let value_size = datatype_element_size(base)?;
            let value_size_u32 = u32::try_from(value_size).map_err(|_| {
                Error::UnsupportedFeature("enum datatype value size exceeds u32 capacity".into())
            })?;
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&class_word(8, 3, n_members).to_le_bytes());
            bytes.extend_from_slice(&value_size_u32.to_le_bytes());
            bytes.extend_from_slice(&encode_datatype_message(base)?);
            let mut names = std::collections::BTreeSet::new();
            for member in members {
                encode_datatype_name(&member.name, "enum member")?;
                if !names.insert(member.name.as_str()) {
                    return Err(Error::InvalidDefinition(format!(
                        "duplicate enum member '{}'",
                        member.name
                    )));
                }
                bytes.extend_from_slice(member.name.as_bytes());
                bytes.push(0);
            }
            for member in members {
                if member.value.len() != value_size {
                    return Err(Error::InvalidDefinition(format!(
                        "enum member '{}' value byte length must be {value_size}, got {}",
                        member.name,
                        member.value.len()
                    )));
                }
                bytes.extend_from_slice(&member.value);
            }
            Ok(bytes)
        }
        Datatype::Array { base, dims } => {
            let rank = u8::try_from(dims.len()).map_err(|_| {
                Error::UnsupportedFeature("array datatype rank exceeds HDF5 capacity".into())
            })?;
            let element_size = datatype_element_size(datatype)?;
            let element_size_u32 = u32::try_from(element_size).map_err(|_| {
                Error::UnsupportedFeature("array datatype element size exceeds u32 capacity".into())
            })?;
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&class_word(10, 3, 0).to_le_bytes());
            bytes.extend_from_slice(&element_size_u32.to_le_bytes());
            bytes.push(rank);
            for &dim in dims {
                let dim = u32::try_from(dim).map_err(|_| {
                    Error::UnsupportedFeature(
                        "array datatype dimension exceeds HDF5 u32 capacity".into(),
                    )
                })?;
                bytes.extend_from_slice(&dim.to_le_bytes());
            }
            bytes.extend_from_slice(&encode_datatype_message(base)?);
            Ok(bytes)
        }
    }
}

fn encode_datatype_name(name: &str, context: &str) -> Result<()> {
    if name.is_empty() {
        return Err(Error::InvalidDefinition(format!(
            "{context} name must not be empty"
        )));
    }
    if name.as_bytes().contains(&0) {
        return Err(Error::InvalidDefinition(format!(
            "{context} name cannot contain NUL bytes"
        )));
    }
    Ok(())
}

fn compound_member_offset_size(size: u32) -> usize {
    match size {
        0..=0xff => 1,
        0x100..=0xffff => 2,
        0x1_0000..=0xff_ffff => 3,
        _ => 4,
    }
}

fn align_vec(bytes: &mut Vec<u8>, align: usize) {
    let padding = (align - (bytes.len() % align)) % align;
    bytes.resize(bytes.len() + padding, 0);
}

fn encode_contiguous_layout_message(address: u64, size: u64) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(18);
    bytes.push(3);
    bytes.push(1);
    bytes.extend_from_slice(&address.to_le_bytes());
    bytes.extend_from_slice(&size.to_le_bytes());
    bytes
}

fn encode_compact_layout_message(data: &[u8]) -> Result<Vec<u8>> {
    let size = u16::try_from(data.len()).map_err(|_| {
        Error::UnsupportedFeature(
            "compact HDF5 dataset data exceeds u16 byte length capacity".into(),
        )
    })?;
    let mut bytes = Vec::with_capacity(4 + data.len());
    bytes.push(3);
    bytes.push(0);
    bytes.extend_from_slice(&size.to_le_bytes());
    bytes.extend_from_slice(data);
    Ok(bytes)
}

/// Space-allocation-time codes for the v3 fill value message.
const ALLOC_TIME_EARLY: u8 = 1;
const ALLOC_TIME_LATE: u8 = 2;
const ALLOC_TIME_INCREMENTAL: u8 = 3;
/// Fill-value-write-time "if set" (bits 2-3) shared by every emitted message.
const FILL_WRITE_TIME_IFSET: u8 = 2 << 2;
/// Flag bit 5: the message carries a defined fill value (size + data).
const FILL_VALUE_DEFINED: u8 = 0x20;

/// When the file's data for each layout is allocated. Implicit chunk indexes
/// require early allocation (the writer materializes every chunk up front);
/// libhdf5 pairs the other layouts with these defaults.
fn space_allocation_time(dataset: &PreparedDataset<'_>) -> u8 {
    match &dataset.dataset.layout {
        PlannedLayout::Compact => ALLOC_TIME_EARLY,
        PlannedLayout::Contiguous => ALLOC_TIME_LATE,
        PlannedLayout::Chunked { .. } => {
            if dataset.filters.is_empty() {
                ALLOC_TIME_EARLY
            } else {
                ALLOC_TIME_INCREMENTAL
            }
        }
    }
}

fn encode_fill_value_message(fill_value: &[u8], alloc_time: u8) -> Result<Vec<u8>> {
    let size = u32::try_from(fill_value.len()).map_err(|_| {
        Error::UnsupportedFeature("HDF5 fill value exceeds u32 byte length capacity".into())
    })?;
    let mut bytes = Vec::with_capacity(6 + fill_value.len());
    bytes.push(3);
    bytes.push(alloc_time | FILL_WRITE_TIME_IFSET | FILL_VALUE_DEFINED);
    bytes.extend_from_slice(&size.to_le_bytes());
    bytes.extend_from_slice(fill_value);
    Ok(bytes)
}

/// Fill value message for datasets without a user fill: allocation/write time
/// only, no defined value (readers use the library default of all zeros).
fn encode_default_fill_value_message(alloc_time: u8) -> Vec<u8> {
    vec![3, alloc_time | FILL_WRITE_TIME_IFSET]
}

fn encode_filter_pipeline_message(filters: &[FilterDescription]) -> Result<Vec<u8>> {
    let filter_count = u8::try_from(filters.len()).map_err(|_| {
        Error::UnsupportedFeature("HDF5 filter pipeline exceeds u8 filter count capacity".into())
    })?;
    let mut bytes = Vec::new();
    bytes.push(2);
    bytes.push(filter_count);
    for filter in filters {
        bytes.extend_from_slice(&filter.id.to_le_bytes());
        if filter.id >= 256 {
            let name_len = filter.name.as_ref().map_or(0usize, |name| name.len() + 1);
            let name_len = u16::try_from(name_len).map_err(|_| {
                Error::UnsupportedFeature(format!(
                    "HDF5 filter {} name exceeds u16 byte length capacity",
                    filter.id
                ))
            })?;
            bytes.extend_from_slice(&name_len.to_le_bytes());
        }
        bytes.extend_from_slice(&0u16.to_le_bytes());
        let client_count = u16::try_from(filter.client_data.len()).map_err(|_| {
            Error::UnsupportedFeature(format!(
                "HDF5 filter {} client data exceeds u16 count capacity",
                filter.id
            ))
        })?;
        bytes.extend_from_slice(&client_count.to_le_bytes());
        if filter.id >= 256 {
            if let Some(name) = &filter.name {
                bytes.extend_from_slice(name.as_bytes());
                bytes.push(0);
            }
        }
        for value in &filter.client_data {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    Ok(bytes)
}

/// Flag bit 1 of a v4 chunked layout: the single-chunk index carries the
/// filtered chunk size and filter mask inline.
const V4_LAYOUT_SINGLE_INDEX_WITH_FILTER: u8 = 0x02;

fn encode_single_chunk_layout_message(
    address: u64,
    chunk_shape: &[u64],
    filtered_size: u64,
    element_size: u32,
) -> Result<Vec<u8>> {
    let mut bytes = Vec::with_capacity(22 + chunk_shape.len() * 4 + usize::from(OFFSET_SIZE));
    encode_v4_chunked_layout_preamble(
        &mut bytes,
        V4_LAYOUT_SINGLE_INDEX_WITH_FILTER,
        chunk_shape,
        element_size,
    )?;
    bytes.push(1);
    bytes.extend_from_slice(&filtered_size.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&address.to_le_bytes());
    Ok(bytes)
}

/// Write the v4 chunked layout preamble: version, class, flags, rank, and the
/// chunk dimensions with the element size appended as the trailing dimension
/// (readers validate that dimension against the datatype size).
///
/// Dimensions use the minimal byte width for the largest value; readers
/// recompute that width from the decoded dimensions and reject a mismatch.
fn encode_v4_chunked_layout_preamble(
    bytes: &mut Vec<u8>,
    flags: u8,
    chunk_shape: &[u64],
    element_size: u32,
) -> Result<()> {
    if chunk_shape.len() + 1 > u8::MAX as usize {
        return Err(Error::InvalidDefinition(
            "chunked dataset rank exceeds HDF5 rank field capacity".into(),
        ));
    }
    let max_dim = chunk_shape
        .iter()
        .copied()
        .chain(std::iter::once(u64::from(element_size)))
        .max()
        .unwrap_or(1);
    let dim_bytes = minimal_uint_width(max_dim);
    bytes.push(4);
    bytes.push(2);
    bytes.push(flags);
    bytes.push((chunk_shape.len() + 1) as u8);
    bytes.push(dim_bytes);
    for &dim in chunk_shape {
        write_uvar(dim, usize::from(dim_bytes), bytes);
    }
    write_uvar(u64::from(element_size), usize::from(dim_bytes), bytes);
    Ok(())
}

/// Minimal number of bytes needed to encode `value` (at least 1).
fn minimal_uint_width(value: u64) -> u8 {
    let bits = 64 - value.leading_zeros();
    (bits.max(1)).div_ceil(8) as u8
}

fn encode_implicit_chunked_layout_message(
    address: u64,
    chunk_shape: &[u64],
    element_size: u32,
) -> Result<Vec<u8>> {
    let mut bytes = Vec::with_capacity(10 + chunk_shape.len() * 4 + usize::from(OFFSET_SIZE));
    encode_v4_chunked_layout_preamble(&mut bytes, 0, chunk_shape, element_size)?;
    bytes.push(2);
    bytes.extend_from_slice(&address.to_le_bytes());
    Ok(bytes)
}

fn encode_fixed_array_chunked_layout_message(
    address: u64,
    chunk_shape: &[u64],
    element_size: u32,
    page_bits: u8,
) -> Result<Vec<u8>> {
    let mut bytes = Vec::with_capacity(11 + chunk_shape.len() * 4 + usize::from(OFFSET_SIZE));
    encode_v4_chunked_layout_preamble(&mut bytes, 0, chunk_shape, element_size)?;
    bytes.push(3);
    bytes.push(page_bits);
    bytes.extend_from_slice(&address.to_le_bytes());
    Ok(bytes)
}

/// v2 B-tree chunk-index node parameters. `split`/`merge` percentages match
/// libhdf5's defaults; readers ignore them for a depth-0 tree.
const BTREE_V2_SPLIT_PERCENT: u8 = 100;
const BTREE_V2_MERGE_PERCENT: u8 = 40;
const BTREE_V2_LEAF_OVERHEAD: u64 = 4 + 1 + 1 + 4; // signature + version + type + checksum
const BTREE_V2_RECORD_TYPE_UNFILTERED: u8 = 10;
const BTREE_V2_RECORD_TYPE_FILTERED: u8 = 11;

/// Byte size of one v2 B-tree chunk record.
fn btree_v2_record_size(ndims: usize, chunk_size_len: u8, filtered: bool) -> Result<u16> {
    let offsets = ndims
        .checked_mul(8)
        .ok_or_else(|| Error::InvalidDefinition("chunk offset bytes overflow".into()))?;
    let fixed = usize::from(OFFSET_SIZE)
        + offsets
        + if filtered {
            usize::from(chunk_size_len) + 4
        } else {
            0
        };
    u16::try_from(fixed)
        .map_err(|_| Error::UnsupportedFeature("v2 B-tree chunk record exceeds u16".into()))
}

/// Single-leaf node size: exactly large enough for the leaf header, all
/// records, and the trailing checksum, 8-byte aligned.
fn btree_v2_node_size(num_records: u64, record_size: u16) -> Result<u32> {
    let records_bytes = num_records
        .checked_mul(u64::from(record_size))
        .ok_or_else(|| Error::InvalidDefinition("v2 B-tree leaf size overflow".into()))?;
    let size = align_u64(
        checked_add_u64(BTREE_V2_LEAF_OVERHEAD, records_bytes, "v2 B-tree leaf size")?,
        8,
    );
    u32::try_from(size)
        .map_err(|_| Error::UnsupportedFeature("v2 B-tree node size exceeds u32".into()))
}

fn encode_btree_v2_chunked_layout_message(
    address: u64,
    chunk_shape: &[u64],
    element_size: u32,
    node_size: u32,
) -> Result<Vec<u8>> {
    let mut bytes = Vec::with_capacity(15 + chunk_shape.len() * 4 + usize::from(OFFSET_SIZE));
    encode_v4_chunked_layout_preamble(&mut bytes, 0, chunk_shape, element_size)?;
    bytes.push(5); // index type: v2 B-tree
    bytes.extend_from_slice(&node_size.to_le_bytes());
    bytes.push(BTREE_V2_SPLIT_PERCENT);
    bytes.push(BTREE_V2_MERGE_PERCENT);
    bytes.extend_from_slice(&address.to_le_bytes());
    Ok(bytes)
}

fn encode_btree_v2_header(
    record_type: u8,
    record_size: u16,
    node_size: u32,
    root_node_address: u64,
    num_records: u64,
) -> Result<Vec<u8>> {
    let mut bytes = Vec::with_capacity(4 + 1 + 1 + 4 + 2 + 2 + 1 + 1 + 8 + 2 + 8 + 4);
    bytes.extend_from_slice(b"BTHD");
    bytes.push(0); // version
    bytes.push(record_type);
    bytes.extend_from_slice(&node_size.to_le_bytes());
    bytes.extend_from_slice(&record_size.to_le_bytes());
    bytes.extend_from_slice(&0u16.to_le_bytes()); // depth 0 (single leaf)
    bytes.push(BTREE_V2_SPLIT_PERCENT);
    bytes.push(BTREE_V2_MERGE_PERCENT);
    bytes.extend_from_slice(&root_node_address.to_le_bytes());
    let num_records_u16 = u16::try_from(num_records)
        .map_err(|_| Error::UnsupportedFeature("v2 B-tree root record count exceeds u16".into()))?;
    bytes.extend_from_slice(&num_records_u16.to_le_bytes());
    bytes.extend_from_slice(&num_records.to_le_bytes()); // total records (length size 8)
    let checksum = jenkins_lookup3(&bytes);
    bytes.extend_from_slice(&checksum.to_le_bytes());
    Ok(bytes)
}

/// Encode the single leaf node holding every chunk record. The node is written
/// at its full `node_size`; the checksum sits immediately after the records
/// and the remainder is zero padding.
fn encode_btree_v2_leaf(
    record_type: u8,
    record_size: u16,
    node_size: u32,
    entries: &[FixedArrayChunkEntry],
    chunk_size_len: u8,
) -> Result<Vec<u8>> {
    let filtered = record_type == BTREE_V2_RECORD_TYPE_FILTERED;
    let ndims = entries
        .first()
        .map_or(0, |entry| entry.scaled_offsets.len());
    let mut records = Vec::new();
    records.extend_from_slice(b"BTLF");
    records.push(0); // version
    records.push(record_type);
    let size_limit = if chunk_size_len >= 8 {
        u64::MAX
    } else {
        (1u64 << (8 * u32::from(chunk_size_len))) - 1
    };
    for entry in entries {
        if entry.scaled_offsets.len() != ndims {
            return Err(Error::InvalidDefinition(
                "v2 B-tree chunk records have inconsistent rank".into(),
            ));
        }
        records.extend_from_slice(&entry.address.to_le_bytes());
        if filtered {
            if entry.size > size_limit {
                return Err(Error::InvalidDefinition(format!(
                    "filtered chunk byte length {} exceeds the {chunk_size_len}-byte record field",
                    entry.size
                )));
            }
            write_uvar(entry.size, usize::from(chunk_size_len), &mut records);
            records.extend_from_slice(&entry.filter_mask.to_le_bytes());
        }
        for &scaled in &entry.scaled_offsets {
            records.extend_from_slice(&scaled.to_le_bytes());
        }
    }
    let _ = record_size;
    let checksum = jenkins_lookup3(&records);
    records.extend_from_slice(&checksum.to_le_bytes());
    let node_size = checked_usize(u64::from(node_size), "v2 B-tree node size")?;
    if records.len() > node_size {
        return Err(Error::InvalidDefinition(
            "v2 B-tree leaf records exceed node size".into(),
        ));
    }
    records.resize(node_size, 0);
    Ok(records)
}

fn encode_fixed_array_chunk_index_header(
    num_entries: u64,
    chunk_size_len: u8,
    data_block_address: u64,
) -> Result<Vec<u8>> {
    let entry_size = u8::try_from(usize::from(OFFSET_SIZE) + usize::from(chunk_size_len) + 4)
        .map_err(|_| Error::UnsupportedFeature("fixed array entry size exceeds u8".into()))?;
    let mut bytes = Vec::with_capacity(4 + 1 + 1 + 1 + 1 + 8 + 8 + 4);
    bytes.extend_from_slice(b"FAHD");
    bytes.push(0);
    bytes.push(1);
    bytes.push(entry_size);
    bytes.push(fixed_array_page_bits(num_entries)?);
    bytes.extend_from_slice(&num_entries.to_le_bytes());
    bytes.extend_from_slice(&data_block_address.to_le_bytes());
    let checksum = jenkins_lookup3(&bytes);
    bytes.extend_from_slice(&checksum.to_le_bytes());
    Ok(bytes)
}

fn encode_fixed_array_chunk_index_data_block(
    header_address: u64,
    entries: &[FixedArrayChunkEntry],
    chunk_size_len: u8,
) -> Result<Vec<u8>> {
    let entry_size = usize::from(OFFSET_SIZE) + usize::from(chunk_size_len) + 4;
    let entries_len = checked_mul_usize(entries.len(), entry_size, "fixed array entries size")?;
    let capacity = checked_add_usize(
        4 + 1 + 1 + usize::from(OFFSET_SIZE),
        checked_add_usize(entries_len, 4, "fixed array data block size")?,
        "fixed array data block size",
    )?;
    let size_limit = if chunk_size_len >= 8 {
        u64::MAX
    } else {
        (1u64 << (8 * u32::from(chunk_size_len))) - 1
    };
    let mut bytes = Vec::with_capacity(capacity);
    bytes.extend_from_slice(b"FADB");
    bytes.push(0);
    bytes.push(1);
    bytes.extend_from_slice(&header_address.to_le_bytes());
    for entry in entries {
        if entry.size > size_limit {
            return Err(Error::InvalidDefinition(format!(
                "filtered chunk byte length {} exceeds the {chunk_size_len}-byte entry field",
                entry.size
            )));
        }
        bytes.extend_from_slice(&entry.address.to_le_bytes());
        write_uvar(entry.size, usize::from(chunk_size_len), &mut bytes);
        bytes.extend_from_slice(&entry.filter_mask.to_le_bytes());
    }
    let checksum = jenkins_lookup3(&bytes);
    bytes.extend_from_slice(&checksum.to_le_bytes());
    Ok(bytes)
}

fn class_word(class: u8, version: u8, flags: u32) -> u32 {
    u32::from(class) | (u32::from(version) << 4) | (flags << 8)
}

fn byte_order_flag(byte_order: ByteOrder) -> u32 {
    match byte_order {
        ByteOrder::LittleEndian => 0,
        ByteOrder::BigEndian => 1,
    }
}

fn string_padding_bits(padding: StringPadding) -> u32 {
    match padding {
        StringPadding::NullTerminate => 0,
        StringPadding::NullPad => 1,
        StringPadding::SpacePad => 2,
    }
}

fn string_encoding_bits(encoding: StringEncoding) -> u32 {
    match encoding {
        StringEncoding::Ascii => 0,
        StringEncoding::Utf8 => 1,
    }
}

fn datatype_element_size(datatype: &Datatype) -> Result<usize> {
    match datatype {
        Datatype::FixedPoint { size, .. }
        | Datatype::FloatingPoint { size, .. }
        | Datatype::Bitfield { size, .. }
        | Datatype::Reference { size, .. } => Ok(*size as usize),
        Datatype::String {
            size: StringSize::Fixed(size),
            ..
        } => Ok(*size as usize),
        Datatype::Compound { size, .. } | Datatype::Opaque { size, .. } => Ok(*size as usize),
        Datatype::Array { base, dims } => {
            let base_size = datatype_element_size(base)?;
            let count = dims.iter().try_fold(1usize, |acc, &dim| {
                let dim = checked_usize(dim, "array datatype dimension")?;
                checked_mul_usize(acc, dim, "array datatype element count")
            })?;
            checked_mul_usize(base_size, count, "array datatype byte size")
        }
        Datatype::Enum { base, .. } => datatype_element_size(base),
        Datatype::String {
            size: StringSize::Variable,
            ..
        } => Ok(16),
        Datatype::VarLen { .. } => Ok(16),
    }
}

fn validate_vlen_sequence_base(datatype: &Datatype) -> Result<()> {
    if datatype_element_size(datatype)? == 0 {
        return Err(Error::InvalidDefinition(
            "variable-length sequence base datatype must have non-zero byte size".into(),
        ));
    }
    match datatype {
        Datatype::String {
            size: StringSize::Variable,
            ..
        }
        | Datatype::VarLen { .. } => Err(Error::UnsupportedFeature(
            "nested heap-backed variable-length sequence bases are not emitted yet".into(),
        )),
        Datatype::Compound { fields, .. } => {
            for field in fields {
                validate_vlen_sequence_base(&field.datatype)?;
            }
            Ok(())
        }
        Datatype::Array { base, .. } | Datatype::Enum { base, .. } => {
            validate_vlen_sequence_base(base)
        }
        Datatype::FixedPoint { .. }
        | Datatype::FloatingPoint { .. }
        | Datatype::Bitfield { .. }
        | Datatype::Reference { .. }
        | Datatype::String {
            size: StringSize::Fixed(_),
            ..
        }
        | Datatype::Opaque { .. } => Ok(()),
    }
}

fn validate_writer_filters(
    dataset: &DatasetBuilder,
    element_size: usize,
) -> Result<Vec<FilterDescription>> {
    if dataset.filters.is_empty() {
        return Ok(Vec::new());
    }
    let PlannedLayout::Chunked { chunk_shape } = &dataset.layout else {
        return Err(Error::InvalidDefinition(
            "filtered HDF5 datasets must use chunked layout".into(),
        ));
    };
    let _ = chunk_count(&dataset.shape, chunk_shape)?;
    let mut filters = Vec::with_capacity(dataset.filters.len());
    for filter in &dataset.filters {
        match filter.id {
            FILTER_DEFLATE => {
                if filter.client_data.len() != 1 || filter.client_data[0] > 9 {
                    return Err(Error::InvalidDefinition(
                        "deflate filter requires one compression level in 0..=9".into(),
                    ));
                }
                if filter.name.is_some() {
                    return Err(Error::InvalidDefinition(
                        "well-known HDF5 deflate filter must not carry a name".into(),
                    ));
                }
                filters.push(filter.clone());
            }
            FILTER_SHUFFLE => {
                if filter.name.is_some() {
                    return Err(Error::InvalidDefinition(
                        "well-known HDF5 shuffle filter must not carry a name".into(),
                    ));
                }
                let element_size_client_data = u32::try_from(element_size).map_err(|_| {
                    Error::UnsupportedFeature(
                        "shuffle filter element size exceeds u32 capacity".into(),
                    )
                })?;
                match filter.client_data.as_slice() {
                    [] => filters.push(FilterDescription {
                        id: FILTER_SHUFFLE,
                        name: None,
                        client_data: vec![element_size_client_data],
                    }),
                    [size] if *size == element_size_client_data => filters.push(filter.clone()),
                    [_] => {
                        return Err(Error::InvalidDefinition(format!(
                            "shuffle filter element size must match datatype element size {element_size}"
                        )));
                    }
                    _ => {
                        return Err(Error::InvalidDefinition(
                            "shuffle filter requires zero or one element-size client value".into(),
                        ));
                    }
                }
            }
            FILTER_FLETCHER32 => {
                if filter.name.is_some() {
                    return Err(Error::InvalidDefinition(
                        "well-known HDF5 Fletcher32 filter must not carry a name".into(),
                    ));
                }
                if !filter.client_data.is_empty() {
                    return Err(Error::InvalidDefinition(
                        "Fletcher32 filter must not carry client data".into(),
                    ));
                }
                filters.push(filter.clone());
            }
            other => {
                return Err(Error::UnsupportedFeature(format!(
                    "HDF5 filter id {other} is not implemented for writing"
                )));
            }
        }
    }
    Ok(filters)
}

fn apply_write_filters(
    data: &[u8],
    filters: &[FilterDescription],
    element_size: usize,
) -> Result<Vec<u8>> {
    let mut current = data.to_vec();
    for filter in filters {
        current = match filter.id {
            FILTER_SHUFFLE => shuffle_filter(&current, element_size),
            FILTER_DEFLATE => deflate_filter(&current, filter)?,
            FILTER_FLETCHER32 => fletcher32_filter(&current),
            other => {
                return Err(Error::UnsupportedFeature(format!(
                    "HDF5 filter id {other} is not implemented for writing"
                )));
            }
        };
    }
    Ok(current)
}

fn shuffle_filter(data: &[u8], element_size: usize) -> Vec<u8> {
    if element_size <= 1 || data.is_empty() {
        return data.to_vec();
    }

    let n_elements = data.len() / element_size;
    if n_elements == 0 {
        return data.to_vec();
    }

    let mut output = vec![0u8; data.len()];
    for byte_idx in 0..element_size {
        let dst_start = byte_idx * n_elements;
        for elem in 0..n_elements {
            output[dst_start + elem] = data[elem * element_size + byte_idx];
        }
    }

    let complete = n_elements * element_size;
    if complete < data.len() {
        output[complete..].copy_from_slice(&data[complete..]);
    }
    output
}

fn fletcher32_filter(data: &[u8]) -> Vec<u8> {
    let mut output = Vec::with_capacity(data.len() + 4);
    output.extend_from_slice(data);
    output.extend_from_slice(&fletcher32(data).to_le_bytes());
    output
}

fn deflate_filter(data: &[u8], filter: &FilterDescription) -> Result<Vec<u8>> {
    let level = filter.client_data[0];
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::new(level));
    encoder.write_all(data)?;
    Ok(encoder.finish()?)
}

fn ensure_datatype_matches_element<T: H5WriteElement>(datatype: &Datatype) -> Result<()> {
    let expected = T::hdf5_type_with_order(numeric_datatype_order(datatype)?);
    if &expected != datatype {
        return Err(Error::InvalidDefinition(format!(
            "typed data does not match dataset datatype: expected {datatype:?}, got {expected:?}"
        )));
    }
    Ok(())
}

fn chunk_count(shape: &[u64], chunk_shape: &[u64]) -> Result<u64> {
    if shape.contains(&0) {
        return Ok(0);
    }
    shape
        .iter()
        .zip(chunk_shape)
        .try_fold(1u64, |acc, (&dim, &chunk_dim)| {
            checked_mul_u64(acc, dim.div_ceil(chunk_dim), "chunk count")
        })
}

fn chunked_storage_data(
    raw_data: &[u8],
    shape: &[u64],
    chunk_shape: &[u64],
    element_size: usize,
) -> Result<Vec<u8>> {
    if shape.is_empty() {
        return Err(Error::InvalidDefinition(
            "chunked scalar HDF5 datasets are not supported".into(),
        ));
    }
    if chunk_shape.len() != shape.len() {
        return Err(Error::InvalidDefinition(
            "chunk shape rank must match dataset rank".into(),
        ));
    }

    let mut chunks_per_dim = Vec::with_capacity(shape.len());
    for (&dim, &chunk_dim) in shape.iter().zip(chunk_shape) {
        if chunk_dim == 0 {
            return Err(Error::InvalidDefinition(
                "chunk dimensions must be non-zero".into(),
            ));
        }
        u32::try_from(chunk_dim).map_err(|_| {
            Error::InvalidDefinition("chunk dimension exceeds HDF5 v4 layout capacity".into())
        })?;
        chunks_per_dim.push(dim.div_ceil(chunk_dim));
    }
    if shape.contains(&0) {
        return Ok(Vec::new());
    }

    let chunk_elements = chunk_shape.iter().try_fold(1u64, |acc, &dim| {
        checked_mul_u64(acc, dim, "chunk element count")
    })?;
    let element_size_u64 = u64::try_from(element_size).map_err(|_| {
        Error::InvalidDefinition("datatype element size exceeds u64 capacity".into())
    })?;
    let chunk_bytes_u64 = checked_mul_u64(chunk_elements, element_size_u64, "chunk byte length")?;
    let total_chunks = chunks_per_dim.iter().try_fold(1u64, |acc, &count| {
        checked_mul_u64(acc, count, "chunk count")
    })?;
    let storage_bytes_u64 = checked_mul_u64(total_chunks, chunk_bytes_u64, "chunked data length")?;
    let storage_bytes = checked_usize(storage_bytes_u64, "chunked data length")?;
    let mut storage = vec![0u8; storage_bytes];

    let dataset_strides = row_major_strides(shape, "dataset stride")?;
    let chunk_strides = row_major_strides(chunk_shape, "chunk stride")?;
    let mut chunk_indices = vec![0u64; shape.len()];

    for chunk_linear in 0..total_chunks {
        let chunk_base = checked_usize(
            checked_mul_u64(chunk_linear, chunk_bytes_u64, "chunk byte offset")?,
            "chunk byte offset",
        )?;
        let mut starts = Vec::with_capacity(shape.len());
        let mut counts = Vec::with_capacity(shape.len());
        for ((&chunk_index, &chunk_dim), &dim) in chunk_indices.iter().zip(chunk_shape).zip(shape) {
            let start = checked_mul_u64(chunk_index, chunk_dim, "chunk start")?;
            starts.push(start);
            counts.push((dim - start).min(chunk_dim));
        }
        let src_start =
            starts
                .iter()
                .zip(&dataset_strides)
                .try_fold(0u64, |acc, (&start, &stride)| {
                    checked_add_u64(
                        acc,
                        checked_mul_u64(start, stride, "chunk source element offset")?,
                        "chunk source element offset",
                    )
                })?;
        copy_chunk_rows(
            raw_data,
            &mut storage,
            element_size,
            &dataset_strides,
            &chunk_strides,
            &counts,
            0,
            src_start,
            0,
            chunk_base,
        )?;
        increment_row_major_index(&mut chunk_indices, &chunks_per_dim);
    }

    Ok(storage)
}

fn filtered_chunk_storage_data(
    raw_data: &[u8],
    shape: &[u64],
    chunk_shape: &[u64],
    element_size: usize,
    filters: &[FilterDescription],
    kind: ChunkIndexKind,
) -> Result<(Vec<u8>, Option<PreparedChunkIndex>)> {
    if shape.is_empty() {
        return Err(Error::InvalidDefinition(
            "chunked scalar HDF5 datasets are not supported".into(),
        ));
    }
    if chunk_shape.len() != shape.len() {
        return Err(Error::InvalidDefinition(
            "chunk shape rank must match dataset rank".into(),
        ));
    }

    let mut chunks_per_dim = Vec::with_capacity(shape.len());
    for (&dim, &chunk_dim) in shape.iter().zip(chunk_shape) {
        if chunk_dim == 0 {
            return Err(Error::InvalidDefinition(
                "chunk dimensions must be non-zero".into(),
            ));
        }
        u32::try_from(chunk_dim).map_err(|_| {
            Error::InvalidDefinition("chunk dimension exceeds HDF5 v4 layout capacity".into())
        })?;
        chunks_per_dim.push(dim.div_ceil(chunk_dim));
    }
    if shape.contains(&0) {
        return Ok((Vec::new(), None));
    }

    let chunk_elements = chunk_shape.iter().try_fold(1u64, |acc, &dim| {
        checked_mul_u64(acc, dim, "chunk element count")
    })?;
    let element_size_u64 = u64::try_from(element_size).map_err(|_| {
        Error::InvalidDefinition("datatype element size exceeds u64 capacity".into())
    })?;
    let chunk_bytes_u64 = checked_mul_u64(chunk_elements, element_size_u64, "chunk byte length")?;
    let chunk_bytes = checked_usize(chunk_bytes_u64, "chunk byte length")?;
    let total_chunks = chunks_per_dim.iter().try_fold(1u64, |acc, &count| {
        checked_mul_u64(acc, count, "chunk count")
    })?;

    let dataset_strides = row_major_strides(shape, "dataset stride")?;
    let chunk_strides = row_major_strides(chunk_shape, "chunk stride")?;
    let mut chunk_indices = vec![0u64; shape.len()];
    let mut storage = Vec::new();
    let mut entries = Vec::with_capacity(checked_usize(total_chunks, "chunk count")?);

    for _ in 0..total_chunks {
        let mut starts = Vec::with_capacity(shape.len());
        let mut counts = Vec::with_capacity(shape.len());
        for ((&chunk_index, &chunk_dim), &dim) in chunk_indices.iter().zip(chunk_shape).zip(shape) {
            let start = checked_mul_u64(chunk_index, chunk_dim, "chunk start")?;
            starts.push(start);
            counts.push((dim - start).min(chunk_dim));
        }
        let src_start =
            starts
                .iter()
                .zip(&dataset_strides)
                .try_fold(0u64, |acc, (&start, &stride)| {
                    checked_add_u64(
                        acc,
                        checked_mul_u64(start, stride, "chunk source element offset")?,
                        "chunk source element offset",
                    )
                })?;

        let mut chunk = vec![0u8; chunk_bytes];
        copy_chunk_rows(
            raw_data,
            &mut chunk,
            element_size,
            &dataset_strides,
            &chunk_strides,
            &counts,
            0,
            src_start,
            0,
            0,
        )?;

        let encoded = apply_write_filters(&chunk, filters, element_size)?;
        let relative_address = u64::try_from(storage.len()).map_err(|_| {
            Error::InvalidDefinition("filtered chunk storage exceeds u64 capacity".into())
        })?;
        let size = u64::try_from(encoded.len()).map_err(|_| {
            Error::InvalidDefinition("filtered chunk byte length exceeds u64 capacity".into())
        })?;
        storage.extend_from_slice(&encoded);
        entries.push(PreparedChunkIndexEntry {
            relative_address,
            size,
            filter_mask: 0,
            scaled_offsets: chunk_indices.clone(),
        });

        increment_row_major_index(&mut chunk_indices, &chunks_per_dim);
    }

    Ok((
        storage,
        Some(PreparedChunkIndex {
            kind,
            chunks: entries,
            chunk_size_len: fixed_array_chunk_size_len(chunk_bytes_u64),
        }),
    ))
}

/// Build a B-tree v2 chunk index over an unfiltered, contiguously-laid-out
/// chunked dataset (the same byte layout the implicit index would use). Needed
/// because datasets with unlimited maximum dimensions may not use the implicit
/// index.
fn unfiltered_btree_chunk_index(
    shape: &[u64],
    chunk_shape: &[u64],
    element_size: usize,
) -> Result<Option<PreparedChunkIndex>> {
    if shape.is_empty() || shape.contains(&0) {
        return Ok(None);
    }
    let mut chunks_per_dim = Vec::with_capacity(shape.len());
    for (&dim, &chunk_dim) in shape.iter().zip(chunk_shape) {
        if chunk_dim == 0 {
            return Err(Error::InvalidDefinition(
                "chunk dimensions must be non-zero".into(),
            ));
        }
        chunks_per_dim.push(dim.div_ceil(chunk_dim));
    }
    let chunk_elements = chunk_shape.iter().try_fold(1u64, |acc, &dim| {
        checked_mul_u64(acc, dim, "chunk element count")
    })?;
    let element_size_u64 = u64::try_from(element_size).map_err(|_| {
        Error::InvalidDefinition("datatype element size exceeds u64 capacity".into())
    })?;
    let chunk_bytes = checked_mul_u64(chunk_elements, element_size_u64, "chunk byte length")?;
    let total_chunks = chunks_per_dim.iter().try_fold(1u64, |acc, &count| {
        checked_mul_u64(acc, count, "chunk count")
    })?;

    let mut chunk_indices = vec![0u64; shape.len()];
    let mut entries = Vec::with_capacity(checked_usize(total_chunks, "chunk count")?);
    let mut relative_address = 0u64;
    for _ in 0..total_chunks {
        entries.push(PreparedChunkIndexEntry {
            relative_address,
            size: chunk_bytes,
            filter_mask: 0,
            scaled_offsets: chunk_indices.clone(),
        });
        relative_address = checked_add_u64(relative_address, chunk_bytes, "chunk offset")?;
        increment_row_major_index(&mut chunk_indices, &chunks_per_dim);
    }
    Ok(Some(PreparedChunkIndex {
        kind: ChunkIndexKind::BtreeV2,
        chunks: entries,
        // Unfiltered records store no chunk-size field.
        chunk_size_len: 0,
    }))
}

/// Whether a max-shape declares any unlimited dimension.
fn has_unlimited_max_shape(max_shape: Option<&[u64]>) -> bool {
    max_shape.is_some_and(|dims| dims.contains(&UNLIMITED))
}

/// Width of the encoded chunk-size field in filtered fixed-array entries.
/// Mirrors libhdf5, which sizes the field from the nominal chunk byte size
/// with one byte of headroom (filters can expand a chunk); readers derive the
/// same width from the dataset properties rather than the stored entry size.
fn fixed_array_chunk_size_len(chunk_bytes: u64) -> u8 {
    let log2 = 63u32.saturating_sub(chunk_bytes.leading_zeros());
    (1 + (log2 + 8) / 8).min(8) as u8
}

#[allow(clippy::too_many_arguments)]
fn copy_chunk_rows(
    src: &[u8],
    dst: &mut [u8],
    element_size: usize,
    src_strides: &[u64],
    chunk_strides: &[u64],
    counts: &[u64],
    dim: usize,
    src_element: u64,
    dst_element: u64,
    dst_chunk_base: usize,
) -> Result<()> {
    let last_dim = counts.len() - 1;
    if dim == last_dim {
        let count = checked_usize(counts[dim], "chunk row element count")?;
        let byte_count = checked_mul_usize(count, element_size, "chunk row byte length")?;
        let src_start = element_byte_offset(src_element, element_size, "chunk source offset")?;
        let dst_relative =
            element_byte_offset(dst_element, element_size, "chunk destination offset")?;
        let dst_start = dst_chunk_base.checked_add(dst_relative).ok_or_else(|| {
            Error::InvalidDefinition("chunk destination offset overflows usize".into())
        })?;
        let src_end = src_start
            .checked_add(byte_count)
            .ok_or_else(|| Error::InvalidDefinition("chunk source end overflows usize".into()))?;
        let dst_end = dst_start.checked_add(byte_count).ok_or_else(|| {
            Error::InvalidDefinition("chunk destination end overflows usize".into())
        })?;
        if src_end > src.len() || dst_end > dst.len() {
            return Err(Error::InvalidDefinition(
                "chunk packing range exceeds dataset storage".into(),
            ));
        }
        dst[dst_start..dst_end].copy_from_slice(&src[src_start..src_end]);
        return Ok(());
    }

    for index in 0..counts[dim] {
        copy_chunk_rows(
            src,
            dst,
            element_size,
            src_strides,
            chunk_strides,
            counts,
            dim + 1,
            checked_add_u64(
                src_element,
                checked_mul_u64(index, src_strides[dim], "chunk source stride")?,
                "chunk source stride",
            )?,
            checked_add_u64(
                dst_element,
                checked_mul_u64(index, chunk_strides[dim], "chunk destination stride")?,
                "chunk destination stride",
            )?,
            dst_chunk_base,
        )?;
    }
    Ok(())
}

fn row_major_strides(dims: &[u64], context: &str) -> Result<Vec<u64>> {
    let mut strides = vec![1u64; dims.len()];
    let mut stride = 1u64;
    for index in (0..dims.len()).rev() {
        strides[index] = stride;
        stride = checked_mul_u64(stride, dims[index], context)?;
    }
    Ok(strides)
}

fn increment_row_major_index(indices: &mut [u64], limits: &[u64]) {
    for dim in (0..indices.len()).rev() {
        indices[dim] += 1;
        if indices[dim] < limits[dim] {
            return;
        }
        indices[dim] = 0;
    }
}

fn element_byte_offset(element_offset: u64, element_size: usize, context: &str) -> Result<usize> {
    let element_size = u64::try_from(element_size).map_err(|_| {
        Error::InvalidDefinition(format!("{context} element size exceeds u64 capacity"))
    })?;
    checked_usize(
        checked_mul_u64(element_offset, element_size, context)?,
        context,
    )
}

fn numeric_datatype_order(datatype: &Datatype) -> Result<ByteOrder> {
    datatype_numeric_order(datatype).ok_or_else(|| {
        Error::UnsupportedFeature(format!(
            "typed data encoding is not implemented for {datatype:?}"
        ))
    })
}

fn datatype_numeric_order(datatype: &Datatype) -> Option<ByteOrder> {
    match datatype {
        Datatype::FixedPoint { byte_order, .. } | Datatype::FloatingPoint { byte_order, .. } => {
            Some(*byte_order)
        }
        _ => None,
    }
}

fn encode_fixed_string_values<S: AsRef<str>>(values: &[S]) -> Result<(Datatype, Vec<u8>)> {
    let mut width = 1usize;
    let mut encoding = StringEncoding::Ascii;
    for value in values {
        let value = value.as_ref();
        if value.as_bytes().contains(&0) {
            return Err(Error::InvalidDefinition(
                "fixed string values cannot contain NUL bytes".into(),
            ));
        }
        width = width.max(value.len());
        if !value.is_ascii() {
            encoding = StringEncoding::Utf8;
        }
    }

    let width_u32 = u32::try_from(width).map_err(|_| {
        Error::InvalidDefinition("fixed string element width exceeds HDF5 capacity".into())
    })?;
    let capacity = checked_mul_usize(values.len(), width, "fixed string data byte length")?;
    let mut raw_data = Vec::with_capacity(capacity);
    for value in values {
        let element_end = raw_data.len().checked_add(width).ok_or_else(|| {
            Error::InvalidDefinition("fixed string data byte length exceeds usize capacity".into())
        })?;
        raw_data.extend_from_slice(value.as_ref().as_bytes());
        raw_data.resize(element_end, 0);
    }

    Ok((
        Datatype::String {
            size: StringSize::Fixed(width_u32),
            encoding,
            padding: StringPadding::NullPad,
        },
        raw_data,
    ))
}

fn expected_element_count(shape: &[u64]) -> Result<usize> {
    if shape.is_empty() {
        return Ok(1);
    }
    shape.iter().try_fold(1usize, |acc, &dim| {
        checked_mul_usize(
            acc,
            checked_usize(dim, "dataset dimension")?,
            "dataset element count",
        )
    })
}

fn expected_data_len(shape: &[u64], element_size: usize) -> Result<usize> {
    let elements = expected_element_count(shape)?;
    checked_mul_usize(elements, element_size, "dataset byte length")
}

fn size_width_for(value: u64) -> Result<(u8, usize)> {
    if value <= u8::MAX as u64 {
        Ok((0, 1))
    } else if value <= u16::MAX as u64 {
        Ok((1, 2))
    } else if value <= u32::MAX as u64 {
        Ok((2, 4))
    } else {
        Ok((3, 8))
    }
}

fn write_uvar(value: u64, width: usize, dst: &mut Vec<u8>) {
    let bytes = value.to_le_bytes();
    dst.extend_from_slice(&bytes[..width]);
}

fn pad_to_address(bytes: &mut Vec<u8>, address: u64) -> Result<()> {
    let address = checked_usize(address, "HDF5 address")?;
    if bytes.len() > address {
        return Err(Error::InvalidDefinition(format!(
            "internal HDF5 layout overlap: current offset {} exceeds target {address}",
            bytes.len()
        )));
    }
    bytes.resize(address, 0);
    Ok(())
}

fn align_u64(value: u64, alignment: u64) -> u64 {
    debug_assert!(alignment.is_power_of_two());
    (value + alignment - 1) & !(alignment - 1)
}

fn checked_usize(value: u64, context: &str) -> Result<usize> {
    usize::try_from(value).map_err(|_| {
        Error::InvalidDefinition(format!(
            "{context} value {value} exceeds platform usize capacity"
        ))
    })
}

fn checked_mul_usize(lhs: usize, rhs: usize, context: &str) -> Result<usize> {
    lhs.checked_mul(rhs).ok_or_else(|| {
        Error::InvalidDefinition(format!("{context} exceeds platform usize capacity"))
    })
}

fn checked_add_usize(lhs: usize, rhs: usize, context: &str) -> Result<usize> {
    lhs.checked_add(rhs).ok_or_else(|| {
        Error::InvalidDefinition(format!("{context} exceeds platform usize capacity"))
    })
}

fn checked_mul_u64(lhs: u64, rhs: u64, context: &str) -> Result<u64> {
    lhs.checked_mul(rhs)
        .ok_or_else(|| Error::InvalidDefinition(format!("{context} exceeds u64 capacity")))
}

fn checked_add_u64(lhs: u64, rhs: u64, context: &str) -> Result<u64> {
    lhs.checked_add(rhs)
        .ok_or_else(|| Error::InvalidDefinition(format!("{context} exceeds u64 capacity")))
}
