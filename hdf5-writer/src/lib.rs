//! Pure-Rust HDF5 writer crate.
//!
//! This crate owns the write-side API surface and shared encoding decisions.
//! The first implementation milestone exposes validated builders and format
//! planning types. Full HDF5 serialization is intentionally gated until the
//! object-header, heap, chunk-index, and checksum encoders are complete.

use flate2::{write::ZlibEncoder, Compression};
use std::io::{Seek, SeekFrom, Write};

pub use hdf5_core::{
    jenkins_lookup3, ByteOrder, ChunkIndexing, DataLayout, DataspaceMessage, DataspaceType,
    Datatype, FillTime, FillValueMessage, FilterDescription, FilterPipelineMessage, ReferenceType,
    StringEncoding, StringPadding, StringSize, VarLenKind, FILTER_DEFLATE, FILTER_FLETCHER32,
    FILTER_LZ4, FILTER_NBIT, FILTER_SCALEOFFSET, FILTER_SHUFFLE, FILTER_SZIP, UNLIMITED,
};

const HDF5_MAGIC: [u8; 8] = [0x89, b'H', b'D', b'F', 0x0d, 0x0a, 0x1a, 0x0a];
const OFFSET_SIZE: u8 = 8;
const LENGTH_SIZE: u8 = 8;
const UNDEFINED_ADDRESS: u64 = u64::MAX;

const MSG_DATASPACE: u8 = 0x01;
const MSG_DATATYPE: u8 = 0x03;
const MSG_FILL_VALUE: u8 = 0x05;
const MSG_LINK: u8 = 0x06;
const MSG_DATA_LAYOUT: u8 = 0x08;
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

    #[error("unsupported write feature: {0}")]
    UnsupportedFeature(String),

    #[error("writer has already been finalized")]
    AlreadyFinalized,
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
            return Err(Error::InvalidDefinition(format!(
                "dataset '{}' expects {expected} string elements, got {}",
                name,
                values.len()
            )));
        }
        let (datatype, raw_data) = encode_fixed_string_values(values)?;
        Ok(Self::new(name, datatype, shape).raw_data(raw_data))
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
        self
    }

    pub fn data<T: H5WriteElement>(mut self, values: &[T]) -> Result<Self> {
        let byte_order = numeric_datatype_order(&self.datatype)?;
        ensure_datatype_matches_element::<T>(&self.datatype)?;
        let expected = expected_data_len(&self.shape, datatype_element_size(&self.datatype)?)?;
        let actual = std::mem::size_of_val(values);
        if actual != expected {
            return Err(Error::InvalidDefinition(format!(
                "dataset '{}' expects {expected} data bytes, got {actual}",
                self.name
            )));
        }

        let mut bytes = Vec::with_capacity(actual);
        for &value in values {
            value.write_one(byte_order, &mut bytes);
        }
        self.raw_data = Some(bytes);
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
        validate_attributes(&self.attributes)?;
        Ok(())
    }
}

/// Attribute definition used by HDF5 object headers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttributeBuilder {
    name: String,
    datatype: Datatype,
    shape: Vec<u64>,
    raw_data: Vec<u8>,
    vlen_object_reference_targets: Option<Vec<Vec<String>>>,
    vlen_string_values: Option<Vec<String>>,
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
        if let Some(target_sequences) = &self.vlen_object_reference_targets {
            if self.shape != [target_sequences.len() as u64] {
                return Err(Error::InvalidDefinition(format!(
                    "attribute '{}' vlen reference shape must match sequence count",
                    self.name
                )));
            }
            for sequence in target_sequences {
                if sequence.is_empty() {
                    return Err(Error::UnsupportedFeature(format!(
                        "attribute '{}' contains an empty vlen object-reference sequence",
                        self.name
                    )));
                }
                for target in sequence {
                    validate_name(target)?;
                }
            }
            return Ok(());
        }
        let expected = expected_data_len(&self.shape, datatype_element_size(&self.datatype)?)?;
        if self.raw_data.len() != expected {
            return Err(Error::InvalidDefinition(format!(
                "attribute '{}' expects {expected} data bytes, got {}",
                self.name,
                self.raw_data.len()
            )));
        }
        Ok(())
    }
}

/// Root HDF5 file builder.
#[derive(Debug, Clone, Default)]
pub struct Hdf5Builder {
    datasets: Vec<DatasetBuilder>,
    attributes: Vec<AttributeBuilder>,
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

    pub fn validate(&self) -> Result<()> {
        let mut names = std::collections::BTreeSet::new();
        let mut implicit_group_paths = std::collections::BTreeSet::new();
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
                implicit_group_paths.insert(path);
                parent = parent_path(path);
            }
        }
        for dataset in &self.datasets {
            if implicit_group_paths.contains(dataset.name.as_str()) {
                return Err(Error::InvalidDefinition(format!(
                    "dataset '{}' conflicts with an implicit HDF5 group at the same path",
                    dataset.name
                )));
            }
        }
        validate_attributes(&self.attributes)?;
        Ok(())
    }

    pub fn into_plan(self) -> Result<Hdf5WritePlan> {
        self.validate()?;
        Ok(Hdf5WritePlan {
            datasets: self.datasets,
            attributes: self.attributes,
        })
    }
}

/// Validated HDF5 write plan.
#[derive(Debug, Clone)]
pub struct Hdf5WritePlan {
    datasets: Vec<DatasetBuilder>,
    attributes: Vec<AttributeBuilder>,
}

impl Hdf5WritePlan {
    pub fn datasets(&self) -> &[DatasetBuilder] {
        &self.datasets
    }

    pub fn attributes(&self) -> &[AttributeBuilder] {
        &self.attributes
    }

    pub fn validate(&self) -> Result<()> {
        Hdf5Builder {
            datasets: self.datasets.clone(),
            attributes: self.attributes.clone(),
        }
        .validate()
    }
}

/// Streaming HDF5 writer placeholder. It validates definitions now and will
/// host the binary emitters as they are added.
pub struct Hdf5Writer<W: Write + Seek> {
    sink: W,
    options: WriteOptions,
    finalized: bool,
}

impl<W: Write + Seek> Hdf5Writer<W> {
    pub fn new(sink: W, options: WriteOptions) -> Self {
        Self {
            sink,
            options,
            finalized: false,
        }
    }

    pub fn finish(mut self, plan: Hdf5WritePlan) -> Result<W> {
        if self.finalized {
            return Err(Error::AlreadyFinalized);
        }
        let bytes = encode_hdf5_file(&plan, self.options)?;
        self.sink.seek(SeekFrom::Start(0))?;
        self.sink.write_all(&bytes)?;
        self.finalized = true;
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

#[derive(Debug)]
struct DatasetEmission<'a> {
    dataset: &'a DatasetBuilder,
    raw_data: &'a [u8],
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
    datasets: Vec<Vec<PlannedAttribute>>,
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
                let mut raw_data =
                    Vec::with_capacity(refs.len() * (4 + usize::from(OFFSET_SIZE) + 4));
                for reference in refs {
                    raw_data.extend_from_slice(&reference.sequence_len.to_le_bytes());
                    raw_data.extend_from_slice(&heap_address.to_le_bytes());
                    raw_data.extend_from_slice(&(u32::from(reference.heap_index)).to_le_bytes());
                }
                raw_data
            }
        }
    }
}

fn encode_hdf5_file(plan: &Hdf5WritePlan, options: WriteOptions) -> Result<Vec<u8>> {
    if options.variant != Hdf5Variant::Modern {
        return Err(Error::UnsupportedFeature(format!(
            "unsupported HDF5 variant {:?}",
            options.variant
        )));
    }

    let prepared = prepare_datasets(plan, options)?;
    let groups = plan_group_hierarchy(&prepared)?;
    let placeholder_group_addresses = vec![0; groups.paths.len()];
    let placeholder_dataset_addresses = vec![0; prepared.len()];
    let placeholder_targets =
        placeholder_target_addresses(&prepared, &groups, &placeholder_group_addresses);
    let placeholder_attributes = plan_attributes(plan, &placeholder_targets)?;
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
    for group_links in &groups.group_links {
        group_addresses.push(next_address);
        let placeholder = encode_group_header_from_plan(
            group_links,
            &placeholder_group_addresses,
            &placeholder_dataset_addresses,
            &[],
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
            let header_address = next_address;
            let placeholder_header =
                encode_fixed_array_chunk_index_header(fixed_array_entry_count(chunk_index)?, 0)?;
            next_address = align_u64(
                checked_add_u64(
                    next_address,
                    placeholder_header.len() as u64,
                    "fixed array chunk index header end",
                )?,
                8,
            );

            let data_block_address = next_address;
            let placeholder_entries = fixed_array_entries(chunk_index, 0)?;
            let placeholder_data_block =
                encode_fixed_array_chunk_index_data_block(header_address, &placeholder_entries)?;
            next_address = align_u64(
                checked_add_u64(
                    next_address,
                    placeholder_data_block.len() as u64,
                    "fixed array chunk index data block end",
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

    let mut data_addresses = Vec::with_capacity(prepared.len());
    for prepared_dataset in &prepared {
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

    let target_addresses =
        target_addresses(&prepared, &header_addresses, &groups, &group_addresses);
    let planned_attributes = plan_attributes(plan, &target_addresses)?;

    let heap_address = if planned_attributes.heap_objects.is_empty() {
        0
    } else {
        next_address
    };
    let heap = if planned_attributes.heap_objects.is_empty() {
        Vec::new()
    } else {
        encode_global_heap_collection(&planned_attributes.heap_objects)?
    };
    next_address = align_u64(
        checked_add_u64(next_address, heap.len() as u64, "global heap end")?,
        8,
    );

    let root_header = encode_group_header_from_plan(
        &groups.root_links,
        &group_addresses,
        &header_addresses,
        &planned_attributes.root,
        heap_address,
    )?;

    let mut group_emissions = Vec::with_capacity(groups.paths.len());
    for (group_links, header_address) in groups
        .group_links
        .iter()
        .zip(group_addresses.iter().copied())
    {
        let header = encode_group_header_from_plan(
            group_links,
            &group_addresses,
            &header_addresses,
            &[],
            heap_address,
        )?;
        group_emissions.push(GroupEmission {
            header_address,
            header,
        });
    }

    let mut emissions = Vec::with_capacity(prepared.len());
    for ((((prepared_dataset, attributes), header_address), data_address), chunk_index_address) in
        prepared
            .iter()
            .zip(&planned_attributes.datasets)
            .zip(header_addresses.iter().copied())
            .zip(data_addresses.iter().copied())
            .zip(chunk_index_addresses.iter().copied())
    {
        let layout_address = chunk_index_address.map_or(data_address, |address| address.header);
        let header =
            encode_dataset_header(prepared_dataset, attributes, layout_address, heap_address)?;
        let chunk_index = if let (Some(chunk_index), Some(addresses)) =
            (&prepared_dataset.chunk_index, chunk_index_address)
        {
            let entries = fixed_array_entries(chunk_index, data_address)?;
            Some(FixedArrayChunkIndexEmission {
                header_address: addresses.header,
                data_block_address: addresses.data_block,
                header: encode_fixed_array_chunk_index_header(
                    fixed_array_entry_count(chunk_index)?,
                    addresses.data_block,
                )?,
                data_block: encode_fixed_array_chunk_index_data_block(addresses.header, &entries)?,
            })
        } else {
            None
        };
        emissions.push(DatasetEmission {
            dataset: prepared_dataset.dataset,
            raw_data: &prepared_dataset.raw_data,
            header_address,
            data_address,
            header,
            chunk_index,
        });
    }

    let eof_address = next_address;
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
            file.extend_from_slice(emission.raw_data);
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
    raw_data: Vec<u8>,
    data_size: u64,
    chunk_index: Option<PreparedChunkIndex>,
}

#[derive(Debug)]
struct PreparedChunkIndex {
    chunks: Vec<PreparedChunkIndexEntry>,
}

#[derive(Debug)]
struct PreparedChunkIndexEntry {
    relative_address: u64,
    size: u64,
    filter_mask: u32,
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
        validate_writer_filters(dataset)?;
        if let Some(datatype_order) = datatype_numeric_order(&dataset.datatype) {
            if datatype_order != options.byte_order {
                return Err(Error::InvalidDefinition(format!(
                    "dataset '{}' datatype byte order {:?} does not match writer byte order {:?}",
                    dataset.name, datatype_order, options.byte_order
                )));
            }
        }
        let expected = expected_data_len(&dataset.shape, element_size)?;
        let (storage_data, chunk_index) = if let Some(raw_data) = dataset.raw_data.as_deref() {
            if raw_data.len() != expected {
                return Err(Error::InvalidDefinition(format!(
                    "dataset '{}' expects {expected} data bytes, got {}",
                    dataset.name,
                    raw_data.len()
                )));
            }
            match &dataset.layout {
                PlannedLayout::Contiguous => (raw_data.to_vec(), None),
                PlannedLayout::Compact => (raw_data.to_vec(), None),
                PlannedLayout::Chunked { chunk_shape } => {
                    if dataset.filters.is_empty() {
                        (
                            chunked_storage_data(
                                raw_data,
                                &dataset.shape,
                                chunk_shape,
                                element_size,
                            )?,
                            None,
                        )
                    } else if chunk_count(&dataset.shape, chunk_shape)? <= 1 {
                        let storage_data = chunked_storage_data(
                            raw_data,
                            &dataset.shape,
                            chunk_shape,
                            element_size,
                        )?;
                        if storage_data.is_empty() {
                            (storage_data, None)
                        } else {
                            (apply_write_filters(&storage_data, &dataset.filters)?, None)
                        }
                    } else {
                        filtered_chunk_storage_data(
                            raw_data,
                            &dataset.shape,
                            chunk_shape,
                            element_size,
                            &dataset.filters,
                        )?
                    }
                }
            }
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
            raw_data: storage_data,
            data_size,
            chunk_index,
        });
    }
    Ok(prepared)
}

fn plan_group_hierarchy(prepared: &[PreparedDataset<'_>]) -> Result<GroupHierarchy> {
    let mut group_paths = std::collections::BTreeSet::new();
    for prepared_dataset in prepared {
        let mut parent = parent_path(&prepared_dataset.dataset.name);
        while let Some(path) = parent {
            group_paths.insert(path.to_string());
            parent = parent_path(path);
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
            })
        })
        .collect()
}

fn fixed_array_entry_count(chunk_index: &PreparedChunkIndex) -> Result<u64> {
    u64::try_from(chunk_index.chunks.len()).map_err(|_| {
        Error::InvalidDefinition("fixed array chunk entry count exceeds u64 capacity".into())
    })
}

fn plan_attributes(
    plan: &Hdf5WritePlan,
    target_addresses: &std::collections::BTreeMap<String, u64>,
) -> Result<PlannedAttributes> {
    let mut heap_objects = Vec::new();
    let root = plan_attribute_list(&plan.attributes, target_addresses, &mut heap_objects)?;
    let datasets = plan
        .datasets
        .iter()
        .map(|dataset| {
            plan_attribute_list(&dataset.attributes, target_addresses, &mut heap_objects)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(PlannedAttributes {
        root,
        datasets,
        heap_objects,
    })
}

fn plan_attribute_list(
    attributes: &[AttributeBuilder],
    target_addresses: &std::collections::BTreeMap<String, u64>,
    heap_objects: &mut Vec<GlobalHeapObjectPlan>,
) -> Result<Vec<PlannedAttribute>> {
    attributes
        .iter()
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
        let mut refs = Vec::with_capacity(values.len());
        for value in values {
            let index = checked_heap_index(heap_objects.len() + 1)?;
            let mut data = Vec::with_capacity(value.len() + 1);
            data.extend_from_slice(value.as_bytes());
            data.push(0);
            let sequence_len = u32::try_from(data.len()).map_err(|_| {
                Error::UnsupportedFeature(format!(
                    "attribute '{}' vlen string exceeds u32 capacity",
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
    } else if let Some(target_sequences) = &attribute.vlen_object_reference_targets {
        let mut refs = Vec::with_capacity(target_sequences.len());
        for sequence in target_sequences {
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
    } else {
        Ok(PlannedAttribute {
            name: attribute.name.clone(),
            datatype: attribute.datatype.clone(),
            shape: attribute.shape.clone(),
            raw_data: PlannedAttributeRaw::Inline(attribute.raw_data.clone()),
        })
    }
}

fn checked_heap_index(index: usize) -> Result<u16> {
    u16::try_from(index).map_err(|_| {
        Error::UnsupportedFeature("global heap object count exceeds u16 capacity".into())
    })
}

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
    body.extend_from_slice(&0u16.to_le_bytes());

    let collection_size = checked_add_u64(16, body.len() as u64, "global heap size")?;
    let mut bytes = Vec::with_capacity(checked_usize(collection_size, "global heap size")?);
    bytes.extend_from_slice(b"GCOL");
    bytes.push(1);
    bytes.extend_from_slice(&[0, 0, 0]);
    bytes.extend_from_slice(&collection_size.to_le_bytes());
    bytes.extend_from_slice(&body);
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
    let mut messages = encode_attribute_header_messages(attributes, heap_address)?;
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
    if let Some(fill_value) = &dataset.dataset.fill_value {
        messages.push(HeaderMessage {
            type_id: MSG_FILL_VALUE,
            payload: encode_fill_value_message(fill_value)?,
        });
    }
    if !dataset.dataset.filters.is_empty() {
        messages.push(HeaderMessage {
            type_id: MSG_FILTER_PIPELINE,
            payload: encode_filter_pipeline_message(&dataset.dataset.filters)?,
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
            if dataset.dataset.filters.is_empty() {
                encode_implicit_chunked_layout_message(storage_address, chunk_shape)
            } else if dataset.chunk_index.is_some() {
                encode_fixed_array_chunked_layout_message(storage_address, chunk_shape)
            } else {
                encode_single_chunk_layout_message(storage_address, chunk_shape, dataset.data_size)
            }
        }
        PlannedLayout::Compact => encode_compact_layout_message(&dataset.raw_data),
    }
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
            let (exp_location, exp_size, mantissa_size, exp_bias) = match *size {
                4 => (23u8, 8u8, 23u8, 127u32),
                8 => (52u8, 11u8, 52u8, 1023u32),
                other => {
                    return Err(Error::UnsupportedFeature(format!(
                        "unsupported floating-point byte width {other}"
                    )))
                }
            };
            let flags = 0x20 | byte_order_flag(*byte_order);
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
        } => {
            let size = match size {
                StringSize::Fixed(size) => *size,
                StringSize::Variable => 0,
            };
            let flags = string_padding_bits(*padding) | (string_encoding_bits(*encoding) << 4);
            let mut bytes = Vec::new();
            bytes.extend_from_slice(&class_word(3, 1, flags).to_le_bytes());
            bytes.extend_from_slice(&size.to_le_bytes());
            Ok(bytes)
        }
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
        other => Err(Error::UnsupportedFeature(format!(
            "datatype emission is not implemented for {other:?}"
        ))),
    }
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

fn encode_fill_value_message(fill_value: &[u8]) -> Result<Vec<u8>> {
    let size = u32::try_from(fill_value.len()).map_err(|_| {
        Error::UnsupportedFeature("HDF5 fill value exceeds u32 byte length capacity".into())
    })?;
    let mut bytes = Vec::with_capacity(6 + fill_value.len());
    bytes.push(3);
    bytes.push(0x10);
    bytes.extend_from_slice(&size.to_le_bytes());
    bytes.extend_from_slice(fill_value);
    Ok(bytes)
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

fn encode_single_chunk_layout_message(
    address: u64,
    chunk_shape: &[u64],
    filtered_size: u64,
) -> Result<Vec<u8>> {
    if chunk_shape.len() > u8::MAX as usize {
        return Err(Error::InvalidDefinition(
            "chunked dataset rank exceeds HDF5 rank field capacity".into(),
        ));
    }
    let mut bytes = Vec::with_capacity(18 + chunk_shape.len() * 4 + usize::from(OFFSET_SIZE));
    bytes.push(4);
    bytes.push(2);
    bytes.push(1);
    bytes.push(chunk_shape.len() as u8);
    bytes.push(4);
    for &dim in chunk_shape {
        let dim = u32::try_from(dim).map_err(|_| {
            Error::InvalidDefinition("chunk dimension exceeds HDF5 v4 layout capacity".into())
        })?;
        bytes.extend_from_slice(&dim.to_le_bytes());
    }
    bytes.push(1);
    bytes.extend_from_slice(&filtered_size.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&address.to_le_bytes());
    Ok(bytes)
}

fn encode_implicit_chunked_layout_message(address: u64, chunk_shape: &[u64]) -> Result<Vec<u8>> {
    if chunk_shape.len() > u8::MAX as usize {
        return Err(Error::InvalidDefinition(
            "chunked dataset rank exceeds HDF5 rank field capacity".into(),
        ));
    }
    let mut bytes = Vec::with_capacity(6 + chunk_shape.len() * 4 + usize::from(OFFSET_SIZE));
    bytes.push(4);
    bytes.push(2);
    bytes.push(0);
    bytes.push(chunk_shape.len() as u8);
    bytes.push(4);
    for &dim in chunk_shape {
        let dim = u32::try_from(dim).map_err(|_| {
            Error::InvalidDefinition("chunk dimension exceeds HDF5 v4 layout capacity".into())
        })?;
        bytes.extend_from_slice(&dim.to_le_bytes());
    }
    bytes.push(2);
    bytes.extend_from_slice(&address.to_le_bytes());
    Ok(bytes)
}

fn encode_fixed_array_chunked_layout_message(address: u64, chunk_shape: &[u64]) -> Result<Vec<u8>> {
    if chunk_shape.len() > u8::MAX as usize {
        return Err(Error::InvalidDefinition(
            "chunked dataset rank exceeds HDF5 rank field capacity".into(),
        ));
    }
    let mut bytes = Vec::with_capacity(7 + chunk_shape.len() * 4 + usize::from(OFFSET_SIZE));
    bytes.push(4);
    bytes.push(2);
    bytes.push(1);
    bytes.push(chunk_shape.len() as u8);
    bytes.push(4);
    for &dim in chunk_shape {
        let dim = u32::try_from(dim).map_err(|_| {
            Error::InvalidDefinition("chunk dimension exceeds HDF5 v4 layout capacity".into())
        })?;
        bytes.extend_from_slice(&dim.to_le_bytes());
    }
    bytes.push(3);
    bytes.push(0);
    bytes.extend_from_slice(&address.to_le_bytes());
    Ok(bytes)
}

fn encode_fixed_array_chunk_index_header(
    num_entries: u64,
    data_block_address: u64,
) -> Result<Vec<u8>> {
    let entry_size = u8::try_from(usize::from(OFFSET_SIZE) + usize::from(LENGTH_SIZE) + 4)
        .map_err(|_| Error::UnsupportedFeature("fixed array entry size exceeds u8".into()))?;
    let mut bytes = Vec::with_capacity(4 + 1 + 1 + 1 + 1 + 8 + 8 + 4);
    bytes.extend_from_slice(b"FAHD");
    bytes.push(0);
    bytes.push(1);
    bytes.push(entry_size);
    bytes.push(0);
    bytes.extend_from_slice(&num_entries.to_le_bytes());
    bytes.extend_from_slice(&data_block_address.to_le_bytes());
    let checksum = jenkins_lookup3(&bytes);
    bytes.extend_from_slice(&checksum.to_le_bytes());
    Ok(bytes)
}

fn encode_fixed_array_chunk_index_data_block(
    header_address: u64,
    entries: &[FixedArrayChunkEntry],
) -> Result<Vec<u8>> {
    let entry_size = usize::from(OFFSET_SIZE) + usize::from(LENGTH_SIZE) + 4;
    let entries_len = checked_mul_usize(entries.len(), entry_size, "fixed array entries size")?;
    let capacity = checked_add_usize(
        4 + 1 + 1 + usize::from(OFFSET_SIZE),
        checked_add_usize(entries_len, 4, "fixed array data block size")?,
        "fixed array data block size",
    )?;
    let mut bytes = Vec::with_capacity(capacity);
    bytes.extend_from_slice(b"FADB");
    bytes.push(0);
    bytes.push(1);
    bytes.extend_from_slice(&header_address.to_le_bytes());
    for entry in entries {
        bytes.extend_from_slice(&entry.address.to_le_bytes());
        bytes.extend_from_slice(&entry.size.to_le_bytes());
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

fn validate_writer_filters(dataset: &DatasetBuilder) -> Result<()> {
    if dataset.filters.is_empty() {
        return Ok(());
    }
    let PlannedLayout::Chunked { chunk_shape } = &dataset.layout else {
        return Err(Error::InvalidDefinition(
            "filtered HDF5 datasets must use chunked layout".into(),
        ));
    };
    let _ = chunk_count(&dataset.shape, chunk_shape)?;
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
            }
            other => {
                return Err(Error::UnsupportedFeature(format!(
                    "HDF5 filter id {other} is not implemented for writing"
                )));
            }
        }
    }
    Ok(())
}

fn apply_write_filters(data: &[u8], filters: &[FilterDescription]) -> Result<Vec<u8>> {
    let mut current = data.to_vec();
    for filter in filters {
        current = match filter.id {
            FILTER_DEFLATE => deflate_filter(&current, filter)?,
            other => {
                return Err(Error::UnsupportedFeature(format!(
                    "HDF5 filter id {other} is not implemented for writing"
                )));
            }
        };
    }
    Ok(current)
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

        let encoded = apply_write_filters(&chunk, filters)?;
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
        });

        increment_row_major_index(&mut chunk_indices, &chunks_per_dim);
    }

    Ok((storage, Some(PreparedChunkIndex { chunks: entries })))
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
    match width {
        1 => dst.push(value as u8),
        2 => dst.extend_from_slice(&(value as u16).to_le_bytes()),
        4 => dst.extend_from_slice(&(value as u32).to_le_bytes()),
        8 => dst.extend_from_slice(&value.to_le_bytes()),
        _ => unreachable!("validated HDF5 variable integer width"),
    }
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
