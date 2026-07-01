//! Pure-Rust HDF5 writer crate.
//!
//! This crate owns the write-side API surface and shared encoding decisions.
//! The first implementation milestone exposes validated builders and format
//! planning types. Full HDF5 serialization is intentionally gated until the
//! object-header, heap, chunk-index, and checksum encoders are complete.

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
const MSG_LINK: u8 = 0x06;
const MSG_DATA_LAYOUT: u8 = 0x08;
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
        for dataset in &self.datasets {
            dataset.validate()?;
            if !names.insert(dataset.name.as_str()) {
                return Err(Error::InvalidDefinition(format!(
                    "duplicate root dataset '{}'",
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
    header_address: u64,
    data_address: u64,
    header: Vec<u8>,
}

#[derive(Debug)]
struct HeaderMessage {
    type_id: u8,
    payload: Vec<u8>,
}

fn encode_hdf5_file(plan: &Hdf5WritePlan, options: WriteOptions) -> Result<Vec<u8>> {
    if options.variant != Hdf5Variant::Modern {
        return Err(Error::UnsupportedFeature(format!(
            "unsupported HDF5 variant {:?}",
            options.variant
        )));
    }

    let prepared = prepare_datasets(plan, options)?;
    let root_header_size = encode_root_header(&prepared, &plan.attributes)?.len();
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

    let mut header_addresses = Vec::with_capacity(prepared.len());
    for prepared_dataset in &prepared {
        header_addresses.push(next_address);
        let placeholder = encode_dataset_header(prepared_dataset, 0)?;
        next_address = align_u64(
            checked_add_u64(
                next_address,
                placeholder.len() as u64,
                "dataset object header end",
            )?,
            8,
        );
    }

    let mut data_addresses = Vec::with_capacity(prepared.len());
    for prepared_dataset in &prepared {
        data_addresses.push(next_address);
        next_address = align_u64(
            checked_add_u64(
                next_address,
                prepared_dataset.raw_data.len() as u64,
                "dataset raw data end",
            )?,
            8,
        );
    }

    let links: Vec<_> = prepared
        .iter()
        .zip(header_addresses.iter().copied())
        .map(|(prepared_dataset, address)| (prepared_dataset.dataset.name.as_str(), address))
        .collect();
    let root_header = encode_root_header_from_links(&links, &plan.attributes)?;

    let mut emissions = Vec::with_capacity(prepared.len());
    for ((prepared_dataset, header_address), data_address) in prepared
        .iter()
        .zip(header_addresses.iter().copied())
        .zip(data_addresses.iter().copied())
    {
        let header = encode_dataset_header(prepared_dataset, data_address)?;
        emissions.push(DatasetEmission {
            dataset: prepared_dataset.dataset,
            header_address,
            data_address,
            header,
        });
    }

    let eof_address = next_address;
    let mut file = Vec::with_capacity(checked_usize(eof_address, "HDF5 file size")?);
    file.extend_from_slice(&encode_superblock_v2(root_address, eof_address));
    pad_to_address(&mut file, root_address)?;
    file.extend_from_slice(&root_header);

    for emission in &emissions {
        pad_to_address(&mut file, emission.header_address)?;
        file.extend_from_slice(&emission.header);
    }

    for emission in &emissions {
        pad_to_address(&mut file, emission.data_address)?;
        file.extend_from_slice(
            emission
                .dataset
                .raw_data
                .as_deref()
                .expect("prepared datasets always have raw data"),
        );
    }

    pad_to_address(&mut file, eof_address)?;
    Ok(file)
}

#[derive(Debug)]
struct PreparedDataset<'a> {
    dataset: &'a DatasetBuilder,
    raw_data: &'a [u8],
    data_size: u64,
}

fn prepare_datasets(
    plan: &Hdf5WritePlan,
    options: WriteOptions,
) -> Result<Vec<PreparedDataset<'_>>> {
    plan.validate()?;
    let mut prepared = Vec::with_capacity(plan.datasets.len());
    for dataset in &plan.datasets {
        if dataset.name.contains('/') {
            return Err(Error::UnsupportedFeature(format!(
                "nested HDF5 paths are not emitted yet: '{}'",
                dataset.name
            )));
        }
        if dataset.max_shape.is_some() {
            return Err(Error::UnsupportedFeature(format!(
                "resizable HDF5 dataspace is not emitted yet: '{}'",
                dataset.name
            )));
        }
        if !matches!(dataset.layout, PlannedLayout::Contiguous) {
            return Err(Error::UnsupportedFeature(format!(
                "only contiguous HDF5 datasets are emitted now: '{}'",
                dataset.name
            )));
        }
        if !dataset.filters.is_empty() {
            return Err(Error::UnsupportedFeature(format!(
                "filtered HDF5 datasets are not emitted yet: '{}'",
                dataset.name
            )));
        }
        if dataset.fill_value.is_some() {
            return Err(Error::UnsupportedFeature(format!(
                "HDF5 fill value messages are not emitted yet: '{}'",
                dataset.name
            )));
        }

        let element_size = datatype_element_size(&dataset.datatype)?;
        let datatype_order = numeric_datatype_order(&dataset.datatype)?;
        if datatype_order != options.byte_order {
            return Err(Error::InvalidDefinition(format!(
                "dataset '{}' datatype byte order {:?} does not match writer byte order {:?}",
                dataset.name, datatype_order, options.byte_order
            )));
        }
        let expected = expected_data_len(&dataset.shape, element_size)?;
        let raw_data = dataset.raw_data.as_deref().ok_or_else(|| {
            Error::InvalidDefinition(format!(
                "dataset '{}' has no raw data for binary emission",
                dataset.name
            ))
        })?;
        if raw_data.len() != expected {
            return Err(Error::InvalidDefinition(format!(
                "dataset '{}' expects {expected} data bytes, got {}",
                dataset.name,
                raw_data.len()
            )));
        }

        prepared.push(PreparedDataset {
            dataset,
            raw_data,
            data_size: expected as u64,
        });
    }
    Ok(prepared)
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

fn encode_root_header(
    prepared: &[PreparedDataset<'_>],
    attributes: &[AttributeBuilder],
) -> Result<Vec<u8>> {
    let links: Vec<_> = prepared
        .iter()
        .map(|prepared_dataset| (prepared_dataset.dataset.name.as_str(), 0u64))
        .collect();
    encode_root_header_from_links(&links, attributes)
}

fn encode_root_header_from_links(
    links: &[(&str, u64)],
    attributes: &[AttributeBuilder],
) -> Result<Vec<u8>> {
    let mut messages = encode_attribute_header_messages(attributes)?;
    let link_messages: Result<Vec<_>> = links
        .iter()
        .map(|(name, address)| {
            Ok(HeaderMessage {
                type_id: MSG_LINK,
                payload: encode_hard_link_message(name, *address)?,
            })
        })
        .collect();
    messages.extend(link_messages?);
    encode_object_header_v2(&messages)
}

fn encode_dataset_header(dataset: &PreparedDataset<'_>, data_address: u64) -> Result<Vec<u8>> {
    let mut messages = vec![
        HeaderMessage {
            type_id: MSG_DATASPACE,
            payload: encode_dataspace_message(
                &dataset.dataset.shape,
                dataset.dataset.max_shape.as_deref(),
            )?,
        },
        HeaderMessage {
            type_id: MSG_DATATYPE,
            payload: encode_datatype_message(&dataset.dataset.datatype)?,
        },
        HeaderMessage {
            type_id: MSG_DATA_LAYOUT,
            payload: encode_contiguous_layout_message(data_address, dataset.data_size),
        },
    ];
    messages.extend(encode_attribute_header_messages(
        &dataset.dataset.attributes,
    )?);
    encode_object_header_v2(&messages)
}

fn encode_attribute_header_messages(attributes: &[AttributeBuilder]) -> Result<Vec<HeaderMessage>> {
    attributes
        .iter()
        .map(|attribute| {
            Ok(HeaderMessage {
                type_id: MSG_ATTRIBUTE,
                payload: encode_attribute_message(attribute)?,
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

fn encode_attribute_message(attribute: &AttributeBuilder) -> Result<Vec<u8>> {
    attribute.validate()?;
    let name_bytes = attribute.name.as_bytes();
    let datatype = encode_datatype_message(&attribute.datatype)?;
    let dataspace = encode_dataspace_message(&attribute.shape, None)?;
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
    bytes.extend_from_slice(&attribute.raw_data);
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

fn ensure_datatype_matches_element<T: H5WriteElement>(datatype: &Datatype) -> Result<()> {
    let expected = T::hdf5_type_with_order(numeric_datatype_order(datatype)?);
    if &expected != datatype {
        return Err(Error::InvalidDefinition(format!(
            "typed data does not match dataset datatype: expected {datatype:?}, got {expected:?}"
        )));
    }
    Ok(())
}

fn numeric_datatype_order(datatype: &Datatype) -> Result<ByteOrder> {
    match datatype {
        Datatype::FixedPoint { byte_order, .. } | Datatype::FloatingPoint { byte_order, .. } => {
            Ok(*byte_order)
        }
        other => Err(Error::UnsupportedFeature(format!(
            "typed data encoding is not implemented for {other:?}"
        ))),
    }
}

fn expected_data_len(shape: &[u64], element_size: usize) -> Result<usize> {
    let elements = if shape.is_empty() {
        1usize
    } else {
        shape.iter().try_fold(1usize, |acc, &dim| {
            checked_mul_usize(
                acc,
                checked_usize(dim, "dataset dimension")?,
                "dataset element count",
            )
        })?
    };
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

fn checked_add_u64(lhs: u64, rhs: u64, context: &str) -> Result<u64> {
    lhs.checked_add(rhs)
        .ok_or_else(|| Error::InvalidDefinition(format!("{context} exceeds u64 capacity")))
}
