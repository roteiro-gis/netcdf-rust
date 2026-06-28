//! Pure-Rust HDF5 writer crate.
//!
//! This crate owns the write-side API surface and shared encoding decisions.
//! The first implementation milestone exposes validated builders and format
//! planning types. Full HDF5 serialization is intentionally gated until the
//! object-header, heap, chunk-index, and checksum encoders are complete.

use std::io::{Seek, Write};

pub use hdf5_core::{
    ByteOrder, ChunkIndexing, DataLayout, DataspaceMessage, DataspaceType, Datatype, FillTime,
    FillValueMessage, FilterDescription, FilterPipelineMessage, StringEncoding, StringPadding,
    StringSize, VarLenKind, FILTER_DEFLATE, FILTER_FLETCHER32, FILTER_LZ4, FILTER_NBIT,
    FILTER_SCALEOFFSET, FILTER_SHUFFLE, FILTER_SZIP, UNLIMITED,
};

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

macro_rules! impl_int_type {
    ($ty:ty, $size:expr, $signed:expr) => {
        impl H5WriteType for $ty {
            fn hdf5_type() -> Datatype {
                Datatype::FixedPoint {
                    size: $size,
                    signed: $signed,
                    byte_order: native_order(),
                }
            }
        }
    };
}

impl_int_type!(i8, 1, true);
impl_int_type!(u8, 1, false);
impl_int_type!(i16, 2, true);
impl_int_type!(u16, 2, false);
impl_int_type!(i32, 4, true);
impl_int_type!(u32, 4, false);
impl_int_type!(i64, 8, true);
impl_int_type!(u64, 8, false);

impl H5WriteType for f32 {
    fn hdf5_type() -> Datatype {
        Datatype::FloatingPoint {
            size: 4,
            byte_order: native_order(),
        }
    }
}

impl H5WriteType for f64 {
    fn hdf5_type() -> Datatype {
        Datatype::FloatingPoint {
            size: 8,
            byte_order: native_order(),
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
        }
    }

    pub fn typed<T: H5WriteType>(name: impl Into<String>, shape: impl Into<Vec<u64>>) -> Self {
        Self::new(name, T::hdf5_type(), shape)
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

    pub fn datatype(&self) -> &Datatype {
        &self.datatype
    }

    pub fn shape(&self) -> &[u64] {
        &self.shape
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
        Ok(())
    }
}

/// Root HDF5 file builder.
#[derive(Debug, Clone, Default)]
pub struct Hdf5Builder {
    datasets: Vec<DatasetBuilder>,
}

impl Hdf5Builder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn dataset(mut self, dataset: DatasetBuilder) -> Self {
        self.datasets.push(dataset);
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
        Ok(())
    }

    pub fn into_plan(self) -> Result<Hdf5WritePlan> {
        self.validate()?;
        Ok(Hdf5WritePlan {
            datasets: self.datasets,
        })
    }
}

/// Validated HDF5 write plan.
#[derive(Debug, Clone)]
pub struct Hdf5WritePlan {
    datasets: Vec<DatasetBuilder>,
}

impl Hdf5WritePlan {
    pub fn datasets(&self) -> &[DatasetBuilder] {
        &self.datasets
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
        let _ = self.options;
        let _ = plan;
        self.finalized = true;
        Err(Error::UnsupportedFeature(
            "HDF5 binary emission is not implemented yet".into(),
        ))
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
