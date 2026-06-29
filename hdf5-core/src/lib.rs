//! Shared HDF5 format model used by reader and writer crates.

use std::fmt;

/// Errors produced by shared HDF5 model helpers.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid data: {0}")]
    InvalidData(String),
}

pub type Result<T> = std::result::Result<T, Error>;

/// Byte order for numeric data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ByteOrder {
    LittleEndian,
    BigEndian,
}

impl fmt::Display for ByteOrder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ByteOrder::LittleEndian => write!(f, "little-endian"),
            ByteOrder::BigEndian => write!(f, "big-endian"),
        }
    }
}

/// How a string's length is determined.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StringSize {
    /// Fixed-length, padded to `n` bytes.
    Fixed(u32),
    /// Variable-length.
    Variable,
}

/// String character encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StringEncoding {
    Ascii,
    Utf8,
}

/// String padding type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StringPadding {
    NullTerminate,
    NullPad,
    SpacePad,
}

/// HDF5 variable-length datatype flavor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VarLenKind {
    /// Variable-length sequence of values.
    Sequence,
    /// Variable-length string.
    String,
    /// Unknown HDF5 vlen kind; retained for metadata fidelity.
    Unknown(u8),
}

/// A field within a compound datatype.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompoundField {
    pub name: String,
    pub byte_offset: u32,
    pub datatype: Datatype,
}

/// A member of an enumeration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnumMember {
    pub name: String,
    pub value: Vec<u8>,
}

/// HDF5 reference type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceType {
    /// Object reference.
    Object,
    /// Dataset region reference.
    DatasetRegion,
}

/// Describes the element type of a dataset or attribute.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Datatype {
    FixedPoint {
        size: u8,
        signed: bool,
        byte_order: ByteOrder,
    },
    FloatingPoint {
        size: u8,
        byte_order: ByteOrder,
    },
    String {
        size: StringSize,
        encoding: StringEncoding,
        padding: StringPadding,
    },
    Compound {
        size: u32,
        fields: Vec<CompoundField>,
    },
    Array {
        base: Box<Datatype>,
        dims: Vec<u64>,
    },
    Enum {
        base: Box<Datatype>,
        members: Vec<EnumMember>,
    },
    VarLen {
        base: Box<Datatype>,
        kind: VarLenKind,
        encoding: StringEncoding,
        padding: StringPadding,
    },
    Opaque {
        size: u32,
        tag: String,
    },
    Reference {
        ref_type: ReferenceType,
        size: u8,
    },
    Bitfield {
        size: u8,
        byte_order: ByteOrder,
    },
}

/// Datatype message payload plus element size from the message header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DatatypeMessage {
    pub datatype: Datatype,
    pub size: u32,
}

/// Unlimited dimension sentinel value.
pub const UNLIMITED: u64 = u64::MAX;

/// HDF5 dataspace kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataspaceType {
    Null,
    Scalar,
    Simple,
}

/// Parsed dataspace message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataspaceMessage {
    pub rank: u8,
    pub dims: Vec<u64>,
    pub max_dims: Option<Vec<u64>>,
    pub dataspace_type: DataspaceType,
}

impl DataspaceMessage {
    /// Total number of elements in the dataspace.
    pub fn num_elements(&self) -> Result<u64> {
        if self.dims.is_empty() {
            return Ok(match self.dataspace_type {
                DataspaceType::Scalar => 1,
                _ => 0,
            });
        }
        self.dims.iter().try_fold(1u64, |acc, &dim| {
            acc.checked_mul(dim).ok_or_else(|| {
                Error::InvalidData("dataspace element count overflows u64".to_string())
            })
        })
    }
}

/// Chunk indexing method for v4/v5 layout messages.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChunkIndexing {
    SingleChunk {
        filtered_size: u64,
        filters: u32,
    },
    Implicit,
    FixedArray {
        page_bits: u8,
        chunk_size_len: u8,
    },
    ExtensibleArray {
        max_bits: u8,
        index_bits: u8,
        min_pointers: u8,
        min_elements: u8,
        chunk_size_len: u8,
    },
    BTreeV2,
}

/// Raw data storage layout for a dataset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataLayout {
    Compact {
        data: Vec<u8>,
    },
    Contiguous {
        address: u64,
        size: u64,
    },
    Chunked {
        address: u64,
        dims: Vec<u32>,
        element_size: u32,
        chunk_indexing: Option<ChunkIndexing>,
    },
}

/// Parsed data layout message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataLayoutMessage {
    pub layout: DataLayout,
}

/// When to write the fill value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FillTime {
    IfSet,
    Always,
    Never,
}

/// Parsed fill value message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FillValueMessage {
    pub defined: bool,
    pub fill_time: FillTime,
    pub value: Option<Vec<u8>>,
}

/// Standard HDF5 filter IDs.
pub const FILTER_DEFLATE: u16 = 1;
pub const FILTER_SHUFFLE: u16 = 2;
pub const FILTER_FLETCHER32: u16 = 3;
pub const FILTER_SZIP: u16 = 4;
pub const FILTER_NBIT: u16 = 5;
pub const FILTER_SCALEOFFSET: u16 = 6;
pub const FILTER_LZ4: u16 = 32004;

/// A single filter in a filter pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FilterDescription {
    pub id: u16,
    pub name: Option<String>,
    pub client_data: Vec<u32>,
}

/// Parsed filter pipeline message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FilterPipelineMessage {
    pub filters: Vec<FilterDescription>,
}

/// A single external raw data file slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExternalFileSlot {
    pub name_offset: u64,
    pub offset: u64,
    pub size: u64,
}

/// Parsed external raw data files message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExternalFilesMessage {
    pub heap_address: u64,
    pub slots: Vec<ExternalFileSlot>,
}

/// Where an HDF5 link points.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LinkTarget {
    Hard { address: u64 },
    Soft { path: String },
    External { filename: String, path: String },
}

/// Parsed link message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LinkMessage {
    pub name: String,
    pub target: LinkTarget,
    pub creation_order: Option<u64>,
}
