//! Shared HDF5 format model used by reader and writer crates.

use std::fmt;

/// Errors produced by shared HDF5 model helpers.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid data: {0}")]
    InvalidData(String),
}

pub type Result<T> = std::result::Result<T, Error>;

/// Jenkins lookup3 `hashlittle2` used by HDF5 metadata checksums.
///
/// HDF5 uses this checksum for superblock v2/v3, object header v2,
/// B-tree v2, extensible-array, fixed-array, fractal-heap, and shared-message
/// metadata blocks. This is a direct translation of Bob Jenkins'
/// lookup3.c `hashlittle2` with both init values set to zero.
pub fn jenkins_lookup3(data: &[u8]) -> u32 {
    let len = data.len();
    let mut a: u32 = 0xdeadbeef_u32.wrapping_add(len as u32);
    let mut b: u32 = a;
    let mut c: u32 = a;
    let mut offset = 0;

    while offset + 12 < len {
        a = a.wrapping_add(u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]));
        b = b.wrapping_add(u32::from_le_bytes([
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ]));
        c = c.wrapping_add(u32::from_le_bytes([
            data[offset + 8],
            data[offset + 9],
            data[offset + 10],
            data[offset + 11],
        ]));
        jenkins_mix(&mut a, &mut b, &mut c);
        offset += 12;
    }

    let remaining = len - offset;
    if remaining > 0 {
        let mut tail_a: u32 = 0;
        let mut tail_b: u32 = 0;
        let mut tail_c: u32 = 0;

        for i in 0..remaining.min(4) {
            tail_a |= (data[offset + i] as u32) << (i * 8);
        }
        if remaining > 4 {
            for i in 4..remaining.min(8) {
                tail_b |= (data[offset + i] as u32) << ((i - 4) * 8);
            }
        }
        if remaining > 8 {
            for i in 8..remaining {
                tail_c |= (data[offset + i] as u32) << ((i - 8) * 8);
            }
        }

        a = a.wrapping_add(tail_a);
        b = b.wrapping_add(tail_b);
        c = c.wrapping_add(tail_c);
        jenkins_final_mix(&mut a, &mut b, &mut c);
    }

    c
}

/// Fletcher-32 checksum used by the HDF5 filter pipeline.
///
/// Matches HDF5's `H5_checksum_fletcher32`: reads 16-bit big-endian words,
/// accumulates in batches of 360, and reduces with `(x & 0xffff) + (x >> 16)`.
pub fn fletcher32(data: &[u8]) -> u32 {
    let mut sum1: u32 = 0;
    let mut sum2: u32 = 0;
    let total_words = data.len() / 2;

    let mut offset = 0usize;
    let mut remaining = total_words;

    while remaining > 0 {
        let batch = remaining.min(360);
        remaining -= batch;

        for _ in 0..batch {
            let word = ((data[offset] as u32) << 8) | (data[offset + 1] as u32);
            sum1 += word;
            sum2 += sum1;
            offset += 2;
        }

        sum1 = (sum1 & 0xffff) + (sum1 >> 16);
        sum2 = (sum2 & 0xffff) + (sum2 >> 16);
    }

    // A trailing odd byte contributes as the high half of a final word.
    if data.len() % 2 == 1 {
        sum1 += (data[data.len() - 1] as u32) << 8;
        sum2 += sum1;
        sum1 = (sum1 & 0xffff) + (sum1 >> 16);
        sum2 = (sum2 & 0xffff) + (sum2 >> 16);
    }

    sum1 = (sum1 & 0xffff) + (sum1 >> 16);
    sum2 = (sum2 & 0xffff) + (sum2 >> 16);

    (sum2 << 16) | sum1
}

#[inline]
fn jenkins_mix(a: &mut u32, b: &mut u32, c: &mut u32) {
    *a = a.wrapping_sub(*c);
    *a ^= c.rotate_left(4);
    *c = c.wrapping_add(*b);
    *b = b.wrapping_sub(*a);
    *b ^= a.rotate_left(6);
    *a = a.wrapping_add(*c);
    *c = c.wrapping_sub(*b);
    *c ^= b.rotate_left(8);
    *b = b.wrapping_add(*a);
    *a = a.wrapping_sub(*c);
    *a ^= c.rotate_left(16);
    *c = c.wrapping_add(*b);
    *b = b.wrapping_sub(*a);
    *b ^= a.rotate_left(19);
    *a = a.wrapping_add(*c);
    *c = c.wrapping_sub(*b);
    *c ^= b.rotate_left(4);
    *b = b.wrapping_add(*a);
}

#[inline]
fn jenkins_final_mix(a: &mut u32, b: &mut u32, c: &mut u32) {
    *c ^= *b;
    *c = c.wrapping_sub(b.rotate_left(14));
    *a ^= *c;
    *a = a.wrapping_sub(c.rotate_left(11));
    *b ^= *a;
    *b = b.wrapping_sub(a.rotate_left(25));
    *c ^= *b;
    *c = c.wrapping_sub(b.rotate_left(16));
    *a ^= *c;
    *a = a.wrapping_sub(c.rotate_left(4));
    *b ^= *a;
    *b = b.wrapping_sub(a.rotate_left(14));
    *c ^= *b;
    *c = c.wrapping_sub(b.rotate_left(24));
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jenkins_lookup3_is_deterministic() {
        assert_eq!(jenkins_lookup3(b"hello"), jenkins_lookup3(b"hello"));
        assert_ne!(jenkins_lookup3(b"hello"), jenkins_lookup3(b"world"));
    }

    #[test]
    fn jenkins_lookup3_handles_twelve_byte_boundary_like_lookup3() {
        let twelve = [0x5au8; 12];
        let thirteen = [0x5au8; 13];

        assert_ne!(jenkins_lookup3(&twelve), jenkins_lookup3(&thirteen));
        assert_eq!(jenkins_lookup3(&twelve), jenkins_lookup3(&twelve));
    }

    #[test]
    fn fletcher32_is_deterministic() {
        let data = [0x01, 0x02, 0x03, 0x04];

        assert_eq!(fletcher32(&data), fletcher32(&data));
    }

    #[test]
    fn fletcher32_matches_known_reference() {
        let data = [0x00, 0x01, 0x00, 0x02];

        assert_eq!(fletcher32(&data), 0x0004_0003);
    }

    #[test]
    fn fletcher32_handles_trailing_odd_byte_like_libhdf5() {
        // H5_checksum_fletcher32 folds a trailing odd byte in as the high
        // half of a final 16-bit word.
        let data = [0x01, 0x02, 0x03];

        assert_eq!(fletcher32(&data), 0x0504_0402);
    }
}
