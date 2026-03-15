//! HDF5 Data Layout message (type 0x0008).
//!
//! Describes how raw data for a dataset is stored: compact (inline in the
//! object header), contiguous (a single block in the file), or chunked
//! (split into fixed-size chunks, indexed by a B-tree).

use crate::error::{Error, Result};
use crate::io::Cursor;

/// Chunk indexing method (version 4 only).
#[derive(Debug, Clone)]
pub enum ChunkIndexing {
    /// Single chunk — the entire dataset is one chunk.
    SingleChunk { filtered_size: u64, filters: u32 },
    /// Implicit indexing — chunk addresses are computed, not stored.
    Implicit,
    /// Fixed array indexing.
    FixedArray { page_bits: u8 },
    /// Extensible array indexing.
    ExtensibleArray {
        max_bits: u8,
        index_bits: u8,
        min_pointers: u8,
        min_elements: u8,
    },
    /// Version 2 B-tree indexing.
    BTreeV2,
}

/// The storage layout for a dataset's raw data.
#[derive(Debug, Clone)]
pub enum DataLayout {
    /// Data is stored inline in the object header.
    Compact { data: Vec<u8> },
    /// Data is stored in a single contiguous block in the file.
    Contiguous { address: u64, size: u64 },
    /// Data is split into fixed-size chunks.
    Chunked {
        /// Address of the chunk index (B-tree root or similar).
        address: u64,
        /// Chunk dimensions.
        dims: Vec<u32>,
        /// Element size (encoded in the last "dimension" for v1-v3).
        element_size: u32,
        /// Chunk indexing type (v4 only).
        chunk_indexing: Option<ChunkIndexing>,
    },
}

/// Parsed data layout message.
#[derive(Debug, Clone)]
pub struct DataLayoutMessage {
    pub layout: DataLayout,
}

/// Parse a data layout message.
pub fn parse(
    cursor: &mut Cursor<'_>,
    offset_size: u8,
    length_size: u8,
    msg_size: usize,
) -> Result<DataLayoutMessage> {
    let start = cursor.position();
    let version = cursor.read_u8()?;

    let layout = match version {
        1 | 2 => parse_v1_v2(cursor, offset_size, length_size, version)?,
        3 => parse_v3(cursor, offset_size, length_size)?,
        4 => parse_v4(cursor, offset_size)?,
        v => return Err(Error::UnsupportedLayoutVersion(v)),
    };

    let consumed = (cursor.position() - start) as usize;
    if consumed < msg_size {
        cursor.skip(msg_size - consumed)?;
    }

    Ok(DataLayoutMessage { layout })
}

// ---------------------------------------------------------------------------
// Version 1 / 2
// ---------------------------------------------------------------------------

fn parse_v1_v2(
    cursor: &mut Cursor<'_>,
    offset_size: u8,
    _length_size: u8,
    version: u8,
) -> Result<DataLayout> {
    let dimensionality = cursor.read_u8()?;
    let layout_class = cursor.read_u8()?;
    let _reserved = cursor.read_bytes(if version == 1 { 5 } else { 3 })?;

    // For v1 there is an optional compact data size field.
    // data_address is only meaningful for contiguous and chunked.
    let data_address = if layout_class != 0 {
        cursor.read_offset(offset_size)?
    } else {
        // For compact, there is no address; skip the offset-sized field.
        cursor.read_offset(offset_size)?
    };

    // Read dimension sizes. Each is 4 bytes. The number of dimensions:
    // For contiguous: dimensionality values (unused data size).
    // For chunked: (dimensionality-1) chunk dims + 1 element size.
    let mut dim_values = Vec::with_capacity(dimensionality as usize);
    for _ in 0..dimensionality {
        dim_values.push(cursor.read_u32_le()?);
    }

    match layout_class {
        0 => {
            // Compact
            let compact_size = cursor.read_u32_le()? as usize;
            let data = cursor.read_bytes(compact_size)?.to_vec();
            Ok(DataLayout::Compact { data })
        }
        1 => {
            // Contiguous
            // Size is not explicitly stored in v1/v2 for contiguous. The dims
            // encode the logical size but the actual file extent comes from the
            // dataspace * element size. We store the product as size.
            let size = if dim_values.is_empty() {
                0
            } else {
                dim_values.iter().map(|d| *d as u64).product()
            };
            Ok(DataLayout::Contiguous {
                address: data_address,
                size,
            })
        }
        2 => {
            // Chunked — last dimension is the element size
            let (element_size, chunk_dims) = if dim_values.is_empty() {
                (0u32, vec![])
            } else {
                let es = *dim_values.last().unwrap();
                let cd: Vec<u32> = dim_values[..dim_values.len() - 1].to_vec();
                (es, cd)
            };
            Ok(DataLayout::Chunked {
                address: data_address,
                dims: chunk_dims,
                element_size,
                chunk_indexing: None,
            })
        }
        c => Err(Error::UnsupportedLayoutClass(c)),
    }
}

// ---------------------------------------------------------------------------
// Version 3
// ---------------------------------------------------------------------------

fn parse_v3(cursor: &mut Cursor<'_>, offset_size: u8, length_size: u8) -> Result<DataLayout> {
    let layout_class = cursor.read_u8()?;

    match layout_class {
        0 => {
            // Compact
            let size = cursor.read_u16_le()? as usize;
            let data = cursor.read_bytes(size)?.to_vec();
            Ok(DataLayout::Compact { data })
        }
        1 => {
            // Contiguous
            let address = cursor.read_offset(offset_size)?;
            let size = cursor.read_length(length_size)?;
            Ok(DataLayout::Contiguous { address, size })
        }
        2 => {
            // Chunked
            let dimensionality = cursor.read_u8()?;
            let address = cursor.read_offset(offset_size)?;

            // (dimensionality - 1) chunk dims + 1 element size (each 4 bytes)
            let n = dimensionality as usize;
            let mut raw_dims = Vec::with_capacity(n);
            for _ in 0..n {
                raw_dims.push(cursor.read_u32_le()?);
            }

            let (element_size, chunk_dims) = if raw_dims.is_empty() {
                (0, vec![])
            } else {
                let es = *raw_dims.last().unwrap();
                let cd = raw_dims[..raw_dims.len() - 1].to_vec();
                (es, cd)
            };

            Ok(DataLayout::Chunked {
                address,
                dims: chunk_dims,
                element_size,
                chunk_indexing: None,
            })
        }
        c => Err(Error::UnsupportedLayoutClass(c)),
    }
}

// ---------------------------------------------------------------------------
// Version 4
// ---------------------------------------------------------------------------

fn parse_v4(cursor: &mut Cursor<'_>, offset_size: u8) -> Result<DataLayout> {
    let layout_class = cursor.read_u8()?;

    match layout_class {
        0 => {
            // Compact
            let size = cursor.read_u16_le()? as usize;
            let data = cursor.read_bytes(size)?.to_vec();
            Ok(DataLayout::Compact { data })
        }
        1 => {
            // Contiguous
            let address = cursor.read_offset(offset_size)?;
            let size = cursor.read_u64_le()?;
            Ok(DataLayout::Contiguous { address, size })
        }
        2 => {
            // Chunked (v4 specific)
            let flags = cursor.read_u8()?;
            let dimensionality = cursor.read_u8()? as usize;

            // Encoded dimension size (number of bytes per dim element minus 1)
            let dim_size_enc = cursor.read_u8()?;
            let dim_bytes = (dim_size_enc + 1) as usize;

            // Read chunk dimensions
            let mut dims = Vec::with_capacity(dimensionality);
            for _ in 0..dimensionality {
                dims.push(cursor.read_uvar(dim_bytes)? as u32);
            }

            // Encoded chunk size (type-size encoded in variable-width)
            let element_size = cursor.read_uvar(dim_bytes)? as u32;

            // Chunk indexing type
            let index_type = cursor.read_u8()?;

            let chunk_indexing = match index_type {
                0 => {
                    // Single chunk
                    let _idx_flags = if (flags & 0x01) != 0 {
                        // Filters present — read filtered size + filter mask
                        let filtered_size = cursor.read_u64_le()?;
                        let filter_mask = cursor.read_u32_le()?;
                        Some((filtered_size, filter_mask))
                    } else {
                        None
                    };
                    let (fs, fm) = _idx_flags.unwrap_or((0, 0));
                    ChunkIndexing::SingleChunk {
                        filtered_size: fs,
                        filters: fm,
                    }
                }
                1 => ChunkIndexing::Implicit,
                2 => {
                    // Fixed array
                    let page_bits = cursor.read_u8()?;
                    ChunkIndexing::FixedArray { page_bits }
                }
                3 => {
                    // Extensible array
                    let max_bits = cursor.read_u8()?;
                    let index_bits = cursor.read_u8()?;
                    let min_pointers = cursor.read_u8()?;
                    let min_elements = cursor.read_u8()?;
                    ChunkIndexing::ExtensibleArray {
                        max_bits,
                        index_bits,
                        min_pointers,
                        min_elements,
                    }
                }
                4 => ChunkIndexing::BTreeV2,
                t => return Err(Error::UnsupportedChunkIndexType(t)),
            };

            // Address of the chunk index
            let address = cursor.read_offset(offset_size)?;

            Ok(DataLayout::Chunked {
                address,
                dims,
                element_size,
                chunk_indexing: Some(chunk_indexing),
            })
        }
        c => Err(Error::UnsupportedLayoutClass(c)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_v3_contiguous() {
        let mut data = vec![
            0x03, // version 3
            0x01, // layout class = contiguous
        ];
        // address (8 bytes)
        data.extend_from_slice(&0x1000u64.to_le_bytes());
        // size (8 bytes)
        data.extend_from_slice(&4096u64.to_le_bytes());

        let mut cursor = Cursor::new(&data);
        let msg = parse(&mut cursor, 8, 8, data.len()).unwrap();
        match &msg.layout {
            DataLayout::Contiguous { address, size } => {
                assert_eq!(*address, 0x1000);
                assert_eq!(*size, 4096);
            }
            other => panic!("expected Contiguous, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_v3_compact() {
        let mut data = vec![
            0x03, // version 3
            0x00, // layout class = compact
        ];
        // compact size = 4
        data.extend_from_slice(&4u16.to_le_bytes());
        // inline data
        data.extend_from_slice(&[0x01, 0x02, 0x03, 0x04]);

        let mut cursor = Cursor::new(&data);
        let msg = parse(&mut cursor, 8, 8, data.len()).unwrap();
        match &msg.layout {
            DataLayout::Compact { data } => {
                assert_eq!(data, &[0x01, 0x02, 0x03, 0x04]);
            }
            other => panic!("expected Compact, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_v3_chunked() {
        let mut data = vec![
            0x03, // version 3
            0x02, // layout class = chunked
            0x03, // dimensionality = 3 (2 chunk dims + 1 element size)
        ];
        // address
        data.extend_from_slice(&0x2000u64.to_le_bytes());
        // dim[0] = 256
        data.extend_from_slice(&256u32.to_le_bytes());
        // dim[1] = 128
        data.extend_from_slice(&128u32.to_le_bytes());
        // element size = 4
        data.extend_from_slice(&4u32.to_le_bytes());

        let mut cursor = Cursor::new(&data);
        let msg = parse(&mut cursor, 8, 8, data.len()).unwrap();
        match &msg.layout {
            DataLayout::Chunked {
                address,
                dims,
                element_size,
                chunk_indexing,
            } => {
                assert_eq!(*address, 0x2000);
                assert_eq!(dims, &[256, 128]);
                assert_eq!(*element_size, 4);
                assert!(chunk_indexing.is_none());
            }
            other => panic!("expected Chunked, got {:?}", other),
        }
    }
}
