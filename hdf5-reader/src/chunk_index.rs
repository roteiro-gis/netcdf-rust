//! Chunk indexing — resolves chunk locations from various storage strategies.
//!
//! Supports:
//! - V1 B-tree chunk indexing (btree_v1 type 1) — dispatched externally
//! - V2 B-tree chunk indexing (btree_v2 types 10 and 11)
//! - Single chunk indexing
//! - Implicit chunk indexing
//!
//! Fixed array and extensible array return errors (not yet implemented).

use crate::error::Result;

/// A resolved chunk location within the file.
#[derive(Debug, Clone)]
pub struct ChunkEntry {
    /// Absolute file address of the chunk data.
    pub address: u64,
    /// Size of the chunk data in bytes (after filtering, i.e., on-disk size).
    pub size: u64,
    /// Filter mask — each bit indicates whether the corresponding filter
    /// in the pipeline was skipped (1 = skipped).
    pub filter_mask: u32,
    /// Chunk offsets within the dataset (one per dimension).
    pub offsets: Vec<u64>,
}

/// Collect chunk entries from a B-tree v2 chunk index.
pub fn collect_v2_chunk_entries(
    data: &[u8],
    btree_address: u64,
    offset_size: u8,
    length_size: u8,
    ndim: u32,
) -> Result<Vec<ChunkEntry>> {
    let mut cursor = crate::io::Cursor::new(data);
    cursor.set_position(btree_address);
    let header =
        crate::btree_v2::BTreeV2Header::parse(&mut cursor, offset_size, length_size)?;

    let records = crate::btree_v2::collect_btree_v2_records(
        data,
        &header,
        offset_size,
        length_size,
        Some(ndim),
    )?;

    let mut entries = Vec::with_capacity(records.len());
    for record in records {
        match record {
            crate::btree_v2::BTreeV2Record::ChunkedNonFiltered { address, offsets } => {
                entries.push(ChunkEntry {
                    address,
                    size: 0, // caller must compute from chunk dims * elem_size
                    filter_mask: 0,
                    offsets,
                });
            }
            crate::btree_v2::BTreeV2Record::ChunkedFiltered {
                address,
                chunk_size,
                filter_mask,
                offsets,
            } => {
                entries.push(ChunkEntry {
                    address,
                    size: chunk_size,
                    filter_mask,
                    offsets,
                });
            }
            _ => {
                // Skip non-chunk records
            }
        }
    }

    Ok(entries)
}

/// Collect chunk entries for implicit indexing.
///
/// Implicit chunks are laid out sequentially starting at the given address.
/// Each chunk has the same size = product(chunk_dims) * elem_size.
pub fn collect_implicit_chunk_entries(
    start_address: u64,
    dataset_shape: &[u64],
    chunk_dims: &[u32],
    elem_size: usize,
) -> Vec<ChunkEntry> {
    let chunk_bytes: u64 = chunk_dims.iter().map(|&d| d as u64).product::<u64>() * elem_size as u64;
    let ndim = dataset_shape.len();

    // Compute how many chunks along each dimension
    let chunks_per_dim: Vec<u64> = (0..ndim)
        .map(|i| dataset_shape[i].div_ceil(chunk_dims[i] as u64))
        .collect();

    let total_chunks: u64 = chunks_per_dim.iter().product();
    let mut entries = Vec::with_capacity(total_chunks as usize);

    for chunk_idx in 0..total_chunks {
        // Convert linear chunk index to multi-dimensional offsets
        let mut remaining = chunk_idx;
        let mut offsets = vec![0u64; ndim];
        for d in (0..ndim).rev() {
            offsets[d] = (remaining % chunks_per_dim[d]) * chunk_dims[d] as u64;
            remaining /= chunks_per_dim[d];
        }

        entries.push(ChunkEntry {
            address: start_address + chunk_idx * chunk_bytes,
            size: chunk_bytes,
            filter_mask: 0,
            offsets,
        });
    }

    entries
}

/// Resolve a single-chunk layout.
///
/// The entire dataset is stored as one chunk at the given address.
pub fn single_chunk_entry(
    address: u64,
    filtered_size: u64,
    filter_mask: u32,
    ndim: usize,
) -> ChunkEntry {
    ChunkEntry {
        address,
        size: filtered_size,
        filter_mask,
        offsets: vec![0u64; ndim],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_entry_debug_clone() {
        let entry = ChunkEntry {
            address: 0x1000,
            size: 4096,
            filter_mask: 0,
            offsets: vec![0, 0],
        };
        let entry2 = entry.clone();
        assert_eq!(entry2.address, 0x1000);
        let _ = format!("{:?}", entry);
    }

    #[test]
    fn test_implicit_chunk_entries() {
        let entries = collect_implicit_chunk_entries(
            1000,
            &[10, 20],
            &[5, 10],
            4,
        );
        // 2 chunks along dim 0, 2 chunks along dim 1 = 4 total
        assert_eq!(entries.len(), 4);
        assert_eq!(entries[0].address, 1000);
        assert_eq!(entries[0].offsets, vec![0, 0]);
        assert_eq!(entries[1].address, 1000 + 200); // 5*10*4 = 200
        assert_eq!(entries[1].offsets, vec![0, 10]);
        assert_eq!(entries[2].offsets, vec![5, 0]);
        assert_eq!(entries[3].offsets, vec![5, 10]);
    }

    #[test]
    fn test_single_chunk_entry() {
        let entry = single_chunk_entry(0x2000, 8192, 0, 3);
        assert_eq!(entry.address, 0x2000);
        assert_eq!(entry.size, 8192);
        assert_eq!(entry.offsets, vec![0, 0, 0]);
    }
}
