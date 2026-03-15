//! Chunk indexing — resolves chunk locations from various storage strategies.
//!
//! This is a Phase 3 stub. The full implementation will handle:
//! - V1 B-tree chunk indexing (btree_v1 type 1)
//! - V2 B-tree chunk indexing (btree_v2 types 10 and 11)
//! - Fixed array indexing
//! - Extensible array indexing
//! - Single chunk indexing
//!
//! For now, all calls return an error indicating that chunk indexing is not
//! yet implemented.

use crate::error::{Error, Result};

/// A resolved chunk location within the file.
#[derive(Debug, Clone)]
pub struct ChunkLocation {
    /// Absolute file address of the chunk data.
    pub address: u64,
    /// Size of the chunk data in bytes (after filtering, i.e., on-disk size).
    pub size: u64,
    /// Filter mask — each bit indicates whether the corresponding filter
    /// in the pipeline was skipped (1 = skipped).
    pub filter_mask: u32,
}

/// Resolve chunk locations from a dataset's layout information.
///
/// Returns a list of `(chunk_offsets, ChunkLocation)` pairs where
/// `chunk_offsets` is the position of the chunk within the logical dataset
/// (one element per dimension).
///
/// **Phase 3 stub** — currently returns an error.
pub fn resolve_chunk_locations(
    _data: &[u8],
    _offset_size: u8,
    _length_size: u8,
) -> Result<Vec<(Vec<u64>, ChunkLocation)>> {
    Err(Error::Other("chunk indexing not yet implemented".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stub_returns_error() {
        let data = vec![0u8; 100];
        let result = resolve_chunk_locations(&data, 8, 8);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, Error::Other(_)));
    }

    #[test]
    fn test_chunk_location_debug() {
        let loc = ChunkLocation {
            address: 0x1000,
            size: 4096,
            filter_mask: 0,
        };
        // Verify Debug trait is implemented.
        let _ = format!("{:?}", loc);
    }

    #[test]
    fn test_chunk_location_clone() {
        let loc = ChunkLocation {
            address: 0x2000,
            size: 8192,
            filter_mask: 0x01,
        };
        let loc2 = loc.clone();
        assert_eq!(loc2.address, 0x2000);
        assert_eq!(loc2.size, 8192);
        assert_eq!(loc2.filter_mask, 0x01);
    }
}
