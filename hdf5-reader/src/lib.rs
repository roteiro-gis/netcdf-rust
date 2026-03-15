pub mod checksum;
pub mod error;
pub mod io;

// Level 0 — File Metadata
pub mod superblock;

// Level 1 — File Infrastructure
pub mod btree_v1;
pub mod btree_v2;
pub mod chunk_index;
pub mod fractal_heap;
pub mod global_heap;
pub mod local_heap;
pub mod symbol_table;

// Level 2 — Data Objects
pub mod messages;
pub mod object_header;

// High-level API
pub mod attribute_api;
pub mod dataset;
pub mod datatype_api;
pub mod group;

// Filters
pub mod filters;

// Utilities
pub mod cache;

use std::path::Path;
use std::sync::Arc;

use memmap2::Mmap;

use cache::ChunkCache;
use error::{Error, Result};
use group::Group;
use io::Cursor;
use superblock::Superblock;

// Re-exports
pub use attribute_api::Attribute;
pub use dataset::{Dataset, SliceInfo, SliceInfoElem};
pub use datatype_api::{
    dtype_element_size, CompoundField, EnumMember, H5Type, ReferenceType, StringEncoding,
    StringPadding, StringSize,
};
pub use error::ByteOrder;
pub use messages::datatype::Datatype;

/// Configuration options for opening an HDF5 file.
pub struct OpenOptions {
    /// Maximum bytes for the chunk cache. Default: 64 MiB.
    pub chunk_cache_bytes: usize,
    /// Maximum number of chunk cache slots. Default: 521.
    pub chunk_cache_slots: usize,
}

impl Default for OpenOptions {
    fn default() -> Self {
        OpenOptions {
            chunk_cache_bytes: 64 * 1024 * 1024,
            chunk_cache_slots: 521,
        }
    }
}

/// An opened HDF5 file.
///
/// This is the main entry point for reading HDF5 files. The file data is
/// memory-mapped for efficient access.
pub struct Hdf5File {
    /// Memory-mapped file data (or owned bytes for `from_bytes`).
    data: FileData,
    /// Parsed superblock.
    superblock: Superblock,
    /// Shared chunk cache.
    chunk_cache: Arc<ChunkCache>,
}

enum FileData {
    Mmap(Mmap),
    Bytes(Vec<u8>),
}

impl FileData {
    fn as_slice(&self) -> &[u8] {
        match self {
            FileData::Mmap(m) => m,
            FileData::Bytes(b) => b,
        }
    }
}

impl Hdf5File {
    /// Open an HDF5 file with default options.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_with_options(path, OpenOptions::default())
    }

    /// Open an HDF5 file with custom options.
    pub fn open_with_options(path: impl AsRef<Path>, options: OpenOptions) -> Result<Self> {
        let file = std::fs::File::open(path.as_ref())?;
        // SAFETY: We only read from the mapping, and the file isn't modified
        // while we hold the mapping. The caller is responsible for not
        // modifying the file concurrently.
        let mmap = unsafe { Mmap::map(&file)? };

        let mut cursor = Cursor::new(&mmap);
        let superblock = Superblock::parse(&mut cursor)?;

        let cache = Arc::new(ChunkCache::new(
            options.chunk_cache_bytes,
            options.chunk_cache_slots,
        ));

        Ok(Hdf5File {
            data: FileData::Mmap(mmap),
            superblock,
            chunk_cache: cache,
        })
    }

    /// Open an HDF5 file from an in-memory byte slice.
    ///
    /// The data is copied into an owned buffer.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        let owned = data.to_vec();
        let mut cursor = Cursor::new(&owned);
        let superblock = Superblock::parse(&mut cursor)?;

        Ok(Hdf5File {
            data: FileData::Bytes(owned),
            superblock,
            chunk_cache: Arc::new(ChunkCache::default()),
        })
    }

    /// Get the parsed superblock.
    pub fn superblock(&self) -> &Superblock {
        &self.superblock
    }

    /// Get the root group of the file.
    pub fn root_group(&self) -> Result<Group<'_>> {
        let data = self.data.as_slice();
        let addr = self.superblock.root_object_header_address()?;

        Ok(Group::new(
            data,
            addr,
            "/".to_string(),
            self.superblock.offset_size,
            self.superblock.length_size,
            self.chunk_cache.clone(),
        ))
    }

    /// Convenience: get a dataset at a path like "/group1/dataset".
    pub fn dataset(&self, path: &str) -> Result<Dataset<'_>> {
        let parts: Vec<&str> = path
            .trim_start_matches('/')
            .split('/')
            .filter(|s| !s.is_empty())
            .collect();

        if parts.is_empty() {
            return Err(Error::DatasetNotFound(path.to_string()));
        }

        let mut group = self.root_group()?;
        for &part in &parts[..parts.len() - 1] {
            group = group.group(part)?;
        }

        group.dataset(parts[parts.len() - 1])
    }

    /// Convenience: get a group at a path like "/group1/subgroup".
    pub fn group(&self, path: &str) -> Result<Group<'_>> {
        let parts: Vec<&str> = path
            .trim_start_matches('/')
            .split('/')
            .filter(|s| !s.is_empty())
            .collect();

        let mut group = self.root_group()?;
        for &part in &parts {
            group = group.group(part)?;
        }

        Ok(group)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_open_options_default() {
        let opts = OpenOptions::default();
        assert_eq!(opts.chunk_cache_bytes, 64 * 1024 * 1024);
        assert_eq!(opts.chunk_cache_slots, 521);
    }

    #[test]
    fn test_invalid_file() {
        let data = b"this is not an HDF5 file";
        let result = Hdf5File::from_bytes(data);
        assert!(result.is_err());
    }
}
