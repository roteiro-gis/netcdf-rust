pub mod checksum;
pub mod error;
pub mod io;

// Level 0 — File Metadata
pub mod superblock;

// Level 1 — File Infrastructure
pub mod btree_v1;
pub mod btree_v2;
pub mod chunk_index;
pub mod extensible_array;
pub mod fixed_array;
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
pub mod reference;

// Filters
pub mod filters;

// Utilities
pub mod cache;

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use memmap2::Mmap;
// parking_lot::Mutex used via fully-qualified paths in HeaderCache and constructors.

use cache::ChunkCache;
use error::{Error, Result};
use group::Group;
use io::Cursor;
use object_header::ObjectHeader;
use superblock::Superblock;

// Re-exports
pub use attribute_api::Attribute;
use dataset::DatasetTemplate;
pub use dataset::{Dataset, SliceInfo, SliceInfoElem};
pub use datatype_api::{
    dtype_element_size, CompoundField, EnumMember, H5Type, ReferenceType, StringEncoding,
    StringPadding, StringSize,
};
pub use error::ByteOrder;
pub use filters::FilterRegistry;
pub use messages::datatype::Datatype;

/// Configuration options for opening an HDF5 file.
pub struct OpenOptions {
    /// Maximum bytes for the chunk cache. Default: 64 MiB.
    pub chunk_cache_bytes: usize,
    /// Maximum number of chunk cache slots. Default: 521.
    pub chunk_cache_slots: usize,
    /// Custom filter registry. If `None`, the default built-in filters are used.
    pub filter_registry: Option<FilterRegistry>,
}

impl Default for OpenOptions {
    fn default() -> Self {
        OpenOptions {
            chunk_cache_bytes: 64 * 1024 * 1024,
            chunk_cache_slots: 521,
            filter_registry: None,
        }
    }
}

/// Cache for parsed object headers, keyed by file address.
pub type HeaderCache = Arc<parking_lot::Mutex<HashMap<u64, Arc<ObjectHeader>>>>;

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
    /// Object header cache — avoids re-parsing the same header.
    header_cache: HeaderCache,
    /// Dataset path cache — avoids repeated path traversal and metadata rebuilds.
    dataset_path_cache: Arc<parking_lot::Mutex<HashMap<String, Arc<DatasetTemplate>>>>,
    /// Filter registry for decompression — users can register custom filters.
    filter_registry: Arc<FilterRegistry>,
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
    fn from_file_data(data: FileData, options: OpenOptions) -> Result<Self> {
        let mut cursor = Cursor::new(data.as_slice());
        let superblock = Superblock::parse(&mut cursor)?;
        let cache = Arc::new(ChunkCache::new(
            options.chunk_cache_bytes,
            options.chunk_cache_slots,
        ));
        let registry = options.filter_registry.unwrap_or_default();

        Ok(Hdf5File {
            data,
            superblock,
            chunk_cache: cache,
            header_cache: Arc::new(parking_lot::Mutex::new(HashMap::new())),
            dataset_path_cache: Arc::new(parking_lot::Mutex::new(HashMap::new())),
            filter_registry: Arc::new(registry),
        })
    }

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
        Self::from_mmap_with_options(mmap, options)
    }

    /// Open an HDF5 file from an in-memory byte slice.
    ///
    /// The data is copied into an owned buffer.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        Self::from_bytes_with_options(data, OpenOptions::default())
    }

    /// Open an HDF5 file from an in-memory byte slice with custom options.
    ///
    /// The data is copied into an owned buffer.
    pub fn from_bytes_with_options(data: &[u8], options: OpenOptions) -> Result<Self> {
        Self::from_vec_with_options(data.to_vec(), options)
    }

    /// Open an HDF5 file from an owned byte vector without copying.
    pub fn from_vec(data: Vec<u8>) -> Result<Self> {
        Self::from_vec_with_options(data, OpenOptions::default())
    }

    /// Open an HDF5 file from an owned byte vector with custom options.
    pub fn from_vec_with_options(data: Vec<u8>, options: OpenOptions) -> Result<Self> {
        Self::from_file_data(FileData::Bytes(data), options)
    }

    /// Open an HDF5 file from an existing memory map with custom options.
    ///
    /// This avoids remapping when the caller already owns a read-only mapping.
    pub fn from_mmap_with_options(mmap: Mmap, options: OpenOptions) -> Result<Self> {
        Self::from_file_data(FileData::Mmap(mmap), options)
    }

    /// Get the parsed superblock.
    pub fn superblock(&self) -> &Superblock {
        &self.superblock
    }

    /// Look up or parse an object header at the given address.
    ///
    /// Uses the internal cache to avoid re-parsing the same header.
    pub fn get_or_parse_header(&self, addr: u64) -> Result<Arc<ObjectHeader>> {
        {
            let cache = self.header_cache.lock();
            if let Some(hdr) = cache.get(&addr) {
                return Ok(Arc::clone(hdr));
            }
        }
        let data = self.data.as_slice();
        let mut hdr = ObjectHeader::parse_at(
            data,
            addr,
            self.superblock.offset_size,
            self.superblock.length_size,
        )?;
        hdr.resolve_shared_messages(
            data,
            self.superblock.offset_size,
            self.superblock.length_size,
        )?;
        let arc = Arc::new(hdr);
        let mut cache = self.header_cache.lock();
        cache.insert(addr, Arc::clone(&arc));
        Ok(arc)
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
            addr, // root_address = self
            self.chunk_cache.clone(),
            self.header_cache.clone(),
            self.filter_registry.clone(),
        ))
    }

    /// Convenience: get a dataset at a path like "/group1/dataset".
    pub fn dataset(&self, path: &str) -> Result<Dataset<'_>> {
        let parts: Vec<&str> = path
            .trim_start_matches('/')
            .split('/')
            .filter(|s| !s.is_empty())
            .collect();
        let normalized_path = format!("/{}", parts.join("/"));

        if parts.is_empty() {
            return Err(Error::DatasetNotFound(path.to_string()).with_context(path));
        }

        if let Some(template) = self
            .dataset_path_cache
            .lock()
            .get(&normalized_path)
            .cloned()
        {
            return Ok(Dataset::from_template(
                self.data.as_slice(),
                self.superblock.offset_size,
                self.superblock.length_size,
                template,
                self.chunk_cache.clone(),
                self.filter_registry.clone(),
            ));
        }

        let mut group = self.root_group()?;
        for &part in &parts[..parts.len() - 1] {
            group = group.group(part).map_err(|e| e.with_context(path))?;
        }

        let dataset = group
            .dataset(parts[parts.len() - 1])
            .map_err(|e| e.with_context(path))?;
        self.dataset_path_cache
            .lock()
            .insert(normalized_path, dataset.template());
        Ok(dataset)
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
