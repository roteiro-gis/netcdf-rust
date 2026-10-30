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
pub mod shared_message_table;
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
pub mod storage;

// Filters
pub mod filters;

// Utilities
pub mod cache;

use std::collections::HashMap;
#[cfg(unix)]
use std::ffi::{CString, OsStr};
use std::fs::File;
use std::io::ErrorKind;
#[cfg(unix)]
use std::os::fd::{AsRawFd, FromRawFd};
#[cfg(unix)]
use std::os::unix::ffi::OsStrExt;
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, OnceLock};

use memmap2::Mmap;
// parking_lot::Mutex used via fully-qualified paths in HeaderCache and constructors.

use cache::ChunkCache;
use error::{Error, Result};
use group::Group;
use messages::HdfMessage;
use object_header::ObjectHeader;
use shared_message_table::SharedMessageTableRef;
use storage::DynStorage;
use superblock::Superblock;

// Re-exports
pub use attribute_api::{Attribute, RawAttribute};
pub use cache::ChunkCacheStats;
use dataset::DatasetTemplate;
pub use dataset::{Dataset, DatasetChunk, DatasetChunkIterator, SliceInfo, SliceInfoElem};
pub use datatype_api::{
    dtype_element_size, CompoundField, EnumMember, H5Type, ReferenceType, StringEncoding,
    StringPadding, StringSize, VarLenKind,
};
pub use error::ByteOrder;
pub use filters::FilterRegistry;
pub use messages::datatype::Datatype;
pub use storage::{
    BlockCacheStats, BlockCacheStorage, BytesStorage, FileStorage, MmapStorage,
    RangeRequestStorage, Storage, StorageBuffer,
};

/// Default largest decoded chunk a dataset read will allocate: 256 MiB.
///
/// Real HDF5 writers keep chunks far smaller (h5py and netCDF target about
/// 1-4 MiB). The bound applies before any buffer for the chunk is allocated,
/// and each parallel decode worker can hold one such chunk.
pub const DEFAULT_MAX_CHUNK_BYTES: usize = 256 * 1024 * 1024;
/// Largest decoded chunk the HDF5 format permits (chunk sizes are 32-bit).
pub const HDF5_MAX_CHUNK_BYTES: usize = u32::MAX as usize;

/// Default largest decoded selection one dataset read will allocate:
/// unlimited.
///
/// A dataset's extent comes from the file, so callers reading untrusted files
/// should set [`OpenOptions::max_read_bytes`] to what they can afford.
pub const DEFAULT_MAX_READ_BYTES: usize = usize::MAX;

/// Default largest object header, across all of its chunks, the reader will
/// read: 64 MiB.
///
/// Real object headers are a few KiB; each message is at most 64 KiB and
/// attributes past a handful move to dense storage outside the header. The
/// bound applies before each header chunk is read, so continuation chunks
/// that overlap or repeat cannot multiply the bytes parsed for one object.
pub const DEFAULT_MAX_OBJECT_HEADER_BYTES: usize = 64 * 1024 * 1024;

/// Default byte capacity of a file's chunk cache (64 MiB).
pub const DEFAULT_CHUNK_CACHE_BYTES: usize = 64 * 1024 * 1024;
/// Default number of chunk cache slots.
pub const DEFAULT_CHUNK_CACHE_SLOTS: usize = 521;

/// Configuration options for opening an HDF5 file.
pub struct OpenOptions {
    /// Maximum bytes for the chunk cache. Default: 64 MiB.
    /// Zero disables both chunk retention and the optional 32 MiB hot-array copy.
    pub chunk_cache_bytes: usize,
    /// Maximum number of chunk cache slots. Default: 521.
    pub chunk_cache_slots: usize,
    /// Custom filter registry. If `None`, the default built-in filters are used.
    pub filter_registry: Option<FilterRegistry>,
    /// Resolver for HDF5 external raw data files. If `None`, external raw data
    /// files are not resolved.
    pub external_file_resolver: Option<Arc<dyn ExternalFileResolver>>,
    /// Optional resolver for HDF5 external links.
    pub external_link_resolver: Option<Arc<dyn ExternalLinkResolver>>,
    /// Largest decoded chunk (chunk dimensions times element size) a read
    /// may allocate. Larger chunks are refused with [`Error::ResourceLimit`]
    /// before allocation. Default: [`DEFAULT_MAX_CHUNK_BYTES`]; values above
    /// [`HDF5_MAX_CHUNK_BYTES`] are clamped to it.
    pub max_chunk_bytes: usize,
    /// Largest decoded selection (selected elements times the dataset's
    /// element size) one read call may allocate, for `read_array*`,
    /// `read_slice*`, `read_rows_columns_chunked*`, `read_raw_bytes`,
    /// `read_native_bytes`, `read_string` and `read_strings`. Larger
    /// selections are refused with [`Error::ResourceLimit`] before
    /// allocation. Reads into caller-provided buffers are not limited.
    /// Default: [`DEFAULT_MAX_READ_BYTES`] (unlimited).
    pub max_read_bytes: usize,
    /// Largest object header, summed over its first chunk and every
    /// continuation chunk, the reader will read and parse. Larger headers
    /// are refused with [`Error::ResourceLimit`] before the chunk that would
    /// exceed the limit is read. Default: [`DEFAULT_MAX_OBJECT_HEADER_BYTES`].
    pub max_object_header_bytes: usize,
}

impl Default for OpenOptions {
    fn default() -> Self {
        OpenOptions {
            chunk_cache_bytes: DEFAULT_CHUNK_CACHE_BYTES,
            chunk_cache_slots: DEFAULT_CHUNK_CACHE_SLOTS,
            filter_registry: None,
            external_file_resolver: None,
            external_link_resolver: None,
            max_chunk_bytes: DEFAULT_MAX_CHUNK_BYTES,
            max_read_bytes: DEFAULT_MAX_READ_BYTES,
            max_object_header_bytes: DEFAULT_MAX_OBJECT_HEADER_BYTES,
        }
    }
}

impl OpenOptions {
    /// Set [`OpenOptions::chunk_cache_bytes`].
    pub fn with_chunk_cache_bytes(mut self, chunk_cache_bytes: usize) -> Self {
        self.chunk_cache_bytes = chunk_cache_bytes;
        self
    }

    /// Set [`OpenOptions::chunk_cache_slots`].
    pub fn with_chunk_cache_slots(mut self, chunk_cache_slots: usize) -> Self {
        self.chunk_cache_slots = chunk_cache_slots;
        self
    }

    /// Set [`OpenOptions::filter_registry`].
    pub fn with_filter_registry(mut self, filter_registry: FilterRegistry) -> Self {
        self.filter_registry = Some(filter_registry);
        self
    }

    /// Set [`OpenOptions::external_file_resolver`].
    pub fn with_external_file_resolver(mut self, resolver: Arc<dyn ExternalFileResolver>) -> Self {
        self.external_file_resolver = Some(resolver);
        self
    }

    /// Set [`OpenOptions::external_link_resolver`].
    pub fn with_external_link_resolver(mut self, resolver: Arc<dyn ExternalLinkResolver>) -> Self {
        self.external_link_resolver = Some(resolver);
        self
    }

    /// Set [`OpenOptions::max_chunk_bytes`].
    pub fn with_max_chunk_bytes(mut self, max_chunk_bytes: usize) -> Self {
        self.max_chunk_bytes = max_chunk_bytes;
        self
    }

    /// Set [`OpenOptions::max_read_bytes`].
    pub fn with_max_read_bytes(mut self, max_read_bytes: usize) -> Self {
        self.max_read_bytes = max_read_bytes;
        self
    }

    /// Set [`OpenOptions::max_object_header_bytes`].
    pub fn with_max_object_header_bytes(mut self, max_object_header_bytes: usize) -> Self {
        self.max_object_header_bytes = max_object_header_bytes;
        self
    }
}

/// Resolves file names from HDF5 External Data Files messages to storage.
///
/// Implementations are responsible for their own path security policy. The
/// built-in [`FilesystemExternalFileResolver`] confines normal paths to a base
/// directory and, on Unix, opens paths through an anchored directory handle
/// without following symlinks. On non-Unix platforms it falls back to
/// canonicalize-then-open, so attacker-writable resolver roots are out of
/// scope there.
pub trait ExternalFileResolver: Send + Sync {
    fn resolve_external_file(&self, filename: &str) -> Result<Option<DynStorage>>;
}

/// Resolves HDF5 external links to another opened file.
///
/// Implementations are responsible for their own path security policy. The
/// built-in [`FilesystemExternalLinkResolver`] confines normal paths to a base
/// directory and, on Unix, opens paths through an anchored directory handle
/// without following symlinks. On non-Unix platforms it falls back to
/// canonicalize-then-open, so attacker-writable resolver roots are out of
/// scope there.
pub trait ExternalLinkResolver: Send + Sync {
    fn resolve_external_link(&self, filename: &str) -> Result<Option<Hdf5File>>;
}

fn normalize_resolver_path(filename: &str, description: &str) -> Result<PathBuf> {
    let path = Path::new(filename);
    if path.as_os_str().is_empty() {
        return Err(Error::InvalidData(format!("{description} path is empty")));
    }

    if path.is_absolute() {
        return Err(Error::InvalidData(format!(
            "{description} path must be relative: {filename}"
        )));
    }

    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Normal(name) => normalized.push(name),
            Component::CurDir => {}
            Component::Prefix(_) | Component::RootDir | Component::ParentDir => {
                return Err(Error::InvalidData(format!(
                    "{description} path must stay within the resolver base directory: {filename}"
                )));
            }
        }
    }

    if normalized.as_os_str().is_empty() {
        return Err(Error::InvalidData(format!("{description} path is empty")));
    }

    Ok(normalized)
}

#[cfg(not(unix))]
fn open_external_file_within_base(
    base_dir: &Path,
    relative_path: &Path,
    description: &str,
    filename: &str,
) -> Result<Option<File>> {
    let base = match base_dir.canonicalize() {
        Ok(path) => path,
        Err(err) if err.kind() == ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(err.into()),
    };
    let candidate = base.join(relative_path);
    let resolved = match candidate.canonicalize() {
        Ok(path) => path,
        Err(err) if err.kind() == ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(err.into()),
    };

    if !resolved.starts_with(&base) {
        return Err(Error::InvalidData(format!(
            "{description} path escapes the resolver base directory: {filename}"
        )));
    }

    Ok(Some(File::open(resolved)?))
}

#[cfg(unix)]
fn open_external_file_within_base(
    base_dir: &Path,
    relative_path: &Path,
    description: &str,
    filename: &str,
) -> Result<Option<File>> {
    let mut dir = match open_unix_path(
        base_dir,
        libc::O_RDONLY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
    ) {
        Ok(file) => file,
        Err(err) if path_lookup_is_missing(&err) => return Ok(None),
        Err(err) if path_lookup_is_symlink(&err) => {
            return Err(Error::InvalidData(format!(
                "{description} resolver base directory must not be a symlink"
            )));
        }
        Err(err) => return Err(err.into()),
    };
    if !dir.metadata()?.is_dir() {
        return Ok(None);
    }

    let parts: Vec<&OsStr> = relative_path
        .components()
        .filter_map(|component| match component {
            Component::Normal(name) => Some(name),
            _ => None,
        })
        .collect();

    let Some((leaf, parents)) = parts.split_last() else {
        return Err(Error::InvalidData(format!("{description} path is empty")));
    };

    for parent in parents {
        dir = match open_unix_child(
            &dir,
            parent,
            libc::O_RDONLY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
        ) {
            Ok(file) => file,
            Err(err) if path_lookup_is_missing(&err) => return Ok(None),
            Err(err) if path_lookup_is_symlink(&err) => {
                return Err(symlink_resolver_error(description, filename));
            }
            Err(err) => return Err(err.into()),
        };
        if !dir.metadata()?.is_dir() {
            return Ok(None);
        }
    }

    let file = match open_unix_child(
        &dir,
        leaf,
        libc::O_RDONLY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
    ) {
        Ok(file) => file,
        Err(err) if path_lookup_is_missing(&err) => return Ok(None),
        Err(err) if path_lookup_is_symlink(&err) => {
            return Err(symlink_resolver_error(description, filename));
        }
        Err(err) => return Err(err.into()),
    };

    if file.metadata()?.is_dir() {
        return Err(Error::InvalidData(format!(
            "{description} path resolves to a directory: {filename}"
        )));
    }

    Ok(Some(file))
}

#[cfg(unix)]
fn open_unix_path(path: &Path, flags: libc::c_int) -> std::io::Result<File> {
    let path = CString::new(path.as_os_str().as_bytes()).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "filesystem path contains an interior NUL byte",
        )
    })?;
    // SAFETY: `path` is NUL-terminated, `flags` is passed through unchanged,
    // and the returned descriptor is checked before ownership is assumed.
    let fd = unsafe { libc::open(path.as_ptr(), flags) };
    file_from_unix_fd(fd)
}

#[cfg(unix)]
fn open_unix_child(dir: &File, name: &OsStr, flags: libc::c_int) -> std::io::Result<File> {
    let name = CString::new(name.as_bytes()).map_err(|_| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "filesystem path contains an interior NUL byte",
        )
    })?;
    // SAFETY: `dir` owns a live descriptor, `name` is NUL-terminated, and the
    // returned descriptor is checked before ownership is assumed.
    let fd = unsafe { libc::openat(dir.as_raw_fd(), name.as_ptr(), flags) };
    file_from_unix_fd(fd)
}

#[cfg(unix)]
fn file_from_unix_fd(fd: libc::c_int) -> std::io::Result<File> {
    if fd < 0 {
        Err(std::io::Error::last_os_error())
    } else {
        // SAFETY: a successful `open`/`openat` returned an owned descriptor,
        // and this transfers that ownership exactly once to `File`.
        Ok(unsafe { File::from_raw_fd(fd) })
    }
}

#[cfg(unix)]
fn path_lookup_is_missing(err: &std::io::Error) -> bool {
    err.kind() == ErrorKind::NotFound
        || matches!(err.raw_os_error(), Some(code) if code == libc::ENOTDIR)
}

#[cfg(unix)]
fn path_lookup_is_symlink(err: &std::io::Error) -> bool {
    matches!(err.raw_os_error(), Some(code) if code == libc::ELOOP)
}

#[cfg(unix)]
fn symlink_resolver_error(description: &str, filename: &str) -> Error {
    Error::InvalidData(format!(
        "{description} path escapes the resolver base directory or uses a symlink: {filename}"
    ))
}

/// Filesystem resolver for external raw data files.
///
/// The resolver rejects absolute paths and `..` components. On Unix, it opens
/// paths relative to `base_dir` using `openat` and `O_NOFOLLOW`, so symlinks
/// are rejected instead of being followed. On non-Unix platforms,
/// attacker-writable resolver roots are out of scope.
#[derive(Debug, Clone)]
pub struct FilesystemExternalFileResolver {
    base_dir: PathBuf,
}

impl FilesystemExternalFileResolver {
    pub fn new(base_dir: impl Into<PathBuf>) -> Self {
        Self {
            base_dir: base_dir.into(),
        }
    }

    fn relative_path_for(&self, filename: &str) -> Result<PathBuf> {
        normalize_resolver_path(filename, "external raw data file")
    }
}

impl ExternalFileResolver for FilesystemExternalFileResolver {
    fn resolve_external_file(&self, filename: &str) -> Result<Option<DynStorage>> {
        let relative_path = self.relative_path_for(filename)?;
        let Some(file) = open_external_file_within_base(
            &self.base_dir,
            &relative_path,
            "external raw data file",
            filename,
        )?
        else {
            return Ok(None);
        };
        Ok(Some(Arc::new(FileStorage::from_file(file)?)))
    }
}

/// Filesystem resolver for external links. Linked files are cached after the
/// first successful open.
///
/// The resolver rejects absolute paths and `..` components. On Unix, it opens
/// paths relative to `base_dir` using `openat` and `O_NOFOLLOW`, so symlinks
/// are rejected instead of being followed. On non-Unix platforms,
/// attacker-writable resolver roots are out of scope.
pub struct FilesystemExternalLinkResolver {
    base_dir: PathBuf,
    cache: parking_lot::Mutex<HashMap<PathBuf, Hdf5File>>,
}

impl FilesystemExternalLinkResolver {
    pub fn new(base_dir: impl Into<PathBuf>) -> Self {
        Self {
            base_dir: base_dir.into(),
            cache: parking_lot::Mutex::new(HashMap::new()),
        }
    }

    fn relative_path_for(&self, filename: &str) -> Result<PathBuf> {
        normalize_resolver_path(filename, "external link")
    }
}

impl ExternalLinkResolver for FilesystemExternalLinkResolver {
    fn resolve_external_link(&self, filename: &str) -> Result<Option<Hdf5File>> {
        let relative_path = self.relative_path_for(filename)?;

        if let Some(file) = self.cache.lock().get(&relative_path).cloned() {
            return Ok(Some(file));
        }

        let Some(opened) = open_external_file_within_base(
            &self.base_dir,
            &relative_path,
            "external link",
            filename,
        )?
        else {
            return Ok(None);
        };

        let file = Hdf5File::from_storage(Arc::new(FileStorage::from_file(opened)?))?;
        self.cache.lock().insert(relative_path, file.clone());
        Ok(Some(file))
    }
}

/// Cache for parsed object headers, keyed by file address.
pub type HeaderCache = Arc<parking_lot::Mutex<HashMap<u64, Arc<ObjectHeader>>>>;

/// An opened HDF5 file.
///
/// This is the main entry point for reading HDF5 files. Storage is random-
/// access and range-based, so metadata and data reads do not require an eager
/// whole-file mapping.
#[derive(Clone)]
pub struct Hdf5File {
    context: Arc<FileContext>,
}

pub(crate) struct FileContext {
    pub(crate) storage: DynStorage,
    pub(crate) superblock: Superblock,
    pub(crate) chunk_cache: Arc<ChunkCache>,
    pub(crate) header_cache: HeaderCache,
    pub(crate) old_groups: parking_lot::Mutex<HashMap<u64, Arc<group::OldGroupEntries>>>,
    pub(crate) dataset_path_cache: Arc<parking_lot::Mutex<HashMap<String, Arc<DatasetTemplate>>>>,
    pub(crate) filter_registry: Arc<FilterRegistry>,
    pub(crate) external_file_resolver: Option<Arc<dyn ExternalFileResolver>>,
    pub(crate) external_link_resolver: Option<Arc<dyn ExternalLinkResolver>>,
    pub(crate) external_file_cache: parking_lot::Mutex<HashMap<String, DynStorage>>,
    pub(crate) max_chunk_bytes: usize,
    pub(crate) max_read_bytes: usize,
    pub(crate) max_object_header_bytes: usize,
    sohm_table: OnceLock<std::result::Result<Option<SharedMessageTableRef>, String>>,
    full_file_cache: OnceLock<StorageBuffer>,
}

impl FileContext {
    pub(crate) fn read_range(&self, offset: u64, len: usize) -> Result<StorageBuffer> {
        self.storage.read_range(offset, len)
    }

    pub(crate) fn full_file_data(&self) -> Result<StorageBuffer> {
        if let Some(buffer) = self.full_file_cache.get() {
            return Ok(buffer.clone());
        }

        let len = usize::try_from(self.storage.len()).map_err(|_| {
            Error::InvalidData("file size exceeds platform usize capacity".to_string())
        })?;
        let buffer = self.storage.read_range(0, len)?;
        let _ = self.full_file_cache.set(buffer);
        Ok(self
            .full_file_cache
            .get()
            .expect("full-file buffer must exist after successful initialization")
            .clone())
    }

    pub(crate) fn get_or_parse_header(&self, addr: u64) -> Result<Arc<ObjectHeader>> {
        {
            let cache = self.header_cache.lock();
            if let Some(hdr) = cache.get(&addr) {
                return Ok(Arc::clone(hdr));
            }
        }

        let mut hdr = ObjectHeader::parse_at_storage_with_limit(
            self.storage.as_ref(),
            addr,
            self.superblock.offset_size,
            self.superblock.length_size,
            self.max_object_header_bytes,
        )?;
        hdr.resolve_shared_messages_storage_with_sohm(
            self.storage.as_ref(),
            self.superblock.offset_size,
            self.superblock.length_size,
            self.max_object_header_bytes,
            |heap_id, message_type| self.resolve_sohm_message(heap_id, message_type),
        )?;
        let arc = Arc::new(hdr);
        let mut cache = self.header_cache.lock();
        cache.insert(addr, Arc::clone(&arc));
        Ok(arc)
    }

    /// Parse the object header at `addr` again, bypassing the header cache,
    /// and record the exact datatype encodings parsed for it, including those
    /// of shared messages it refers to.
    pub(crate) fn parse_header_capturing(
        &self,
        addr: u64,
    ) -> Result<(ObjectHeader, messages::CapturedEncodings)> {
        let capture = messages::RawCapture::start();
        let mut hdr = ObjectHeader::parse_at_storage_with_limit(
            self.storage.as_ref(),
            addr,
            self.superblock.offset_size,
            self.superblock.length_size,
            self.max_object_header_bytes,
        )?;
        hdr.resolve_shared_messages_storage_with_sohm(
            self.storage.as_ref(),
            self.superblock.offset_size,
            self.superblock.length_size,
            self.max_object_header_bytes,
            |heap_id, message_type| self.resolve_sohm_message(heap_id, message_type),
        )?;
        Ok((hdr, capture.finish()))
    }

    fn resolve_sohm_message(
        &self,
        heap_id: &[u8],
        message_type: u16,
    ) -> Result<Option<HdfMessage>> {
        let Some(table) = self.sohm_table()? else {
            return Ok(None);
        };
        table.resolve_heap_message(
            heap_id,
            message_type,
            self.storage.as_ref(),
            self.superblock.offset_size,
            self.superblock.length_size,
            Some(self.filter_registry.as_ref()),
        )
    }

    fn sohm_table(&self) -> Result<Option<SharedMessageTableRef>> {
        let cached = self
            .sohm_table
            .get_or_init(|| self.load_sohm_table().map_err(|err| err.to_string()));
        match cached {
            Ok(table) => Ok(table.clone()),
            Err(message) => Err(Error::InvalidData(format!(
                "failed to load SOHM table: {message}"
            ))),
        }
    }

    fn load_sohm_table(&self) -> Result<Option<SharedMessageTableRef>> {
        let Some(extension_address) = self.superblock.extension_address else {
            return Ok(None);
        };
        let extension = ObjectHeader::parse_at_storage_with_limit(
            self.storage.as_ref(),
            extension_address,
            self.superblock.offset_size,
            self.superblock.length_size,
            self.max_object_header_bytes,
        )?;

        let shared_table = extension.messages.iter().find_map(|message| match message {
            HdfMessage::SharedTable(table) => Some(table),
            _ => None,
        });
        let Some(shared_table) = shared_table else {
            return Ok(None);
        };

        let table = crate::shared_message_table::SharedMessageTable::parse_at_storage(
            self.storage.as_ref(),
            shared_table.table_address,
            shared_table.num_indices,
            self.superblock.offset_size,
        )?;
        Ok(Some(Arc::new(table)))
    }

    pub(crate) fn resolve_external_file(&self, filename: &str) -> Result<Option<DynStorage>> {
        if let Some(storage) = self.external_file_cache.lock().get(filename).cloned() {
            return Ok(Some(storage));
        }

        let Some(resolver) = self.external_file_resolver.as_ref() else {
            return Ok(None);
        };
        let Some(storage) = resolver.resolve_external_file(filename)? else {
            return Ok(None);
        };
        self.external_file_cache
            .lock()
            .insert(filename.to_string(), storage.clone());
        Ok(Some(storage))
    }
}

impl Hdf5File {
    fn from_storage_impl(storage: DynStorage, options: OpenOptions) -> Result<Self> {
        let superblock = Superblock::parse_from_storage(storage.as_ref())?;
        let cache = Arc::new(ChunkCache::new(
            options.chunk_cache_bytes,
            options.chunk_cache_slots,
        ));
        let registry = options.filter_registry.unwrap_or_default();
        let external_file_resolver = options.external_file_resolver;
        let external_link_resolver = options.external_link_resolver;

        Ok(Hdf5File {
            context: Arc::new(FileContext {
                storage,
                superblock,
                chunk_cache: cache,
                header_cache: Arc::new(parking_lot::Mutex::new(HashMap::new())),
                old_groups: Default::default(),
                dataset_path_cache: Arc::new(parking_lot::Mutex::new(HashMap::new())),
                filter_registry: Arc::new(registry),
                external_file_resolver,
                external_link_resolver,
                external_file_cache: parking_lot::Mutex::new(HashMap::new()),
                max_chunk_bytes: options.max_chunk_bytes.min(HDF5_MAX_CHUNK_BYTES),
                max_read_bytes: options.max_read_bytes,
                max_object_header_bytes: options.max_object_header_bytes,
                sohm_table: OnceLock::new(),
                full_file_cache: OnceLock::new(),
            }),
        })
    }

    /// Open an HDF5 file with default options.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_with_options(path, OpenOptions::default())
    }

    /// Open an HDF5 file with custom options.
    pub fn open_with_options(path: impl AsRef<Path>, options: OpenOptions) -> Result<Self> {
        let path = path.as_ref();
        Self::from_storage_with_options(Arc::new(FileStorage::open(path)?), options)
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

    /// Open an HDF5 file from shared bytes without copying them. The caller can keep its own
    /// handle, to open the same bytes again or read them otherwise.
    pub fn from_shared_bytes(data: Arc<[u8]>) -> Result<Self> {
        Self::from_shared_bytes_with_options(data, OpenOptions::default())
    }

    /// [`Self::from_shared_bytes`] with custom options.
    pub fn from_shared_bytes_with_options(data: Arc<[u8]>, options: OpenOptions) -> Result<Self> {
        Self::from_storage_with_options(Arc::new(BytesStorage::from_shared(data)), options)
    }

    /// Open an HDF5 file from the bytes of any owner (a `Box<[u8]>`, a `bytes::Bytes`, ...)
    /// without copying them.
    pub fn from_owner<T>(owner: T, options: OpenOptions) -> Result<Self>
    where
        T: AsRef<[u8]> + Send + Sync + 'static,
    {
        Self::from_storage_with_options(Arc::new(BytesStorage::from_owner(owner)), options)
    }

    /// Open an HDF5 file from an owned byte vector with custom options.
    pub fn from_vec_with_options(data: Vec<u8>, options: OpenOptions) -> Result<Self> {
        Self::from_storage_with_options(Arc::new(BytesStorage::new(data)), options)
    }

    /// Open an HDF5 file from an existing memory map with custom options.
    ///
    /// This avoids remapping when the caller already owns a read-only mapping.
    pub fn from_mmap_with_options(mmap: Mmap, options: OpenOptions) -> Result<Self> {
        Self::from_storage_with_options(Arc::new(MmapStorage::new(mmap)), options)
    }

    /// Open an HDF5 file from a custom random-access storage backend.
    pub fn from_storage(storage: DynStorage) -> Result<Self> {
        Self::from_storage_with_options(storage, OpenOptions::default())
    }

    /// Open an HDF5 file from a custom random-access storage backend.
    pub fn from_storage_with_options(storage: DynStorage, options: OpenOptions) -> Result<Self> {
        Self::from_storage_impl(storage, options)
    }

    /// Get the parsed superblock.
    pub fn superblock(&self) -> &Superblock {
        &self.context.superblock
    }

    /// Access the underlying random-access storage backend.
    pub fn storage(&self) -> &dyn Storage {
        self.context.storage.as_ref()
    }

    /// Return current chunk-cache statistics for this file.
    pub fn chunk_cache_stats(&self) -> ChunkCacheStats {
        self.context.chunk_cache.stats()
    }

    /// Look up or parse an object header at the given address.
    ///
    /// Uses the internal cache to avoid re-parsing the same header.
    pub fn get_or_parse_header(&self, addr: u64) -> Result<Arc<ObjectHeader>> {
        self.context.get_or_parse_header(addr)
    }

    /// Get the root group of the file.
    pub fn root_group(&self) -> Result<Group> {
        let addr = self.context.superblock.root_object_header_address()?;

        Ok(Group::new(
            self.context.clone(),
            addr,
            "/".to_string(),
            addr, // root_address = self
        ))
    }

    /// Convenience: get a dataset at a path like "/group1/dataset".
    pub fn dataset(&self, path: &str) -> Result<Dataset> {
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
            .context
            .dataset_path_cache
            .lock()
            .get(&normalized_path)
            .cloned()
        {
            return Ok(Dataset::from_template(self.context.clone(), template));
        }

        let mut group = self.root_group()?;
        for &part in &parts[..parts.len() - 1] {
            group = group.group(part).map_err(|e| e.with_context(path))?;
        }

        let dataset = group
            .dataset(parts[parts.len() - 1])
            .map_err(|e| e.with_context(path))?;
        if Arc::ptr_eq(&dataset.context, &self.context) {
            self.context
                .dataset_path_cache
                .lock()
                .insert(normalized_path, dataset.template());
        }
        Ok(dataset)
    }

    /// Convenience: get a group at a path like "/group1/subgroup".
    pub fn group(&self, path: &str) -> Result<Group> {
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

    fn testdata(name: &str) -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../testdata/hdf5")
            .join(name)
    }

    #[test]
    fn old_group_cache_is_filled_once_and_does_not_keep_the_file() {
        let path = testdata("old_format_large_group.h5");
        if !path.exists() {
            eprintln!("SKIPPED: fixture old_format_large_group.h5 not found");
            return;
        }
        let file = Hdf5File::open(path).unwrap();
        let weak = Arc::downgrade(&file.context);
        let g = file.root_group().unwrap();
        let names = g.member_names().unwrap();
        let objects = g.members().unwrap();
        let group_names: Vec<String> = objects.0.iter().map(|g| g.name().to_string()).collect();
        let dataset_names: Vec<String> = objects.1.iter().map(|d| d.name().to_string()).collect();
        assert_eq!(names, (group_names, dataset_names));
        assert_eq!(file.context.old_groups.lock().len(), 1);
        assert_eq!(g.member_names().unwrap(), names);
        assert_eq!(file.context.old_groups.lock().len(), 1);
        drop(objects);
        drop(g);
        drop(file);
        assert!(
            weak.upgrade().is_none(),
            "the group cache must not keep the file context alive"
        );
    }

    #[test]
    fn old_group_cache_does_not_hold_its_lock_during_storage_reads() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct LockProbeStorage {
            inner: crate::storage::BytesStorage,
            context: parking_lot::Mutex<Option<std::sync::Weak<FileContext>>>,
            checked_reads: AtomicUsize,
            locked_reads: AtomicUsize,
        }

        impl crate::storage::Storage for LockProbeStorage {
            fn len(&self) -> u64 {
                self.inner.len()
            }

            fn read_range(&self, offset: u64, len: usize) -> crate::error::Result<StorageBuffer> {
                if let Some(context) = self.context.lock().as_ref().and_then(|weak| weak.upgrade())
                {
                    self.checked_reads.fetch_add(1, Ordering::Relaxed);
                    if context.old_groups.try_lock().is_none() {
                        self.locked_reads.fetch_add(1, Ordering::Relaxed);
                    }
                }
                self.inner.read_range(offset, len)
            }
        }

        let path = testdata("old_format_large_group.h5");
        if !path.exists() {
            eprintln!("SKIPPED: fixture old_format_large_group.h5 not found");
            return;
        }
        let storage = Arc::new(LockProbeStorage {
            inner: crate::storage::BytesStorage::new(std::fs::read(path).unwrap()),
            context: parking_lot::Mutex::new(None),
            checked_reads: AtomicUsize::new(0),
            locked_reads: AtomicUsize::new(0),
        });
        let file = Hdf5File::from_storage(storage.clone()).unwrap();
        *storage.context.lock() = Some(Arc::downgrade(&file.context));

        file.root_group().unwrap().member_names().unwrap();
        assert!(storage.checked_reads.load(Ordering::Relaxed) > 0);
        assert_eq!(storage.locked_reads.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn empty_group_does_not_read_unused_heap_segment() {
        let path = testdata("old_format_empty_heap_past_eof.h5");
        if !path.exists() {
            eprintln!("SKIPPED: fixture old_format_empty_heap_past_eof.h5 not found");
            return;
        }
        let file = Hdf5File::open(path).unwrap();
        assert!(file.root_group().unwrap().members().unwrap().0.is_empty());
        assert_eq!(
            file.root_group().unwrap().member_names().unwrap(),
            (vec![], vec![])
        );
    }
    #[test]
    fn open_options_builder_sets_every_field() {
        let opts = OpenOptions::default()
            .with_chunk_cache_bytes(1024)
            .with_chunk_cache_slots(7)
            .with_filter_registry(FilterRegistry::new())
            .with_max_chunk_bytes(4096)
            .with_max_read_bytes(8192)
            .with_max_object_header_bytes(16384);
        assert_eq!(opts.chunk_cache_bytes, 1024);
        assert_eq!(opts.chunk_cache_slots, 7);
        assert!(opts.filter_registry.is_some());
        assert!(opts.external_file_resolver.is_none());
        assert!(opts.external_link_resolver.is_none());
        assert_eq!(opts.max_chunk_bytes, 4096);
        assert_eq!(opts.max_read_bytes, 8192);
        assert_eq!(opts.max_object_header_bytes, 16384);
    }

    #[test]
    fn open_options_default() {
        let opts = OpenOptions::default();
        assert_eq!(opts.chunk_cache_bytes, 64 * 1024 * 1024);
        assert_eq!(opts.chunk_cache_slots, 521);
        assert!(opts.external_file_resolver.is_none());
    }

    #[test]
    fn invalid_file() {
        let data = b"this is not an HDF5 file";
        let result = Hdf5File::from_bytes(data);
        assert!(result.is_err());
    }

    #[test]
    fn btree_v2_chunked_fixture_uses_btree_v2_index() {
        use crate::messages::layout::{ChunkIndexing, DataLayout};

        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("testdata/hdf5/btree_v2_chunked.h5");
        if !path.exists() {
            eprintln!("SKIPPED: fixture btree_v2_chunked.h5 not found");
            return;
        }

        let file = Hdf5File::open(path).unwrap();
        let dataset = file.dataset("/data").unwrap();
        assert!(matches!(
            dataset.layout,
            DataLayout::Chunked {
                chunk_indexing: Some(ChunkIndexing::BTreeV2),
                ..
            }
        ));
    }

    #[test]
    fn filesystem_external_file_resolver_reads_relative_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("raw.bin");
        std::fs::write(&path, b"abcdef").unwrap();

        let resolver = FilesystemExternalFileResolver::new(dir.path());
        let storage = resolver
            .resolve_external_file("raw.bin")
            .unwrap()
            .expect("raw file should resolve");
        let bytes = storage.read_range(2, 3).unwrap();
        assert_eq!(bytes.as_ref(), b"cde");
    }

    #[test]
    fn filesystem_external_file_resolver_rejects_absolute_path() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("raw.bin");
        std::fs::write(&path, b"abcdef").unwrap();

        let resolver = FilesystemExternalFileResolver::new(dir.path());
        let err = match resolver.resolve_external_file(path.to_str().unwrap()) {
            Ok(_) => panic!("absolute external file path should be rejected"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("must be relative"));
    }

    #[test]
    fn filesystem_external_file_resolver_rejects_parent_component() {
        let dir = tempfile::tempdir().unwrap();
        let resolver = FilesystemExternalFileResolver::new(dir.path());

        let err = match resolver.resolve_external_file("../raw.bin") {
            Ok(_) => panic!("parent external file path should be rejected"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("resolver base directory"));
    }

    #[cfg(unix)]
    #[test]
    fn filesystem_external_file_resolver_rejects_symlink_escape() {
        use std::os::unix::fs::symlink;

        let dir = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let outside_path = outside.path().join("raw.bin");
        std::fs::write(&outside_path, b"abcdef").unwrap();
        symlink(&outside_path, dir.path().join("raw.bin")).unwrap();

        let resolver = FilesystemExternalFileResolver::new(dir.path());
        let err = match resolver.resolve_external_file("raw.bin") {
            Ok(_) => panic!("symlink escape should be rejected"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("escapes"));
    }

    #[cfg(unix)]
    #[test]
    fn filesystem_external_file_resolver_rejects_symlink_inside_base() {
        use std::os::unix::fs::symlink;

        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("raw.bin"), b"abcdef").unwrap();
        symlink("raw.bin", dir.path().join("link.bin")).unwrap();

        let resolver = FilesystemExternalFileResolver::new(dir.path());
        let err = match resolver.resolve_external_file("link.bin") {
            Ok(_) => panic!("symlinks should be rejected even when they point inside the base"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("symlink"));
    }

    #[cfg(unix)]
    #[test]
    fn filesystem_external_file_resolver_rejects_symlink_directory_component() {
        use std::os::unix::fs::symlink;

        let dir = tempfile::tempdir().unwrap();
        let real_dir = dir.path().join("real");
        std::fs::create_dir(&real_dir).unwrap();
        std::fs::write(real_dir.join("raw.bin"), b"abcdef").unwrap();
        symlink("real", dir.path().join("linkdir")).unwrap();

        let resolver = FilesystemExternalFileResolver::new(dir.path());
        let err = match resolver.resolve_external_file("linkdir/raw.bin") {
            Ok(_) => panic!("symlinked directory components should be rejected"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("symlink"));
    }

    #[test]
    fn filesystem_external_link_resolver_rejects_absolute_path() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("linked.h5");
        std::fs::write(&path, b"not really hdf5").unwrap();

        let resolver = FilesystemExternalLinkResolver::new(dir.path());
        let err = match resolver.resolve_external_link(path.to_str().unwrap()) {
            Ok(_) => panic!("absolute external link path should be rejected"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("must be relative"));
    }

    #[test]
    fn filesystem_external_link_resolver_rejects_parent_component() {
        let dir = tempfile::tempdir().unwrap();
        let resolver = FilesystemExternalLinkResolver::new(dir.path());

        let err = match resolver.resolve_external_link("../linked.h5") {
            Ok(_) => panic!("parent external link path should be rejected"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("resolver base directory"));
    }

    #[cfg(unix)]
    #[test]
    fn filesystem_external_link_resolver_rejects_symlink_escape() {
        use std::os::unix::fs::symlink;

        let dir = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let outside_path = outside.path().join("linked.h5");
        std::fs::write(&outside_path, b"not really hdf5").unwrap();
        symlink(&outside_path, dir.path().join("linked.h5")).unwrap();

        let resolver = FilesystemExternalLinkResolver::new(dir.path());
        let err = match resolver.resolve_external_link("linked.h5") {
            Ok(_) => panic!("symlink escape should be rejected"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("escapes"));
    }
}
