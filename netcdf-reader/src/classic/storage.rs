use std::ops::Deref;
use std::sync::Arc;

use memmap2::Mmap;

use crate::error::{Error, Result};

/// Header prefix read on the first pass for range-backed classic opens.
#[cfg(feature = "netcdf4")]
const INITIAL_HEADER_READ: usize = 64 * 1024;

#[derive(Clone)]
enum ClassicStorageBacking {
    Bytes(Arc<[u8]>),
    Mmap(Arc<Mmap>),
    #[cfg(feature = "netcdf4")]
    Range(hdf5_reader::storage::DynStorage),
}

#[derive(Clone)]
pub(crate) struct ClassicStorage {
    backing: ClassicStorageBacking,
    len: u64,
}

#[derive(Clone)]
pub(crate) enum ClassicBuffer {
    Bytes {
        data: Arc<[u8]>,
        start: usize,
        len: usize,
    },
    Mmap {
        mmap: Arc<Mmap>,
        start: usize,
        len: usize,
    },
    #[cfg(feature = "netcdf4")]
    Range(hdf5_reader::StorageBuffer),
}

impl ClassicStorage {
    pub(crate) fn from_bytes(bytes: Vec<u8>) -> Self {
        let len = bytes.len() as u64;
        Self {
            backing: ClassicStorageBacking::Bytes(Arc::<[u8]>::from(bytes)),
            len,
        }
    }

    pub(crate) fn from_mmap(mmap: Mmap) -> Self {
        let len = mmap.len() as u64;
        Self {
            backing: ClassicStorageBacking::Mmap(Arc::new(mmap)),
            len,
        }
    }

    #[cfg(feature = "netcdf4")]
    pub(crate) fn from_range(storage: hdf5_reader::storage::DynStorage) -> Self {
        let len = storage.len();
        Self {
            backing: ClassicStorageBacking::Range(storage),
            len,
        }
    }

    #[cfg(feature = "netcdf4")]
    pub(crate) fn len(&self) -> u64 {
        self.len
    }

    #[cfg(feature = "netcdf4")]
    pub(crate) fn initial_header_len(&self) -> usize {
        usize::try_from(self.len.min(INITIAL_HEADER_READ as u64)).unwrap_or(INITIAL_HEADER_READ)
    }

    pub(crate) fn read_range(&self, offset: u64, len: usize) -> Result<ClassicBuffer> {
        let needed = u64::try_from(len)
            .map_err(|_| Error::InvalidData("classic range length exceeds u64".to_string()))?;
        let end = offset
            .checked_add(needed)
            .ok_or_else(|| Error::InvalidData("classic byte range exceeds u64".to_string()))?;
        if end > self.len {
            return Err(Error::UnexpectedEof {
                offset,
                needed,
                available: self.len.saturating_sub(offset),
            });
        }

        match &self.backing {
            ClassicStorageBacking::Bytes(data) => {
                let start = usize::try_from(offset).map_err(|_| {
                    Error::InvalidData("classic byte offset exceeds platform usize".to_string())
                })?;
                Ok(ClassicBuffer::Bytes {
                    data: data.clone(),
                    start,
                    len,
                })
            }
            ClassicStorageBacking::Mmap(mmap) => {
                let start = usize::try_from(offset).map_err(|_| {
                    Error::InvalidData("classic byte offset exceeds platform usize".to_string())
                })?;
                Ok(ClassicBuffer::Mmap {
                    mmap: mmap.clone(),
                    start,
                    len,
                })
            }
            #[cfg(feature = "netcdf4")]
            ClassicStorageBacking::Range(storage) => {
                Ok(ClassicBuffer::Range(storage.read_range(offset, len)?))
            }
        }
    }

    #[cfg(feature = "netcdf4")]
    pub(crate) fn read_header_prefix(&self, len: usize) -> Result<ClassicBuffer> {
        let capped = len.min(usize::try_from(self.len).unwrap_or(usize::MAX));
        self.read_range(0, capped)
    }
}

impl ClassicBuffer {
    #[cfg(feature = "netcdf4")]
    pub(crate) fn len(&self) -> usize {
        match self {
            ClassicBuffer::Bytes { len, .. } | ClassicBuffer::Mmap { len, .. } => *len,
            #[cfg(feature = "netcdf4")]
            ClassicBuffer::Range(buffer) => buffer.len(),
        }
    }
}

impl AsRef<[u8]> for ClassicBuffer {
    fn as_ref(&self) -> &[u8] {
        self
    }
}

impl Deref for ClassicBuffer {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        match self {
            ClassicBuffer::Bytes { data, start, len } => &data[*start..*start + *len],
            ClassicBuffer::Mmap { mmap, start, len } => &mmap[*start..*start + *len],
            #[cfg(feature = "netcdf4")]
            ClassicBuffer::Range(buffer) => buffer.as_ref(),
        }
    }
}
