//! HDF5 Global Heap Collection (GCOL).
//!
//! Global heaps store variable-length data such as variable-length strings
//! and VL arrays. Each collection is a contiguous block in the file that
//! contains multiple heap objects. Objects are referenced by a global heap
//! ID that encodes the collection address and the object index.
//!
//! An object with index 0 marks the free space sentinel and terminates
//! parsing of the collection.

use std::sync::Arc;

use lru::LruCache;

use crate::error::{Error, Result};
use crate::io::Cursor;
use crate::storage::Storage;

/// Signature bytes for a Global Heap Collection: ASCII `GCOL`.
const GCOL_SIGNATURE: [u8; 4] = *b"GCOL";

/// Bytes of object data a [`GlobalHeapCache`] holds before it evicts.
const GLOBAL_HEAP_CACHE_MAX_BYTES: usize = 64 * 1024 * 1024;

/// A single object within a global heap collection.
#[derive(Debug, Clone)]
pub struct GlobalHeapObject {
    /// Heap object index (1-based; index 0 is the free-space sentinel).
    pub index: u16,
    /// Reference count.
    pub reference_count: u16,
    /// Raw object data.
    pub data: Vec<u8>,
}

/// A parsed global heap collection containing zero or more heap objects.
#[derive(Debug, Clone)]
pub struct GlobalHeapCollection {
    /// The heap objects in this collection.
    pub objects: Vec<GlobalHeapObject>,
}

impl GlobalHeapCollection {
    /// Parse a global heap collection at the current cursor position.
    ///
    /// Format:
    /// - Signature: `GCOL` (4 bytes)
    /// - Version: 1 (1 byte)
    /// - Reserved: 3 bytes
    /// - Collection size (`length_size` bytes) — total size including header
    /// - Then global heap objects until the collection is exhausted.
    ///
    /// Each global heap object:
    /// - Heap object index (u16 LE)
    /// - Reference count (u16 LE)
    /// - Reserved (4 bytes)
    /// - Object size (`length_size` bytes)
    /// - Object data (padded to 8-byte boundary)
    /// - An index of 0 signals free space / end of objects.
    pub fn parse(cursor: &mut Cursor, _offset_size: u8, length_size: u8) -> Result<Self> {
        let header_start = cursor.position();

        let sig = cursor.read_bytes(4)?;
        if sig != GCOL_SIGNATURE {
            return Err(Error::InvalidGlobalHeapSignature);
        }

        let version = cursor.read_u8()?;
        if version != 1 {
            return Err(Error::UnsupportedGlobalHeapVersion(version));
        }

        // Reserved 3 bytes
        cursor.skip(3)?;

        let collection_size = cursor.read_length(length_size)?;

        // The collection_size includes the header we just read. Calculate the
        // end boundary so we don't read past it.
        let collection_end = header_start
            .checked_add(collection_size)
            .ok_or_else(|| Error::InvalidData("global heap collection size overflows".into()))?;

        let mut objects = Vec::new();

        loop {
            // Check if we have enough room for at least an object header
            // (2 + 2 + 4 + length_size bytes minimum).
            let min_obj_header = 8 + length_size as u64;
            if cursor.position().saturating_add(min_obj_header) > collection_end {
                break;
            }

            let index = cursor.read_u16_le()?;

            // Index 0 = free space sentinel — stop parsing.
            if index == 0 {
                break;
            }

            let reference_count = cursor.read_u16_le()?;
            // Reserved 4 bytes
            cursor.skip(4)?;
            let obj_size = cursor.read_length(length_size)?;

            // Guard against reading past the collection.
            if cursor.position().saturating_add(obj_size) > collection_end {
                return Err(Error::UnexpectedEof {
                    offset: cursor.position(),
                    needed: obj_size,
                    available: collection_end.saturating_sub(cursor.position()),
                });
            }

            let obj_len = usize::try_from(obj_size).map_err(|_| {
                Error::InvalidData("global heap object size exceeds platform usize capacity".into())
            })?;
            let data = cursor.read_bytes(obj_len)?.to_vec();

            // Object data is padded to an 8-byte boundary.
            let padding = obj_size.wrapping_neg() & 7;
            if padding > 0 && cursor.position().saturating_add(padding) <= collection_end {
                cursor.skip(padding as usize)?;
            }

            objects.push(GlobalHeapObject {
                index,
                reference_count,
                data,
            });
        }

        Ok(GlobalHeapCollection { objects })
    }

    /// Parse a global heap collection from random-access storage.
    pub fn parse_at_storage(
        storage: &dyn Storage,
        address: u64,
        offset_size: u8,
        length_size: u8,
    ) -> Result<Self> {
        let header_len = 4 + 1 + 3 + usize::from(length_size);
        let header = storage.read_range(address, header_len)?;
        let mut cursor = Cursor::new(header.as_ref());
        let sig = cursor.read_bytes(4)?;
        if sig != GCOL_SIGNATURE {
            return Err(Error::InvalidGlobalHeapSignature);
        }

        let version = cursor.read_u8()?;
        if version != 1 {
            return Err(Error::UnsupportedGlobalHeapVersion(version));
        }

        cursor.skip(3)?;
        let collection_size = cursor.read_length(length_size)?;
        let collection_len = usize::try_from(collection_size).map_err(|_| {
            Error::InvalidData("global heap collection exceeds platform usize capacity".into())
        })?;
        let bytes = storage.read_range(address, collection_len)?;
        let mut full_cursor = Cursor::new(bytes.as_ref());
        Self::parse(&mut full_cursor, offset_size, length_size)
    }

    /// Look up an object by its index within this collection.
    pub fn get_object(&self, index: u16) -> Option<&GlobalHeapObject> {
        // Writers number objects from 1 in order, so the position is usually the index.
        usize::from(index)
            .checked_sub(1)
            .and_then(|position| self.objects.get(position))
            .filter(|object| object.index == index)
            .or_else(|| self.objects.iter().find(|object| object.index == index))
    }

    fn data_bytes(&self) -> usize {
        self.objects.iter().map(|object| object.data.len()).sum()
    }
}

/// Recently parsed collections by address, so that reading the elements of a
/// variable-length dataset parses each collection once rather than once per element.
pub(crate) struct GlobalHeapCache {
    collections: LruCache<u64, Arc<GlobalHeapCollection>>,
    bytes: usize,
    max_bytes: usize,
}

impl Default for GlobalHeapCache {
    fn default() -> Self {
        Self::with_max_bytes(GLOBAL_HEAP_CACHE_MAX_BYTES)
    }
}

impl GlobalHeapCache {
    fn with_max_bytes(max_bytes: usize) -> Self {
        Self {
            collections: LruCache::unbounded(),
            bytes: 0,
            max_bytes,
        }
    }

    pub(crate) fn get(&mut self, address: u64) -> Option<Arc<GlobalHeapCollection>> {
        self.collections.get(&address).cloned()
    }

    /// Cache `collection`, evicting the least recently used others once over budget. The
    /// newest collection always stays, however large, because the next element of the read
    /// that loaded it most likely needs it again.
    pub(crate) fn insert(&mut self, address: u64, collection: Arc<GlobalHeapCollection>) {
        self.bytes += collection.data_bytes();
        if let Some(replaced) = self.collections.put(address, collection) {
            self.bytes -= replaced.data_bytes();
        }
        while self.bytes > self.max_bytes && self.collections.len() > 1 {
            if let Some((_, evicted)) = self.collections.pop_lru() {
                self.bytes -= evicted.data_bytes();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal global heap collection with the given objects.
    /// Each object is (index, ref_count, data).
    fn build_gcol(objects: &[(u16, u16, &[u8])], length_size: u8) -> Vec<u8> {
        let mut body = Vec::new();
        for &(index, ref_count, data) in objects {
            body.extend_from_slice(&index.to_le_bytes());
            body.extend_from_slice(&ref_count.to_le_bytes());
            body.extend_from_slice(&[0u8; 4]); // reserved
            match length_size {
                4 => body.extend_from_slice(&(data.len() as u32).to_le_bytes()),
                8 => body.extend_from_slice(&(data.len() as u64).to_le_bytes()),
                _ => panic!("test only supports 4/8"),
            }
            body.extend_from_slice(data);
            // Pad to 8-byte boundary
            let padded = (data.len() + 7) & !7;
            body.resize(body.len() + (padded - data.len()), 0);
        }
        // Free space sentinel (index 0)
        body.extend_from_slice(&0u16.to_le_bytes());

        // Build full collection
        let header_size = 4 + 1 + 3 + length_size as usize; // sig + ver + reserved + size
        let collection_size = header_size + body.len();

        let mut buf = Vec::new();
        buf.extend_from_slice(b"GCOL");
        buf.push(1); // version
        buf.extend_from_slice(&[0, 0, 0]); // reserved
        match length_size {
            4 => buf.extend_from_slice(&(collection_size as u32).to_le_bytes()),
            8 => buf.extend_from_slice(&(collection_size as u64).to_le_bytes()),
            _ => panic!("test only supports 4/8"),
        }
        buf.extend(body);
        buf
    }

    fn collection(objects: &[(u16, &[u8])]) -> Arc<GlobalHeapCollection> {
        Arc::new(GlobalHeapCollection {
            objects: objects
                .iter()
                .map(|&(index, data)| GlobalHeapObject {
                    index,
                    reference_count: 1,
                    data: data.to_vec(),
                })
                .collect(),
        })
    }

    #[test]
    fn get_object_finds_objects_stored_out_of_order() {
        let collection = collection(&[(2, b"b"), (1, b"a"), (9, b"z")]);
        assert_eq!(collection.get_object(1).unwrap().data, b"a");
        assert_eq!(collection.get_object(2).unwrap().data, b"b");
        assert_eq!(collection.get_object(9).unwrap().data, b"z");
        assert!(collection.get_object(0).is_none());
        assert!(collection.get_object(3).is_none());
    }

    #[test]
    fn cache_evicts_least_recently_used_collections_over_budget() {
        let mut cache = GlobalHeapCache::with_max_bytes(10);
        cache.insert(1, collection(&[(1, &[0; 4])]));
        cache.insert(2, collection(&[(1, &[0; 4])]));
        assert!(cache.get(1).is_some());

        cache.insert(3, collection(&[(1, &[0; 4])]));
        assert!(cache.get(2).is_none());
        assert!(cache.get(1).is_some());
        assert!(cache.get(3).is_some());
    }

    #[test]
    fn cache_keeps_a_collection_larger_than_its_budget() {
        let mut cache = GlobalHeapCache::with_max_bytes(10);
        cache.insert(1, collection(&[(1, &[0; 4])]));
        cache.insert(2, collection(&[(1, &[0; 64])]));
        assert!(cache.get(1).is_none());
        assert!(cache.get(2).is_some());
    }

    #[test]
    fn replacing_a_cached_collection_does_not_leak_its_bytes() {
        let mut cache = GlobalHeapCache::with_max_bytes(10);
        cache.insert(1, collection(&[(1, &[0; 8])]));
        cache.insert(1, collection(&[(1, &[0; 8])]));
        cache.insert(2, collection(&[(1, &[0; 2])]));
        assert!(cache.get(1).is_some());
        assert!(cache.get(2).is_some());
    }

    #[test]
    fn sizes_that_overflow_are_errors() {
        // Collection size u64::MAX at offset 8: its end does not fit in u64.
        let mut data = vec![0u8; 8];
        let mut gcol = build_gcol(&[(1, 1, b"x")], 8);
        gcol[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
        data.extend(gcol);
        let mut cursor = Cursor::new(&data);
        cursor.set_position(8);
        assert!(matches!(
            GlobalHeapCollection::parse(&mut cursor, 8, 8),
            Err(Error::InvalidData(_))
        ));

        // Object size u64::MAX: past the end of the collection.
        let mut data = build_gcol(&[(1, 1, b"x")], 8);
        data[24..32].copy_from_slice(&u64::MAX.to_le_bytes());
        let mut cursor = Cursor::new(&data);
        assert!(matches!(
            GlobalHeapCollection::parse(&mut cursor, 8, 8),
            Err(Error::UnexpectedEof { .. })
        ));
    }

    #[test]
    fn parse_empty_collection() {
        let data = build_gcol(&[], 8);
        let mut cursor = Cursor::new(&data);
        let col = GlobalHeapCollection::parse(&mut cursor, 8, 8).unwrap();
        assert!(col.objects.is_empty());
    }

    #[test]
    fn parse_single_object() {
        let obj_data = b"hello world";
        let data = build_gcol(&[(1, 1, obj_data)], 8);
        let mut cursor = Cursor::new(&data);
        let col = GlobalHeapCollection::parse(&mut cursor, 8, 8).unwrap();

        assert_eq!(col.objects.len(), 1);
        assert_eq!(col.objects[0].index, 1);
        assert_eq!(col.objects[0].reference_count, 1);
        assert_eq!(col.objects[0].data, obj_data);
    }

    #[test]
    fn parse_multiple_objects() {
        let data = build_gcol(
            &[
                (1, 1, b"alpha"),
                (2, 3, b"beta"),
                (5, 0, b"gamma123"), // 8 bytes, no padding needed
            ],
            8,
        );
        let mut cursor = Cursor::new(&data);
        let col = GlobalHeapCollection::parse(&mut cursor, 8, 8).unwrap();

        assert_eq!(col.objects.len(), 3);

        let obj1 = col.get_object(1).unwrap();
        assert_eq!(obj1.data, b"alpha");
        assert_eq!(obj1.reference_count, 1);

        let obj2 = col.get_object(2).unwrap();
        assert_eq!(obj2.data, b"beta");
        assert_eq!(obj2.reference_count, 3);

        let obj5 = col.get_object(5).unwrap();
        assert_eq!(obj5.data, b"gamma123");

        assert!(col.get_object(99).is_none());
    }

    #[test]
    fn parse_4byte_lengths() {
        let data = build_gcol(&[(1, 2, b"test")], 4);
        let mut cursor = Cursor::new(&data);
        let col = GlobalHeapCollection::parse(&mut cursor, 4, 4).unwrap();

        assert_eq!(col.objects.len(), 1);
        assert_eq!(col.objects[0].data, b"test");
    }

    #[test]
    fn bad_signature() {
        let mut data = build_gcol(&[], 8);
        data[0] = b'X';
        let mut cursor = Cursor::new(&data);
        assert!(matches!(
            GlobalHeapCollection::parse(&mut cursor, 8, 8),
            Err(Error::InvalidGlobalHeapSignature)
        ));
    }

    #[test]
    fn bad_version() {
        let mut data = build_gcol(&[], 8);
        data[4] = 2; // version 2
        let mut cursor = Cursor::new(&data);
        assert!(matches!(
            GlobalHeapCollection::parse(&mut cursor, 8, 8),
            Err(Error::UnsupportedGlobalHeapVersion(2))
        ));
    }
}
