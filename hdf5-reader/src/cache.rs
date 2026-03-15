use std::sync::{Arc, Mutex};

use lru::LruCache;
use std::num::NonZeroUsize;

/// Key for the chunk cache: (dataset object header address, chunk offset tuple).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ChunkKey {
    pub dataset_addr: u64,
    pub chunk_offsets: Vec<u64>,
}

/// LRU cache for decompressed chunks.
///
/// Thread-safe via `Mutex`. Values are `Arc<Vec<u8>>` so multiple readers
/// can share the same decompressed chunk data.
pub struct ChunkCache {
    inner: Mutex<LruCache<ChunkKey, Arc<Vec<u8>>>>,
    max_bytes: usize,
    current_bytes: Mutex<usize>,
}

impl ChunkCache {
    /// Create a new chunk cache.
    ///
    /// - `max_bytes`: maximum total bytes of decompressed data to cache (default 64 MiB)
    /// - `max_slots`: maximum number of entries (default 521)
    pub fn new(max_bytes: usize, max_slots: usize) -> Self {
        let slots = NonZeroUsize::new(max_slots).unwrap_or(NonZeroUsize::new(521).unwrap());
        ChunkCache {
            inner: Mutex::new(LruCache::new(slots)),
            max_bytes,
            current_bytes: Mutex::new(0),
        }
    }

    /// Get a cached chunk, if present.
    pub fn get(&self, key: &ChunkKey) -> Option<Arc<Vec<u8>>> {
        let mut cache = self.inner.lock().unwrap();
        cache.get(key).cloned()
    }

    /// Insert a chunk into the cache. Evicts LRU entries if over capacity.
    pub fn insert(&self, key: ChunkKey, data: Vec<u8>) -> Arc<Vec<u8>> {
        let data_len = data.len();
        let arc = Arc::new(data);

        let mut cache = self.inner.lock().unwrap();
        let mut current = self.current_bytes.lock().unwrap();

        // Evict until we have room
        while *current + data_len > self.max_bytes && !cache.is_empty() {
            if let Some((_, evicted)) = cache.pop_lru() {
                *current = current.saturating_sub(evicted.len());
            }
        }

        *current += data_len;
        cache.put(key, arc.clone());

        arc
    }
}

impl Default for ChunkCache {
    fn default() -> Self {
        Self::new(64 * 1024 * 1024, 521)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_insert_and_get() {
        let cache = ChunkCache::new(1024, 10);
        let key = ChunkKey {
            dataset_addr: 100,
            chunk_offsets: vec![0, 0],
        };
        cache.insert(key.clone(), vec![1, 2, 3]);
        let val = cache.get(&key).unwrap();
        assert_eq!(&**val, &[1, 2, 3]);
    }

    #[test]
    fn test_cache_eviction() {
        let cache = ChunkCache::new(10, 10); // 10 bytes max
        for i in 0..5 {
            let key = ChunkKey {
                dataset_addr: 100,
                chunk_offsets: vec![i],
            };
            cache.insert(key, vec![0; 4]); // 4 bytes each
        }
        // Should have evicted older entries to stay under 10 bytes
        // At most 2 entries of 4 bytes each = 8 bytes
        let first_key = ChunkKey {
            dataset_addr: 100,
            chunk_offsets: vec![0],
        };
        assert!(cache.get(&first_key).is_none()); // should be evicted
    }
}
