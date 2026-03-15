pub mod deflate;
pub mod fletcher32;
pub mod shuffle;

use std::collections::HashMap;

use crate::error::{Error, Result};
use crate::messages::filter_pipeline::FilterDescription;

/// Standard HDF5 filter IDs.
pub const FILTER_DEFLATE: u16 = 1;
pub const FILTER_SHUFFLE: u16 = 2;
pub const FILTER_FLETCHER32: u16 = 3;
pub const FILTER_SZIP: u16 = 4;
pub const FILTER_NBIT: u16 = 5;
pub const FILTER_SCALEOFFSET: u16 = 6;

/// A user-supplied filter function.
///
/// Takes the input data and element size, returns the decoded output.
pub type FilterFn = Box<dyn Fn(&[u8], usize) -> Result<Vec<u8>> + Send + Sync>;

/// A registry of filter implementations.
///
/// Comes pre-loaded with deflate, shuffle, and fletcher32. Users can register
/// additional filters (e.g., Blosc, LZ4, ZFP) before reading datasets.
pub struct FilterRegistry {
    filters: HashMap<u16, FilterFn>,
}

impl FilterRegistry {
    /// Create a new registry with the built-in filters pre-registered.
    pub fn new() -> Self {
        let mut registry = FilterRegistry {
            filters: HashMap::new(),
        };
        registry.register(FILTER_DEFLATE, Box::new(|data, _| deflate::decompress(data)));
        registry.register(
            FILTER_SHUFFLE,
            Box::new(|data, elem_size| Ok(shuffle::unshuffle(data, elem_size))),
        );
        registry.register(
            FILTER_FLETCHER32,
            Box::new(|data, _| fletcher32::verify_and_strip(data)),
        );
        registry
    }

    /// Register a custom filter implementation for the given filter ID.
    ///
    /// Overwrites any previously registered filter with the same ID.
    pub fn register(&mut self, id: u16, f: FilterFn) {
        self.filters.insert(id, f);
    }

    /// Apply a single filter by ID.
    pub fn apply(&self, id: u16, data: &[u8], element_size: usize) -> Result<Vec<u8>> {
        match self.filters.get(&id) {
            Some(f) => f(data, element_size),
            None => Err(Error::UnsupportedFilter(format!("filter id {}", id))),
        }
    }
}

impl Default for FilterRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Apply the filter pipeline in reverse (decompression direction) to a chunk.
///
/// HDF5 stores filters in the order they were applied during writing.
/// On read, we apply them in reverse order.
///
/// `filter_mask` is a bitmask where bit N being set means filter N should be skipped.
pub fn apply_pipeline(
    data: &[u8],
    filters: &[FilterDescription],
    filter_mask: u32,
    element_size: usize,
) -> Result<Vec<u8>> {
    let mut buf = data.to_vec();

    // Apply filters in reverse order
    for (i, filter) in filters.iter().enumerate().rev() {
        // Check if this filter is masked out
        if filter_mask & (1 << i) != 0 {
            continue;
        }

        buf = apply_filter(filter, &buf, element_size)?;
    }

    Ok(buf)
}

fn apply_filter(filter: &FilterDescription, data: &[u8], element_size: usize) -> Result<Vec<u8>> {
    match filter.id {
        FILTER_DEFLATE => deflate::decompress(data),
        FILTER_SHUFFLE => Ok(shuffle::unshuffle(data, element_size)),
        FILTER_FLETCHER32 => fletcher32::verify_and_strip(data),
        FILTER_SZIP => Err(Error::UnsupportedFilter("szip".into())),
        FILTER_NBIT => Err(Error::UnsupportedFilter("nbit".into())),
        FILTER_SCALEOFFSET => Err(Error::UnsupportedFilter("scaleoffset".into())),
        id => Err(Error::UnsupportedFilter(format!("filter id {}", id))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_registry_default() {
        let registry = FilterRegistry::new();
        // Built-in filters should be registered
        assert!(registry.filters.contains_key(&FILTER_DEFLATE));
        assert!(registry.filters.contains_key(&FILTER_SHUFFLE));
        assert!(registry.filters.contains_key(&FILTER_FLETCHER32));
    }

    #[test]
    fn test_filter_registry_custom() {
        let mut registry = FilterRegistry::new();
        // Register a no-op custom filter
        registry.register(32000, Box::new(|data, _| Ok(data.to_vec())));
        let result = registry.apply(32000, &[1, 2, 3], 1).unwrap();
        assert_eq!(result, vec![1, 2, 3]);
    }

    #[test]
    fn test_filter_registry_unknown() {
        let registry = FilterRegistry::new();
        let err = registry.apply(9999, &[1, 2, 3], 1).unwrap_err();
        assert!(matches!(err, Error::UnsupportedFilter(_)));
    }
}
