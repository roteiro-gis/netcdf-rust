pub mod deflate;
pub mod fletcher32;
#[cfg(feature = "lz4")]
pub mod lz4;
pub mod nbit;
pub mod scaleoffset;
pub mod shuffle;
pub mod szip;

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
/// HDF5 registered LZ4 filter.
pub const FILTER_LZ4: u16 = 32004;

/// A user-supplied filter function.
///
/// Takes the filter description, input data, and element size, then returns
/// the decoded output.
pub type FilterFn = Box<dyn Fn(&FilterDescription, &[u8], usize) -> Result<Vec<u8>> + Send + Sync>;

enum FilterImplementation {
    Builtin,
    Custom(FilterFn),
}

/// A registry of filter implementations.
///
/// Comes pre-loaded with deflate, shuffle, and fletcher32. Users can register
/// additional filters (e.g., Blosc, LZ4, ZFP) before reading datasets.
pub struct FilterRegistry {
    filters: HashMap<u16, FilterImplementation>,
}

impl FilterRegistry {
    /// Create a new registry with the built-in filters pre-registered.
    pub fn new() -> Self {
        let mut registry = FilterRegistry {
            filters: HashMap::new(),
        };
        registry.register_builtin(FILTER_DEFLATE);
        registry.register_builtin(FILTER_SHUFFLE);
        registry.register_builtin(FILTER_FLETCHER32);
        registry.register_builtin(FILTER_SZIP);
        registry.register_builtin(FILTER_NBIT);
        registry.register_builtin(FILTER_SCALEOFFSET);
        #[cfg(feature = "lz4")]
        registry.register_builtin(FILTER_LZ4);
        registry
    }

    fn register_builtin(&mut self, id: u16) {
        self.filters.insert(id, FilterImplementation::Builtin);
    }

    /// Register a custom filter implementation for the given filter ID.
    ///
    /// Overwrites any previously registered filter with the same ID.
    pub fn register(&mut self, id: u16, f: FilterFn) {
        self.filters.insert(id, FilterImplementation::Custom(f));
    }

    /// Apply a single filter by ID.
    pub fn apply(
        &self,
        filter: &FilterDescription,
        data: &[u8],
        element_size: usize,
    ) -> Result<Vec<u8>> {
        self.apply_with_limit(filter, data, element_size, None)
    }

    /// Apply a single filter by ID, passing a maximum decoded output length to
    /// built-in filters that can enforce it while decoding.
    pub fn apply_with_limit(
        &self,
        filter: &FilterDescription,
        data: &[u8],
        element_size: usize,
        max_output_len: Option<usize>,
    ) -> Result<Vec<u8>> {
        match self.filters.get(&filter.id) {
            Some(FilterImplementation::Builtin) => {
                apply_builtin_filter_with_limit(filter, data, element_size, max_output_len)
            }
            Some(FilterImplementation::Custom(f)) => f(filter, data, element_size),
            None => Err(Error::UnsupportedFilter(format!("filter id {}", filter.id))),
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
/// If `registry` is `None`, the built-in filter set is used.
///
/// `filter_mask` is a bitmask where bit N being set means filter N should be skipped.
pub fn apply_pipeline(
    data: &[u8],
    filters: &[FilterDescription],
    filter_mask: u32,
    element_size: usize,
    registry: Option<&FilterRegistry>,
) -> Result<Vec<u8>> {
    apply_pipeline_with_limit(data, filters, filter_mask, element_size, registry, None)
}

/// Apply the filter pipeline in reverse (decompression direction) to a chunk,
/// passing a maximum decoded output length to built-in filters that support
/// bounded decompression.
pub fn apply_pipeline_with_limit(
    data: &[u8],
    filters: &[FilterDescription],
    filter_mask: u32,
    element_size: usize,
    registry: Option<&FilterRegistry>,
    max_output_len: Option<usize>,
) -> Result<Vec<u8>> {
    // Count active filters so an all-skipped pipeline can avoid the loop.
    let active_count = filters
        .iter()
        .enumerate()
        .rev()
        .filter(|(i, _)| filter_mask & (1 << i) == 0)
        .count();

    if active_count == 0 {
        validate_output_limit("filter pipeline", data.len(), max_output_len)?;
        return Ok(data.to_vec());
    }

    // For a single active filter, avoid the double-buffer loop overhead.
    if active_count == 1 {
        for (i, filter) in filters.iter().enumerate().rev() {
            if filter_mask & (1 << i) != 0 {
                continue;
            }
            let output = if let Some(reg) = registry {
                reg.apply_with_limit(filter, data, element_size, max_output_len)?
            } else {
                apply_builtin_filter_with_limit(filter, data, element_size, max_output_len)?
            };
            validate_output_limit("filter pipeline", output.len(), max_output_len)?;
            return Ok(output);
        }
    }

    // Multi-filter pipeline: the first stage reads from the borrowed input
    // slice (avoiding a copy), subsequent stages consume the previous output.
    // Each filter stage necessarily allocates (output sizes are unpredictable),
    // but we avoid the initial data.to_vec() copy.
    let mut owned: Option<Vec<u8>> = None;

    for (i, filter) in filters.iter().enumerate().rev() {
        if filter_mask & (1 << i) != 0 {
            continue;
        }

        let input: &[u8] = match &owned {
            Some(buf) => buf,
            None => data,
        };

        // A decoded intermediate may still be compressed by an earlier filter.
        // Its bound is the forward encoded-size bound of that prefix, not the
        // final chunk size (e.g. two DEFLATE filters on an eight-byte chunk).
        // A prefix with no known bound leaves the stage unbounded; the final
        // output is still checked against the limit.
        let stage_limit = max_output_len.and_then(|n| {
            filters[..i]
                .iter()
                .enumerate()
                .filter(|(j, _)| filter_mask & (1 << j) == 0)
                .try_fold(n, |n, (_, earlier)| {
                    let custom = registry.is_some_and(|reg| {
                        matches!(
                            reg.filters.get(&earlier.id),
                            Some(FilterImplementation::Custom(_))
                        )
                    });
                    if custom {
                        None
                    } else {
                        encoded_size_bound(earlier.id, n)
                    }
                })
        });
        owned = Some(if let Some(reg) = registry {
            reg.apply_with_limit(filter, input, element_size, stage_limit)?
        } else {
            apply_builtin_filter_with_limit(filter, input, element_size, stage_limit)?
        });
    }

    let output = owned.unwrap_or_else(|| data.to_vec());
    validate_output_limit("filter pipeline", output.len(), max_output_len)?;
    Ok(output)
}

/// Absolute ceiling on one built-in filter's decoded output: one byte past
/// the largest chunk HDF5 permits, so the usual `expected + 1` limit still
/// detects over-long streams at the format maximum.
const MAX_FILTER_OUTPUT: usize = crate::HDF5_MAX_CHUNK_BYTES.saturating_add(1);

fn apply_builtin_filter_with_limit(
    filter: &FilterDescription,
    data: &[u8],
    element_size: usize,
    max_output_len: Option<usize>,
) -> Result<Vec<u8>> {
    // Built-in decoders size their output from file content, so every call is
    // bounded, including callers that pass no limit.
    let limit = max_output_len.map_or(MAX_FILTER_OUTPUT, |n| n.min(MAX_FILTER_OUTPUT));
    match filter.id {
        FILTER_DEFLATE => deflate::decompress_with_limit(data, limit),
        FILTER_SHUFFLE => Ok(shuffle::unshuffle(data, element_size)),
        FILTER_FLETCHER32 => fletcher32::verify_and_strip(data),
        FILTER_SZIP => szip::decompress_with_limit(data, &filter.client_data, limit),
        FILTER_NBIT => nbit::decompress_with_limit(data, &filter.client_data, limit),
        FILTER_SCALEOFFSET => scaleoffset::decompress_with_limit(data, &filter.client_data, limit),
        #[cfg(feature = "lz4")]
        FILTER_LZ4 => lz4::decompress_with_limit(data, limit),
        id => Err(Error::UnsupportedFilter(format!("filter id {}", id))),
    }
}

/// A zero-filled filter output buffer, or an error when it cannot be allocated: a corrupted
/// chunk shape or size field must fail the read, not abort the process.
pub(crate) fn zeroed_output(len: usize, context: &str) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    out.try_reserve_exact(len).map_err(|_| {
        Error::DecompressionError(format!("{context}: cannot allocate {len} output bytes"))
    })?;
    out.resize(len, 0);
    Ok(out)
}

fn validate_output_limit(context: &str, len: usize, max_output_len: Option<usize>) -> Result<()> {
    if let Some(max_output_len) = max_output_len {
        if len > max_output_len {
            return Err(Error::DecompressionError(format!(
                "{context} decoded to {len} bytes, limit {max_output_len}"
            )));
        }
    }
    Ok(())
}

/// Forward encoded-size bound of a built-in filter applied to `n` bytes, or
/// `None` when the codec has no bound known here (unknown filters, LZ4).
fn encoded_size_bound(id: u16, n: usize) -> Option<usize> {
    Some(match id {
        // zlib's compressBound.
        FILTER_DEFLATE => n
            .saturating_add(n >> 12)
            .saturating_add(n >> 14)
            .saturating_add(n >> 25)
            .saturating_add(13),
        FILTER_FLETCHER32 => n.saturating_add(4),
        // HDF5 scale-offset header plus at most one padding byte.
        FILTER_SCALEOFFSET => n.saturating_add(22),
        FILTER_SHUFFLE | FILTER_NBIT => n,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::write::ZlibEncoder;
    use flate2::Compression;
    use std::io::Write;

    fn zlib(data: &[u8]) -> Vec<u8> {
        let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(data).unwrap();
        encoder.finish().unwrap()
    }

    #[test]
    fn intermediate_stages_are_bounded_by_the_earlier_filters() {
        // DEFLATE twice on an eight-byte chunk: the inner stream is longer
        // than the chunk.
        let raw = (-3.0f64).to_le_bytes();
        let once = zlib(&raw);
        let twice = zlib(&once);
        assert!(once.len() > raw.len());
        let filters = vec![
            FilterDescription {
                id: FILTER_DEFLATE,
                name: None,
                client_data: vec![1],
            };
            2
        ];
        for registry in [None, Some(FilterRegistry::new())] {
            let registry = registry.as_ref();
            assert_eq!(
                apply_pipeline_with_limit(&twice, &filters, 0, 8, registry, Some(8)).unwrap(),
                raw
            );
            // Second filter skipped (mask bit 1): one DEFLATE stream remains.
            assert_eq!(
                apply_pipeline_with_limit(&once, &filters, 0b10, 8, registry, Some(8)).unwrap(),
                raw
            );
        }

        assert_eq!(encoded_size_bound(FILTER_SCALEOFFSET, 256), Some(278));
        assert_eq!(
            encoded_size_bound(FILTER_DEFLATE, usize::MAX),
            Some(usize::MAX)
        );
        assert_eq!(encoded_size_bound(50000, 256), None);
    }

    #[test]
    fn filter_registry_default() {
        let registry = FilterRegistry::new();
        // Built-in filters should be registered
        assert!(registry.filters.contains_key(&FILTER_DEFLATE));
        assert!(registry.filters.contains_key(&FILTER_SHUFFLE));
        assert!(registry.filters.contains_key(&FILTER_FLETCHER32));
        assert!(registry.filters.contains_key(&FILTER_NBIT));
        assert!(registry.filters.contains_key(&FILTER_SCALEOFFSET));
    }

    #[test]
    fn filter_registry_custom() {
        let mut registry = FilterRegistry::new();
        // Register a no-op custom filter
        registry.register(32000, Box::new(|_, data, _| Ok(data.to_vec())));
        let filter = FilterDescription {
            id: 32000,
            name: None,
            client_data: Vec::new(),
        };
        let result = registry.apply(&filter, &[1, 2, 3], 1).unwrap();
        assert_eq!(result, vec![1, 2, 3]);
    }

    #[test]
    fn filter_registry_unknown() {
        let registry = FilterRegistry::new();
        let filter = FilterDescription {
            id: 9999,
            name: None,
            client_data: Vec::new(),
        };
        let err = registry.apply(&filter, &[1, 2, 3], 1).unwrap_err();
        assert!(matches!(err, Error::UnsupportedFilter(_)));
    }

    #[test]
    fn apply_pipeline_with_limit_caps_registry_deflate_output() {
        let original = vec![0u8; 4096];
        let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&original).unwrap();
        let compressed = encoder.finish().unwrap();
        let filter = FilterDescription {
            id: FILTER_DEFLATE,
            name: None,
            client_data: vec![6],
        };
        let registry = FilterRegistry::new();

        let decoded =
            apply_pipeline_with_limit(&compressed, &[filter], 0, 1, Some(&registry), Some(65))
                .unwrap();

        assert_eq!(decoded.len(), 65);
        assert_eq!(decoded, original[..65]);
    }

    #[test]
    fn apply_pipeline_with_limit_rejects_oversized_final_output() {
        let filter = FilterDescription {
            id: FILTER_SHUFFLE,
            name: None,
            client_data: Vec::new(),
        };

        let err =
            apply_pipeline_with_limit(&[1, 2, 3, 4], &[filter], 0, 1, None, Some(3)).unwrap_err();

        assert!(err.to_string().contains("limit 3"));
    }
}
