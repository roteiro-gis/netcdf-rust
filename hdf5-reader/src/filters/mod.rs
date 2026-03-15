pub mod deflate;
pub mod fletcher32;
pub mod shuffle;

use crate::error::{Error, Result};
use crate::messages::filter_pipeline::FilterDescription;

/// Standard HDF5 filter IDs.
pub const FILTER_DEFLATE: u16 = 1;
pub const FILTER_SHUFFLE: u16 = 2;
pub const FILTER_FLETCHER32: u16 = 3;
pub const FILTER_SZIP: u16 = 4;
pub const FILTER_NBIT: u16 = 5;
pub const FILTER_SCALEOFFSET: u16 = 6;

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
