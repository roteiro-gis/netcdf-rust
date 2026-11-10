//! Chunk indexing — resolves chunk locations from various storage strategies.
//!
//! Supports all six HDF5 chunk indexing types:
//! - V1 B-tree chunk indexing (btree_v1 type 1) — dispatched externally
//! - V2 B-tree chunk indexing (btree_v2 types 10 and 11)
//! - Single chunk indexing
//! - Implicit chunk indexing
//! - Fixed array indexing (fixed_array module)
//! - Extensible array indexing (extensible_array module)

use crate::error::{Error, Result};
use crate::storage::Storage;

/// A resolved chunk location within the file.
#[derive(Debug, Clone)]
pub struct ChunkEntry {
    /// Absolute file address of the chunk data.
    pub address: u64,
    /// Size of the chunk data in bytes (after filtering, i.e., on-disk size).
    pub size: u64,
    /// Filter mask — each bit indicates whether the corresponding filter
    /// in the pipeline was skipped (1 = skipped).
    pub filter_mask: u32,
    /// Chunk offsets within the dataset (one per dimension).
    pub offsets: Vec<u64>,
}

/// The chunks an index walk collects: those inside `bounds` (if given) and
/// in `set` (if given).
#[derive(Clone, Copy, Default)]
pub(crate) struct ChunkQuery<'a> {
    pub(crate) bounds: Option<(&'a [u64], &'a [u64])>,
    pub(crate) set: Option<&'a ChunkSet>,
}

impl ChunkQuery<'_> {
    pub(crate) fn matches(&self, offsets: &[u64], chunk_dims: &[u32]) -> bool {
        chunk_overlaps_bounds(offsets, chunk_dims, self.bounds)
            && self
                .set
                .is_none_or(|set| set.contains_offsets(offsets, chunk_dims))
    }

    /// Whether a B-tree child whose keys lie between `low` and `high` may
    /// hold a collected chunk. Only a set prunes children.
    pub(crate) fn may_contain_between(
        &self,
        low: Option<&[u64]>,
        high: Option<&[u64]>,
        chunk_dims: &[u32],
    ) -> bool {
        self.set
            .is_none_or(|set| set.may_contain_between(low, high, chunk_dims))
    }
}

/// A sparse chunk selection: the cross product of a sorted, duplicate-free
/// list of chunk indices per dimension.
///
/// Index walks given a set look up only its chunks, so their work scales with
/// the number of selected chunks rather than with the span between them.
#[derive(Debug, Clone)]
pub(crate) struct ChunkSet {
    dims: Vec<Vec<u64>>,
}

impl ChunkSet {
    pub(crate) fn new(mut dims: Vec<Vec<u64>>) -> Self {
        for indices in &mut dims {
            indices.sort_unstable();
            indices.dedup();
        }
        ChunkSet { dims }
    }

    pub(crate) fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Whether the chunk at `chunk_indices` (at least `rank` of them) is selected.
    pub(crate) fn contains(&self, chunk_indices: &[u64]) -> bool {
        chunk_indices.len() >= self.dims.len()
            && self
                .dims
                .iter()
                .zip(chunk_indices)
                .all(|(indices, index)| indices.binary_search(index).is_ok())
    }

    /// Whether the chunk with element offsets `offsets` is selected.
    pub(crate) fn contains_offsets(&self, offsets: &[u64], chunk_dims: &[u32]) -> bool {
        offsets.len() >= self.dims.len()
            && chunk_dims.len() >= self.dims.len()
            && self.contains(&scaled_offsets(&offsets[..self.dims.len()], chunk_dims))
    }

    /// The selected chunks in row-major (lexicographic) order.
    pub(crate) fn chunks(&self) -> impl Iterator<Item = Vec<u64>> + '_ {
        let mut next = if self.dims.iter().any(Vec::is_empty) {
            None
        } else {
            Some(vec![0usize; self.dims.len()])
        };
        std::iter::from_fn(move || {
            let positions = next.as_mut()?;
            let chunk = positions
                .iter()
                .zip(&self.dims)
                .map(|(&pos, indices)| indices[pos])
                .collect();
            let mut advanced = false;
            for dim in (0..positions.len()).rev() {
                if positions[dim] + 1 < self.dims[dim].len() {
                    positions[dim] += 1;
                    positions[dim + 1..].fill(0);
                    advanced = true;
                    break;
                }
            }
            if !advanced {
                next = None;
            }
            Some(chunk)
        })
    }

    /// The first selected chunk at or after `low` in lexicographic order.
    fn first_at_or_after(&self, low: &[u64]) -> Option<Vec<u64>> {
        let n = self.dims.len();
        if n == 0 || self.dims.iter().any(Vec::is_empty) {
            return None;
        }
        // Length of the prefix of `low` that is selected dimension by dimension.
        let prefix = self
            .dims
            .iter()
            .zip(low)
            .take_while(|(indices, index)| indices.binary_search(index).is_ok())
            .count();
        if prefix == n {
            return Some(low[..n].to_vec());
        }
        // Keep `low[..dim]`, raise dimension `dim` and take the smallest
        // selected index in every later dimension.
        for dim in (0..=prefix).rev() {
            let indices = &self.dims[dim];
            let pos = if dim == prefix {
                indices.partition_point(|&index| index < low[dim])
            } else {
                indices.partition_point(|&index| index <= low[dim])
            };
            if let Some(&index) = indices.get(pos) {
                let mut chunk = low[..dim].to_vec();
                chunk.push(index);
                chunk.extend(self.dims[dim + 1..].iter().map(|indices| indices[0]));
                return Some(chunk);
            }
        }
        None
    }

    /// Whether a selected chunk may lie between the chunks at element
    /// offsets `low` and `high` (both inclusive, lexicographic; `None` is
    /// unbounded), as between the keys of a B-tree child. Offsets past the
    /// set's rank are ignored.
    pub(crate) fn may_contain_between(
        &self,
        low: Option<&[u64]>,
        high: Option<&[u64]>,
        chunk_dims: &[u32],
    ) -> bool {
        let n = self.dims.len();
        let scale = |offsets: &[u64]| {
            (offsets.len() >= n && chunk_dims.len() >= n)
                .then(|| scaled_offsets(&offsets[..n], chunk_dims))
        };
        let low = match low.map(scale) {
            Some(None) => return true,
            Some(Some(low)) => low,
            None => vec![0; n],
        };
        let Some(first) = self.first_at_or_after(&low) else {
            return false;
        };
        match high.map(scale) {
            Some(None) | None => true,
            Some(Some(high)) => first <= high,
        }
    }
}

fn scaled_offsets(offsets: &[u64], chunk_dims: &[u32]) -> Vec<u64> {
    offsets
        .iter()
        .zip(chunk_dims)
        .map(|(&offset, &dim)| offset / u64::from(dim.max(1)))
        .collect()
}

fn chunk_overlaps_bounds(
    offsets: &[u64],
    chunk_dims: &[u32],
    chunk_bounds: Option<(&[u64], &[u64])>,
) -> bool {
    let Some((first_chunk, last_chunk)) = chunk_bounds else {
        return true;
    };

    offsets.iter().enumerate().all(|(dim, offset)| {
        let chunk_index = *offset / u64::from(chunk_dims[dim]);
        chunk_index >= first_chunk[dim] && chunk_index <= last_chunk[dim]
    })
}

fn checked_mul_u64(lhs: u64, rhs: u64, context: &str) -> Result<u64> {
    lhs.checked_mul(rhs)
        .ok_or_else(|| Error::InvalidData(format!("{context} overflows u64")))
}

fn checked_add_u64(lhs: u64, rhs: u64, context: &str) -> Result<u64> {
    lhs.checked_add(rhs)
        .ok_or_else(|| Error::InvalidData(format!("{context} overflows u64")))
}

fn checked_usize(value: u64, context: &str) -> Result<usize> {
    usize::try_from(value).map_err(|_| {
        Error::InvalidData(format!(
            "{context} value {value} exceeds platform usize capacity"
        ))
    })
}

fn chunk_linear_index(chunk_indices: &[u64], chunks_per_dim: &[u64]) -> Result<u64> {
    let mut linear = 0u64;
    for (dim, chunk_index) in chunk_indices.iter().enumerate() {
        linear = checked_mul_u64(linear, chunks_per_dim[dim], "implicit chunk linear index")?;
        linear = checked_add_u64(linear, *chunk_index, "implicit chunk linear index")?;
    }
    Ok(linear)
}

/// Collect chunk entries from a B-tree v2 chunk index.
pub fn collect_v2_chunk_entries(
    data: &[u8],
    btree_address: u64,
    offset_size: u8,
    length_size: u8,
    ndim: u32,
    chunk_dims: &[u32],
    chunk_bounds: Option<(&[u64], &[u64])>,
) -> Result<Vec<ChunkEntry>> {
    let mut cursor = crate::io::Cursor::new(data);
    cursor.set_position(btree_address);
    let header = crate::btree_v2::BTreeV2Header::parse(&mut cursor, offset_size, length_size)?;

    let records = crate::btree_v2::collect_btree_v2_records(
        data,
        &header,
        offset_size,
        length_size,
        Some(ndim),
        chunk_dims,
        chunk_bounds,
    )?;

    let mut entries = Vec::with_capacity(records.len());
    for record in records {
        match record {
            crate::btree_v2::BTreeV2Record::ChunkedNonFiltered { address, offsets }
                if chunk_overlaps_bounds(&offsets, chunk_dims, chunk_bounds) =>
            {
                entries.push(ChunkEntry {
                    address,
                    size: 0, // caller must compute from chunk dims * elem_size
                    filter_mask: 0,
                    offsets,
                });
            }
            crate::btree_v2::BTreeV2Record::ChunkedFiltered {
                address,
                chunk_size,
                filter_mask,
                offsets,
            } if chunk_overlaps_bounds(&offsets, chunk_dims, chunk_bounds) => {
                entries.push(ChunkEntry {
                    address,
                    size: chunk_size,
                    filter_mask,
                    offsets,
                });
            }
            _ => {
                // Skip non-chunk records
            }
        }
    }

    Ok(entries)
}

/// Collect chunk entries from a B-tree v2 chunk index using random-access storage.
pub fn collect_v2_chunk_entries_storage(
    storage: &dyn Storage,
    btree_address: u64,
    offset_size: u8,
    length_size: u8,
    ndim: u32,
    chunk_dims: &[u32],
    chunk_bounds: Option<(&[u64], &[u64])>,
) -> Result<Vec<ChunkEntry>> {
    collect_v2_chunk_entries_query_storage(
        storage,
        btree_address,
        offset_size,
        length_size,
        ndim,
        chunk_dims,
        ChunkQuery {
            bounds: chunk_bounds,
            set: None,
        },
    )
}

/// Collect the entries of the chunks in `set` from a B-tree v2 chunk index,
/// entering only the nodes whose key range may hold a selected chunk.
pub(crate) fn collect_v2_chunk_entries_in_set_storage(
    storage: &dyn Storage,
    btree_address: u64,
    offset_size: u8,
    length_size: u8,
    ndim: u32,
    chunk_dims: &[u32],
    set: &ChunkSet,
) -> Result<Vec<ChunkEntry>> {
    collect_v2_chunk_entries_query_storage(
        storage,
        btree_address,
        offset_size,
        length_size,
        ndim,
        chunk_dims,
        ChunkQuery {
            bounds: None,
            set: Some(set),
        },
    )
}

fn collect_v2_chunk_entries_query_storage(
    storage: &dyn Storage,
    btree_address: u64,
    offset_size: u8,
    length_size: u8,
    ndim: u32,
    chunk_dims: &[u32],
    query: ChunkQuery<'_>,
) -> Result<Vec<ChunkEntry>> {
    let header = crate::btree_v2::BTreeV2Header::parse_at_storage(
        storage,
        btree_address,
        offset_size,
        length_size,
    )?;
    let records = crate::btree_v2::collect_btree_v2_records_query_storage(
        storage,
        &header,
        offset_size,
        length_size,
        Some(ndim),
        chunk_dims,
        query,
    )?;

    let mut entries = Vec::with_capacity(records.len());
    for record in records {
        match record {
            crate::btree_v2::BTreeV2Record::ChunkedNonFiltered { address, offsets }
                if query.matches(&offsets, chunk_dims) =>
            {
                entries.push(ChunkEntry {
                    address,
                    size: 0,
                    filter_mask: 0,
                    offsets,
                });
            }
            crate::btree_v2::BTreeV2Record::ChunkedFiltered {
                address,
                chunk_size,
                filter_mask,
                offsets,
            } if query.matches(&offsets, chunk_dims) => {
                entries.push(ChunkEntry {
                    address,
                    size: chunk_size,
                    filter_mask,
                    offsets,
                });
            }
            _ => {}
        }
    }

    Ok(entries)
}

/// Collect chunk entries for implicit indexing.
///
/// Implicit chunks are laid out sequentially starting at the given address.
/// Each chunk has the same size = product(chunk_dims) * elem_size.
pub fn collect_implicit_chunk_entries(
    start_address: u64,
    dataset_shape: &[u64],
    chunk_dims: &[u32],
    elem_size: usize,
    chunk_bounds: Option<(&[u64], &[u64])>,
    storage_len: u64,
) -> Result<Vec<ChunkEntry>> {
    let ndim = dataset_shape.len();
    let (chunks_per_dim, chunk_bytes) = implicit_layout(dataset_shape, chunk_dims, elem_size)?;
    if dataset_shape.contains(&0) {
        return Ok(Vec::new());
    }

    if ndim == 0 {
        return Ok(vec![ChunkEntry {
            address: start_address,
            size: chunk_bytes,
            filter_mask: 0,
            offsets: Vec::new(),
        }]);
    }

    let (first_chunk, last_chunk): (Vec<u64>, Vec<u64>) = match chunk_bounds {
        Some((first, last)) => (first.to_vec(), last.to_vec()),
        None => (
            vec![0u64; ndim],
            chunks_per_dim
                .iter()
                .map(|count| count.saturating_sub(1))
                .collect(),
        ),
    };

    let mut chunk_counts = Vec::with_capacity(ndim);
    for dim in 0..ndim {
        let selected = last_chunk[dim]
            .checked_sub(first_chunk[dim])
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| {
                Error::InvalidData("implicit chunk selection bounds are invalid".to_string())
            })?;
        chunk_counts.push(selected);
    }
    let total_selected_chunks = chunk_counts.iter().try_fold(1u64, |acc, &count| {
        checked_mul_u64(acc, count, "implicit selected chunk count")
    })?;
    // An implicit index stores its chunks contiguously and unfiltered, so the
    // total chunk data cannot exceed the file. Reject a declared chunk count
    // whose data would not fit before pre-allocating the entry vector, so a
    // tiny file cannot declare billions of chunks.
    if chunk_bytes > 0 {
        let max_chunks = storage_len / chunk_bytes;
        if total_selected_chunks > max_chunks.saturating_add(1) {
            return Err(Error::InvalidData(format!(
                "implicit index declares {total_selected_chunks} chunks ({} bytes each) but the \
                 file has only {storage_len} bytes",
                chunk_bytes
            )));
        }
    }
    let mut entries = Vec::with_capacity(checked_usize(
        total_selected_chunks,
        "implicit selected chunk count",
    )?);
    let mut chunk_indices = first_chunk.clone();

    loop {
        let chunk_idx = chunk_linear_index(&chunk_indices, &chunks_per_dim)?;
        let offsets = chunk_indices
            .iter()
            .enumerate()
            .map(|(dim, chunk_index)| {
                checked_mul_u64(
                    *chunk_index,
                    u64::from(chunk_dims[dim]),
                    "implicit chunk offset",
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let chunk_data_offset =
            checked_mul_u64(chunk_idx, chunk_bytes, "implicit chunk byte offset")?;

        entries.push(ChunkEntry {
            address: checked_add_u64(start_address, chunk_data_offset, "implicit chunk address")?,
            size: chunk_bytes,
            filter_mask: 0,
            offsets,
        });

        let mut advanced = false;
        for dim in (0..ndim).rev() {
            if chunk_indices[dim] < last_chunk[dim] {
                chunk_indices[dim] += 1;
                if dim + 1 < ndim {
                    chunk_indices[(dim + 1)..ndim].copy_from_slice(&first_chunk[(dim + 1)..ndim]);
                }
                advanced = true;
                break;
            }
        }

        if !advanced {
            break;
        }
    }

    Ok(entries)
}

/// Chunk counts per dimension and the byte size of each chunk of an implicit index.
fn implicit_layout(
    dataset_shape: &[u64],
    chunk_dims: &[u32],
    elem_size: usize,
) -> Result<(Vec<u64>, u64)> {
    let ndim = dataset_shape.len();
    if chunk_dims.len() != ndim {
        return Err(Error::InvalidData(format!(
            "implicit chunk rank {} does not match dataset rank {ndim}",
            chunk_dims.len()
        )));
    }

    // Compute how many chunks along each dimension
    let mut chunks_per_dim = Vec::with_capacity(ndim);
    for i in 0..ndim {
        let chunk_dim = u64::from(chunk_dims[i]);
        if chunk_dim == 0 {
            return Err(Error::InvalidData(format!(
                "implicit chunk dimension {i} has zero extent"
            )));
        }
        chunks_per_dim.push(dataset_shape[i].div_ceil(chunk_dim));
    }

    let chunk_elements = chunk_dims.iter().try_fold(1u64, |acc, &dim| {
        checked_mul_u64(acc, u64::from(dim), "implicit chunk element count")
    })?;
    let elem_size = u64::try_from(elem_size).map_err(|_| {
        Error::InvalidData("implicit chunk element size exceeds u64 capacity".to_string())
    })?;
    let chunk_bytes = checked_mul_u64(chunk_elements, elem_size, "implicit chunk byte size")?;
    Ok((chunks_per_dim, chunk_bytes))
}

/// Chunk entries of the chunks in `set` of an implicit index, computed
/// without enumerating the other chunks.
pub(crate) fn collect_implicit_chunk_entries_in_set(
    start_address: u64,
    dataset_shape: &[u64],
    chunk_dims: &[u32],
    elem_size: usize,
    set: &ChunkSet,
) -> Result<Vec<ChunkEntry>> {
    let (chunks_per_dim, chunk_bytes) = implicit_layout(dataset_shape, chunk_dims, elem_size)?;
    if dataset_shape.contains(&0) || set.rank() != dataset_shape.len() {
        return Ok(Vec::new());
    }
    let mut entries = Vec::new();
    for chunk_indices in set.chunks() {
        if chunk_indices
            .iter()
            .zip(&chunks_per_dim)
            .any(|(index, count)| index >= count)
        {
            continue;
        }
        let chunk_idx = chunk_linear_index(&chunk_indices, &chunks_per_dim)?;
        let offsets = chunk_indices
            .iter()
            .zip(chunk_dims)
            .map(|(&index, &dim)| checked_mul_u64(index, u64::from(dim), "implicit chunk offset"))
            .collect::<Result<Vec<_>>>()?;
        let chunk_data_offset =
            checked_mul_u64(chunk_idx, chunk_bytes, "implicit chunk byte offset")?;
        entries.push(ChunkEntry {
            address: checked_add_u64(start_address, chunk_data_offset, "implicit chunk address")?,
            size: chunk_bytes,
            filter_mask: 0,
            offsets,
        });
    }
    Ok(entries)
}

/// Resolve a single-chunk layout.
///
/// The entire dataset is stored as one chunk at the given address.
pub fn single_chunk_entry(
    address: u64,
    filtered_size: u64,
    filter_mask: u32,
    ndim: usize,
) -> ChunkEntry {
    ChunkEntry {
        address,
        size: filtered_size,
        filter_mask,
        offsets: vec![0u64; ndim],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_entry_debug_clone() {
        let entry = ChunkEntry {
            address: 0x1000,
            size: 4096,
            filter_mask: 0,
            offsets: vec![0, 0],
        };
        let entry2 = entry.clone();
        assert_eq!(entry2.address, 0x1000);
        let _ = format!("{:?}", entry);
    }

    #[test]
    fn chunk_set_lookups_match_brute_force() {
        let set = ChunkSet::new(vec![vec![4, 1, 1], vec![0, 7, 3]]);
        let all: Vec<Vec<u64>> = (0..6)
            .flat_map(|r| (0..9).map(move |c| vec![r, c]))
            .collect();
        let selected: Vec<Vec<u64>> = all
            .iter()
            .filter(|chunk| [1, 4].contains(&chunk[0]) && [0, 3, 7].contains(&chunk[1]))
            .cloned()
            .collect();
        assert_eq!(set.chunks().collect::<Vec<_>>(), selected);
        for chunk in &all {
            assert_eq!(set.contains(chunk), selected.contains(chunk), "{chunk:?}");
        }
        // Chunk dims (2, 3): element offsets are chunk indices times those.
        let offsets = |chunk: &[u64]| vec![chunk[0] * 2, chunk[1] * 3];
        for low in &all {
            for high in &all {
                let expected = selected.iter().any(|chunk| chunk >= low && chunk <= high);
                assert_eq!(
                    set.may_contain_between(Some(&offsets(low)), Some(&offsets(high)), &[2, 3]),
                    expected,
                    "{low:?}..={high:?}"
                );
            }
            let after = selected.iter().any(|chunk| chunk >= low);
            assert_eq!(
                set.may_contain_between(Some(&offsets(low)), None, &[2, 3]),
                after
            );
            let before = selected.iter().any(|chunk| chunk <= low);
            assert_eq!(
                set.may_contain_between(None, Some(&offsets(low)), &[2, 3]),
                before
            );
        }
        assert!(!ChunkSet::new(vec![vec![], vec![1]]).may_contain_between(None, None, &[1, 1]));
    }

    #[test]
    fn implicit_chunk_entries() {
        let entries =
            collect_implicit_chunk_entries(1000, &[10, 20], &[5, 10], 4, None, u64::MAX).unwrap();
        // 2 chunks along dim 0, 2 chunks along dim 1 = 4 total
        assert_eq!(entries.len(), 4);
        assert_eq!(entries[0].address, 1000);
        assert_eq!(entries[0].offsets, vec![0, 0]);
        assert_eq!(entries[1].address, 1000 + 200); // 5*10*4 = 200
        assert_eq!(entries[1].offsets, vec![0, 10]);
        assert_eq!(entries[2].offsets, vec![5, 0]);
        assert_eq!(entries[3].offsets, vec![5, 10]);
    }

    #[test]
    fn implicit_chunk_entries_empty_extent() {
        let entries =
            collect_implicit_chunk_entries(1000, &[0, 20], &[5, 10], 4, None, u64::MAX).unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn implicit_chunk_entries_reject_chunk_byte_overflow() {
        let err = collect_implicit_chunk_entries(
            1000,
            &[10, 10],
            &[u32::MAX, u32::MAX],
            2,
            None,
            u64::MAX,
        )
        .unwrap_err();
        assert!(err.to_string().contains("implicit chunk byte size"));
    }

    #[test]
    fn implicit_chunk_entries_reject_address_overflow() {
        let err =
            collect_implicit_chunk_entries(u64::MAX, &[2], &[1], 1, Some((&[1], &[1])), u64::MAX)
                .unwrap_err();
        assert!(err.to_string().contains("implicit chunk address"));
    }

    #[test]
    fn implicit_chunk_entries_reject_declared_count_beyond_file() {
        // 2^32 chunks of 4 bytes each declared in a 1 KiB file must be
        // rejected before the entry vector is allocated.
        let err = collect_implicit_chunk_entries(1000, &[1u64 << 34, 1], &[4, 1], 1, None, 1024)
            .unwrap_err();
        assert!(
            err.to_string().contains("file has only"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn single_chunk_entry_uses_origin_offsets() {
        let entry = single_chunk_entry(0x2000, 8192, 0, 3);
        assert_eq!(entry.address, 0x2000);
        assert_eq!(entry.size, 8192);
        assert_eq!(entry.offsets, vec![0, 0, 0]);
    }
}
