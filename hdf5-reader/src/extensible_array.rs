//! HDF5 Extensible Array (EA) chunk index.
//!
//! This is the default chunk index for datasets with one unlimited dimension
//! and `libver='latest'`. It uses a three-level hierarchy:
//!
//! - `EAHD` — Extensible Array Header
//! - `EAIB` — Extensible Array Index Block
//! - `EADB` — Extensible Array Data Block
//! - `EASB` — Extensible Array Secondary Block

use crate::checksum::jenkins_lookup3;
use crate::chunk_index::ChunkEntry;
use crate::error::{Error, Result};
use crate::io::Cursor;

const EAHD_SIGNATURE: [u8; 4] = *b"EAHD";
const EAIB_SIGNATURE: [u8; 4] = *b"EAIB";
const EADB_SIGNATURE: [u8; 4] = *b"EADB";
const EASB_SIGNATURE: [u8; 4] = *b"EASB";

/// Parsed Extensible Array Header.
#[derive(Debug)]
struct EaHeader {
    client_id: u8,
    _element_size: u8,
    _max_nelmts_bits: u8,
    idx_blk_elmts: u8,
    data_blk_min_elmts: u8,
    sec_blk_min_data_ptrs: u8,
    max_dblk_page_nelmts_bits: u8,
    _nelmts: u64,
    index_block_address: u64,
}

/// Parse the Extensible Array Header.
///
/// On-disk layout (from H5EA_HEADER_SIZE):
/// sig(4) + ver(1) + client_id(1) +
/// element_size(1) + max_nelmts_bits(1) + idx_blk_elmts(1) +
/// data_blk_min_elmts(1) + sec_blk_min_data_ptrs(1) + max_dblk_page_nelmts_bits(1) +
/// 6 statistics fields (each length_size) +
/// index_block_address(offset_size) + checksum(4)
fn parse_header(data: &[u8], address: u64, offset_size: u8, length_size: u8) -> Result<EaHeader> {
    let mut cursor = Cursor::new(data);
    cursor.set_position(address);

    let sig = cursor.read_bytes(4)?;
    if sig != EAHD_SIGNATURE {
        return Err(Error::InvalidExtensibleArraySignature {
            context: "header signature mismatch",
        });
    }

    let version = cursor.read_u8()?;
    if version != 0 {
        return Err(Error::Other(format!(
            "unsupported extensible array header version {}",
            version
        )));
    }

    let client_id = cursor.read_u8()?;
    let element_size = cursor.read_u8()?;
    let max_nelmts_bits = cursor.read_u8()?;
    let idx_blk_elmts = cursor.read_u8()?;
    let data_blk_min_elmts = cursor.read_u8()?;
    let sec_blk_min_data_ptrs = cursor.read_u8()?;
    let max_dblk_page_nelmts_bits = cursor.read_u8()?;

    // Statistics (6 fields, each length_size bytes)
    let _nsuper_blks = cursor.read_length(length_size)?;
    let _super_blk_size = cursor.read_length(length_size)?;
    let _ndata_blks = cursor.read_length(length_size)?;
    let _data_blk_size = cursor.read_length(length_size)?;
    let _max_idx_set = cursor.read_length(length_size)?;
    let nelmts = cursor.read_length(length_size)?;

    let index_block_address = cursor.read_offset(offset_size)?;

    // Checksum
    let header_end = cursor.position();
    let header_bytes = &data[address as usize..header_end as usize];
    let stored_checksum = cursor.read_u32_le()?;
    let computed = jenkins_lookup3(header_bytes);
    if stored_checksum != computed {
        return Err(Error::ChecksumMismatch {
            expected: stored_checksum,
            actual: computed,
        });
    }

    Ok(EaHeader {
        client_id,
        _element_size: element_size,
        _max_nelmts_bits: max_nelmts_bits,
        idx_blk_elmts,
        data_blk_min_elmts,
        sec_blk_min_data_ptrs,
        max_dblk_page_nelmts_bits,
        _nelmts: nelmts,
        index_block_address,
    })
}

/// Compute the super block layout.
///
/// Returns a vec of (elements_per_data_block, num_data_blocks) for each super block.
fn compute_super_block_layout(header: &EaHeader) -> Vec<(u64, u64)> {
    let mut layout = Vec::new();
    let dblk_min = header.data_blk_min_elmts as u64;
    let sblk_min = header.sec_blk_min_data_ptrs as u64;

    // We generate super blocks until we exceed reasonable limits.
    // In practice, HDF5 files use a small number of super blocks.
    for sb_idx in 0u32..64 {
        let elmts_per_dblk = dblk_min * (1u64 << (sb_idx / 2));
        let num_dblks = sblk_min * (1u64 << (sb_idx.div_ceil(2)));
        layout.push((elmts_per_dblk, num_dblks));
    }

    layout
}

/// A single raw entry.
struct EaRawEntry {
    address: u64,
    chunk_size: u64,
    filter_mask: u32,
}

/// Read `count` entries from the cursor.
fn read_entries(
    cursor: &mut Cursor<'_>,
    count: usize,
    is_filtered: bool,
    offset_size: u8,
    length_size: u8,
) -> Result<Vec<EaRawEntry>> {
    let mut entries = Vec::with_capacity(count);
    for _ in 0..count {
        let address = cursor.read_offset(offset_size)?;
        let (chunk_size, filter_mask) = if is_filtered {
            let cs = cursor.read_length(length_size)?;
            let fm = cursor.read_u32_le()?;
            (cs, fm)
        } else {
            (0, 0)
        };
        entries.push(EaRawEntry {
            address,
            chunk_size,
            filter_mask,
        });
    }
    Ok(entries)
}

/// Parse a data block and return its entries.
///
/// `sizeof_nelmts` is `ceil(max_nelmts_bits / 8)` — used for the block_off field.
#[allow(clippy::too_many_arguments)]
fn parse_data_block(
    data: &[u8],
    address: u64,
    num_entries: usize,
    is_filtered: bool,
    max_page_bits: u8,
    offset_size: u8,
    length_size: u8,
    sizeof_nelmts: usize,
) -> Result<Vec<EaRawEntry>> {
    let mut cursor = Cursor::new(data);
    cursor.set_position(address);

    let sig = cursor.read_bytes(4)?;
    if sig != EADB_SIGNATURE {
        return Err(Error::InvalidExtensibleArraySignature {
            context: "data block signature mismatch",
        });
    }

    let version = cursor.read_u8()?;
    if version != 0 {
        return Err(Error::Other(format!(
            "unsupported extensible array data block version {}",
            version
        )));
    }

    let _client_id = cursor.read_u8()?;
    let _header_address = cursor.read_offset(offset_size)?;

    // Block offset: sizeof_nelmts bytes indicating this block's element index offset.
    cursor.skip(sizeof_nelmts)?;

    // Paging is used only when nelmts exceeds 2^page_bits.
    let page_nelmts = if max_page_bits > 0 {
        1usize << max_page_bits
    } else {
        0
    };

    if page_nelmts > 0 && num_entries > page_nelmts {
        // Paged data block
        let num_pages = num_entries.div_ceil(page_nelmts);
        let bitmap_bytes = num_pages.div_ceil(8);
        let page_bitmap = cursor.read_bytes(bitmap_bytes)?.to_vec();

        let mut all_entries = Vec::with_capacity(num_entries);
        for page_idx in 0..num_pages {
            let byte_idx = page_idx / 8;
            let bit_idx = page_idx % 8;
            let page_initialized =
                byte_idx < page_bitmap.len() && (page_bitmap[byte_idx] & (1 << bit_idx)) != 0;

            let entries_in_page = if page_idx == num_pages - 1 {
                let remainder = num_entries % page_nelmts;
                if remainder == 0 {
                    page_nelmts
                } else {
                    remainder
                }
            } else {
                page_nelmts
            };

            if page_initialized {
                let page_entries = read_entries(
                    &mut cursor,
                    entries_in_page,
                    is_filtered,
                    offset_size,
                    length_size,
                )?;
                let _page_checksum = cursor.read_u32_le()?;
                all_entries.extend(page_entries);
            } else {
                for _ in 0..entries_in_page {
                    all_entries.push(EaRawEntry {
                        address: u64::MAX,
                        chunk_size: 0,
                        filter_mask: 0,
                    });
                }
            }
        }
        Ok(all_entries)
    } else {
        // Non-paged data block
        let entries = read_entries(
            &mut cursor,
            num_entries,
            is_filtered,
            offset_size,
            length_size,
        )?;
        let _checksum = cursor.read_u32_le()?;
        Ok(entries)
    }
}

/// Parse a secondary block and return its data block addresses.
fn parse_secondary_block(
    data: &[u8],
    address: u64,
    num_dblk_addrs: usize,
    offset_size: u8,
) -> Result<Vec<u64>> {
    let mut cursor = Cursor::new(data);
    cursor.set_position(address);

    let sig = cursor.read_bytes(4)?;
    if sig != EASB_SIGNATURE {
        return Err(Error::InvalidExtensibleArraySignature {
            context: "secondary block signature mismatch",
        });
    }

    let version = cursor.read_u8()?;
    if version != 0 {
        return Err(Error::Other(format!(
            "unsupported extensible array secondary block version {}",
            version
        )));
    }

    let _client_id = cursor.read_u8()?;
    let _header_address = cursor.read_offset(offset_size)?;
    let _block_offset = cursor.read_u64_le()?;

    // Read page init bitmap if paging is used — we skip over it for now
    // since we just need the data block addresses.
    // (Pages within data blocks are handled by parse_data_block.)

    let mut addrs = Vec::with_capacity(num_dblk_addrs);
    for _ in 0..num_dblk_addrs {
        addrs.push(cursor.read_offset(offset_size)?);
    }

    // Skip checksum
    let _checksum = cursor.read_u32_le()?;

    Ok(addrs)
}

/// Collect chunk entries from an Extensible Array index.
///
/// Walks the EAHD → EAIB → (EADB / EASB → EADB) hierarchy and converts
/// linear entry indices to multi-dimensional chunk offsets.
pub fn collect_extensible_array_chunk_entries(
    data: &[u8],
    header_address: u64,
    offset_size: u8,
    length_size: u8,
    dataset_shape: &[u64],
    chunk_dims: &[u32],
) -> Result<Vec<ChunkEntry>> {
    let header = parse_header(data, header_address, offset_size, length_size)?;

    if Cursor::is_undefined_offset(header.index_block_address, offset_size) {
        return Ok(Vec::new());
    }

    let is_filtered = header.client_id == 1;
    let sb_layout = compute_super_block_layout(&header);
    let sizeof_nelmts = (header._max_nelmts_bits as usize).div_ceil(8);

    // Parse the index block.
    let mut cursor = Cursor::new(data);
    cursor.set_position(header.index_block_address);

    let sig = cursor.read_bytes(4)?;
    if sig != EAIB_SIGNATURE {
        return Err(Error::InvalidExtensibleArraySignature {
            context: "index block signature mismatch",
        });
    }

    let version = cursor.read_u8()?;
    if version != 0 {
        return Err(Error::Other(format!(
            "unsupported extensible array index block version {}",
            version
        )));
    }

    let _client_id = cursor.read_u8()?;
    let _header_address = cursor.read_offset(offset_size)?;

    // 1. Inline elements (idx_blk_elmts entries stored directly).
    let num_inline = header.idx_blk_elmts as usize;
    let inline_entries = read_entries(
        &mut cursor,
        num_inline,
        is_filtered,
        offset_size,
        length_size,
    )?;

    // 2. Data block addresses stored directly in the index block.
    // The number is 2 * sec_blk_min_data_ptrs (from HDF5: EA_IBLOCK_NDBLK_ADDRS).
    let ndblk_addrs = 2 * header.sec_blk_min_data_ptrs as usize;
    let mut direct_dblk_addrs = Vec::with_capacity(ndblk_addrs);
    for _ in 0..ndblk_addrs {
        direct_dblk_addrs.push(cursor.read_offset(offset_size)?);
    }

    // 3. Secondary block addresses for super blocks 2+.
    // nsblk_addrs = max(0, nsblks - ndblk_addrs) where nsblks is the total
    // number of super blocks needed to cover nelmts.
    let nelmts = header._nelmts;

    // Compute total super blocks needed (nsblks).
    let mut nsblks = 0usize;
    {
        let mut covered = num_inline as u64;
        for sb in &sb_layout {
            if covered >= nelmts {
                break;
            }
            covered += sb.0 * sb.1;
            nsblks += 1;
        }
    }

    let nsblk_addrs = nsblks.saturating_sub(ndblk_addrs);
    let mut sec_block_addrs = Vec::with_capacity(nsblk_addrs);
    for _ in 0..nsblk_addrs {
        sec_block_addrs.push(cursor.read_offset(offset_size)?);
    }

    // Skip checksum at end of index block
    let _checksum = cursor.read_u32_le()?;

    // Now collect all entries.
    let mut all_entries: Vec<EaRawEntry> = Vec::new();

    // Inline entries
    all_entries.extend(inline_entries);

    // Data blocks from direct addresses (super blocks 0-1)
    let mut dblk_addr_idx = 0;
    for sb_idx_iter in 0..2usize.min(nsblks) {
        if sb_idx_iter >= sb_layout.len() {
            break;
        }
        let (elmts_per_dblk, num_dblks) = sb_layout[sb_idx_iter];
        for _ in 0..num_dblks {
            if dblk_addr_idx >= direct_dblk_addrs.len() {
                break;
            }
            let dblk_addr = direct_dblk_addrs[dblk_addr_idx];
            dblk_addr_idx += 1;

            if Cursor::is_undefined_offset(dblk_addr, offset_size) {
                for _ in 0..elmts_per_dblk {
                    all_entries.push(EaRawEntry {
                        address: u64::MAX,
                        chunk_size: 0,
                        filter_mask: 0,
                    });
                }
            } else {
                let dblk_entries = parse_data_block(
                    data,
                    dblk_addr,
                    elmts_per_dblk as usize,
                    is_filtered,
                    header.max_dblk_page_nelmts_bits,
                    offset_size,
                    length_size,
                    sizeof_nelmts,
                )?;
                all_entries.extend(dblk_entries);
            }
        }
    }

    // Data blocks from super blocks 2+ (via secondary blocks)
    for (sec_idx, &sec_addr) in sec_block_addrs.iter().enumerate() {
        let sb_idx_iter = sec_idx + 2;
        if sb_idx_iter >= sb_layout.len() {
            break;
        }
        let (elmts_per_dblk, num_dblks) = sb_layout[sb_idx_iter];

        if Cursor::is_undefined_offset(sec_addr, offset_size) {
            for _ in 0..(elmts_per_dblk * num_dblks) {
                all_entries.push(EaRawEntry {
                    address: u64::MAX,
                    chunk_size: 0,
                    filter_mask: 0,
                });
            }
            continue;
        }

        let dblk_addrs = parse_secondary_block(data, sec_addr, num_dblks as usize, offset_size)?;

        for &dblk_addr in &dblk_addrs {
            if Cursor::is_undefined_offset(dblk_addr, offset_size) {
                for _ in 0..elmts_per_dblk {
                    all_entries.push(EaRawEntry {
                        address: u64::MAX,
                        chunk_size: 0,
                        filter_mask: 0,
                    });
                }
            } else {
                let dblk_entries = parse_data_block(
                    data,
                    dblk_addr,
                    elmts_per_dblk as usize,
                    is_filtered,
                    header.max_dblk_page_nelmts_bits,
                    offset_size,
                    length_size,
                    sizeof_nelmts,
                )?;
                all_entries.extend(dblk_entries);
            }
        }
    }

    // Convert linear indices to chunk offsets.
    let ndim = dataset_shape.len();
    let chunks_per_dim: Vec<u64> = (0..ndim)
        .map(|i| dataset_shape[i].div_ceil(chunk_dims[i] as u64))
        .collect();

    let mut entries = Vec::new();
    for (linear_idx, raw) in all_entries.iter().enumerate() {
        if Cursor::is_undefined_offset(raw.address, offset_size) {
            continue;
        }

        let mut remaining = linear_idx as u64;
        let mut offsets = vec![0u64; ndim];
        for d in (0..ndim).rev() {
            offsets[d] = (remaining % chunks_per_dim[d]) * chunk_dims[d] as u64;
            remaining /= chunks_per_dim[d];
        }

        entries.push(ChunkEntry {
            address: raw.address,
            size: raw.chunk_size,
            filter_mask: raw.filter_mask,
            offsets,
        });
    }

    Ok(entries)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eahd_bad_signature() {
        let mut data = vec![0u8; 64];
        data[0..4].copy_from_slice(b"XXXX");
        let err = parse_header(&data, 0, 8, 8).unwrap_err();
        assert!(matches!(err, Error::InvalidExtensibleArraySignature { .. }));
    }

    #[test]
    fn test_compute_super_block_layout() {
        let header = EaHeader {
            client_id: 0,
            _element_size: 8,
            _max_nelmts_bits: 32,
            idx_blk_elmts: 2,
            data_blk_min_elmts: 2,
            sec_blk_min_data_ptrs: 2,
            max_dblk_page_nelmts_bits: 0,
            _nelmts: 100,
            index_block_address: 0,
        };
        let layout = compute_super_block_layout(&header);
        // sb 0: elmts_per_dblk = 2 * 2^0 = 2, num_dblks = 2 * 2^0 = 2  (cap = 4 elements)
        assert_eq!(layout[0], (2, 2));
        // sb 1: elmts_per_dblk = 2 * 2^0 = 2, num_dblks = 2 * 2^1 = 4  (cap = 8 elements)
        assert_eq!(layout[1], (2, 4));
        // sb 2: elmts_per_dblk = 2 * 2^1 = 4, num_dblks = 2 * 2^1 = 4  (cap = 16 elements)
        assert_eq!(layout[2], (4, 4));
        // sb 3: elmts_per_dblk = 2 * 2^1 = 4, num_dblks = 2 * 2^2 = 8  (cap = 32 elements)
        assert_eq!(layout[3], (4, 8));
    }
}
