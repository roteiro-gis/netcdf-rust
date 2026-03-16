use std::sync::Arc;

use ndarray::{ArrayD, IxDyn};

use crate::attribute_api::Attribute;
use crate::cache::{ChunkCache, ChunkKey};
use crate::chunk_index;
use crate::datatype_api::{dtype_element_size, H5Type};
use crate::error::{Error, Result};
use crate::filters::{self, FilterRegistry};
use crate::io::Cursor;
use crate::messages::attribute::AttributeMessage;
use crate::messages::dataspace::{DataspaceMessage, DataspaceType};
use crate::messages::datatype::Datatype;
use crate::messages::fill_value::FillValueMessage;
use crate::messages::filter_pipeline::FilterPipelineMessage;
use crate::messages::layout::{ChunkIndexing, DataLayout};
use crate::messages::HdfMessage;
use crate::object_header::ObjectHeader;

/// Hyperslab selection for reading slices of datasets.
#[derive(Debug, Clone)]
pub struct SliceInfo {
    pub selections: Vec<SliceInfoElem>,
}

/// A single dimension's selection.
#[derive(Debug, Clone)]
pub enum SliceInfoElem {
    /// Select a single index (reduces dimensionality).
    Index(u64),
    /// Select a range with optional step.
    Slice { start: u64, end: u64, step: u64 },
}

impl SliceInfo {
    /// Create a selection that reads everything.
    pub fn all(ndim: usize) -> Self {
        SliceInfo {
            selections: vec![
                SliceInfoElem::Slice {
                    start: 0,
                    end: u64::MAX,
                    step: 1,
                };
                ndim
            ],
        }
    }
}

/// A dataset within an HDF5 file.
pub struct Dataset<'f> {
    file_data: &'f [u8],
    offset_size: u8,
    length_size: u8,
    pub(crate) name: String,
    pub(crate) data_address: u64,
    pub(crate) dataspace: DataspaceMessage,
    pub(crate) datatype: Datatype,
    pub(crate) layout: DataLayout,
    pub(crate) fill_value: Option<FillValueMessage>,
    pub(crate) filters: Option<FilterPipelineMessage>,
    pub(crate) attributes: Vec<AttributeMessage>,
    pub(crate) chunk_cache: Arc<ChunkCache>,
    pub(crate) filter_registry: Arc<FilterRegistry>,
}

impl<'f> Dataset<'f> {
    /// Parse a dataset from an object header at the given address.
    pub(crate) fn from_object_header(
        file_data: &'f [u8],
        address: u64,
        name: String,
        offset_size: u8,
        length_size: u8,
        chunk_cache: Arc<ChunkCache>,
        filter_registry: Arc<FilterRegistry>,
    ) -> Result<Self> {
        let mut header = ObjectHeader::parse_at(file_data, address, offset_size, length_size)?;
        header.resolve_shared_messages(file_data, offset_size, length_size);

        let mut dataspace: Option<DataspaceMessage> = None;
        let mut datatype: Option<Datatype> = None;
        let mut layout: Option<DataLayout> = None;
        let mut fill_value: Option<FillValueMessage> = None;
        let mut filter_pipeline: Option<FilterPipelineMessage> = None;
        let mut attributes = Vec::new();

        for msg in header.messages {
            match msg {
                HdfMessage::Dataspace(ds) => dataspace = Some(ds),
                HdfMessage::Datatype(dt) => datatype = Some(dt.datatype),
                HdfMessage::DataLayout(dl) => layout = Some(dl.layout),
                HdfMessage::FillValue(fv) => fill_value = Some(fv),
                HdfMessage::FilterPipeline(fp) => filter_pipeline = Some(fp),
                HdfMessage::Attribute(attr) => attributes.push(attr),
                _ => {}
            }
        }

        let dataspace =
            dataspace.ok_or_else(|| Error::InvalidData("dataset missing dataspace".into()))?;
        let dt = datatype.ok_or_else(|| Error::InvalidData("dataset missing datatype".into()))?;
        let layout =
            layout.ok_or_else(|| Error::InvalidData("dataset missing data layout".into()))?;

        Ok(Dataset {
            file_data,
            offset_size,
            length_size,
            name,
            data_address: address,
            dataspace,
            datatype: dt,
            layout,
            fill_value,
            filters: filter_pipeline,
            attributes,
            chunk_cache,
            filter_registry,
        })
    }

    /// Dataset name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The object header address used to parse this dataset.
    /// Useful as an opaque identifier or for NC4 data_offset.
    pub fn address(&self) -> u64 {
        self.data_address
    }

    /// Shape of the dataset (dimensions).
    pub fn shape(&self) -> &[u64] {
        &self.dataspace.dims
    }

    /// Datatype of the dataset.
    pub fn dtype(&self) -> &Datatype {
        &self.datatype
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        self.dataspace.dims.len()
    }

    /// Maximum dimension sizes, if defined. `u64::MAX` indicates unlimited.
    pub fn max_dims(&self) -> Option<&[u64]> {
        self.dataspace.max_dims.as_deref()
    }

    /// Chunk dimensions, if the dataset is chunked.
    pub fn chunks(&self) -> Option<Vec<u32>> {
        match &self.layout {
            DataLayout::Chunked { dims, .. } => Some(dims.clone()),
            _ => None,
        }
    }

    /// Fill value, if defined.
    pub fn fill_value(&self) -> Option<&FillValueMessage> {
        self.fill_value.as_ref()
    }

    /// Dataset attributes.
    pub fn attributes(&self) -> Vec<Attribute> {
        self.attributes
            .iter()
            .map(|a| Attribute::from_message(a.clone()))
            .collect()
    }

    /// Find an attribute by name.
    pub fn attribute(&self, name: &str) -> Result<Attribute> {
        self.attributes
            .iter()
            .find(|a| a.name == name)
            .map(|a| Attribute::from_message(a.clone()))
            .ok_or_else(|| Error::AttributeNotFound(name.to_string()))
    }

    /// Total number of elements in the dataset.
    pub fn num_elements(&self) -> u64 {
        if self.dataspace.dims.is_empty() {
            match self.dataspace.dataspace_type {
                DataspaceType::Scalar => 1,
                DataspaceType::Null => 0,
                DataspaceType::Simple => 0,
            }
        } else {
            self.dataspace.dims.iter().product()
        }
    }

    /// Read the entire dataset into an n-dimensional array.
    pub fn read_array<T: H5Type>(&self) -> Result<ArrayD<T>> {
        match &self.layout {
            DataLayout::Compact { data } => self.read_compact::<T>(data),
            DataLayout::Contiguous { address, size } => self.read_contiguous::<T>(*address, *size),
            DataLayout::Chunked {
                address,
                dims,
                element_size,
                chunk_indexing,
            } => self.read_chunked::<T>(*address, dims, *element_size, chunk_indexing.as_ref()),
        }
    }

    /// Read a hyperslab of the dataset.
    pub fn read_slice<T: H5Type>(&self, selection: &SliceInfo) -> Result<ArrayD<T>> {
        if selection.selections.len() != self.ndim() {
            return Err(Error::InvalidData(format!(
                "slice has {} dimensions but dataset has {}",
                selection.selections.len(),
                self.ndim()
            )));
        }

        match &self.layout {
            DataLayout::Contiguous { address, size } => {
                self.read_contiguous_slice::<T>(*address, *size, selection)
            }
            DataLayout::Compact { data } => self.read_compact_slice::<T>(data, selection),
            DataLayout::Chunked {
                address,
                dims,
                element_size,
                chunk_indexing,
            } => self.read_chunked_slice::<T>(
                *address,
                dims,
                *element_size,
                chunk_indexing.as_ref(),
                selection,
            ),
        }
    }

    fn read_compact<T: H5Type>(&self, data: &[u8]) -> Result<ArrayD<T>> {
        self.decode_raw_data::<T>(data)
    }

    fn read_contiguous<T: H5Type>(&self, address: u64, size: u64) -> Result<ArrayD<T>> {
        if Cursor::is_undefined_offset(address, self.offset_size) || size == 0 {
            // Dataset with no data written — return fill values
            return self.make_fill_array::<T>();
        }

        let addr = address as usize;
        let sz = size as usize;
        if addr + sz > self.file_data.len() {
            return Err(Error::OffsetOutOfBounds(address));
        }

        let raw = &self.file_data[addr..addr + sz];
        self.decode_raw_data::<T>(raw)
    }

    fn read_chunked<T: H5Type>(
        &self,
        index_address: u64,
        chunk_dims: &[u32],
        _element_size: u32,
        chunk_indexing: Option<&ChunkIndexing>,
    ) -> Result<ArrayD<T>> {
        if Cursor::is_undefined_offset(index_address, self.offset_size) {
            return self.make_fill_array::<T>();
        }

        let ndim = self.ndim();
        let shape = &self.dataspace.dims;
        let elem_size = dtype_element_size(&self.datatype);
        let chunk_shape: Vec<u64> = chunk_dims.iter().map(|&d| d as u64).collect();

        // Allocate output initialized from the dataset's fill value.
        let total_elements = self.num_elements() as usize;
        let total_bytes = total_elements * elem_size;
        let mut flat_data = if let Some(ref fv) = self.fill_value {
            if let Some(ref fill_bytes) = fv.value {
                let mut buf = vec![0u8; total_bytes];
                if !fill_bytes.is_empty() {
                    for chunk in buf.chunks_exact_mut(fill_bytes.len()) {
                        chunk.copy_from_slice(fill_bytes);
                    }
                }
                buf
            } else {
                vec![0u8; total_bytes]
            }
        } else {
            vec![0u8; total_bytes]
        };

        let entries = self.collect_all_chunk_entries(
            index_address,
            chunk_dims,
            chunk_indexing,
            shape,
            ndim,
            elem_size,
        )?;

        for entry in &entries {
            self.assemble_chunk(
                entry,
                index_address,
                &chunk_shape,
                shape,
                elem_size,
                &mut flat_data,
            )?;
        }

        self.decode_raw_data::<T>(&flat_data)
    }

    /// Collect all chunk entries by dispatching on the chunk indexing type.
    ///
    /// Shared by `read_chunked` and `read_chunked_slice`.
    fn collect_all_chunk_entries(
        &self,
        index_address: u64,
        chunk_dims: &[u32],
        chunk_indexing: Option<&ChunkIndexing>,
        shape: &[u64],
        ndim: usize,
        elem_size: usize,
    ) -> Result<Vec<chunk_index::ChunkEntry>> {
        match chunk_indexing {
            None => {
                // V1-V3: B-tree v1 chunk indexing
                self.collect_btree_v1_entries(index_address, ndim)
            }
            Some(ChunkIndexing::SingleChunk {
                filtered_size,
                filters,
            }) => Ok(vec![chunk_index::single_chunk_entry(
                index_address,
                *filtered_size,
                *filters,
                ndim,
            )]),
            Some(ChunkIndexing::BTreeV2) => chunk_index::collect_v2_chunk_entries(
                self.file_data,
                index_address,
                self.offset_size,
                self.length_size,
                ndim as u32,
            ),
            Some(ChunkIndexing::Implicit) => Ok(chunk_index::collect_implicit_chunk_entries(
                index_address,
                shape,
                chunk_dims,
                elem_size,
            )),
            Some(ChunkIndexing::FixedArray { .. }) => {
                crate::fixed_array::collect_fixed_array_chunk_entries(
                    self.file_data,
                    index_address,
                    self.offset_size,
                    self.length_size,
                    shape,
                    chunk_dims,
                )
            }
            Some(ChunkIndexing::ExtensibleArray { .. }) => {
                crate::extensible_array::collect_extensible_array_chunk_entries(
                    self.file_data,
                    index_address,
                    self.offset_size,
                    self.length_size,
                    shape,
                    chunk_dims,
                )
            }
        }
    }

    /// Collect chunk entries from a B-tree v1 index.
    fn collect_btree_v1_entries(
        &self,
        btree_address: u64,
        ndim: usize,
    ) -> Result<Vec<chunk_index::ChunkEntry>> {
        let leaves = crate::btree_v1::collect_btree_v1_leaves(
            self.file_data,
            btree_address,
            self.offset_size,
            self.length_size,
            Some(ndim as u32),
        )?;

        let mut entries = Vec::with_capacity(leaves.len());
        for (key, chunk_addr) in &leaves {
            match key {
                crate::btree_v1::BTreeV1Key::RawData {
                    chunk_size,
                    filter_mask,
                    offsets,
                } => {
                    entries.push(chunk_index::ChunkEntry {
                        address: *chunk_addr,
                        size: *chunk_size as u64,
                        filter_mask: *filter_mask,
                        offsets: offsets[..ndim].to_vec(),
                    });
                }
                _ => {
                    return Err(Error::InvalidData(
                        "expected raw data key in chunk B-tree".into(),
                    ))
                }
            }
        }
        Ok(entries)
    }

    /// Read, decompress, and copy a single chunk into the flat output buffer.
    fn assemble_chunk(
        &self,
        entry: &chunk_index::ChunkEntry,
        dataset_addr: u64,
        chunk_shape: &[u64],
        shape: &[u64],
        elem_size: usize,
        flat_data: &mut [u8],
    ) -> Result<()> {
        let cache_key = ChunkKey {
            dataset_addr,
            chunk_offsets: entry.offsets.clone(),
        };

        let chunk_data = if let Some(cached) = self.chunk_cache.get(&cache_key) {
            cached
        } else {
            let addr = entry.address as usize;
            // Determine on-disk size: use entry.size if nonzero, otherwise
            // compute from uncompressed chunk volume.
            let size = if entry.size > 0 {
                entry.size as usize
            } else {
                chunk_shape.iter().product::<u64>() as usize * elem_size
            };
            if addr + size > self.file_data.len() {
                return Err(Error::OffsetOutOfBounds(entry.address));
            }
            let raw = &self.file_data[addr..addr + size];

            let decoded = if let Some(ref pipeline) = self.filters {
                filters::apply_pipeline(
                    raw,
                    &pipeline.filters,
                    entry.filter_mask,
                    elem_size,
                    Some(&self.filter_registry),
                )?
            } else {
                raw.to_vec()
            };

            self.chunk_cache.insert(cache_key, decoded)
        };

        copy_chunk_to_flat(
            &chunk_data,
            flat_data,
            &entry.offsets,
            chunk_shape,
            shape,
            elem_size,
        );
        Ok(())
    }

    /// Chunked slice: only read chunks that overlap the selection.
    ///
    /// Resolves each `SliceInfoElem` to concrete ranges, computes the chunk
    /// grid range per dimension, and only decompresses overlapping chunks.
    fn read_chunked_slice<T: H5Type>(
        &self,
        index_address: u64,
        chunk_dims: &[u32],
        _element_size: u32,
        chunk_indexing: Option<&ChunkIndexing>,
        selection: &SliceInfo,
    ) -> Result<ArrayD<T>> {
        if Cursor::is_undefined_offset(index_address, self.offset_size) {
            // No data — build a fill-value array then slice it.
            let full = self.make_fill_array::<T>()?;
            return slice_array(&full, selection, &self.dataspace.dims);
        }

        let ndim = self.ndim();
        let shape = &self.dataspace.dims;
        let elem_size = dtype_element_size(&self.datatype);
        let chunk_shape: Vec<u64> = chunk_dims.iter().map(|&d| d as u64).collect();

        // Resolve each selection dimension to (start, end_exclusive, step, count).
        let mut sel_ranges: Vec<(u64, u64, u64, usize)> = Vec::with_capacity(ndim);
        let mut result_shape: Vec<usize> = Vec::new();

        for (i, sel) in selection.selections.iter().enumerate() {
            let dim_size = shape[i];
            match sel {
                SliceInfoElem::Index(idx) => {
                    if *idx >= dim_size {
                        return Err(Error::SliceOutOfBounds {
                            dim: i,
                            index: *idx,
                            size: dim_size,
                        });
                    }
                    sel_ranges.push((*idx, *idx + 1, 1, 1));
                    // Don't add to result_shape — collapsed dim
                }
                SliceInfoElem::Slice { start, end, step } => {
                    let actual_end = if *end == u64::MAX {
                        dim_size
                    } else {
                        (*end).min(dim_size)
                    };
                    if *step == 0 {
                        return Err(Error::InvalidData("slice step cannot be 0".into()));
                    }
                    let count = (actual_end - *start).div_ceil(*step) as usize;
                    sel_ranges.push((*start, actual_end, *step, count));
                    result_shape.push(count);
                }
            }
        }

        // Compute chunk grid range per dimension.
        let mut first_chunk = vec![0u64; ndim];
        let mut last_chunk = vec![0u64; ndim];
        for d in 0..ndim {
            first_chunk[d] = sel_ranges[d].0 / chunk_shape[d];
            last_chunk[d] = if sel_ranges[d].1 == 0 {
                0
            } else {
                (sel_ranges[d].1 - 1) / chunk_shape[d]
            };
        }

        // Collect all chunk entries.
        let all_entries = self.collect_all_chunk_entries(
            index_address,
            chunk_dims,
            chunk_indexing,
            shape,
            ndim,
            elem_size,
        )?;

        // Filter to only chunks overlapping the selection.
        let overlapping: Vec<&chunk_index::ChunkEntry> = all_entries
            .iter()
            .filter(|entry| {
                for d in 0..ndim {
                    let chunk_idx = entry.offsets[d] / chunk_shape[d];
                    if chunk_idx < first_chunk[d] || chunk_idx > last_chunk[d] {
                        return false;
                    }
                }
                true
            })
            .collect();

        // Allocate result buffer (raw bytes) initialized from fill value.
        let result_elements: usize = sel_ranges.iter().map(|r| r.3).product();
        let result_total_bytes = result_elements * elem_size;
        let mut result_buf = if let Some(ref fv) = self.fill_value {
            if let Some(ref fill_bytes) = fv.value {
                let mut buf = vec![0u8; result_total_bytes];
                if !fill_bytes.is_empty() {
                    for chunk in buf.chunks_exact_mut(fill_bytes.len()) {
                        chunk.copy_from_slice(fill_bytes);
                    }
                }
                buf
            } else {
                vec![0u8; result_total_bytes]
            }
        } else {
            vec![0u8; result_total_bytes]
        };

        // Compute result strides (including collapsed dims — they have count=1).
        let result_dims: Vec<usize> = sel_ranges.iter().map(|r| r.3).collect();
        let mut result_strides = vec![1usize; ndim];
        for d in (0..ndim - 1).rev() {
            result_strides[d] = result_strides[d + 1] * result_dims[d + 1];
        }

        // For each overlapping chunk: decompress and copy matching elements.
        for entry in overlapping {
            let cache_key = crate::cache::ChunkKey {
                dataset_addr: index_address,
                chunk_offsets: entry.offsets.clone(),
            };

            let chunk_data = if let Some(cached) = self.chunk_cache.get(&cache_key) {
                cached
            } else {
                let addr = entry.address as usize;
                let size = if entry.size > 0 {
                    entry.size as usize
                } else {
                    chunk_shape.iter().product::<u64>() as usize * elem_size
                };
                if addr + size > self.file_data.len() {
                    return Err(Error::OffsetOutOfBounds(entry.address));
                }
                let raw = &self.file_data[addr..addr + size];
                let decoded = if let Some(ref pipeline) = self.filters {
                    filters::apply_pipeline(
                        raw,
                        &pipeline.filters,
                        entry.filter_mask,
                        elem_size,
                        Some(&self.filter_registry),
                    )?
                } else {
                    raw.to_vec()
                };
                self.chunk_cache.insert(cache_key, decoded)
            };

            // Chunk strides
            let mut chunk_strides = vec![1usize; ndim];
            for d in (0..ndim - 1).rev() {
                chunk_strides[d] = chunk_strides[d + 1] * chunk_shape[d + 1] as usize;
            }

            // For each dimension, compute which elements within this chunk fall
            // within the selection.
            let mut dim_indices: Vec<Vec<(usize, usize)>> = Vec::with_capacity(ndim);
            for d in 0..ndim {
                let chunk_start = entry.offsets[d];
                let chunk_end = (chunk_start + chunk_shape[d]).min(shape[d]);
                let (sel_start, sel_end, sel_step, _) = sel_ranges[d];
                let mut indices = Vec::new();

                // Find first selected index >= chunk_start
                let first_sel = if sel_start >= chunk_start {
                    sel_start
                } else {
                    let steps_to_skip = (chunk_start - sel_start).div_ceil(sel_step);
                    sel_start + steps_to_skip * sel_step
                };

                let mut sel_idx = first_sel;
                while sel_idx < sel_end && sel_idx < chunk_end {
                    let chunk_local = (sel_idx - chunk_start) as usize;
                    // Compute result-space index for this dimension.
                    let result_dim_idx = ((sel_idx - sel_ranges[d].0) / sel_step) as usize;
                    indices.push((chunk_local, result_dim_idx));
                    sel_idx += sel_step;
                }

                dim_indices.push(indices);
            }

            // Iterate over the cartesian product of matching indices.
            copy_selected_elements(
                &chunk_data,
                &mut result_buf,
                &dim_indices,
                &chunk_strides,
                &result_strides,
                elem_size,
                ndim,
            );
        }

        // Decode the result buffer into typed elements.
        let total = result_elements;
        let mut elements = Vec::with_capacity(total);
        for i in 0..total {
            let start = i * elem_size;
            let end = start + elem_size;
            if end <= result_buf.len() {
                elements.push(T::from_bytes(&result_buf[start..end], &self.datatype)?);
            } else {
                elements.push(T::from_bytes(&vec![0u8; elem_size], &self.datatype)?);
            }
        }

        ArrayD::from_shape_vec(IxDyn(&result_shape), elements)
            .map_err(|e| Error::InvalidData(format!("slice shape error: {e}")))
    }

    fn read_contiguous_slice<T: H5Type>(
        &self,
        address: u64,
        size: u64,
        selection: &SliceInfo,
    ) -> Result<ArrayD<T>> {
        // For contiguous layout, read the full data then slice
        let full = self.read_contiguous::<T>(address, size)?;
        slice_array(&full, selection, &self.dataspace.dims)
    }

    fn read_compact_slice<T: H5Type>(
        &self,
        data: &[u8],
        selection: &SliceInfo,
    ) -> Result<ArrayD<T>> {
        let full = self.read_compact::<T>(data)?;
        slice_array(&full, selection, &self.dataspace.dims)
    }

    fn decode_raw_data<T: H5Type>(&self, raw: &[u8]) -> Result<ArrayD<T>> {
        let n = self.num_elements() as usize;
        let elem_size = dtype_element_size(&self.datatype);

        let shape: Vec<usize> = if self.dataspace.dims.is_empty() {
            vec![] // scalar
        } else {
            self.dataspace.dims.iter().map(|&d| d as usize).collect()
        };

        let mut elements = Vec::with_capacity(n);
        for i in 0..n {
            let start = i * elem_size;
            let end = start + elem_size;
            if end > raw.len() {
                // Pad with fill value or zeros if data is short
                let padded = if end <= raw.len() + elem_size {
                    let mut buf = vec![0u8; elem_size];
                    let available = raw.len().saturating_sub(start);
                    if available > 0 {
                        buf[..available].copy_from_slice(&raw[start..start + available]);
                    }
                    T::from_bytes(&buf, &self.datatype)?
                } else {
                    T::from_bytes(&vec![0u8; elem_size], &self.datatype)?
                };
                elements.push(padded);
            } else {
                elements.push(T::from_bytes(&raw[start..end], &self.datatype)?);
            }
        }

        if shape.is_empty() {
            // Scalar
            Ok(ArrayD::from_shape_vec(IxDyn(&[]), elements)
                .map_err(|e| Error::InvalidData(format!("array shape error: {e}")))?)
        } else {
            Ok(ArrayD::from_shape_vec(IxDyn(&shape), elements)
                .map_err(|e| Error::InvalidData(format!("array shape error: {e}")))?)
        }
    }

    fn make_fill_array<T: H5Type>(&self) -> Result<ArrayD<T>> {
        let n = self.num_elements() as usize;
        let elem_size = dtype_element_size(&self.datatype);

        let fill_bytes = if let Some(ref fv) = self.fill_value {
            fv.value.clone().unwrap_or(vec![0u8; elem_size])
        } else {
            vec![0u8; elem_size]
        };

        let mut elements = Vec::with_capacity(n);
        for _ in 0..n {
            elements.push(T::from_bytes(&fill_bytes, &self.datatype)?);
        }

        let shape: Vec<usize> = self.dataspace.dims.iter().map(|&d| d as usize).collect();
        if shape.is_empty() {
            Ok(ArrayD::from_shape_vec(IxDyn(&[]), elements)
                .map_err(|e| Error::InvalidData(format!("array shape error: {e}")))?)
        } else {
            Ok(ArrayD::from_shape_vec(IxDyn(&shape), elements)
                .map_err(|e| Error::InvalidData(format!("array shape error: {e}")))?)
        }
    }
}

/// Copy a chunk's data into the flat output buffer at the correct position.
fn copy_chunk_to_flat(
    chunk_data: &[u8],
    flat: &mut [u8],
    chunk_offsets: &[u64],
    chunk_shape: &[u64],
    dataset_shape: &[u64],
    elem_size: usize,
) {
    let ndim = dataset_shape.len();

    // Compute strides for the dataset (row-major)
    let mut dataset_strides = vec![1usize; ndim];
    for i in (0..ndim - 1).rev() {
        dataset_strides[i] = dataset_strides[i + 1] * dataset_shape[i + 1] as usize;
    }

    // Compute strides for the chunk
    let mut chunk_strides = vec![1usize; ndim];
    for i in (0..ndim - 1).rev() {
        chunk_strides[i] = chunk_strides[i + 1] * chunk_shape[i + 1] as usize;
    }

    // Total elements in this chunk (clamped to dataset boundaries)
    let mut actual_chunk_shape = Vec::with_capacity(ndim);
    for i in 0..ndim {
        let remaining = dataset_shape[i] - chunk_offsets[i];
        actual_chunk_shape.push(remaining.min(chunk_shape[i]) as usize);
    }

    let total_chunk_elements: usize = actual_chunk_shape.iter().product();

    // Iterate over all elements in the chunk
    let mut chunk_idx = vec![0usize; ndim];
    for _ in 0..total_chunk_elements {
        // Compute flat offset in chunk data
        let mut chunk_flat = 0;
        for d in 0..ndim {
            chunk_flat += chunk_idx[d] * chunk_strides[d];
        }

        // Compute flat offset in dataset
        let mut dataset_flat = 0;
        for d in 0..ndim {
            dataset_flat += (chunk_offsets[d] as usize + chunk_idx[d]) * dataset_strides[d];
        }

        let src_start = chunk_flat * elem_size;
        let dst_start = dataset_flat * elem_size;
        let src_end = src_start + elem_size;
        let dst_end = dst_start + elem_size;

        if src_end <= chunk_data.len() && dst_end <= flat.len() {
            flat[dst_start..dst_end].copy_from_slice(&chunk_data[src_start..src_end]);
        }

        // Increment chunk index (row-major order)
        let mut carry = true;
        for d in (0..ndim).rev() {
            if carry {
                chunk_idx[d] += 1;
                if chunk_idx[d] < actual_chunk_shape[d] {
                    carry = false;
                } else {
                    chunk_idx[d] = 0;
                }
            }
        }
    }
}

/// Copy selected elements from a chunk into the result buffer.
///
/// `dim_indices[d]` is a list of `(chunk_local_idx, result_dim_idx)` pairs for dimension `d`.
fn copy_selected_elements(
    chunk_data: &[u8],
    result_buf: &mut [u8],
    dim_indices: &[Vec<(usize, usize)>],
    chunk_strides: &[usize],
    result_strides: &[usize],
    elem_size: usize,
    ndim: usize,
) {
    // Check for empty selection
    if dim_indices.iter().any(|v| v.is_empty()) {
        return;
    }

    // Recursive cartesian-product iteration, but unrolled iteratively.
    let total: usize = dim_indices.iter().map(|v| v.len()).product();
    let mut counters = vec![0usize; ndim];

    for _ in 0..total {
        let mut chunk_flat = 0;
        let mut result_flat = 0;
        for d in 0..ndim {
            let (cl, ri) = dim_indices[d][counters[d]];
            chunk_flat += cl * chunk_strides[d];
            result_flat += ri * result_strides[d];
        }

        let src_start = chunk_flat * elem_size;
        let dst_start = result_flat * elem_size;
        let src_end = src_start + elem_size;
        let dst_end = dst_start + elem_size;

        if src_end <= chunk_data.len() && dst_end <= result_buf.len() {
            result_buf[dst_start..dst_end].copy_from_slice(&chunk_data[src_start..src_end]);
        }

        // Increment counters (row-major)
        let mut carry = true;
        for d in (0..ndim).rev() {
            if carry {
                counters[d] += 1;
                if counters[d] < dim_indices[d].len() {
                    carry = false;
                } else {
                    counters[d] = 0;
                }
            }
        }
    }
}

/// Slice an ndarray according to a SliceInfo selection.
fn slice_array<T: H5Type + Clone>(
    array: &ArrayD<T>,
    selection: &SliceInfo,
    shape: &[u64],
) -> Result<ArrayD<T>> {
    // Build result shape
    let mut result_shape = Vec::new();

    for (i, sel) in selection.selections.iter().enumerate() {
        let dim_size = shape[i];
        match sel {
            SliceInfoElem::Index(idx) => {
                if *idx >= dim_size {
                    return Err(Error::SliceOutOfBounds {
                        dim: i,
                        index: *idx,
                        size: dim_size,
                    });
                }
                // Don't add to result_shape — this dimension is collapsed
            }
            SliceInfoElem::Slice { start, end, step } => {
                let actual_end = if *end == u64::MAX {
                    dim_size as usize
                } else {
                    (*end as usize).min(dim_size as usize)
                };
                let actual_start = *start as usize;
                let actual_step = *step as usize;
                if actual_step == 0 {
                    return Err(Error::InvalidData("slice step cannot be 0".into()));
                }
                let n = (actual_end - actual_start).div_ceil(actual_step);
                result_shape.push(n);
            }
        }
    }

    // Extract elements manually (ndarray's slicing API is complex with dynamic dims)
    let ndim = shape.len();
    let total: usize = result_shape.iter().product();
    let mut elements = Vec::with_capacity(total);

    // Generate all indices in the result
    let mut result_idx = vec![0usize; result_shape.len()];

    for _ in 0..total {
        // Map result index to source index
        let mut src_idx = Vec::with_capacity(ndim);
        let mut ri = 0;
        for sel in selection.selections.iter() {
            match sel {
                SliceInfoElem::Index(idx) => {
                    src_idx.push(*idx as usize);
                }
                SliceInfoElem::Slice { start, step, .. } => {
                    src_idx.push(*start as usize + result_idx[ri] * *step as usize);
                    ri += 1;
                }
            }
        }

        elements.push(array[IxDyn(&src_idx)].clone());

        // Increment result index
        if !result_shape.is_empty() {
            let mut carry = true;
            for d in (0..result_shape.len()).rev() {
                if carry {
                    result_idx[d] += 1;
                    if result_idx[d] < result_shape[d] {
                        carry = false;
                    } else {
                        result_idx[d] = 0;
                    }
                }
            }
        }
    }

    ArrayD::from_shape_vec(IxDyn(&result_shape), elements)
        .map_err(|e| Error::InvalidData(format!("slice shape error: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slice_info_all() {
        let s = SliceInfo::all(3);
        assert_eq!(s.selections.len(), 3);
    }

    #[test]
    fn test_copy_chunk_1d() {
        let chunk_data = vec![1u8, 2, 3, 4]; // 4 elements of 1 byte each
        let mut flat = vec![0u8; 8];
        let chunk_offsets = vec![2u64]; // starts at index 2
        let chunk_shape = vec![4u64];
        let dataset_shape = vec![8u64];

        copy_chunk_to_flat(
            &chunk_data,
            &mut flat,
            &chunk_offsets,
            &chunk_shape,
            &dataset_shape,
            1,
        );
        assert_eq!(flat, vec![0, 0, 1, 2, 3, 4, 0, 0]);
    }
}
