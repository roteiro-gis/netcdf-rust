//! NetCDF-4 (HDF5-backed) format support.
//!
//! This module maps the HDF5 data model to the NetCDF data model:
//! - HDF5 groups become NetCDF groups
//! - HDF5 datasets become NetCDF variables
//! - HDF5 attributes become NetCDF attributes
//! - Dimensions are reconstructed from `DIMENSION_LIST` and `_Netcdf4Dimid` attributes
//!
//! Requires the `netcdf4` feature (enabled by default).

pub mod attributes;
pub mod dimensions;
pub mod groups;
pub mod types;
pub mod variables;

use std::path::Path;
use std::sync::{Mutex, OnceLock};

use hdf5_reader::datatype_api::H5Type;
use hdf5_reader::Hdf5File;
use ndarray::ArrayD;
#[cfg(feature = "rayon")]
use rayon::ThreadPool;

use crate::error::{Error, Result};
use crate::types::{NcAttribute, NcDimension, NcGroup, NcType, NcVariable};

/// Dispatch on `NcType` to read data and promote to `f64`.
///
/// `$dtype` must be an expression of type `&NcType`.
/// `$read_expr` is a macro-like callback: for each numeric type `$T`,
/// the macro evaluates `$read_expr` with `$T` substituted in.
///
/// Usage:
/// ```ignore
/// dispatch_read_as_f64!(&var.dtype, |$T| dataset.read_array::<$T>())
/// ```
macro_rules! dispatch_read_as_f64 {
    ($dtype:expr, |$T:ident| $read_expr:expr) => {{
        use crate::types::NcType;
        match $dtype {
            NcType::Byte => {
                type $T = i8;
                let arr = $read_expr?;
                Ok(arr.mapv(|v| v as f64))
            }
            NcType::Short => {
                type $T = i16;
                let arr = $read_expr?;
                Ok(arr.mapv(|v| v as f64))
            }
            NcType::Int => {
                type $T = i32;
                let arr = $read_expr?;
                Ok(arr.mapv(|v| v as f64))
            }
            NcType::Float => {
                type $T = f32;
                let arr = $read_expr?;
                Ok(arr.mapv(|v| v as f64))
            }
            NcType::Double => {
                type $T = f64;
                Ok($read_expr?)
            }
            NcType::UByte => {
                type $T = u8;
                let arr = $read_expr?;
                Ok(arr.mapv(|v| v as f64))
            }
            NcType::UShort => {
                type $T = u16;
                let arr = $read_expr?;
                Ok(arr.mapv(|v| v as f64))
            }
            NcType::UInt => {
                type $T = u32;
                let arr = $read_expr?;
                Ok(arr.mapv(|v| v as f64))
            }
            NcType::Int64 => {
                type $T = i64;
                let arr = $read_expr?;
                Ok(arr.mapv(|v| v as f64))
            }
            NcType::UInt64 => {
                type $T = u64;
                let arr = $read_expr?;
                Ok(arr.mapv(|v| v as f64))
            }
            NcType::Char => Err(Error::TypeMismatch {
                expected: "numeric type".to_string(),
                actual: "Char".to_string(),
            }),
            NcType::String => Err(Error::TypeMismatch {
                expected: "numeric type".to_string(),
                actual: "String".to_string(),
            }),
            other => Err(Error::TypeMismatch {
                expected: "numeric type".to_string(),
                actual: format!("{:?}", other),
            }),
        }
    }};
}

/// An opened NetCDF-4 file (backed by HDF5).
pub struct Nc4File {
    hdf5: Hdf5File,
    metadata_mode: crate::NcMetadataMode,
    root_metadata: OnceLock<NcGroup>,
    subtree_group_cache: MetadataArena<NcGroup>,
    variable_cache: MetadataArena<NcVariable>,
    dimension_cache: MetadataArena<NcDimension>,
    attribute_cache: MetadataArena<NcAttribute>,
}

struct MetadataArena<T> {
    state: Mutex<MetadataArenaState<T>>,
}

struct MetadataArenaState<T> {
    entries: Vec<Box<T>>,
    index: std::collections::HashMap<String, usize>,
}

impl<T> MetadataArena<T> {
    fn new() -> Self {
        Self {
            state: Mutex::new(MetadataArenaState {
                entries: Vec::new(),
                index: std::collections::HashMap::new(),
            }),
        }
    }

    fn get_or_try_insert_with<E, F>(&self, key: &str, build: F) -> std::result::Result<&T, E>
    where
        F: FnOnce() -> std::result::Result<T, E>,
    {
        if let Some(existing) = self.get(key) {
            return Ok(existing);
        }

        let value = build()?;
        let mut state = self.state.lock().expect("metadata arena mutex poisoned");
        if let Some(&index) = state.index.get(key) {
            let ptr: *const T = state.entries[index].as_ref();
            // SAFETY: `entries` stores boxed values that are never removed or
            // moved after insertion, so references derived from their inner
            // pointers remain valid for the lifetime of the arena.
            return Ok(unsafe { &*ptr });
        }

        let boxed = Box::new(value);
        state.entries.push(boxed);
        let index = state.entries.len() - 1;
        state.index.insert(key.to_string(), index);
        let ptr: *const T = state.entries[index].as_ref();
        // SAFETY: `ptr` references the boxed value now owned by `entries`,
        // which will live for the lifetime of `self`.
        Ok(unsafe { &*ptr })
    }

    fn get(&self, key: &str) -> Option<&T> {
        let state = self.state.lock().expect("metadata arena mutex poisoned");
        let index = *state.index.get(key)?;
        let ptr: *const T = state.entries[index].as_ref();
        drop(state);
        // SAFETY: pointers in the arena always reference boxed values stored
        // in `entries`, which are never removed or moved after insertion.
        Some(unsafe { &*ptr })
    }
}

impl Nc4File {
    pub(crate) fn from_hdf5(hdf5: Hdf5File, metadata_mode: crate::NcMetadataMode) -> Result<Self> {
        Ok(Nc4File {
            hdf5,
            metadata_mode,
            root_metadata: OnceLock::new(),
            subtree_group_cache: MetadataArena::new(),
            variable_cache: MetadataArena::new(),
            dimension_cache: MetadataArena::new(),
            attribute_cache: MetadataArena::new(),
        })
    }

    /// Open a NetCDF-4 file from disk.
    pub fn open(path: &Path) -> Result<Self> {
        Self::open_with_options(path, crate::NcOpenOptions::default())
    }

    /// Open a NetCDF-4 file from disk with custom options.
    pub fn open_with_options(path: &Path, options: crate::NcOpenOptions) -> Result<Self> {
        let hdf5 = Hdf5File::open_with_options(
            path,
            hdf5_reader::OpenOptions {
                chunk_cache_bytes: options.chunk_cache_bytes,
                chunk_cache_slots: options.chunk_cache_slots,
                filter_registry: options.filter_registry,
            },
        )?;
        Nc4File::from_hdf5(hdf5, options.metadata_mode)
    }

    /// Open a NetCDF-4 file from in-memory bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        Self::from_bytes_with_options(data, crate::NcOpenOptions::default())
    }

    /// Open a NetCDF-4 file from in-memory bytes with custom options.
    pub fn from_bytes_with_options(data: &[u8], options: crate::NcOpenOptions) -> Result<Self> {
        let hdf5 = Hdf5File::from_bytes_with_options(
            data,
            hdf5_reader::OpenOptions {
                chunk_cache_bytes: options.chunk_cache_bytes,
                chunk_cache_slots: options.chunk_cache_slots,
                filter_registry: options.filter_registry,
            },
        )?;
        Nc4File::from_hdf5(hdf5, options.metadata_mode)
    }

    /// The root group.
    pub fn root_group(&self) -> Result<&NcGroup> {
        self.subtree_group_cache.get_or_try_insert_with("/", || {
            groups::build_group_at_path(&self.hdf5, "/", self.metadata_mode, true)
        })
    }

    /// Check if this file uses the classic data model (`_nc3_strict`).
    ///
    /// This checks the raw HDF5 root group attributes (before the internal
    /// attribute filter removes `_nc3_strict`).
    pub fn is_classic_model(&self) -> bool {
        self.hdf5
            .root_group()
            .ok()
            .and_then(|g| g.attribute("_nc3_strict").ok())
            .is_some()
    }

    pub fn dimensions(&self) -> Result<&[crate::types::NcDimension]> {
        Ok(&self.root_metadata()?.dimensions)
    }

    pub fn variables(&self) -> Result<&[crate::types::NcVariable]> {
        Ok(&self.root_metadata()?.variables)
    }

    pub fn global_attributes(&self) -> Result<&[crate::types::NcAttribute]> {
        Ok(&self.root_metadata()?.attributes)
    }

    pub fn group(&self, path: &str) -> Result<&NcGroup> {
        let normalized = normalize_group_path(path)?;
        self.subtree_group_cache
            .get_or_try_insert_with(normalized, || {
                groups::build_group_at_path(&self.hdf5, normalized, self.metadata_mode, true)
            })
    }

    pub fn variable(&self, path: &str) -> Result<&crate::types::NcVariable> {
        let (group_path, variable_name) = split_parent_path_required(path, "variable")?;
        let normalized_group = normalize_group_path(group_path)?;
        let normalized_dataset = normalize_dataset_path(path)?;
        self.variable_cache
            .get_or_try_insert_with(normalized_dataset, || {
                let context = groups::group_context_at_path(
                    &self.hdf5,
                    normalized_group,
                    self.metadata_mode,
                )?;
                let dataset = context
                    .group
                    .dataset(variable_name)
                    .map_err(|_| Error::VariableNotFound(path.to_string()))?;
                variables::extract_variable(
                    &dataset,
                    &context.group,
                    &context.visible_dimensions,
                    &context.visible_dim_addr_map,
                    self.metadata_mode,
                )?
                .ok_or_else(|| Error::VariableNotFound(path.to_string()))
            })
    }

    pub fn dimension(&self, path: &str) -> Result<&crate::types::NcDimension> {
        let (group_path, dimension_name) = split_parent_path_required(path, "dimension")?;
        let context = groups::group_context_at_path(&self.hdf5, group_path, self.metadata_mode)?;
        let dim = context
            .visible_dimensions
            .into_iter()
            .find(|dim| dim.name == dimension_name)
            .ok_or_else(|| Error::DimensionNotFound(path.to_string()))?;
        self.dimension_cache
            .get_or_try_insert_with(&format!("dim:{path}"), || Ok(dim))
    }

    pub fn global_attribute(&self, path: &str) -> Result<&crate::types::NcAttribute> {
        let (group_path, attr_name) = split_parent_path_required(path, "attribute")?;
        self.attribute_cache
            .get_or_try_insert_with(&format!("attr:{path}"), || {
                let normalized = normalize_group_path(group_path)?;
                let group = if normalized.is_empty() {
                    self.hdf5.root_group()?
                } else {
                    self.hdf5.group(normalized)?
                };
                let attr = group
                    .attribute(attr_name)
                    .map_err(|_| Error::AttributeNotFound(path.to_string()))?;
                attributes::convert_visible_attribute(&attr, self.metadata_mode)?
                    .ok_or_else(|| Error::AttributeNotFound(path.to_string()))
            })
    }

    /// Read a variable's data as a typed array.
    ///
    /// Looks up the variable by path relative to the root group, then opens the
    /// matching HDF5 dataset and reads the data.
    pub fn read_variable<T: H5Type>(&self, path: &str) -> Result<ArrayD<T>> {
        let normalized = normalize_dataset_path(path)?;
        let dataset = self.hdf5.dataset(normalized)?;
        Ok(dataset.read_array::<T>()?)
    }

    /// Read a string variable as a single string.
    pub fn read_variable_as_string(&self, path: &str) -> Result<String> {
        let mut strings = self.read_variable_as_strings(path)?;
        match strings.len() {
            1 => Ok(strings.swap_remove(0)),
            0 => Err(Error::InvalidData(format!(
                "variable '{}' contains no string elements",
                path
            ))),
            count => Err(Error::InvalidData(format!(
                "variable '{}' contains {count} string elements; use read_variable_as_strings()",
                path
            ))),
        }
    }

    /// Read a string variable as a flat vector of strings.
    pub fn read_variable_as_strings(&self, path: &str) -> Result<Vec<String>> {
        let normalized = normalize_dataset_path(path)?;
        let dataset = self.hdf5.dataset(normalized)?;
        let dtype = dataset_nc_type(&dataset)?;
        if dtype != NcType::String {
            return Err(Error::TypeMismatch {
                expected: "String".to_string(),
                actual: format!("{dtype:?}"),
            });
        }
        Ok(dataset.read_strings()?)
    }

    #[cfg(feature = "rayon")]
    pub fn read_variable_parallel<T: H5Type>(&self, path: &str) -> Result<ArrayD<T>> {
        let normalized = normalize_dataset_path(path)?;
        let dataset = self.hdf5.dataset(normalized)?;
        Ok(dataset.read_array_parallel::<T>()?)
    }

    #[cfg(feature = "rayon")]
    pub fn read_variable_in_pool<T: H5Type>(
        &self,
        path: &str,
        pool: &ThreadPool,
    ) -> Result<ArrayD<T>> {
        let normalized = normalize_dataset_path(path)?;
        let dataset = self.hdf5.dataset(normalized)?;
        Ok(dataset.read_array_in_pool::<T>(pool)?)
    }
}

impl Nc4File {
    fn root_metadata(&self) -> Result<&NcGroup> {
        if let Some(group) = self.root_metadata.get() {
            return Ok(group);
        }
        let root_metadata = groups::build_root_group_metadata(&self.hdf5, self.metadata_mode)?;
        let _ = self.root_metadata.set(root_metadata);
        Ok(self
            .root_metadata
            .get()
            .expect("root metadata must be initialized after successful build"))
    }
}

impl Nc4File {
    /// Read a variable with automatic type promotion to f64.
    ///
    /// Reads in the native HDF5 type and promotes to f64 via `mapv`.
    pub fn read_variable_as_f64(&self, path: &str) -> Result<ArrayD<f64>> {
        let normalized = normalize_dataset_path(path)?;
        let dataset = self.hdf5.dataset(normalized)?;
        let dtype = dataset_nc_type(&dataset)?;
        dispatch_read_as_f64!(&dtype, |T| dataset.read_array::<T>())
    }

    /// Read a slice of a variable with automatic type promotion to f64.
    pub fn read_variable_slice_as_f64(
        &self,
        path: &str,
        selection: &crate::types::NcSliceInfo,
    ) -> Result<ArrayD<f64>> {
        let normalized = normalize_dataset_path(path)?;
        let dataset = self.hdf5.dataset(normalized)?;
        let hdf5_sel = selection.to_hdf5_slice_info();
        let dtype = dataset_nc_type(&dataset)?;
        dispatch_read_as_f64!(&dtype, |T| dataset.read_slice::<T>(&hdf5_sel))
    }

    /// Read a typed slice of a variable (NC4 delegation).
    pub fn read_variable_slice<T: H5Type>(
        &self,
        path: &str,
        selection: &crate::types::NcSliceInfo,
    ) -> Result<ArrayD<T>> {
        let normalized = normalize_dataset_path(path)?;
        let dataset = self.hdf5.dataset(normalized)?;
        let hdf5_sel = selection.to_hdf5_slice_info();
        Ok(dataset.read_slice::<T>(&hdf5_sel)?)
    }

    /// Read a typed slice of a variable using chunk-level parallelism.
    ///
    /// Chunked datasets decompress overlapping chunks in parallel via Rayon.
    /// Non-chunked layouts fall back to `read_variable_slice`.
    #[cfg(feature = "rayon")]
    pub fn read_variable_slice_parallel<T: H5Type>(
        &self,
        path: &str,
        selection: &crate::types::NcSliceInfo,
    ) -> Result<ArrayD<T>> {
        let normalized = normalize_dataset_path(path)?;
        let dataset = self.hdf5.dataset(normalized)?;
        let hdf5_sel = selection.to_hdf5_slice_info();
        Ok(dataset.read_slice_parallel::<T>(&hdf5_sel)?)
    }
}

fn normalize_dataset_path(path: &str) -> Result<&str> {
    let trimmed = path.trim_matches('/');
    if trimmed.is_empty() {
        return Err(Error::VariableNotFound(path.to_string()));
    }
    Ok(trimmed)
}

fn normalize_group_path(path: &str) -> Result<&str> {
    Ok(path.trim_matches('/'))
}

fn split_parent_path_required<'a>(path: &'a str, kind: &str) -> Result<(&'a str, &'a str)> {
    let trimmed = path.trim_matches('/');
    if trimmed.is_empty() {
        let err = match kind {
            "group" => Error::GroupNotFound(path.to_string()),
            "dimension" => Error::DimensionNotFound(path.to_string()),
            "attribute" => Error::AttributeNotFound(path.to_string()),
            _ => Error::VariableNotFound(path.to_string()),
        };
        return Err(err);
    }

    Ok(match trimmed.rsplit_once('/') {
        Some((group_path, leaf)) if !leaf.is_empty() => (group_path, leaf),
        Some(_) => ("", trimmed),
        None => ("", trimmed),
    })
}

fn dataset_nc_type(dataset: &hdf5_reader::Dataset<'_>) -> Result<NcType> {
    self::types::hdf5_to_nc_type(dataset.dtype()).map_err(|err| {
        Error::InvalidData(format!(
            "dataset '{}' cannot be mapped to a NetCDF-4 type: {err}",
            dataset.name()
        ))
    })
}
