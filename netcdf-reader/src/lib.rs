//! Pure-Rust NetCDF file reader.
//!
//! Supports:
//! - **CDF-1** (classic): `CDF\x01` magic
//! - **CDF-2** (64-bit offset): `CDF\x02` magic
//! - **CDF-5** (64-bit data): `CDF\x05` magic
//! - **NetCDF-4** (HDF5-backed): `\x89HDF\r\n\x1a\n` magic (requires `netcdf4` feature)
//!
//! # Example
//!
//! ```no_run
//! use netcdf_reader::NcFile;
//!
//! let file = NcFile::open("example.nc").unwrap();
//! println!("format: {:?}", file.format());
//! for var in file.variables() {
//!     println!("  variable: {} shape={:?}", var.name(), var.shape());
//! }
//! ```

pub mod classic;
pub mod error;
pub mod types;

#[cfg(feature = "netcdf4")]
pub mod nc4;

#[cfg(feature = "cf")]
pub mod cf;

pub use error::{Error, Result};
pub use types::*;

use std::fs::File;
use std::path::Path;

use memmap2::Mmap;
use ndarray::ArrayD;

/// Trait alias for types readable from both classic and NetCDF-4 files.
///
/// This unifies `classic::data::NcReadType` (for CDF-1/2/5) and
/// `hdf5_reader::H5Type` (for NetCDF-4/HDF5) so that `NcFile::read_variable`
/// works across all formats with a single type parameter.
#[cfg(feature = "netcdf4")]
pub trait NcReadable: classic::data::NcReadType + hdf5_reader::H5Type {}
#[cfg(feature = "netcdf4")]
impl<T: classic::data::NcReadType + hdf5_reader::H5Type> NcReadable for T {}

#[cfg(not(feature = "netcdf4"))]
pub trait NcReadable: classic::data::NcReadType {}
#[cfg(not(feature = "netcdf4"))]
impl<T: classic::data::NcReadType> NcReadable for T {}

/// NetCDF file format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NcFormat {
    /// CDF-1 classic format.
    Classic,
    /// CDF-2 64-bit offset format.
    Offset64,
    /// CDF-5 64-bit data format.
    Cdf5,
    /// NetCDF-4 (HDF5-backed).
    Nc4,
    /// NetCDF-4 classic model (HDF5-backed, restricted data model).
    Nc4Classic,
}

/// An opened NetCDF file.
pub struct NcFile {
    format: NcFormat,
    inner: NcFileInner,
}

enum NcFileInner {
    Classic(classic::ClassicFile),
    #[cfg(feature = "netcdf4")]
    Nc4(nc4::Nc4File),
}

/// HDF5 magic bytes: `\x89HDF\r\n\x1a\n`
const HDF5_MAGIC: [u8; 8] = [0x89, b'H', b'D', b'F', 0x0D, 0x0A, 0x1A, 0x0A];

/// Detect the NetCDF format from the first bytes of a file.
fn detect_format(data: &[u8]) -> Result<NcFormat> {
    if data.len() < 4 {
        return Err(Error::InvalidMagic);
    }

    // Check for CDF magic: "CDF" followed by version byte.
    if data[0] == b'C' && data[1] == b'D' && data[2] == b'F' {
        return match data[3] {
            1 => Ok(NcFormat::Classic),
            2 => Ok(NcFormat::Offset64),
            5 => Ok(NcFormat::Cdf5),
            v => Err(Error::UnsupportedVersion(v)),
        };
    }

    // Check for HDF5 magic (8 bytes).
    if data.len() >= 8 && data[..8] == HDF5_MAGIC {
        return Ok(NcFormat::Nc4);
    }

    Err(Error::InvalidMagic)
}

impl NcFile {
    /// Open a NetCDF file from a path.
    ///
    /// The format is auto-detected from the file's magic bytes.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path)?;
        // SAFETY: read-only mapping; caller must not modify the file concurrently.
        let mmap = unsafe { Mmap::map(&file)? };
        let format = detect_format(&mmap)?;

        match format {
            NcFormat::Classic | NcFormat::Offset64 | NcFormat::Cdf5 => {
                // For classic formats, the ClassicFile will re-mmap, so we just
                // pass the path. (We could also hand it the mmap, but re-opening
                // is cleaner for ownership.)
                let classic = classic::ClassicFile::open(path, format)?;
                Ok(NcFile {
                    format,
                    inner: NcFileInner::Classic(classic),
                })
            }
            NcFormat::Nc4 | NcFormat::Nc4Classic => {
                #[cfg(feature = "netcdf4")]
                {
                    let nc4 = nc4::Nc4File::open(path)?;
                    let actual_format = if nc4.is_classic_model() {
                        NcFormat::Nc4Classic
                    } else {
                        NcFormat::Nc4
                    };
                    Ok(NcFile {
                        format: actual_format,
                        inner: NcFileInner::Nc4(nc4),
                    })
                }
                #[cfg(not(feature = "netcdf4"))]
                {
                    Err(Error::Nc4NotEnabled)
                }
            }
        }
    }

    /// Open a NetCDF file from in-memory bytes.
    ///
    /// The format is auto-detected from the magic bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        let format = detect_format(data)?;

        match format {
            NcFormat::Classic | NcFormat::Offset64 | NcFormat::Cdf5 => {
                let classic = classic::ClassicFile::from_bytes(data, format)?;
                Ok(NcFile {
                    format,
                    inner: NcFileInner::Classic(classic),
                })
            }
            NcFormat::Nc4 | NcFormat::Nc4Classic => {
                #[cfg(feature = "netcdf4")]
                {
                    let nc4 = nc4::Nc4File::from_bytes(data)?;
                    let actual_format = if nc4.is_classic_model() {
                        NcFormat::Nc4Classic
                    } else {
                        NcFormat::Nc4
                    };
                    Ok(NcFile {
                        format: actual_format,
                        inner: NcFileInner::Nc4(nc4),
                    })
                }
                #[cfg(not(feature = "netcdf4"))]
                {
                    Err(Error::Nc4NotEnabled)
                }
            }
        }
    }

    /// The detected file format.
    pub fn format(&self) -> NcFormat {
        self.format
    }

    /// The root group of the file.
    ///
    /// Classic files have a single implicit root group containing all
    /// dimensions, variables, and global attributes. NetCDF-4 files
    /// can have nested sub-groups.
    pub fn root_group(&self) -> &NcGroup {
        match &self.inner {
            NcFileInner::Classic(c) => c.root_group(),
            #[cfg(feature = "netcdf4")]
            NcFileInner::Nc4(n) => n.root_group(),
        }
    }

    /// Convenience: dimensions in the root group.
    pub fn dimensions(&self) -> &[NcDimension] {
        &self.root_group().dimensions
    }

    /// Convenience: variables in the root group.
    pub fn variables(&self) -> &[NcVariable] {
        &self.root_group().variables
    }

    /// Convenience: global attributes (attributes of the root group).
    pub fn global_attributes(&self) -> &[NcAttribute] {
        &self.root_group().attributes
    }

    /// Find a group by path relative to the root group.
    pub fn group(&self, path: &str) -> Result<&NcGroup> {
        self.root_group()
            .group(path)
            .ok_or_else(|| Error::GroupNotFound(path.to_string()))
    }

    /// Find a variable by name or path relative to the root group.
    pub fn variable(&self, name: &str) -> Result<&NcVariable> {
        self.root_group()
            .variable(name)
            .ok_or_else(|| Error::VariableNotFound(name.to_string()))
    }

    /// Find a dimension by name or path relative to the root group.
    pub fn dimension(&self, name: &str) -> Result<&NcDimension> {
        self.root_group()
            .dimension(name)
            .ok_or_else(|| Error::DimensionNotFound(name.to_string()))
    }

    /// Find a group attribute by name or path relative to the root group.
    pub fn global_attribute(&self, name: &str) -> Result<&NcAttribute> {
        self.root_group()
            .attribute(name)
            .ok_or_else(|| Error::AttributeNotFound(name.to_string()))
    }

    /// Read a variable's data as a typed array.
    ///
    /// Works for both classic (CDF-1/2/5) and NetCDF-4 files. NetCDF-4 nested
    /// variables can be addressed with paths like `group/subgroup/var`. The type
    /// parameter `T` must implement `NcReadable`, which is satisfied by:
    /// `i8, u8, i16, u16, i32, u32, i64, u64, f32, f64`.
    pub fn read_variable<T: NcReadable>(&self, name: &str) -> Result<ArrayD<T>> {
        match &self.inner {
            NcFileInner::Classic(c) => c.read_variable::<T>(name),
            #[cfg(feature = "netcdf4")]
            NcFileInner::Nc4(n) => Ok(n.read_variable::<T>(name)?),
        }
    }

    /// Access the underlying classic file (for reading data).
    ///
    /// Returns `None` if this is a NetCDF-4 file.
    pub fn as_classic(&self) -> Option<&classic::ClassicFile> {
        match &self.inner {
            NcFileInner::Classic(c) => Some(c),
            #[cfg(feature = "netcdf4")]
            NcFileInner::Nc4(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_cdf1() {
        let data = b"CDF\x01rest_of_file";
        assert_eq!(detect_format(data).unwrap(), NcFormat::Classic);
    }

    #[test]
    fn test_detect_cdf2() {
        let data = b"CDF\x02rest_of_file";
        assert_eq!(detect_format(data).unwrap(), NcFormat::Offset64);
    }

    #[test]
    fn test_detect_cdf5() {
        let data = b"CDF\x05rest_of_file";
        assert_eq!(detect_format(data).unwrap(), NcFormat::Cdf5);
    }

    #[test]
    fn test_detect_hdf5() {
        let mut data = vec![0x89, b'H', b'D', b'F', 0x0D, 0x0A, 0x1A, 0x0A];
        data.extend_from_slice(b"rest_of_file");
        assert_eq!(detect_format(&data).unwrap(), NcFormat::Nc4);
    }

    #[test]
    fn test_detect_invalid_magic() {
        let data = b"XXXX";
        assert!(matches!(
            detect_format(data).unwrap_err(),
            Error::InvalidMagic
        ));
    }

    #[test]
    fn test_detect_unsupported_version() {
        let data = b"CDF\x03";
        assert!(matches!(
            detect_format(data).unwrap_err(),
            Error::UnsupportedVersion(3)
        ));
    }

    #[test]
    fn test_detect_too_short() {
        let data = b"CD";
        assert!(matches!(
            detect_format(data).unwrap_err(),
            Error::InvalidMagic
        ));
    }

    #[test]
    fn test_from_bytes_minimal_cdf1() {
        // Minimal valid CDF-1 file: magic + numrecs + absent dim/att/var lists.
        let mut data = Vec::new();
        data.extend_from_slice(b"CDF\x01");
        data.extend_from_slice(&0u32.to_be_bytes()); // numrecs = 0
                                                     // dim_list: ABSENT
        data.extend_from_slice(&0u32.to_be_bytes()); // tag = 0
        data.extend_from_slice(&0u32.to_be_bytes()); // count = 0
                                                     // att_list: ABSENT
        data.extend_from_slice(&0u32.to_be_bytes());
        data.extend_from_slice(&0u32.to_be_bytes());
        // var_list: ABSENT
        data.extend_from_slice(&0u32.to_be_bytes());
        data.extend_from_slice(&0u32.to_be_bytes());

        let file = NcFile::from_bytes(&data).unwrap();
        assert_eq!(file.format(), NcFormat::Classic);
        assert!(file.dimensions().is_empty());
        assert!(file.variables().is_empty());
        assert!(file.global_attributes().is_empty());
    }

    #[test]
    fn test_from_bytes_cdf1_with_data() {
        // Build a CDF-1 file with one dimension, one global attribute, and one variable.
        let mut data = Vec::new();
        data.extend_from_slice(b"CDF\x01");
        data.extend_from_slice(&0u32.to_be_bytes()); // numrecs = 0

        // dim_list: 1 dimension "x" with size 3
        data.extend_from_slice(&0x0000_000Au32.to_be_bytes()); // NC_DIMENSION tag
        data.extend_from_slice(&1u32.to_be_bytes()); // nelems = 1
                                                     // name "x": length=1, "x", 3 bytes padding
        data.extend_from_slice(&1u32.to_be_bytes());
        data.push(b'x');
        data.extend_from_slice(&[0, 0, 0]); // padding to 4
                                            // dim size
        data.extend_from_slice(&3u32.to_be_bytes());

        // att_list: 1 attribute "title" = "test"
        data.extend_from_slice(&0x0000_000Cu32.to_be_bytes()); // NC_ATTRIBUTE tag
        data.extend_from_slice(&1u32.to_be_bytes()); // nelems = 1
                                                     // name "title"
        data.extend_from_slice(&5u32.to_be_bytes());
        data.extend_from_slice(b"title");
        data.extend_from_slice(&[0, 0, 0]); // padding
                                            // nc_type = NC_CHAR = 2
        data.extend_from_slice(&2u32.to_be_bytes());
        // nvalues = 4
        data.extend_from_slice(&4u32.to_be_bytes());
        data.extend_from_slice(b"test"); // exactly 4 bytes, no padding needed

        // var_list: 1 variable "vals" with dim x, type float
        data.extend_from_slice(&0x0000_000Bu32.to_be_bytes()); // NC_VARIABLE tag
        data.extend_from_slice(&1u32.to_be_bytes()); // nelems = 1
                                                     // name "vals"
        data.extend_from_slice(&4u32.to_be_bytes());
        data.extend_from_slice(b"vals");
        // ndims = 1
        data.extend_from_slice(&1u32.to_be_bytes());
        // dimid = 0
        data.extend_from_slice(&0u32.to_be_bytes());
        // att_list: absent
        data.extend_from_slice(&0u32.to_be_bytes());
        data.extend_from_slice(&0u32.to_be_bytes());
        // nc_type = NC_FLOAT = 5
        data.extend_from_slice(&5u32.to_be_bytes());
        // vsize = 12 (3 floats * 4 bytes)
        data.extend_from_slice(&12u32.to_be_bytes());
        // begin (offset): we'll put data right after this header
        let data_offset = data.len() as u32 + 4; // +4 for this field itself
        data.extend_from_slice(&data_offset.to_be_bytes());

        // Now append the variable data: 3 floats
        data.extend_from_slice(&1.5f32.to_be_bytes());
        data.extend_from_slice(&2.5f32.to_be_bytes());
        data.extend_from_slice(&3.5f32.to_be_bytes());

        let file = NcFile::from_bytes(&data).unwrap();
        assert_eq!(file.format(), NcFormat::Classic);
        assert_eq!(file.dimensions().len(), 1);
        assert_eq!(file.dimensions()[0].name, "x");
        assert_eq!(file.dimensions()[0].size, 3);

        assert_eq!(file.global_attributes().len(), 1);
        assert_eq!(file.global_attributes()[0].name, "title");
        assert_eq!(
            file.global_attributes()[0].value.as_string().unwrap(),
            "test"
        );

        assert_eq!(file.variables().len(), 1);
        let var = file.variable("vals").unwrap();
        assert_eq!(var.dtype(), NcType::Float);
        assert_eq!(var.shape(), vec![3]);

        // Read the actual data through the classic file.
        let classic = file.as_classic().unwrap();
        let arr: ndarray::ArrayD<f32> = classic.read_variable("vals").unwrap();
        assert_eq!(arr.shape(), &[3]);
        assert_eq!(arr[[0]], 1.5f32);
        assert_eq!(arr[[1]], 2.5f32);
        assert_eq!(arr[[2]], 3.5f32);
    }

    #[test]
    fn test_variable_not_found() {
        let mut data = Vec::new();
        data.extend_from_slice(b"CDF\x01");
        data.extend_from_slice(&0u32.to_be_bytes());
        // All absent.
        data.extend_from_slice(&0u32.to_be_bytes());
        data.extend_from_slice(&0u32.to_be_bytes());
        data.extend_from_slice(&0u32.to_be_bytes());
        data.extend_from_slice(&0u32.to_be_bytes());
        data.extend_from_slice(&0u32.to_be_bytes());
        data.extend_from_slice(&0u32.to_be_bytes());

        let file = NcFile::from_bytes(&data).unwrap();
        assert!(matches!(
            file.variable("nonexistent").unwrap_err(),
            Error::VariableNotFound(_)
        ));
    }

    #[test]
    fn test_group_not_found() {
        let mut data = Vec::new();
        data.extend_from_slice(b"CDF\x01");
        data.extend_from_slice(&0u32.to_be_bytes());
        data.extend_from_slice(&0u32.to_be_bytes());
        data.extend_from_slice(&0u32.to_be_bytes());
        data.extend_from_slice(&0u32.to_be_bytes());
        data.extend_from_slice(&0u32.to_be_bytes());
        data.extend_from_slice(&0u32.to_be_bytes());
        data.extend_from_slice(&0u32.to_be_bytes());

        let file = NcFile::from_bytes(&data).unwrap();
        assert!(matches!(
            file.group("nonexistent").unwrap_err(),
            Error::GroupNotFound(_)
        ));
    }
}
