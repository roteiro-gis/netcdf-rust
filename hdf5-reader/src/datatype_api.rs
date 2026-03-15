use crate::error::{ByteOrder, Error, Result};
use crate::messages::datatype::Datatype;

// Re-export types from the datatype message module so users don't need to
// reach into messages::datatype.
pub use crate::messages::datatype::{
    CompoundField, EnumMember, ReferenceType, StringEncoding, StringPadding, StringSize,
};

/// Trait for types that can be read from HDF5 datasets.
///
/// Implemented for primitive numeric types. Users can implement this
/// for custom types (e.g., compound types).
pub trait H5Type: Sized + Send + Clone {
    /// The HDF5 datatype that this Rust type corresponds to.
    fn hdf5_type() -> Datatype;

    /// Decode a single value from raw bytes with the given datatype.
    fn from_bytes(bytes: &[u8], dtype: &Datatype) -> Result<Self>;

    /// Size of a single element in bytes.
    fn element_size(dtype: &Datatype) -> usize;
}

/// Read a numeric value from bytes, handling byte-order conversion.
fn read_numeric<const N: usize>(bytes: &[u8], byte_order: ByteOrder) -> Result<[u8; N]> {
    if bytes.len() < N {
        return Err(Error::InvalidData(format!(
            "expected {} bytes, got {}",
            N,
            bytes.len()
        )));
    }
    let mut arr = [0u8; N];
    arr.copy_from_slice(&bytes[..N]);

    // Swap bytes if the source endianness doesn't match native
    #[cfg(target_endian = "little")]
    if byte_order == ByteOrder::BigEndian {
        arr.reverse();
    }
    #[cfg(target_endian = "big")]
    if byte_order == ByteOrder::LittleEndian {
        arr.reverse();
    }

    Ok(arr)
}

macro_rules! impl_h5type_int {
    ($ty:ty, $size:expr, $signed:expr) => {
        impl H5Type for $ty {
            fn hdf5_type() -> Datatype {
                Datatype::FixedPoint {
                    size: $size,
                    signed: $signed,
                    byte_order: if cfg!(target_endian = "little") {
                        ByteOrder::LittleEndian
                    } else {
                        ByteOrder::BigEndian
                    },
                }
            }

            fn from_bytes(bytes: &[u8], dtype: &Datatype) -> Result<Self> {
                match dtype {
                    Datatype::FixedPoint {
                        size, byte_order, ..
                    } => {
                        if *size as usize != std::mem::size_of::<$ty>() {
                            return Err(Error::TypeMismatch {
                                expected: stringify!($ty).into(),
                                actual: format!("FixedPoint(size={})", size),
                            });
                        }
                        let arr = read_numeric::<$size>(bytes, *byte_order)?;
                        Ok(<$ty>::from_ne_bytes(arr))
                    }
                    _ => Err(Error::TypeMismatch {
                        expected: stringify!($ty).into(),
                        actual: format!("{:?}", dtype),
                    }),
                }
            }

            fn element_size(_dtype: &Datatype) -> usize {
                $size
            }
        }
    };
}

impl_h5type_int!(i8, 1, true);
impl_h5type_int!(u8, 1, false);
impl_h5type_int!(i16, 2, true);
impl_h5type_int!(u16, 2, false);
impl_h5type_int!(i32, 4, true);
impl_h5type_int!(u32, 4, false);
impl_h5type_int!(i64, 8, true);
impl_h5type_int!(u64, 8, false);

impl H5Type for f32 {
    fn hdf5_type() -> Datatype {
        Datatype::FloatingPoint {
            size: 4,
            byte_order: if cfg!(target_endian = "little") {
                ByteOrder::LittleEndian
            } else {
                ByteOrder::BigEndian
            },
        }
    }

    fn from_bytes(bytes: &[u8], dtype: &Datatype) -> Result<Self> {
        match dtype {
            Datatype::FloatingPoint { size, byte_order } => {
                if *size != 4 {
                    return Err(Error::TypeMismatch {
                        expected: "f32".into(),
                        actual: format!("FloatingPoint(size={})", size),
                    });
                }
                let arr = read_numeric::<4>(bytes, *byte_order)?;
                Ok(f32::from_ne_bytes(arr))
            }
            _ => Err(Error::TypeMismatch {
                expected: "f32".into(),
                actual: format!("{:?}", dtype),
            }),
        }
    }

    fn element_size(_dtype: &Datatype) -> usize {
        4
    }
}

impl H5Type for f64 {
    fn hdf5_type() -> Datatype {
        Datatype::FloatingPoint {
            size: 8,
            byte_order: if cfg!(target_endian = "little") {
                ByteOrder::LittleEndian
            } else {
                ByteOrder::BigEndian
            },
        }
    }

    fn from_bytes(bytes: &[u8], dtype: &Datatype) -> Result<Self> {
        match dtype {
            Datatype::FloatingPoint { size, byte_order } => {
                if *size != 8 {
                    return Err(Error::TypeMismatch {
                        expected: "f64".into(),
                        actual: format!("FloatingPoint(size={})", size),
                    });
                }
                let arr = read_numeric::<8>(bytes, *byte_order)?;
                Ok(f64::from_ne_bytes(arr))
            }
            _ => Err(Error::TypeMismatch {
                expected: "f64".into(),
                actual: format!("{:?}", dtype),
            }),
        }
    }

    fn element_size(_dtype: &Datatype) -> usize {
        8
    }
}

/// Get the element size from a datatype.
pub fn dtype_element_size(dtype: &Datatype) -> usize {
    match dtype {
        Datatype::FixedPoint { size, .. } => *size as usize,
        Datatype::FloatingPoint { size, .. } => *size as usize,
        Datatype::String {
            size: StringSize::Fixed(n),
            ..
        } => *n as usize,
        Datatype::String {
            size: StringSize::Variable,
            ..
        } => 16,
        Datatype::Compound { size, .. } => *size as usize,
        Datatype::Array { base, dims } => {
            let base_size = dtype_element_size(base);
            let count: u64 = dims.iter().product();
            base_size * count as usize
        }
        Datatype::Enum { base, .. } => dtype_element_size(base),
        Datatype::VarLen { .. } => 16,
        Datatype::Opaque { size, .. } => *size as usize,
        Datatype::Reference { size, .. } => *size as usize,
        Datatype::Bitfield { size, .. } => *size as usize,
    }
}
