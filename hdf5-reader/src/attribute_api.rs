use crate::error::{Error, Result};
use crate::messages::attribute::AttributeMessage;
use crate::messages::dataspace::DataspaceType;
use crate::messages::datatype::{Datatype, StringEncoding, StringPadding, StringSize};

/// A parsed, high-level HDF5 attribute.
#[derive(Debug, Clone)]
pub struct Attribute {
    pub name: String,
    pub datatype: Datatype,
    pub shape: Vec<u64>,
    pub raw_data: Vec<u8>,
}

impl Attribute {
    /// Create from a parsed attribute message.
    pub fn from_message(msg: AttributeMessage) -> Self {
        let shape = match msg.dataspace.dataspace_type {
            DataspaceType::Scalar => vec![],
            DataspaceType::Null => vec![0],
            DataspaceType::Simple => msg.dataspace.dims.clone(),
        };
        Attribute {
            name: msg.name,
            datatype: msg.datatype,
            shape,
            raw_data: msg.raw_data,
        }
    }

    /// Total number of elements.
    pub fn num_elements(&self) -> u64 {
        if self.shape.is_empty() {
            1 // scalar
        } else {
            self.shape.iter().product()
        }
    }

    /// Read the attribute value as a scalar of the given type.
    pub fn read_scalar<T: crate::datatype_api::H5Type>(&self) -> Result<T> {
        T::from_bytes(&self.raw_data, &self.datatype)
    }

    /// Read the attribute as a 1-D vector of the given type.
    pub fn read_1d<T: crate::datatype_api::H5Type>(&self) -> Result<Vec<T>> {
        let elem_size = T::element_size(&self.datatype);
        let n = self.num_elements() as usize;
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            let start = i * elem_size;
            let end = start + elem_size;
            if end > self.raw_data.len() {
                return Err(Error::InvalidData(format!(
                    "attribute data too short: need {} bytes, have {}",
                    end,
                    self.raw_data.len()
                )));
            }
            result.push(T::from_bytes(&self.raw_data[start..end], &self.datatype)?);
        }
        Ok(result)
    }

    /// Read the attribute as a string (for string-typed attributes).
    pub fn read_string(&self) -> Result<String> {
        match &self.datatype {
            Datatype::String {
                size,
                encoding,
                padding,
            } => match size {
                StringSize::Fixed(len) => {
                    let len = *len as usize;
                    let bytes = if self.raw_data.len() < len {
                        &self.raw_data
                    } else {
                        &self.raw_data[..len]
                    };
                    decode_string(bytes, *padding, *encoding)
                }
                StringSize::Variable => {
                    // For inline vlen strings in attributes, try direct decode.
                    decode_string(&self.raw_data, *padding, *encoding)
                }
            },
            _ => Err(Error::TypeMismatch {
                expected: "String".into(),
                actual: format!("{:?}", self.datatype),
            }),
        }
    }

    /// Read the attribute as a vector of strings.
    pub fn read_strings(&self) -> Result<Vec<String>> {
        match &self.datatype {
            Datatype::String {
                size: StringSize::Fixed(len),
                encoding,
                padding,
            } => {
                let len = *len as usize;
                let n = self.num_elements() as usize;
                let mut result = Vec::with_capacity(n);
                for i in 0..n {
                    let start = i * len;
                    let end = (start + len).min(self.raw_data.len());
                    if start >= self.raw_data.len() {
                        break;
                    }
                    result.push(decode_string(
                        &self.raw_data[start..end],
                        *padding,
                        *encoding,
                    )?);
                }
                Ok(result)
            }
            _ => Err(Error::TypeMismatch {
                expected: "String array".into(),
                actual: format!("{:?}", self.datatype),
            }),
        }
    }

    /// Read an attribute as f64 (with automatic promotion from int types).
    pub fn read_as_f64(&self) -> Result<f64> {
        match &self.datatype {
            Datatype::FloatingPoint { size, .. } => {
                let val: f64 = match size {
                    4 => {
                        let v = self.read_scalar::<f32>()?;
                        v as f64
                    }
                    8 => self.read_scalar::<f64>()?,
                    _ => {
                        return Err(Error::TypeMismatch {
                            expected: "f32 or f64".into(),
                            actual: format!("FloatingPoint(size={})", size),
                        })
                    }
                };
                Ok(val)
            }
            Datatype::FixedPoint { size, signed, .. } => {
                let val = match (size, signed) {
                    (1, true) => self.read_scalar::<i8>()? as f64,
                    (1, false) => self.read_scalar::<u8>()? as f64,
                    (2, true) => self.read_scalar::<i16>()? as f64,
                    (2, false) => self.read_scalar::<u16>()? as f64,
                    (4, true) => self.read_scalar::<i32>()? as f64,
                    (4, false) => self.read_scalar::<u32>()? as f64,
                    (8, true) => self.read_scalar::<i64>()? as f64,
                    (8, false) => self.read_scalar::<u64>()? as f64,
                    _ => {
                        return Err(Error::TypeMismatch {
                            expected: "numeric".into(),
                            actual: format!("FixedPoint(size={})", size),
                        })
                    }
                };
                Ok(val)
            }
            _ => Err(Error::TypeMismatch {
                expected: "numeric".into(),
                actual: format!("{:?}", self.datatype),
            }),
        }
    }
}

/// Decode a byte slice into a String, handling padding and encoding.
fn decode_string(
    bytes: &[u8],
    padding: StringPadding,
    _encoding: StringEncoding,
) -> Result<String> {
    let trimmed = match padding {
        StringPadding::NullTerminate => {
            let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
            &bytes[..end]
        }
        StringPadding::NullPad => {
            let end = bytes.iter().rposition(|&b| b != 0).map_or(0, |i| i + 1);
            &bytes[..end]
        }
        StringPadding::SpacePad => {
            let end = bytes.iter().rposition(|&b| b != b' ').map_or(0, |i| i + 1);
            &bytes[..end]
        }
    };

    // Both ASCII and UTF-8 are valid UTF-8 (ASCII is a subset)
    String::from_utf8(trimmed.to_vec())
        .map_err(|e| Error::InvalidData(format!("invalid string data: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ByteOrder;

    #[test]
    fn test_scalar_f64_attribute() {
        let value: f64 = 3.14;
        let raw_data = value.to_le_bytes().to_vec();
        let attr = Attribute {
            name: "pi".to_string(),
            datatype: Datatype::FloatingPoint {
                size: 8,
                byte_order: ByteOrder::LittleEndian,
            },
            shape: vec![],
            raw_data,
        };
        let val = attr.read_scalar::<f64>().unwrap();
        assert!((val - 3.14).abs() < 1e-10);
    }

    #[test]
    fn test_1d_i32_attribute() {
        let values = [1i32, 2, 3, 4];
        let mut raw_data = Vec::new();
        for v in &values {
            raw_data.extend_from_slice(&v.to_le_bytes());
        }
        let attr = Attribute {
            name: "data".to_string(),
            datatype: Datatype::FixedPoint {
                size: 4,
                signed: true,
                byte_order: ByteOrder::LittleEndian,
            },
            shape: vec![4],
            raw_data,
        };
        let result = attr.read_1d::<i32>().unwrap();
        assert_eq!(result, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_string_attribute() {
        let attr = Attribute {
            name: "units".to_string(),
            datatype: Datatype::String {
                size: StringSize::Fixed(10),
                encoding: StringEncoding::Ascii,
                padding: StringPadding::NullPad,
            },
            shape: vec![],
            raw_data: b"meters\0\0\0\0".to_vec(),
        };
        assert_eq!(attr.read_string().unwrap(), "meters");
    }

    #[test]
    fn test_read_as_f64_from_int() {
        let raw_data = 42i32.to_le_bytes().to_vec();
        let attr = Attribute {
            name: "count".to_string(),
            datatype: Datatype::FixedPoint {
                size: 4,
                signed: true,
                byte_order: ByteOrder::LittleEndian,
            },
            shape: vec![],
            raw_data,
        };
        let val = attr.read_as_f64().unwrap();
        assert!((val - 42.0).abs() < 1e-10);
    }
}
