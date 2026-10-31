//! scale_factor / add_offset unpacking for NetCDF variables.
//!
//! The CF convention defines:
//!   `actual_value = stored_value * scale_factor + add_offset`
//!
//! Defaults: `scale_factor = 1.0`, `add_offset = 0.0`.
//!
//! The `_Unsigned` attribute marks integers stored in the type of the opposite signedness,
//! usually because the classic format had no unsigned types. [`SignParams`] recovers their
//! values, which comes before masking and unpacking.

use ndarray::ArrayD;

use crate::types::{NcType, NcVariable};

/// How to reinterpret integers whose `_Unsigned` attribute contradicts their stored type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SignParams {
    modulus: f64,
    to_unsigned: bool,
}

impl SignParams {
    /// Returns `None` unless the variable is an integer whose `_Unsigned` attribute says its
    /// signedness differs from the stored type.
    pub fn from_variable(var: &NcVariable) -> Option<Self> {
        let unsigned = var.attribute("_Unsigned")?.value.as_string()?;
        let unsigned = if unsigned.eq_ignore_ascii_case("true") {
            true
        } else if unsigned.eq_ignore_ascii_case("false") {
            false
        } else {
            return None;
        };

        let (bits, stored_unsigned) = match var.dtype {
            NcType::Byte => (8, false),
            NcType::Short => (16, false),
            NcType::Int => (32, false),
            NcType::Int64 => (64, false),
            NcType::UByte => (8, true),
            NcType::UShort => (16, true),
            NcType::UInt => (32, true),
            NcType::UInt64 => (64, true),
            _ => return None,
        };

        (unsigned != stored_unsigned).then_some(SignParams {
            modulus: 2f64.powi(bits),
            to_unsigned: unsigned,
        })
    }

    /// The value of one stored integer under the signedness `_Unsigned` declares.
    pub fn reinterpret(&self, value: f64) -> f64 {
        if self.to_unsigned && value < 0.0 {
            value + self.modulus
        } else if !self.to_unsigned && value >= self.modulus / 2.0 {
            value - self.modulus
        } else {
            value
        }
    }

    /// Reinterpret every value in `data`.
    pub fn apply(&self, data: &mut ArrayD<f64>) {
        data.mapv_inplace(|value| self.reinterpret(value));
    }
}

/// Parameters for unpacking packed variable data.
#[derive(Debug, Clone, Copy)]
pub struct UnpackParams {
    pub scale_factor: f64,
    pub add_offset: f64,
}

impl UnpackParams {
    /// Extract unpacking parameters from a variable's attributes.
    ///
    /// Returns `None` if neither `scale_factor` nor `add_offset` is present
    /// (i.e., no unpacking needed).
    pub fn from_variable(var: &NcVariable) -> Option<Self> {
        let scale = var.attribute("scale_factor").and_then(|a| a.value.as_f64());
        let offset = var.attribute("add_offset").and_then(|a| a.value.as_f64());

        if scale.is_none() && offset.is_none() {
            return None;
        }

        Some(UnpackParams {
            scale_factor: scale.unwrap_or(1.0),
            add_offset: offset.unwrap_or(0.0),
        })
    }

    /// Apply unpacking: `actual = stored * scale_factor + add_offset`.
    pub fn apply(&self, data: &mut ArrayD<f64>) {
        if self.scale_factor == 1.0 && self.add_offset == 0.0 {
            return;
        }
        data.mapv_inplace(|v| v * self.scale_factor + self.add_offset);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{NcAttrValue, NcAttribute, NcType, NcVariable};
    use ndarray::arr1;

    fn make_var(attrs: Vec<NcAttribute>) -> NcVariable {
        typed_var(NcType::Int, attrs)
    }

    fn typed_var(dtype: NcType, attributes: Vec<NcAttribute>) -> NcVariable {
        NcVariable {
            name: "test".into(),
            dimensions: vec![],
            dtype,
            attributes,
            data_offset: 0,
            _data_size: 0,
            is_record_var: false,
            record_size: 0,
        }
    }

    fn unsigned_attr(value: &str) -> Vec<NcAttribute> {
        vec![NcAttribute {
            name: "_Unsigned".into(),
            value: NcAttrValue::Chars(value.into()),
        }]
    }

    #[test]
    fn unsigned_true_recovers_values_of_signed_storage() {
        let cases = [
            (NcType::Byte, -1.0, 255.0),
            (NcType::Short, -32768.0, 32768.0),
            (NcType::Int, -1.0, 4294967295.0),
            (NcType::Short, 5.0, 5.0),
        ];
        for (dtype, stored, expected) in cases {
            let var = typed_var(dtype, unsigned_attr("true"));
            let params = SignParams::from_variable(&var).unwrap();
            assert_eq!(params.reinterpret(stored), expected);
        }
    }

    #[test]
    fn unsigned_false_recovers_values_of_unsigned_storage() {
        let var = typed_var(NcType::UShort, unsigned_attr("False"));
        let params = SignParams::from_variable(&var).unwrap();
        assert_eq!(params.reinterpret(65535.0), -1.0);
        assert_eq!(params.reinterpret(32768.0), -32768.0);
        assert_eq!(params.reinterpret(32767.0), 32767.0);
    }

    #[test]
    fn sign_params_need_a_contradiction_with_the_stored_type() {
        for (dtype, value) in [
            (NcType::Short, "false"),
            (NcType::UShort, "true"),
            (NcType::Float, "true"),
            (NcType::Short, "maybe"),
        ] {
            assert!(SignParams::from_variable(&typed_var(dtype, unsigned_attr(value))).is_none());
        }
        assert!(SignParams::from_variable(&typed_var(NcType::Short, vec![])).is_none());
    }

    #[test]
    fn reinterpreting_leaves_nan_alone() {
        let var = typed_var(NcType::Short, unsigned_attr("true"));
        assert!(SignParams::from_variable(&var)
            .unwrap()
            .reinterpret(f64::NAN)
            .is_nan());
    }

    #[test]
    fn no_unpack_attrs() {
        let var = make_var(vec![]);
        assert!(UnpackParams::from_variable(&var).is_none());
    }

    #[test]
    fn scale_only() {
        let var = make_var(vec![NcAttribute {
            name: "scale_factor".into(),
            value: NcAttrValue::Doubles(vec![0.01]),
        }]);
        let params = UnpackParams::from_variable(&var).unwrap();
        assert_eq!(params.scale_factor, 0.01);
        assert_eq!(params.add_offset, 0.0);
    }

    #[test]
    fn offset_only() {
        let var = make_var(vec![NcAttribute {
            name: "add_offset".into(),
            value: NcAttrValue::Doubles(vec![273.15]),
        }]);
        let params = UnpackParams::from_variable(&var).unwrap();
        assert_eq!(params.scale_factor, 1.0);
        assert_eq!(params.add_offset, 273.15);
    }

    #[test]
    fn both() {
        let var = make_var(vec![
            NcAttribute {
                name: "scale_factor".into(),
                value: NcAttrValue::Doubles(vec![0.1]),
            },
            NcAttribute {
                name: "add_offset".into(),
                value: NcAttrValue::Doubles(vec![10.0]),
            },
        ]);
        let params = UnpackParams::from_variable(&var).unwrap();
        let mut data = arr1(&[100.0, 200.0, 300.0]).into_dyn();
        params.apply(&mut data);
        assert_eq!(data[[0]], 20.0); // 100 * 0.1 + 10
        assert_eq!(data[[1]], 30.0); // 200 * 0.1 + 10
        assert_eq!(data[[2]], 40.0); // 300 * 0.1 + 10
    }
}
