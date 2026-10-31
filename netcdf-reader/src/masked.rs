//! _FillValue / missing_value masking for NetCDF variables.
//!
//! Replaces fill/missing values with NaN according to CF conventions.
//! Checks `_FillValue`, `missing_value`, `valid_min`, `valid_max`, and `valid_range`.

use ndarray::ArrayD;

use crate::types::{NcType, NcVariable};
use crate::unpack::SignParams;

/// Parameters for masking invalid data.
#[derive(Debug, Clone)]
pub struct MaskParams {
    pub fill_value: Option<f64>,
    /// `missing_value` may be a vector, each element marking missing data.
    pub missing_values: Vec<f64>,
    pub valid_min: Option<f64>,
    pub valid_max: Option<f64>,
}

impl MaskParams {
    /// Extract masking parameters from a variable's attributes.
    ///
    /// Values are converted to the variable's stored type, as it is what they are compared
    /// with, and take any signedness its `_Unsigned` attribute declares.
    ///
    /// Returns `None` if no masking attributes are present.
    pub fn from_variable(var: &NcVariable) -> Option<Self> {
        let sign = SignParams::from_variable(var);
        let value = |value: f64| {
            let value = if var.dtype == NcType::Float {
                f64::from(value as f32)
            } else {
                value
            };
            sign.map_or(value, |sign| sign.reinterpret(value))
        };
        let scalar = |name: &str| {
            var.attribute(name)
                .and_then(|a| a.value.as_f64())
                .map(value)
        };

        let fill = scalar("_FillValue");
        let missing: Vec<f64> = var
            .attribute("missing_value")
            .and_then(|a| a.value.as_f64_vec())
            .unwrap_or_default()
            .into_iter()
            .map(value)
            .collect();

        // valid_range takes precedence over valid_min/valid_max individually
        let (vmin, vmax) = if let Some(range) = var
            .attribute("valid_range")
            .and_then(|a| a.value.as_f64_vec())
        {
            if range.len() >= 2 {
                (Some(value(range[0])), Some(value(range[1])))
            } else {
                (None, None)
            }
        } else {
            (scalar("valid_min"), scalar("valid_max"))
        };

        if fill.is_none() && missing.is_empty() && vmin.is_none() && vmax.is_none() {
            return None;
        }

        Some(MaskParams {
            fill_value: fill,
            missing_values: missing,
            valid_min: vmin,
            valid_max: vmax,
        })
    }

    /// Replace fill/missing values with NaN and mask values outside valid range.
    ///
    /// NaN data stays NaN, so a NaN fill value needs no comparison of its own.
    pub fn apply(&self, data: &mut ArrayD<f64>) {
        data.mapv_inplace(|v| if self.masks(v) { f64::NAN } else { v });
    }

    fn masks(&self, v: f64) -> bool {
        self.fill_value == Some(v)
            || self.missing_values.contains(&v)
            || self.valid_min.is_some_and(|vmin| v < vmin)
            || self.valid_max.is_some_and(|vmax| v > vmax)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{NcAttrValue, NcAttribute, NcType, NcVariable};
    use ndarray::arr1;

    fn make_var(attrs: Vec<NcAttribute>) -> NcVariable {
        NcVariable {
            name: "test".into(),
            dimensions: vec![],
            dtype: NcType::Float,
            attributes: attrs,
            data_offset: 0,
            _data_size: 0,
            is_record_var: false,
            record_size: 0,
        }
    }

    #[test]
    fn no_mask_attrs() {
        let var = make_var(vec![]);
        assert!(MaskParams::from_variable(&var).is_none());
    }

    #[test]
    fn fill_value() {
        let var = make_var(vec![NcAttribute {
            name: "_FillValue".into(),
            value: NcAttrValue::Floats(vec![-9999.0]),
        }]);
        let params = MaskParams::from_variable(&var).unwrap();
        let mut data = arr1(&[1.0, -9999.0, 3.0]).into_dyn();
        params.apply(&mut data);
        assert_eq!(data[[0]], 1.0);
        assert!(data[[1]].is_nan());
        assert_eq!(data[[2]], 3.0);
    }

    #[test]
    fn missing_value() {
        let var = make_var(vec![NcAttribute {
            name: "missing_value".into(),
            value: NcAttrValue::Doubles(vec![-999.0]),
        }]);
        let params = MaskParams::from_variable(&var).unwrap();
        let mut data = arr1(&[-999.0, 5.0]).into_dyn();
        params.apply(&mut data);
        assert!(data[[0]].is_nan());
        assert_eq!(data[[1]], 5.0);
    }

    #[test]
    fn every_element_of_a_missing_value_vector_masks() {
        let var = make_var(vec![NcAttribute {
            name: "missing_value".into(),
            value: NcAttrValue::Doubles(vec![-1.0, -2.0]),
        }]);
        let params = MaskParams::from_variable(&var).unwrap();
        let mut data = arr1(&[1.0, -1.0, -2.0, 3.0]).into_dyn();
        params.apply(&mut data);
        assert_eq!(data[[0]], 1.0);
        assert!(data[[1]].is_nan());
        assert!(data[[2]].is_nan());
        assert_eq!(data[[3]], 3.0);
    }

    #[test]
    fn a_double_fill_value_masks_float_data() {
        // 1e20 has no exact f32 form, so the stored value widens to something else.
        let var = make_var(vec![NcAttribute {
            name: "missing_value".into(),
            value: NcAttrValue::Doubles(vec![1e20]),
        }]);
        let params = MaskParams::from_variable(&var).unwrap();
        let mut data = arr1(&[f64::from(1e20_f32), 1.0]).into_dyn();
        params.apply(&mut data);
        assert!(data[[0]].is_nan());
        assert_eq!(data[[1]], 1.0);
    }

    #[test]
    fn a_nan_fill_value_keeps_nan_data_masked() {
        let var = make_var(vec![NcAttribute {
            name: "_FillValue".into(),
            value: NcAttrValue::Floats(vec![f32::NAN]),
        }]);
        let params = MaskParams::from_variable(&var).unwrap();
        let mut data = arr1(&[f64::NAN, 1.0]).into_dyn();
        params.apply(&mut data);
        assert!(data[[0]].is_nan());
        assert_eq!(data[[1]], 1.0);
    }

    #[test]
    fn zero_fill_values_mask_negative_zero() {
        let var = make_var(vec![NcAttribute {
            name: "_FillValue".into(),
            value: NcAttrValue::Floats(vec![0.0]),
        }]);
        let params = MaskParams::from_variable(&var).unwrap();
        let mut data = arr1(&[-0.0, 1.0]).into_dyn();
        params.apply(&mut data);
        assert!(data[[0]].is_nan());
    }

    #[test]
    fn unsigned_attribute_reinterprets_mask_values() {
        let var = NcVariable {
            dtype: NcType::Short,
            ..make_var(vec![
                NcAttribute {
                    name: "_Unsigned".into(),
                    value: NcAttrValue::Chars("true".into()),
                },
                NcAttribute {
                    name: "_FillValue".into(),
                    value: NcAttrValue::Shorts(vec![-1]),
                },
                NcAttribute {
                    name: "valid_range".into(),
                    value: NcAttrValue::Shorts(vec![0, -2]),
                },
            ])
        };
        let params = MaskParams::from_variable(&var).unwrap();
        assert_eq!(params.fill_value, Some(65535.0));
        assert_eq!(params.valid_min, Some(0.0));
        assert_eq!(params.valid_max, Some(65534.0));
    }

    #[test]
    fn valid_range() {
        let var = make_var(vec![NcAttribute {
            name: "valid_range".into(),
            value: NcAttrValue::Doubles(vec![0.0, 100.0]),
        }]);
        let params = MaskParams::from_variable(&var).unwrap();
        let mut data = arr1(&[-5.0, 50.0, 150.0]).into_dyn();
        params.apply(&mut data);
        assert!(data[[0]].is_nan());
        assert_eq!(data[[1]], 50.0);
        assert!(data[[2]].is_nan());
    }

    #[test]
    fn valid_min_max() {
        let var = make_var(vec![
            NcAttribute {
                name: "valid_min".into(),
                value: NcAttrValue::Doubles(vec![0.0]),
            },
            NcAttribute {
                name: "valid_max".into(),
                value: NcAttrValue::Doubles(vec![50.0]),
            },
        ]);
        let params = MaskParams::from_variable(&var).unwrap();
        let mut data = arr1(&[-1.0, 25.0, 51.0]).into_dyn();
        params.apply(&mut data);
        assert!(data[[0]].is_nan());
        assert_eq!(data[[1]], 25.0);
        assert!(data[[2]].is_nan());
    }
}
