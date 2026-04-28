//! CF time coordinate decoding.
//!
//! Parses time units strings like "days since 1970-01-01 00:00:00" and
//! converts numeric time values to chrono DateTime objects.
//!
//! Supported calendars:
//! - standard (mixed Gregorian/Julian)
//! - proleptic_gregorian
//! - noleap / 365_day
//! - all_leap / 366_day
//! - 360_day
//! - julian
//!
//! Reference: CF Conventions §4.4 "Time Coordinate"

use chrono::{DateTime, NaiveDate, NaiveDateTime, TimeDelta, Utc};

use crate::error::{Error, Result};
use crate::types::{NcDimension, NcGroup, NcVariable};

/// Supported CF calendar types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CfCalendar {
    /// Mixed Gregorian/Julian (default).
    Standard,
    /// Proleptic Gregorian (no Julian transition).
    ProlepticGregorian,
    /// No leap years, every year has 365 days.
    NoLeap,
    /// Every year has 366 days.
    AllLeap,
    /// Every month has 30 days (360 days/year).
    Day360,
    /// Julian calendar.
    Julian,
}

impl CfCalendar {
    /// Parse a calendar name from a CF `calendar` attribute value.
    pub fn parse(s: &str) -> Self {
        match s.trim().to_lowercase().as_str() {
            "standard" | "gregorian" => CfCalendar::Standard,
            "proleptic_gregorian" => CfCalendar::ProlepticGregorian,
            "noleap" | "365_day" => CfCalendar::NoLeap,
            "all_leap" | "366_day" => CfCalendar::AllLeap,
            "360_day" => CfCalendar::Day360,
            "julian" => CfCalendar::Julian,
            _ => CfCalendar::Standard, // Default per CF spec
        }
    }
}

/// Time unit for CF time coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CfTimeUnit {
    Seconds,
    Minutes,
    Hours,
    Days,
    /// Common month (~30.44 days)
    Months,
}

/// Parsed CF time reference.
#[derive(Debug, Clone)]
pub struct CfTimeRef {
    pub unit: CfTimeUnit,
    pub epoch: NaiveDateTime,
    pub calendar: CfCalendar,
}

/// A discovered CF time coordinate backed by a NetCDF coordinate variable.
#[derive(Debug, Clone)]
pub struct CfTimeCoordinate<'a> {
    /// The coordinate variable carrying CF time metadata.
    pub variable: &'a NcVariable,
    /// The dimension represented by the coordinate variable.
    pub dimension: &'a NcDimension,
    /// Parsed `units` and `calendar` metadata for decoding values.
    pub time_ref: CfTimeRef,
}

/// Parse a CF time units string like "days since 1970-01-01 00:00:00".
///
/// Format: `<unit> since <date>[ <time>]`
pub fn parse_time_units(units: &str, calendar: CfCalendar) -> Result<CfTimeRef> {
    let lower = units.trim().to_lowercase();
    let parts: Vec<&str> = lower.splitn(2, " since ").collect();
    if parts.len() != 2 {
        return Err(Error::InvalidData(format!(
            "invalid CF time units '{}': expected '<unit> since <date>'",
            units
        )));
    }

    let unit = match parts[0].trim() {
        "second" | "seconds" | "s" => CfTimeUnit::Seconds,
        "minute" | "minutes" | "min" => CfTimeUnit::Minutes,
        "hour" | "hours" | "hr" | "h" => CfTimeUnit::Hours,
        "day" | "days" | "d" => CfTimeUnit::Days,
        "month" | "months" => CfTimeUnit::Months,
        u => {
            return Err(Error::InvalidData(format!(
                "unsupported CF time unit '{}'",
                u
            )));
        }
    };

    let epoch = parse_epoch(parts[1].trim())?;

    Ok(CfTimeRef {
        unit,
        epoch,
        calendar,
    })
}

/// Parse CF time metadata from a variable.
///
/// Returns `Ok(None)` when the variable has no CF time units. Invalid CF time
/// units return an error so malformed time coordinates are not silently hidden.
pub fn time_ref_from_variable(var: &NcVariable) -> Result<Option<CfTimeRef>> {
    let Some(units) = var
        .attribute("units")
        .and_then(|attr| attr.value.as_string())
    else {
        return Ok(None);
    };

    if !units.trim().to_lowercase().contains(" since ") {
        return Ok(None);
    }

    let calendar = var
        .attribute("calendar")
        .and_then(|attr| attr.value.as_string())
        .map(|value| CfCalendar::parse(&value))
        .unwrap_or(CfCalendar::Standard);

    parse_time_units(&units, calendar).map(Some)
}

/// Discover CF time coordinate variables in a group.
///
/// Only true coordinate variables are considered. Non-time coordinate variables
/// are skipped, while malformed time metadata is returned as an error.
pub fn discover_time_coordinates(group: &NcGroup) -> Result<Vec<CfTimeCoordinate<'_>>> {
    let mut coordinates = Vec::new();
    for variable in group.coordinate_variables() {
        let Some(time_ref) = time_ref_from_variable(variable)? else {
            continue;
        };
        let Some(dimension) = variable.coordinate_dimension() else {
            continue;
        };
        coordinates.push(CfTimeCoordinate {
            variable,
            dimension,
            time_ref,
        });
    }
    Ok(coordinates)
}

/// Discover the CF time coordinate used by a variable, if one exists.
pub fn discover_variable_time_coordinate<'a>(
    var: &NcVariable,
    group: &'a NcGroup,
) -> Result<Option<CfTimeCoordinate<'a>>> {
    for dimension in var.dimensions() {
        let Some(variable) = group.coordinate_variable(&dimension.name) else {
            continue;
        };
        let Some(time_ref) = time_ref_from_variable(variable)? else {
            continue;
        };
        let Some(coordinate_dimension) = variable.coordinate_dimension() else {
            continue;
        };
        return Ok(Some(CfTimeCoordinate {
            variable,
            dimension: coordinate_dimension,
            time_ref,
        }));
    }

    Ok(None)
}

/// Parse the epoch date/time string.
fn parse_epoch(s: &str) -> Result<NaiveDateTime> {
    // Try date + time first
    for fmt in &[
        "%Y-%m-%d %H:%M:%S%.f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%.f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
    ] {
        if let Ok(dt) = NaiveDateTime::parse_from_str(s, fmt) {
            return Ok(dt);
        }
    }

    // Try date-only
    if let Ok(d) = NaiveDate::parse_from_str(s, "%Y-%m-%d") {
        return Ok(d.and_hms_opt(0, 0, 0).unwrap());
    }

    Err(Error::InvalidData(format!("cannot parse CF epoch '{}'", s)))
}

/// Decode a numeric time value to a UTC DateTime.
///
/// For `Standard` and `ProlepticGregorian` calendars, the result is exact.
///
/// For non-standard calendars (`NoLeap`, `AllLeap`, `Day360`, `Julian`), this
/// function applies a Gregorian approximation: it adds the time delta directly
/// to a `chrono::NaiveDateTime`, which uses the Gregorian calendar. This means:
/// - `NoLeap`/`365_day`: dates that fall on Feb 29 in the Gregorian calendar
///   will appear in output even though the source calendar has no leap years.
/// - `Day360`: months are not 30-day uniform; Gregorian month lengths apply.
/// - `Julian`: the Julian–Gregorian transition is not modeled.
///
/// For exact non-standard calendar handling, decode to raw numeric offsets
/// and apply calendar logic in application code.
pub fn decode_time(value: f64, time_ref: &CfTimeRef) -> Result<DateTime<Utc>> {
    let delta = match time_ref.unit {
        CfTimeUnit::Seconds => TimeDelta::milliseconds((value * 1000.0) as i64),
        CfTimeUnit::Minutes => TimeDelta::seconds((value * 60.0) as i64),
        CfTimeUnit::Hours => TimeDelta::seconds((value * 3600.0) as i64),
        CfTimeUnit::Days => TimeDelta::milliseconds((value * 86_400_000.0) as i64),
        CfTimeUnit::Months => {
            // Approximate: 1 month ≈ 30.44 days
            TimeDelta::milliseconds((value * 30.44 * 86_400_000.0) as i64)
        }
    };

    let naive = time_ref
        .epoch
        .checked_add_signed(delta)
        .ok_or_else(|| Error::InvalidData(format!("time value {} out of range", value)))?;

    Ok(DateTime::<Utc>::from_naive_utc_and_offset(naive, Utc))
}

/// Decode a vector of numeric time values.
pub fn decode_times(values: &[f64], time_ref: &CfTimeRef) -> Result<Vec<DateTime<Utc>>> {
    values.iter().map(|&v| decode_time(v, time_ref)).collect()
}

/// Decode numeric values using the CF time metadata on a variable.
pub fn decode_time_coordinate_values(
    var: &NcVariable,
    values: &[f64],
) -> Result<Option<Vec<DateTime<Utc>>>> {
    let Some(time_ref) = time_ref_from_variable(var)? else {
        return Ok(None);
    };
    decode_times(values, &time_ref).map(Some)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{NcAttrValue, NcAttribute, NcType};

    fn attr(name: &str, value: &str) -> NcAttribute {
        NcAttribute {
            name: name.into(),
            value: NcAttrValue::Chars(value.into()),
        }
    }

    fn coordinate_var(name: &str, size: u64, attrs: Vec<NcAttribute>) -> NcVariable {
        NcVariable {
            name: name.into(),
            dimensions: vec![NcDimension {
                name: name.into(),
                size,
                is_unlimited: false,
            }],
            dtype: NcType::Double,
            attributes: attrs,
            data_offset: 0,
            _data_size: 0,
            is_record_var: false,
            record_size: 0,
        }
    }

    #[test]
    fn test_parse_days_since() {
        let tr = parse_time_units("days since 1970-01-01", CfCalendar::Standard).unwrap();
        assert_eq!(tr.unit, CfTimeUnit::Days);
        assert_eq!(
            tr.epoch,
            NaiveDate::from_ymd_opt(1970, 1, 1)
                .unwrap()
                .and_hms_opt(0, 0, 0)
                .unwrap()
        );
    }

    #[test]
    fn test_parse_hours_since_with_time() {
        let tr = parse_time_units("hours since 2000-01-01 00:00:00", CfCalendar::Standard).unwrap();
        assert_eq!(tr.unit, CfTimeUnit::Hours);
        assert_eq!(
            tr.epoch,
            NaiveDate::from_ymd_opt(2000, 1, 1)
                .unwrap()
                .and_hms_opt(0, 0, 0)
                .unwrap()
        );
    }

    #[test]
    fn test_decode_days() {
        let tr = parse_time_units("days since 1970-01-01", CfCalendar::Standard).unwrap();
        let dt = decode_time(365.0, &tr).unwrap();
        assert_eq!(dt.format("%Y-%m-%d").to_string(), "1971-01-01");
    }

    #[test]
    fn test_decode_hours() {
        let tr = parse_time_units("hours since 2000-01-01 00:00:00", CfCalendar::Standard).unwrap();
        let dt = decode_time(24.0, &tr).unwrap();
        assert_eq!(dt.format("%Y-%m-%d").to_string(), "2000-01-02");
    }

    #[test]
    fn test_calendar_from_str() {
        assert_eq!(CfCalendar::parse("standard"), CfCalendar::Standard);
        assert_eq!(CfCalendar::parse("noleap"), CfCalendar::NoLeap);
        assert_eq!(CfCalendar::parse("365_day"), CfCalendar::NoLeap);
        assert_eq!(CfCalendar::parse("360_day"), CfCalendar::Day360);
        assert_eq!(
            CfCalendar::parse("proleptic_gregorian"),
            CfCalendar::ProlepticGregorian
        );
    }

    #[test]
    fn test_invalid_units() {
        assert!(parse_time_units("invalid", CfCalendar::Standard).is_err());
        assert!(parse_time_units("furlongs since yesterday", CfCalendar::Standard).is_err());
    }

    #[test]
    fn test_time_ref_from_variable() {
        let var = coordinate_var(
            "time",
            3,
            vec![
                attr("units", "hours since 2001-01-01 12:00:00"),
                attr("calendar", "proleptic_gregorian"),
            ],
        );

        let time_ref = time_ref_from_variable(&var).unwrap().unwrap();
        assert_eq!(time_ref.unit, CfTimeUnit::Hours);
        assert_eq!(time_ref.calendar, CfCalendar::ProlepticGregorian);
        assert_eq!(
            time_ref.epoch,
            NaiveDate::from_ymd_opt(2001, 1, 1)
                .unwrap()
                .and_hms_opt(12, 0, 0)
                .unwrap()
        );
    }

    #[test]
    fn test_discover_time_coordinates_only_uses_coordinate_variables() {
        let time = coordinate_var("time", 3, vec![attr("units", "days since 1970-01-01")]);
        let data_time = NcVariable {
            name: "data_time".into(),
            dimensions: vec![NcDimension {
                name: "obs".into(),
                size: 3,
                is_unlimited: false,
            }],
            dtype: NcType::Double,
            attributes: vec![attr("units", "days since 1970-01-01")],
            data_offset: 0,
            _data_size: 0,
            is_record_var: false,
            record_size: 0,
        };
        let group = NcGroup {
            name: "/".into(),
            dimensions: vec![time.dimensions()[0].clone()],
            variables: vec![data_time, time],
            attributes: vec![],
            groups: vec![],
        };

        let times = discover_time_coordinates(&group).unwrap();
        assert_eq!(times.len(), 1);
        assert_eq!(times[0].variable.name(), "time");
        assert_eq!(times[0].dimension.name, "time");
    }

    #[test]
    fn test_discover_variable_time_coordinate() {
        let time = coordinate_var("time", 3, vec![attr("units", "hours since 2000-01-01")]);
        let lat = coordinate_var("lat", 2, vec![attr("units", "degrees_north")]);
        let temperature = NcVariable {
            name: "temperature".into(),
            dimensions: vec![time.dimensions()[0].clone(), lat.dimensions()[0].clone()],
            dtype: NcType::Float,
            attributes: vec![],
            data_offset: 0,
            _data_size: 0,
            is_record_var: false,
            record_size: 0,
        };
        let group = NcGroup {
            name: "/".into(),
            dimensions: vec![time.dimensions()[0].clone(), lat.dimensions()[0].clone()],
            variables: vec![lat, time, temperature.clone()],
            attributes: vec![],
            groups: vec![],
        };

        let discovered = discover_variable_time_coordinate(&temperature, &group)
            .unwrap()
            .unwrap();
        assert_eq!(discovered.variable.name(), "time");
        assert_eq!(discovered.time_ref.unit, CfTimeUnit::Hours);
    }

    #[test]
    fn test_decode_time_coordinate_values() {
        let var = coordinate_var("time", 2, vec![attr("units", "days since 1970-01-01")]);

        let decoded = decode_time_coordinate_values(&var, &[0.0, 1.0])
            .unwrap()
            .unwrap();
        assert_eq!(decoded[0].format("%Y-%m-%d").to_string(), "1970-01-01");
        assert_eq!(decoded[1].format("%Y-%m-%d").to_string(), "1970-01-02");
    }
}
