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
//! TODO: Phase 6
