//! CF Conventions support for NetCDF files.
//!
//! This module provides helpers for interpreting NetCDF data according to
//! the CF (Climate and Forecast) Conventions:
//! - Axis identification (T, X, Y, Z) from standard_name and axis attributes
//! - Coordinate reference system (CRS) extraction from grid_mapping
//! - Time coordinate decoding (units like "days since ...")
//! - Bounds variables for cell boundaries
//!
//! TODO: Phase 6

pub mod axes;
pub mod bounds;
pub mod crs;
pub mod time;
