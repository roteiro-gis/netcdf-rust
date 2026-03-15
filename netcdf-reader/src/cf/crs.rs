//! CF coordinate reference system (CRS) extraction.
//!
//! Reads `grid_mapping` attributes to determine the projection/CRS of
//! spatial data. Supports common projections:
//! - latitude_longitude (EPSG:4326)
//! - transverse_mercator
//! - lambert_conformal_conic
//! - polar_stereographic
//! - rotated_latitude_longitude
//!
//! TODO: Phase 6
