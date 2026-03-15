//! CF axis identification.
//!
//! Determines the role of each coordinate variable (T, X, Y, Z) based on:
//! - The `axis` attribute (most explicit)
//! - The `standard_name` attribute (e.g., "latitude", "longitude", "time")
//! - The `units` attribute (e.g., "degrees_north", "degrees_east")
//! - The `positive` attribute for vertical axes
//!
//! TODO: Phase 6
