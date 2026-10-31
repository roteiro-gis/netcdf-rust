//! Compare open and metadata costs across files of increasing size.

use std::{hint::black_box, time::Instant};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let paths: Vec<_> = std::env::args().skip(1).collect();
    if paths.is_empty() {
        return Err("usage: metadata_scaling file.nc [file.nc ...]".into());
    }
    println!("variables,hdf5_list_ms,nc_open_ms,nc_open_and_metadata_ms");
    for path in paths {
        let count = netcdf_reader::NcFile::open(&path)?.variables()?.len();
        let mut samples = [Vec::new(), Vec::new(), Vec::new()];
        for _ in 0..7 {
            let start = Instant::now();
            let hdf5 = hdf5_reader::Hdf5File::open(&path)?;
            black_box(hdf5.root_group()?.datasets()?);
            samples[0].push(start.elapsed().as_secs_f64() * 1000.0);
            let start = Instant::now();
            let file = netcdf_reader::NcFile::open(&path)?;
            black_box(&file);
            samples[1].push(start.elapsed().as_secs_f64() * 1000.0);
            let start = Instant::now();
            let file = netcdf_reader::NcFile::open(&path)?;
            assert_eq!(file.variables()?.len(), count);
            samples[2].push(start.elapsed().as_secs_f64() * 1000.0);
        }
        for values in &mut samples {
            values.sort_by(f64::total_cmp);
        }
        println!(
            "{count},{:.3},{:.3},{:.3}",
            samples[0][3], samples[1][3], samples[2][3]
        );
    }
    Ok(())
}
