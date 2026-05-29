use crate::error::{Error, Result};
use flate2::read::ZlibDecoder;
use std::io::Read;

/// Decompress DEFLATE/zlib-compressed data.
pub fn decompress(data: &[u8]) -> Result<Vec<u8>> {
    decompress_inner(data, None)
}

/// Decompress DEFLATE/zlib-compressed data, reading at most `max_output_len`
/// bytes from the decoded stream.
pub fn decompress_with_limit(data: &[u8], max_output_len: usize) -> Result<Vec<u8>> {
    decompress_inner(data, Some(max_output_len))
}

fn decompress_inner(data: &[u8], max_output_len: Option<usize>) -> Result<Vec<u8>> {
    let mut decoder = ZlibDecoder::new(data);
    let mut output = Vec::new();
    if let Some(max_output_len) = max_output_len {
        decoder
            .take(max_output_len as u64)
            .read_to_end(&mut output)
            .map_err(|e| Error::DecompressionError(format!("DEFLATE decompression failed: {e}")))?;
    } else {
        decoder
            .read_to_end(&mut output)
            .map_err(|e| Error::DecompressionError(format!("DEFLATE decompression failed: {e}")))?;
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::write::ZlibEncoder;
    use flate2::Compression;
    use std::io::Write;

    #[test]
    fn roundtrip() {
        let original = b"hello world, this is a test of zlib compression!";
        let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(original).unwrap();
        let compressed = encoder.finish().unwrap();

        let decompressed = decompress(&compressed).unwrap();
        assert_eq!(decompressed, original);
    }

    #[test]
    fn decompress_with_limit_stops_at_limit() {
        let original = vec![0x5a; 4096];
        let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&original).unwrap();
        let compressed = encoder.finish().unwrap();

        let decompressed = decompress_with_limit(&compressed, 17).unwrap();
        assert_eq!(decompressed.len(), 17);
        assert_eq!(decompressed, original[..17]);
    }
}
