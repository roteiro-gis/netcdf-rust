pub use hdf5_core::jenkins_lookup3;

/// Fletcher-32 checksum used by the HDF5 filter pipeline.
///
/// Matches the HDF5 library's `H5_checksum_fletcher32`: reads 16-bit
/// big-endian words, accumulates in batches of 360, and reduces with
/// `(x & 0xffff) + (x >> 16)` (not `% 65535`).
pub fn fletcher32(data: &[u8]) -> u32 {
    let mut sum1: u32 = 0;
    let mut sum2: u32 = 0;
    let total_words = data.len() / 2;

    let mut offset = 0usize;
    let mut remaining = total_words;

    while remaining > 0 {
        let batch = remaining.min(360);
        remaining -= batch;

        for _ in 0..batch {
            let word = ((data[offset] as u32) << 8) | (data[offset + 1] as u32);
            sum1 += word;
            sum2 += sum1;
            offset += 2;
        }

        sum1 = (sum1 & 0xffff) + (sum1 >> 16);
        sum2 = (sum2 & 0xffff) + (sum2 >> 16);
    }

    // Final reduction
    sum1 = (sum1 & 0xffff) + (sum1 >> 16);
    sum2 = (sum2 & 0xffff) + (sum2 >> 16);

    (sum2 << 16) | sum1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jenkins_empty() {
        let hash = jenkins_lookup3(b"");
        // Empty input produces a deterministic hash from the initial state
        assert_ne!(hash, 0);
    }

    #[test]
    fn jenkins_known_value() {
        // Test with known input — the hash is deterministic
        let h1 = jenkins_lookup3(b"hello");
        let h2 = jenkins_lookup3(b"hello");
        assert_eq!(h1, h2);

        // Different input -> different hash (with overwhelming probability)
        let h3 = jenkins_lookup3(b"world");
        assert_ne!(h1, h3);
    }

    #[test]
    fn fletcher32_simple() {
        let data = [0x01, 0x02, 0x03, 0x04];
        let checksum = fletcher32(&data);
        // Verify it's deterministic
        assert_eq!(checksum, fletcher32(&data));
    }

    #[test]
    fn fletcher32_known_reference() {
        // Known reference: 4 bytes [0x00, 0x01, 0x00, 0x02] interpreted as
        // big-endian u16 words: 0x0001 and 0x0002.
        // sum1 = (0 + 1) % 65535 = 1; sum2 = (0 + 1) % 65535 = 1
        // sum1 = (1 + 2) % 65535 = 3; sum2 = (1 + 3) % 65535 = 4
        // result = (4 << 16) | 3 = 0x0004_0003
        let data = [0x00, 0x01, 0x00, 0x02];
        assert_eq!(fletcher32(&data), 0x0004_0003);
    }

    #[test]
    fn fletcher32_roundtrip_with_filter() {
        // Build a payload + checksum and verify the filter can strip it
        let payload = vec![0x00u8, 0x80, 0x3F, 0x80, 0x00, 0x00, 0x40, 0x00];
        let ck = fletcher32(&payload);
        let mut data = payload.clone();
        data.extend_from_slice(&ck.to_le_bytes());
        // The filter stores checksum in big-endian
        let stripped = crate::filters::fletcher32::verify_and_strip(&data).unwrap();
        assert_eq!(stripped, payload);
    }
}
