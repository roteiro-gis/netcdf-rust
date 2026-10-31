//! HDF5 SZIP filter (filter ID 4). Decoding only.
//!
//! SZIP is the extended-Rice coding of CCSDS 121.0-B. On disk a chunk is:
//! - `orig_size: u32 LE`, the decoded byte length
//! - a bit stream of reference sample intervals (RSIs), each a run of blocks
//!
//! Each block starts with an option id that selects how its samples are coded. With
//! nearest-neighbour preprocessing, samples are mapped differences from the previous sample, and
//! each RSI starts with a raw reference sample. The framing around the bit stream follows libaec:
//! 32- and 64-bit pixels are coded as 8-bit samples, one byte plane at a time, and each scanline
//! is one RSI, padded to a whole number of blocks.

use super::zeroed_output;
use crate::error::{Error, Result};

const OPTION_MSB: u32 = 16;
const OPTION_NEAREST_NEIGHBOR: u32 = 32;

/// The zero-block count that means the rest of the segment, not a number of blocks.
const REMAINDER_OF_SEGMENT: u64 = 5;
/// A zero-block run stops at the next multiple of this many blocks.
const ZERO_BLOCK_SEGMENT: usize = 64;
/// Second-extension codes run through the pairs with a sum below this.
const SECOND_EXTENSION_MAX_SUM: u64 = 13;
/// The largest block the format allows.
const MAX_PIXELS_PER_BLOCK: u32 = 32;
/// The longest RSI the format allows.
const MAX_BLOCKS_PER_SCANLINE: u32 = 4096;

/// The four client-data values HDF5 stores for the filter.
#[derive(Clone, Copy)]
struct Parameters {
    msb: bool,
    preprocessed: bool,
    pixels_per_block: usize,
    bits_per_pixel: u32,
    pixels_per_scanline: usize,
}

impl Parameters {
    fn parse(client_data: &[u32]) -> Result<Self> {
        let &[options, pixels_per_block, bits_per_pixel, pixels_per_scanline] = client_data else {
            return Err(error(format!(
                "expected 4 client data values, got {}",
                client_data.len()
            )));
        };
        if !(1..=MAX_PIXELS_PER_BLOCK).contains(&pixels_per_block)
            || pixels_per_scanline == 0
            || pixels_per_scanline.div_ceil(pixels_per_block) > MAX_BLOCKS_PER_SCANLINE
        {
            return Err(error(format!(
                "invalid sizes: {pixels_per_block} pixels per block, \
                 {pixels_per_scanline} per scanline"
            )));
        }
        if !matches!(bits_per_pixel, 1..=32 | 64) {
            return Err(error(format!("unsupported {bits_per_pixel}-bit pixels")));
        }
        Ok(Self {
            msb: options & OPTION_MSB != 0,
            preprocessed: options & OPTION_NEAREST_NEIGHBOR != 0,
            pixels_per_block: pixels_per_block as usize,
            bits_per_pixel,
            pixels_per_scanline: pixels_per_scanline as usize,
        })
    }

    /// Pixels of 32 and 64 bits are coded a byte plane at a time.
    fn is_interleaved(&self) -> bool {
        matches!(self.bits_per_pixel, 32 | 64)
    }

    fn bits_per_sample(&self) -> u32 {
        if self.is_interleaved() {
            8
        } else {
            self.bits_per_pixel
        }
    }

    fn bytes_per_sample(&self) -> usize {
        match self.bits_per_sample() {
            1..=8 => 1,
            9..=16 => 2,
            _ => 4,
        }
    }

    fn blocks_per_rsi(&self) -> usize {
        self.pixels_per_scanline.div_ceil(self.pixels_per_block)
    }
}

/// Decompress SZIP-filtered data, rejecting declared output sizes above `max_output_len` before
/// allocating the output buffer.
pub fn decompress_with_limit(
    data: &[u8],
    client_data: &[u32],
    max_output_len: usize,
) -> Result<Vec<u8>> {
    let parameters = Parameters::parse(client_data)?;
    let (size, stream) = data
        .split_first_chunk::<4>()
        .ok_or_else(|| error("input too short for header"))?;
    let output_len = u32::from_le_bytes(*size) as usize;
    if output_len > max_output_len {
        return Err(error(format!(
            "declared output size {output_len} exceeds the limit of {max_output_len}"
        )));
    }

    let pixel_bytes = if parameters.is_interleaved() {
        (parameters.bits_per_pixel / 8) as usize
    } else {
        parameters.bytes_per_sample()
    };
    if output_len % pixel_bytes != 0 {
        return Err(error(format!(
            "output size {output_len} is not a whole number of {pixel_bytes}-byte pixels"
        )));
    }

    let mut output = zeroed_output(output_len, "SZIP")?;
    Decoder::new(stream, parameters).decode(&mut output)?;
    if parameters.is_interleaved() {
        output = deinterleave(&output, pixel_bytes)?;
    }
    Ok(output)
}

fn error(message: impl std::fmt::Display) -> Error {
    Error::DecompressionError(format!("SZIP: {message}"))
}

/// Reassemble pixels from their byte planes, the first byte of every pixel coming first.
fn deinterleave(planes: &[u8], pixel_bytes: usize) -> Result<Vec<u8>> {
    let pixels = planes.len() / pixel_bytes;
    let mut output = zeroed_output(planes.len(), "SZIP")?;
    for (byte, plane) in planes.chunks_exact(pixels.max(1)).enumerate() {
        for (pixel, &value) in plane.iter().enumerate() {
            output[pixel * pixel_bytes + byte] = value;
        }
    }
    Ok(output)
}

/// Reads the bit stream most significant bit first.
struct BitReader<'a> {
    data: &'a [u8],
    next: usize,
    /// Unread bits, starting at the top.
    buffer: u64,
    buffered: u32,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            next: 0,
            buffer: 0,
            buffered: 0,
        }
    }

    fn refill(&mut self) {
        while self.buffered <= 56 {
            let Some(&byte) = self.data.get(self.next) else {
                break;
            };
            self.buffer |= u64::from(byte) << (56 - self.buffered);
            self.buffered += 8;
            self.next += 1;
        }
    }

    fn consume(&mut self, bits: u32) {
        self.buffer = if bits == 64 { 0 } else { self.buffer << bits };
        self.buffered -= bits;
    }

    /// The next `bits` bits, at most 32.
    fn read(&mut self, bits: u32) -> Result<u32> {
        if bits == 0 {
            return Ok(0);
        }
        if self.buffered < bits {
            self.refill();
            if self.buffered < bits {
                return Err(error("input ended early"));
            }
        }
        let value = (self.buffer >> (64 - bits)) as u32;
        self.consume(bits);
        Ok(value)
    }

    /// A fundamental sequence: the number of 0 bits before the next 1 bit.
    fn read_fundamental(&mut self) -> Result<u64> {
        let mut zeros = 0u64;
        loop {
            if self.buffered == 0 {
                self.refill();
                if self.buffered == 0 {
                    return Err(error("input ended early"));
                }
            }
            let leading = self.buffer.leading_zeros();
            if leading < self.buffered {
                self.consume(leading + 1);
                return Ok(zeros + u64::from(leading));
            }
            zeros += u64::from(self.buffered);
            let buffered = self.buffered;
            self.consume(buffered);
        }
    }
}

struct Decoder<'a> {
    bits: BitReader<'a>,
    parameters: Parameters,
    sample_bits: u32,
    sample_max: u32,
    id_bits: u32,
    /// One RSI's samples, reused between intervals.
    segment: Vec<u32>,
}

impl<'a> Decoder<'a> {
    fn new(stream: &'a [u8], parameters: Parameters) -> Self {
        let sample_bits = parameters.bits_per_sample();
        Self {
            bits: BitReader::new(stream),
            parameters,
            sample_bits,
            sample_max: u32::MAX >> (32 - sample_bits),
            id_bits: match sample_bits {
                1..=8 => 3,
                9..=16 => 4,
                _ => 5,
            },
            segment: vec![0; parameters.blocks_per_rsi() * parameters.pixels_per_block],
        }
    }

    fn decode(&mut self, output: &mut [u8]) -> Result<()> {
        let sample_bytes = self.parameters.bytes_per_sample();
        let line_bytes = self.parameters.pixels_per_scanline * sample_bytes;
        for line in output.chunks_mut(line_bytes) {
            // A whole scanline consumes its RSI's padding too; the last can stop early.
            let wanted = if line.len() == line_bytes {
                self.segment.len()
            } else {
                line.len().div_ceil(sample_bytes)
            };
            let filled = self.decode_segment(wanted)?;
            if self.parameters.preprocessed {
                undo_prediction(&mut self.segment[..filled], self.sample_max);
            }
            for (out, &sample) in line.chunks_mut(sample_bytes).zip(&self.segment) {
                if self.parameters.msb {
                    let be = sample.to_be_bytes();
                    out.copy_from_slice(&be[4 - sample_bytes..][..out.len()]);
                } else {
                    out.copy_from_slice(&sample.to_le_bytes()[..out.len()]);
                }
            }
        }
        Ok(())
    }

    /// Decode blocks of one RSI until at least `wanted` samples are filled, returning how many.
    fn decode_segment(&mut self, wanted: usize) -> Result<usize> {
        let block_size = self.parameters.pixels_per_block;
        let uncompressed_id = (1u32 << self.id_bits) - 1;
        let mut filled = 0;
        let mut has_reference = self.parameters.preprocessed;

        while filled < wanted {
            let id = self.bits.read(self.id_bits)?;
            if id == uncompressed_id {
                self.read_samples(filled, block_size)?;
                filled += block_size;
            } else if id == 0 {
                let second_extension = self.bits.read(1)? == 1;
                if has_reference {
                    self.read_samples(filled, 1)?;
                }
                filled = if second_extension {
                    self.decode_second_extension(filled, has_reference)?
                } else {
                    self.decode_zero_blocks(filled, has_reference)?
                };
            } else {
                let start = filled;
                if has_reference {
                    self.read_samples(filled, 1)?;
                    filled += 1;
                }
                filled = self.decode_split(filled, start + block_size, id - 1)?;
            }
            has_reference = false;
        }
        Ok(filled)
    }

    fn read_samples(&mut self, start: usize, count: usize) -> Result<()> {
        let sample_bits = self.sample_bits;
        let samples = self
            .segment
            .get_mut(start..start + count)
            .ok_or_else(|| error("block runs past its interval"))?;
        for sample in samples {
            *sample = self.bits.read(sample_bits)?;
        }
        Ok(())
    }

    fn decode_split(&mut self, start: usize, end: usize, split_bits: u32) -> Result<usize> {
        if split_bits >= self.sample_bits {
            return Err(error(format!("invalid split of {split_bits} bits")));
        }
        let sample_max = self.sample_max;
        let bits = &mut self.bits;
        let samples = self
            .segment
            .get_mut(start..end)
            .ok_or_else(|| error("block runs past its interval"))?;
        for sample in samples.iter_mut() {
            *sample = ((bits.read_fundamental()? << split_bits) & u64::from(sample_max)) as u32;
        }
        for sample in samples {
            *sample |= bits.read(split_bits)?;
        }
        Ok(end)
    }

    fn decode_second_extension(
        &mut self,
        block_start: usize,
        has_reference: bool,
    ) -> Result<usize> {
        let block_size = self.parameters.pixels_per_block;
        let end = block_start + block_size;
        if end > self.segment.len() {
            return Err(error("block runs past its interval"));
        }

        let mut position = block_start + usize::from(has_reference);
        while position < end {
            let (first, second) = second_extension_pair(self.bits.read_fundamental()?)?;
            // A pair starts on an even position, so a reference sample leaves its first half out.
            if (position - block_start) % 2 == 0 {
                self.segment[position] = first;
                position += 1;
                if position == end {
                    break;
                }
            }
            self.segment[position] = second;
            position += 1;
        }
        Ok(end)
    }

    fn decode_zero_blocks(&mut self, block_start: usize, has_reference: bool) -> Result<usize> {
        let block_size = self.parameters.pixels_per_block;
        let blocks_per_rsi = self.parameters.blocks_per_rsi();
        let code = self.bits.read_fundamental()? + 1;

        let block = block_start / block_size;
        let blocks = if code == REMAINDER_OF_SEGMENT {
            (blocks_per_rsi - block).min(ZERO_BLOCK_SEGMENT - block % ZERO_BLOCK_SEGMENT)
        } else {
            let count = if code > REMAINDER_OF_SEGMENT {
                code - 1
            } else {
                code
            };
            usize::try_from(count).map_err(|_| error("zero block run overflows"))?
        };

        let end = blocks
            .checked_mul(block_size)
            .and_then(|samples| samples.checked_add(block_start))
            .filter(|&end| end <= self.segment.len())
            .ok_or_else(|| error("zero block run passes its interval"))?;
        self.segment[block_start + usize::from(has_reference)..end].fill(0);
        Ok(end)
    }
}

/// The two mapped residuals a second-extension code stands for. Codes count through the pairs by
/// their sum `s`, then by `second` within a sum, so a sum starts at code `s * (s + 1) / 2`.
fn second_extension_pair(code: u64) -> Result<(u32, u32)> {
    let mut sum_start = 0;
    for sum in 0..SECOND_EXTENSION_MAX_SUM {
        if code <= sum_start + sum {
            let second = (code - sum_start) as u32;
            return Ok((sum as u32 - second, second));
        }
        sum_start += sum + 1;
    }
    Err(error(format!("invalid second extension code {code}")))
}

/// Undo nearest-neighbour preprocessing: `samples[0]` is the reference sample, and each other
/// sample is its value's mapped difference from the one before.
fn undo_prediction(samples: &mut [u32], sample_max: u32) {
    let midpoint = sample_max / 2 + 1;
    let Some(mut previous) = samples.first().copied() else {
        return;
    };
    for sample in &mut samples[1..] {
        let mapped = *sample;
        let magnitude = (mapped >> 1) + (mapped & 1);
        let headroom = if previous < midpoint {
            previous
        } else {
            sample_max - previous
        };
        previous = if magnitude <= headroom {
            if mapped & 1 == 1 {
                previous - magnitude
            } else {
                previous + magnitude
            }
        } else if previous < midpoint {
            mapped
        } else {
            sample_max - mapped
        };
        *sample = previous;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RAW: u32 = 128;
    const LSB: u32 = 8;
    const NN: u32 = OPTION_NEAREST_NEIGHBOR;

    /// Writes bit streams most significant bit first.
    #[derive(Default)]
    struct BitWriter {
        bytes: Vec<u8>,
        used: u32,
    }

    impl BitWriter {
        fn write(&mut self, value: u32, bits: u32) {
            for bit in (0..bits).rev() {
                if self.used % 8 == 0 {
                    self.bytes.push(0);
                }
                let last = self.bytes.len() - 1;
                self.bytes[last] |= (((value >> bit) & 1) as u8) << (7 - self.used % 8);
                self.used += 1;
            }
        }

        fn fundamental(&mut self, zeros: u32) {
            for _ in 0..zeros {
                self.write(0, 1);
            }
            self.write(1, 1);
        }

        fn into_filtered(self, output_len: u32) -> Vec<u8> {
            let mut filtered = output_len.to_le_bytes().to_vec();
            filtered.extend(self.bytes);
            filtered
        }
    }

    #[test]
    fn second_extension_codes_count_through_pairs_by_sum() {
        assert_eq!(second_extension_pair(0).unwrap(), (0, 0));
        assert_eq!(second_extension_pair(1).unwrap(), (1, 0));
        assert_eq!(second_extension_pair(2).unwrap(), (0, 1));
        assert_eq!(second_extension_pair(3).unwrap(), (2, 0));
        assert_eq!(second_extension_pair(5).unwrap(), (0, 2));
        assert_eq!(second_extension_pair(90).unwrap(), (0, 12));
        assert!(second_extension_pair(91).is_err());
    }

    #[test]
    fn prediction_is_undone_at_both_ends_of_the_range() {
        let mut samples = [100, 4, 3, 255];
        undo_prediction(&mut samples, 255);
        // +2, then -2, then past the room below 100 the mapped value is the sample itself.
        assert_eq!(samples, [100, 102, 100, 255]);

        let mut samples = [250, 20, 200];
        undo_prediction(&mut samples, 255);
        // 250 has 5 of room above it, so the mapped value counts down from the maximum.
        assert_eq!(samples, [250, 235, 55]);
    }

    #[test]
    fn uncompressed_blocks_hold_raw_samples() {
        let mut stream = BitWriter::default();
        stream.write(0b111, 3);
        for value in [10, 20, 30, 40] {
            stream.write(value, 8);
        }
        let output =
            decompress_with_limit(&stream.into_filtered(4), &[RAW | LSB, 4, 8, 4], usize::MAX)
                .unwrap();
        assert_eq!(output, [10, 20, 30, 40]);
    }

    #[test]
    fn padding_past_a_scanline_is_dropped() {
        // Six pixels a scanline take two blocks of four.
        let mut stream = BitWriter::default();
        for line in [[1, 2, 3, 4, 5, 6, 77, 88], [7, 8, 9, 10, 11, 12, 99, 99]] {
            for block in line.chunks(4) {
                stream.write(0b111, 3);
                for &value in block {
                    stream.write(value, 8);
                }
            }
        }
        let output =
            decompress_with_limit(&stream.into_filtered(12), &[RAW | LSB, 4, 8, 6], usize::MAX)
                .unwrap();
        assert_eq!(output, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]);
    }

    #[test]
    fn split_blocks_join_fundamental_and_low_bits() {
        let mut stream = BitWriter::default();
        stream.write(2, 3); // id 2: one split bit
        for high in [0, 1, 2, 3] {
            stream.fundamental(high);
        }
        for low in [1, 0, 1, 0] {
            stream.write(low, 1);
        }
        let output =
            decompress_with_limit(&stream.into_filtered(4), &[RAW | LSB, 4, 8, 4], usize::MAX)
                .unwrap();
        assert_eq!(output, [1, 2, 5, 6]);
    }

    #[test]
    fn zero_blocks_can_cover_the_rest_of_an_interval() {
        let mut stream = BitWriter::default();
        stream.write(0, 3);
        stream.write(0, 1); // zero blocks, not second extension
        stream.write(9, 8); // reference sample
        stream.fundamental(4); // remainder of segment
        let output = decompress_with_limit(
            &stream.into_filtered(8),
            &[RAW | LSB | NN, 4, 8, 8],
            usize::MAX,
        )
        .unwrap();
        assert_eq!(output, [9; 8]);
    }

    #[test]
    fn wide_pixels_come_back_from_their_byte_planes() {
        assert_eq!(
            deinterleave(&[1, 2, 3, 4, 5, 6], 2).unwrap(),
            [1, 4, 2, 5, 3, 6]
        );
    }

    #[test]
    fn declared_sizes_past_the_limit_are_refused_before_allocating() {
        let filtered = u32::MAX.to_le_bytes();
        let err = decompress_with_limit(&filtered, &[RAW | LSB, 8, 8, 64], 1024).unwrap_err();
        assert!(err.to_string().contains("exceeds the limit"), "{err}");
    }

    #[test]
    fn output_sizes_must_be_whole_pixels() {
        let filtered = 6u32.to_le_bytes();
        let err =
            decompress_with_limit(&filtered, &[RAW | LSB, 8, 32, 64], usize::MAX).unwrap_err();
        assert!(err.to_string().contains("whole number"), "{err}");
    }

    #[test]
    fn truncated_streams_are_errors() {
        let mut stream = BitWriter::default();
        stream.write(0b111, 3);
        stream.write(1, 8);
        let err =
            decompress_with_limit(&stream.into_filtered(4), &[RAW | LSB, 4, 8, 4], usize::MAX)
                .unwrap_err();
        assert!(err.to_string().contains("ended early"), "{err}");
    }

    #[test]
    fn invalid_parameters_are_errors() {
        for client_data in [
            &[0, 8, 8][..],
            &[0, 0, 8, 8][..],
            &[0, 33, 8, 33][..],
            &[0, 8, 0, 8][..],
            &[0, 8, 33, 8][..],
            &[0, 8, 8, 0][..],
            &[0, 8, 8, 8 * 4096 + 1][..],
        ] {
            assert!(decompress_with_limit(&[0; 8], client_data, usize::MAX).is_err());
        }
    }

    #[test]
    fn zero_block_runs_cannot_pass_their_interval() {
        let mut stream = BitWriter::default();
        stream.write(0, 3);
        stream.write(0, 1);
        stream.fundamental(30);
        let filtered = stream.into_filtered(8);
        let err = decompress_with_limit(&filtered, &[RAW | LSB, 4, 8, 8], usize::MAX).unwrap_err();
        assert!(err.to_string().contains("passes its interval"), "{err}");
    }
}
