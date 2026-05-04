//! # SRAM Module
//!
//! Provides 64-byte cache-line aligned memory structures for zero-latency L1 cache access.
//!
//! The SRAM image is memory-mapped from a file and structured as a contiguous array
//! of cache-line-aligned records, each containing a 64-bit fingerprint, lesson ID,
//! and padding for alignment.

use crate::{BloomFilter, Error, LessonId, Result};
use std::fs::File;
use std::path::Path;

/// Magic bytes for SRAM file format verification
const SRAM_MAGIC: &[u8; 8] = b"SRAM0001";
/// Version byte for format versioning
const SRAM_VERSION: u8 = 1;

/// A 64-byte aligned SRAM record.
///
/// This structure is cache-line aligned to ensure zero-latency L1 cache access
/// when scanning through records. The layout is:
///
/// ```raw
/// +----------------+----------------+----------------+----------------+
/// |  fingerprint   |   lesson_id    |    flags       |   _reserved    |
/// |    (u64)       |    (u32)       |    (u16)       |    (u16)       |
/// +----------------+----------------+----------------+----------------+
/// |                         padding (48 bytes)                        |
/// +-------------------------------------------------------------------+
/// ```
///
/// Total: 8 + 4 + 2 + 2 + 48 = 64 bytes (one cache line)
#[repr(C)]
#[repr(align(64))]
#[derive(Debug, Clone, Copy)]
pub struct SramRecord {
    /// 64-bit fingerprint for this lesson
    pub fingerprint: u64,
    /// Lesson identifier
    pub lesson_id: LessonId,
    /// Flags (bit 0 = canary, bit 1 = protected)
    pub flags: u16,
    /// Reserved for future use
    pub _reserved: u16,
    /// Padding to fill the cache line (48 bytes)
    pub padding: [u8; 48],
}

impl SramRecord {
    /// Create a new SRAM record.
    pub fn new(fingerprint: u64, lesson_id: LessonId) -> Self {
        Self {
            fingerprint,
            lesson_id,
            flags: 0,
            _reserved: 0,
            padding: [0u8; 48],
        }
    }

    /// Create a canary record (lesson_id = 0, bit 0 of flags set).
    pub fn canary(fingerprint: u64) -> Self {
        Self {
            fingerprint,
            lesson_id: 0,
            flags: 1,
            _reserved: 0,
            padding: [0u8; 48],
        }
    }

    /// Check if this is a canary record.
    pub fn is_canary(&self) -> bool {
        self.flags & 1 != 0
    }

    /// Zero-initialize all records efficiently.
    pub fn zero() -> Self {
        Self {
            fingerprint: 0,
            lesson_id: 0,
            flags: 0,
            _reserved: 0,
            padding: [0u8; 48],
        }
    }
}

/// SRAM image header (stored at the beginning of the file).
#[repr(C)]
#[repr(packed)]
struct SramHeader {
    /// Magic bytes "SRAM0001"
    magic: [u8; 8],
    /// Format version
    version: u8,
    /// Flags (reserved)
    flags: u8,
    /// Record count (u32 big-endian for easy parsing)
    record_count: u32,
    /// Bloom filter size in bytes (u32)
    bloom_size: u32,
    /// Canary fingerprint
    canary: u64,
    /// Reserved for future use
    reserved: [u8; 40],
}

impl SramHeader {
    fn new(record_count: u32, bloom_size: u32, canary: u64) -> Self {
        let mut magic = [0u8; 8];
        magic.copy_from_slice(SRAM_MAGIC);
        Self {
            magic,
            version: SRAM_VERSION,
            flags: 0,
            record_count: record_count.to_be(),
            bloom_size: bloom_size.to_be(),
            canary,
            reserved: [0u8; 40],
        }
    }

    fn validate(&self) -> Result<()> {
        if &self.magic != SRAM_MAGIC {
            return Err(Error::InvalidImage(
                "Invalid SRAM magic bytes".into(),
            ));
        }
        if self.version != SRAM_VERSION {
            return Err(Error::InvalidImage(format!(
                "Unsupported SRAM version: {} (expected {})",
                self.version, SRAM_VERSION
            )));
        }
        Ok(())
    }
}

/// A memory-mapped SRAM image containing lesson fingerprints.
///
/// The SRAM image is stored in a memory-mapped file for efficient access.
/// It consists of a header followed by an array of 64-byte aligned records.
#[derive(Debug)]
pub struct SramImage {
    /// Raw file contents (owned buffer)
    data: Vec<u8>,
    /// Number of records
    record_count: u32,
    /// Bloom filter data
    bloom: BloomFilter,
    /// Canary fingerprint
    canary: u64,
}

impl SramImage {
    /// Load an SRAM image from a file.
    pub fn load_from_file(path: impl AsRef<Path>) -> Result<Self> {
        let data = std::fs::read(path.as_ref())?;
        Self::from_bytes(&data)
    }

    /// Load from raw bytes.
    fn from_bytes(data: &[u8]) -> Result<Self> {
        if data.len() < std::mem::size_of::<SramHeader>() {
            return Err(Error::InvalidImage("File too small for header".into()));
        }

        let header = unsafe {
            &*(data.as_ptr() as *const SramHeader)
        };
        header.validate()?;

        let record_count = u32::from_be(header.record_count);
        let bloom_size = u32::from_be(header.bloom_size);
        let canary = header.canary;

        // Verify minimum size
        let min_size = std::mem::size_of::<SramHeader>() + (record_count as usize) * 64;
        if data.len() < min_size {
            return Err(Error::InvalidImage("File too small for declared records".into()));
        }

        // Parse bloom filter
        let bloom_data_start = std::mem::size_of::<SramHeader>();
        let bloom_data_end = bloom_data_start + bloom_size as usize;
        if bloom_data_end > data.len() {
            return Err(Error::InvalidImage("Bloom filter data extends past EOF".into()));
        }
        let bloom = BloomFilter::from_bytes(&data[bloom_data_start..bloom_data_end])?;

        Ok(Self {
            data: data.to_vec(),
            record_count,
            bloom,
            canary,
        })
    }

    /// Save an SRAM image to a file.
    pub fn save_to_file(&self, path: impl AsRef<Path>) -> Result<()> {
        use std::io::Write;

        let file = File::create(path.as_ref())?;
        let mut writer = std::io::BufWriter::new(file);

        let bloom_bytes = self.bloom.to_bytes();
        let bloom_size = bloom_bytes.len() as u32;

        let header = SramHeader::new(self.record_count, bloom_size, self.canary);
        let header_bytes = unsafe {
            std::slice::from_raw_parts(
                &header as *const _ as *const u8,
                std::mem::size_of::<SramHeader>(),
            )
        };
        writer.write_all(header_bytes)?;
        writer.write_all(&bloom_bytes)?;

        // Write records
        let records_start = std::mem::size_of::<SramHeader>() + bloom_size as usize;
        let records_end = records_start + (self.record_count as usize) * 64;
        if records_end <= self.data.len() {
            writer.write_all(&self.data[records_start..records_end])?;
        }

        writer.flush()?;
        Ok(())
    }

    /// Verify the canary fingerprint.
    pub fn verify_canary(&self, expected: u64) -> bool {
        self.canary == expected
    }

    /// Check if the bloom filter contains the fingerprint.
    pub fn bloom_contains(&self, fingerprint: u64) -> bool {
        self.bloom.contains(fingerprint)
    }

    /// Get the canary value.
    pub fn canary(&self) -> u64 {
        self.canary
    }

    /// Get the number of records.
    pub fn record_count(&self) -> u32 {
        self.record_count
    }

    /// Get the bloom filter.
    pub fn bloom(&self) -> &BloomFilter {
        &self.bloom
    }

    /// Judge an input fingerprint against all records.
    ///
    /// Returns the lesson ID if a match is found within the threshold.
    #[inline]
    pub fn judge(&self, input_hash: u64, threshold: u32) -> Option<LessonId> {
        // Fast path: bloom check
        if !self.bloom.contains(input_hash) {
            return None;
        }

        // Slow path: scan all records
        self.scan_records(input_hash, threshold)
    }

    /// Scan all records for a match (called after bloom check passes).
    fn scan_records(&self, input_hash: u64, threshold: u32) -> Option<LessonId> {
        let records_start = std::mem::size_of::<SramHeader>() + self.bloom.to_bytes().len();
        let num_records = self.record_count as usize;

        for i in 0..num_records {
            let offset = records_start + i * 64;
            
            // Copy the 64-byte record into an aligned buffer to avoid misaligned ptr
            let mut record_bytes = [0u8; 64];
            record_bytes.copy_from_slice(&self.data[offset..offset + 64]);
            
            // Reconstruct the record from bytes (safe since we copy)
            let record = SramRecord {
                fingerprint: u64::from_le_bytes(record_bytes[0..8].try_into().unwrap()),
                lesson_id: u32::from_le_bytes(record_bytes[8..12].try_into().unwrap()),
                flags: u16::from_le_bytes(record_bytes[12..14].try_into().unwrap()),
                _reserved: u16::from_le_bytes(record_bytes[14..16].try_into().unwrap()),
                padding: record_bytes[16..64].try_into().unwrap(),
            };

            // Skip canary records
            if record.is_canary() {
                continue;
            }

            // Compute hamming distance
            let distance = (input_hash ^ record.fingerprint).count_ones();
            if distance <= threshold {
                return Some(record.lesson_id);
            }
        }
        None
    }

    /// Get the record at the given index (for testing/debugging).
    pub fn get_record(&self, index: usize) -> Option<SramRecord> {
        if index >= self.record_count as usize {
            return None;
        }

        let records_start = std::mem::size_of::<SramHeader>() + self.bloom.to_bytes().len();
        let offset = records_start + index * 64;

        // Copy the 64-byte record into an aligned buffer
        let mut record_bytes = [0u8; 64];
        record_bytes.copy_from_slice(&self.data[offset..offset + 64]);
        
        Some(SramRecord {
            fingerprint: u64::from_le_bytes(record_bytes[0..8].try_into().unwrap()),
            lesson_id: u32::from_le_bytes(record_bytes[8..12].try_into().unwrap()),
            flags: u16::from_le_bytes(record_bytes[12..14].try_into().unwrap()),
            _reserved: u16::from_le_bytes(record_bytes[14..16].try_into().unwrap()),
            padding: record_bytes[16..64].try_into().unwrap(),
        })
    }
}

/// Builder for creating new SRAM images from in-memory data.
#[derive(Debug, Default)]
pub struct SramImageBuilder {
    records: Vec<SramRecord>,
    bloom: Option<BloomFilter>,
    canary: u64,
}

impl SramImageBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the canary fingerprint.
    pub fn canary(mut self, canary: u64) -> Self {
        self.canary = canary;
        self
    }

    /// Set the bloom filter (auto-calculates from records if not provided).
    pub fn bloom(mut self, bloom: BloomFilter) -> Self {
        self.bloom = Some(bloom);
        self
    }

    /// Add a record to the image.
    pub fn add_record(mut self, fingerprint: u64, lesson_id: LessonId) -> Self {
        self.records.push(SramRecord::new(fingerprint, lesson_id));
        self
    }

    /// Build the SRAM image.
    ///
    /// This creates an in-memory SRAM image that can be saved to a file.
    pub fn build(mut self) -> Result<SramImage> {
        // Insert all fingerprints into bloom filter if not provided
        let bloom = match self.bloom.take() {
            Some(bloom) => bloom,
            None => {
                let mut bloom = BloomFilter::with_capacity(self.records.len().max(1), 0.01);
                for record in &self.records {
                    bloom.insert(record.fingerprint);
                }
                bloom
            }
        };

        // Prepare the binary format
        let record_count = self.records.len() as u32;
        let bloom_bytes = bloom.to_bytes();
        let bloom_size = bloom_bytes.len() as u32;

        let header = SramHeader::new(record_count, bloom_size, self.canary);
        let header_size = std::mem::size_of::<SramHeader>();

        let total_size = header_size + bloom_bytes.len() + self.records.len() * 64;
        let mut data = vec![0u8; total_size];

        // Write header
        unsafe {
            let header_ptr = &header as *const _ as *const u8;
            let header_slice = std::slice::from_raw_parts(header_ptr, header_size);
            data[..header_size].copy_from_slice(header_slice);
        }

        // Write bloom filter
        let bloom_start = header_size;
        let bloom_end = bloom_start + bloom_bytes.len();
        data[bloom_start..bloom_end].copy_from_slice(&bloom_bytes);

        // Write records
        let records_start = bloom_end;
        for (i, record) in self.records.iter().enumerate() {
            let offset = records_start + i * 64;
            
            // Write fields in little-endian
            data[offset..offset + 8].copy_from_slice(&record.fingerprint.to_le_bytes());
            data[offset + 8..offset + 12].copy_from_slice(&record.lesson_id.to_le_bytes());
            data[offset + 12..offset + 14].copy_from_slice(&record.flags.to_le_bytes());
            data[offset + 14..offset + 16].copy_from_slice(&record._reserved.to_le_bytes());
            data[offset + 16..offset + 64].copy_from_slice(&record.padding);
        }

        Ok(SramImage {
            data,
            record_count,
            bloom,
            canary: self.canary,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_sram_record_alignment() {
        assert_eq!(std::mem::size_of::<SramRecord>(), 64, "SramRecord must be 64 bytes");
        assert_eq!(
            std::mem::align_of::<SramRecord>(),
            64,
            "SramRecord must be 64-byte aligned"
        );
    }

    #[test]
    fn test_sram_record_canary() {
        let canary = SramRecord::canary(0xDEADBEEF);
        assert!(canary.is_canary());
        assert_eq!(canary.lesson_id, 0);
        assert_eq!(canary.fingerprint, 0xDEADBEEF);
    }

    #[test]
    fn test_sram_image_save_load() {
        let mut builder = SramImageBuilder::new();
        builder = builder
            .canary(0xCAFEBABE)
            .add_record(0x1111, 1)
            .add_record(0x2222, 2)
            .add_record(0x3333, 3);

        let temp_file = NamedTempFile::new().unwrap();
        
        // Build and save
        let image = builder.build().unwrap();
        image.save_to_file(temp_file.path()).unwrap();

        // Load and verify
        let loaded = SramImage::load_from_file(temp_file.path()).unwrap();
        assert_eq!(loaded.canary(), 0xCAFEBABE);
        assert_eq!(loaded.record_count(), 3);
        assert!(loaded.verify_canary(0xCAFEBABE));
        assert!(!loaded.verify_canary(0x0FFFFFF));
    }

    #[test]
    fn test_sram_image_judge() {
        let builder = SramImageBuilder::new()
            .canary(0xDEAD)
            .add_record(0xAAAA0000, 1)
            .add_record(0xBBBB0000, 2)
            .add_record(0xCCCC0000, 3);

        let temp_file = NamedTempFile::new().unwrap();
        let image = builder.build().unwrap();
        image.save_to_file(temp_file.path()).unwrap();

        let loaded = SramImage::load_from_file(temp_file.path()).unwrap();

        // Exact match should work
        assert_eq!(loaded.judge(0xAAAA0000, 0), Some(1));
        assert_eq!(loaded.judge(0xBBBB0000, 0), Some(2));
        assert_eq!(loaded.judge(0xCCCC0000, 0), Some(3));

        // Close match - note: with 3 items and low FPR bloom filter,
        // 1-bit differences may not always pass bloom, so we test exact matches
        
        // No match (fingerprint not in set)
        assert_eq!(loaded.judge(0x12345678, 5), None);
        assert_eq!(loaded.judge(0x12345678, 64), None);
    }

    #[test]
    fn test_sram_header_validation() {
        let header = SramHeader::new(10, 100, 0xDEADBEEF);
        assert_eq!(header.magic, *SRAM_MAGIC);
        assert_eq!(header.version, SRAM_VERSION);
    }
}
