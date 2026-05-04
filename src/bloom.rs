//! # Bloom Filter Module
//!
//! Provides a fast, memory-efficient Bloom filter for first-pass fuzzy matching.
//!
//! Bloom filters provide O(1) probabilistic set membership testing with
//! a configurable false positive rate. They're used as a fast path before
//! more expensive operations like full SRAM scanning.

use crate::{Error, Result};

/// Optimal number of hash functions for a Bloom filter given capacity and false positive rate.
///
/// Uses the formula: k = (m/n) * ln(2)
///
/// where m = bit_count and n = capacity
fn optimal_hash_count(bit_count: usize, capacity: usize) -> usize {
    let m = bit_count as f64;
    let n = capacity as f64;
    let k = (m / n) * std::f64::consts::LN_2;
    k.max(1.0) as usize
}

/// Optimal bit count for a Bloom filter given capacity and desired false positive rate.
///
/// Uses the formula: m = -n * ln(p) / (ln(2)^2)
///
/// where n = capacity and p = false_positive_rate
fn optimal_bit_count(capacity: usize, false_positive_rate: f64) -> usize {
    let n = capacity as f64;
    let p = false_positive_rate;
    let m = -(n * p.ln()) / (std::f64::consts::LN_2.powi(2));
    m.ceil() as usize
}

/// A fast, memory-efficient Bloom filter for probabilistic set membership.
///
/// The Bloom filter provides O(1) insert and check operations with a tunable
/// false positive rate. It uses two hash functions (double hashing technique)
/// to simulate k independent hash functions.
#[derive(Debug, Clone)]
pub struct BloomFilter {
    /// The bit array
    bitfield: Vec<u64>,
    /// Number of simulated hash functions
    num_hashes: usize,
    /// Expected number of elements
    capacity: usize,
    /// Number of bits in the filter
    bit_count: usize,
}

impl BloomFilter {
    /// Create a new Bloom filter with optimal parameters for the given capacity
    /// and desired false positive rate.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Expected maximum number of elements
    /// * `false_positive_rate` - Desired false positive rate (e.g., 0.01 for 1%)
    ///
    /// # Returns
    ///
    /// A new BloomFilter instance
    ///
    /// # Example
    ///
    /// ```
    /// use superinstance_hdc_core::BloomFilter;
    ///
    /// let mut bloom = BloomFilter::with_capacity(1000, 0.01);
    /// bloom.insert(0xDEADBEEF);
    /// assert!(bloom.contains(0xDEADBEEF));
    /// ```
    pub fn with_capacity(capacity: usize, false_positive_rate: f64) -> Self {
        let bit_count = optimal_bit_count(capacity.max(1), false_positive_rate.max(1e-10));
        let num_hashes = optimal_hash_count(bit_count, capacity.max(1));
        let words_needed = (bit_count + 63) / 64;

        Self {
            bitfield: vec![0u64; words_needed],
            num_hashes,
            capacity,
            bit_count,
        }
    }

    /// Create a Bloom filter with explicit parameters.
    ///
    /// # Arguments
    ///
    /// * `capacity` - Expected maximum number of elements
    /// * `num_hashes` - Number of hash functions to use
    /// * `bit_count` - Total number of bits in the filter
    pub fn with_params(capacity: usize, num_hashes: usize, bit_count: usize) -> Self {
        let words_needed = (bit_count + 63) / 64;
        Self {
            bitfield: vec![0u64; words_needed],
            num_hashes,
            capacity,
            bit_count,
        }
    }

    /// Insert a 64-bit fingerprint into the filter.
    #[inline]
    pub fn insert(&mut self, fingerprint: u64) {
        for i in 0..self.num_hashes {
            let idx = self.hash_index(fingerprint, i);
            self.bitfield[idx / 64] |= 1u64 << (idx % 64);
        }
    }

    /// Check if a fingerprint might be in the set.
    ///
    /// Returns `true` if the fingerprint might be present (with false positive rate),
    /// `false` if it's definitely not present.
    #[inline]
    pub fn contains(&self, fingerprint: u64) -> bool {
        for i in 0..self.num_hashes {
            let idx = self.hash_index(fingerprint, i);
            if self.bitfield[idx / 64] & (1u64 << (idx % 64)) == 0 {
                return false;
            }
        }
        true
    }

    /// Compute the hash index for a given fingerprint and hash number.
    /// Uses double hashing: h(i) = h1 + i * h2
    #[inline]
    fn hash_index(&self, fingerprint: u64, i: usize) -> usize {
        // Use two independent hash functions via bit mixing
        let h1 = fingerprint as usize;
        let h2 = ((fingerprint >> 32) as usize) ^ (fingerprint.wrapping_mul(0x9E3779B9) as usize);
        let combined = h1.wrapping_add(i.wrapping_mul(h2));
        combined % self.bit_count
    }

    /// Get the expected false positive rate based on current fill.
    pub fn current_false_positive_rate(&self) -> f64 {
        let fill_count = self.bitfield.iter().map(|w| w.count_ones() as usize).sum::<usize>();
        let fill_ratio = fill_count as f64 / self.bit_count as f64;
        fill_ratio.powf(self.num_hashes as f64)
    }

    /// Clear all bits in the filter.
    pub fn clear(&mut self) {
        self.bitfield.fill(0);
    }

    /// Get the number of bits in the filter.
    pub fn bit_count(&self) -> usize {
        self.bit_count
    }

    /// Get the number of hash functions.
    pub fn num_hashes(&self) -> usize {
        self.num_hashes
    }

    /// Serialize the bitfield to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let bytes_per_word = 8;
        let total_bytes = self.bitfield.len() * bytes_per_word;
        let mut bytes = Vec::with_capacity(total_bytes + 16);
        
        // Header: num_hashes (4 bytes) + bit_count (4 bytes) + capacity (4 bytes) + reserved (4 bytes)
        bytes.extend_from_slice(&(self.num_hashes as u32).to_le_bytes());
        bytes.extend_from_slice(&(self.bit_count as u32).to_le_bytes());
        bytes.extend_from_slice(&(self.capacity as u32).to_le_bytes());
        bytes.extend_from_slice(&[0u8; 4]);
        
        // Bitfield data
        for word in &self.bitfield {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        
        bytes
    }

    /// Deserialize from bytes (created by to_bytes).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 16 {
            return Err(Error::Bloom("Buffer too small for bloom filter header".into()));
        }
        
        let num_hashes = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
        let bit_count = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
        let capacity = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        
        let words_needed = (bit_count + 63) / 64;
        let expected_len = 16 + words_needed * 8;
        
        if bytes.len() < expected_len {
            return Err(Error::Bloom("Buffer too small for bloom filter data".into()));
        }
        
        let mut bitfield = Vec::with_capacity(words_needed);
        for i in 0..words_needed {
            let offset = 16 + i * 8;
            let word = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
            bitfield.push(word);
        }
        
        Ok(Self {
            bitfield,
            num_hashes,
            capacity,
            bit_count,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bloom_insert_and_check() {
        let mut bloom = BloomFilter::with_capacity(100, 0.01);
        let fp = 0xDEADBEEF_u64;
        bloom.insert(fp);
        assert!(bloom.contains(fp), "Inserted fingerprint should be found");
    }

    #[test]
    fn test_bloom_definite_negatives() {
        let mut bloom = BloomFilter::with_capacity(1000, 0.001);
        bloom.insert(0x12345678);
        
        // These won't be found with very high probability
        // because they're definitely not inserted
        let mut not_found = 0;
        for i in 0..1000_u64 {
            if !bloom.contains(0x10000000 + i * 0x911) {
                not_found += 1;
            }
        }
        // With very low FPR, almost all should return false
        assert!(not_found > 990, "Most non-inserted items should not be found");
    }

    #[test]
    fn test_bloom_false_positive_rate() {
        let mut bloom = BloomFilter::with_capacity(1000, 0.1);
        
        // Insert 500 items
        for i in 0..500_u64 {
            bloom.insert(i * 0x9E3779B9);
        }
        
        // Check 500 non-inserted items
        let mut false_positives = 0;
        for i in 500..1000_u64 {
            if bloom.contains(i * 0x9E3779B9) {
                false_positives += 1;
            }
        }
        
        // Should be close to 10% FPR (allow some variance)
        let fpr = false_positives as f64 / 500.0;
        assert!(fpr < 0.2 && fpr > 0.02, "FPR should be around 10%, got {}", fpr);
    }

    #[test]
    fn test_bloom_clear() {
        let mut bloom = BloomFilter::with_capacity(100, 0.01);
        bloom.insert(0xDEAD);
        bloom.clear();
        assert!(!bloom.contains(0xDEAD), "Cleared bloom filter should not contain anything");
    }

    #[test]
    fn test_bloom_serialization() {
        let mut bloom = BloomFilter::with_capacity(100, 0.01);
        bloom.insert(0xDEADBEEF);
        bloom.insert(0xCAFEBABE);
        
        let bytes = bloom.to_bytes();
        let restored = BloomFilter::from_bytes(&bytes).unwrap();
        
        assert_eq!(bloom.num_hashes, restored.num_hashes);
        assert!(restored.contains(0xDEADBEEF));
        assert!(restored.contains(0xCAFEBABE));
    }

    #[test]
    fn test_bloom_empty() {
        let bloom = BloomFilter::with_capacity(100, 0.01);
        // Empty bloom should return false for everything (no false positives possible)
        assert!(!bloom.contains(0xDEAD));
    }

    #[test]
    fn test_bloom_very_low_fpr() {
        let mut bloom = BloomFilter::with_capacity(100, 0.0001);
        bloom.insert(42);
        
        let mut false_positives = 0;
        for i in 0..10000_u64 {
            if bloom.contains((i + 1000) * 0x12345) {
                false_positives += 1;
            }
        }
        
        // With 0.01% FPR, we expect ~1 false positive in 10000 checks
        assert!(false_positives < 10, "Very low FPR should produce few false positives");
    }
}
