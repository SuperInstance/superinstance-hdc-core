//! # Fingerprint Module
//!
//! Provides MurmurHash3-based 64-bit fingerprinting for text concepts.
//!
//! MurmurHash3 is a fast, non-crypto hash function ideal for generating
//! deterministic 64-bit fingerprints from arbitrary text.

use murmurhash::sum128with_seed;

/// Generate a 64-bit fingerprint for the given text using MurmurHash3.
///
/// This function uses the x64_128 variant and extracts the lower 64 bits
/// to produce a fast, deterministic hash suitable for Bloom filters and
/// SRAM record matching.
///
/// # Arguments
///
/// * `text` - The input text to fingerprint
/// * `seed` - A seed value to differentiate fingerprints for the same text
///
/// # Returns
///
/// A 64-bit unsigned integer fingerprint
///
/// # Example
///
/// ```
/// use superinstance_hdc_core::fingerprint;
///
/// let fp = fingerprint("hello world", 0xDEADBEEF);
/// assert_eq!(fp, fingerprint("hello world", 0xDEADBEEF));
/// ```
#[inline]
pub fn fingerprint(text: &str, seed: u64) -> u64 {
    // Use MurmurHash3 x64_128 and take lower 64 bits
    let (low, _high) = sum128with_seed(text.as_bytes(), seed);
    low
}

/// Generate multiple fingerprints for the same text with different seeds.
/// Useful for generating atomic fingerprints that will be bundled into
/// a hypervector.
pub fn multi_fingerprint(text: &str, base_seed: u64, count: usize) -> Vec<u64> {
    (0..count).map(|i| fingerprint(text, base_seed.wrapping_add(i as u64))).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fingerprint_empty_string() {
        let fp = fingerprint("", 0xDEAD);
        assert_ne!(fp, 0, "Empty string should still produce a hash");
    }

    #[test]
    fn test_fingerprint_deterministic() {
        let text = "test input";
        let seed = 0x12345678;
        let fp1 = fingerprint(text, seed);
        let fp2 = fingerprint(text, seed);
        assert_eq!(fp1, fp2, "Must be deterministic");
    }

    #[test]
    fn test_fingerprint_seed_sensitivity() {
        let text = "same text";
        let fp1 = fingerprint(text, 0x1111);
        let fp2 = fingerprint(text, 0x2222);
        assert_ne!(fp1, fp2, "Different seeds must produce different fingerprints");
    }

    #[test]
    fn test_fingerprint_unicode() {
        let fp1 = fingerprint("héllo", 0xDEAD);
        let fp2 = fingerprint("héllo", 0xDEAD);
        assert_eq!(fp1, fp2, "Unicode must be handled correctly");
    }

    #[test]
    fn test_fingerprint_long_text() {
        let text = "a".repeat(10000);
        let fp = fingerprint(&text, 0xDEAD);
        assert_ne!(fp, 0, "Long text must produce valid hash");
    }

    #[test]
    fn test_multi_fingerprint() {
        let fps = multi_fingerprint("test", 0x1000, 5);
        assert_eq!(fps.len(), 5);
        // Each should be different (with very high probability)
        for i in 0..fps.len() {
            for j in (i + 1)..fps.len() {
                assert_ne!(fps[i], fps[j]);
            }
        }
    }
}
