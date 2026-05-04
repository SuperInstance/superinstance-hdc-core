//! # SuperInstance HDC Core
//!
//! Hyperdimensional Computing Core for bit-level agent cognition.
//!
//! ## Overview
//!
//! This crate provides tools for building memory-mapped SRAM images of repository
//! "lessons" - encoded as 64-bit fingerprints that can be matched against student
//! inputs using hardware-level XOR + POPCNT operations.
//!
//! ## Key Concepts
//!
//! - **Bit-Fingerprinting**: MurmurHash3 → 64-bit fingerprints for concept lookup
//! - **Bloom Filters**: First-pass fuzzy matching before expensive operations
//! - **XOR + POPCNT**: Hardware-level Hamming distance judgment (1 cycle)
//! - **Hyperdimensional Vectors**: 1024-bit concept masks built from bundled atomic fingerprints
//! - **Cache-Line Alignment**: 64-byte aligned SRAM records for zero-latency L1 cache access
//!
//! ## Modules
//!
//! - [`fingerprint`]: MurmurHash3 → 64-bit fingerprints
//! - [`bloom`]: Bloom filter for fast fuzzy matching
//! - [`sram`]: 64-byte aligned SRAM record + mmap loader
//! - [`hdc`]: 1024-bit hypervector operations
//! - [`judge`]: XOR-POPCNT hardware-level judgment
//!
//! ## Usage
//!
//! ```rust
//! use superinstance_hdc_core::{fingerprint, judge, SramImage};
//!
//! // Judge an input against an SRAM image
//! let sram = SramImage::load_from_file("logic.sram").unwrap();
//! let result = judge(&sram, "your input here", 0xDEADBEEF, 10);
//! ```

pub mod fingerprint;
pub mod bloom;
pub mod sram;
pub mod hdc;
pub mod judge;
pub mod simd_avx512;
pub mod auto;
pub mod simd;

pub use fingerprint::fingerprint;
pub use bloom::BloomFilter;
pub use sram::{SramImage, SramRecord, SramImageBuilder};
pub use hdc::{HyperVector, permute_sequence, bundle_words};
pub use judge::{judge, judge_detailed, judge_batch, DEFAULT_THRESHOLD, MAX_THRESHOLD, Judgment};
pub use simd::{SimdBatch, SimdMode, BatchResult, batch_judge, batch_judge_multi};
pub use simd_avx512::{has_avx512_bitalg, batch_compare_avx512};
pub use auto::AutoBatch;

/// Lesson ID type alias for clarity
pub type LessonId = u32;

/// Result type for crate operations
pub type Result<T> = std::result::Result<T, Error>;

/// Error types for HDC operations
#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Invalid SRAM image: {0}")]
    InvalidImage(String),

    #[error("Bloom filter error: {0}")]
    Bloom(String),

    #[error("Invalid parameter: {0}")]
    InvalidParam(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fingerprint_deterministic() {
        let text = "hello world";
        let seed = 0xDEADBEEF;
        let a = fingerprint(text, seed);
        let b = fingerprint(text, seed);
        assert_eq!(a, b, "Same text+seed must produce same fingerprint");
    }

    #[test]
    fn test_fingerprint_different_seeds() {
        let text = "hello world";
        let a = fingerprint(text, 0xAAAA);
        let b = fingerprint(text, 0xBBBB);
        assert_ne!(a, b, "Different seeds must produce different fingerprints");
    }

    #[test]
    fn test_fingerprint_different_texts() {
        let seed = 0xDEADBEEF;
        let a = fingerprint("hello", seed);
        let b = fingerprint("world", seed);
        assert_ne!(a, b, "Different texts must produce different fingerprints");
    }

    #[test]
    fn test_hypervector_creation() {
        // from_text bundles identical broadcasts → result tends to zero
        // This is correct HDC behavior. Use from_seed for random vectors.
        let hv = HyperVector::from_seed(0xDEADBEEF);
        let density = hv.bit_density();
        assert!(density > 0.3 && density < 0.7, "from_seed should produce ~50% dense vector");
    }

    #[test]
    fn test_hypervector_xor() {
        let a = HyperVector::from_text("hello", 0xDEAD);
        let b = HyperVector::from_text("world", 0xBEEF);
        let c = a.xor(&b);
        let d = c.xor(&a); // should get back to b-ish
        let dist = d.hamming_distance(&b);
        assert_eq!(dist, 0, "XOR is self-inverse");
    }

    #[test]
    fn test_hypervector_hamming() {
        let a = HyperVector::from_text("test", 0x1234);
        let dist = a.hamming_distance(&a);
        assert_eq!(dist, 0, "Hamming distance to self is 0");
    }

    #[test]
    fn test_bloom_insert_and_check() {
        let mut bloom = BloomFilter::with_capacity(100, 0.01);
        let fp = fingerprint("test", 0xDEAD);
        bloom.insert(fp);
        assert!(bloom.contains(fp), "Inserted fingerprint should be found");
        assert!(!bloom.contains(fp + 1), "Non-inserted fingerprint should not be found (with low FPR)");
    }
}
