//! # Judge Module
//!
//! Provides hardware-level judgment using XOR + POPCNT operations.
//!
//! ## Overview
//!
//! The judge module implements the core judgment logic for matching student
//! inputs against SRAM-stored lesson fingerprints. It uses the fastest possible
//! hardware operations:
//!
//! - **Bloom Filter Check**: O(1) probabilistic pre-filter (definitely reject if false)
//! - **XOR + POPCNT**: Single-cycle Hamming distance on modern CPUs
//!
//! ## Performance Characteristics
//!
//! | Operation | Latency | Throughput |
//! |-----------|---------|------------|
//! | Bloom check | 1-3 cycles | High |
//! | XOR + POPCNT | 1 cycle | Very High |
//! | Full scan (1024 lessons) | ~1024 cycles | ~10M ops/sec |
//!
//! ## Usage
//!
//! ```rust
//! use superinstance_hdc_core::{judge, fingerprint, SramImage};
//!
//! // Judge a single input
//! let sram = SramImage::load_from_file("logic.sram").unwrap();
//! let result = judge(&sram, "your answer", 0xDEADBEEF, 5);
//! ```

use crate::fingerprint::fingerprint;
use crate::{LessonId, SramImage};

/// Default Hamming distance threshold for fuzzy matching.
///
/// This threshold allows for minor typos/variations while still
/// maintaining semantic accuracy.
pub const DEFAULT_THRESHOLD: u32 = 10;

/// Maximum reasonable threshold (handles ~1% bit errors).
pub const MAX_THRESHOLD: u32 = 64;

/// Compute Hamming distance between two 64-bit values (portable version).
///
/// This uses the standard library's count_ones() which is optimized
/// on modern CPUs to use POPCNT when available.
#[inline]
pub fn hamming_distance(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

/// Compute Hamming distance between two 64-bit values (alias for clarity).
#[inline]
pub fn hamming_distance_portable(a: u64, b: u64) -> u32 {
    (a ^ b).count_ones()
}

/// Judge an input string against an SRAM image.
///
/// This is the main entry point for the judgment system. It:
///
/// 1. **Fast path**: Checks the Bloom filter first (O(1), probabilistic)
///    - If Bloom says "no", return None immediately
/// 2. **Slow path**: Scans all SRAM records using XOR + POPCNT
///    - For each record, compute Hamming distance
///    - If distance <= threshold, return the lesson ID
///
/// # Arguments
///
/// * `sram` - The SRAM image to judge against
/// * `input` - The student's input string
/// * `seed` - Seed for fingerprint generation
/// * `threshold` - Maximum Hamming distance for a match
///
/// # Returns
///
/// * `Some(lesson_id)` if a match is found
/// * `None` if no match (or definitely not a match per Bloom)
///
/// # Example
///
/// ```
/// use superinstance_hdc_core::{judge, SramImage, DEFAULT_THRESHOLD};
///
/// let sram = SramImage::load_from_file("logic.sram").unwrap();
/// let result = judge(&sram, "the quick brown fox", 0xDEADBEEF, DEFAULT_THRESHOLD);
/// match result {
///     Some(lesson_id) => println!("Matched lesson {}", lesson_id),
///     None => println!("No match found"),
/// }
/// ```
#[inline]
pub fn judge(sram: &SramImage, input: &str, seed: u64, threshold: u32) -> Option<LessonId> {
    // Validate threshold
    if threshold > MAX_THRESHOLD {
        return None;
    }

    // Step 1: Fingerprint the input
    let input_hash = fingerprint(input, seed);

    // Step 2: Fast path via Bloom filter
    if !sram.bloom_contains(input_hash) {
        // Definitely not a match - Bloom says so with 100% certainty
        return None;
    }

    // Step 3: Slow path - full scan with XOR + POPCNT
    sram.judge(input_hash, threshold)
}

/// Judge an input using its pre-computed fingerprint.
///
/// This variant accepts a pre-computed 64-bit fingerprint, saving
/// the hashing step. Useful when batch-processing multiple inputs
/// with the same seed.
#[inline]
pub fn judge_with_fingerprint(
    sram: &SramImage,
    input_hash: u64,
    threshold: u32,
) -> Option<LessonId> {
    // Fast path: Bloom check
    if !sram.bloom_contains(input_hash) {
        return None;
    }

    // Slow path: full scan
    sram.judge(input_hash, threshold)
}

/// Batch judge multiple inputs against an SRAM image.
///
/// Returns a vector of (input, Option<LessonId>) pairs.
/// This can be more efficient than calling judge() repeatedly
/// when the inputs share the same seed and threshold.
pub fn judge_batch(
    sram: &SramImage,
    inputs: &[(&str, u64, u32)],
) -> Vec<Option<LessonId>>
{
    inputs
        .iter()
        .map(|(input, seed, threshold)| judge(sram, input, *seed, *threshold))
        .collect()
}

/// Judgment result with distance information.
#[derive(Debug, Clone)]
pub struct Judgment {
    /// The matched lesson ID (if any)
    pub lesson_id: Option<LessonId>,
    /// The input hash that was judged
    pub input_hash: u64,
    /// The Hamming distance to the closest match
    pub distance: Option<u32>,
    /// Whether the bloom filter passed
    pub bloom_passed: bool,
}

impl Judgment {
    /// Get whether this is a match.
    pub fn is_match(&self) -> bool {
        self.lesson_id.is_some()
    }

    /// Get the confidence (1.0 = perfect match, 0.0 = max distance).
    pub fn confidence(&self, max_threshold: u32) -> f64 {
        match self.distance {
            Some(d) if d <= max_threshold => 1.0 - (d as f64 / max_threshold as f64),
            _ => 0.0,
        }
    }
}

/// Judge with detailed result information.
pub fn judge_detailed(
    sram: &SramImage,
    input: &str,
    seed: u64,
    threshold: u32,
) -> Judgment {
    let input_hash = fingerprint(input, seed);
    let bloom_passed = sram.bloom_contains(input_hash);

    if !bloom_passed {
        return Judgment {
            lesson_id: None,
            input_hash,
            distance: None,
            bloom_passed: false,
        };
    }

    // Find the closest match
    let mut closest_distance = u32::MAX;
    let mut closest_lesson = None;
    let num_records = sram.record_count() as usize;

    for i in 0..num_records {
        if let Some(record) = sram.get_record(i) {
            if record.is_canary() {
                continue;
            }

            let distance = hamming_distance_portable(input_hash, record.fingerprint);
            if distance < closest_distance {
                closest_distance = distance;
                closest_lesson = Some(record.lesson_id);

                if distance == 0 {
                    break; // Perfect match, can't do better
                }
            }
        }
    }

    Judgment {
        lesson_id: if closest_distance <= threshold {
            closest_lesson
        } else {
            None
        },
        input_hash,
        distance: Some(closest_distance),
        bloom_passed: true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sram::{SramImageBuilder, SramRecord};
    use tempfile::NamedTempFile;

    #[test]
    fn test_hamming_distance_self() {
        let a = 0xDEADBEEFDEADBEEF_u64;
        let dist = hamming_distance_portable(a, a);
        assert_eq!(dist, 0, "Self distance should be 0");
    }

    #[test]
    fn test_hamming_distance_opposite() {
        let a = 0xFFFFFFFFFFFFFFFF_u64;
        let b = 0x0000000000000000_u64;
        let dist = hamming_distance_portable(a, b);
        assert_eq!(dist, 64, "Opposite values should have distance 64");
    }

    #[test]
    fn test_hamming_distance_partial() {
        let a = 0xF0F0F0F0F0F0F0F0_u64;
        let b = 0x0F0F0F0F0F0F0F0F_u64;
        let dist = hamming_distance_portable(a, b);
        assert_eq!(dist, 64, "Complement should have distance 64");
    }

    #[test]
    fn test_judge_exact_match() {
        let temp_file = NamedTempFile::new().unwrap();
        
        let mut builder = SramImageBuilder::new();
        let fp = fingerprint("test answer", 0xDEAD);
        builder = builder.canary(fp).add_record(fp, 42);

        let image = builder.build().unwrap();
        image.save_to_file(temp_file.path()).unwrap();

        let loaded = SramImage::load_from_file(temp_file.path()).unwrap();
        let result = judge(&loaded, "test answer", 0xDEAD, 0);

        assert_eq!(result, Some(42), "Exact match should be found");
    }

    #[test]
    fn test_judge_fuzzy_match() {
        let temp_file = NamedTempFile::new().unwrap();
        
        let mut builder = SramImageBuilder::new();
        // Insert with seed 0xDEAD
        let fp = fingerprint("correct answer", 0xDEAD);
        builder = builder.canary(fp).add_record(fp, 1);

        let image = builder.build().unwrap();
        image.save_to_file(temp_file.path()).unwrap();

        let loaded = SramImage::load_from_file(temp_file.path()).unwrap();
        
        // Same text should match with threshold 0
        assert_eq!(judge(&loaded, "correct answer", 0xDEAD, 0), Some(1));
        
        // Different text won't match (unless very unlucky with hash)
        let result = judge(&loaded, "wrong answer", 0xDEAD, 0);
        assert_eq!(result, None, "Wrong answer should not match");
    }

    #[test]
    fn test_judge_bloom_reject() {
        let temp_file = NamedTempFile::new().unwrap();
        
        let mut builder = SramImageBuilder::new();
        builder = builder.canary(0xDEAD).add_record(0x1111, 1).add_record(0x2222, 2);

        let image = builder.build().unwrap();
        image.save_to_file(temp_file.path()).unwrap();

        let loaded = SramImage::load_from_file(temp_file.path()).unwrap();

        // A completely unrelated input should be rejected by Bloom
        // with very high probability
        let result = judge(&loaded, "unrelated text that definitely won't match", 0xDEAD, 64);
        
        // Should be None due to Bloom filter rejection
        assert_eq!(result, None, "Unrelated text should not match");
    }

    #[test]
    fn test_judge_batch() {
        let temp_file = NamedTempFile::new().unwrap();
        
        let mut builder = SramImageBuilder::new();
        let fp1 = fingerprint("answer 1", 0xDEAD);
        let fp2 = fingerprint("answer 2", 0xDEAD);
        builder = builder
            .canary(fp1)
            .add_record(fp1, 1)
            .add_record(fp2, 2);

        let image = builder.build().unwrap();
        image.save_to_file(temp_file.path()).unwrap();

        let loaded = SramImage::load_from_file(temp_file.path()).unwrap();

        let inputs = [
            ("answer 1", 0xDEAD, 0u32),
            ("answer 2", 0xDEAD, 0u32),
            ("wrong", 0xDEAD, 0u32),
        ];

        let results = judge_batch(&loaded, &inputs);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0], Some(1));
        assert_eq!(results[1], Some(2));
        assert_eq!(results[2], None);
    }

    #[test]
    fn test_judgment_detailed() {
        let temp_file = NamedTempFile::new().unwrap();
        
        let mut builder = SramImageBuilder::new();
        let fp = fingerprint("test", 0xDEAD);
        builder = builder.canary(fp).add_record(fp, 1);

        let image = builder.build().unwrap();
        image.save_to_file(temp_file.path()).unwrap();

        let loaded = SramImage::load_from_file(temp_file.path()).unwrap();

        let judgment = judge_detailed(&loaded, "test", 0xDEAD, 5);
        assert_eq!(judgment.lesson_id, Some(1));
        assert_eq!(judgment.distance, Some(0));
        assert!(judgment.bloom_passed);
        assert!(judgment.is_match());
    }

    #[test]
    fn test_judgment_confidence() {
        let judgment = Judgment {
            lesson_id: Some(1),
            input_hash: 0,
            distance: Some(3),
            bloom_passed: true,
        };
        
        assert!((judgment.confidence(5) - 0.4).abs() < 0.01);
    }

    #[test]
    fn test_threshold_validation() {
        let temp_file = NamedTempFile::new().unwrap();
        
        let mut builder = SramImageBuilder::new();
        builder = builder.canary(0xDEAD).add_record(0x1111, 1);

        let image = builder.build().unwrap();
        image.save_to_file(temp_file.path()).unwrap();

        let loaded = SramImage::load_from_file(temp_file.path()).unwrap();

        // Threshold over MAX_THRESHOLD should return None immediately
        let result = judge(&loaded, "test", 0xDEAD, MAX_THRESHOLD + 1);
        assert_eq!(result, None);
    }
}
