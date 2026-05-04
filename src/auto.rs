//! # Auto-Detect SIMD Module
//!
//! Picks the fastest available implementation at runtime:
//! - AVX-512 BITALG (1 cycle per 8 lanes) on Zen 4 / Ryzen AI 9
//! - Scalar fallback (portable, works everywhere)
//!
//! Usage:
//! ```rust
//! use superinstance_hdc_core::auto::AutoBatch;
//!
//! let batch = AutoBatch::new(records);
//! let matches = batch.compare(query, threshold); // AVX-512 if available
//! ```

use crate::simd::{SimdBatch, BatchResult};
use crate::simd_avx512;

/// Runtime-dispatched batch comparison.
///
/// On first use, detects CPU features. Subsequent calls use the fastest path.
pub struct AutoBatch {
    records: Vec<u64>,
    has_avx512: bool,
}

impl AutoBatch {
    /// Create a new auto-detected batch.
    pub fn new(records: Vec<u64>) -> Self {
        Self {
            records,
            has_avx512: simd_avx512::has_avx512_bitalg(),
        }
    }

    /// Compare query against all records, using AVX-512 if available.
    pub fn compare(&self, query: u64, threshold: u32) -> Vec<BatchResult> {
        if self.has_avx512 {
            simd_avx512::batch_compare_avx512(&self.records, query, threshold)
        } else {
            let batch = SimdBatch::new(self.records.clone());
            batch.compare_against(query, threshold)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fingerprint::fingerprint;

    #[test]
    fn test_auto_batch_basic() {
        let records: Vec<u64> = (0..100)
            .map(|i| fingerprint(&format!("record_{}", i), 0xDEAD))
            .collect();
        let query = fingerprint("record_50", 0xDEAD);
        
        let batch = AutoBatch::new(records);
        let matches = batch.compare(query, 0);
        
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].index, 50);
        assert_eq!(matches[0].distance, 0);
    }
}
