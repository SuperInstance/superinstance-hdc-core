//! # AVX-512 Intrinsics Implementation
//!
//! Real AVX-512 code for HDC batch comparison.
//! Uses VPXORQ + VPOPCNTDQ on x86_64 with AVX-512 BITALG.
//!
//! ## Target Features Required
//!
//! - `avx512f` — Foundation (512-bit operations)
//! - `avx512bitalg` — VPOPCNTDQ (population count on 512-bit vectors)
//!
//! ## Runtime Detection
//!
//! `is_x86_feature_detected!("avx512bitalg")` enables the fast path.
//! Falls back to scalar on unsupported CPUs.
//!
//! ## Performance (Expected)
//!
//! | Operation | Scalar | AVX-512 | Speedup |
//! |-----------|--------|---------|---------|
//! | 8× XOR+POPCNT | 8 cycles | 1 cycle | **8x** |
//! | 16× XOR+POPCNT | 16 cycles | 1 cycle | **16x** |
//! | Full batch (1024) | ~1024 cycles | ~64 cycles | **16x** |

#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
use std::arch::x86_64::*;

/// Check if AVX-512 BITALG (VPOPCNTDQ) is available at runtime.
#[inline]
pub fn has_avx512_bitalg() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bitalg")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// AVX-512 batch compare: 8 fingerprints per iteration.
///
/// Uses 512-bit vectors to compare 8 × 64-bit fingerprints simultaneously.
/// Each iteration does:
/// 1. Load 8 records into ZMM register
/// 2. Broadcast query to all lanes
/// 3. VPXORQ (8 XORs in parallel)
/// 4. VPOPCNTDQ (8 POPCNTs in parallel)
/// 5. Compare against threshold, store matches
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bitalg")]
unsafe fn compare_8_avx512(
    records: *const u64,
    query: __m512i,
    threshold: u32,
    start_index: usize,
) -> Vec<crate::simd::BatchResult> {
    let mut results = Vec::new();
    
    // Load 8 records
    let records_vec = _mm512_loadu_si512(records as *const i32);
    
    // XOR all lanes with query
    let xored = _mm512_xor_epi64(records_vec, query);
    
    // Population count per lane (VPOPCNTDQ)
    let counts = _mm512_popcnt_epi64(xored);
    
    // Compare counts <= threshold
    // Create threshold vector (all lanes = threshold)
    let thresh_vec = _mm512_set1_epi64(threshold as i64);
    let mask = _mm512_cmple_epu64_mask(counts, thresh_vec);
    
    // Extract matching lanes
    if mask != 0 {
        for lane in 0..8 {
            if (mask & (1 << lane)) != 0 {
                let distance = _mm512_extract_epi64(counts, lane) as u32;
                results.push(crate::simd::BatchResult {
                    index: start_index + lane,
                    distance,
                    is_match: true,
                });
            }
        }
    }
    
    results
}

/// AVX-512 batch compare entry point.
///
/// Runtime-dispatched: uses AVX-512 if available, scalar fallback otherwise.
/// Processes fingerprints in chunks of 8 for vector alignment.
pub fn batch_compare_avx512(
    records: &[u64],
    query: u64,
    threshold: u32,
) -> Vec<crate::simd::BatchResult> {
    #[cfg(target_arch = "x86_64")]
    {
        if has_avx512_bitalg() {
            return unsafe { batch_compare_avx512_inner(records, query, threshold) };
        }
    }
    
    // Fallback to scalar
    let batch = crate::simd::SimdBatch::new(records.to_vec());
    batch.compare_against(query, threshold)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bitalg")]
unsafe fn batch_compare_avx512_inner(
    records: &[u64],
    query: u64,
    threshold: u32,
) -> Vec<crate::simd::BatchResult> {
    let mut results = Vec::new();
    let len = records.len();
    
    // Broadcast query to all 8 lanes
    let query_vec = _mm512_set1_epi64(query as i64);
    
    // Process chunks of 8
    let chunks = len / 8;
    for i in 0..chunks {
        let offset = i * 8;
        let ptr = records.as_ptr().add(offset);
        let chunk_results = compare_8_avx512(ptr, query_vec, threshold, offset);
        results.extend(chunk_results);
    }
    
    // Handle remainder with scalar
    let remainder_start = chunks * 8;
    for i in remainder_start..len {
        let distance = (query ^ records[i]).count_ones();
        if distance <= threshold {
            results.push(crate::simd::BatchResult {
                index: i,
                distance,
                is_match: true,
            });
        }
    }
    
    results
}

/// 512-bit hypervector XOR (16 × 64-bit words).
///
/// Processes an entire 1024-bit hypervector in one instruction pair.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
pub unsafe fn hypervector_xor_avx512(
    a: &[u64; 16],
    b: &[u64; 16],
    out: &mut [u64; 16],
) {
    // Two ZMM loads cover all 16 words
    let a0 = _mm512_loadu_si512(a.as_ptr() as *const i32);
    let a1 = _mm512_loadu_si512(a.as_ptr().add(8) as *const i32);
    let b0 = _mm512_loadu_si512(b.as_ptr() as *const i32);
    let b1 = _mm512_loadu_si512(b.as_ptr().add(8) as *const i32);
    
    let r0 = _mm512_xor_epi64(a0, b0);
    let r1 = _mm512_xor_epi64(a1, b1);
    
    _mm512_storeu_si512(out.as_mut_ptr() as *mut i32, r0);
    _mm512_storeu_si512(out.as_mut_ptr().add(8) as *mut i32, r1);
}

/// 512-bit hypervector Hamming distance (1024 bits).
///
/// Uses two ZMM POPCNTs + horizontal add.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bitalg")]
pub unsafe fn hypervector_hamming_avx512(
    a: &[u64; 16],
    b: &[u64; 16],
) -> u32 {
    let a0 = _mm512_loadu_si512(a.as_ptr() as *const i32);
    let a1 = _mm512_loadu_si512(a.as_ptr().add(8) as *const i32);
    let b0 = _mm512_loadu_si512(b.as_ptr() as *const i32);
    let b1 = _mm512_loadu_si512(b.as_ptr().add(8) as *const i32);
    
    let x0 = _mm512_xor_epi64(a0, b0);
    let x1 = _mm512_xor_epi64(a1, b1);
    
    let c0 = _mm512_popcnt_epi64(x0);
    let c1 = _mm512_popcnt_epi64(x1);
    
    // Horizontal sum: reduce 8 lanes to scalar
    let sum0 = _mm512_reduce_add_epi64(c0) as u32;
    let sum1 = _mm512_reduce_add_epi64(c1) as u32;
    
    sum0 + sum1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_has_avx512_detection() {
        // Just verify it doesn't panic
        let _ = has_avx512_bitalg();
    }

    #[test]
    fn test_hypervector_xor_matches_scalar() {
        let a: [u64; 16] = [0xDEADBEEF; 16];
        let b: [u64; 16] = [0x12345678; 16];
        let mut out = [0u64; 16];
        
        #[cfg(target_arch = "x86_64")]
        unsafe {
            if has_avx512_bitalg() {
                hypervector_xor_avx512(&a, &b, &mut out);
                
                // Verify against scalar
                for i in 0..16 {
                    assert_eq!(out[i], a[i] ^ b[i], "Lane {} mismatch", i);
                }
            }
        }
    }

    #[test]
    fn test_batch_compare_matches_scalar() {
        let records: Vec<u64> = (0..100).map(|i| i * 0x1111).collect();
        let query = 0x2222;
        let threshold = 10;
        
        let avx_results = batch_compare_avx512(&records, query, threshold);
        let scalar = crate::simd::SimdBatch::new(records);
        let scalar_results = scalar.compare_against(query, threshold);
        
        assert_eq!(avx_results.len(), scalar_results.len());
        for (a, s) in avx_results.iter().zip(scalar_results.iter()) {
            assert_eq!(a.index, s.index);
            assert_eq!(a.distance, s.distance);
        }
    }
}
