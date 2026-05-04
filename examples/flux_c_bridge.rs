//! # FLUX-C + HDC Integration Examples
//!
//! Shows how HDC primitives map to FLUX-C instructions for
//! certified constraint checking on bare metal.
//!
//! ## Mapping
//!
//! | HDC Operation | FLUX-C Opcode | Cycles |
//! |---------------|---------------|--------|
//! | MurmurHash3 | XOR + ROT + MUL (emulated) | ~20 |
//! | XOR + POPCNT | XOR, POPCNT | 1 |
//! | Bloom check | AND, SHIFT, TEST | 3-5 |
//! | Batch compare | AVX-512 VPXOR, VPOPCNT | 1/16 |
//!
//! ## Architecture
//!
//! FLUX-C runs on the CPU (AVX-512), pre-screening constraints.
//! Only complex constraints graduate to FLUX-X (GPU).
//! The bridge between them is the HDC similarity score.

use superinstance_hdc_core as hdc;

/// Example: Check a fleet constraint using HDC judgment.
///
/// In FLUX-C, this compiles to:
/// ```flux
/// ; Load constraint SRAM image to zone 0x1000
/// LOAD.SRAM %r0, #0x1000, "constraint.sram"
///
/// ; Fingerprint the agent's claim
/// HASH %r1, "agent_claim", #SEED
///
/// ; Judge: XOR + POPCNT against all records
/// JUDGE %r2, %r0, %r1, #THRESHOLD
///
/// ; Branch on match
/// JNZ %r2, .violation_detected
/// ```
pub fn flux_c_constraint_check(
    sram: &hdc::SramImage,
    agent_claim: &str,
    seed: u64,
    threshold: u32,
) -> Option<u32> {
    hdc::judge::judge(sram, agent_claim, seed, threshold)
}

/// Example: Batch screen 100 agents against a policy.
///
/// In FLUX-C with AVX-512:
/// ```flux
/// ; Load 16 records into ZMM registers
/// LOAD.ZMM %zmm0, [%r0 + #0]
/// ...
/// LOAD.ZMM %zmm15, [%r0 + #960]
///
/// ; Broadcast query fingerprint
/// VBROADCASTQ %zmm16, %r1
///
/// ; XOR all 16 lanes simultaneously
/// VPXORQ %zmm0, %zmm0, %zmm16
/// ...
/// VPXORQ %zmm15, %zmm15, %zmm16
///
/// ; POPCNT all lanes
/// VPOPCNTQ %zmm0, %zmm0
/// ...
/// VPOPCNTQ %zmm15, %zmm15
///
/// ; Compare against threshold
/// VPCMPULT %k0, %zmm0, #THRESHOLD
/// ; ... etc
/// ```
pub fn flux_c_batch_screen(
    sram: &hdc::SramImage,
    agent_claims: &[&str],
    seed: u64,
    threshold: u32,
) -> Vec<Option<u32>> {
    agent_claims
        .iter()
        .map(|claim| hdc::judge::judge(sram, claim, seed, threshold))
        .collect()
}

/// Example: Build an HDC concept mask from a FLUX capability descriptor.
///
/// A capability like "can_write_to_zone_0x2000" becomes a 1024-bit
/// hypervector. Multiple capabilities bundle together.
pub fn capability_to_hypervector(capabilities: &[&str], seed: u64) -> hdc::HyperVector {
    hdc::hdc::bundle_words(capabilities, seed)
}

/// Example: Verify capability subset.
///
/// Does agent A's capability set subsume the required set?
/// HDC similarity > 0.9 means "yes".
pub fn check_capability_subset(
    required: &hdc::HyperVector,
    agent_caps: &hdc::HyperVector,
    threshold: f64,
) -> bool {
    let sim = required.similarity(agent_caps);
    sim >= threshold
}

#[cfg(test)]
mod tests {
    use super::*;
    use hdc::{SramImageBuilder, HyperVector};

    #[test]
    fn test_flux_c_constraint_check() {
        let mut builder = SramImageBuilder::new();
        let fp = hdc::fingerprint("unsafe_alloc", 0xDEAD);
        builder = builder.canary(fp).add_record(fp, 42);
        let sram = builder.build().unwrap();

        let result = flux_c_constraint_check(&sram, "unsafe_alloc", 0xDEAD, 0
);
        assert_eq!(result, Some(42));
    }

    #[test]
    fn test_flux_c_batch_screen() {
        let mut builder = SramImageBuilder::new();
        for i in 0..10 {
            let fp = hdc::fingerprint(&format!("violation_{}", i), 0xDEAD);
            builder = builder.canary(fp).add_record(fp, i as u32);
        }
        let sram = builder.build().unwrap();

        let claims = ["violation_3", "violation_7", "safe_behavior"];
        let results = flux_c_batch_screen(&sram, &claims, 0xDEAD, 0);

        assert_eq!(results[0], Some(3));
        assert_eq!(results[1], Some(7));
        assert_eq!(results[2], None);
    }

    #[test]
    fn test_capability_mask() {
        let required = capability_to_hypervector(
            &["read", "write", "execute"], 0xDEAD);
        let agent = capability_to_hypervector(
            &["read", "write", "execute", "admin"], 0xDEAD);
        let limited = capability_to_hypervector(
            &["read"], 0xDEAD);

        assert!(check_capability_subset(&required, &agent, 0.9));
        assert!(!check_capability_subset(&required, &limited, 0.9));
    }
}
