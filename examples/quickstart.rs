//! # Quickstart Example
//!
//! Demonstrates the basic usage of the SuperInstance HDC Core library.
//!
//! ## Running
//!
//! ```bash
//! cargo run --example quickstart
//! ```
//!
//! ## Concepts Demonstrated
//!
//! - Creating fingerprints from text
//! - Building hypervectors from text
//! - XOR binding, permutation, and bundling
//! - Bloom filter usage
//! - Judging inputs against SRAM

use superinstance_hdc_core::{
    fingerprint, BloomFilter, HyperVector, SramImageBuilder,
    permute_sequence, bundle_words,
};

fn main() {
    println!("=== SuperInstance HDC Core Quickstart ===\n");

    // 1. Fingerprinting
    println!("1. FINGERPRINTING");
    println!("   Fingerprints are 64-bit hashes from MurmurHash3");
    let fp1 = fingerprint("hello world", 0xDEADBEEF);
    let fp2 = fingerprint("hello world", 0xDEADBEEF);
    let fp3 = fingerprint("hello world", 0xCAFEBABE);
    println!("   fingerprint(\"hello world\", 0xDEADBEEF) = {:016x}", fp1);
    println!("   Same text+seed → same fingerprint: {}", fp1 == fp2);
    println!("   Different seed → different fingerprint: {}", fp1 != fp3);
    println!();

    // 2. Hypervectors
    println!("2. HYPERVECTOR OPERATIONS");
    let hv1 = HyperVector::from_text("concept A", 0x1000);
    let hv2 = HyperVector::from_text("concept B", 0x1000);
    let _hv3 = HyperVector::from_text("concept A", 0x1000);

    println!("   Created hypervectors from text");
    println!("   Bit density of A: {:.2}", hv1.bit_density());
    println!("   Bit density of B: {:.2}", hv2.bit_density());

    let bundled = hv1.bundle(&[hv2]);
    println!("   Bundled (A + B) density: {:.2}", bundled.bit_density());

    let xor_bound = hv1.xor(&hv2);
    println!("   XOR bound (A ⊕ B) self-similarity: {:.2}", xor_bound.similarity(&xor_bound));
    println!();

    // 3. Sequences
    println!("3. SEQUENCE ENCODING");
    let seq1 = permute_sequence(&["hello", "world"], 0xDEAD);
    let seq2 = permute_sequence(&["world", "hello"], 0xDEAD);
    let seq3 = permute_sequence(&["hello", "world"], 0xDEAD);

    println!("   \"hello then world\": dist = {}", seq1.hamming_distance(&seq3));
    println!("   \"world then hello\": dist = {}", seq2.hamming_distance(&seq3));
    println!("   Order matters: {} (0 = same)", seq1.hamming_distance(&seq2));
    println!();

    // 4. Order-independent bags
    println!("4. BAG ENCODING (order-independent)");
    let bag1 = bundle_words(&["A", "B", "C"], 0xDEAD);
    let bag2 = bundle_words(&["C", "B", "A"], 0xDEAD);
    let bag3 = bundle_words(&["X", "Y", "Z"], 0xDEAD);

    println!("   \"A B C\" == \"C B A\": {}", bag1 == bag2);
    println!("   \"A B C\" != \"X Y Z\": {}", bag1 != bag3);
    println!();

    // 5. Bloom filter
    println!("5. BLOOM FILTER");
    let mut bloom = BloomFilter::with_capacity(1000, 0.01);
    bloom.insert(fp1);
    bloom.insert(fp2);
    bloom.insert(fp3);

    println!("   Capacity: 1000, FPR: 1%");
    println!("   Contains fp1: {}", bloom.contains(fp1));
    println!("   Contains random: {}", bloom.contains(0x123456789ABCDEF0));
    println!();

    // 6. SRAM Image
    println!("6. SRAM IMAGE BUILDING");
    let sram = SramImageBuilder::new()
        .canary(fp1)
        .add_record(fingerprint("answer:42", 0xDEAD), 1)
        .add_record(fingerprint("answer:3.14", 0xDEAD), 2)
        .add_record(fingerprint("answer:green", 0xDEAD), 3)
        .build()
        .expect("Failed to build SRAM image");

    println!("   Built SRAM image with {} records", sram.record_count());
    println!("   Canary: {:016x}", sram.canary());
    println!();

    // 7. Judgment
    println!("7. JUDGMENT");
    println!("   Input: \"answer:42\" threshold=0 → {:?}", 
        superinstance_hdc_core::judge(&sram, "answer:42", 0xDEAD, 0));
    println!("   Input: \"answer:wrong\" threshold=10 → {:?}",
        superinstance_hdc_core::judge(&sram, "answer:wrong", 0xDEAD, 10));
    println!();

    println!("=== Quickstart Complete ===");
}
