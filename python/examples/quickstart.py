#!/usr/bin/env python3
"""
HDC Quickstart — Python bindings example.

Shows the core HDC pipeline:
1. Fingerprint text with MurmurHash3 (10x faster than SHA)
2. Build a Bloom filter for O(1) pre-filtering
3. Load SRAM image and judge with XOR+POPCNT (1 cycle)
4. Batch compare with SIMD-ready batch operations
"""

import superinstance_hdc_py as hdc

# 1. Fingerprint some concepts
seed = 0xDEADBEEF
fp_hello = hdc.fingerprint("hello world", seed)
fp_goodbye = hdc.fingerprint("goodbye world", seed)

print(f"Fingerprint 'hello world': 0x{fp_hello:016x}")
print(f"Fingerprint 'goodbye world': 0x{fp_goodbye:016x}")

# 2. Hamming distance (1 CPU cycle)
dist = hdc.hamming_distance(fp_hello, fp_goodbye)
print(f"Hamming distance: {dist} bits")

# 3. Hypervectors (1024-bit concept masks)
hv_hello = hdc.hypervector_from_text("hello world", seed)
hv_goodbye = hdc.hypervector_from_text("goodbye world", seed)

sim = hdc.hypervector_similarity(hv_hello, hv_goodbye)
print(f"Hypervector similarity: {sim:.3f}")

# 4. Bloom filter (O(1) fuzzy match)
bloom = hdc.PyBloomFilter(1000, 0.01)
bloom.insert(fp_hello)

print(f"Bloom contains 'hello': {bloom.contains(fp_hello)}")
print(f"Bloom contains 'goodbye': {bloom.contains(fp_goodbye)} (probably False)")

# 5. Batch judgment (AVX-512 ready)
records = [hdc.fingerprint(f"concept_{i}", seed) for i in range(100)]
queries = [hdc.fingerprint("concept_50", seed), hdc.fingerprint("unrelated", seed)]

matches = hdc.batch_judge_py(queries, records, threshold=5)
print(f"Batch matches: {matches}")
