# SuperInstance HDC Core

Hyperdimensional Computing Core for bit-level agent cognition.

## Overview

This crate provides tools for building memory-mapped SRAM images of repository "lessons" - encoded as 64-bit fingerprints that can be matched against student inputs using hardware-level XOR + POPCNT operations.

## Key Concepts

- **Bit-Fingerprinting**: MurmurHash3 → 64-bit fingerprints for concept lookup
- **Bloom Filters**: First-pass fuzzy matching before expensive operations
- **XOR + POPCNT**: Hardware-level Hamming distance judgment (1 cycle on modern CPU)
- **Hyperdimensional Vectors**: 1024-bit concept masks built from bundled atomic fingerprints
- **Cache-Line Alignment**: 64-byte aligned SRAM records for zero-latency L1 cache access

## Architecture

```
superinstance-hdc-core/
├── src/
│   ├── lib.rs          # Main entry point
│   ├── fingerprint.rs  # MurmurHash3 → 64-bit fingerprints
│   ├── bloom.rs        # Bloom filter for fast fuzzy matching
│   ├── sram.rs         # 64-byte aligned SRAM record + mmap loader
│   ├── hdc.rs          # 1024-bit hypervector operations
│   └── judge.rs        # XOR-POPCNT hardware-level judgment
├── src/bin/
│   ├── bake.rs         # CLI: bake repo lessons to SRAM image
│   ├── judge.rs        # CLI: judge student input against SRAM
│   └── monitor.rs      # CLI: resonance HUD
├── examples/
│   └── quickstart.rs   # Basic usage example
├── Cargo.toml
└── README.md
```

## Installation

```toml
[dependencies]
superinstance-hdc-core = "0.1.0"
```

## Usage

### Library Usage

```rust
use superinstance_hdc_core::{
    fingerprint, judge, judge_detailed, BloomFilter, HyperVector, 
    SramImage, SramImageBuilder, permute_sequence, bundle_words,
};

// Create a fingerprint
let fp = fingerprint("hello world", 0xDEADBEEF);

// Create a hypervector from text
let hv = HyperVector::from_text("concept", 0x1000);

// Build an SRAM image
let sram = SramImageBuilder::new()
    .canary(fp)
    .add_record(fingerprint("answer:42", 0xDEAD), 1)
    .add_record(fingerprint("answer:3.14", 0xDEAD), 2)
    .build()?;

// Judge an input
let result = judge(&sram, "answer:42", 0xDEAD, 5);
match result {
    Some(lesson_id) => println!("Matched lesson {}", lesson_id),
    None => println!("No match"),
}
```

### CLI Usage

#### Bake Lessons to SRAM

```bash
# Create a lessons directory with .txt or .md files
mkdir -p lessons
echo "The answer is 42" > lessons/lesson_001.txt
echo "Pi is approximately 3.14" > lessons/lesson_002.txt

# Bake lessons to SRAM image
cargo run --bin bake -- --dir lessons --output logic.sram --seed 0xDEADBEEF
```

#### Judge Student Input

```bash
# Judge from stdin
echo "The answer is 42" | cargo run --bin judge -- --sram logic.sram --seed 0xDEADBEEF

# Judge from argument
cargo run --bin judge -- "The answer is 42" --sram logic.sram --threshold 5

# Detailed output
cargo run --bin judge -- "The answer is 42" --sram logic.sram --detailed
```

#### Monitor Resonance HUD

```bash
# Monitor state file every 100ms
cargo run --bin monitor -- --state /dev/shm/superinstance/state.bin

# Custom interval and bar width
cargo run --bin monitor -- --interval 50 --width 60
```

### Quickstart Example

```bash
cargo run --example quickstart
```

## SRAM Image Format

The SRAM image format (`logic.sram`):

```
+----------------+----------------+----------------+----------------+
|     Header     |  Bloom Filter  |    Record 0    |    Record 1    | ...
|   (64 bytes)   |  (variable)    |   (64 bytes)   |   (64 bytes)   |
+----------------+----------------+----------------+----------------+
```

- **Header**: Magic, version, record count, bloom size, canary
- **Bloom Filter**: Serialized bloom filter for O(1) pre-checking
- **Records**: 64-byte aligned SRAM records (cache-line aligned)

Each record:
- `fingerprint`: u64 - 64-bit lesson fingerprint
- `lesson_id`: u32 - Lesson identifier
- `flags`: u16 - Flags (bit 0 = canary)
- `_reserved`: u16 - Reserved
- `padding`: [u8; 48] - Padding to 64 bytes

## Hypervector Operations

### XOR Binding

```rust
let a = HyperVector::from_text("concept A", seed);
let b = HyperVector::from_text("concept B", seed);
let bound = a.xor(&b); // "A bound to B"

// Self-inverse: (A XOR B) XOR B = A
assert_eq!(bound.xor(&b), a);
```

### Permutation (Sequence)

```rust
// "A then B" differs from "B then A"
let seq1 = permute_sequence(&["A", "B"], seed);
let seq2 = permute_sequence(&["B", "A"], seed);
assert_ne!(seq1, seq2);
```

### Bundling (Majority Rule)

```rust
// Combine multiple vectors into one
let a = HyperVector::from_fingerprint(fp1);
let b = HyperVector::from_fingerprint(fp2);
let c = HyperVector::from_fingerprint(fp3);
let bundled = a.bundle(&[b, c]); // Majority rule
```

## Performance

| Operation | Latency | Throughput |
|-----------|---------|-------------|
| Bloom check | 1-3 cycles | ~10B ops/sec |
| XOR + POPCNT | 1 cycle | ~10B ops/sec |
| Full scan (1024 lessons) | ~1024 cycles | ~10M ops/sec |

## Dependencies

- `mmh3-rust`: MurmurHash3 implementation
- `memmap2`: Zero-copy mmap for SRAM loading
- `thiserror`: Error handling
- `serde`: Serialization support
- `structopt`: CLI argument parsing (binaries only)
- `termcolor`: Colored terminal output (monitor only)

## License

MIT
