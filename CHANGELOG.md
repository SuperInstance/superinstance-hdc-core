# Changelog

All notable changes to this project will be documented in this file.

## [0.2.2] — 2026-05-04

### Added
- WASM target (`--features wasm`) for browser deployment
- `examples/wasm.rs`: C-exported functions for JS integration
- `examples/wasm-demo.html`: Browser demo page
- Optional dependencies: `memmap2`, `thiserror`, `structopt`, `termcolor` now gated behind `std` feature

## [0.2.1] — 2026-05-04

### Added
- `simd_avx512.rs`: Real AVX-512 intrinsics (VPXORQ + VPOPCNTDQ)
- `auto.rs`: `AutoBatch` struct with runtime feature detection
- 512-bit hypervector XOR and Hamming distance via ZMM registers
- AVX-512 tests verify bit-for-bit scalar equivalence

## [0.2.0] — 2026-05-04

### Added
- `simd.rs`: Batch comparison framework (AVX-512-ready, scalar fallback)
- `python/`: PyO3 bindings exposing all HDC primitives to Python
- `examples/flux_c_bridge.rs`: FLUX-C opcode mappings for certified execution
- `benches/`: Criterion.rs benchmark suite (fingerprint, judge)
- `.github/workflows/ci.yml`: GitHub Actions CI (test, bench, docs)
- `examples/plato_tile_matching.rs`: MUD tile duplicate detection + routing

### Changed
- `Cargo.toml`: Added `criterion` dev-dependency
- `README.md`: Complete rewrite with performance table, installation, usage
- `lib.rs`: Exported `simd`, `AutoBatch`, `has_avx512_bitalg`

## [0.1.0] — 2026-05-04

### Added
- Initial release by Oracle1
- `fingerprint.rs`: MurmurHash3 → 64-bit fingerprints
- `bloom.rs`: Bloom filter for O(1) fuzzy matching
- `sram.rs`: 64-byte aligned SRAM records + mmap loader
- `hdc.rs`: 1024-bit hypervector operations
- `judge.rs`: XOR-POPCNT hardware-level judgment
- CLI binaries: `bake`, `judge`, `monitor`
