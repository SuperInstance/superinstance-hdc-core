use pyo3::prelude::*;
use superinstance_hdc_core as hdc;

/// Python module for SuperInstance HDC Core.
///
/// Provides high-performance hyperdimensional computing primitives:
/// - MurmurHash3 fingerprinting (10x faster than SHA)
/// - XOR+POPCNT Hamming distance (1 cycle on modern CPU)
/// - Bloom filter O(1) fuzzy matching
/// - 1024-bit hypervector operations
/// - Batch SIMD comparison (AVX-512 ready)
#[pymodule]
fn superinstance_hdc_py(m: &Bound<'_ , PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fingerprint, m)?)?;
    m.add_function(wrap_pyfunction!(hamming_distance, m)?)?;
    m.add_function(wrap_pyfunction!(batch_judge_py, m)?)?;
    m.add_function(wrap_pyfunction!(hypervector_from_text, m)?)?;
    m.add_function(wrap_pyfunction!(hypervector_similarity, m)?)?;
    m.add_class::<<PySramImage>>()?;
    m.add_class::<<PyHyperVector>>()?;
    m.add_class::<<PyBloomFilter>>()?;
    Ok(())
}

/// Generate a 64-bit fingerprint from text using MurmurHash3.
///
/// Args:
///     text: Input string to fingerprint
///     seed: Seed value for deterministic hashing
///
/// Returns:
///     64-bit integer fingerprint
#[pyfunction]
fn fingerprint(text: &str, seed: u64) -> u64 {
    hdc::fingerprint::fingerprint(text, seed)
}

/// Compute Hamming distance between two 64-bit fingerprints.
///
/// Uses hardware POPCNT — 1 cycle on modern x86_64.
#[pyfunction]
fn hamming_distance(a: u64, b: u64) -> u32 {
    hdc::judge::hamming_distance(a, b)
}

/// Batch judge multiple query fingerprints against a set of records.
///
/// Args:
///     queries: List of query fingerprints
///     records: List of record fingerprints
///     threshold: Maximum Hamming distance for match
///
/// Returns:
///     List of (query_index, record_index, distance) tuples for matches
#[pyfunction]
fn batch_judge_py(queries: Vec<u64>, records: Vec<u64>, threshold: u32) -> Vec<(u32, u32, u32)> {
    let batch = hdc::simd::SimdBatch::new(records);
    let mut results = Vec::new();
    
    for (q_idx, query) in queries.iter().enumerate() {
        let matches = batch.compare_against(*query, threshold);
        for m in matches {
            results.push((q_idx as u32, m.index as u32, m.distance));
        }
    }
    
    results
}

/// Create a 1024-bit hypervector from text.
#[pyfunction]
fn hypervector_from_text(text: &str, seed: u64) -> PyHyperVector {
    let hv = hdc::HyperVector::from_text(text, seed);
    PyHyperVector { inner: hv }
}

/// Compute similarity between two hypervectors (0.0 to 1.0).
#[pyfunction]
fn hypervector_similarity(a: &PyHyperVector, b: &PyHyperVector) -> f64 {
    a.inner.similarity(&b.inner)
}

/// Python wrapper for SRAM image.
#[pyclass]
struct PySramImage {
    inner: hdc::SramImage,
}

#[pymethods]
impl PySramImage {
    /// Load an SRAM image from a file path.
    #[staticmethod]
    fn load(path: &str) -> PyResult<PySramImage> {
        let inner = hdc::SramImage::load_from_file(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(PySramImage { inner })
    }
    
    /// Judge an input string against this SRAM image.
    fn judge(&self, input: &str, seed: u64, threshold: u32) -> Option<u32> {
        hdc::judge::judge(&self.inner, input, seed, threshold)
    }
    
    /// Get the number of records in the image.
    #[getter]
    fn record_count(&self) -> usize {
        self.inner.record_count() as usize
    }
}

/// Python wrapper for 1024-bit HyperVector.
#[pyclass]
struct PyHyperVector {
    inner: hdc::HyperVector,
}

#[pymethods]
impl PyHyperVector {
    /// Create from text.
    #[staticmethod]
    fn from_text(text: &str, seed: u64) -> PyHyperVector {
        PyHyperVector {
            inner: hdc::HyperVector::from_text(text, seed),
        }
    }
    
    /// XOR with another hypervector.
    fn xor(&self, other: &PyHyperVector) -> PyHyperVector {
        PyHyperVector {
            inner: self.inner.xor(&other.inner),
        }
    }
    
    /// Hamming distance to another hypervector.
    fn hamming_distance(&self, other: &PyHyperVector) -> u32 {
        self.inner.hamming_distance(&other.inner)
    }
    
    /// Normalized similarity (0.0 = opposite, 1.0 = identical).
    fn similarity(&self, other: &PyHyperVector) -> f64 {
        self.inner.similarity(&other.inner)
    }
    
    /// Bit density (ratio of 1-bits, ~0.5 for random).
    #[getter]
    fn bit_density(&self) -> f64 {
        self.inner.bit_density()
    }
    
    /// Raw bytes (128 bytes).
    fn to_bytes(&self) -> [u8; 128] {
        self.inner.to_raw()
    }
}

/// Python wrapper for Bloom Filter.
#[pyclass]
struct PyBloomFilter {
    inner: hdc::BloomFilter,
}

#[pymethods]
impl PyBloomFilter {
    /// Create a new Bloom filter with given capacity and false-positive rate.
    #[new]
    fn new(capacity: usize, fpr: f64) -> PyBloomFilter {
        PyBloomFilter {
            inner: hdc::BloomFilter::with_capacity(capacity, fpr),
        }
    }
    
    /// Insert a fingerprint.
    fn insert(&mut self, fp: u64) {
        self.inner.insert(fp);
    }
    
    /// Check if a fingerprint might be present.
    fn contains(&self, fp: u64) -> bool {
        self.inner.contains(fp)
    }
}
