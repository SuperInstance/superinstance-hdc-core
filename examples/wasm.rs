//! # WASM Example — HDC in the Browser
//!
//! Compile with:
//! ```bash
//! cargo build --target wasm32-unknown-unknown --release --features wasm --no-default-features
//! ```
//!
//! Then use wasm-bindgen or direct JS integration.

use superinstance_hdc_core as hdc;

/// WASM-exported function: fingerprint text.
///
/// JavaScript usage:
/// ```js
/// const fp = wasm.fingerprint("hello world", 0xDEADBEEF);
/// console.log(fp.toString(16)); // 64-bit hex
/// ```
#[no_mangle]
pub extern "C" fn wasm_fingerprint(text_ptr: *const u8, text_len: usize, seed: u64) -> u64 {
    let text = unsafe {
        std::str::from_utf8(std::slice::from_raw_parts(text_ptr, text_len))
            .unwrap_or("")
    };
    hdc::fingerprint::fingerprint(text, seed)
}

/// WASM-exported function: hamming distance between two fingerprints.
#[no_mangle]
pub extern "C" fn wasm_hamming_distance(a: u64, b: u64) -> u32 {
    hdc::judge::hamming_distance(a, b)
}

/// WASM-exported function: create hypervector from text.
///
/// Returns pointer to 128-byte buffer. JS must free with `wasm_free(ptr)`.
#[no_mangle]
pub extern "C" fn wasm_hypervector_from_text(
    text_ptr: *const u8,
    text_len: usize,
    seed: u64,
) -> *mut u8 {
    let text = unsafe {
        std::str::from_utf8(std::slice::from_raw_parts(text_ptr, text_len))
            .unwrap_or("")
    };
    let hv = hdc::HyperVector::from_text(text, seed);
    let bytes = hv.to_raw();
    
    let ptr = unsafe { std::alloc::alloc(std::alloc::Layout::from_size_align(128, 8).unwrap()) };
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, 128);
    }
    ptr
}

/// Free a buffer allocated by WASM.
#[no_mangle]
pub extern "C" fn wasm_free(ptr: *mut u8) {
    unsafe {
        std::alloc::dealloc(ptr, std::alloc::Layout::from_size_align(128, 8).unwrap());
    }
}

/// WASM-exported function: hypervector similarity.
#[no_mangle]
pub extern "C" fn wasm_hypervector_similarity(a_ptr: *const u8, b_ptr: *const u8) -> f64 {
    let a_bytes = unsafe { std::slice::from_raw_parts(a_ptr, 128) };
    let b_bytes = unsafe { std::slice::from_raw_parts(b_ptr, 128) };
    
    let a = hdc::HyperVector::from_raw(&a_bytes.try_into().unwrap_or([0u8; 128])
    );
    let b = hdc::HyperVector::from_raw(
        &b_bytes.try_into().unwrap_or([0u8; 128])
    );
    
    a.similarity(&b)
}
