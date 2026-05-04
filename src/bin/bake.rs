//! # bake CLI
//!
//! Bakes repository lessons into a memory-mapped SRAM image.
//!
//! This tool:
//! 1. Reads all `.txt` and `.md` lesson files from a directory
//! 2. Generates MurmurHash3 fingerprints for each lesson
//! 3. Creates a Bloom filter for fast pre-checking
//! 4. Writes a 64-byte aligned binary SRAM image
//! 5. Generates a verification canary

use std::fs;
use std::path::PathBuf;
use std::process;
use structopt::StructOpt;

use superinstance_hdc_core::{
    fingerprint,
    sram::SramImageBuilder,
    Error, Result,
};

/// CLI for baking lessons into SRAM image
#[derive(Debug, StructOpt)]
#[structopt(
    name = "bake",
    about = "Bake repository lessons to SRAM image",
    author = "SuperInstance"
)]
struct Opt {
    /// Lesson files directory
    #[structopt(short = "d", long = "dir", default_value = "./lessons")]
    lessons_dir: PathBuf,

    /// Output SRAM file path
    #[structopt(short = "o", long = "output", default_value = "./logic.sram")]
    output: PathBuf,

    /// Seed for fingerprint generation
    #[structopt(short = "s", long = "seed", default_value = "0xDEADBEEF")]
    seed: u64,

    /// Bloom filter false positive rate
    #[structopt(short = "f", long = "fpr", default_value = "0.01")]
    fpr: f64,

    /// Verbose output
    #[structopt(short = "v", long = "verbose")]
    verbose: bool,
}

fn find_lesson_files(dir: &PathBuf) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();

    if !dir.exists() {
        return Err(Error::InvalidParam(format!(
            "Lessons directory does not exist: {}",
            dir.display()
        )));
    }

    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            if let Some(ext) = path.extension() {
                let ext_str = ext.to_string_lossy().to_lowercase();
                if ext_str == "txt" || ext_str == "md" {
                    files.push(path);
                }
            }
        }
    }

    files.sort();
    Ok(files)
}

fn extract_lesson_id(path: &PathBuf) -> u32 {
    // Extract lesson ID from filename (e.g., "lesson_001.txt" -> 1)
    let filename = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("0");

    // Try to extract numeric suffix
    let numeric_part = filename
        .chars()
        .filter(|c| c.is_ascii_digit())
        .collect::<String>();

    numeric_part.parse().unwrap_or(0)
}

fn read_lesson_content(path: &PathBuf) -> Result<String> {
    let content = fs::read_to_string(path)?;
    Ok(content.trim().to_string())
}

fn main() {
    let opt = Opt::from_args();

    // Find lesson files
    let lesson_files = match find_lesson_files(&opt.lessons_dir) {
        Ok(files) => files,
        Err(e) => {
            eprintln!("Error finding lesson files: {}", e);
            process::exit(1);
        }
    };

    if opt.verbose {
        println!("Found {} lesson files", lesson_files.len());
    }

    if lesson_files.is_empty() {
        eprintln!("No lesson files found in {}", opt.lessons_dir.display());
        process::exit(1);
    }

    // Process each lesson
    let mut builder = SramImageBuilder::new();
    let mut canary_fp: u64 = 0;

    for (i, path) in lesson_files.iter().enumerate() {
        let lesson_id = extract_lesson_id(path);
        let content = match read_lesson_content(path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Error reading {}: {}", path.display(), e);
                continue;
            }
        };

        let fp = fingerprint(&content, opt.seed);

        if opt.verbose {
            let filename = path.file_name().unwrap_or_default().to_string_lossy();
            println!(
                "  [{}] {} -> lesson {}: {:016x}",
                i + 1,
                filename,
                lesson_id,
                fp
            );
        }

        if i == 0 {
            canary_fp = fp;
        }

        builder = builder.add_record(fp, lesson_id);
    }

    // Set canary
    builder = builder.canary(canary_fp);

    // Build SRAM image
    let sram_image = match builder.build() {
        Ok(img) => img,
        Err(e) => {
            eprintln!("Error building SRAM image: {}", e);
            process::exit(1);
        }
    };

    // Save to file
    if let Err(e) = sram_image.save_to_file(&opt.output) {
        eprintln!("Error saving SRAM image: {}", e);
        process::exit(1);
    }

    println!(
        "Baked {} lessons to {} (canary: {:016x})",
        lesson_files.len(),
        opt.output.display(),
        canary_fp
    );
}
