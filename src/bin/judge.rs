//! # judge CLI
//!
//! Judges student input against a baked SRAM image.
//!
//! Usage:
//! ```bash
//! echo "your answer" | judge --sram logic.sram --seed 0xDEADBEEF
//! judge "your answer" --sram logic.sram --threshold 5
//! ```

use std::io::{self, Read};
use std::process;
use structopt::StructOpt;

use superinstance_hdc_core::{judge, judge_detailed, SramImage};

/// CLI for judging student input against SRAM image
#[derive(Debug, StructOpt)]
#[structopt(
    name = "judge",
    about = "Judge student input against SRAM image",
    author = "SuperInstance"
)]
struct Opt {
    /// SRAM image file path
    #[structopt(short = "s", long = "sram", default_value = "./logic.sram")]
    sram: String,

    /// Seed for fingerprint generation
    #[structopt(short = "S", long = "seed", default_value = "0xDEADBEEF")]
    seed: u64,

    /// Hamming distance threshold for fuzzy matching
    #[structopt(short = "t", long = "threshold", default_value = "10")]
    threshold: u32,

    /// Input text (if not provided, reads from stdin)
    #[structopt(default_value = "")]
    input: String,

    /// Show detailed output (distance, confidence, etc.)
    #[structopt(short = "d", long = "detailed")]
    detailed: bool,

    /// Verbose output
    #[structopt(short = "v", long = "verbose")]
    verbose: bool,
}

fn main() {
    let opt = Opt::from_args();

    // Load SRAM image
    let sram = match SramImage::load_from_file(&opt.sram) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error loading SRAM image: {}", e);
            process::exit(1);
        }
    };

    if opt.verbose {
        println!(
            "Loaded SRAM with {} records (canary: {:016x})",
            sram.record_count(),
            sram.canary()
        );
    }

    // Get input (stdin or argument)
    let input = if opt.input.is_empty() {
        let mut buf = String::new();
        if let Err(e) = io::stdin().read_to_string(&mut buf) {
            eprintln!("Error reading from stdin: {}", e);
            process::exit(1);
        }
        buf.trim().to_string()
    } else {
        opt.input.clone()
    };

    if input.is_empty() {
        eprintln!("No input provided");
        process::exit(1);
    }

    if opt.verbose {
        let display_len = input.len().min(50);
        println!("Judging input: {}...", &input[..display_len]);
    }

    // Perform judgment
    if opt.detailed {
        let judgment = judge_detailed(&sram, &input, opt.seed, opt.threshold);

        println!("Input hash: {:016x}", judgment.input_hash);
        println!("Bloom passed: {}", judgment.bloom_passed);

        if let Some(distance) = judgment.distance {
            println!("Closest distance: {}", distance);
            let confidence = 1.0 - (distance as f64 / opt.threshold as f64);
            println!("Confidence: {:.2}", confidence);
        }

        match judgment.lesson_id {
            Some(lesson_id) => {
                println!("MATCH: Lesson {}", lesson_id);
                process::exit(0);
            }
            None => {
                println!("NOMATCH");
                process::exit(1);
            }
        }
    } else {
        let result = judge(&sram, &input, opt.seed, opt.threshold);

        match result {
            Some(lesson_id) => {
                println!("MATCH {} ", lesson_id);
                process::exit(0);
            }
            None => {
                println!("NOMATCH");
                process::exit(1);
            }
        }
    }
}
