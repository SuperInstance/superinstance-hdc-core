//! # monitor CLI
//!
//! Monitors the resonance HUD from shared memory state.
//!
//! Reads `/dev/shm/superinstance/state.bin` every 100ms and displays
//! a real-time resonance meter.
//!
//! Format: [density: f32][lesson_id: u32][magnitude: f32]
//!
//! Example output:
//! ```text
//! [Lesson 003] Resonance: 87.32% |██████████████████████████████████|
//! ```

use std::fs::File;
use std::io::{self, Read};
use std::path::Path;
use std::time::Duration;
use structopt::StructOpt;
use termcolor::{Color, ColorChoice, ColorSpec, StandardStream, WriteColor};

/// State bin format packed structure
#[repr(C)]
struct StateBin {
    density: f32,
    lesson_id: u32,
    magnitude: f32,
}

/// CLI for monitoring resonance HUD
#[derive(Debug, StructOpt)]
#[structopt(
    name = "monitor",
    about = "Monitor resonance HUD from shared memory state",
    author = "SuperInstance"
)]
struct Opt {
    /// State bin file path
    #[structopt(short = "s", long = "state", default_value = "/dev/shm/superinstance/state.bin")]
    state_file: String,

    /// Update interval in milliseconds
    #[structopt(short = "i", long = "interval", default_value = "100")]
    interval: u64,

    /// Lesson name mapping file (optional)
    #[structopt(short = "m", long = "mapping")]
    mapping: Option<String>,

    /// Maximum bar width
    #[structopt(short = "w", long = "width", default_value = "40")]
    bar_width: usize,

    /// Verbose output
    #[structopt(short = "v", long = "verbose")]
    verbose: bool,
}

impl Opt {
    fn read_state(&self) -> Option<StateBin> {
        let path = Path::new(&self.state_file);
        if !path.exists() {
            return None;
        }

        let mut file = match File::open(path) {
            Ok(f) => f,
            Err(_) => return None,
        };

        let mut buffer = [0u8; 12]; // 4 + 4 + 4 = 12 bytes
        if file.read_exact(&mut buffer).is_err() {
            return None;
        }

        Some(StateBin {
            density: f32::from_le_bytes([buffer[0], buffer[1], buffer[2], buffer[3]]),
            lesson_id: u32::from_le_bytes([buffer[4], buffer[5], buffer[6], buffer[7]]),
            magnitude: f32::from_le_bytes([buffer[8], buffer[9], buffer[10], buffer[11]]),
        })
    }

    fn format_bar(&self, resonance: f32) -> String {
        let filled = ((resonance / 100.0) * self.bar_width as f32) as usize;
        let empty = self.bar_width.saturating_sub(filled);

        let filled_str = "█".repeat(filled.min(self.bar_width));
        let empty_str = "░".repeat(empty.min(self.bar_width));

        format!("|{}{}|", filled_str, empty_str)
    }

    fn color_for_resonance(resonance: f32) -> Color {
        if resonance >= 80.0 {
            Color::Green
        } else if resonance >= 50.0 {
            Color::Yellow
        } else if resonance >= 20.0 {
            Color::Magenta
        } else {
            Color::Red
        }
    }
}

fn main() {
    let opt = Opt::from_args();

    if opt.verbose {
        println!(
            "Monitoring {} every {}ms",
            opt.state_file, opt.interval
        );
    }

    let mut stdout = StandardStream::stdout(ColorChoice::Auto);

    // Load lesson name mapping if provided
    let lesson_names: Option<std::collections::HashMap<u32, String>> = if let Some(ref mapping_file) = opt.mapping {
        load_lesson_mapping(mapping_file).ok()
    } else {
        None
    };

    loop {
        if let Some(state) = opt.read_state() {
            let resonance = state.density * 100.0;
            let lesson_display: String = if let Some(ref names) = lesson_names {
                names
                    .get(&state.lesson_id)
                    .cloned()
                    .unwrap_or_else(|| format!("Lesson {:03}", state.lesson_id))
            } else {
                format!("Lesson {:03}", state.lesson_id)
            };

            let bar = opt.format_bar(resonance);
            let color = Opt::color_for_resonance(resonance);

            // Color and print the line
            if let Err(_) = stdout.set_color(ColorSpec::new().set_fg(Some(color))) {
                // Fallback if colors not supported
                println!(
                    "[{}] Resonance: {:5.2}% {} ({:.2}m)",
                    lesson_display, resonance, bar, state.magnitude
                );
            } else {
                println!(
                    "[{}] Resonance: {:5.2}% {} ({:.2}m)",
                    lesson_display, resonance, bar, state.magnitude
                );
                let _ = stdout.set_color(&ColorSpec::new());
            }
        } else {
            // No state available
            if let Err(_) = stdout.set_color(ColorSpec::new().set_fg(Some(Color::Blue))) {
                println!("[--] Waiting for state... ");
            } else {
                println!("[--] Waiting for state... ");
                let _ = stdout.set_color(&ColorSpec::new());
            }
        }

        std::thread::sleep(Duration::from_millis(opt.interval));
    }
}

fn load_lesson_mapping(path: &str) -> io::Result<std::collections::HashMap<u32, String>> {
    let content = std::fs::read_to_string(path)?;
    let mut map = std::collections::HashMap::new();

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let parts: Vec<&str> = line.splitn(2, ':').collect();
        if parts.len() == 2 {
            if let Ok(id) = parts[0].trim().parse::<u32>() {
                map.insert(id, parts[1].trim().to_string());
            }
        }
    }

    Ok(map)
}
