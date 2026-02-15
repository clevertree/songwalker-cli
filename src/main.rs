//! SongWalker CLI — Compile, render, and tune .sw files and presets.
//!
//! Usage:
//!   songwalker_cli <input.sw> [output.wav]
//!   songwalker_cli --check <input.sw>
//!   songwalker_cli --ast <input.sw>
//!   songwalker_cli detect-pitch <preset.json|directory> [--recursive]
//!   songwalker_cli tune <preset.json|directory> [--apply] [--recursive] [--threshold <cents>]

use songwalker_core::{compiler, dsp, parse};
use std::env;
use std::fs;

use std::process;

mod tuner_cli;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_usage(&args[0]);
        process::exit(1);
    }

    match args[1].as_str() {
        "--check" => {
            if args.len() < 3 {
                eprintln!("Error: --check requires a file argument");
                process::exit(1);
            }
            cmd_check(&args[2]);
        }
        "--ast" => {
            if args.len() < 3 {
                eprintln!("Error: --ast requires a file argument");
                process::exit(1);
            }
            cmd_ast(&args[2]);
        }
        "detect-pitch" => {
            tuner_cli::cmd_detect_pitch(&args[2..]);
        }
        "tune" => {
            tuner_cli::cmd_tune(&args[2..]);
        }
        "--help" | "-h" | "help" => {
            print_usage(&args[0]);
        }
        _ => {
            let input = &args[1];
            let output = if args.len() >= 3 {
                args[2].clone()
            } else {
                // Replace .sw extension with .wav, or append .wav
                if input.ends_with(".sw") {
                    format!("{}.wav", &input[..input.len() - 3])
                } else {
                    format!("{input}.wav")
                }
            };
            cmd_render(input, &output);
        }
    }
}

fn print_usage(program: &str) {
    eprintln!("SongWalker CLI v{}", env!("CARGO_PKG_VERSION"));
    eprintln!();
    eprintln!("Usage:");
    eprintln!("  {program} <input.sw> [output.wav]           Render to WAV");
    eprintln!("  {program} --check <input.sw>                Check syntax only");
    eprintln!("  {program} --ast <input.sw>                  Print AST");
    eprintln!();
    eprintln!("Preset tuning:");
    eprintln!("  {program} detect-pitch <path> [--recursive]");
    eprintln!("      Detect pitch of each zone in preset(s).");
    eprintln!("      <path> can be a preset.json or a directory.");
    eprintln!();
    eprintln!("  {program} tune <path> [options]");
    eprintln!("      Analyse and optionally fix tuning of preset(s).");
    eprintln!("      --apply            Write corrected pitch to preset files");
    eprintln!("      --recursive        Walk directories recursively");
    eprintln!("      --threshold <N>    Adjustment threshold in cents (default: 10)");
}

fn read_source(path: &str) -> String {
    match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error reading '{path}': {e}");
            process::exit(1);
        }
    }
}

fn cmd_check(path: &str) {
    let source = read_source(path);

    match parse(&source) {
        Ok(program) => {
            match compiler::compile(&program) {
                Ok(event_list) => {
                    println!(
                        "✓ {path}: {} events, {:.1} beats",
                        event_list.events.len(),
                        event_list.total_beats,
                    );
                }
                Err(e) => {
                    eprintln!("Compile error in '{path}': {e}");
                    process::exit(1);
                }
            }
        }
        Err(e) => {
            eprintln!("Parse error in '{path}': {e}");
            process::exit(1);
        }
    }
}

fn cmd_ast(path: &str) {
    let source = read_source(path);

    match parse(&source) {
        Ok(program) => {
            println!("{program:#?}");
        }
        Err(e) => {
            eprintln!("Parse error in '{path}': {e}");
            process::exit(1);
        }
    }
}

fn cmd_render(input: &str, output: &str) {
    let source = read_source(input);

    // Parse
    let program = match parse(&source) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Parse error: {e}");
            process::exit(1);
        }
    };

    // Compile
    let event_list = match compiler::compile(&program) {
        Ok(el) => el,
        Err(e) => {
            eprintln!("Compile error: {e}");
            process::exit(1);
        }
    };

    let sample_rate = 44100;
    let total_beats = event_list.total_beats;
    let num_events = event_list.events.len();

    // Render to WAV
    let wav_data = dsp::renderer::render_wav(&event_list, sample_rate);

    // Write output
    match fs::write(output, &wav_data) {
        Ok(()) => {
            let duration_sec = total_beats * 60.0 / 120.0; // default BPM for display
            let size_kb = wav_data.len() / 1024;
            println!("✓ Rendered '{input}' → '{output}'");
            println!(
                "  {num_events} events, {total_beats:.1} beats, ~{duration_sec:.1}s, {size_kb} KB",
            );
        }
        Err(e) => {
            eprintln!("Error writing '{output}': {e}");
            process::exit(1);
        }
    }
}
