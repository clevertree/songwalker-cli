//! CLI commands for preset pitch detection and tuning.
//!
//! Subcommands:
//!   detect-pitch <path> [--recursive]     Detect pitch of each zone
//!   tune <path> [--apply] [--recursive] [--threshold <cents>]

use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use songwalker_core::dsp::tuner::{PitchEstimate, detect_pitch};
use std::fs;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::process;
use walkdir::WalkDir;

// ── Preset JSON helpers (serde_json::Value-based) ───────────

/// Extract sampler zones from a preset JSON value.
/// Handles both `node.config.zones` and `graph.config.zones` layouts.
fn get_zones(preset: &serde_json::Value) -> Option<&Vec<serde_json::Value>> {
    let node = preset.get("node").or_else(|| preset.get("graph"))?;
    let config = node.get("config")?;
    config.get("zones")?.as_array()
}

/// Mutable version: extract zones array for modification.
fn get_zones_mut(preset: &mut serde_json::Value) -> Option<&mut Vec<serde_json::Value>> {
    // Try "node" first, fall back to "graph"
    let node = if preset.get("node").is_some() {
        preset.get_mut("node").unwrap()
    } else {
        preset.get_mut("graph")?
    };
    let config = node.get_mut("config")?;
    config.get_mut("zones")?.as_array_mut()
}

/// Read audio samples from a zone, returning (samples_f64, sample_rate).
fn read_zone_audio(zone: &serde_json::Value, preset_dir: &Path) -> Result<(Vec<f64>, u32), String> {
    let sample_rate = zone.get("sampleRate")
        .and_then(|v| v.as_u64())
        .unwrap_or(32000) as u32;

    let audio = zone.get("audio").ok_or("zone has no 'audio' field")?;
    let audio_type = audio.get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    match audio_type {
        "inline-pcm" => {
            let data_b64 = audio.get("data")
                .and_then(|v| v.as_str())
                .ok_or("inline-pcm missing 'data'")?;
            let bps = audio.get("bitsPerSample")
                .and_then(|v| v.as_u64())
                .unwrap_or(16) as u8;
            let raw = BASE64.decode(data_b64)
                .map_err(|e| format!("base64 decode error: {e}"))?;
            let samples = pcm_bytes_to_f64(&raw, bps);
            Ok((samples, sample_rate))
        }
        "external" => {
            let url = audio.get("url")
                .and_then(|v| v.as_str())
                .ok_or("external audio missing 'url'")?;
            let codec = audio.get("codec")
                .and_then(|v| v.as_str())
                .unwrap_or("wav");
            let file_path = preset_dir.join(url);
            if !file_path.exists() {
                return Err(format!("audio file not found: {}", file_path.display()));
            }
            decode_audio_file(&file_path, codec, sample_rate)
        }
        other => Err(format!("unsupported audio type: '{other}'")),
    }
}

/// Convert raw PCM bytes (little-endian) to f64 samples in [-1, 1].
fn pcm_bytes_to_f64(raw: &[u8], bits_per_sample: u8) -> Vec<f64> {
    match bits_per_sample {
        16 => {
            raw.chunks_exact(2)
                .map(|c| {
                    let sample = i16::from_le_bytes([c[0], c[1]]);
                    sample as f64 / 32768.0
                })
                .collect()
        }
        24 => {
            raw.chunks_exact(3)
                .map(|c| {
                    let sample = ((c[0] as i32) | ((c[1] as i32) << 8) | ((c[2] as i32) << 16))
                        << 8 >> 8; // sign-extend
                    sample as f64 / 8388608.0
                })
                .collect()
        }
        8 => {
            raw.iter().map(|&b| (b as f64 - 128.0) / 128.0).collect()
        }
        _ => {
            eprintln!("  Warning: unsupported bitsPerSample={bits_per_sample}, treating as 16-bit");
            pcm_bytes_to_f64(raw, 16)
        }
    }
}

/// Decode an audio file (MP3 or WAV) to mono f64 samples.
fn decode_audio_file(path: &Path, codec: &str, _declared_rate: u32) -> Result<(Vec<f64>, u32), String> {
    let file_data = fs::read(path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;

    match codec {
        "mp3" => decode_mp3(&file_data),
        "wav" => decode_wav(path),
        "flac" => Err("FLAC decoding not yet supported".into()),
        "ogg" => Err("OGG decoding not yet supported".into()),
        other => Err(format!("unsupported codec: '{other}'")),
    }
}

/// Decode MP3 data to mono f64 samples using minimp3.
fn decode_mp3(data: &[u8]) -> Result<(Vec<f64>, u32), String> {
    let mut decoder = minimp3::Decoder::new(Cursor::new(data));
    let mut all_samples: Vec<f64> = Vec::new();
    let mut sample_rate = 44100u32;

    loop {
        match decoder.next_frame() {
            Ok(frame) => {
                sample_rate = frame.sample_rate as u32;
                let channels = frame.channels;
                // Mix to mono
                if channels == 1 {
                    all_samples.extend(frame.data.iter().map(|&s| s as f64 / 32768.0));
                } else {
                    for chunk in frame.data.chunks(channels) {
                        let mono: f64 = chunk.iter().map(|&s| s as f64).sum::<f64>()
                            / (channels as f64 * 32768.0);
                        all_samples.push(mono);
                    }
                }
            }
            Err(minimp3::Error::Eof) => break,
            Err(e) => return Err(format!("MP3 decode error: {e:?}")),
        }
    }

    if all_samples.is_empty() {
        return Err("MP3 decoded to 0 samples".into());
    }
    Ok((all_samples, sample_rate))
}

/// Decode WAV file to mono f64 samples using hound.
fn decode_wav(path: &Path) -> Result<(Vec<f64>, u32), String> {
    let reader = hound::WavReader::open(path)
        .map_err(|e| format!("WAV open error: {e}"))?;
    let spec = reader.spec();
    let sample_rate = spec.sample_rate;
    let channels = spec.channels as usize;

    let samples: Vec<f64> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max_val = (1i64 << (spec.bits_per_sample - 1)) as f64;
            let all: Vec<i32> = reader.into_samples::<i32>()
                .filter_map(|s| s.ok())
                .collect();
            // Mix to mono
            if channels == 1 {
                all.iter().map(|&s| s as f64 / max_val).collect()
            } else {
                all.chunks(channels)
                    .map(|c| c.iter().map(|&s| s as f64).sum::<f64>() / (channels as f64 * max_val))
                    .collect()
            }
        }
        hound::SampleFormat::Float => {
            let all: Vec<f32> = reader.into_samples::<f32>()
                .filter_map(|s| s.ok())
                .collect();
            if channels == 1 {
                all.iter().map(|&s| s as f64).collect()
            } else {
                all.chunks(channels)
                    .map(|c| c.iter().map(|&s| s as f64).sum::<f64>() / channels as f64)
                    .collect()
            }
        }
    };

    Ok((samples, sample_rate))
}

// ── File discovery ──────────────────────────────────────────

/// Find all preset.json files under a path.
fn find_presets(path: &Path, recursive: bool) -> Vec<PathBuf> {
    if path.is_file() {
        return vec![path.to_path_buf()];
    }

    if !path.is_dir() {
        eprintln!("Error: '{}' is not a file or directory", path.display());
        process::exit(1);
    }

    if recursive {
        WalkDir::new(path)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name() == "preset.json")
            .map(|e| e.into_path())
            .collect()
    } else {
        // Non-recursive: only direct children
        let mut results = Vec::new();
        if let Ok(entries) = fs::read_dir(path) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.is_file() && p.file_name().is_some_and(|n| n == "preset.json") {
                    results.push(p);
                } else if p.is_dir() {
                    let candidate = p.join("preset.json");
                    if candidate.exists() {
                        results.push(candidate);
                    }
                }
            }
        }
        results.sort();
        results
    }
}

/// Compute a reasonable pitch detection frequency range for a given MIDI note.
/// Returns (min_freq, max_freq) centered around the expected fundamental,
/// spanning roughly ±1 octave.
fn detection_range_for_note(midi_note: u8) -> (f64, f64) {
    let expected_freq = 440.0 * 2.0_f64.powf((midi_note as f64 - 69.0) / 12.0);
    // Search ±1 octave around expected pitch, clamped to reasonable limits
    let min_f = (expected_freq / 2.0).max(20.0);
    let max_f = (expected_freq * 2.0).min(8000.0);
    (min_f, max_f)
}

/// Extract a stable segment of audio for pitch analysis.
///
/// MP3-compressed samples often have a long decay — the YIN algorithm works
/// best on a stable, sustained portion of the waveform. This function finds
/// the loudest portion and returns a window around it.
fn extract_stable_segment(samples: &[f64], sample_rate: u32) -> Vec<f64> {
    if samples.len() < 4096 {
        return samples.to_vec();
    }

    let sr = sample_rate as usize;
    // Window size: ~50ms, enough for several cycles of low notes
    let window = (sr / 20).max(2048).min(samples.len() / 4);

    // Find the point with maximum RMS energy using a sliding window
    let step = window / 4;
    let mut best_start = 0;
    let mut best_rms = 0.0f64;

    let mut pos = 0;
    while pos + window <= samples.len() {
        let rms: f64 = (samples[pos..pos + window]
            .iter()
            .map(|s| s * s)
            .sum::<f64>()
            / window as f64)
            .sqrt();
        if rms > best_rms {
            best_rms = rms;
            best_start = pos;
        }
        pos += step;
    }

    // Take a segment around the loudest point, at least 3x window for YIN
    let segment_len = (window * 4).min(samples.len());
    let start = if best_start + segment_len > samples.len() {
        samples.len().saturating_sub(segment_len)
    } else {
        best_start
    };
    let end = (start + segment_len).min(samples.len());

    samples[start..end].to_vec()
}

/// MIDI note name from number.
fn midi_note_name(note: u8) -> String {
    const NAMES: [&str; 12] = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
    let octave = (note as i32 / 12) - 1;
    let name = NAMES[note as usize % 12];
    format!("{name}{octave}")
}

// ── detect-pitch command ────────────────────────────────────

pub fn cmd_detect_pitch(args: &[String]) {
    if args.is_empty() {
        eprintln!("Error: detect-pitch requires a path argument");
        eprintln!("Usage: songwalker_cli detect-pitch <preset.json|directory> [--recursive]");
        process::exit(1);
    }

    let path = Path::new(&args[0]);
    let recursive = args.iter().any(|a| a == "--recursive" || a == "-r");
    let presets = find_presets(path, recursive);

    if presets.is_empty() {
        eprintln!("No preset.json files found in '{}'", path.display());
        process::exit(1);
    }

    println!("Scanning {} preset(s)...\n", presets.len());

    let mut total_zones = 0usize;
    let mut melodic_zones = 0usize;
    let mut noise_zones = 0usize;

    for preset_path in &presets {
        let preset_dir = preset_path.parent().unwrap_or(Path::new("."));
        let content = match fs::read_to_string(preset_path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("  ✗ {}: read error: {e}", preset_path.display());
                continue;
            }
        };
        let preset: serde_json::Value = match serde_json::from_str(&content) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("  ✗ {}: JSON parse error: {e}", preset_path.display());
                continue;
            }
        };

        let name = preset.get("name").and_then(|v| v.as_str()).unwrap_or("?");
        let zones = match get_zones(&preset) {
            Some(z) => z,
            None => {
                // Not a sampler preset — skip silently
                continue;
            }
        };

        if zones.is_empty() {
            continue;
        }

        println!("── {} ({} zones)", name, zones.len());

        for (i, zone) in zones.iter().enumerate() {
            total_zones += 1;
            let pitch = zone.get("pitch");
            let root_note = pitch.and_then(|p| p.get("rootNote")).and_then(|v| v.as_u64()).unwrap_or(60) as u8;
            let fine_cents = pitch.and_then(|p| p.get("fineTuneCents")).and_then(|v| v.as_f64()).unwrap_or(0.0);
            let key_range = zone.get("keyRange");
            let kr_low = key_range.and_then(|kr| kr.get("low")).and_then(|v| v.as_u64()).unwrap_or(0) as u8;
            let kr_high = key_range.and_then(|kr| kr.get("high")).and_then(|v| v.as_u64()).unwrap_or(127) as u8;

            match read_zone_audio(zone, preset_dir) {
                Ok((samples, sr)) => {
                    // Auto-adjust detection range based on expected root note
                    let (min_f, max_f) = detection_range_for_note(root_note);
                    // For pitch detection, use a segment around the loudest part
                    let analysis_samples = extract_stable_segment(&samples, sr);
                    let estimate = detect_pitch(&analysis_samples, sr, Some(min_f), Some(max_f));
                    if estimate.is_noise {
                        noise_zones += 1;
                        println!("  zone[{i}] ({}-{}): NOISE/PERCUSSION  confidence={:.2}  declared={}",
                            midi_note_name(kr_low), midi_note_name(kr_high),
                            estimate.confidence,
                            midi_note_name(root_note));
                    } else {
                        melodic_zones += 1;
                        let expected_freq = 440.0 * 2.0_f64.powf(
                            (root_note as f64 - 69.0 + fine_cents / 100.0) / 12.0
                        );
                        let deviation = if estimate.frequency > 0.0 && expected_freq > 0.0 {
                            1200.0 * (estimate.frequency / expected_freq).log2()
                        } else {
                            0.0
                        };
                        let flag = if deviation.abs() > 10.0 { " ⚠" } else { "" };
                        println!("  zone[{i}] ({}-{}): detected={} ({:.1}Hz) confidence={:.2}  \
                                  declared={} ({:.1}Hz)  deviation={:+.1}¢{}",
                            midi_note_name(kr_low), midi_note_name(kr_high),
                            midi_note_name(estimate.midi_note), estimate.frequency,
                            estimate.confidence,
                            midi_note_name(root_note), expected_freq,
                            deviation, flag);
                    }
                }
                Err(e) => {
                    eprintln!("  zone[{i}] ({}-{}): audio error: {e}",
                        midi_note_name(kr_low), midi_note_name(kr_high));
                }
            }
        }
        println!();
    }

    println!("Summary: {total_zones} zones scanned, {melodic_zones} melodic, {noise_zones} noise/percussion");
}

// ── tune command ────────────────────────────────────────────

pub fn cmd_tune(args: &[String]) {
    if args.is_empty() {
        eprintln!("Error: tune requires a path argument");
        eprintln!("Usage: songwalker_cli tune <preset.json|directory> [--apply] [--recursive] [--threshold <cents>]");
        process::exit(1);
    }

    let path = Path::new(&args[0]);
    let apply = args.iter().any(|a| a == "--apply");
    let recursive = args.iter().any(|a| a == "--recursive" || a == "-r");
    let threshold: f64 = args.windows(2)
        .find(|w| w[0] == "--threshold" || w[0] == "-t")
        .and_then(|w| w[1].parse().ok())
        .unwrap_or(10.0);

    let presets = find_presets(path, recursive);

    if presets.is_empty() {
        eprintln!("No preset.json files found in '{}'", path.display());
        process::exit(1);
    }

    println!("Tuning {} preset(s)  threshold={threshold}¢  apply={apply}\n", presets.len());

    let mut stats = TuneStats::default();

    for preset_path in &presets {
        tune_preset(preset_path, threshold, apply, &mut stats);
    }

    println!("\n═══ Summary ═══");
    println!("  Presets scanned:  {}", stats.presets_scanned);
    println!("  Total zones:      {}", stats.total_zones);
    println!("  Melodic zones:    {}", stats.melodic_zones);
    println!("  Noise/percussion: {}", stats.noise_zones);
    println!("  Needs adjustment: {} (>{threshold}¢ deviation)", stats.needs_adjustment);
    println!("  Adjusted:         {}", stats.adjusted);
    println!("  Errors:           {}", stats.errors);
    if !apply && stats.needs_adjustment > 0 {
        println!("\n  Run with --apply to write corrections.");
    }
}

#[derive(Default)]
struct TuneStats {
    presets_scanned: usize,
    total_zones: usize,
    melodic_zones: usize,
    noise_zones: usize,
    needs_adjustment: usize,
    adjusted: usize,
    errors: usize,
}

fn tune_preset(preset_path: &Path, threshold: f64, apply: bool, stats: &mut TuneStats) {
    let preset_dir = preset_path.parent().unwrap_or(Path::new("."));
    stats.presets_scanned += 1;

    let content = match fs::read_to_string(preset_path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("  ✗ {}: read error: {e}", preset_path.display());
            stats.errors += 1;
            return;
        }
    };

    let mut preset: serde_json::Value = match serde_json::from_str(&content) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("  ✗ {}: JSON parse error: {e}", preset_path.display());
            stats.errors += 1;
            return;
        }
    };

    let name = preset.get("name").and_then(|v| v.as_str()).unwrap_or("?").to_string();

    // First pass: read-only analysis using immutable zones
    let zones_ref = match get_zones(&preset) {
        Some(z) => z.clone(),
        None => return, // not a sampler
    };

    if zones_ref.is_empty() {
        return;
    }

    // Analyse each zone and collect corrections
    struct ZoneCorrection {
        index: usize,
        estimate: PitchEstimate,
        deviation_cents: f64,
        trustworthy: bool,
    }

    let mut corrections: Vec<ZoneCorrection> = Vec::new();
    for (i, zone) in zones_ref.iter().enumerate() {
        stats.total_zones += 1;

        let pitch = zone.get("pitch");
        let root_note = pitch.and_then(|p| p.get("rootNote")).and_then(|v| v.as_u64()).unwrap_or(60) as u8;
        let fine_cents = pitch.and_then(|p| p.get("fineTuneCents")).and_then(|v| v.as_f64()).unwrap_or(0.0);

        match read_zone_audio(zone, preset_dir) {
            Ok((samples, sr)) => {
                let (min_f, max_f) = detection_range_for_note(root_note);
                let analysis_samples = extract_stable_segment(&samples, sr);
                let estimate = detect_pitch(&analysis_samples, sr, Some(min_f), Some(max_f));

                if estimate.is_noise {
                    stats.noise_zones += 1;
                    continue;
                }

                stats.melodic_zones += 1;
                let expected_freq = 440.0 * 2.0_f64.powf(
                    (root_note as f64 - 69.0 + fine_cents / 100.0) / 12.0
                );
                let deviation = if estimate.frequency > 0.0 && expected_freq > 0.0 {
                    1200.0 * (estimate.frequency / expected_freq).log2()
                } else {
                    0.0
                };

                if deviation.abs() > threshold {
                    stats.needs_adjustment += 1;
                    // Only trust corrections with high confidence and reasonable deviation.
                    // Deviations >100¢ (one semitone) usually indicate a detection error, not a tuning problem.
                    let trustworthy = estimate.confidence >= 0.70 && deviation.abs() <= 100.0;
                    corrections.push(ZoneCorrection {
                        index: i,
                        estimate,
                        deviation_cents: deviation,
                        trustworthy,
                    });
                }
            }
            Err(e) => {
                eprintln!("  ✗ zone[{i}] audio error: {e}");
                stats.errors += 1;
            }
        }
    }

    if !corrections.is_empty() {
        println!("── {} ({} zones, {} need adjustment)", name, zones_ref.len(), corrections.len());

        for correction in &corrections {
            let i = correction.index;
            let est = &correction.estimate;
            let zone = &zones_ref[i];
            let pitch = zone.get("pitch");
            let root_note = pitch.and_then(|p| p.get("rootNote")).and_then(|v| v.as_u64()).unwrap_or(60) as u8;
            let kr = zone.get("keyRange");
            let kr_low = kr.and_then(|k| k.get("low")).and_then(|v| v.as_u64()).unwrap_or(0) as u8;
            let kr_high = kr.and_then(|k| k.get("high")).and_then(|v| v.as_u64()).unwrap_or(127) as u8;

            let trust_flag = if correction.trustworthy { "" } else { " ⚠ LOW CONFIDENCE" };
            println!("  zone[{i}] ({}-{}): {} → {} ({:+.1}¢)  confidence={:.2}{trust_flag}",
                midi_note_name(kr_low), midi_note_name(kr_high),
                midi_note_name(root_note), midi_note_name(est.midi_note),
                correction.deviation_cents, est.confidence);
        }

        // Apply corrections
        if apply {
            let trusted: Vec<&ZoneCorrection> = corrections.iter().filter(|c| c.trustworthy).collect();
            let skipped = corrections.len() - trusted.len();
            if skipped > 0 {
                println!("  ⚠ Skipping {skipped} zone(s) with low-confidence detections");
            }
            if let Some(zones_mut) = get_zones_mut(&mut preset) {
                for correction in &trusted {
                    let zone = &mut zones_mut[correction.index];
                    let est = &correction.estimate;

                    if let Some(pitch) = zone.get_mut("pitch") {
                        pitch["rootNote"] = serde_json::json!(est.midi_note);
                        pitch["fineTuneCents"] = serde_json::json!((est.fine_tune_cents * 10.0).round() / 10.0);
                    }
                    stats.adjusted += 1;
                }
            }

            // Update tuning metadata
            let all_melodic = corrections.iter().all(|c| !c.estimate.is_noise);
            let first_est = &corrections[0].estimate;
            preset["tuning"] = serde_json::json!({
                "verified": false,
                "isMelodic": all_melodic,
                "detectedPitchHz": first_est.frequency,
                "needsAdjustment": false,
            });

            // Write back
            let formatted = serde_json::to_string_pretty(&preset).unwrap();
            match fs::write(preset_path, formatted.as_bytes()) {
                Ok(()) => println!("  ✓ Written: {}", preset_path.display()),
                Err(e) => {
                    eprintln!("  ✗ Write error: {e}");
                    stats.errors += 1;
                }
            }
        }
    }
}
