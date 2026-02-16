# Copilot Instructions — songwalker-cli

## Project Overview

`songwalker-cli` is the command-line interface for SongWalker. It renders `.sw` files to audio and provides a tuner utility.

## Dependency on songwalker-core

Direct Cargo path dependency:
```toml
songwalker_core = { path = "../songwalker-core" }
```

Changes to songwalker-core are picked up automatically on `cargo build` / `cargo test`.

## Key Files

- `src/main.rs` — CLI entry: file rendering, WAV export
- `src/tuner_cli.rs` — interactive tuner utility

## Testing

```bash
cargo test
```

No unit tests exist yet. Tests from songwalker-core run independently.

## Deploying Core Updates

When songwalker-core changes:
```bash
cd /home/ari/dev/songwalker-cli
cargo test    # recompiles with updated core automatically
```

No additional build steps needed — the path dependency ensures the latest core is used.
