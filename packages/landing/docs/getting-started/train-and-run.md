# Train and run

## Train the model (optional)

A prebuilt artifact may already be present; retrain to reproduce it:

```bash
cd crates
cargo run --release -p linguae-data -- all   # build the WiLI, Tatoeba and word CSVs
cargo run --release -p linguae-train -- train
```

The trainer writes the artifact to `crates/linguae-train/artifacts/linguae`
and SVG plots to `crates/linguae-train/artifacts/images`.

## Predict

```bash
crates/target/release/linguae "Bonjour tout le monde"
```

The CLI predicts exactly one language per input string.
