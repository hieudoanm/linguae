# linguae-model-export

Exports the linguae CharCNN artifact (`../artifacts/linguae`) to
self-contained single-file model formats under `../artifacts/models`:

- `linguae.pt` — PyTorch checkpoint (`config` + `state_dict` + `vocab` + `labels`)
- `linguae.pkl` — pickle of `config` + numpy `weights` + `vocab` + `labels`
- `linguae.joblib` — same payload via joblib

## Usage

```bash
uv sync
uv run python export_models.py [artifact_dir] [out_dir]
```

Defaults: `../artifacts/linguae` → `../artifacts/models`.
