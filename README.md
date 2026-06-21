# SURE

SURE learns tract-level urban representations with shared motifs,
profile-residual decomposition, gated residual injection, and spatial context
refinement. This repository contains the code and processed top-100-city data
needed to reproduce the SURE row in the main downstream table.

Generated checkpoints, logs, embeddings, and result tables are written to
`outputs/` and are not tracked.

## Layout

- `src/model.py`: SURE encoder and self-supervised losses.
- `src/pretrain.py`: pretraining loop.
- `src/downstream.py`: downstream linear probes.
- `src/summarize_sure_main.py`: main-table summarizer.
- `src/dataset.py`: processed graph tensor loader.
- `src/graph_utils.py`: graph utilities used by the encoder.
- `scripts/`: runnable entry points.
- `data/`: processed top-100-city tensors and downstream labels.

## Install

Run all commands from the repository root.

```bash
conda create -n sure python=3.11 -y
conda activate sure
pip install -r requirements.txt
```

The scripts use `PYTHON_BIN` if it is set. Otherwise they use `python` from the
active environment.

## Data

The included data bundle contains:

- `data/processed_source_destination/*.pt`
- `data/raw/us_top100_cities_counties.csv`
- `data/raw/downstream/task_npy_466county/`
- `data/top100_processed_counties_manifest.csv`

Check the bundle:

```bash
python - <<'PY'
from pathlib import Path
root = Path("data/processed_source_destination")
print("processed county tensors:", len(list(root.glob("*.pt"))))
print("county file:", Path("data/raw/us_top100_cities_counties.csv").exists())
print("downstream labels:", Path("data/raw/downstream/task_npy_466county").exists())
PY
```

## Main Experiment

The released main configuration is:

```text
num_motifs=32
dim=256
smooth_steps=3
view_smooth_type=learnable_diffuse
residual_fusion=attn_gated
tract_context_type=gcn
tract_context_position=post_residual
tract_context_layers=2
contrast_loss_weight=0.1
graph_contrast_loss_weight=0.03
balance_weight=0.003
poi_loss_weight=0.01
landuse_loss_weight=0.01
mobility_recon_loss_weight=0.03
```

Run the SURE main-table reproduction:

```bash
CUDA_VISIBLE_DEVICES=0 \
SEEDS=7,42,2026 \
RUN_TAG=sure_main_multiseed \
bash scripts/run_main_sure_multiseed.sh
```

Main outputs:

```text
outputs/sure_main_multiseed/summary/sure_main_table_row.csv
outputs/sure_main_multiseed/summary/sure_main_selected_metrics.csv
outputs/sure_main_multiseed/summary/sure_main_all_metrics.csv
outputs/sure_main_multiseed/summary/sure_main_per_seed_task_probe.csv
```

## Stage Commands

Single pretraining run:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/pretrain.sh \
  --nproc-per-node 1 \
  --processed-root data/processed_source_destination \
  --county-fips-file data/raw/us_top100_cities_counties.csv \
  --out-dir outputs/sure_top100/pretrain \
  --num-motifs 32 \
  --dim 256 \
  --smooth-steps 3 \
  --view-smooth-type learnable_diffuse \
  --residual-fusion attn_gated \
  --tract-context-type gcn \
  --tract-context-position post_residual \
  --tract-context-layers 2 \
  --contrast-loss-weight 0.1 \
  --graph-contrast-loss-weight 0.03 \
  --balance-weight 0.003 \
  --poi-loss-weight 0.01 \
  --landuse-loss-weight 0.01 \
  --mobility-recon-loss-weight 0.03 \
  --epochs 200 \
  --early-stopping-patience 20 \
  --amp bf16
```

Downstream probes:

```bash
bash scripts/downstream.sh \
  --processed-root data/processed_source_destination \
  --checkpoint outputs/sure_top100/pretrain/best.pt \
  --task-root data/raw/downstream/task_npy_466county \
  --county-fips-file data/raw/us_top100_cities_counties.csv \
  --tasks co2,employment,income,landcover,population,safety \
  --n-splits 3 \
  --out-dir outputs/sure_top100/downstream \
  --output-prefix sure_top100 \
  --model-name SURE \
  --embedding-path outputs/sure_top100/downstream/sure_top100_embeddings.npz \
  --embedding-type z \
  --num-motifs 32 \
  --dim 256 \
  --smooth-steps 3 \
  --view-smooth-type learnable_diffuse \
  --residual-fusion attn_gated \
  --tract-context-type gcn \
  --tract-context-position post_residual \
  --tract-context-layers 2 \
  --amp bf16
```

The six downstream tasks are `co2`, `employment`, `income`, `landcover`,
`population`, and `safety`.

## Upload Note

The processed graph tensors are about 340 MB, and the full `data/` directory is
about 358 MB. The largest tensor is below GitHub's 100 MB hard limit, but the
full bundle is still large. If a hosting service rejects the data directory,
upload `data/processed_source_destination/*.pt` as a separate anonymous artifact
and keep the same relative path after download.

Before uploading a local run directory, remove generated files:

```bash
rm -rf outputs
find . -type d -name '__pycache__' -prune -exec rm -rf {} +
find . -type f -name '*.pyc' -delete
```
