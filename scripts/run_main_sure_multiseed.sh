#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  bash scripts/run_main_sure_multiseed.sh

Purpose:
  Reproduce the SURE main-table row.

Useful overrides:
  SEEDS=7,42,2026
  CUDA_VISIBLE_DEVICES=0
  RUN_TAG=sure_main_multiseed
  EPOCHS=200
  TASKS=co2,employment,income,landcover,population,safety

Default SURE main setting:
  K=32, T_d=3, dim=256
  reconstruction weights = 0.01/0.01/0.03 for POI/land-use/mobility
  contrast=0.1, graph_contrast=0.03, balance=0.003
EOF
  exit 0
fi

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    echo "python executable not found. Activate the environment or set PYTHON_BIN." >&2
    exit 1
  fi
fi
export PYTHON_BIN

HOST_TAG="${HOST_TAG:-$(hostname -s 2>/dev/null || echo host)}"
RUN_TAG="${RUN_TAG:-sure_main_multiseed_${HOST_TAG}_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/outputs/${RUN_TAG}}"

PROCESSED_ROOT="${PROCESSED_ROOT:-${ROOT_DIR}/data/processed_source_destination}"
TASK_ROOT="${TASK_ROOT:-${ROOT_DIR}/data/raw/downstream/task_npy_466county}"
COUNTY_FIPS_FILE="${COUNTY_FIPS_FILE:-${ROOT_DIR}/data/raw/us_top100_cities_counties.csv}"

SEEDS="${SEEDS:-7,42,2026}"
TASKS="${TASKS:-co2,employment,income,landcover,population,safety}"
N_SPLITS="${N_SPLITS:-3}"
DOWNSTREAM_SEED="${DOWNSTREAM_SEED:-2026}"
DOWNSTREAM_JOBS="${DOWNSTREAM_JOBS:-1}"

NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
EPOCHS="${EPOCHS:-200}"
VAL_RATIO="${VAL_RATIO:-0.1}"
EVAL_EVERY="${EVAL_EVERY:-1}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-20}"
EARLY_STOPPING_MIN_DELTA="${EARLY_STOPPING_MIN_DELTA:-0.0}"
EARLY_STOPPING_WARMUP="${EARLY_STOPPING_WARMUP:-0}"

NUM_MOTIFS="${NUM_MOTIFS:-32}"
DIM="${DIM:-256}"
SMOOTH_STEPS="${SMOOTH_STEPS:-3}"
TEMPERATURE="${TEMPERATURE:-0.15}"
VIEW_SMOOTH_TYPE="${VIEW_SMOOTH_TYPE:-learnable_diffuse}"
TRACT_CONTEXT_TYPE="${TRACT_CONTEXT_TYPE:-gcn}"
TRACT_CONTEXT_POSITION="${TRACT_CONTEXT_POSITION:-post_residual}"
TRACT_CONTEXT_GRAPH="${TRACT_CONTEXT_GRAPH:-spatial}"
TRACT_CONTEXT_LAYERS="${TRACT_CONTEXT_LAYERS:-2}"
RESIDUAL_FUSION="${RESIDUAL_FUSION:-attn_gated}"
DROPOUT="${DROPOUT:-0.05}"
MOTIF_CONSENSUS_TYPE="${MOTIF_CONSENSUS_TYPE:-shared}"
PROFILE_RESIDUAL_DECOMP="${PROFILE_RESIDUAL_DECOMP:-on}"
ENABLED_VIEWS="${ENABLED_VIEWS:-all}"

POI_LOSS_WEIGHT="${POI_LOSS_WEIGHT:-0.01}"
LANDUSE_LOSS_WEIGHT="${LANDUSE_LOSS_WEIGHT:-0.01}"
MOBILITY_RECON_LOSS_WEIGHT="${MOBILITY_RECON_LOSS_WEIGHT:-0.03}"
CONTRAST_LOSS_WEIGHT="${CONTRAST_LOSS_WEIGHT:-0.1}"
CONTRAST_TEMPERATURE="${CONTRAST_TEMPERATURE:-0.5}"
GRAPH_CONTRAST_LOSS_WEIGHT="${GRAPH_CONTRAST_LOSS_WEIGHT:-0.03}"
GRAPH_CONTRAST_TEMPERATURE="${GRAPH_CONTRAST_TEMPERATURE:-0.2}"
GRAPH_RECON_TEMPERATURE="${GRAPH_RECON_TEMPERATURE:-0.2}"
BALANCE_WEIGHT="${BALANCE_WEIGHT:-0.003}"
PAIRWISE_LOSS_SAMPLE_SIZE="${PAIRWISE_LOSS_SAMPLE_SIZE:-1024}"
MAX_ABS_EMBEDDING="${MAX_ABS_EMBEDDING:-20.0}"
LOGIT_CLIP="${LOGIT_CLIP:-30.0}"

LR="${LR:-0.0003}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.1}"
LOCAL_EPOCHS_PER_BATCH="${LOCAL_EPOCHS_PER_BATCH:-1}"
CITIES_PER_GPU="${CITIES_PER_GPU:-0}"
MAX_NODES_PER_GPU="${MAX_NODES_PER_GPU:-12000}"
BATCH_SORT_POOL_SIZE="${BATCH_SORT_POOL_SIZE:-32}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-4}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
DATASET_CACHE="${DATASET_CACHE:-cpu}"
AMP_MODE="${AMP_MODE:-bf16}"
PIN_MEMORY="${PIN_MEMORY:-1}"
PERSISTENT_WORKERS="${PERSISTENT_WORKERS:-1}"
FIND_UNUSED_PARAMETERS="${FIND_UNUSED_PARAMETERS:-1}"
RESUME_PRETRAIN="${RESUME_PRETRAIN:-1}"
FORCE_PRETRAIN="${FORCE_PRETRAIN:-0}"
FORCE_DOWNSTREAM="${FORCE_DOWNSTREAM:-0}"

mkdir -p "${OUT_ROOT}/logs"

run_stage() {
  local run_dir="$1"
  local stage="$2"
  shift 2
  mkdir -p "${run_dir}/logs"
  "$@" 2>&1 | tee "${run_dir}/logs/${stage}.log"
}

write_manifest() {
  local run_dir="$1"
  local seed="$2"
  cat > "${run_dir}/sure_main_manifest.env" <<EOF
RUN_TAG=${RUN_TAG}
RUN_DIR=${run_dir}
PROCESSED_ROOT=${PROCESSED_ROOT}
TASK_ROOT=${TASK_ROOT}
COUNTY_FIPS_FILE=${COUNTY_FIPS_FILE}
TASKS=${TASKS}
SEED=${seed}
DOWNSTREAM_SEED=${DOWNSTREAM_SEED}
NUM_MOTIFS=${NUM_MOTIFS}
DIM=${DIM}
SMOOTH_STEPS=${SMOOTH_STEPS}
VIEW_SMOOTH_TYPE=${VIEW_SMOOTH_TYPE}
RESIDUAL_FUSION=${RESIDUAL_FUSION}
TRACT_CONTEXT_TYPE=${TRACT_CONTEXT_TYPE}
TRACT_CONTEXT_POSITION=${TRACT_CONTEXT_POSITION}
TRACT_CONTEXT_GRAPH=${TRACT_CONTEXT_GRAPH}
TRACT_CONTEXT_LAYERS=${TRACT_CONTEXT_LAYERS}
POI_LOSS_WEIGHT=${POI_LOSS_WEIGHT}
LANDUSE_LOSS_WEIGHT=${LANDUSE_LOSS_WEIGHT}
MOBILITY_RECON_LOSS_WEIGHT=${MOBILITY_RECON_LOSS_WEIGHT}
CONTRAST_LOSS_WEIGHT=${CONTRAST_LOSS_WEIGHT}
GRAPH_CONTRAST_LOSS_WEIGHT=${GRAPH_CONTRAST_LOSS_WEIGHT}
BALANCE_WEIGHT=${BALANCE_WEIGHT}
EPOCHS=${EPOCHS}
EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE}
EOF
}

run_one_seed() {
  local seed="$1"
  local tag="seed_${seed}"
  local model_name="SURE_seed${seed}"
  local output_prefix="sure_main_seed${seed}"
  local run_dir="${OUT_ROOT}/${tag}"
  local pretrain_out="${run_dir}/pretrain"
  local downstream_out="${run_dir}/downstream"
  mkdir -p "${pretrain_out}" "${downstream_out}"
  write_manifest "${run_dir}" "${seed}"

  local pretrain_done="${run_dir}/pretrain.done"
  local downstream_done="${run_dir}/downstream.done"

  local pretrain_cmd=(
    bash "${ROOT_DIR}/scripts/pretrain.sh"
    --nproc-per-node "${NPROC_PER_NODE}"
    --processed-root "${PROCESSED_ROOT}"
    --county-fips-file "${COUNTY_FIPS_FILE}"
    --out-dir "${pretrain_out}"
    --val-ratio "${VAL_RATIO}"
    --eval-every "${EVAL_EVERY}"
    --early-stopping-patience "${EARLY_STOPPING_PATIENCE}"
    --early-stopping-min-delta "${EARLY_STOPPING_MIN_DELTA}"
    --early-stopping-warmup "${EARLY_STOPPING_WARMUP}"
    --num-motifs "${NUM_MOTIFS}"
    --dim "${DIM}"
    --smooth-steps "${SMOOTH_STEPS}"
    --temperature "${TEMPERATURE}"
    --view-smooth-type "${VIEW_SMOOTH_TYPE}"
    --tract-context-type "${TRACT_CONTEXT_TYPE}"
    --tract-context-position "${TRACT_CONTEXT_POSITION}"
    --tract-context-graph "${TRACT_CONTEXT_GRAPH}"
    --tract-context-layers "${TRACT_CONTEXT_LAYERS}"
    --residual-fusion "${RESIDUAL_FUSION}"
    --motif-consensus-type "${MOTIF_CONSENSUS_TYPE}"
    --profile-residual-decomp "${PROFILE_RESIDUAL_DECOMP}"
    --enabled-views "${ENABLED_VIEWS}"
    --poi-loss-weight "${POI_LOSS_WEIGHT}"
    --landuse-loss-weight "${LANDUSE_LOSS_WEIGHT}"
    --mobility-recon-loss-weight "${MOBILITY_RECON_LOSS_WEIGHT}"
    --contrast-loss-weight "${CONTRAST_LOSS_WEIGHT}"
    --contrast-temperature "${CONTRAST_TEMPERATURE}"
    --graph-contrast-loss-weight "${GRAPH_CONTRAST_LOSS_WEIGHT}"
    --graph-contrast-temperature "${GRAPH_CONTRAST_TEMPERATURE}"
    --graph-recon-temperature "${GRAPH_RECON_TEMPERATURE}"
    --balance-weight "${BALANCE_WEIGHT}"
    --pairwise-loss-sample-size "${PAIRWISE_LOSS_SAMPLE_SIZE}"
    --max-abs-embedding "${MAX_ABS_EMBEDDING}"
    --logit-clip "${LOGIT_CLIP}"
    --dropout "${DROPOUT}"
    --lr "${LR}"
    --weight-decay "${WEIGHT_DECAY}"
    --warmup-epochs "${WARMUP_EPOCHS}"
    --min-lr-ratio "${MIN_LR_RATIO}"
    --local-epochs-per-batch "${LOCAL_EPOCHS_PER_BATCH}"
    --epochs "${EPOCHS}"
    --cities-per-gpu "${CITIES_PER_GPU}"
    --max-nodes-per-gpu "${MAX_NODES_PER_GPU}"
    --batch-sort-pool-size "${BATCH_SORT_POOL_SIZE}"
    --grad-accum-steps "${GRAD_ACCUM_STEPS}"
    --grad-clip "${GRAD_CLIP}"
    --num-workers "${TRAIN_NUM_WORKERS}"
    --prefetch-factor "${PREFETCH_FACTOR}"
    --dataset-cache "${DATASET_CACHE}"
    --amp "${AMP_MODE}"
    --seed "${seed}"
  )
  [[ "${PIN_MEMORY}" == "1" ]] && pretrain_cmd+=(--pin-memory)
  [[ "${PERSISTENT_WORKERS}" == "1" ]] && pretrain_cmd+=(--persistent-workers)
  [[ "${FIND_UNUSED_PARAMETERS}" == "1" ]] && pretrain_cmd+=(--find-unused-parameters)
  [[ "${RESUME_PRETRAIN}" == "1" && "${FORCE_PRETRAIN}" != "1" && -f "${pretrain_out}/last.pt" ]] && pretrain_cmd+=(--resume "${pretrain_out}/last.pt")

  echo "[sure-main] seed=${seed} out=${run_dir}"
  if [[ "${FORCE_PRETRAIN}" == "1" || ! -f "${pretrain_done}" ]]; then
    run_stage "${run_dir}" pretrain "${pretrain_cmd[@]}"
    date '+%F %T' > "${pretrain_done}"
  else
    echo "[skip] pretrain seed=${seed}"
  fi

  local checkpoint="${pretrain_out}/best.pt"
  [[ -f "${checkpoint}" ]] || checkpoint="${pretrain_out}/last.pt"
  if [[ ! -f "${checkpoint}" ]]; then
    echo "Missing checkpoint for seed=${seed}: ${pretrain_out}" >&2
    exit 1
  fi

  local downstream_cmd=(
    bash "${ROOT_DIR}/scripts/downstream.sh"
    --processed-root "${PROCESSED_ROOT}"
    --checkpoint "${checkpoint}"
    --task-root "${TASK_ROOT}"
    --county-fips-file "${COUNTY_FIPS_FILE}"
    --tasks "${TASKS}"
    --n-splits "${N_SPLITS}"
    --seed "${DOWNSTREAM_SEED}"
    --n-jobs "${DOWNSTREAM_JOBS}"
    --out-dir "${downstream_out}"
    --output-prefix "${output_prefix}"
    --model-name "${model_name}"
    --embedding-path "${downstream_out}/${output_prefix}_embeddings.npz"
    --embedding-type z
    --num-motifs "${NUM_MOTIFS}"
    --dim "${DIM}"
    --smooth-steps "${SMOOTH_STEPS}"
    --view-smooth-type "${VIEW_SMOOTH_TYPE}"
    --tract-context-type "${TRACT_CONTEXT_TYPE}"
    --tract-context-position "${TRACT_CONTEXT_POSITION}"
    --tract-context-graph "${TRACT_CONTEXT_GRAPH}"
    --tract-context-layers "${TRACT_CONTEXT_LAYERS}"
    --residual-fusion "${RESIDUAL_FUSION}"
    --motif-consensus-type "${MOTIF_CONSENSUS_TYPE}"
    --profile-residual-decomp "${PROFILE_RESIDUAL_DECOMP}"
    --enabled-views "${ENABLED_VIEWS}"
    --amp "${AMP_MODE}"
  )

  if [[ "${FORCE_DOWNSTREAM}" == "1" || ! -f "${downstream_done}" ]]; then
    run_stage "${run_dir}" downstream "${downstream_cmd[@]}"
    date '+%F %T' > "${downstream_done}"
  else
    echo "[skip] downstream seed=${seed}"
  fi
}

IFS=',' read -r -a seed_array <<< "${SEEDS}"
for seed in "${seed_array[@]}"; do
  seed="$(echo "${seed}" | xargs)"
  [[ -z "${seed}" ]] && continue
  run_one_seed "${seed}"
done

"${PYTHON_BIN}" -m summarize_sure_main \
  --run-root "${OUT_ROOT}" \
  --out-dir "${OUT_ROOT}/summary"

echo "[sure-main] summary=${OUT_ROOT}/summary/sure_main_table_row.csv"
