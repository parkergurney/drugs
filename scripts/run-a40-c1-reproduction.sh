#!/usr/bin/env bash
set -euo pipefail

readonly EXPERIMENT_ID="C1-A40-P1-REPRODUCTION"
readonly P1_SOURCE="data/frozen/p1"
readonly P1_DATASET="artifacts/datasets/p1"
readonly C1_STAGING="results/datasets/c1-confirmatory"
readonly C1_DATASET="artifacts/datasets/c1"
readonly TRIALS="artifacts/trials/c1-reproduction"
readonly ANALYSIS="artifacts/analysis/c1-reproduction"
readonly SYSTEM_DIR="artifacts/system"
readonly BUNDLES="artifacts/bundles"
readonly EXPECTED_P1_SHA256="f7e759b6f0050b82eae88ff99416a2d43f50eac9e2e944a7524e80eaff40a28d"
readonly EXPECTED_MODEL_SHA256="9bd176f569bb26925f5d8ae7779e01babafaed42bece49f34cb1f561925a8149"

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "Refusing scientific run from a dirty tracked worktree" >&2
  exit 1
fi

gpu_count="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')"
if [[ "$gpu_count" != "1" ]]; then
  echo "Expected exactly one visible GPU, found: $gpu_count" >&2
  exit 1
fi
gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')"
if [[ "$gpu_name" != *"A40"* ]]; then
  echo "Expected an NVIDIA A40, found: $gpu_name" >&2
  exit 1
fi

sha256sum -c data/rmd17/GPU-SHA256SUMS
observed_p1_sha256="$(sha256sum "$P1_SOURCE/frames.npz" | awk '{print $1}')"
if [[ "$observed_p1_sha256" != "$EXPECTED_P1_SHA256" ]]; then
  echo "P1 frame checksum mismatch: $observed_p1_sha256" >&2
  exit 1
fi

mkdir -p "$SYSTEM_DIR" "$BUNDLES"
git rev-parse HEAD > "$SYSTEM_DIR/git-commit.txt"
git status --porcelain > "$SYSTEM_DIR/git-status.txt"
uname -a > "$SYSTEM_DIR/platform.txt"
nvidia-smi -q > "$SYSTEM_DIR/nvidia-smi-before.txt"

uv sync --extra ml --extra test
uv run pytest
uv run python -c \
  'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
uv pip freeze > "$SYSTEM_DIR/environment.txt"

# P1 is imported, never recomputed. The study manifest supplies legacy provenance.
uv run precision-md freeze-dataset \
  --source "$P1_SOURCE" \
  --output "$P1_DATASET" \
  --dataset-id p1 \
  --provenance studies/pilot-p1-a40/manifest.json

# C1 preparation is FP32-only and resumes candidate scoring from its staging table.
# The immutable bundle is created only after the complete staging output validates.
if [[ ! -d "$C1_DATASET" ]]; then
  uv run precision-md prepare-data --config configs/c1-dataset.yaml
  uv run precision-md freeze-dataset \
    --source "$C1_STAGING" \
    --output "$C1_DATASET" \
    --dataset-id c1-confirmatory
fi
uv run precision-md validate-dataset --dataset "$C1_DATASET" --dataset-id c1-confirmatory

tar -C artifacts/datasets -czf "$BUNDLES/precision-md-c1-dataset.tar.gz" c1
sha256sum "$BUNDLES/precision-md-c1-dataset.tar.gz" \
  > "$BUNDLES/precision-md-c1-dataset.tar.gz.sha256"

wait_for_idle_gpu() {
  local consecutive=0
  local attempt utilization
  for attempt in $(seq 1 120); do
    utilization="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | sed -n '1p' | tr -d ' ')"
    if [[ "$utilization" =~ ^[0-9]+$ ]] && (( utilization <= 5 )); then
      consecutive=$((consecutive + 1))
      if (( consecutive >= 3 )); then
        return 0
      fi
    else
      consecutive=0
    fi
    sleep 1
  done
  echo "GPU did not remain idle before the next trial" >&2
  return 1
}

run_ids=(c1-run-01 c1-run-02 c1-run-03 c1-run-04 c1-run-05)
timing_seeds=(2026081901 2026081902 2026081903 2026081904 2026081905)

for index in "${!run_ids[@]}"; do
  run_id="${run_ids[$index]}"
  timing_seed="${timing_seeds[$index]}"
  run_dir="$TRIALS/$run_id"
  if [[ -d "$run_dir" ]]; then
    if uv run precision-md validate-trial \
      --trial "$run_dir" --dataset "$P1_DATASET" --experiment-id "$EXPERIMENT_ID"; then
      echo "Skipping complete verified trial: $run_id"
      continue
    fi
    echo "Refusing to overwrite incomplete or invalid trial: $run_dir" >&2
    exit 1
  fi

  wait_for_idle_gpu
  mkdir -p "$run_dir"
  nvidia-smi \
    --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,power.draw,clocks.sm,clocks.mem,memory.used \
    --format=csv > "$run_dir/gpu-before.csv"

  (
    nvidia-smi \
      --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,power.draw,clocks.sm,clocks.mem,memory.used \
      --format=csv -l 1 > "$run_dir/gpu-telemetry.csv" &
    telemetry_pid=$!
    cleanup_telemetry() {
      kill "$telemetry_pid" 2>/dev/null || true
      wait "$telemetry_pid" 2>/dev/null || true
    }
    trap cleanup_telemetry EXIT INT TERM
    uv run precision-md benchmark \
      --config configs/c1-reproduction.yaml \
      --frames "$P1_DATASET/frames.npz" \
      --run-id "$run_id" \
      --timing-seed "$timing_seed" \
      --experiment-id "$EXPERIMENT_ID" \
      --allow-gpu-benchmark
    cleanup_telemetry
    trap - EXIT INT TERM
  )

  nvidia-smi \
    --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,power.draw,clocks.sm,clocks.mem,memory.used \
    --format=csv > "$run_dir/gpu-after.csv"
  uv run precision-md validate-trial \
    --trial "$run_dir" --dataset "$P1_DATASET" --experiment-id "$EXPERIMENT_ID"
  uv run python -c \
    "import json; p=json.load(open('$run_dir/manifest.json')); assert p['model_hash']=='$EXPECTED_MODEL_SHA256'"
done

uv run precision-md analyze-trials --trials "$TRIALS" --output "$ANALYSIS"
nvidia-smi -q > "$SYSTEM_DIR/nvidia-smi-after.txt"

find artifacts -path "$BUNDLES" -prune -o -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum > artifacts/SHA256SUMS
tar -C artifacts -czf "$BUNDLES/precision-md-c1-a40-results.tar.gz" \
  datasets trials analysis system SHA256SUMS
sha256sum "$BUNDLES/precision-md-c1-a40-results.tar.gz" \
  > "$BUNDLES/precision-md-c1-a40-results.tar.gz.sha256"

echo "C1 preparation and five-process P1 reproduction are complete."
echo "Download artifacts/bundles and verify both archive checksums before terminating the instance."
echo "The C1 dataset has not been evaluated under TF32 or BF16."
