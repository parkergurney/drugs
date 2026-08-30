#!/usr/bin/env bash
set -euo pipefail

readonly CONFIG="configs/d1-p1.yaml"
readonly P1_SOURCE="data/frozen/p1"
readonly P1_DATASET="artifacts/datasets/p1"
readonly D1_ROOT="artifacts/diagnostics/d1-p1"
readonly SYSTEM_DIR="artifacts/system/d1-p1"
readonly BUNDLES="artifacts/bundles"
readonly EXPECTED_TAG="d1-p1-tracer-fix-v2"

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "Refusing D1 from a dirty tracked worktree" >&2
  exit 1
fi
if [[ "$(git describe --exact-match --tags 2>/dev/null || true)" != "$EXPECTED_TAG" ]]; then
  echo "D1 must run from tag $EXPECTED_TAG" >&2
  exit 1
fi

gpu_count="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')"
gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')"
if [[ "$gpu_count" != "1" || "$gpu_name" != *"A40"* ]]; then
  echo "D1 requires exactly one visible NVIDIA A40; found $gpu_count: $gpu_name" >&2
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
  'import torch; assert torch.__version__.startswith("2.4.1"); assert torch.cuda.is_available(); assert "A40" in torch.cuda.get_device_name(0); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
uv pip freeze > "$SYSTEM_DIR/environment.txt"

uv run precision-md freeze-dataset \
  --source "$P1_SOURCE" --output "$P1_DATASET" --dataset-id p1 \
  --provenance studies/pilot-p1-a40/manifest.json
uv run precision-md validate-dataset --dataset "$P1_DATASET" --dataset-id p1
uv run precision-md select-d1 --config "$CONFIG"
uv run python -c \
  'import json; p=json.load(open("artifacts/diagnostics/d1-p1/d1-selection.json")); assert p["frame_count"] == 27'

wait_for_idle_gpu() {
  local consecutive=0 utilization
  for _ in $(seq 1 120); do
    utilization="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')"
    if [[ "$utilization" =~ ^[0-9]+$ ]] && (( utilization <= 5 )); then
      consecutive=$((consecutive + 1))
      if (( consecutive >= 3 )); then return 0; fi
    else
      consecutive=0
    fi
    sleep 1
  done
  echo "GPU did not remain idle" >&2
  return 1
}

run_with_telemetry() {
  local run_dir="$1"
  shift
  mkdir -p "$run_dir"
  wait_for_idle_gpu
  nvidia-smi \
    --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,power.draw,clocks.sm,clocks.mem,memory.used \
    --format=csv > "$run_dir/gpu-before.csv"
  (
    nvidia-smi \
      --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,power.draw,clocks.sm,clocks.mem,memory.used \
      --format=csv -l 1 > "$run_dir/gpu-telemetry.csv" &
    telemetry_pid=$!
    cleanup() {
      kill "$telemetry_pid" 2>/dev/null || true
      wait "$telemetry_pid" 2>/dev/null || true
    }
    trap cleanup EXIT INT TERM
    "$@"
    cleanup
    trap - EXIT INT TERM
  )
  nvidia-smi \
    --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,power.draw,clocks.sm,clocks.mem,memory.used \
    --format=csv > "$run_dir/gpu-after.csv"
}

trace_dir="$D1_ROOT/runs/d1-trace-01"
if [[ ! -f "$trace_dir/manifest.json" ]]; then
  run_with_telemetry "$trace_dir" uv run precision-md diagnose-d1 \
    --config "$CONFIG" --run-id d1-trace-01 --allow-gpu-diagnostic
fi

run_ids=(d1-time-01 d1-time-02 d1-time-03)
timing_seeds=(2026082801 2026082802 2026082803)
for index in "${!run_ids[@]}"; do
  run_id="${run_ids[$index]}"
  run_dir="$D1_ROOT/runs/$run_id"
  if [[ -f "$run_dir/manifest.json" ]]; then
    echo "Skipping completed timing process: $run_id"
    continue
  fi
  run_with_telemetry "$run_dir" uv run precision-md time-d1 \
    --config "$CONFIG" --run-id "$run_id" \
    --timing-seed "${timing_seeds[$index]}" --allow-gpu-benchmark
done

if [[ ! -f "$D1_ROOT/manifest.json" ]]; then
  uv run precision-md analyze-d1 --config "$CONFIG"
fi
uv run precision-md validate-d1 --config "$CONFIG"
nvidia-smi -q > "$SYSTEM_DIR/nvidia-smi-after.txt"

tar -C artifacts -czf "$BUNDLES/precision-md-d1-a40-results.tar.gz" \
  diagnostics/d1-p1 system/d1-p1
sha256sum "$BUNDLES/precision-md-d1-a40-results.tar.gz" \
  > "$BUNDLES/precision-md-d1-a40-results.tar.gz.sha256"

echo "D1 is complete. Download both bundle files and verify the archive checksum before terminating the Pod."
