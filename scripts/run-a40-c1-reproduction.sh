#!/usr/bin/env bash
set -euo pipefail

readonly FRAMES="data/frozen/p1/frames.npz"
readonly EXPECTED_FRAMES_SHA256="f7e759b6f0050b82eae88ff99416a2d43f50eac9e2e944a7524e80eaff40a28d"
readonly EXPECTED_MODEL_SHA256="9bd176f569bb26925f5d8ae7779e01babafaed42bece49f34cb1f561925a8149"
readonly SYSTEM_DIR="results/system/c1-reproduction"

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "Refusing scientific run from a dirty tracked worktree" >&2
  exit 1
fi

test -f "$FRAMES"
observed_frames_sha256="$(sha256sum "$FRAMES" | awk '{print $1}')"
test "$observed_frames_sha256" = "$EXPECTED_FRAMES_SHA256"

gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')"
if [[ "$gpu_name" != *"A40"* ]]; then
  echo "Expected an NVIDIA A40, found: $gpu_name" >&2
  exit 1
fi

mkdir -p "$SYSTEM_DIR"
git rev-parse HEAD > "$SYSTEM_DIR/git-commit.txt"
git status --porcelain > "$SYSTEM_DIR/git-status.txt"
uv pip freeze > "$SYSTEM_DIR/environment.txt"
nvidia-smi -q > "$SYSTEM_DIR/nvidia-smi-before.txt"

run_ids=(c1-run-01 c1-run-02 c1-run-03 c1-run-04 c1-run-05)
timing_seeds=(2026081901 2026081902 2026081903 2026081904 2026081905)

for index in "${!run_ids[@]}"; do
  run_id="${run_ids[$index]}"
  timing_seed="${timing_seeds[$index]}"
  run_dir="results/gate1/$run_id"
  mkdir -p "$run_dir"
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
    --config configs/gate1.yaml \
    --frames "$FRAMES" \
    --run-id "$run_id" \
    --timing-seed "$timing_seed" \
    --allow-gpu-benchmark

  cleanup_telemetry
  trap - EXIT INT TERM

  uv run python -c \
    "import json; p=json.load(open('$run_dir/manifest.json')); assert p['frames_sha256']=='$EXPECTED_FRAMES_SHA256'; assert p['model_hash']=='$EXPECTED_MODEL_SHA256'; assert p['run_id']=='$run_id'; assert p['timing_seed']==$timing_seed"
done

nvidia-smi -q > "$SYSTEM_DIR/nvidia-smi-after.txt"
uv run precision-md analyze-trials \
  --trials results/gate1 \
  --output results/c1-analysis

echo "Five-process A40 reproduction complete; preserve results/ before shutdown."
