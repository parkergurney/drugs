#!/usr/bin/env bash
set -euo pipefail
test "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -ge 1
uv run precision-md prepare-data --config configs/gate1.yaml
uv run precision-md benchmark --config configs/gate1.yaml --allow-gpu-benchmark
uv run precision-md analyze --results results
uv run precision-md render-report --results results --output report/decision.md
