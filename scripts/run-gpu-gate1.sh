#!/usr/bin/env bash
set -euo pipefail
test "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -ge 1
precision-md prepare-data --config configs/gate1.yaml
precision-md benchmark --config configs/gate1.yaml --allow-gpu-benchmark
precision-md analyze --results results
precision-md render-report --results results --output report/decision.md
