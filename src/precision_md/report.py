from __future__ import annotations
import json
from pathlib import Path


def render_report(results, output):
    results, output = Path(results), Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    analysis = json.loads((results / "analysis.json").read_text()) if (results / "analysis.json").exists() else {"verdict": "STOP"}
    manifests = []
    for path in results.glob("**/manifest.json"):
        manifests.append((str(path.relative_to(results)), json.loads(path.read_text())))
    text = ["# Adaptive precision feasibility decision", "", f"## Verdict: `{analysis['verdict']}`", "",
            "This report is a feasibility result. It does not establish long-run ensemble equivalence.", "",
            "## Hardware and software manifests", ""]
    if manifests:
        for name, value in manifests: text += [f"### {name}", "", "```json", json.dumps(value, indent=2), "```", ""]
    else: text += ["No completed benchmark manifest was found.", ""]
    text += ["## Precision-policy failures", "", "See `gate1/evaluations.parquet` for unsupported operations and nonfinite results.", "",
             "## Numerical discrepancy, speed, and memory", "", "See `gate1/summary.json`, timing samples, and generated plots.", "",
             "## Unsafe blocks, predictors, and oracle economics", "", json.dumps(analysis.get("gate2", {}), indent=2), "",
             "## Second-architecture confirmation", "",
             "Not run because the primary-hardware feasibility gate did not pass.", ""]
    output.write_text("\n".join(text), encoding="utf-8")
