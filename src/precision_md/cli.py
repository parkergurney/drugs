from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from .analysis import analyze_results, analyze_trials
from .benchmark import run_benchmark
from .config import Gate1Config, Gate2Config, load_config
from .data import Frame, close_contact, ordinary_indices, select_high_force
from .model import MaceEvaluator
from .types import AtomicBatch
from .report import render_report


def prepare_data(config):
    """Load simple rMD17 NPZ archives without downloading licensed/remote assets."""
    config.output_dir.mkdir(parents=True, exist_ok=True); frames = []
    archives = {}
    for mol in config.molecules:
        path = config.dataset_dir / f"{mol}.npz"
        if not path.exists(): raise FileNotFoundError(f"missing rMD17 archive: {path}")
        archives[mol] = np.load(path)
    indices = ordinary_indices({m: len(a["R"]) for m, a in archives.items()}, config.ordinary_counts, config.seed)
    for mol, ids in indices.items():
        archive = archives[mol]
        numbers = archive["z"] if "z" in archive else archive["nuclear_charges"]
        for idx in ids: frames.append({"frame_id": f"{mol}-{idx}", "molecule": mol, "stratum": "ordinary", "atomic_numbers": numbers, "positions": archive["R"][idx]})
    if len(frames) != 100: raise RuntimeError("ordinary selection did not produce 100 frames")
    evaluator = MaceEvaluator(config.model, config.device)
    score_path = config.output_dir / "candidate_scores.parquet"
    candidates = []
    if score_path.exists():
        candidates = pd.read_parquet(score_path).to_dict("records")
        if candidates and any(row["seed"] != config.seed or row["model_hash"] != evaluator.model_hash
                              for row in candidates):
            raise RuntimeError(f"candidate score checkpoint does not match this run: {score_path}")
        print(f"Resuming from {len(candidates)}/{config.candidate_pool} candidate scores", flush=True)
    completed = {row["frame_id"] for row in candidates}
    rng = np.random.default_rng(config.seed + 1)
    allocations = [config.candidate_pool // len(config.molecules)] * len(config.molecules)
    for i in range(config.candidate_pool % len(config.molecules)): allocations[i] += 1
    for mol, n in zip(config.molecules, allocations, strict=True):
        archive = archives[mol]; numbers = archive["z"] if "z" in archive else archive["nuclear_charges"]
        for idx in np.sort(rng.choice(len(archive["R"]), n, replace=False)):
            frame_id = f"{mol}-{idx}"
            if frame_id in completed:
                continue
            result = evaluator.evaluate(AtomicBatch(numbers, archive["R"][idx]), "fp32")
            if not result.finite: raise RuntimeError(f"FP32 candidate scoring failed: {mol}-{idx}: {result.error}")
            candidates.append({"frame_id": frame_id, "molecule": mol, "frame_index": int(idx),
                               "max_force": float(np.linalg.norm(result.forces, axis=1).max()),
                               "seed": config.seed, "model_hash": evaluator.model_hash})
            completed.add(frame_id)
            if len(candidates) % 50 == 0:
                temporary = score_path.with_suffix(".tmp.parquet")
                pd.DataFrame(candidates).to_parquet(temporary, index=False)
                temporary.replace(score_path)
                print(f"Scored {len(candidates)}/{config.candidate_pool} candidates", flush=True)
    if len(candidates) != config.candidate_pool:
        raise RuntimeError(f"expected {config.candidate_pool} candidate scores, found {len(candidates)}")
    ordinary_ids = {frame["frame_id"] for frame in frames}
    eligible_candidates = [
        candidate for candidate in candidates
        if candidate["frame_id"] not in ordinary_ids
    ]
    selected = select_high_force(
        eligible_candidates, config.high_force_count,
        config.high_force_cap_per_molecule,
    )
    high = []
    for row in selected:
        archive = archives[row["molecule"]]
        numbers = archive["z"] if "z" in archive else archive["nuclear_charges"]
        high.append({"frame_id": row["frame_id"], "molecule": row["molecule"],
                     "stratum": "high_force", "atomic_numbers": numbers,
                     "positions": archive["R"][row["frame_index"]]})
    close = []
    for i, row in enumerate(frames):
        source = Frame(row["frame_id"], row["molecule"], row["atomic_numbers"], row["positions"])
        generated = close_contact(source, config.close_contact_distances[i % len(config.close_contact_distances)], i)
        close.append({"frame_id": generated.frame_id, "molecule": generated.molecule, "stratum": generated.stratum,
                      "atomic_numbers": generated.atomic_numbers, "positions": generated.positions})
    all_frames = frames + high + close
    if len(all_frames) != 300: raise RuntimeError("Gate 1 construction must produce exactly 300 frames")
    np.savez_compressed(config.output_dir / "frames.npz", frames=np.array(all_frames, dtype=object))
    (config.output_dir / "selection.json").write_text(json.dumps({"seed": config.seed, "frames": [f["frame_id"] for f in all_frames]}, indent=2) + "\n")
    print(f"Prepared {len(all_frames)} Gate 1 frames in {config.output_dir}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="precision-md"); sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare-data", "benchmark", "prepare-trajectory", "fork-segments"):
        p = sub.add_parser(name); p.add_argument("--config", required=True)
        if name == "benchmark":
            p.add_argument("--allow-gpu-benchmark", action="store_true")
            p.add_argument("--frames", help="path to one frozen frames.npz")
            p.add_argument("--run-id", help="unique subdirectory under config output_dir")
            p.add_argument("--timing-seed", type=int,
                           help="policy-order seed, independent of frame selection")
    p = sub.add_parser("analyze"); p.add_argument("--results", required=True)
    p = sub.add_parser("analyze-trials")
    p.add_argument("--trials", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("render-report"); p.add_argument("--results", required=True); p.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare-data": prepare_data(load_config(args.config, Gate1Config))
    elif args.command == "benchmark":
        run_benchmark(load_config(args.config, Gate1Config),
                      args.allow_gpu_benchmark, args.frames, args.run_id,
                      args.timing_seed)
    elif args.command in ("prepare-trajectory", "fork-segments"):
        config = load_config(args.config, Gate2Config)
        from .gpu_workflow import fork_segments, prepare_trajectory
        (prepare_trajectory if args.command == "prepare-trajectory" else fork_segments)(config)
    elif args.command == "analyze": print(json.dumps(analyze_results(args.results), indent=2))
    elif args.command == "analyze-trials":
        print(json.dumps(analyze_trials(args.trials, args.output), indent=2))
    elif args.command == "render-report": render_report(args.results, args.output)


if __name__ == "__main__": main()
