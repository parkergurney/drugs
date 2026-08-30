from __future__ import annotations
import argparse, json, shlex, sys
from pathlib import Path
import numpy as np
import pandas as pd
from .analysis import analyze_results, analyze_trials
from .benchmark import run_benchmark
from .config import D1Config, Gate1Config, Gate2Config, load_config
from .data import (
    Frame, close_contact, ensure_disjoint_sources, load_excluded_sources,
    ordinary_indices, select_high_force, sha256_file,
)
from .model import MaceEvaluator
from .types import AtomicBatch
from .report import render_report
from .artifacts import freeze_dataset, validate_frozen_dataset, validate_trial


def prepare_data(config):
    """Load simple rMD17 NPZ archives without downloading licensed/remote assets."""
    config.output_dir.mkdir(parents=True, exist_ok=True); frames = []
    excluded_sources, exclusion_hash = load_excluded_sources(config.exclude_selection)
    archives = {}
    source_hashes = {}
    for mol in config.molecules:
        path = config.dataset_dir / f"{mol}.npz"
        if not path.exists(): raise FileNotFoundError(f"missing rMD17 archive: {path}")
        archives[mol] = np.load(path)
        source_hashes[mol] = sha256_file(path)
    indices = ordinary_indices(
        {m: len(a["R"]) for m, a in archives.items()}, config.ordinary_counts,
        config.seed, excluded_sources,
    )
    for mol, ids in indices.items():
        archive = archives[mol]
        numbers = archive["z"] if "z" in archive else archive["nuclear_charges"]
        for idx in ids:
            frame_id = f"{mol}-{idx}"
            frames.append({"frame_id": frame_id, "source_frame_id": frame_id,
                           "molecule": mol, "stratum": "ordinary",
                           "atomic_numbers": numbers, "positions": archive["R"][idx]})
    if len(frames) != 100: raise RuntimeError("ordinary selection did not produce 100 frames")
    evaluator = MaceEvaluator(config.model, config.device)
    score_path = config.output_dir / "candidate_scores.parquet"
    candidates = []
    if score_path.exists():
        candidates = pd.read_parquet(score_path).to_dict("records")
        if candidates and any(
            row["seed"] != config.seed or row["model_hash"] != evaluator.model_hash
            or row.get("dataset_id", "gate1") != config.dataset_id
            or row.get("exclude_selection_sha256") != exclusion_hash
            for row in candidates
        ):
            raise RuntimeError(f"candidate score checkpoint does not match this run: {score_path}")
        print(f"Resuming from {len(candidates)}/{config.candidate_pool} candidate scores", flush=True)
    completed = {row["frame_id"] for row in candidates}
    rng = np.random.default_rng(config.seed + 1)
    allocations = [config.candidate_pool // len(config.molecules)] * len(config.molecules)
    for i in range(config.candidate_pool % len(config.molecules)): allocations[i] += 1
    for mol, n in zip(config.molecules, allocations, strict=True):
        archive = archives[mol]; numbers = archive["z"] if "z" in archive else archive["nuclear_charges"]
        eligible_indices = np.array([
            index for index in range(len(archive["R"]))
            if f"{mol}-{index}" not in excluded_sources
        ])
        if len(eligible_indices) < n:
            raise RuntimeError(f"insufficient non-excluded candidate frames for {mol}")
        for idx in np.sort(rng.choice(eligible_indices, n, replace=False)):
            frame_id = f"{mol}-{idx}"
            if frame_id in completed:
                continue
            result = evaluator.evaluate(AtomicBatch(numbers, archive["R"][idx]), "fp32")
            if not result.finite: raise RuntimeError(f"FP32 candidate scoring failed: {mol}-{idx}: {result.error}")
            candidates.append({"frame_id": frame_id, "molecule": mol, "frame_index": int(idx),
                               "max_force": float(np.linalg.norm(result.forces, axis=1).max()),
                               "seed": config.seed, "model_hash": evaluator.model_hash,
                               "dataset_id": config.dataset_id,
                               "exclude_selection_sha256": exclusion_hash})
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
                     "source_frame_id": row["frame_id"], "stratum": "high_force",
                     "atomic_numbers": numbers,
                     "positions": archive["R"][row["frame_index"]]})
    close = []
    for i, row in enumerate(frames):
        source = Frame(row["frame_id"], row["molecule"], row["atomic_numbers"], row["positions"])
        generated = close_contact(source, config.close_contact_distances[i % len(config.close_contact_distances)], i)
        close.append({"frame_id": generated.frame_id, "source_frame_id": row["frame_id"],
                      "molecule": generated.molecule, "stratum": generated.stratum,
                      "atomic_numbers": generated.atomic_numbers, "positions": generated.positions})
    all_frames = frames + high + close
    if len(all_frames) != 300: raise RuntimeError("Gate 1 construction must produce exactly 300 frames")
    selected_sources = {frame["source_frame_id"] for frame in all_frames}
    overlap = sorted(selected_sources & excluded_sources)
    ensure_disjoint_sources(selected_sources, excluded_sources)
    np.savez_compressed(config.output_dir / "frames.npz", frames=np.array(all_frames, dtype=object))
    selection = {
        "schema_version": 2,
        "dataset_id": config.dataset_id,
        "seed": config.seed,
        "frames": [frame["frame_id"] for frame in all_frames],
        "source_frames": [frame["source_frame_id"] for frame in all_frames],
        "records": [
            {key: frame[key] for key in ("frame_id", "source_frame_id", "molecule", "stratum")}
            for frame in all_frames
        ],
    }
    selection_path = config.output_dir / "selection.json"
    selection_path.write_text(json.dumps(selection, indent=2) + "\n")
    manifest = {
        "schema_version": 1,
        "dataset_id": config.dataset_id,
        "seed": config.seed,
        "model": config.model,
        "model_hash": evaluator.model_hash,
        "source_dataset_sha256": source_hashes,
        "exclude_selection": str(config.exclude_selection) if config.exclude_selection else None,
        "exclude_selection_sha256": exclusion_hash,
        "excluded_source_count": len(excluded_sources),
        "overlap_count": len(overlap),
        "stratum_counts": {
            stratum: sum(frame["stratum"] == stratum for frame in all_frames)
            for stratum in ("ordinary", "high_force", "close_contact")
        },
        "selection_sha256": sha256_file(selection_path),
        "frames_sha256": sha256_file(config.output_dir / "frames.npz"),
        "selection_policy": "FP32 high-force scoring; no reduced-precision outcomes inspected",
    }
    (config.output_dir / "dataset-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"Prepared {len(all_frames)} Gate 1 frames in {config.output_dir}", flush=True)


def main(argv=None):
    program = Path(sys.argv[0]).name if argv is None else "precisemd"
    parser = argparse.ArgumentParser(prog=program); sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare-data", "benchmark", "prepare-trajectory", "fork-segments"):
        p = sub.add_parser(name); p.add_argument("--config", required=True)
        if name == "benchmark":
            p.add_argument("--allow-gpu-benchmark", action="store_true")
            p.add_argument("--frames", help="path to one frozen frames.npz")
            p.add_argument("--run-id", help="unique subdirectory under config output_dir")
            p.add_argument("--timing-seed", type=int,
                           help="policy-order seed, independent of frame selection")
            p.add_argument("--experiment-id",
                           help="prospective experiment identifier recorded in manifest")
    p = sub.add_parser("analyze"); p.add_argument("--results", required=True)
    p = sub.add_parser("analyze-trials")
    p.add_argument("--trials", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("freeze-dataset")
    p.add_argument("--source", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--dataset-id", required=True)
    p.add_argument("--provenance")
    p = sub.add_parser("validate-dataset")
    p.add_argument("--dataset", required=True)
    p.add_argument("--dataset-id")
    p = sub.add_parser("validate-trial")
    p.add_argument("--trial", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--experiment-id")
    p = sub.add_parser("render-report"); p.add_argument("--results", required=True); p.add_argument("--output", required=True)
    p = sub.add_parser("select-d1")
    p.add_argument("--config", required=True)
    p = sub.add_parser("diagnose-d1")
    p.add_argument("--config", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--allow-gpu-diagnostic", action="store_true")
    p = sub.add_parser("time-d1")
    p.add_argument("--config", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--timing-seed", required=True, type=int)
    p.add_argument("--allow-gpu-benchmark", action="store_true")
    p = sub.add_parser("analyze-d1")
    p.add_argument("--config", required=True)
    p = sub.add_parser("validate-d1")
    p.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare-data": prepare_data(load_config(args.config, Gate1Config))
    elif args.command == "benchmark":
        run_benchmark(load_config(args.config, Gate1Config),
                      args.allow_gpu_benchmark, args.frames, args.run_id,
                      args.timing_seed, args.experiment_id,
                      shlex.join([program, *(argv if argv is not None else sys.argv[1:])]))
    elif args.command in ("prepare-trajectory", "fork-segments"):
        config = load_config(args.config, Gate2Config)
        from .gpu_workflow import fork_segments, prepare_trajectory
        (prepare_trajectory if args.command == "prepare-trajectory" else fork_segments)(config)
    elif args.command == "analyze": print(json.dumps(analyze_results(args.results), indent=2))
    elif args.command == "analyze-trials":
        print(json.dumps(analyze_trials(args.trials, args.output), indent=2))
    elif args.command == "freeze-dataset":
        print(json.dumps(freeze_dataset(args.source, args.output, args.dataset_id,
                                        args.provenance), indent=2))
    elif args.command == "validate-dataset":
        print(json.dumps(validate_frozen_dataset(args.dataset, args.dataset_id), indent=2))
    elif args.command == "validate-trial":
        print(json.dumps(validate_trial(args.trial, args.dataset,
                                        args.experiment_id), indent=2))
    elif args.command == "render-report": render_report(args.results, args.output)
    elif args.command == "select-d1":
        from .d1_selection import write_d1_selection
        print(json.dumps(write_d1_selection(load_config(args.config, D1Config)), indent=2))
    elif args.command == "diagnose-d1":
        from .d1_diagnostics import run_d1_diagnostics
        print(json.dumps(run_d1_diagnostics(
            load_config(args.config, D1Config), args.run_id,
            args.allow_gpu_diagnostic,
        ), indent=2))
    elif args.command == "time-d1":
        from .d1_timing import run_d1_timing
        print(json.dumps(run_d1_timing(
            load_config(args.config, D1Config), args.run_id,
            args.timing_seed, args.allow_gpu_benchmark,
        ), indent=2))
    elif args.command == "analyze-d1":
        from .d1_analysis import analyze_d1
        print(json.dumps(analyze_d1(load_config(args.config, D1Config)), indent=2))
    elif args.command == "validate-d1":
        from .d1_analysis import validate_d1
        print(json.dumps(validate_d1(load_config(args.config, D1Config)), indent=2))


if __name__ == "__main__": main()
