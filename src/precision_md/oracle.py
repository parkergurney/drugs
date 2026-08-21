from __future__ import annotations
import numpy as np


def oracle_cost(row):
    cost = row["fast_seconds"] + row["audit_seconds"] + row.get("checkpoint_seconds", 0.0)
    if row["unsafe"]:
        cost += row["restore_seconds"] + row["fp32_seconds"]
    return cost


def oracle_speedup(rows):
    baseline = sum(r["fp32_seconds"] for r in rows)
    adaptive = sum(oracle_cost(r) for r in rows)
    return baseline / adaptive if adaptive > 0 else np.nan


def checkpoint_bootstrap(rows, statistic=oracle_speedup, iterations=2000, seed=20260819):
    if not rows: return (np.nan, np.nan)
    groups = {}
    for row in rows: groups.setdefault(row["checkpoint_id"], []).append(row)
    ids, rng = list(groups), np.random.default_rng(seed)
    samples = []
    for _ in range(iterations):
        draw = rng.choice(ids, len(ids), replace=True)
        samples.append(statistic([r for cid in draw for r in groups[cid]]))
    return tuple(np.quantile(samples, [0.025, 0.975]))


def fit_threshold(train_rows, predictor, minimum_recall=0.8):
    """Choose threshold with best precision subject to recall, predicting high values unsafe."""
    values = sorted({r[predictor] for r in train_rows})
    candidates = []
    for threshold in values:
        predicted = [r[predictor] >= threshold for r in train_rows]
        tp = sum(p and r["unsafe"] for p, r in zip(predicted, train_rows))
        fp = sum(p and not r["unsafe"] for p, r in zip(predicted, train_rows))
        positives = sum(r["unsafe"] for r in train_rows)
        recall = tp / positives if positives else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        if recall >= minimum_recall: candidates.append((precision, threshold, recall))
    if not candidates: return None
    precision, threshold, recall = max(candidates)
    return {"threshold": threshold, "precision": precision, "recall": recall}


def evaluate_threshold(rows, predictor, threshold):
    pred = [r[predictor] >= threshold for r in rows]
    tp = sum(p and r["unsafe"] for p, r in zip(pred, rows)); fp = sum(p and not r["unsafe"] for p, r in zip(pred, rows))
    fn = sum(not p and r["unsafe"] for p, r in zip(pred, rows))
    return {"precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0}
