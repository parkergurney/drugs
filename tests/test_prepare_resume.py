from pathlib import Path


def test_candidate_checkpoint_uses_atomic_same_directory_replacement(tmp_path):
    score_path = tmp_path / "candidate_scores.parquet"
    temporary = score_path.with_suffix(".tmp.parquet")

    assert temporary.parent == score_path.parent
    assert temporary != score_path
    assert temporary == Path(tmp_path, "candidate_scores.tmp.parquet")
