import numpy as np
from precision_md.trajectories import Checkpoint, restore_checkpoint, serialize_checkpoint


def test_checkpoint_roundtrip(tmp_path):
    cp = Checkpoint(3, np.ones((2,3)), np.zeros((2,3)), np.array([12.,1.]), 5., {"seed":7})
    assert serialize_checkpoint(cp, tmp_path / "c.npz") >= 0
    got, elapsed = restore_checkpoint(tmp_path / "c.npz")
    assert elapsed >= 0 and got.metadata == cp.metadata
    np.testing.assert_array_equal(got.positions, cp.positions)
