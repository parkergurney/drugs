import numpy as np
import pytest
from precision_md.trajectories import pair_distance_rmsd, unsafe_label, wrapped_angle_difference


def endpoint(**updates):
    base = {"conformer":"trans", "dihedral_deg":179., "positions":np.array([[0.,0,0],[1.,0,0]]),
            "initial_energy_ev":0., "endpoint_energy_ev":0.}
    return base | updates


def test_wrapped_angle():
    assert wrapped_angle_difference(179, -179) == pytest.approx(2)


@pytest.mark.parametrize("fast,reason", [
    (endpoint(conformer="gauche+"), "conformer"),
    (endpoint(dihedral_deg=170), "dihedral"),
    (endpoint(positions=np.array([[0.,0,0],[1.1,0,0]])), "pair_distance_rmsd"),
    (endpoint(endpoint_energy_ev=.01), "delta_energy"),
    (endpoint(endpoint_energy_ev=np.nan), "nonfinite"),
])
def test_each_unsafe_condition(fast, reason):
    unsafe, reasons = unsafe_label(endpoint(), fast)
    assert unsafe and reason in reasons


def test_identical_is_safe():
    assert unsafe_label(endpoint(), endpoint()) == (False, [])
