import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "experiments"))
from run_warm_start import build_warm_start_point, build_ext_warm_start_point  # noqa: E402

EXT = {"n_qubits": 30,
       "p1": {"best_betas": [0.1], "best_gammas": [1.0]},
       "p2": {"best_betas": [0.1, 0.2], "best_gammas": [1.0, 2.0]},
       "p3": {"best_betas": [0.1, 0.2, 0.3], "best_gammas": [1.0, 2.0, 3.0]}}


def test_ext_uses_deepest_layer_and_pads_with_zeros():
    assert build_ext_warm_start_point(3, EXT) == [0.1, 0.2, 0.3, 1.0, 2.0, 3.0]
    assert build_ext_warm_start_point(5, EXT) == [0.1, 0.2, 0.3, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0]
    assert build_ext_warm_start_point(2, EXT) == [0.1, 0.2, 1.0, 2.0]


def test_v3_point_unchanged():
    v3 = {"p1_best_beta": 0.1, "p1_best_gamma": 1.0, "p2_best_beta": 0.2, "p2_best_gamma": 2.0}
    assert build_warm_start_point(4, v3) == [0.1, 0.2, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0]
