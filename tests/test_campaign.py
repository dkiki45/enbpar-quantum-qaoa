import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "experiments"))
from run_campaign import interp_layer, initial_point, greedy_mis, tts99, optimize_exact  # noqa: E402
from abc_report import paired_ci  # noqa: E402

GRID = {"p1": {"best_betas": [0.4], "best_gammas": [3.4]},
        "p2": {"best_betas": [0.4, 0.2], "best_gammas": [3.4, 1.9]},
        "p3": {"best_betas": [0.4, 0.2, 3.0], "best_gammas": [3.4, 1.9, 2.2]}}


def test_interp_layer_keeps_ends_and_grows():
    # INTERP: p -> p+1 angles, first = (q/q)*a1, last = (q/q)*a_q
    assert interp_layer([1.0, 2.0]) == pytest.approx([1.0, 1.5, 2.0])
    assert len(interp_layer([0.4, 0.2, 3.0])) == 4


def test_initial_point_extensions_and_perturbation():
    pt, k = initial_point(GRID, 4, 99, "zeros", 0.0, 0.05, seed=1)
    assert k == 3 and pt == [0.4, 0.2, 3.0, 0.0, 3.4, 1.9, 2.2, 0.0]
    pt, k = initial_point(GRID, 4, 2, "zeros", 0.0, 0.05, seed=1)  # old-style init: layers 1-2 only
    assert k == 2 and pt == [0.4, 0.2, 0.0, 0.0, 3.4, 1.9, 0.0, 0.0]
    a, _ = initial_point(GRID, 4, 99, "zeros", 0.05, 0.05, seed=1)
    b, _ = initial_point(GRID, 4, 99, "zeros", 0.05, 0.05, seed=2)
    base = [0.4, 0.2, 3.0, 0.0, 3.4, 1.9, 2.2, 0.0]
    assert a != b and max(abs(x - y) for x, y in zip(a, base)) <= 0.05
    r, _ = initial_point(GRID, 5, 99, "perturb", 0.0, 0.05, seed=3)
    assert len(r) == 10 and all(abs(x) <= 0.05 for x in (r[3], r[4], r[8], r[9]))


def test_greedy_mis_path():
    assert greedy_mis(4, [(0, 1), (1, 2), (2, 3)]) == 2


def test_tts99():
    assert tts99(0.0) is None
    assert tts99(1.0) == 1.0
    assert tts99(0.5) == pytest.approx(6.644, abs=1e-3)       # log(0.01)/log(0.5)
    assert tts99(0.0069) == pytest.approx(665, abs=1)          # P(opt) of the main instance, p = 3


class _Quadratic:
    """Fake evaluator: E = sum((angles - 1)^2), minimum 0 at all angles = 1."""
    def __init__(self):
        self.calls = 0

    def energies(self, points):
        self.calls += 1
        (b, g), = points
        return [sum((x - 1.0) ** 2 for x in list(b) + list(g))]


def test_optimize_exact_finds_minimum_and_counts_evaluations():
    ev = _Quadratic()
    betas, gammas, n = optimize_exact(ev, [0.0, 0.0, 0.0, 0.0], p=2, maxiter=500)
    assert betas == pytest.approx([1.0, 1.0], abs=1e-3) and gammas == pytest.approx([1.0, 1.0], abs=1e-3)
    assert n == ev.calls and 0 < n <= 500


def test_paired_ci():
    m, lo, hi = paired_ci([1.0, 2.0, 3.0])
    assert m == pytest.approx(2.0)
    assert lo == pytest.approx(2.0 - 4.303 * 1.0 / 3 ** 0.5, abs=1e-3)   # t(0.975, 2) = 4.303
    assert hi == pytest.approx(2.0 + 4.303 * 1.0 / 3 ** 0.5, abs=1e-3)
    assert paired_ci([5.0])[1:] == (None, None)
