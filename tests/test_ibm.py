import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "experiments"))
from run_ibm import graph_from_components, sample_metrics, random_baseline  # noqa: E402

CSV = "src/data/paranainterativo.csv"


def test_components_graph_has_exact_size_and_valid_edges():
    nodes, edges, sizes = graph_from_components(CSV, 156, 2.0)
    assert len(nodes) == 156 and sum(sizes) == 156
    assert sizes[0] == 125                      # largest connected group at 40 m
    assert all(0 <= u < v < 156 for u, v in edges)
    assert sizes == [125, 12, 9, 8, 2]          # groups added whole while they fit


def test_sample_metrics_path_graph():
    # path 0-1-2: MIS = {0, 2}, cost -2. Qiskit keys: highest qubit on the left
    edges = [(0, 1), (1, 2)]
    counts = {"101": 6, "001": 2, "011": 2}     # optimum, valid size 1, invalid
    m = sample_metrics(counts, 3, edges, -2.0)
    assert m["P_optimum"] == pytest.approx(0.6)
    assert m["P_feasible"] == pytest.approx(0.8)
    assert m["approx_ratio_all"] == pytest.approx((0.6 * 2 + 0.2 * 1) / 2)
    assert m["hit_optimum"] and m["best_size"] == 2


def test_random_baseline_chance_level():
    rb = random_baseline(3, [(0, 1), (1, 2)], -2.0, shots=200_000, seed=1)
    assert rb["P_feasible"] == pytest.approx(5 / 8, abs=0.01)   # 5 of the 8 bitstrings are independent sets
    assert rb["P_optimum"] == pytest.approx(1 / 8, abs=0.01)


def test_repair_makes_independent_set_and_completes_it():
    from collections import defaultdict
    import numpy as np
    from ibm_analysis import repair, bits_from_key
    adj = defaultdict(set)
    for u, v in [(0, 1), (1, 2), (2, 3)]:                 # path 0-1-2-3, MIS size 2
        adj[u].add(v)
        adj[v].add(u)
    assert repair(np.array([1, 1, 1, 1], dtype=np.int8), adj) == 2   # all chosen -> repaired to a MIS
    assert repair(np.array([0, 0, 0, 0], dtype=np.int8), adj) == 2   # empty -> greedy completion
    assert list(bits_from_key("0011", 4)) == [1, 1, 0, 0]            # Qiskit order: qubit 0 on the right
