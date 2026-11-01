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


def test_ghz_circuit_and_population():
    from ghz_test import ghz_circuit, ghz_population
    qc = ghz_circuit(4)
    assert sum(1 for i in qc.data if i.operation.num_qubits == 2) == 3     # n - 1 two-qubit gates
    pop = ghz_population({"0000": 45, "1111": 45, "0101": 10}, 4)
    assert pop["P_ghz"] == pytest.approx(0.9) and pop["chance"] == pytest.approx(2 / 16)


def test_transpile_best_keeps_highest_esp_on_fake_backend():
    from core.ibm_runner import IBMRunner
    from ghz_test import ghz_circuit
    runner = IBMRunner(fake="FakeFez", min_qubits=5)
    isa, stats, _ = runner.transpile_best(ghz_circuit(5), tries=4)
    assert stats["two_qubit_gates"] == 4                                   # a chain fits without SWAPs
    assert 0 < stats["esp"] <= 1 and stats["esp"] == pytest.approx(stats["esp_range"][1])


def test_mitigation_options_are_accepted_by_the_sampler():
    from qiskit_ibm_runtime.options import SamplerOptions
    from core.ibm_runner import MITIGATIONS, IBMRunner, sampler_options
    assert sampler_options("none") == {}
    both = sampler_options("dd+twirl")
    assert both["dynamical_decoupling"]["enable"] and both["twirling"]["enable_gates"]
    assert "twirling" not in sampler_options("dd") and "dynamical_decoupling" not in sampler_options("twirl")
    for m in MITIGATIONS:
        SamplerOptions(**sampler_options(m))                              # valid for qiskit-ibm-runtime
    with pytest.raises(ValueError):
        IBMRunner(fake="FakeFez", mitigation="zne")


def test_size_sweep_graphs_are_nested_and_noiseless_sampling_works():
    from types import SimpleNamespace
    from size_sweep import build_instance, ideal_counts
    from run_ibm import qaoa_circuit
    args = SimpleNamespace(start_node=30, tolerance=2.0)
    e10, cost10, greedy10, model10 = build_instance(10, args)
    e15, *_ = build_instance(15, args)
    assert set(e10) <= set(e15)                                  # BFS pieces of the same graph
    assert greedy10 == -cost10 == 4
    counts = ideal_counts(qaoa_circuit(model10, [0.4], [3.4]), 500)
    assert sum(counts.values()) == 500 and all(len(k) == 10 for k in counts)
