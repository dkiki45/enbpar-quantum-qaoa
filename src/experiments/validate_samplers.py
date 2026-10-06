"""
Sampler validation: do the Aer method (MPS vs statevector) or the transpile cache
of FastAerSampler change the QAOA results?

Variants (same circuit, same seeds):
  R = Qiskit StatevectorSampler (pure Qiskit reference: no Aer, no cache; transpiled every call)
  A = Aer statevector, circuit transpiled again on EVERY call (no cache)
  B = FastAerSampler + statevector (transpile cache)
  C = FastAerSampler + matrix_product_state (current setup)

Test 1 (fixed): same random angles in every variant, no optimizer. Each energy is compared
               with the exact expectation (computed from the full statevector, no shots).
               A correct simulator stays within shot noise (|z| < ~3).
Test 2 (opt):  full COBYLA optimization per variant and seed, same settings as run_qaoa.

Existing code is not modified. Results: src/results/sampler_validation/n<N>_p<p>/

Usage (from the repo root):
    PYTHONPATH=src python -u src/experiments/validate_samplers.py --nodes 20 --test both
    PYTHONPATH=src python -u src/experiments/validate_samplers.py --nodes 30 --variants A,B,C --test fixed
    PYTHONPATH=src python -u src/experiments/validate_samplers.py --nodes 20 --report
"""
import argparse
import io
import contextlib
import json
import statistics
import time
import warnings
from pathlib import Path

import numpy as np
from qiskit import transpile
from qiskit.circuit.library import QAOAAnsatz
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Statevector
from qiskit_aer.primitives import SamplerV2
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit_algorithms.utils import algorithm_globals

from core.graph_builder import build_graph_from_csv
from core.qubo_formalization import build_mis_qubo, qubo_to_ising, classical_cost
from core.qaoa_solver import FastAerSampler
from core.solution_decoder import decode_distribution, best_feasible_or_none, feasible_probability, integer_to_bits
from core.classical_baseline import solve_exact_ilp
from config import CSV_PATH, SHOTS_PER_EVAL, RESULTS_BASE_DIR

BASIS = ['rx', 'ry', 'rz', 'cx', 'h', 'x', 'y', 'z', 'id']  # same basis as FastAerSampler
MAX_ITER = 300

VARIANTS = {
    "R": "Qiskit StatevectorSampler (reference)",
    "A": "Aer statevector, transpile every call",
    "B": "FastAerSampler + statevector (cache)",
    "C": "FastAerSampler + MPS (current)",
}


def _transpile_pubs(pubs):
    """Transpile every pub's circuit from scratch (no cache)."""
    out = []
    for pub in pubs:
        circ = pub[0] if isinstance(pub, tuple) else pub.circuit
        t = transpile(circ, optimization_level=1, basis_gates=BASIS)
        if isinstance(pub, tuple):
            out.append((t,) + tuple(pub[1:]))
        else:
            out.append((t, pub.parameter_values, pub.shots) if pub.shots else (t, pub.parameter_values))
    return out


class TranspileEveryCallSampler(SamplerV2):
    """Aer SamplerV2 that transpiles the circuit on every run() call: no cache at all."""

    def run(self, pubs, **kwargs):
        return super().run(_transpile_pubs(pubs), **kwargs)


class QiskitReferenceSampler(StatevectorSampler):
    """Pure Qiskit statevector simulation (no Aer, no cache). The circuit is transpiled to
    basic gates first: on the raw QAOA ansatz Qiskit builds dense evolution matrices,
    which takes seconds at 10 qubits and is infeasible at 20."""

    def run(self, pubs, **kwargs):
        return super().run(_transpile_pubs(pubs), **kwargs)


def make_sampler(key, shots, seed):
    if key == "R":
        return QiskitReferenceSampler(default_shots=shots, seed=seed)
    if key == "A":
        return TranspileEveryCallSampler(default_shots=shots, seed=seed,
                                         options={"backend_options": {"method": "statevector"}})
    if key == "B":
        return FastAerSampler(default_shots=shots, seed=seed,
                              options={"backend_options": {"method": "statevector"}})
    if key == "C":
        return FastAerSampler(default_shots=shots, seed=seed,
                              options={"backend_options": {"method": "matrix_product_state"}})
    raise ValueError(key)


def cost_of_index(idx, n, edges):
    return classical_cost(integer_to_bits(idx, n), edges)


# ------------------------------------------------------------------ Test 1
def test_fixed(model, n, edges, p, variants, n_points, shots, out_dir):
    op = model.to_sparse_pauli_op()
    ansatz = QAOAAnsatz(op, reps=p)
    measured = ansatz.copy(); measured.measure_all()
    rng = np.random.default_rng(12345)
    points = [rng.uniform(0, np.pi, size=ansatz.num_parameters).tolist() for _ in range(n_points)]

    # Exact expectation and shot-noise std from the full statevector (no sampling)
    t_ans = transpile(ansatz, optimization_level=1, basis_gates=BASIS)
    exact = []
    cost_cache = {}
    for pt in points:
        probs = Statevector(t_ans.assign_parameters(pt)).probabilities()
        nz = np.nonzero(probs > 1e-15)[0]
        costs = np.array([cost_cache.setdefault(i, cost_of_index(i, n, edges)) for i in nz])
        pr = probs[nz]
        mean = float(np.dot(pr, costs))
        var = float(np.dot(pr, (costs - mean) ** 2))
        exact.append({"mean": mean, "se": (var / shots) ** 0.5})

    rows = []
    for key in variants:
        sampler = make_sampler(key, shots, seed=7)
        for k, pt in enumerate(points):
            t0 = time.time()
            res = sampler.run([(measured, pt)]).result()[0]
            dt = time.time() - t0
            counts = res.data.meas.get_counts()
            tot = sum(counts.values())
            e = sum(classical_cost(integer_to_bits(int(b, 2), n), edges) * c for b, c in counts.items()) / tot
            z = (e - exact[k]["mean"]) / exact[k]["se"] if exact[k]["se"] > 0 else 0.0
            rows.append({"variant": key, "point": k, "energy": e, "exact": exact[k]["mean"],
                         "z": z, "seconds": dt})
            print(f"  [{key}] point {k}: E={e:9.4f} exact={exact[k]['mean']:9.4f} z={z:+6.2f} ({dt:.2f}s)", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "fixed.json").write_text(json.dumps({"shots": shots, "points": points, "rows": rows}, indent=2))
    return rows


def report_fixed(out_dir, variants):
    f = out_dir / "fixed.json"
    if not f.exists():
        print("\n(Test 1: no results yet)")
        return
    rows = json.loads(f.read_text())["rows"]
    print(f"\nTEST 1 - fixed angles vs exact expectation (shots={json.loads(f.read_text())['shots']})")
    hdr = f"{'var':3s} {'description':40s} {'max|z|':>7s} {'mean|z|':>8s} {'max|E-exact|':>13s} {'s/eval':>7s}  verdict"
    print(hdr + "\n" + "-" * len(hdr))
    for key in variants:
        rs = [r for r in rows if r["variant"] == key]
        if not rs:
            continue
        mz = max(abs(r["z"]) for r in rs)
        verdict = "OK (within shot noise)" if mz < 4 else "DIFFERENT - investigate"
        print(f"{key:3s} {VARIANTS[key]:40s} {mz:7.2f} {statistics.mean(abs(r['z']) for r in rs):8.2f} "
              f"{max(abs(r['energy'] - r['exact']) for r in rs):13.4f} {statistics.mean(r['seconds'] for r in rs):7.2f}  {verdict}")


# ------------------------------------------------------------------ Test 2
def run_opt(model, nodes, edges, exact_cost, p, key, seed, shots, out_dir):
    algorithm_globals.random_seed = seed
    history = []
    sampler = make_sampler(key, shots, seed)
    qaoa = QAOA(sampler=sampler, optimizer=COBYLA(maxiter=MAX_ITER, tol=1e-6), reps=p,
                callback=lambda i, prm, mean, md: history.append(float(np.real(mean))))
    t0 = time.time()
    raw = qaoa.compute_minimum_eigenvalue(model.to_sparse_pauli_op())
    dt = time.time() - t0
    dist = {int(k, 2) if isinstance(k, str) else int(k): float(np.real(v)) for k, v in dict(raw.eigenstate).items()}
    cands = decode_distribution(dist, len(nodes), edges, nodes=nodes)
    best = best_feasible_or_none(cands)
    s = {"variant": key, "seed": seed, "reps": p,
         "expectation_qubo": float(np.real(raw.eigenvalue)) + model.offset,
         "best_cost": best.cost if best else None, "exact_cost": exact_cost,
         "hit_optimum": bool(best and best.cost == exact_cost),
         "feasible_probability": feasible_probability(cands),
         "n_evaluations": len(history), "runtime_s": dt,
         "optimal_parameters": [float(v) for v in raw.optimal_point]}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(s, indent=2))
    return s


def report_opt(base, variants, seeds):
    print("\nTEST 2 - full COBYLA optimization")
    hdr = f"{'var':3s} {'description':40s} {'n':>2s} {'E_qubo mean':>11s} {'std':>6s} {'optimum':>8s} {'P(feas)':>8s} {'evals':>6s} {'min/run':>7s}"
    print(hdr + "\n" + "-" * len(hdr))
    for key in variants:
        runs = [json.loads((base / key / f"seed{s}" / "summary.json").read_text())
                for s in seeds if (base / key / f"seed{s}" / "summary.json").exists()]
        if not runs:
            print(f"{key:3s} {VARIANTS[key]:40s}  (no results yet)")
            continue
        e = [r["expectation_qubo"] for r in runs]
        print(f"{key:3s} {VARIANTS[key]:40s} {len(runs):2d} {statistics.mean(e):11.3f} "
              f"{(statistics.stdev(e) if len(e) > 1 else 0):6.3f} {sum(r['hit_optimum'] for r in runs):>3d}/{len(runs):<4d} "
              f"{statistics.mean(r['feasible_probability'] for r in runs):8.3f} "
              f"{statistics.mean(r['n_evaluations'] for r in runs):6.0f} {statistics.mean(r['runtime_s'] for r in runs) / 60:7.1f}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description="Validate QAOA samplers (method / transpile cache)")
    ap.add_argument("--nodes", type=int, default=20, help="number of qubits (default 20)")
    ap.add_argument("--depth", type=int, default=1, help="QAOA depth p (default 1)")
    ap.add_argument("--variants", default="R,A,B,C", help="subset, e.g. A,B,C")
    ap.add_argument("--test", choices=["fixed", "opt", "both"], default="both")
    ap.add_argument("--points", type=int, default=5, help="angle sets for test 1 (default 5)")
    ap.add_argument("--fixed-shots", type=int, default=8192, help="shots for test 1 (default 8192)")
    ap.add_argument("--seeds", type=int, default=3, help="seeds for test 2 (default 3)")
    ap.add_argument("--report", action="store_true", help="only print tables from saved results")
    args = ap.parse_args()

    keys = [k.strip().upper() for k in args.variants.split(",") if k.strip()]
    bad = [k for k in keys if k not in VARIANTS]
    if bad:
        raise SystemExit(f"Unknown variant(s) {bad}. Options: {list(VARIANTS)}")
    base = RESULTS_BASE_DIR / "sampler_validation" / f"n{args.nodes}_p{args.depth}"
    seeds = list(range(1, args.seeds + 1))

    if not args.report:
        nodes, edges = build_graph_from_csv(CSV_PATH, args.nodes)
        model = qubo_to_ising(len(nodes), *build_mis_qubo(len(nodes), edges))
        _, exact_cost = solve_exact_ilp(len(nodes), edges)
        print(f"\n{'=' * 60}\nSAMPLER VALIDATION: {len(nodes)} qubits, {len(edges)} edges, p={args.depth}, variants {keys}\n{'=' * 60}")

        if args.test in ("fixed", "both"):
            print(f"\n>> Test 1: fixed angles ({args.points} points, {args.fixed_shots} shots)")
            test_fixed(model, len(nodes), edges, args.depth, keys, args.points, args.fixed_shots, base)

        if args.test in ("opt", "both"):
            print(f"\n>> Test 2: COBYLA optimization ({len(seeds)} seeds, {SHOTS_PER_EVAL} shots)")
            for key in keys:
                for s in seeds:
                    d = base / key / f"seed{s}"
                    if (d / "summary.json").exists():
                        print(f"  [{key}] seed {s} already done, skipping")
                        continue
                    with contextlib.redirect_stdout(io.StringIO()):
                        r = run_opt(model, nodes, edges, exact_cost, args.depth, key, s, SHOTS_PER_EVAL, d)
                    print(f"  [{key}] seed {s} | E_qubo {r['expectation_qubo']:8.3f} | best {r['best_cost']} "
                          f"| evals {r['n_evaluations']} | {r['runtime_s'] / 60:.1f} min", flush=True)

    report_fixed(base, keys)
    report_opt(base, keys, seeds)
