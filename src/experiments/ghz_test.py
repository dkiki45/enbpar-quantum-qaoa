"""
GHZ noise curve on IBM Quantum hardware: how much signal survives as two-qubit gates grow.

A GHZ state of n qubits (H on qubit 0, then a chain of CNOTs 0-1, 1-2, ...) ideally gives only
"00...0" or "11...1", each half the time. Its circuit is the simplest one with n - 1 two-qubit gates,
so the measured fraction P(00...0) + P(11...1) is a direct measure of how much the hardware noise
destroys a circuit with that many two-qubit gates (chance level: 2 / 2^n).

All sizes run in ONE job (one queue wait). Each circuit is transpiled --tries times and the one with
the highest estimated success probability (ESP, from the backend calibration) is kept, so the curve
also shows how well the ESP predicts the measured result. The QAOA runs can then be placed on the
same axis (two-qubit gates) to see why 30 poles still keep signal and 156 do not.

Results: src/results/ibm/<name>/ghz_summary.json and ghz_counts.json

Usage (from the repo root):
    PYTHONPATH=src python -u src/experiments/ghz_test.py --name ghz_dry --fake FakeFez --dry-run
    PYTHONPATH=src python -u src/experiments/ghz_test.py --name ghz_kingston --backend ibm_kingston
    PYTHONPATH=src python -u src/experiments/ghz_test.py --name ghz_kingston_big --backend ibm_kingston --sizes 60,80,100,156
"""
import argparse
import json
import warnings

from qiskit import QuantumCircuit

import config
from core.ibm_runner import IBMRunner


def ghz_circuit(n):
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n - 1):
        qc.cx(i, i + 1)
    return qc


def ghz_population(counts, n):
    """Fraction of shots in the two ideal outcomes, 00...0 and 11...1."""
    total = sum(counts.values())
    p0 = counts.get("0" * n, 0) / total
    p1 = counts.get("1" * n, 0) / total
    return {"P_all0": p0, "P_all1": p1, "P_ghz": p0 + p1, "chance": 2 / 2 ** n}


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description="GHZ noise curve on IBM Quantum")
    ap.add_argument("--name", required=True, help="results folder under src/results/ibm")
    ap.add_argument("--sizes", default="5,10,15,20,30,50", help="numbers of qubits, comma-separated")
    ap.add_argument("--shots", type=int, default=4000)
    ap.add_argument("--backend", default=None)
    ap.add_argument("--fake", default=None, help="fake backend for --dry-run, e.g. FakeFez")
    ap.add_argument("--tries", type=int, default=10)
    ap.add_argument("--optimization-level", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.fake and not args.dry_run:
        raise SystemExit("--fake is only for --dry-run")

    sizes = [int(x) for x in args.sizes.split(",")]
    runner = IBMRunner(args.backend, min_qubits=max(sizes), optimization_level=args.optimization_level,
                       fake=args.fake)
    rows, circuits = [], []
    for n in sizes:
        isa, stats, t = runner.transpile_best(ghz_circuit(n), tries=args.tries)
        circuits.append(isa)
        rows.append({"n": n, "circuit": stats, "transpile_s": t})
        print(f"  n={n:3d}: 2-qubit gates {stats['two_qubit_gates']} (depth {stats['depth_2q']}), "
              f"ESP {stats['esp']:.3f}", flush=True)

    out = config.RESULTS_BASE_DIR / "ibm" / args.name
    out.mkdir(parents=True, exist_ok=True)
    summary = {"name": args.name, "backend": runner.name, "fake_backend": runner.is_fake, "shots": args.shots,
               "tries": args.tries, "optimization_level": args.optimization_level, "dry_run": args.dry_run,
               "sizes": rows}
    if not args.dry_run:
        counts, job = runner.run_many(circuits, args.shots)
        summary["job"] = job
        for row, c in zip(rows, counts):
            row.update(ghz_population(c, row["n"]))
        (out / "ghz_counts.json").write_text(json.dumps({str(r["n"]): c for r, c in zip(rows, counts)}),
                                             encoding="utf-8")
        print(f"\n## GHZ on {runner.name} ({args.shots} shots per size, QPU {job['qpu_seconds']} s)")
        print("| qubits | 2-qubit gates | ESP (predicted) | P(00..0) + P(11..1) measured | chance |")
        print("| --- | --- | --- | --- | --- |")
        for r in rows:
            print(f"| {r['n']} | {r['circuit']['two_qubit_gates']} | {r['circuit']['esp']:.3f} | "
                  f"{r['P_ghz']:.3f} | {r['chance']:.1e} |")
    (out / "ghz_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved {out / 'ghz_summary.json'}")
