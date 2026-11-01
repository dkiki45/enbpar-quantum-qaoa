"""
Size sweep on IBM Quantum hardware: the same fixed-angle QAOA on 10, 15, 20, 25 and 30 poles, to see
at what size (how many two-qubit gates) the signal fades.

Every size is a BFS-connected piece of the same streetlight graph (start --start-node, conflict distance
20 m x --tolerance), so the smaller graphs are pieces of the 30-pole instance. All sizes use the SAME
angles (by default the best p = 3 grid angles of the 30-pole simulation): fixed angles carry over between
similar graphs, and the noiseless reference below checks that they still work on each size.

For each size:
  * ILP optimum and greedy MIS (classical references);
  * circuit transpiled --tries times, the one with the highest ESP kept;
  * noiseless reference (sizes up to --ideal-max): the logical circuit sampled with the same number of
    shots on Aer's statevector simulator, WITHOUT any noise model. It gives the share of valid samples
    the hardware would reach without noise, so "retained" = hardware / noiseless.
All circuits go to the QPU in ONE job (one queue wait, one calibration for every size).

Each size is saved like a run_ibm.py run (src/results/ibm/<name>_n<size>/summary.json + counts.json),
so ibm_analysis.py works on them; the sweep table goes to src/results/ibm/<name>/sweep.json.

Usage (from the repo root):
    PYTHONPATH=src python -u src/experiments/size_sweep.py --name sweep_dry --fake FakeFez --dry-run
    PYTHONPATH=src python -u src/experiments/size_sweep.py --name sweep_p3_kingston --backend ibm_kingston --tries 20
    PYTHONPATH=src python -u src/experiments/size_sweep.py --name sweep_p2_kingston --p 2 --backend ibm_kingston --tries 20
"""
import argparse
import json
import time
import warnings

import config
from core.graph_builder import build_graph_from_csv
from core.qubo_formalization import build_mis_qubo, qubo_to_ising
from core.classical_baseline import solve_exact_ilp
from core.ibm_runner import MITIGATIONS, IBMRunner, circuit_stats, estimated_success_probability, fetch_job
from run_campaign import greedy_mis
from run_ibm import load_angles, qaoa_circuit, sample_metrics, random_baseline


def ideal_counts(circuit, shots, seed=0):
    """Samples the logical circuit without noise (Aer statevector)."""
    from qiskit import transpile
    from qiskit_aer import AerSimulator
    sim = AerSimulator(method="statevector")
    qc = circuit.remove_final_measurements(inplace=False)
    qc.measure_all()
    return sim.run(transpile(qc, sim), shots=shots, seed_simulator=seed).result().get_counts()


def build_instance(n, args):
    nodes, edges = build_graph_from_csv(config.CSV_PATH, n, start_node=args.start_node,
                                        tolerance_factor=args.tolerance)
    if len(nodes) < n:
        raise SystemExit(f"Only {len(nodes)} connected poles from node {args.start_node}")
    _, exact_cost = solve_exact_ilp(n, edges)
    model = qubo_to_ising(n, *build_mis_qubo(n, edges))
    return edges, exact_cost, greedy_mis(n, edges), model


def pct(x):
    return "-" if x is None else f"{100 * x:.1f}%"


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description="QAOA size sweep on IBM Quantum (one job for every size)")
    ap.add_argument("--name", required=True, help="prefix of the results folders under src/results/ibm")
    ap.add_argument("--sizes", default="10,15,20,25,30", help="numbers of poles, comma-separated")
    ap.add_argument("--start-node", type=int, default=config.START_NODE)
    ap.add_argument("--tolerance", type=float, default=2.0, help="conflict distance = 20 m x tolerance")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--angles", choices=["grid", "campaign", "file"], default="grid")
    ap.add_argument("--grid", default="n30_statevector_GPU", help="grid_search_ext folder")
    ap.add_argument("--campaign", default="abc_C")
    ap.add_argument("--seed", type=int, default=1, help="campaign run to take the angles from")
    ap.add_argument("--angles-file", default=None)
    ap.add_argument("--shots", type=int, default=10_000)
    ap.add_argument("--backend", default=None)
    ap.add_argument("--fake", default=None, help="fake backend for --dry-run, e.g. FakeFez")
    ap.add_argument("--optimization-level", type=int, default=3)
    ap.add_argument("--tries", type=int, default=20)
    ap.add_argument("--mitigation", choices=MITIGATIONS, default="none")
    ap.add_argument("--ideal-max", type=int, default=26,
                    help="largest size simulated without noise (memory grows 2^n: 26 qubits = 1 GB)")
    ap.add_argument("--dry-run", action="store_true", help="transpile and simulate only (no QPU time)")
    ap.add_argument("--job-id", default=None, help="retrieve a job already sent with the same --sizes")
    args = ap.parse_args()
    if args.fake and not args.dry_run:
        raise SystemExit("--fake is only for --dry-run")

    sizes = [int(x) for x in args.sizes.split(",")]
    betas, gammas, source = load_angles(args)
    if len(betas) != args.p:
        raise SystemExit(f"angles have {len(betas)} layers, --p is {args.p}")
    fetched = None
    if args.job_id:
        fetched = fetch_job(args.job_id)
        args.backend = fetched[1]["backend"]
        if len(fetched[0]) != len(sizes):
            raise SystemExit(f"job has {len(fetched[0])} circuits but --sizes has {len(sizes)}")
    runner = IBMRunner(args.backend, min_qubits=max(sizes), optimization_level=args.optimization_level,
                       fake=args.fake, mitigation=args.mitigation)
    print(f"Sweep {args.name} | sizes {sizes} | p = {args.p}, angles {source} | backend {runner.name}"
          f"{' (fake, dry run)' if runner.is_fake else ''} | mitigation {args.mitigation}", flush=True)

    rows, circuits = [], []
    for i, n in enumerate(sizes):
        edges, exact_cost, greedy, model = build_instance(n, args)
        circ = qaoa_circuit(model, betas, gammas)
        if fetched and fetched[2][i] is not None:  # stats of the circuit that actually ran
            isa = fetched[2][i]
            stats = circuit_stats(isa) | {"esp": estimated_success_probability(isa, runner.backend.target),
                                          "from_job": True}
        else:
            isa, stats, _ = runner.transpile_best(circ, tries=args.tries)
        ideal = None
        if n <= args.ideal_max:
            t0 = time.time()
            ideal = sample_metrics(ideal_counts(circ, args.shots), n, edges, exact_cost)
            ideal["seconds"] = time.time() - t0
        circuits.append(isa)
        rows.append({"n": n, "edges": edges, "exact_cost": exact_cost, "greedy": greedy, "circuit": stats,
                     "ideal": ideal})
        print(f"  n={n:2d}: {len(edges)} edges, MIS {-exact_cost:.0f} | 2-qubit gates {stats['two_qubit_gates']}, "
              f"ESP {stats['esp']:.3f} | noiseless: valid {pct(ideal and ideal['P_feasible'])}, "
              f"optimum {pct(ideal and ideal['P_optimum'])}", flush=True)

    counts_list, job = None, None
    if fetched:
        counts_list, job = fetched[0], fetched[1]
    elif not args.dry_run:
        counts_list, job = runner.run_many(circuits, args.shots)

    table = []
    for i, r in enumerate(rows):
        n, edges, exact_cost = r["n"], r["edges"], r["exact_cost"]
        summary = {
            "name": f"{args.name}_n{n}", "sweep": args.name, "n_nodes": n, "n_edges": len(edges),
            "graph": "connected", "group_sizes": [n], "tolerance": args.tolerance, "start_node": args.start_node,
            "exact_cost": exact_cost, "mis_size": int(-exact_cost), "greedy_size": r["greedy"],
            "p": args.p, "angles_source": source, "betas": betas, "gammas": gammas,
            "backend": runner.name, "fake_backend": runner.is_fake, "optimization_level": args.optimization_level,
            "circuit": r["circuit"], "mitigation": args.mitigation, "dry_run": args.dry_run,
            "ideal": r["ideal"], "random_baseline": random_baseline(n, edges, exact_cost, args.shots),
        }
        row = {"n": n, "edges": len(edges), "mis": int(-exact_cost), "two_qubit_gates": r["circuit"]["two_qubit_gates"],
               "esp": r["circuit"]["esp"], "ideal_valid": r["ideal"] and r["ideal"]["P_feasible"],
               "ideal_optimum": r["ideal"] and r["ideal"]["P_optimum"]}
        if counts_list is not None:
            counts = counts_list[i]
            summary["job"] = job
            summary.update(sample_metrics(counts, n, edges, exact_cost))
            row |= {"hw_valid": summary["P_feasible"], "hw_optimum": summary["P_optimum"],
                    "hw_optimum_samples": round(summary["P_optimum"] * summary["shots"]),
                    "retained": summary["P_feasible"] / row["ideal_valid"] if row["ideal_valid"] else None}
            out = config.RESULTS_BASE_DIR / "ibm" / summary["name"]
            out.mkdir(parents=True, exist_ok=True)
            (out / "counts.json").write_text(json.dumps(counts), encoding="utf-8")
            (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        table.append(row)

    out = config.RESULTS_BASE_DIR / "ibm" / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "sweep.json").write_text(json.dumps({"name": args.name, "backend": runner.name, "p": args.p,
                                                "angles_source": source, "shots": args.shots,
                                                "mitigation": args.mitigation, "job": job, "rows": table},
                                               indent=2), encoding="utf-8")
    print(f"\n## Size sweep on {runner.name}, p = {args.p}"
          + (f" (QPU {job['qpu_seconds']} s)" if job else " (dry run: no hardware columns)"))
    print("| poles | edges | MIS | 2-qubit gates | ESP | valid: noiseless | valid: hardware | retained "
          "| optimum: noiseless | optimum samples: hardware |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for t in table:
        ret = t.get("retained")
        print(f"| {t['n']} | {t['edges']} | {t['mis']} | {t['two_qubit_gates']} | {t['esp']:.3f} "
              f"| {pct(t['ideal_valid'])} | {pct(t.get('hw_valid'))} | {'-' if ret is None else f'{100 * ret:.0f}%'} "
              f"| {pct(t['ideal_optimum'])} | {t.get('hw_optimum_samples', '-')} |")
    print(f"Saved {out / 'sweep.json'}")
    if counts_list is not None:
        names = ",".join(f"{args.name}_n{n}" for n in sizes)
        print(f"Same-marginals baseline: PYTHONPATH=src python src/experiments/ibm_analysis.py --names {names}")
