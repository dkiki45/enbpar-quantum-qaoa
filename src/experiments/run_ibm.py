"""
QAOA for the streetlight MIS on IBM Quantum hardware, with fixed angles (no optimization on the QPU).

One run = one graph + one set of angles:
  1. Graph: --nodes streetlights at the conflict distance 20 m x --tolerance.
       --graph connected   BFS-connected subgraph from --start-node (same as the campaign; up to
                           125 poles at 40 m, the largest connected group of the CSV)
       --graph components  the largest connected groups added whole until --nodes is reached
                           (156 poles at 40 m = groups of 125 + 12 + 9 + 8 + 2)
  2. Angles (p layers), from the GPU simulations:
       --angles grid       best angles of the extended grid (grid_search_ext summary.json)
       --angles campaign   final angles of one campaign run (e.g. abc_C, optimized with exact energy)
       --angles file       a JSON file {"betas": [...], "gammas": [...]}
  3. Circuit: QAOAAnsatz of the MIS QUBO with those angles, Z measurement of every qubit,
     transpiled for the backend (depth and two-qubit gates are saved).
  4. --dry-run: stop here (no QPU time). Otherwise sample --shots on the QPU.
  5. Metrics, as in the campaign: P(optimum), P(feasible), mean valid size, approximation ratios
     and TTS99, against the ILP optimum; plus the greedy MIS and a random-sampling baseline
     (uniform random bitstrings, same number of shots) as the chance level.

Results: src/results/ibm/<name>/summary.json and counts.json

Usage (from the repo root; needs `pip install qiskit-ibm-runtime` and the saved IBM key):
    PYTHONPATH=src python -u src/experiments/run_ibm.py --name dry30 --nodes 30 --fake FakeFez --dry-run
    PYTHONPATH=src python -u src/experiments/run_ibm.py --name dry156 --nodes 156 --graph components --fake FakeFez --dry-run
    PYTHONPATH=src python -u src/experiments/run_ibm.py --name n30_grid --nodes 30 --angles grid --backend ibm_fez
    PYTHONPATH=src python -u src/experiments/run_ibm.py --name n30_abcC --nodes 30 --angles campaign --campaign abc_C --backend ibm_fez
    PYTHONPATH=src python -u src/experiments/run_ibm.py --name n156_grid --nodes 156 --graph components --angles grid --backend ibm_fez
"""
import argparse
import json
import time
import warnings
from collections import defaultdict, deque

import numpy as np
import pandas as pd
from qiskit.circuit.library import QAOAAnsatz

import config
from core.graph_builder import build_graph_from_csv, haversine_m
from core.qubo_formalization import build_mis_qubo, qubo_to_ising
from core.classical_baseline import solve_exact_ilp
from core.solution_decoder import decode_distribution, best_feasible_or_none, feasible_probability
from core.ibm_runner import IBMRunner, circuit_stats
from run_campaign import greedy_mis, tts99


# ---------- graph ----------

def graph_from_components(path, n_nodes, tolerance, radius_m=20.0):
    """The largest connected groups of the whole CSV, added whole (largest first) while they fit,
    until exactly n_nodes poles; if no group fits the remainder, a BFS piece of the next one."""
    df = pd.read_csv(path).dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
    if "id" not in df.columns:
        df["id"] = df.index.astype(str)
    xy = df[["latitude", "longitude"]].to_numpy()
    if n_nodes > len(xy):
        raise SystemExit(f"The CSV has only {len(xy)} poles")
    thr = radius_m * tolerance
    adj = defaultdict(set)
    for i in range(len(xy)):
        for j in range(i + 1, len(xy)):
            if haversine_m(*xy[i], *xy[j]) < thr:
                adj[i].add(j)
                adj[j].add(i)
    seen, comps = set(), []
    for v in range(len(xy)):
        if v in seen:
            continue
        order, queue = [], deque([v])
        seen.add(v)
        while queue:  # BFS order, so any prefix is connected
            u = queue.popleft()
            order.append(u)
            for w in sorted(adj[u]):
                if w not in seen:
                    seen.add(w)
                    queue.append(w)
        comps.append(order)
    comps.sort(key=len, reverse=True)
    chosen, sizes = [], []
    for c in comps:
        if len(chosen) + len(c) <= n_nodes:
            chosen += c
            sizes.append(len(c))
        if len(chosen) == n_nodes:
            break
    if len(chosen) < n_nodes:  # fill with a connected piece of the largest group not used
        rest = next(c for c in comps if c[0] not in set(chosen))
        piece = rest[:n_nodes - len(chosen)]
        chosen += piece
        sizes.append(len(piece))
    index = {old: new for new, old in enumerate(chosen)}
    edges = sorted({(min(index[u], index[w]), max(index[u], index[w]))
                    for u in chosen for w in adj[u] if w in index})
    nodes = df.loc[chosen, ["id", "latitude", "longitude"]].to_dict("records")
    return nodes, edges, sizes


# ---------- angles ----------

def load_angles(args):
    if args.angles == "file":
        d = json.loads(open(args.angles_file, encoding="utf-8").read())
        return list(d["betas"]), list(d["gammas"]), f"file:{args.angles_file}"
    if args.angles == "campaign":
        f = config.RESULTS_BASE_DIR / "campaign" / args.campaign / f"p{args.p}_seed{args.seed}" / "summary.json"
        d = json.loads(f.read_text(encoding="utf-8"))
        return list(d["final_betas"]), list(d["final_gammas"]), f"campaign:{args.campaign}/p{args.p}_seed{args.seed}"
    f = config.RESULTS_BASE_DIR / "grid_search_ext" / args.grid / "summary.json"
    d = json.loads(f.read_text(encoding="utf-8"))
    layer = d.get(f"p{args.p}")
    if layer is None:
        raise SystemExit(f"{f} has no layer p{args.p}")
    return list(layer["best_betas"][:args.p]), list(layer["best_gammas"][:args.p]), f"grid:{args.grid}/p{args.p}"


def qaoa_circuit(model, betas, gammas):
    p = len(betas)
    circ = QAOAAnsatz(model.to_sparse_pauli_op(), reps=p)
    by_name = {q.name: q for q in circ.parameters}
    values = {by_name[f"β[{i}]"]: betas[i] for i in range(p)} | {by_name[f"γ[{i}]"]: gammas[i] for i in range(p)}
    return circ.assign_parameters(values)


# ---------- metrics ----------

def sample_metrics(counts, n, edges, exact_cost):
    """Same metrics as the campaign's final evaluation, from measured counts."""
    total = sum(counts.values())
    cands = decode_distribution({int(k, 2): v / total for k, v in counts.items()}, n, edges)
    best = best_feasible_or_none(cands)
    p_feas = feasible_probability(cands)
    valid = [c for c in cands if not c.violations]
    mis = -exact_cost
    p_opt = float(sum(c.probability for c in valid if c.cost == exact_cost))
    mass = float(sum(c.probability * c.selected for c in valid))
    return {
        "shots": total,
        "P_optimum": p_opt,
        "P_feasible": p_feas,
        "mean_valid_size": mass / p_feas if p_feas > 0 else None,
        "approx_ratio_all": mass / mis,
        "approx_ratio_valid": mass / p_feas / mis if p_feas > 0 else None,
        "TTS99_shots": tts99(p_opt),
        "best_size": best.selected if best else None,
        "hit_optimum": bool(best and best.cost == exact_cost),
        "distinct_bitstrings": len(counts),
    }


def random_baseline(n, edges, exact_cost, shots, seed=0):
    """Chance level: uniform random bitstrings, same number of shots."""
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=(shots, n))
    e = np.asarray(edges, dtype=int).reshape(-1, 2)
    viol = (bits[:, e[:, 0]] & bits[:, e[:, 1]]).sum(axis=1) if len(e) else np.zeros(shots, dtype=int)
    size = bits.sum(axis=1)
    ok = viol == 0
    mis = -exact_cost
    return {"P_feasible": float(ok.mean()), "P_optimum": float((ok & (size == mis)).mean()),
            "best_size": int(size[ok].max()) if ok.any() else None,
            "approx_ratio_all": float((size * ok).mean() / mis)}


# ---------- main ----------

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description="Fixed-angle QAOA for the streetlight MIS on IBM Quantum")
    ap.add_argument("--name", required=True, help="results folder under src/results/ibm")
    ap.add_argument("--nodes", type=int, default=config.LIMIT_NODES)
    ap.add_argument("--graph", choices=["connected", "components"], default="connected")
    ap.add_argument("--start-node", type=int, default=config.START_NODE, help="--graph connected only")
    ap.add_argument("--tolerance", type=float, default=2.0, help="conflict distance = 20 m x tolerance")
    ap.add_argument("--p", type=int, default=3)
    ap.add_argument("--angles", choices=["grid", "campaign", "file"], default="grid")
    ap.add_argument("--grid", default="n30_statevector_GPU", help="grid_search_ext folder")
    ap.add_argument("--campaign", default="abc_C")
    ap.add_argument("--seed", type=int, default=1, help="campaign run to take the angles from")
    ap.add_argument("--angles-file", default=None)
    ap.add_argument("--shots", type=int, default=10_000)
    ap.add_argument("--backend", default=None, help="e.g. ibm_fez; default: least busy with enough qubits")
    ap.add_argument("--fake", default=None, help="fake backend for --dry-run without an account, e.g. FakeFez")
    ap.add_argument("--optimization-level", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true", help="transpile and report the circuit size only (no QPU time)")
    args = ap.parse_args()
    if args.fake and not args.dry_run:
        raise SystemExit("--fake is only for --dry-run (fake backends are never used to execute)")

    # graph, optimum and classical references
    if args.graph == "components":
        nodes, edges, sizes = graph_from_components(config.CSV_PATH, args.nodes, args.tolerance)
    else:
        nodes, edges = build_graph_from_csv(config.CSV_PATH, args.nodes, start_node=args.start_node,
                                            tolerance_factor=args.tolerance)
        sizes = [len(nodes)]
        if len(nodes) < args.nodes:
            raise SystemExit(f"Only {len(nodes)} connected poles from node {args.start_node} at this distance; "
                             f"use --graph components")
    n = len(nodes)
    t0 = time.time()
    _, exact_cost = solve_exact_ilp(n, edges)
    ilp_s = time.time() - t0
    t0 = time.time()
    greedy = greedy_mis(n, edges)
    greedy_s = time.time() - t0
    model = qubo_to_ising(n, *build_mis_qubo(n, edges))
    betas, gammas, source = load_angles(args)
    if len(betas) != args.p:
        raise SystemExit(f"angles have {len(betas)} layers, --p is {args.p}")

    runner = IBMRunner(args.backend, min_qubits=n, optimization_level=args.optimization_level, fake=args.fake)
    isa, transpile_s = runner.transpile(qaoa_circuit(model, betas, gammas))
    stats = circuit_stats(isa)
    print(f"\n{'=' * 70}\nIBM {args.name} | {n} poles, {len(edges)} edges (groups {sizes}) | MIS {-exact_cost:.0f} "
          f"(ILP {ilp_s:.3f} s, greedy {greedy})\nangles {source} | backend {runner.name}"
          f"{' (fake, dry run)' if runner.is_fake else ''} | transpiled in {transpile_s:.1f} s: depth {stats['depth']}, "
          f"2-qubit depth {stats['depth_2q']}, 2-qubit gates {stats['two_qubit_gates']}\n{'=' * 70}", flush=True)

    out = config.RESULTS_BASE_DIR / "ibm" / args.name
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "name": args.name, "n_nodes": n, "n_edges": len(edges), "graph": args.graph, "group_sizes": sizes,
        "tolerance": args.tolerance, "start_node": args.start_node if args.graph == "connected" else None,
        "exact_cost": exact_cost, "mis_size": int(-exact_cost), "ilp_s": ilp_s,
        "greedy_size": greedy, "greedy_s": greedy_s,
        "p": args.p, "angles_source": source, "betas": betas, "gammas": gammas,
        "backend": runner.name, "fake_backend": runner.is_fake, "optimization_level": args.optimization_level,
        "transpile_s": transpile_s, "circuit": stats, "dry_run": args.dry_run,
        "random_baseline": random_baseline(n, edges, exact_cost, args.shots),
    }
    if not args.dry_run:
        counts, job = runner.run(isa, args.shots)
        summary["job"] = job
        summary.update(sample_metrics(counts, n, edges, exact_cost))
        (out / "counts.json").write_text(json.dumps(counts), encoding="utf-8")
        print(f"  P(opt) {summary['P_optimum']:.4f} | P(feas) {summary['P_feasible']:.3f} | best valid size "
              f"{summary['best_size']} of {-exact_cost:.0f} | QPU {job['qpu_seconds']} s", flush=True)
    rb = summary["random_baseline"]
    print(f"  random bitstrings ({args.shots}): P(feas) {rb['P_feasible']:.4f} | P(opt) {rb['P_optimum']:.4f} "
          f"| best valid size {rb['best_size']}")
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved {out / 'summary.json'}")
