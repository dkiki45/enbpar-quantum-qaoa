"""
Experiment campaign (Report 04 feedback): COBYLA warm-started from a grid_search_ext.py result,
with a standardized final evaluation of every run.

One run = (depth p, seed). For each run:
  1. Initial point: best angles of the deepest grid layer k <= min(p, --grid-max-layer); the p - k
     extra layers are filled by --extend (zeros | perturb | interp). Then the seed adds a uniform
     perturbation of +-(--perturb) rad to ALL angles, so different seeds start from different
     points (with --perturb 0 the seeds differ only in the shot sampling).
  2. COBYLA optimization (run_qaoa, --maxiter, --shots per evaluation).
  3. Final evaluation at the optimized angles: exact energy (no shots) and an independent
     sampling of --final-shots shots -> P(optimum), P(feasible), mean size of the valid
     solutions, best solution found, number of evaluations, times.

The graph can differ from the grid's (--nodes / --start-node / --tolerance): the grid angles are
then transferred to another instance (generalization test). The optimum is always certified by the
ILP; a greedy MIS heuristic is recorded too, with its time, as a classical reference.

Results: src/results/campaign/<name>/p<p>_seed<s>/summary.json (+ baseline.json, config.json)
Finished runs are skipped. Runs go seed by seed (all depths of seed 1, then seed 2, ...), so an
interrupted campaign still has balanced data across depths.

Usage (from the repo root):
    PYTHONPATH=src python -u src/experiments/run_campaign.py --name exp1_depth --depths 3,4,5,6 --seeds 20
    PYTHONPATH=src python -u src/experiments/run_campaign.py --name exp3_interp --depths 5,6 --seeds 10 --extend interp
    PYTHONPATH=src python -u src/experiments/campaign_report.py      # tables of every campaign
"""
import argparse
import io
import contextlib
import json
import time
import warnings

import numpy as np
from qiskit.circuit.library import QAOAAnsatz

import config
from core.graph_builder import build_graph_from_csv
from core.qubo_formalization import build_mis_qubo, qubo_to_ising
from core.classical_baseline import solve_exact_ilp
from core.qaoa_solver import run_qaoa, FastAerSampler
from core.solution_decoder import decode_distribution, best_feasible_or_none, feasible_probability
from core.sim_backend import backend_options, check_gpu


# ---------- initial point ----------

def interp_layer(angles):
    """INTERP (Zhou et al., 2020): angles for p+1 layers from the p-layer optimum."""
    q = len(angles)
    a = [0.0] + list(angles) + [0.0]  # a[0] = a[q+1] = 0, a[1..q] = angles
    return [((i - 1) / q) * a[i - 1] + ((q - i + 1) / q) * a[i] for i in range(1, q + 2)]


def initial_point(grid_summary, p, max_layer, extend, perturb, extend_sigma, seed):
    """[beta_0..beta_{p-1}, gamma_0..gamma_{p-1}], same order as run_warm_start.py."""
    available = sorted(int(k[1:]) for k in grid_summary if k.startswith("p") and k[1:].isdigit())
    usable = [k for k in available if k <= min(p, max_layer)]
    if not usable:
        raise SystemExit(f"grid has no layer <= {min(p, max_layer)} (available: {available})")
    k = usable[-1]
    betas = list(grid_summary[f"p{k}"]["best_betas"][:k])
    gammas = list(grid_summary[f"p{k}"]["best_gammas"][:k])
    rng = np.random.default_rng(seed)
    while len(betas) < p:
        if extend == "interp":
            betas, gammas = interp_layer(betas), interp_layer(gammas)
        elif extend == "perturb":
            betas.append(float(rng.uniform(-extend_sigma, extend_sigma)))
            gammas.append(float(rng.uniform(-extend_sigma, extend_sigma)))
        else:  # zeros
            betas.append(0.0)
            gammas.append(0.0)
    point = np.array(betas + gammas, dtype=float)
    if perturb > 0:
        point = point + rng.uniform(-perturb, perturb, size=point.size)
    return point.tolist(), k


# ---------- classical references ----------

def greedy_mis(n, edges):
    """Min-degree greedy MIS: repeatedly take the vertex with fewest remaining neighbours."""
    adj = {i: set() for i in range(n)}
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
    alive, chosen = set(range(n)), []
    while alive:
        v = min(alive, key=lambda u: (len(adj[u] & alive), u))
        chosen.append(v)
        alive -= adj[v] | {v}
    return len(chosen)


# ---------- final evaluation ----------

def ordered_angles(optimal_parameters, p):
    betas = [optimal_parameters[f"β[{i}]"] for i in range(p)]
    gammas = [optimal_parameters[f"γ[{i}]"] for i in range(p)]
    return betas, gammas


def final_evaluation(model, nodes, edges, exact_cost, p, betas, gammas, opts, shots, seed):
    from grid_search_ext import LandscapeEvaluator
    t0 = time.time()
    e_exact = float(LandscapeEvaluator(model, p, opts).energies([(betas, gammas)])[0])
    circ = QAOAAnsatz(model.to_sparse_pauli_op(), reps=p)
    circ.measure_all()
    by_name = {q.name: q for q in circ.parameters}
    values = {by_name[f"β[{i}]"]: betas[i] for i in range(p)} | {by_name[f"γ[{i}]"]: gammas[i] for i in range(p)}
    sampler = FastAerSampler(default_shots=shots, seed=100_000 + seed, options={"backend_options": opts})
    counts = sampler.run([circ.assign_parameters(values)]).result()[0].data.meas.get_counts()
    total = sum(counts.values())
    cands = decode_distribution({int(k, 2): v / total for k, v in counts.items()}, len(nodes), edges)
    best = best_feasible_or_none(cands)
    p_feas = feasible_probability(cands)
    valid = [c for c in cands if not c.violations]
    return {
        "final_shots": shots,
        "E_exact_final": e_exact,
        "P_optimum": float(sum(c.probability for c in valid if c.cost == exact_cost)),
        "P_feasible": p_feas,
        "mean_valid_size": float(sum(c.probability * c.selected for c in valid) / p_feas) if p_feas > 0 else None,
        "best_cost": best.cost if best else None,
        "best_size": best.selected if best else None,
        "hit_optimum": bool(best and best.cost == exact_cost),
        "final_eval_s": time.time() - t0,
    }


# ---------- main ----------

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description="Warm-started QAOA campaign with standardized final evaluation")
    ap.add_argument("--name", required=True, help="campaign name = results folder")
    ap.add_argument("--grid", default=None, help="grid_search_ext folder under src/results/grid_search_ext (default n30_<method>_<device>)")
    ap.add_argument("--grid-max-layer", type=int, default=99, help="use grid layers up to this depth only (e.g. 2 = old-style init)")
    ap.add_argument("--depths", default="4", help="QAOA depths, comma-separated")
    ap.add_argument("--seeds", type=int, default=20, help="seeds 1..N")
    ap.add_argument("--extend", choices=["zeros", "perturb", "interp"], default="zeros", help="how to fill layers beyond the grid")
    ap.add_argument("--extend-sigma", type=float, default=0.05, help="range (rad) of the random new layers with --extend perturb")
    ap.add_argument("--perturb", type=float, default=0.05, help="per-seed uniform perturbation of all initial angles (rad); 0 = none")
    ap.add_argument("--maxiter", type=int, default=300)
    ap.add_argument("--shots", type=int, default=1024, help="shots per COBYLA evaluation")
    ap.add_argument("--final-shots", type=int, default=10_000, help="independent shots of the final evaluation")
    ap.add_argument("--nodes", type=int, default=config.LIMIT_NODES)
    ap.add_argument("--start-node", type=int, default=config.START_NODE)
    ap.add_argument("--tolerance", type=float, default=2.0, help="graph density: conflict distance = 20 m x tolerance")
    ap.add_argument("--method", default="statevector")
    ap.add_argument("--device", default="GPU")
    args = ap.parse_args()

    # run_qaoa reads method/device from config: set them for this process
    config.SIMULATION_METHOD, config.SIMULATION_DEVICE = args.method, args.device.upper()
    opts = backend_options(args.method, args.device)
    if opts.get("device") == "GPU":
        err = check_gpu()
        if err:
            raise SystemExit(f"GPU not usable: {err}")

    grid_dir = config.RESULTS_BASE_DIR / "grid_search_ext" / (args.grid or f"n30_{opts['method']}_{opts.get('device', 'CPU')}")
    try:
        grid = json.loads((grid_dir / "summary.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"Grid not found: {grid_dir}/summary.json")
    grid_minutes = {k: v.get("minutes") for k, v in grid.items() if k.startswith("p") and isinstance(v, dict)}

    out = config.RESULTS_BASE_DIR / "campaign" / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({**vars(args), "grid_dir": str(grid_dir), "backend_options": opts}, indent=2), encoding="utf-8")

    nodes, edges = build_graph_from_csv(config.CSV_PATH, args.nodes, start_node=args.start_node, tolerance_factor=args.tolerance)
    model = qubo_to_ising(len(nodes), *build_mis_qubo(len(nodes), edges))
    t0 = time.time()
    _, exact_cost = solve_exact_ilp(len(nodes), edges)
    ilp_s = time.time() - t0
    t0 = time.time()
    greedy = greedy_mis(len(nodes), edges)
    greedy_s = time.time() - t0
    baseline = {"n_nodes": len(nodes), "n_edges": len(edges), "avg_degree": 2 * len(edges) / len(nodes),
                "exact_cost": exact_cost, "mis_size": int(-exact_cost), "ilp_s": ilp_s,
                "greedy_size": greedy, "greedy_s": greedy_s}
    (out / "baseline.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")

    depths = [int(x) for x in args.depths.split(",")]
    seeds = list(range(1, args.seeds + 1))
    print(f"\n{'=' * 70}\nCAMPAIGN {args.name} | {len(nodes)} nodes, {len(edges)} edges | MIS {-exact_cost:.0f} "
          f"(ILP {ilp_s:.3f} s, greedy {greedy}) | p={depths} | seeds 1..{args.seeds}\n"
          f"grid {grid_dir.name} (layers <= {args.grid_max_layer}) | extend {args.extend} | perturb +-{args.perturb} | {opts}\n{'=' * 70}", flush=True)

    t_campaign = time.time()
    for s in seeds:
        for p in depths:
            run_dir = out / f"p{p}_seed{s}"
            if (run_dir / "summary.json").exists():
                print(f"  p={p} seed {s:02d}: already done, skipping")
                continue
            init, k = initial_point(grid, p, args.grid_max_layer, args.extend, args.perturb, args.extend_sigma, seed=1000 * p + s)
            t_opt = time.time()
            with contextlib.redirect_stdout(io.StringIO()):  # run_qaoa prints every iteration
                result = run_qaoa(model, reps=p, shots=args.shots, seed=s, maxiter=args.maxiter,
                                  optimizer_name="COBYLA", initial_point=init)
            opt_s = time.time() - t_opt
            betas, gammas = ordered_angles(result.optimal_parameters, p)
            final = final_evaluation(model, nodes, edges, exact_cost, p, betas, gammas, opts, args.final_shots, s)
            grid_used_min = sum(m for kk, m in grid_minutes.items() if m and int(kk[1:]) <= k)
            summary = {
                "campaign": args.name, "p": p, "seed": s, "n_nodes": len(nodes), "n_edges": len(edges),
                "exact_cost": exact_cost, "grid": grid_dir.name, "grid_layers_used": k, "extend": args.extend,
                "perturb": args.perturb, "initial_point": init,
                "expectation_qubo_opt": result.expectation_qubo,  # last COBYLA estimate (with shots)
                "n_evaluations": len(result.history), "optimize_s": opt_s,
                "grid_minutes_shared": grid_used_min,  # one-off cost of the grid layers used (shared by all runs)
                "total_run_s": opt_s + final["final_eval_s"],
                "final_betas": betas, "final_gammas": gammas,
                "backend_options": opts, **final,
            }
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"  p={p} seed {s:02d}: E_exact {final['E_exact_final']:8.3f} | best {final['best_cost']} "
                  f"| P(opt) {final['P_optimum']:.4f} | P(feas) {final['P_feasible']:.3f} | evals {len(result.history)} "
                  f"| {opt_s / 60:.1f} min", flush=True)
    print(f"\nCampaign {args.name} done in {(time.time() - t_campaign) / 60:.1f} min -> {out}")
