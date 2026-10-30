"""
Extended grid search of the QAOA energy landscape (long runs on the GPU).

Differences from the old grid_search.py (removed from main; see branch arquivo-campanha-gpu):
  * EXACT energy at every grid point (Aer save_expectation_value, no shot noise),
    so the landscape is the true one, not a 1024-shot estimate.
  * Finer grids (--steps, default 50x50) evaluated in batches (one Aer job per grid row).
  * Layer by layer up to --max-p: p=1 full (beta, gamma) grid; for p>=2 the previous layers
    are fixed at the best angles found and the new layer is grid-searched (same idea as
    grid_search.py p=2). After each grid, a zoom grid around the best point refines it.
  * At the best angles of each p, 4096 shots are sampled and compared with the exact ILP
    optimum: best bitstring found, P(feasible) and P(optimum).
  * Resumable: every finished row is saved, so an interrupted run continues where it stopped.
  * --energy shots estimates each grid point with --grid-shots samples instead (the old
    grid_search.py way), to measure how much the exact energy matters. --tag names the
    results folder, so grid variants (e.g. 20x20 exact vs 20x20 shots) never overwrite each other.

Method/device come from config.py or --method/--device, e.g. --method tensor_network --device GPU.

Results: src/results/grid_search_ext/n<N>_<method>_<device>[_<tag>]/
  p<k>_landscape.npy / .png, p<k>_zoom.npy, summary.json (best angles, energies, sampling stats),
  warm_start.json (best angles per layer, same format as the old v3_grid_search)

Usage (from the repo root):
    PYTHONPATH=src python -u src/experiments/grid_search_ext.py --method statevector --device GPU --max-p 3
    PYTHONPATH=src python -u src/experiments/grid_search_ext.py --method tensor_network --device GPU --max-p 3
    PYTHONPATH=src python -u src/experiments/grid_search_ext.py --estimate      # time per point and ETA only
    PYTHONPATH=src python -u src/experiments/grid_search_ext.py --method statevector --device GPU --steps 20 --zoom-steps 0 --energy shots --tag s20_shots
"""
import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
from qiskit.circuit.library import QAOAAnsatz

from core.graph_builder import build_graph_from_csv
from core.qubo_formalization import build_mis_qubo, qubo_to_ising
from core.classical_baseline import solve_exact_ilp
from core.solution_decoder import decode_distribution, best_feasible_or_none, feasible_probability
from core.qaoa_solver import FastAerSampler
from core.evaluators import LandscapeEvaluator, ShotsEvaluator
from core.sim_backend import backend_options, check_gpu
from config import LIMIT_NODES, CSV_PATH, RESULTS_BASE_DIR, SIMULATION_METHOD

BETA_RANGE = (0.0, np.pi)
GAMMA_RANGE = (0.0, 2 * np.pi)


def grid_layer(ev, fixed_b, fixed_g, betas, gammas, store, label):
    """Grid over the LAST layer's (beta, gamma); earlier layers fixed. Resumes from `store` (.npy)."""
    land = np.load(store) if store.exists() else np.full((len(betas), len(gammas)), np.nan)
    if land.shape != (len(betas), len(gammas)):
        raise SystemExit(f"{store} has another grid size: delete it or use the same --steps")
    t_start, done_now = time.time(), 0
    for i, b in enumerate(betas):
        if not np.isnan(land[i]).any():
            continue
        pts = [(fixed_b + [b], fixed_g + [g]) for g in gammas]
        land[i] = ev.energies(pts)
        np.save(store, land)
        done_now += 1
        left = int(np.isnan(land).any(axis=1).sum())
        eta = (time.time() - t_start) / done_now * left / 60
        print(f"  [{label}] row {i + 1}/{len(betas)} | best so far {np.nanmin(land):.4f} | ETA {eta:.1f} min", flush=True)
    i, j = np.unravel_index(np.nanargmin(land), land.shape)
    return land, betas[i], gammas[j], land[i, j]


def zoom(ev, fixed_b, fixed_g, b0, g0, db, dg, steps, store, label):
    """Finer grid of +-2 cells around the best point."""
    betas = np.linspace(b0 - 2 * db, b0 + 2 * db, steps)
    gammas = np.linspace(g0 - 2 * dg, g0 + 2 * dg, steps)
    return grid_layer(ev, fixed_b, fixed_g, betas, gammas, store, label)


def sample_at(model, nodes, edges, exact_cost, p, b, g, sim_opts, shots):
    """Sample the QAOA state at the best angles and compare with the ILP optimum."""
    h = model.to_sparse_pauli_op()
    circ = QAOAAnsatz(h, reps=p)
    circ.measure_all()
    order = {q.name: q for q in circ.parameters}
    values = {order[f"β[{k}]"]: b[k] for k in range(p)} | {order[f"γ[{k}]"]: g[k] for k in range(p)}
    sampler = FastAerSampler(default_shots=shots, seed=7, options={"backend_options": sim_opts})
    counts = sampler.run([circ.assign_parameters(values)]).result()[0].data.meas.get_counts()
    total = sum(counts.values())
    cands = decode_distribution({int(k, 2): v / total for k, v in counts.items()}, len(nodes), edges)
    best = best_feasible_or_none(cands)
    return {
        "shots": shots,
        "best_cost": best.cost if best else None,
        "hit_optimum": bool(best and best.cost == exact_cost),
        "P_feasible": feasible_probability(cands),
        "P_optimum": float(sum(c.probability for c in cands if not c.violations and c.cost == exact_cost)),
        "mean_sampled_energy": float(sum(c.probability * c.cost for c in cands)),
    }


def heatmap(land, betas, gammas, best, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.contourf(gammas, betas, land, 50, cmap="viridis")
    plt.colorbar(label="Exact QUBO energy")
    plt.plot(best[1], best[0], "w*", markersize=15, label="best")
    plt.xlabel(r"$\gamma$ (last layer)")
    plt.ylabel(r"$\beta$ (last layer)")
    plt.title(title)
    plt.legend()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description="Extended exact grid search of the QAOA landscape")
    ap.add_argument("--method", default=SIMULATION_METHOD, help="statevector, tensor_network, matrix_product_state")
    ap.add_argument("--device", default=None, help="CPU or GPU (default config.SIMULATION_DEVICE)")
    ap.add_argument("--max-p", type=int, default=3, help="deepest layer to grid-search (default 3)")
    ap.add_argument("--steps", type=int, default=50, help="grid points per angle (default 50 -> 2500 points per p)")
    ap.add_argument("--zoom-steps", type=int, default=21, help="points per angle in the zoom grid (0 = no zoom)")
    ap.add_argument("--shots", type=int, default=4096, help="shots for the final sampling at the best angles")
    ap.add_argument("--nodes", type=int, default=LIMIT_NODES)
    ap.add_argument("--estimate", action="store_true", help="time one grid row per p and print the ETA, then exit")
    ap.add_argument("--energy", choices=["exact", "shots"], default="exact",
                    help="grid energy: exact (default) or estimated from --grid-shots samples")
    ap.add_argument("--grid-shots", type=int, default=1024, help="samples per grid point with --energy shots")
    ap.add_argument("--tag", default="", help="suffix for the results folder (grid variants)")
    args = ap.parse_args()

    opts = backend_options(args.method, args.device)
    if opts.get("device") == "GPU":
        err = check_gpu()
        if err:
            raise SystemExit(f"GPU not usable: {err}")
    dev = opts.get("device", "CPU")
    out = RESULTS_BASE_DIR / "grid_search_ext" / (f"n{args.nodes}_{opts['method']}_{dev}" + (f"_{args.tag}" if args.tag else ""))
    out.mkdir(parents=True, exist_ok=True)

    nodes, edges = build_graph_from_csv(CSV_PATH, args.nodes)
    model = qubo_to_ising(len(nodes), *build_mis_qubo(len(nodes), edges))
    _, exact_cost = solve_exact_ilp(len(nodes), edges)
    betas = np.linspace(*BETA_RANGE, args.steps)
    gammas = np.linspace(*GAMMA_RANGE, args.steps)
    db, dg = betas[1] - betas[0], gammas[1] - gammas[0]

    print(f"\n{'=' * 64}\nEXTENDED GRID SEARCH | {len(nodes)} qubits | {opts} | p=1..{args.max_p}\n"
          f"grid {args.steps}x{args.steps} (+ zoom {args.zoom_steps}x{args.zoom_steps}) | energy {args.energy}"
          f" | ILP optimum {exact_cost}\n-> {out}\n{'=' * 64}")

    def make_evaluator(p):
        if args.energy == "shots":
            return ShotsEvaluator(model, p, opts, edges, shots=args.grid_shots)
        return LandscapeEvaluator(model, p, opts)

    if args.estimate:
        for p in range(1, args.max_p + 1):
            ev = make_evaluator(p)
            t0 = time.time()
            ev.energies([([0.3] * p, [0.5] * p)])  # warm-up (GPU init, first job)
            t1 = time.time()
            ev.energies([([0.3] * (p - 1) + [b], [0.5] * p) for b in betas])
            per_point = (time.time() - t1) / len(betas)
            n_pts = args.steps ** 2 + args.zoom_steps ** 2
            print(f"p={p}: {per_point:.3f} s/point (first job {t1 - t0:.1f} s) -> ~{per_point * n_pts / 3600:.1f} h for this p")
        raise SystemExit(0)

    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    summary.update({"n_qubits": len(nodes), "n_edges": len(edges), "exact_cost": exact_cost,
                    "backend_options": opts, "steps": args.steps, "zoom_steps": args.zoom_steps,
                    "energy": args.energy, "grid_shots": args.grid_shots if args.energy == "shots" else None,
                    "beta_range": BETA_RANGE, "gamma_range": GAMMA_RANGE})
    fixed_b, fixed_g = [], []
    t_total = time.time()
    for p in range(1, args.max_p + 1):
        fixed_txt = f" | earlier layers fixed: betas {np.round(fixed_b, 4).tolist()}, gammas {np.round(fixed_g, 4).tolist()}" if p > 1 else ""
        print(f"\n>> p={p}{fixed_txt}")
        t_p = time.time()
        ev = make_evaluator(p)
        land, b, g, e = grid_layer(ev, fixed_b, fixed_g, betas, gammas, out / f"p{p}_landscape.npy", f"p={p} grid")
        heatmap(land, betas, gammas, (b, g), out / f"p{p}_landscape.png",
                f"Exact QAOA landscape p={p} ({opts['method']}, {dev})" + (" - earlier layers fixed" if p > 1 else ""))
        grid_best = {"beta": float(b), "gamma": float(g), "energy": float(e)}
        if args.zoom_steps > 0:
            _, b, g, e = zoom(ev, fixed_b, fixed_g, b, g, db, dg, args.zoom_steps, out / f"p{p}_zoom.npy", f"p={p} zoom")
        fixed_b, fixed_g = fixed_b + [float(b)], fixed_g + [float(g)]
        # exact energy of the chosen angles (with --energy shots, `e` is a noisy estimate)
        e_exact = float(LandscapeEvaluator(model, p, opts).energies([(fixed_b, fixed_g)])[0]) if args.energy == "shots" else float(e)
        stats = sample_at(model, nodes, edges, exact_cost, p, fixed_b, fixed_g, opts, args.shots)
        summary[f"p{p}"] = {"grid_best": grid_best, "best_betas": fixed_b, "best_gammas": fixed_g,
                            "best_energy": float(e), "best_energy_exact": e_exact,
                            "minutes": (time.time() - t_p) / 60, **stats}
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"   best p={p}: E={e:.4f} | sampled best {stats['best_cost']} (optimum {exact_cost}) | "
              f"P(feasible) {stats['P_feasible']:.3f} | P(optimum) {stats['P_optimum']:.4f}", flush=True)

    if args.max_p >= 2:  # same format as v3_grid_search/warm_start.json
        ws = {"p1_best_beta": fixed_b[0], "p1_best_gamma": fixed_g[0],
              "p2_best_beta": fixed_b[1], "p2_best_gamma": fixed_g[1]}
        (out / "warm_start.json").write_text(json.dumps(ws, indent=4), encoding="utf-8")
    print(f"\nDone in {(time.time() - t_total) / 60:.1f} min. Results in {out}/")
