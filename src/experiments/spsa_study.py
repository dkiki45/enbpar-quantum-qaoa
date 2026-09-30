"""
SPSA hyperparameter study: does early stopping (patience) or the auto-calibrated
step size explain SPSA's poor results?

Every config runs on the same graph, depth and seeds (paired comparison).
Results: src/results/spsa_study/<scenario>/<config>/seed<N>/summary.json
Finished runs are skipped, so an interrupted study resumes where it stopped.

Usage (from the repo root):
    PYTHONPATH=src python -u src/experiments/spsa_study.py --depth 1 --seeds 10
    PYTHONPATH=src python -u src/experiments/spsa_study.py --depth 3 --warm --seeds 4 --configs A,C
    PYTHONPATH=src python -u src/experiments/spsa_study.py --depth 1 --seeds 10 --report   # table only
"""
import argparse
import json
import statistics
import time
import warnings
from dataclasses import asdict
from pathlib import Path

from core.graph_builder import build_graph_from_csv
from core.qubo_formalization import build_mis_qubo, qubo_to_ising
from core.qaoa_solver import run_qaoa
from core.solution_decoder import decode_distribution, best_feasible_or_none, feasible_probability
from core.classical_baseline import solve_exact_ilp
from config import LIMIT_NODES, CSV_PATH, SHOTS_PER_EVAL, RESULTS_BASE_DIR

MAX_ITER = 300

# A = current behavior (baseline)
CONFIGS = {
    "A": {"label": "patience 15, calibrated (current)", "spsa_patience": 15,   "spsa_learning_rate": None, "spsa_perturbation": None},
    "B": {"label": "patience 50, calibrated",           "spsa_patience": 50,   "spsa_learning_rate": None, "spsa_perturbation": None},
    "C": {"label": "no early stop, calibrated",         "spsa_patience": None, "spsa_learning_rate": None, "spsa_perturbation": None},
    "D": {"label": "patience 15, fixed step 0.05",      "spsa_patience": 15,   "spsa_learning_rate": 0.05, "spsa_perturbation": 0.05},
    "E": {"label": "no early stop, fixed step 0.05",    "spsa_patience": None, "spsa_learning_rate": 0.05, "spsa_perturbation": 0.05},
}


def warm_start_point(p):
    from run_warm_start import build_warm_start_point
    path = RESULTS_BASE_DIR / "v3_grid_search" / "warm_start.json"
    with open(path, "r") as f:
        return build_warm_start_point(p, json.load(f))


def run_one(model, nodes, edges, exact_cost, p, seed, cfg, initial_point, out_dir):
    t0 = time.time()
    result = run_qaoa(model, reps=p, shots=SHOTS_PER_EVAL, seed=seed, maxiter=MAX_ITER,
                      optimizer_name="SPSA", initial_point=initial_point,
                      spsa_patience=cfg["spsa_patience"],
                      spsa_learning_rate=cfg["spsa_learning_rate"],
                      spsa_perturbation=cfg["spsa_perturbation"])
    candidates = decode_distribution(result.distribution, len(nodes), edges, nodes=nodes)
    best = best_feasible_or_none(candidates)
    summary = {
        "n_nodes": len(nodes), "n_edges": len(edges), "reps": p, "seed": seed,
        "config": {k: v for k, v in cfg.items()},
        "expectation_qubo": result.expectation_qubo,
        "feasible": best is not None,
        "feasible_probability": feasible_probability(candidates),
        "best_cost": best.cost if best is not None else None,
        "best_selected": best.selected if best is not None else None,
        "exact_cost": exact_cost,
        "hit_optimum": best is not None and best.cost == exact_cost,
        "n_evaluations": len(result.history),
        "runtime_s": time.time() - t0,
        "warm_start": initial_point,
        "best": asdict(best) if best is not None else None,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def report(base_dir, config_keys, seeds):
    rows = []
    header = f"{'cfg':3s} {'description':34s} {'n':>2s} {'E_qubo mean':>11s} {'std':>6s} {'optimum':>8s} {'P(feas)':>8s} {'evals':>6s} {'min/run':>7s}"
    print("\n" + header + "\n" + "-" * len(header))
    for key in config_keys:
        runs = []
        for s in seeds:
            f = base_dir / key / f"seed{s}" / "summary.json"
            if f.exists():
                runs.append(json.loads(f.read_text(encoding="utf-8")))
        if not runs:
            print(f"{key:3s} {CONFIGS[key]['label']:34s}  (no results yet)")
            continue
        e = [r["expectation_qubo"] for r in runs]
        std = statistics.stdev(e) if len(e) > 1 else 0.0
        hits = sum(r["hit_optimum"] for r in runs)
        pf = statistics.mean(r["feasible_probability"] for r in runs)
        ev = statistics.mean(r["n_evaluations"] for r in runs)
        rt = statistics.mean(r["runtime_s"] for r in runs) / 60
        line = f"{key:3s} {CONFIGS[key]['label']:34s} {len(runs):2d} {statistics.mean(e):11.3f} {std:6.3f} {hits:>3d}/{len(runs):<4d} {pf:8.3f} {ev:6.0f} {rt:7.1f}"
        print(line)
        rows.append(line)
    (base_dir / "report.txt").write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    print(f"\nTable saved to {base_dir / 'report.txt'}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description="SPSA hyperparameter study")
    ap.add_argument("--depth", type=int, default=1, help="QAOA depth p (default 1)")
    ap.add_argument("--seeds", type=int, default=10, help="number of seeds, 1..N (default 10)")
    ap.add_argument("--warm", action="store_true", help="start from grid_search warm-start angles (needs p >= 2)")
    ap.add_argument("--configs", default=",".join(CONFIGS), help="comma-separated subset, e.g. A,C,E")
    ap.add_argument("--report", action="store_true", help="only print the table from existing results")
    args = ap.parse_args()

    keys = [k.strip().upper() for k in args.configs.split(",") if k.strip()]
    unknown = [k for k in keys if k not in CONFIGS]
    if unknown:
        raise SystemExit(f"Unknown config(s): {unknown}. Options: {list(CONFIGS)}")
    if args.warm and args.depth < 2:
        raise SystemExit("--warm needs --depth >= 2 (warm-start uses the p=1 and p=2 grid search angles)")

    seeds = list(range(1, args.seeds + 1))
    scenario = f"n{LIMIT_NODES}_p{args.depth}_{'warm' if args.warm else 'blind'}"
    base_dir = RESULTS_BASE_DIR / "spsa_study" / scenario

    if not args.report:
        nodes, edges = build_graph_from_csv(CSV_PATH, LIMIT_NODES)
        lin, quad, off = build_mis_qubo(len(nodes), edges)
        model = qubo_to_ising(len(nodes), lin, quad, off)
        _, exact_cost = solve_exact_ilp(len(nodes), edges)
        initial_point = warm_start_point(args.depth) if args.warm else None

        print(f"\n{'=' * 60}\nSPSA STUDY: {scenario} | configs {keys} | seeds 1..{args.seeds} | optimum {exact_cost}\n{'=' * 60}")
        total = len(keys) * len(seeds)
        done = 0
        for key in keys:
            cfg = CONFIGS[key]
            print(f"\n>> Config {key}: {cfg['label']}")
            for s in seeds:
                done += 1
                out_dir = base_dir / key / f"seed{s}"
                if (out_dir / "summary.json").exists():
                    print(f"  [{done}/{total}] seed {s:02d} already done, skipping")
                    continue
                r = run_one(model, nodes, edges, exact_cost, args.depth, s, cfg, initial_point, out_dir)
                print(f"  [{done}/{total}] seed {s:02d} | E_qubo {r['expectation_qubo']:8.3f} | best {r['best_cost']} "
                      f"| optimum {'YES' if r['hit_optimum'] else 'no'} | evals {r['n_evaluations']} | {r['runtime_s'] / 60:.1f} min", flush=True)

    report(base_dir, keys, seeds)
