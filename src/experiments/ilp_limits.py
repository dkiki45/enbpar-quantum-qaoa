"""
How far can the exact ILP baseline (scipy.optimize.milp / HiGHS) go?

Solves Maximum Independent Set on growing graphs with a time limit and records whether
optimality was PROVEN (status 'optimal', gap 0) or the limit was hit (best solution found
so far + remaining gap). The difficulty depends on the graph structure much more than on
its size, so several graph families are tested:

  csv    real streetlight graph from config.CSV_PATH (BFS subgraph, up to the component size)
  geo    random geometric graph, average degree ~3 (similar to streetlights, any size)
  er<d>  Erdos-Renyi random graph with average degree d, e.g. er3, er6, er10 (hard cases)

Results: src/results/ilp_limits/<kind>.csv (one row per size, appended as it runs)

Usage (from the repo root):
    PYTHONPATH=src python -u src/experiments/ilp_limits.py --kinds csv --sizes 30,60,90,125
    PYTHONPATH=src python -u src/experiments/ilp_limits.py --kinds geo --sizes 1000,5000,10000,50000
    PYTHONPATH=src python -u src/experiments/ilp_limits.py --kinds er3,er6,er10 --sizes 100,150,200,300,500 --time-limit 300
"""
import argparse
import csv
import time

import networkx as nx
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix

FIELDS = ["kind", "n", "edges", "time_s", "status", "mis", "gap", "bb_nodes", "time_limit_s"]


def solve_mis(n, edges, time_limit):
    """MIS ILP with a sparse constraint matrix (the dense one in classical_baseline does not scale)."""
    m = max(len(edges), 1)
    rows = np.repeat(np.arange(len(edges)), 2)
    cols = np.asarray(edges, dtype=int).ravel() if edges else np.array([], dtype=int)
    A = coo_matrix((np.ones(len(cols)), (rows, cols)), shape=(m, n)).tocsr()
    t0 = time.time()
    res = milp(-np.ones(n), constraints=[LinearConstraint(A, -np.inf, 1)],
               integrality=np.ones(n), bounds=Bounds(0, 1), options={"time_limit": time_limit})
    dt = time.time() - t0
    status = {0: "optimal", 1: "LIMIT"}.get(res.status, f"status{res.status}")
    mis = int(round(-res.fun)) if res.x is not None else None
    return dt, status, mis, getattr(res, "mip_gap", None), getattr(res, "mip_node_count", None)


def make_graph(kind, n):
    if kind == "csv":
        from core.graph_builder import build_graph_from_csv
        from config import CSV_PATH
        nodes, edges = build_graph_from_csv(CSV_PATH, n)
        return len(nodes), edges  # may be smaller than n: limited by the connected component
    if kind == "geo":
        g = nx.random_geometric_graph(n, radius=(3.0 / (np.pi * n)) ** 0.5, seed=1)
    elif kind.startswith("er"):
        d = float(kind[2:])
        g = nx.fast_gnp_random_graph(n, d / (n - 1), seed=1)
    else:
        raise SystemExit(f"Unknown kind {kind!r}: use csv, geo or er<d> (e.g. er6)")
    return n, list(g.edges())


if __name__ == "__main__":
    from config import RESULTS_BASE_DIR

    ap = argparse.ArgumentParser(description="ILP (exact MIS) scaling limits")
    ap.add_argument("--kinds", default="csv,geo,er3,er6", help="graph families, comma-separated")
    ap.add_argument("--sizes", default="30,60,125,250,500,1000", help="number of nodes, comma-separated")
    ap.add_argument("--time-limit", type=float, default=60, help="seconds per instance (default 60)")
    args = ap.parse_args()

    out_dir = RESULTS_BASE_DIR / "ilp_limits"
    out_dir.mkdir(parents=True, exist_ok=True)
    sizes = [int(x) for x in args.sizes.split(",")]

    print(f"{'kind':6s} {'n':>7s} {'edges':>8s} {'time_s':>8s} {'status':>8s} {'MIS':>7s} {'gap':>7s} {'B&B nodes':>9s}")
    for kind in [k.strip() for k in args.kinds.split(",") if k.strip()]:
        path = out_dir / f"{kind}.csv"
        new = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                w.writeheader()
            seen = set()
            for n_req in sizes:
                n, edges = make_graph(kind, n_req)
                if (kind, n) in seen:  # csv: sizes above the component size give the same graph
                    continue
                seen.add((kind, n))
                dt, status, mis, gap, nodes = solve_mis(n, edges, args.time_limit)
                row = {"kind": kind, "n": n, "edges": len(edges), "time_s": round(dt, 3), "status": status,
                       "mis": mis, "gap": None if gap is None else round(gap, 4), "bb_nodes": nodes,
                       "time_limit_s": args.time_limit}
                w.writerow(row)
                f.flush()
                print(f"{kind:6s} {n:7d} {len(edges):8d} {dt:8.2f} {status:>8s} {str(mis):>7s} "
                      f"{str(row['gap']):>7s} {str(nodes):>9s}", flush=True)
    print(f"\nResults saved to {out_dir}/")
