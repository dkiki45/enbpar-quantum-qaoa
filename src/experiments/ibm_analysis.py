"""
Post-analysis of the IBM hardware runs (no QPU time): uses the counts saved by run_ibm.py.

For each run, compares the hardware samples with two random baselines (same number of shots):
  * uniform: each pole chosen with probability 1/2;
  * same marginals: each pole chosen independently with the probability measured on the hardware.
    It has the same density of chosen poles as the hardware but no correlation between poles, so
    beating it means the samples avoid conflicts because of the circuit, not only because they
    choose fewer poles.
Metrics:
  * conflicts per sample: mean number of violated edges (pairs of chosen neighbours).
    A random bitstring violates each edge with probability 1/4, so far fewer conflicts than
    |E|/4 means the circuit still pushes towards valid sets even when no sample is valid.
  * chosen poles per sample (Hamming weight).
  * repaired samples: every sample is made valid by removing, one by one, the pole with the most
    conflicts, and then completed with the min-degree greedy (adds poles with no chosen neighbour,
    fewest free neighbours first). Same repair for the random samples. Reports the mean and best
    repaired size and how often the repair reaches the MIS, next to the greedy from an empty set.
    The repair is classical post-processing (as in Ebadi et al., Science 2022): it shows how good a
    starting point the samples are, not the quantum result alone.

Results: src/results/ibm/<name>/analysis.json

Usage (from the repo root):
    PYTHONPATH=src python src/experiments/ibm_analysis.py                    # every run with counts.json
    PYTHONPATH=src python src/experiments/ibm_analysis.py --names n30_abcC,n156_abcC
"""
import argparse
import json
from collections import defaultdict

import numpy as np

import config
from core.graph_builder import build_graph_from_csv
from run_ibm import graph_from_components


def rebuild_graph(summary):
    if summary["graph"] == "components":
        nodes, edges, _ = graph_from_components(config.CSV_PATH, summary["n_nodes"], summary["tolerance"])
    else:
        nodes, edges = build_graph_from_csv(config.CSV_PATH, summary["n_nodes"], start_node=summary["start_node"],
                                            tolerance_factor=summary["tolerance"])
    if len(nodes) != summary["n_nodes"] or len(edges) != summary["n_edges"]:
        raise SystemExit(f"rebuilt graph differs from the run ({len(nodes)} nodes, {len(edges)} edges)")
    return len(nodes), edges


def bits_from_key(key, n):
    """Qiskit key (highest qubit on the left) -> x_0..x_(n-1)."""
    return np.array([int(c) for c in reversed(key.zfill(n))], dtype=np.int8)


def repair(bits, adj):
    """Remove the most-conflicting pole until no conflict, then add poles greedily (min free degree)."""
    chosen = set(np.flatnonzero(bits).tolist())
    conflicts = {v: len(adj[v] & chosen) for v in chosen}
    while True:
        worst = max(conflicts, key=lambda v: (conflicts[v], v), default=None)
        if worst is None or conflicts[worst] == 0:
            break
        chosen.discard(worst)
        del conflicts[worst]
        for w in adj[worst] & chosen:
            conflicts[w] -= 1
    free = {v for v in adj if v not in chosen and not (adj[v] & chosen)}
    while free:
        v = min(free, key=lambda u: (len(adj[u] & free), u))
        chosen.add(v)
        free -= adj[v] | {v}
    return len(chosen)


def stats(samples, weights, edges, adj, mis):
    """samples: array (k, n) of distinct bitstrings; weights: counts of each."""
    e = np.asarray(edges, dtype=int).reshape(-1, 2)
    w = np.asarray(weights, dtype=float)
    total = w.sum()
    viol = (samples[:, e[:, 0]] & samples[:, e[:, 1]]).sum(axis=1)
    weight = samples.sum(axis=1)
    rep = np.array([repair(s, adj) for s in samples])
    return {
        "mean_conflicts": float((viol * w).sum() / total),
        "mean_chosen": float((weight * w).sum() / total),
        "P_valid": float(w[viol == 0].sum() / total),
        "repaired_mean": float((rep * w).sum() / total),
        "repaired_best": int(rep.max()),
        "P_repaired_optimum": float(w[rep == mis].sum() / total),
    }


def analyse(name):
    folder = config.RESULTS_BASE_DIR / "ibm" / name
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    counts = json.loads((folder / "counts.json").read_text(encoding="utf-8"))
    n, edges = rebuild_graph(summary)
    adj = defaultdict(set)
    for i in range(n):
        adj[i]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    mis = summary["mis_size"]
    keys = list(counts)
    hw_bits = np.array([bits_from_key(k, n) for k in keys])
    hw_w = np.array([counts[k] for k in keys], dtype=float)
    hw = stats(hw_bits, hw_w, edges, adj, mis)
    shots = int(hw_w.sum())
    rng = np.random.default_rng(0)
    rnd = stats(rng.integers(0, 2, size=(shots, n)).astype(np.int8), np.ones(shots), edges, adj, mis)
    marg = (hw_bits * hw_w[:, None]).sum(axis=0) / shots  # P(pole i chosen) on the hardware
    same = stats((rng.random((shots, n)) < marg).astype(np.int8), np.ones(shots), edges, adj, mis)
    out = {"name": name, "backend": summary.get("backend"), "n_nodes": n, "n_edges": len(edges), "mis_size": mis,
           "greedy_size": summary["greedy_size"], "shots": shots,
           "expected_random_conflicts": len(edges) / 4, "hardware": hw, "random_same_marginals": same,
           "random": rnd}
    (folder / "analysis.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def fmt(x, nd=2):
    return "-" if x is None else f"{x:.{nd}f}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Analysis of IBM hardware counts (conflicts and repaired sets)")
    ap.add_argument("--names", default="", help="comma-separated runs (default: every run with counts.json)")
    args = ap.parse_args()
    base = config.RESULTS_BASE_DIR / "ibm"
    names = [x.strip() for x in args.names.split(",") if x.strip()] or \
        sorted(d.name for d in base.iterdir() if (d / "counts.json").exists())
    if not names:
        raise SystemExit(f"No run with counts.json in {base}")
    for name in names:
        r = analyse(name)
        h, m, z = r["hardware"], r["random_same_marginals"], r["random"]
        print(f"\n## {name} ({r['backend']}) | {r['n_nodes']} poles, {r['n_edges']} edges | MIS {r['mis_size']} "
              f"| greedy {r['greedy_size']} | {r['shots']} shots")
        print("| | hardware | random, same marginals | random, uniform |")
        print("| --- | --- | --- | --- |")
        print(f"| conflicts per sample (random expects {r['expected_random_conflicts']:.1f}) "
              f"| {fmt(h['mean_conflicts'])} | {fmt(m['mean_conflicts'])} | {fmt(z['mean_conflicts'])} |")
        print(f"| chosen poles per sample | {fmt(h['mean_chosen'])} | {fmt(m['mean_chosen'])} | {fmt(z['mean_chosen'])} |")
        print(f"| valid samples | {fmt(100 * h['P_valid'])}% | {fmt(100 * m['P_valid'])}% | {fmt(100 * z['P_valid'])}% |")
        print(f"| repaired size, mean | {fmt(h['repaired_mean'])} | {fmt(m['repaired_mean'])} | {fmt(z['repaired_mean'])} |")
        print(f"| repaired size, best | {h['repaired_best']} | {m['repaired_best']} | {z['repaired_best']} |")
        print(f"| repaired = MIS | {fmt(100 * h['P_repaired_optimum'])}% | {fmt(100 * m['P_repaired_optimum'])}% | {fmt(100 * z['P_repaired_optimum'])}% |")
        print(f"Saved {base / name / 'analysis.json'}")
