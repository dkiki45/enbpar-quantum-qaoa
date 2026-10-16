"""
Tables for the run_campaign.py results: one table per campaign, one row per depth p.

Columns: runs, optimum found (best sample = ILP optimum), P(optimum) mean and median,
P(feasible) mean, mean size of valid solutions, exact energy mean +- std, evaluations,
minutes per run, and the classical references (ILP time, greedy MIS size).

Writes src/results/campaign/report.md and all_runs.csv.

Usage (from the repo root):
    PYTHONPATH=src python src/experiments/campaign_report.py
    PYTHONPATH=src python src/experiments/campaign_report.py --only exp1_depth,exp2_grid20_exact
"""
import argparse
import csv
import json
import statistics as st

import config

COLS = ["campaign", "p", "seed", "n_nodes", "exact_cost", "grid", "grid_layers_used", "extend", "perturb",
        "E_exact_final", "P_optimum", "P_feasible", "mean_valid_size", "best_cost", "hit_optimum",
        "n_evaluations", "optimize_s", "final_eval_s", "total_run_s", "grid_minutes_shared"]


def fmt(x, nd=3):
    return "-" if x is None else f"{x:.{nd}f}"


def mean(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else None


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Campaign tables")
    ap.add_argument("--only", default="", help="comma-separated campaign names (default: all)")
    args = ap.parse_args()
    base = config.RESULTS_BASE_DIR / "campaign"
    only = {x.strip() for x in args.only.split(",") if x.strip()}

    lines, rows = [], []
    for camp in sorted(d for d in base.iterdir() if d.is_dir()):
        if only and camp.name not in only:
            continue
        runs = [json.loads(f.read_text(encoding="utf-8")) for f in sorted(camp.glob("p*_seed*/summary.json"))]
        if not runs:
            continue
        rows += runs
        bl = json.loads((camp / "baseline.json").read_text(encoding="utf-8")) if (camp / "baseline.json").exists() else {}
        cfg = json.loads((camp / "config.json").read_text(encoding="utf-8")) if (camp / "config.json").exists() else {}
        lines.append(f"## {camp.name}")
        lines.append(f"grid `{runs[0]['grid']}` (layers <= {cfg.get('grid_max_layer', '?')}) · extend `{runs[0]['extend']}` · "
                     f"perturb ±{runs[0]['perturb']} · {bl.get('n_nodes')} nodes, {bl.get('n_edges')} edges, "
                     f"MIS {bl.get('mis_size')} (ILP {fmt(bl.get('ilp_s'))} s · greedy {bl.get('greedy_size')})")
        lines.append("")
        lines.append("| p | runs | optimum found | P(opt) mean | P(opt) median | P(feas) mean | valid size | E exact mean ± std | evals | min/run |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for p in sorted({r["p"] for r in runs}):
            rs = [r for r in runs if r["p"] == p]
            e = [r["E_exact_final"] for r in rs]
            po = [r["P_optimum"] for r in rs]
            lines.append(
                f"| {p} | {len(rs)} | {sum(r['hit_optimum'] for r in rs)}/{len(rs)} | {fmt(mean(po), 4)} | {fmt(st.median(po), 4)} "
                f"| {fmt(mean([r['P_feasible'] for r in rs]))} | {fmt(mean([r['mean_valid_size'] for r in rs]), 2)} "
                f"| {fmt(mean(e))} ± {fmt(st.stdev(e) if len(e) > 1 else 0.0)} | {fmt(mean([r['n_evaluations'] for r in rs]), 0)} "
                f"| {fmt(mean([r['total_run_s'] for r in rs]) / 60, 1)} |")
        lines.append("")

    text = "\n".join(lines) if lines else "No campaign results yet."
    print(text)
    base.mkdir(parents=True, exist_ok=True)
    (base / "report.md").write_text(text + "\n", encoding="utf-8")
    with open(base / "all_runs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {base / 'report.md'} and {base / 'all_runs.csv'} ({len(rows)} runs)")
