"""
A/B/C comparison (Report 05 feedback): does COBYLA improve the point given by the grid?

Reads three run_campaign.py campaigns run with the same depth, seeds and perturbations:
    A = --mode none   (perturbed grid point, no optimization)
    B = --mode shots  (COBYLA, energy estimated with shots)
    C = --mode exact  (COBYLA, exact energy)

For each seed s the comparison is paired:
    delta_E_s = E_final_exact - E_initial_exact   (initial = grid point after the perturbation of seed s)
    < 0: the optimization improved the point, ~ 0: small change, > 0: it got worse.
    delta_P(opt)_s = P(optimum) of B or C - P(optimum) of A (same start point, same final-sampling seed).
Prints and saves the per-seed table, the mean and 95% confidence interval (t distribution) of the
paired differences, quality per time, and a plot of delta_E per seed with a line at zero.

Outputs in src/results/campaign/abc_report/: report.md, runs.csv, delta_E.png

Usage (from the repo root):
    PYTHONPATH=src python src/experiments/abc_report.py
    PYTHONPATH=src python src/experiments/abc_report.py --names smoke_abc_none,smoke_abc_shots,smoke_abc_exact --p 2
"""
import argparse
import csv
import json
import math
import statistics as st

import config


def paired_ci(values, level=0.95):
    """Mean and confidence interval of paired differences (t distribution). Returns (mean, low, high)."""
    from scipy.stats import t
    n = len(values)
    m = st.mean(values)
    if n < 2:
        return m, None, None
    half = t.ppf(0.5 + level / 2, n - 1) * st.stdev(values) / math.sqrt(n)
    return m, m - half, m + half


def load(name, p):
    base = config.RESULTS_BASE_DIR / "campaign" / name
    runs = {}
    for f in sorted(base.glob(f"p{p}_seed*/summary.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        runs[r["seed"]] = r
    if not runs:
        raise SystemExit(f"No runs with p={p} in {base}")
    return runs


def fmt(x, nd=3):
    return "-" if x is None else f"{x:.{nd}f}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Paired A/B/C comparison")
    ap.add_argument("--names", default="abc_A,abc_B,abc_C", help="campaigns A,B,C (no optimization, shots, exact)")
    ap.add_argument("--p", type=int, default=3)
    args = ap.parse_args()
    names = [x.strip() for x in args.names.split(",")]
    if len(names) != 3:
        raise SystemExit("--names needs three campaigns: A,B,C")
    A, B, C = (load(n, args.p) for n in names)
    seeds = sorted(set(A) & set(B) & set(C))
    if not seeds:
        raise SystemExit("No seed is present in the three campaigns")
    for lab, runs in zip("ABC", (A, B, C)):
        missing = sorted(set(runs) - set(seeds))
        if missing:
            print(f"warning: case {lab} has seeds not in the other cases (ignored): {missing}")
    # same start point in the three cases (same seeds and perturbation) -> the pairing is valid
    for s in seeds:
        e0 = [A[s]["E_exact_initial"], B[s]["E_exact_initial"], C[s]["E_exact_initial"]]
        if max(e0) - min(e0) > 1e-6:
            raise SystemExit(f"seed {s}: the three cases start from different points ({e0}); check --perturb/--extend")

    lines = [f"# A/B/C comparison, p = {args.p}, {len(seeds)} seeds",
             f"A = `{names[0]}` (no optimization) · B = `{names[1]}` (COBYLA, shots) · C = `{names[2]}` (COBYLA, exact energy)",
             f"MIS size {int(-A[seeds[0]]['exact_cost'])} · {A[seeds[0]]['n_nodes']} nodes", ""]

    # per-case summary
    lines += ["## Per case", "",
              "| case | E final mean | ΔE mean [95% CI] | runs with ΔE < 0 | P(opt) mean | TTS99 median (shots) "
              "| P(feas) mean | approx. ratio (invalid = 0) | optimum found | evals mean | optimization min/run | total min/run |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for lab, runs in zip("ABC", (A, B, C)):
        rs = [runs[s] for s in seeds]
        de = [r["delta_E"] for r in rs]
        m, lo, hi = paired_ci(de)
        tts = [r["TTS99_shots"] for r in rs if r.get("TTS99_shots") is not None]
        lines.append(
            f"| {lab} | {fmt(st.mean(r['E_exact_final'] for r in rs))} | {fmt(m)} [{fmt(lo)}, {fmt(hi)}] "
            f"| {sum(d < 0 for d in de)}/{len(de)} | {fmt(st.mean(r['P_optimum'] for r in rs), 4)} "
            f"| {fmt(st.median(tts) if tts else None, 0)} | {fmt(st.mean(r['P_feasible'] for r in rs))} "
            f"| {fmt(st.mean(r['approx_ratio_all'] for r in rs))} | {sum(r['hit_optimum'] for r in rs)}/{len(rs)} "
            f"| {fmt(st.mean(r['n_evaluations'] for r in rs), 0)} | {fmt(st.mean(r['optimize_s'] for r in rs) / 60, 1)} "
            f"| {fmt(st.mean(r['total_run_s'] for r in rs) / 60, 1)} |")
    lines.append("")

    # paired differences against A
    lines += ["## Paired differences against A (same start point, same final-sampling seed)", "",
              "| case | ΔP(opt) vs A mean [95% CI] | ΔP(feas) vs A mean [95% CI] | extra minutes per run vs A |",
              "| --- | --- | --- | --- |"]
    for lab, runs in (("B", B), ("C", C)):
        dpo = [runs[s]["P_optimum"] - A[s]["P_optimum"] for s in seeds]
        dpf = [runs[s]["P_feasible"] - A[s]["P_feasible"] for s in seeds]
        dt = [(runs[s]["total_run_s"] - A[s]["total_run_s"]) / 60 for s in seeds]
        a, alo, ahi = paired_ci(dpo)
        b, blo, bhi = paired_ci(dpf)
        lines.append(f"| {lab} | {fmt(a, 4)} [{fmt(alo, 4)}, {fmt(ahi, 4)}] | {fmt(b)} [{fmt(blo)}, {fmt(bhi)}] "
                     f"| {fmt(st.mean(dt), 1)} |")
    lines += ["", "Reading guide (feedback, section 6): C improves and B does not -> limited by shots; "
              "neither improves -> shot noise alone does not explain it; A same quality in less time -> grid alone "
              "becomes the reference.", ""]

    # per-seed table
    lines += ["## Per seed", "",
              "| seed | E initial | ΔE B | ΔE C | P(opt) A | P(opt) B | P(opt) C | evals B | evals C |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for s in seeds:
        lines.append(f"| {s} | {fmt(A[s]['E_exact_initial'])} | {fmt(B[s]['delta_E'], 4)} | {fmt(C[s]['delta_E'], 4)} "
                     f"| {fmt(A[s]['P_optimum'], 4)} | {fmt(B[s]['P_optimum'], 4)} | {fmt(C[s]['P_optimum'], 4)} "
                     f"| {B[s]['n_evaluations']} | {C[s]['n_evaluations']} |")

    text = "\n".join(lines)
    print(text)
    out = config.RESULTS_BASE_DIR / "campaign" / "abc_report"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(text + "\n", encoding="utf-8")

    cols = ["case", "seed", "E_exact_initial", "E_exact_final", "delta_E", "P_optimum", "P_feasible",
            "approx_ratio_all", "approx_ratio_valid", "TTS99_shots", "hit_optimum", "n_evaluations",
            "init_eval_s", "optimize_s", "final_eval_s", "total_run_s"]
    with open(out / "runs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for lab, runs in zip("ABC", (A, B, C)):
            for s in seeds:
                w.writerow({"case": lab, **runs[s]})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, title in ((axes[0], "delta_E", "ΔE = E final − E initial (exact)"),
                           (axes[1], "dP", "ΔP(optimum) vs A")):
        for off, lab, runs, color in ((-0.15, "B (shots)", B, "tab:gray"), (0.15, "C (exact)", C, "tab:blue")):
            ys = [runs[s]["delta_E"] if key == "delta_E" else runs[s]["P_optimum"] - A[s]["P_optimum"] for s in seeds]
            ax.scatter([s + off for s in seeds], ys, label=lab, color=color, s=22)
            m, lo, hi = paired_ci(ys)
            ax.axhline(m, color=color, lw=1, ls="--")
        ax.axhline(0, color="black", lw=1)
        ax.set_xlabel("seed")
        ax.set_title(title)
        ax.legend()
    fig.suptitle(f"A/B/C, p = {args.p}, {len(seeds)} seeds (dashed = mean)")
    fig.tight_layout()
    fig.savefig(out / "delta_E.png", dpi=150)
    print(f"\nSaved {out / 'report.md'}, {out / 'runs.csv'} and {out / 'delta_E.png'}")
