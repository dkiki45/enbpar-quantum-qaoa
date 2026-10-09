"""
CPU vs GPU benchmark: time and memory of ONE QAOA evaluation (the unit the optimizer repeats
~100-300 times per run) for each simulator setup and depth p.

Every setup gets the same circuit, the same fixed random angles and the same shots, so only
the simulator changes. Each (setup, p) runs in its own subprocess: memory is measured in
isolation, and an out-of-memory crash or timeout is recorded instead of stopping the study.
Finished points are skipped, so an interrupted benchmark resumes where it stopped.

Setups:
  mps_cpu        matrix_product_state on CPU (current setup)
  sv_cpu         statevector on CPU
  sv_gpu         statevector on GPU (double)
  sv_gpu_cusv    statevector on GPU + NVIDIA cuStateVec
  sv_gpu_single  statevector on GPU, single precision (half the memory)

Columns: first = first call (includes transpile), eval = mean time of the next calls,
E_qubo = mean QUBO energy of the samples (should agree across setups within shot noise),
RAM = peak process memory (GB), VRAM = peak GPU memory of the process (GB).

Results: src/results/bench_devices/n<N>/

Usage (from the repo root):
    PYTHONPATH=src python -u src/experiments/bench_devices.py                       # all setups, p=1..6
    PYTHONPATH=src python -u src/experiments/bench_devices.py --setups sv_cpu,sv_gpu --depths 1,3,6
    PYTHONPATH=src python -u src/experiments/bench_devices.py --report              # table only
"""
import argparse
import json
import os
import resource
import subprocess
import sys
import threading
import time
import warnings
from pathlib import Path

SETUPS = {
    "mps_cpu":       {"method": "matrix_product_state", "device": "CPU", "precision": "double", "custatevec": False},
    "sv_cpu":        {"method": "statevector",          "device": "CPU", "precision": "double", "custatevec": False},
    "sv_gpu":        {"method": "statevector",          "device": "GPU", "precision": "double", "custatevec": False},
    "sv_gpu_cusv":   {"method": "statevector",          "device": "GPU", "precision": "double", "custatevec": True},
    "sv_gpu_single": {"method": "statevector",          "device": "GPU", "precision": "single", "custatevec": False},
}


class GpuMemoryWatcher(threading.Thread):
    """Polls nvidia-smi for this process's GPU memory; peak_gb stays None if nvidia-smi fails."""

    def __init__(self, interval=0.2):
        super().__init__(daemon=True)
        self.interval, self.peak_mb, self._halt = interval, None, threading.Event()

    def run(self):
        pid = str(os.getpid())
        while not self._halt.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5).stdout
                for line in out.strip().splitlines():
                    p, mb = [x.strip() for x in line.split(",")]
                    if p == pid:
                        self.peak_mb = max(self.peak_mb or 0, float(mb))
            except Exception:
                pass
            self._halt.wait(self.interval)

    def stop(self):
        self._halt.set()
        self.join(timeout=2)
        return None if self.peak_mb is None else self.peak_mb / 1024


def worker(setup_key, p, n_nodes, shots, evals):
    """Runs one (setup, p) point and prints a JSON line with the measurements."""
    import numpy as np
    from qiskit.circuit.library import QAOAAnsatz
    from core.graph_builder import build_graph_from_csv
    from core.qubo_formalization import build_mis_qubo, qubo_to_ising
    from core.qaoa_solver import FastAerSampler
    from core.solution_decoder import decode_distribution, feasible_probability
    from core.sim_backend import backend_options, check_gpu
    from config import CSV_PATH

    s = SETUPS[setup_key]
    if s["device"] == "GPU":
        err = check_gpu()
        if err:
            print(json.dumps({"status": "no_gpu", "error": err}))
            return

    nodes, edges = build_graph_from_csv(CSV_PATH, n_nodes)
    lin, quad, off = build_mis_qubo(len(nodes), edges)
    model = qubo_to_ising(len(nodes), lin, quad, off)

    circuit = QAOAAnsatz(model.to_sparse_pauli_op(), reps=p)
    circuit.measure_all()
    # Same angles for every setup: fixed RNG, depends only on p
    angles = np.random.default_rng(1234 + p).uniform(0, np.pi, circuit.num_parameters)

    opts = backend_options(s["method"], s["device"], s["precision"], custatevec=s["custatevec"])
    sampler = FastAerSampler(default_shots=shots, seed=7, options={"backend_options": opts})

    watcher = GpuMemoryWatcher() if s["device"] == "GPU" else None
    if watcher:
        watcher.start()

    times, energies, pfeas = [], [], []
    for k in range(evals + 1):  # k = 0 is the first call (includes transpile + GPU init)
        t0 = time.perf_counter()
        counts = sampler.run([(circuit, angles)]).result()[0].data.meas.get_counts()
        times.append(time.perf_counter() - t0)
        total = sum(counts.values())
        if total == 0:  # Aer returns no shots when the state does not fit in memory
            print(json.dumps({"status": "no_memory", "error": "simulator returned 0 shots"}))
            return
        dist = {int(b, 2): c / total for b, c in counts.items()}
        cands = decode_distribution(dist, len(nodes), edges)
        energies.append(sum(c.probability * c.cost for c in cands))
        pfeas.append(feasible_probability(cands))

    vram = watcher.stop() if watcher else None
    ram_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2  # Linux: KB
    print(json.dumps({
        "status": "ok", "setup": setup_key, "options": opts, "p": p, "n_qubits": len(nodes),
        "shots": shots, "evals": evals,
        "first_s": times[0], "eval_s": float(np.mean(times[1:])), "eval_times_s": times[1:],
        "E_qubo": float(np.mean(energies)), "P_feasible": float(np.mean(pfeas)),
        "ram_gb": ram_gb, "vram_gb": vram,
    }))


def run_point(setup_key, p, args, out_file):
    cmd = [sys.executable, "-u", __file__, "--worker", setup_key, str(p),
           "--nodes", str(args.nodes), "--shots", str(args.shots), "--evals", str(args.evals)]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout, env=os.environ.copy())
        lines = [l for l in proc.stdout.strip().splitlines() if l.startswith("{")]
        if lines:
            res = json.loads(lines[-1])
        else:  # crashed (e.g. killed by the OS for using too much memory)
            res = {"status": "crash", "returncode": proc.returncode, "error": proc.stderr.strip()[-800:]}
    except subprocess.TimeoutExpired:
        res = {"status": "timeout", "error": f"more than {args.timeout} s"}
    res.update({"setup": setup_key, "p": p, "wall_s": time.time() - t0})
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


def fmt(x, nd=2):
    return "-" if x is None else f"{x:.{nd}f}"


def report(base, setups, depths):
    header = f"{'setup':14s} {'p':>2s} {'status':>9s} {'first_s':>8s} {'eval_s':>8s} {'x sv_cpu':>8s} {'x mps':>7s} {'E_qubo':>8s} {'P(feas)':>7s} {'RAM GB':>7s} {'VRAM GB':>7s}"
    lines = [header, "-" * len(header)]
    data = {}
    for f in base.glob("*/p*.json"):
        r = json.loads(f.read_text(encoding="utf-8"))
        data[(r["setup"], r["p"])] = r
    for p in depths:
        ref_sv = data.get(("sv_cpu", p), {}).get("eval_s")
        ref_mps = data.get(("mps_cpu", p), {}).get("eval_s")
        for s in setups:
            r = data.get((s, p))
            if r is None:
                continue
            ev = r.get("eval_s")
            sp_sv = ref_sv / ev if ev and ref_sv else None
            sp_mps = ref_mps / ev if ev and ref_mps else None
            lines.append(f"{s:14s} {p:2d} {r['status']:>9s} {fmt(r.get('first_s')):>8s} {fmt(ev):>8s} "
                         f"{fmt(sp_sv, 1):>8s} {fmt(sp_mps, 1):>7s} {fmt(r.get('E_qubo'), 3):>8s} "
                         f"{fmt(r.get('P_feasible'), 3):>7s} {fmt(r.get('ram_gb'), 1):>7s} {fmt(r.get('vram_gb'), 1):>7s}")
        lines.append("")
    text = "\n".join(lines)
    print("\n" + text)
    errors = [f"  {k[0]} p={k[1]}: {v.get('error', '')[:200]}" for k, v in sorted(data.items()) if v["status"] != "ok"]
    if errors:
        print("Not ok:\n" + "\n".join(errors))
    (base / "report.txt").write_text(text + "\n", encoding="utf-8")
    print(f"\nTable saved to {base / 'report.txt'}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    from config import LIMIT_NODES, SHOTS_PER_EVAL, RESULTS_BASE_DIR

    ap = argparse.ArgumentParser(description="CPU vs GPU QAOA benchmark")
    ap.add_argument("--setups", default=",".join(SETUPS), help=f"comma-separated subset of {list(SETUPS)}")
    ap.add_argument("--depths", default="1,2,3,4,5,6", help="QAOA depths p (default 1,2,3,4,5,6)")
    ap.add_argument("--nodes", type=int, default=LIMIT_NODES, help="qubits (default config.LIMIT_NODES)")
    ap.add_argument("--shots", type=int, default=SHOTS_PER_EVAL)
    ap.add_argument("--evals", type=int, default=3, help="timed evaluations after the first call (default 3)")
    ap.add_argument("--timeout", type=int, default=3600, help="seconds per (setup, p) point (default 3600)")
    ap.add_argument("--rerun", action="store_true", help="recompute points that already have results")
    ap.add_argument("--report", action="store_true", help="only print the table from existing results")
    ap.add_argument("--worker", nargs=2, metavar=("SETUP", "P"), help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker:
        worker(args.worker[0], int(args.worker[1]), args.nodes, args.shots, args.evals)
        sys.exit(0)

    setups = [s.strip() for s in args.setups.split(",") if s.strip()]
    unknown = [s for s in setups if s not in SETUPS]
    if unknown:
        raise SystemExit(f"Unknown setup(s): {unknown}. Options: {list(SETUPS)}")
    depths = [int(x) for x in args.depths.split(",")]
    base = RESULTS_BASE_DIR / "bench_devices" / f"n{args.nodes}"

    if not args.report:
        print(f"\n{'=' * 60}\nBENCHMARK n={args.nodes} | setups {setups} | p={depths} | "
              f"{args.shots} shots | {args.evals} evals\n{'=' * 60}")
        for p in depths:  # all setups at a depth before going deeper: comparable data comes first
            for s in setups:
                out = base / s / f"p{p}.json"
                # Points that failed for an external reason (no GPU, crash) are retried automatically
                if out.exists() and not args.rerun:
                    prev = json.loads(out.read_text(encoding="utf-8")).get("status")
                    if prev in ("ok", "timeout", "no_memory"):
                        print(f"  {s:14s} p={p}: already done ({prev}), skipping")
                        continue
                print(f"  {s:14s} p={p}: running...", end=" ", flush=True)
                r = run_point(s, p, args, out)
                if r["status"] == "ok":
                    print(f"eval {r['eval_s']:.2f} s | first {r['first_s']:.2f} s | E_qubo {r['E_qubo']:.3f}", flush=True)
                else:
                    print(f"{r['status'].upper()}: {r.get('error', '')[:150]}", flush=True)
    report(base, setups, depths)
