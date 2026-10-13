import sys
import json
import time
import statistics
import warnings
from pathlib import Path
from dataclasses import asdict

from core.graph_builder import build_graph_from_csv
from core.qubo_formalization import build_mis_qubo, qubo_to_ising
from core.qaoa_solver import run_qaoa
from core.solution_decoder import decode_distribution, best_feasible_or_none, feasible_probability
from core.classical_baseline import solve_exact_ilp
from core.sim_backend import results_suffix, describe, require_gpu_if_selected
from config import LIMIT_NODES, CSV_PATH, SHOTS_PER_EVAL, SEEDS_TO_TEST, DEPTHS_WARM_START, RESULTS_BASE_DIR, SKIP_COMPLETED_RUNS
import config

def build_warm_start_point(p, warm_start_data):
    """
    Builds the initial_point array like a staircase:
    Layer 1 gets p=1 angles, Layer 2 gets p=2 angles, subsequent layers get 0.0.
    Qiskit expects the array format: [beta_0, beta_1, ..., gamma_0, gamma_1, ...]
    """
    b0 = warm_start_data["p1_best_beta"]
    g0 = warm_start_data["p1_best_gamma"]
    b1 = warm_start_data["p2_best_beta"]
    g1 = warm_start_data["p2_best_gamma"]
    
    betas = [b0, b1] + [0.0] * (p - 2)
    gammas = [g0, g1] + [0.0] * (p - 2)
    
    return betas + gammas

def build_ext_warm_start_point(p, ext_summary):
    """
    Initial point from grid_search_ext.py: uses the best angles of the deepest searched
    layer k <= p (summary["p<k>"]), and starts the remaining p - k layers at 0.0.
    Same [beta_0..beta_{p-1}, gamma_0..gamma_{p-1}] format as build_warm_start_point.
    """
    available = sorted(int(k[1:]) for k in ext_summary if k.startswith("p") and k[1:].isdigit())
    usable = [k for k in available if k <= p]
    if not usable:
        raise ValueError(f"grid_search_ext summary has no layer <= p={p} (available: {available})")
    k = usable[-1]
    betas = list(ext_summary[f"p{k}"]["best_betas"][:k]) + [0.0] * (p - k)
    gammas = list(ext_summary[f"p{k}"]["best_gammas"][:k]) + [0.0] * (p - k)
    return betas + gammas

def load_warm_start():
    """Returns (source, data, path) according to config.WARM_START_SOURCE ("v3" or "ext")."""
    source = getattr(config, "WARM_START_SOURCE", "v3")
    if source == "v3":
        path = RESULTS_BASE_DIR / "v3_grid_search" / "warm_start.json"
        hint = "Run grid_search.py first."
    elif source == "ext":
        sub_dir = getattr(config, "WARM_START_EXT_DIR", None) or f"grid_search_ext/n{LIMIT_NODES}_statevector_GPU"
        path = RESULTS_BASE_DIR / sub_dir / "summary.json"
        hint = "Run grid_search_ext.py first or set WARM_START_EXT_DIR."
    else:
        raise SystemExit(f"WARM_START_SOURCE must be 'v3' or 'ext', got {source!r}")
    try:
        return source, json.loads(Path(path).read_text(encoding="utf-8")), path
    except FileNotFoundError:
        raise SystemExit(f"ERROR: Could not find {path}. {hint}")

def initial_point_for(p, source, data):
    return build_ext_warm_start_point(p, data) if source == "ext" else build_warm_start_point(p, data)

def execute_warm_start(csv_path, output, p_depth, warm_start_data, limit=10, shots=1024, seed=2, optimizer_name="COBYLA", max_iter=300,
                       warm_start_source="v3"):
    nodes, edges = build_graph_from_csv(csv_path, limit)
    linear, quadratic, off = build_mis_qubo(len(nodes), edges)
    model = qubo_to_ising(len(nodes), linear, quadratic, off)
    
    initial_pt = initial_point_for(p_depth, warm_start_source, warm_start_data)
    
    result = run_qaoa(
        model, 
        reps=p_depth, 
        shots=shots, 
        seed=seed, 
        maxiter=max_iter, 
        optimizer_name=optimizer_name,
        initial_point=initial_pt
    )
    
    candidates = decode_distribution(result.distribution, len(nodes), edges, nodes=nodes)
    # None when no sample is a valid independent set (optimizer stuck / barren plateau)
    best = best_feasible_or_none(candidates)
    if best is None:
        print(f"    [WARNING] No feasible sample for p={p_depth}, seed={seed} (energy {result.expectation_qubo:.4f})")
    exact_bits, exact_cost = solve_exact_ilp(len(nodes), edges)
    
    summary = {
        "n_nodes": len(nodes), 
        "n_edges": len(edges), 
        "reps": p_depth,
        "shots": shots, 
        "seed": seed, 
        "expectation_qubo": result.expectation_qubo, 
        "feasible": best is not None,
        "feasible_probability": feasible_probability(candidates),
        "best": asdict(best) if best is not None else None,
        "exact_cost": exact_cost,
        "parameters": result.optimal_parameters,
        "warm_start_used": initial_pt,
        "warm_start_source": warm_start_source,
        "simulator": describe()
    }
    
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    
    return summary

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    optimizer = sys.argv[1].upper() if len(sys.argv) > 1 else "COBYLA"
    max_iterations = 300
    
    ws_source, warm_data, json_path = load_warm_start()

    require_gpu_if_selected()
    # "ext" warm-start runs get their own folders, so SKIP_COMPLETED_RUNS never mixes them with "v3" runs
    ws_suffix = "_gridext" if ws_source == "ext" else ""
    base_dir = RESULTS_BASE_DIR / f"warm_start_n{LIMIT_NODES}_{optimizer}{results_suffix()}{ws_suffix}"
    base_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"WARM-START TEST: {LIMIT_NODES} NODES | OPTIMIZER: {optimizer}")
    print(f"Simulator: {describe()} -> {base_dir}")
    print(f"Warm-start: {ws_source} ({json_path})")
    print(f"{'='*50}")

    start_total = time.time()

    for p in DEPTHS_WARM_START:  
        print(f"\n>> Starting Depth p={p} with Warm-Start...")
        energies = []
        infeasible = 0
        
        for s in SEEDS_TO_TEST:
            target_dir = base_dir / f"p{p}_seed{s}"
            summary_path = target_dir / "summary.json"
            if SKIP_COMPLETED_RUNS and summary_path.exists():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                print(f"  - Seed {s:02d} already done, loaded from {summary_path}")
                energies.append(summary["expectation_qubo"])
                infeasible += 0 if summary.get("feasible", True) else 1
                continue
            summary = execute_warm_start(
                csv_path=CSV_PATH, 
                output=str(target_dir), 
                p_depth=p,
                warm_start_data=warm_data,
                limit=LIMIT_NODES, 
                shots=SHOTS_PER_EVAL, 
                seed=s,
                optimizer_name=optimizer,
                max_iter=max_iterations,
                warm_start_source=ws_source
            )
            energies.append(summary["expectation_qubo"])
            feasible = summary.get("feasible", True)
            infeasible += 0 if feasible else 1
            print(f"  - Seed {s:02d} completed | Energy: {summary['expectation_qubo']:.4f} | Feasible: {feasible}")
        
        mean_energy = statistics.mean(energies)
        std_dev = statistics.stdev(energies)
        print(f" SUMMARY {optimizer} (p={p}) -> Mean: {mean_energy:.4f} | Std Dev: {std_dev:.4f} | Infeasible seeds: {infeasible}/{len(SEEDS_TO_TEST)}")

    total_time = (time.time() - start_total) / 60
    print(f"\n {optimizer} WARM-START COMPLETED IN {total_time:.2f} MINUTES.")