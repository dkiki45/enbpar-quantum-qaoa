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
from core.solution_decoder import decode_distribution, best_feasible_candidate
from core.classical_baseline import solve_exact_bruteforce
from config import LIMIT_NODES, CSV_PATH, SHOTS_PER_EVAL, SEEDS_TO_TEST, DEPTHS_WARM_START, RESULTS_BASE_DIR

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

def execute_warm_start(csv_path, output, p_depth, warm_start_data, limit=10, shots=1024, seed=2, optimizer_name="COBYLA", max_iter=300):
    nodes, edges = build_graph_from_csv(csv_path, limit)
    linear, quadratic, off = build_mis_qubo(len(nodes), edges)
    model = qubo_to_ising(len(nodes), linear, quadratic, off)
    
    initial_pt = build_warm_start_point(p_depth, warm_start_data)
    
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
    best = best_feasible_candidate(candidates)
    exact_bits, exact_cost = solve_exact_bruteforce(len(nodes), edges)
    
    summary = {
        "n_nodes": len(nodes), 
        "n_edges": len(edges), 
        "reps": p_depth,
        "shots": shots, 
        "seed": seed, 
        "expectation_qubo": result.expectation_qubo, 
        "best": asdict(best),
        "exact_cost": exact_cost,
        "parameters": result.optimal_parameters,
        "warm_start_used": initial_pt
    }
    
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    
    return summary

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    optimizer = sys.argv[1].upper() if len(sys.argv) > 1 else "COBYLA"
    max_iterations = 100 if optimizer == "SPSA" else 300
    
    json_path = RESULTS_BASE_DIR / "v3_grid_search" / "warm_start.json"
    try:
        with open(json_path, "r") as f:
            warm_data = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Could not find {json_path}. Run grid_search.py first.")
        sys.exit(1)

    base_dir = RESULTS_BASE_DIR / f"warm_start_n{LIMIT_NODES}_{optimizer}"
    base_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"WARM-START TEST: {LIMIT_NODES} NODES | OPTIMIZER: {optimizer}")
    print(f"{'='*50}")

    start_total = time.time()

    for p in DEPTHS_WARM_START:  
        print(f"\n>> Starting Depth p={p} with Warm-Start...")
        energies = []
        
        for s in SEEDS_TO_TEST:
            target_dir = base_dir / f"p{p}_seed{s}"
            summary = execute_warm_start(
                csv_path=CSV_PATH, 
                output=str(target_dir), 
                p_depth=p,
                warm_start_data=warm_data,
                limit=LIMIT_NODES, 
                shots=SHOTS_PER_EVAL, 
                seed=s,
                optimizer_name=optimizer,
                max_iter=max_iterations
            )
            energies.append(summary["expectation_qubo"])
            print(f"  - Seed {s:02d} completed | Energy: {summary['expectation_qubo']:.4f}")
        
        mean_energy = statistics.mean(energies)
        std_dev = statistics.stdev(energies)
        print(f" SUMMARY {optimizer} (p={p}) -> Mean: {mean_energy:.4f} | Std Dev: {std_dev:.4f}")

    total_time = (time.time() - start_total) / 60
    print(f"\n {optimizer} WARM-START COMPLETED IN {total_time:.2f} MINUTES.")