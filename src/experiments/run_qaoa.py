import json
import matplotlib.pyplot as plt
from dataclasses import asdict
from pathlib import Path
import sys
import time
import statistics
from pathlib import Path

from core.graph_builder import build_graph_from_csv
from core.qubo_formalization import build_mis_qubo, qubo_to_ising
from core.qaoa_solver import run_qaoa
from core.solution_decoder import decode_distribution, best_feasible_candidate
from core.classical_baseline import solve_exact_bruteforce

def execute(csv_path, output, limit=15, reps=1, shots=8192, seed=2, optimizer_name="COBYLA"):
    nodes, edges = build_graph_from_csv(csv_path, limit)
    linear, quadratic, off = build_mis_qubo(len(nodes), edges)
    model = qubo_to_ising(len(nodes), linear, quadratic, off)
    
    result = run_qaoa(model, reps, shots, seed, maxiter=300, optimizer_name=optimizer_name)
    candidates = decode_distribution(result.distribution, len(nodes), edges, nodes=nodes)
    best = best_feasible_candidate(candidates)
    
    exact_bits, exact_cost = solve_exact_bruteforce(len(nodes), edges)
    
    summary = {
        "n_nodes": len(nodes), 
        "n_edges": len(edges), 
        "reps": reps,
        "shots": shots, 
        "seed": seed, 
        "expectation_ising": result.expectation_ising,
        "expectation_qubo": result.expectation_qubo, 
        "best": asdict(best),
        "exact_bits": exact_bits, 
        "exact_cost": exact_cost,
        "cardinality_ratio": sum(best.bits) / sum(exact_bits) if sum(exact_bits) > 0 else 0.0,
        "parameters": result.optimal_parameters
    }
    
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    
    return summary, result.history

if __name__ == "__main__":
    optimizer = sys.argv[1].upper() if len(sys.argv) > 1 else "COBYLA"
    
    limit_nodes = 15  

    csv_path = "src/data/paranainterativo.csv"
    shots = 1024
    seeds = list(range(1, 16)) 
    reps_list = [1, 2, 3]

    base_dir = Path(f"src/results/stress_n{limit_nodes}_{optimizer}")
    base_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"STRESS TEST: {limit_nodes} NODES | OPTIMIZER: {optimizer}")
    print(f"{'='*50}")

    start_total = time.time()

    for p in depths:
        print(f"\n>> Starting Depth p={p}...")
        energies = []
        
        for s in seeds:
            target_dir = base_dir / f"p{p}_seed{s}"
            
            summary, _ = execute(
                csv_path=csv_path, 
                output=str(target_dir), 
                limit=limit_nodes, 
                reps=p, 
                shots=shots, 
                seed=s,
                optimizer_name=optimizer
            )
            energies.append(summary["expectation_qubo"])
            print(f"  - Seed {s:02d} completed | Energy: {summary['expectation_qubo']:.4f}")
        
        mean_energy = statistics.mean(energies)
        std_dev = statistics.stdev(energies)
        print(f" SUMMARY {optimizer} (p={p}) -> Mean: {mean_energy:.4f} | Std Dev: {std_dev:.4f}")

    total_time = (time.time() - start_total) / 60
    print(f"\n {optimizer} COMPLETED IN {total_time:.2f} MINUTES.")