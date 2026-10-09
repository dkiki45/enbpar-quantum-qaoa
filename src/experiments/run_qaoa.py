import json
import matplotlib.pyplot as plt
from dataclasses import asdict
from pathlib import Path
import sys
import time
import statistics
import warnings
from pathlib import Path

from core.graph_builder import build_graph_from_csv
from core.qubo_formalization import build_mis_qubo, qubo_to_ising
from core.qaoa_solver import run_qaoa
from core.solution_decoder import decode_distribution, best_feasible_or_none, feasible_probability
from core.classical_baseline import solve_exact_ilp
from core.graph_visualization import plot_graph
from core.sim_backend import results_suffix, describe, require_gpu_if_selected

def execute(csv_path, output, limit=15, reps=1, shots=8192, seed=2, optimizer_name="COBYLA", max_iter=300):
    nodes, edges = build_graph_from_csv(csv_path, limit)

    '''
    #If we want to see the generated graph before QAOA run
    '''
    #plot_graph(nodes, edges, layout="geographic") #layout: geographic / spring
   

    linear, quadratic, off = build_mis_qubo(len(nodes), edges)
    model = qubo_to_ising(len(nodes), linear, quadratic, off)
    
    result = run_qaoa(model, reps, shots, seed, maxiter=max_iter, optimizer_name=optimizer_name)
    candidates = decode_distribution(result.distribution, len(nodes), edges, nodes=nodes)
    # None when no sample is a valid independent set (optimizer stuck / barren plateau)
    best = best_feasible_or_none(candidates)
    if best is None:
        print(f"    [WARNING] No feasible sample for p={reps}, seed={seed} (energy {result.expectation_qubo:.4f})")

    '''
    #Plot the best solution found in the execution
    '''
    #plot_graph(nodes, edges, selected_bits=best.bits, layout="geographic") #layout: geographic / spring
    
    # Exact optimum via ILP (brute force is 2^n and freezes for n >= ~25)
    exact_bits, exact_cost = solve_exact_ilp(len(nodes), edges)
    
    summary = {
        "n_nodes": len(nodes), 
        "n_edges": len(edges), 
        "reps": reps,
        "shots": shots, 
        "seed": seed, 
        "expectation_ising": result.expectation_ising,
        "expectation_qubo": result.expectation_qubo, 
        "feasible": best is not None,
        "feasible_probability": feasible_probability(candidates),
        "best": asdict(best) if best is not None else None,
        "exact_bits": exact_bits, 
        "exact_cost": exact_cost,
        "cardinality_ratio": sum(best.bits) / sum(exact_bits) if best is not None and sum(exact_bits) > 0 else 0.0,
        "parameters": result.optimal_parameters,
        "simulator": describe()
    }
    
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    
    return summary, result.history

import config

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    optimizer = sys.argv[1].upper() if len(sys.argv) > 1 else "COBYLA"
    limite_iteracoes = 300
    
    # Puxando dinamicamente do config.py
    limit_nodes = config.LIMIT_NODES
    csv_path = config.CSV_PATH
    shots = config.SHOTS_PER_EVAL
    seeds = config.SEEDS_TO_TEST
    reps_list = config.DEPTHS_STANDARD

    # Usando o diretório base do config
    require_gpu_if_selected()
    base_dir = config.RESULTS_BASE_DIR / f"stress_n{limit_nodes}_{optimizer}{results_suffix()}"
    print(f"Simulator: {describe()} -> {base_dir}")
    base_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"STRESS TEST: {limit_nodes} NODES | OPTIMIZER: {optimizer}")
    print(f"{'='*50}")

    start_total = time.time()

    for p in reps_list:
        print(f"\n>> Starting Depth p={p}...")
        energies = []
        infeasible = 0
        
        for s in seeds:
            target_dir = base_dir / f"p{p}_seed{s}"
            summary_path = target_dir / "summary.json"

            if config.SKIP_COMPLETED_RUNS and summary_path.exists():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                print(f"  - Seed {s:02d} already done, loaded from {summary_path}")
            else:
                summary, _ = execute(
                    csv_path=csv_path, 
                    output=str(target_dir), 
                    limit=limit_nodes, 
                    reps=p, 
                    shots=shots, 
                    seed=s,
                    optimizer_name=optimizer,
                    max_iter=limite_iteracoes
                )
            energies.append(summary["expectation_qubo"])
            feasible = summary.get("feasible", True)
            infeasible += 0 if feasible else 1
            print(f"  - Seed {s:02d} completed | Energy: {summary['expectation_qubo']:.4f} | Feasible: {feasible}")
        
        mean_energy = statistics.mean(energies)
        std_dev = statistics.stdev(energies)
        print(f" SUMMARY {optimizer} (p={p}) -> Mean: {mean_energy:.4f} | Std Dev: {std_dev:.4f} | Infeasible seeds: {infeasible}/{len(seeds)}")

    total_time = (time.time() - start_total) / 60
    print(f"\n {optimizer} COMPLETED IN {total_time:.2f} MINUTES.")