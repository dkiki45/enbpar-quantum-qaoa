import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
import json
import warnings
from scipy.optimize import OptimizeResult  

from core.graph_builder import build_graph_from_csv
from core.qubo_formalization import build_mis_qubo, qubo_to_ising
from qiskit.primitives import StatevectorSampler
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import Optimizer  

from config import LIMIT_NODES, CSV_PATH, RESULTS_BASE_DIR

# =========================================================
# 1-Step Dummy Optimizer (Officialized for Qiskit)
# =========================================================
class DummyEvaluator(Optimizer):
    def __init__(self):
        super().__init__()
        
    def get_support_level(self):
        return {"gradient": 0, "bounds": 0, "initial_point": 3}
        
    def minimize(self, fun, x0, jac=None, bounds=None):
        energy = fun(x0)
        return OptimizeResult(x=x0, fun=energy, nfev=1)

# =========================================================

def run_grid_search_p1(csv_path, output_dir, limit=10):
    print(f"\n--- Starting Grid Search (QAOA p=1) for {limit} nodes ---")
    start_time = time.time()
    
    nodes, edges = build_graph_from_csv(csv_path, limit)
    linear, quadratic, off = build_mis_qubo(len(nodes), edges)
    model = qubo_to_ising(len(nodes), linear, quadratic, off)
    
    beta_steps, gamma_steps = 20, 20
    betas = np.linspace(0, np.pi, beta_steps)
    gammas = np.linspace(0, 2 * np.pi, gamma_steps)
    
    energy_landscape = np.zeros((beta_steps, gamma_steps))
    sampler = StatevectorSampler(default_shots=1024)
    
    dummy_opt = DummyEvaluator()
    qaoa = QAOA(sampler=sampler, optimizer=dummy_opt, reps=1)
    
    best_energy = float('inf')
    best_angles = (0.0, 0.0)
    
    print("Calculating the p=1 heatmap (this may take a few minutes)...")
    for i, b in enumerate(betas):
        for j, g in enumerate(gammas):
            qaoa.initial_point = [b, g]
            res = qaoa.compute_minimum_eigenvalue(model.to_sparse_pauli_op())
            qubo_energy = res.eigenvalue.real + getattr(model, 'offset', 0)
            energy_landscape[i, j] = qubo_energy
            
            if qubo_energy < best_energy:
                best_energy = qubo_energy
                best_angles = (b, g)

    elapsed_time = (time.time() - start_time) / 60
    print(f"p=1 completed in {elapsed_time:.2f} min. Best Energy: {best_energy:.4f}")

    # Save p=1 energy as well
    warm_start_data = {
        "p1_best_beta": best_angles[0],
        "p1_best_gamma": best_angles[1],
        "p1_best_energy": best_energy
    }
    
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "warm_start.json", "w") as f:
        json.dump(warm_start_data, f, indent=4)
        
    plt.figure(figsize=(8, 6))
    plt.contourf(gammas, betas, energy_landscape, 50, cmap='viridis')
    plt.colorbar(label='QUBO Energy')
    plt.plot(best_angles[1], best_angles[0], 'r*', markersize=15, label='Optimal Point (p=1)')
    plt.title("QAOA Energy Landscape (p=1)")
    plt.xlabel(r"$\gamma_0$")
    plt.ylabel(r"$\beta_0$")
    plt.legend()
    plt.savefig(out_dir / "qaoa_landscape_p1.png", dpi=300, bbox_inches='tight')
    
    return warm_start_data

def run_grid_search_p2(csv_path, output_dir, p1_data, limit=10):
    print(f"\n--- Starting Conditional Grid Search (QAOA p=2) ---")
    start_time = time.time()
    
    nodes, edges = build_graph_from_csv(csv_path, limit)
    linear, quadratic, off = build_mis_qubo(len(nodes), edges)
    model = qubo_to_ising(len(nodes), linear, quadratic, off)
    
    b0 = p1_data["p1_best_beta"]
    g0 = p1_data["p1_best_gamma"]
    
    beta_steps, gamma_steps = 20, 20
    betas = np.linspace(0, np.pi, beta_steps)
    gammas = np.linspace(0, 2 * np.pi, gamma_steps)
    
    energy_landscape = np.zeros((beta_steps, gamma_steps))
    sampler = StatevectorSampler(default_shots=1024)
    
    dummy_opt = DummyEvaluator()
    qaoa = QAOA(sampler=sampler, optimizer=dummy_opt, reps=2)
    
    best_energy = float('inf')
    best_p2_angles = (0.0, 0.0)
    
    print("Calculating the p=2 heatmap (first layer locked)...")
    for i, b1 in enumerate(betas):
        for j, g1 in enumerate(gammas):
            qaoa.initial_point = [b0, b1, g0, g1]
            res = qaoa.compute_minimum_eigenvalue(model.to_sparse_pauli_op())
            qubo_energy = res.eigenvalue.real + getattr(model, 'offset', 0)
            energy_landscape[i, j] = qubo_energy
            
            if qubo_energy < best_energy:
                best_energy = qubo_energy
                best_p2_angles = (b1, g1)

    elapsed_time = (time.time() - start_time) / 60
    print(f"p=2 completed in {elapsed_time:.2f} min. Best Energy: {best_energy:.4f}")

    # Update dictionary with p=2 data and save the final complete JSON
    p1_data["p2_best_beta"] = best_p2_angles[0]
    p1_data["p2_best_gamma"] = best_p2_angles[1]
    p1_data["p2_best_energy"] = best_energy

    out_dir = Path(output_dir)
    with open(out_dir / "warm_start.json", "w") as f:
        json.dump(p1_data, f, indent=4)

    plt.figure(figsize=(8, 6))
    plt.contourf(gammas, betas, energy_landscape, 50, cmap='viridis')
    plt.colorbar(label='QUBO Energy')
    plt.plot(best_p2_angles[1], best_p2_angles[0], 'w*', markersize=15, label='Optimal Point (p=2)')
    plt.title("QAOA Conditional Landscape (p=2) \n First layer locked from p=1")
    plt.xlabel(r"$\gamma_1$")
    plt.ylabel(r"$\beta_1$")
    plt.legend()
    plt.savefig(out_dir / "qaoa_landscape_p2.png", dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    # Silence SciPy warnings
    warnings.filterwarnings("ignore")
    
    target_dir = RESULTS_BASE_DIR / "v3_grid_search"
    
    # 1. Run p=1 and store the angles + energy
    p1_optimal_data = run_grid_search_p1(CSV_PATH, target_dir, limit=LIMIT_NODES)
    
    # 2. Run p=2 (using p=1 as a foundation) and save the complete JSON at the end
    run_grid_search_p2(CSV_PATH, target_dir, p1_optimal_data, limit=LIMIT_NODES)