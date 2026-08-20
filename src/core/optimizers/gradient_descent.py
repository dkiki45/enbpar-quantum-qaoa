import numpy as np
import time
import csv
import os
from datetime import datetime
from typing import List
from qiskit_aer import AerSimulator

from core.lattice_model import QuantumLattice2D
from core.quantum_physics import HamiltonianTerm, evaluate_lattice_energy

def gradient_descent_lattice(lattice: QuantumLattice2D, simulator: AerSimulator, h_terms: List[HamiltonianTerm], generations: int = 100, learning_rate: float = 0.1, h_step: float = 0.05, experiment_name: str = "experiment_gd", qubo_offset: float = 0.0):
    """
    Optimizes the quantum matrix using Gradient Descent.
    Uses the central finite differences method to approximate the derivative 
    of the quantum energy with respect to the theta angle of each Qubit.
    """
    print("\n--- STARTING GRADIENT DESCENT ---")
    
    history_log = []
    start_time = time.time()
 
    best_energy = evaluate_lattice_energy(lattice, simulator, h_terms)
    print(f"Gen 00 [START] | Energy: {best_energy:.4f}")
 
    history_log.append({
        "generation": 0, "energy": round(best_energy, 4), "objective_cx": round(best_energy + qubo_offset, 4),
        "status": "START", "time_elapsed_sec": 0.0
    })
    
    # Extracts the initial angles
    n_qubits = lattice.n_qubits
    thetas = np.zeros(n_qubits)
    for row in range(lattice.rows):
        for col in range(lattice.cols):
            cell = lattice.cells[row, col]
            if cell.occupied:
                thetas[cell.qubit_index] = cell.theta

    for gen in range(1, generations + 1):
        gradients = np.zeros(n_qubits)
        
        # Calculates the gradient for each pole (Qubit)
        for i in range(n_qubits):
            # +h (forward)
            thetas[i] += h_step
            _apply_thetas_to_lattice(lattice, thetas)
            e_plus = evaluate_lattice_energy(lattice, simulator, h_terms)
            
            # -h (backward)
            thetas[i] -= 2 * h_step
            _apply_thetas_to_lattice(lattice, thetas)
            e_minus = evaluate_lattice_energy(lattice, simulator, h_terms)
            
            # Restores
            thetas[i] += h_step
            
            # Central derivative
            gradients[i] = (e_plus - e_minus) / (2 * h_step)
        
        # Updates weights (gradient descent)
        thetas = thetas - learning_rate * gradients
        _apply_thetas_to_lattice(lattice, thetas)
        
        current_energy = evaluate_lattice_energy(lattice, simulator, h_terms)
        if current_energy < best_energy:
            best_energy = current_energy
            status = "IMPROVED"
        else:
            status = "UPDATED"

        print(f"Gen {gen:02d} | Energy: {current_energy:+.4f} | {status}")
 
        history_log.append({
            "generation": gen, "energy": round(current_energy, 4), "objective_cx": round(current_energy + qubo_offset, 4),
            "status": status, "time_elapsed_sec": round(time.time() - start_time, 4)
        })
            
    return _save_optimizer_results(lattice, best_energy, history_log, experiment_name, qubo_offset)

def _apply_thetas_to_lattice(lattice, thetas):
    for row in range(lattice.rows):
        for col in range(lattice.cols):
            cell = lattice.cells[row, col]
            if cell.occupied:
                cell.theta = thetas[cell.qubit_index]

# We kept the saving function identical to the project's
def _save_optimizer_results(lattice, best_energy, history_log, experiment_name, qubo_offset):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = current_dir
    while not os.path.exists(os.path.join(src_dir, "data")):
        src_dir = os.path.dirname(src_dir)
        
    results_dir = os.path.join(src_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    csv_filename = os.path.join(results_dir, f"{experiment_name}_history_{timestamp}.csv")
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(history_log[0].keys()))
        writer.writeheader()
        writer.writerows(history_log)
    
    config_filename = os.path.join(results_dir, f"{experiment_name}_optimal_config_{timestamp}.txt")
    with open(config_filename, 'w') as file:
        file.write(f"Final Minimum Energy (Ising <H>): {best_energy:.4f}\n")
        file.write(f"Final Objective C(x): {best_energy + qubo_offset:.4f}\n")
        file.write("Final LED Grid Configuration (1 = ON, 0 = OFF):\n")
        for i in range(lattice.rows):
            row_str = []
            for j in range(lattice.cols):
                theta = lattice.cells[i, j].theta % (2 * np.pi)
                val = 1 if np.pi/2 < theta < 3*np.pi/2 else 0
                row_str.append(str(val))
            file.write(" ".join(row_str) + "\n")
            
    print(f"\n-> Dados exportados para: {results_dir}")
    return best_energy, lattice