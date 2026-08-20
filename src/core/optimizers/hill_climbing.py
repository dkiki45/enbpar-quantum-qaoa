import numpy as np
import time
import csv
import os
from datetime import datetime
from typing import List
from qiskit_aer import AerSimulator

from core.lattice_model import QuantumLattice2D
from core.quantum_physics import HamiltonianTerm, evaluate_lattice_energy
from core.optimizers.gradient_descent import _save_optimizer_results

def hill_climbing_lattice(lattice: QuantumLattice2D, simulator: AerSimulator, h_terms: List[HamiltonianTerm], generations: int = 100, mutation_rate: float = 0.3, experiment_name: str = "experiment", qubo_offset: float = 0.0, epsilon: float = 0.01):
    """
    The algorithm iteratively mutates a random qubit's angle (theta) and evaluates 
    the new Ising energy. It strictly accepts mutations that lower the global 
    energy beyond a statistical noise-tolerance threshold (epsilon).
    """
    print("\n--- STARTING LATTICE EVOLUTION ---")
    
    history_log = []
    start_time = time.time()
 
    # 1. Initial Evaluation
    best_energy = evaluate_lattice_energy(lattice, simulator, h_terms)
    print(f"Gen 00 [START] | Global Energy (Ising): {best_energy:.4f} | Equivalent C(x): {best_energy + qubo_offset:.4f}")
 
    history_log.append({
        "generation": 0,
        "qubit_mutated": "N/A",
        "energy": round(best_energy, 4),
        "objective_cx": round(best_energy + qubo_offset, 4),
        "status": "START", 
        "time_elapsed_sec": 0.0
    })
    
    for gen in range(1, generations + 1):
        # 1. Pick a random cell in the grid to mutate
        rand_row = np.random.randint(0, lattice.rows)
        rand_col = np.random.randint(0, lattice.cols)
        target_cell = lattice.cells[rand_row, rand_col]
        
        # 2. Save old state and mutate
        old_theta = target_cell.theta
        mutation = np.random.normal(0, mutation_rate)
        target_cell.theta += mutation
        
        # 3. Evaluate the new global energy
        new_energy = evaluate_lattice_energy(lattice, simulator, h_terms)
        
        # 4. Selection
        status = ""
        if new_energy < best_energy - epsilon:
            best_energy = new_energy
            status = "ACCEPTED"
            print(f"Gen {gen:02d} | Mutated Qubit {target_cell.qubit_index} | Energy: {new_energy:+.4f} | ACCEPTED")
        else:
            # Revert mutation if it didn't help the network
            target_cell.theta = old_theta
            status = "REJECTED"
 
        history_log.append({
            "generation": gen,
            "qubit_mutated": target_cell.qubit_index,
            "energy": round(best_energy, 4),
            "objective_cx": round(best_energy + qubo_offset, 4),
            "status": status,
            "time_elapsed_sec": round(time.time() - start_time, 4)
        })
            
    print("\n--- EVOLUTION FINISHED ---")


    return _save_optimizer_results(lattice, best_energy, history_log, experiment_name, qubo_offset)
 