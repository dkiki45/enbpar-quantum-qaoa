import numpy as np
import time
import csv
import os
from datetime import datetime
from typing import List
from qiskit_aer import AerSimulator

from core.lattice_model import QuantumLattice2D
from core.quantum_physics import HamiltonianTerm, evaluate_lattice_energy

def hill_climbing_lattice(lattice: QuantumLattice2D, simulator: AerSimulator, h_terms: List[HamiltonianTerm], generations: int = 100, mutation_rate: float = 0.3, experiment_name: str = "experiment", qubo_offset: float = 0.0, epsilon: float = 0.01):
    """
    qubo_offset: constant returned by qubo_to_ising(). The Ising
    energy (<H>) by itself is NOT equal to the original classical C(x) --
    it equals C(x) - offset. We add the offset back only for reporting
    purposes (the search/acceptance step keeps using the Ising energy,
    since a constant shift doesn't change where the minimum sits).
 
    epsilon: energy tolerance. Energy is measured by
    shot sampling, so tiny improvements can be pure statistical noise
    rather than a real signal. We only accept a mutation when it beats
    the current best by more than epsilon: new_energy < best_energy - epsilon.
    Default 0.01 was chosen because typical shot noise on these small
    circuits (4096 shots) sits around that order of magnitude; tune it
    if you change shot count or circuit size.
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
 
    # ==========================================
    # DATA PERSISTENCE
    # ==========================================
    src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    results_dir = os.path.join(src_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Export CSV
    csv_filename = os.path.join(results_dir, f"{experiment_name}_history_{timestamp}.csv")
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["generation", "qubit_mutated", "energy", "objective_cx", "status", "time_elapsed_sec"])
        writer.writeheader()
        writer.writerows(history_log)
    
    # Optimal Config Export (TXT)
    config_filename = os.path.join(results_dir, f"{experiment_name}_optimal_config_{timestamp}.txt")
    with open(config_filename, 'w') as file:
        file.write(f"Final Minimum Energy (Ising <H>): {best_energy:.4f}\n")
        file.write(f"Final Objective C(x): {best_energy + qubo_offset:.4f}\n")
        file.write("Final LED Grid Configuration (1 = ON, 0 = OFF):\n")
        
        # Extracts the classic matrix and saves it.
        for i in range(lattice.rows):
            row_str = []
            for j in range(lattice.cols):
                theta = lattice.cells[i, j].theta % (2 * np.pi)
                val = 1 if np.pi/2 < theta < 3*np.pi/2 else 0
                row_str.append(str(val))
            file.write(" ".join(row_str) + "\n")
            
    print(f"-> Histórico completo exportado para: {csv_filename}")
    print(f"-> Configuração ótima exportada para: {config_filename}")
 
    print(f"Final Minimum Energy (Ground State, Ising <H>): {best_energy:.4f}")
    print(f"Final Objective C(x): {best_energy + qubo_offset:.4f}")
    print("Final LED Grid Configuration:")
    lattice.print_classical_state()
 
    return best_energy, lattice
 