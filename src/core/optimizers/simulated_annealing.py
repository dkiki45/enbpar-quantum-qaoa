import numpy as np
import time
import csv
import os
from datetime import datetime
from typing import List
from qiskit_aer import AerSimulator

from core.lattice_model import QuantumLattice2D
from core.quantum_physics import HamiltonianTerm, evaluate_lattice_energy

def simulated_annealing_lattice(
    lattice: QuantumLattice2D,
    simulator: AerSimulator,
    h_terms: List[HamiltonianTerm],
    generations: int = 100,
    mutation_rate: float = 0.3,
    experiment_name: str = "experiment_sa",
    qubo_offset: float = 0.0,
    initial_temperature: float = 2.0,
    cooling_rate: float = 0.95,
    min_temperature: float = 0.01,
):
    """
    Difference from Hill Climbing: instead of only accepting strict
    improvements, SA also accepts WORSE mutations with probability
    exp(-delta_E / T), where delta_E = new_energy - current_energy > 0 and
    T is the current temperature. This lets the search escape local minima
    early on (when T is high) while behaving more and more like greedy
    Hill Climbing as T cools down.
 
    Defaults: initial_temperature=2.0, cooling_rate=0.95 (exponential decay
    T = T0 * cooling_rate**generation), min_temperature=0.01 as a floor so
    the acceptance probability calculation never divides by (near) zero.
    These are reasonable starting points for the energy scale seen so far
    (|<H>| roughly in the 0-n_qubits range); retune if the lattice or
    Hamiltonian coefficients change significantly.
    """
    print("\n--- STARTING SIMULATED ANNEALING ---")
 
    history_log = []
    start_time = time.time()
 
    current_energy = evaluate_lattice_energy(lattice, simulator, h_terms)
    best_energy = current_energy
    temperature = initial_temperature
 
    print(f"Gen 00 [START] | Energy: {current_energy:.4f} | T: {temperature:.4f}")
    history_log.append({
        "generation": 0,
        "qubit_mutated": "N/A",
        "energy": round(current_energy, 4),
        "objective_cx": round(current_energy + qubo_offset, 4),
        "temperature": round(temperature, 4),
        "status": "START",
        "time_elapsed_sec": 0.0,
    })
 
    for gen in range(1, generations + 1):
        temperature = max(min_temperature, initial_temperature * (cooling_rate ** gen))
 
        rand_row = np.random.randint(0, lattice.rows)
        rand_col = np.random.randint(0, lattice.cols)
        target_cell = lattice.cells[rand_row, rand_col]
 
        old_theta = target_cell.theta
        mutation = np.random.normal(0, mutation_rate)
        target_cell.theta += mutation
 
        new_energy = evaluate_lattice_energy(lattice, simulator, h_terms)
        delta_e = new_energy - current_energy
 
        if delta_e < 0:
            # Strict improvement: always accept, same as Hill Climbing.
            current_energy = new_energy
            status = "ACCEPTED (improvement)"
        else:
            # Worse mutation: accept anyway with probability exp(-delta_E / T).
            acceptance_prob = np.exp(-delta_e / temperature)
            if np.random.random() < acceptance_prob:
                current_energy = new_energy
                status = f"ACCEPTED (escape, p={acceptance_prob:.3f})"
            else:
                target_cell.theta = old_theta
                status = "REJECTED"
 
        if current_energy < best_energy:
            best_energy = current_energy
 
        print(f"Gen {gen:02d} | Mutated Qubit {target_cell.qubit_index} | Energy: {new_energy:+.4f} | T: {temperature:.4f} | {status}")
 
        history_log.append({
            "generation": gen,
            "qubit_mutated": target_cell.qubit_index,
            "energy": round(current_energy, 4),
            "objective_cx": round(current_energy + qubo_offset, 4),
            "temperature": round(temperature, 4),
            "status": status,
            "time_elapsed_sec": round(time.time() - start_time, 4),
        })
 
    print("\n--- SIMULATED ANNEALING FINISHED ---")
 
    src_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    results_dir = os.path.join(src_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
 
    csv_filename = os.path.join(results_dir, f"{experiment_name}_history_{timestamp}.csv")
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["generation", "qubit_mutated", "energy", "objective_cx", "temperature", "status", "time_elapsed_sec"])
        writer.writeheader()
        writer.writerows(history_log)
 
    config_filename = os.path.join(results_dir, f"{experiment_name}_optimal_config_{timestamp}.txt")
    with open(config_filename, "w") as file:
        file.write(f"Final Minimum Energy (Ising <H>): {best_energy:.4f}\n")
        file.write(f"Final Objective C(x): {best_energy + qubo_offset:.4f}\n")
        file.write("Final LED Grid Configuration (1 = ON, 0 = OFF):\n")
        for i in range(lattice.rows):
            row_str = []
            for j in range(lattice.cols):
                theta = lattice.cells[i, j].theta % (2 * np.pi)
                val = 1 if np.pi / 2 < theta < 3 * np.pi / 2 else 0
                row_str.append(str(val))
            file.write(" ".join(row_str) + "\n")
 
    print(f"-> Histórico completo exportado para: {csv_filename}")
    print(f"-> Configuração ótima exportada para: {config_filename}")
    print(f"Final Minimum Energy (Ising <H>): {best_energy:.4f}")
    print(f"Final Objective C(x): {best_energy + qubo_offset:.4f}")
    lattice.print_classical_state()
 
    return best_energy, lattice