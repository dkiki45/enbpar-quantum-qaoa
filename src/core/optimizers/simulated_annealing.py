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
    Unlike the greedy approach, this algorithm can temporarily accept energy-increasing 
    (worse) mutations with a probability of exp(-delta_E / T) to escape local minima. 
    As the temperature (T) exponentially decays over the generations, it gradually 
    transitions into a strict Hill Climbing behavior to lock into the ground state.
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


    return _save_optimizer_results(lattice, best_energy, history_log, experiment_name, qubo_offset)