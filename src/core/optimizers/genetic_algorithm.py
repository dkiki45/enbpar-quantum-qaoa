import numpy as np
import random
import time
import copy
from typing import List
from qiskit_aer import AerSimulator

from core.lattice_model import QuantumLattice2D
from core.quantum_physics import HamiltonianTerm, evaluate_lattice_energy
from core.optimizers.gradient_descent import _apply_thetas_to_lattice, _save_optimizer_results

def genetic_algorithm_lattice(lattice: QuantumLattice2D, simulator: AerSimulator, h_terms: List[HamiltonianTerm], pop_size: int = 20, generations: int = 50, pc: float = 0.8, pm: float = 0.1, elitism: int = 2, experiment_name: str = "experiment_ga", qubo_offset: float = 0.0):
    """
    Optimizes the quantum matrix using Genetic Algorithms (GA).
    Adapted from binary combinatorial problems to continuous angles (Qubits).
    """
    print("\n--- STARTING GENETIC ALGORITHM ---")
    history_log = []
    start_time = time.time()
    n_qubits = lattice.n_qubits

    # Population Initialization (Random angles between 0 and 2*pi)
    population = [np.random.uniform(0, 2*np.pi, n_qubits) for _ in range(pop_size)]
    
    def evaluate_individual(thetas):
        _apply_thetas_to_lattice(lattice, thetas)
        return evaluate_lattice_energy(lattice, simulator, h_terms)

    # Evaluates Initial Population
    fitness = [evaluate_individual(ind) for ind in population] # We want to MINIMIZE the energy
    
    best_idx = np.argmin(fitness)
    global_best_thetas = copy.deepcopy(population[best_idx])
    global_best_energy = fitness[best_idx]
    
    history_log.append({
        "generation": 0, "energy": round(global_best_energy, 4), "objective_cx": round(global_best_energy + qubo_offset, 4),
        "status": "START", "time_elapsed_sec": 0.0
    })

    def tournament(pop, fit, k=3):
        # Tournament selection adapted for energy minimization
        candidates = random.sample(range(len(pop)), k)
        best_c = min(candidates, key=lambda idx: fit[idx])
        return pop[best_c]

    for gen in range(1, generations + 1):
        new_pop = []
        
        # Elitism: Keeps the best individuals
        elite_indices = np.argsort(fitness)[:elitism]
        for idx in elite_indices:
            new_pop.append(copy.deepcopy(population[idx]))

        # Reproduction
        while len(new_pop) < pop_size:
            p1 = tournament(population, fitness)
            p2 = tournament(population, fitness)
            
            # Uniform Crossover (based on the reference)
            c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)
            if random.random() < pc:
                for i in range(n_qubits):
                    if random.random() < 0.5:
                        c1[i], c2[i] = p2[i], p1[i]
            
            # Mutation (Gaussian Noise on the angles)
            for i in range(n_qubits):
                if random.random() < pm:
                    c1[i] += np.random.normal(0, 0.3)
                if random.random() < pm:
                    c2[i] += np.random.normal(0, 0.3)
                    
            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)

        population = new_pop
        fitness = [evaluate_individual(ind) for ind in population]
        
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < global_best_energy:
            global_best_energy = fitness[current_best_idx]
            global_best_thetas = copy.deepcopy(population[current_best_idx])
            status = "NEW_ELITE"
        else:
            status = "EVOLVING"

        print(f"Gen {gen:02d} | Best Energy: {global_best_energy:+.4f} | {status}")
        history_log.append({
            "generation": gen, "energy": round(global_best_energy, 4), "objective_cx": round(global_best_energy + qubo_offset, 4),
            "status": status, "time_elapsed_sec": round(time.time() - start_time, 4)
        })

    _apply_thetas_to_lattice(lattice, global_best_thetas)
    return _save_optimizer_results(lattice, global_best_energy, history_log, experiment_name, qubo_offset)