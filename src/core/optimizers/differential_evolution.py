import numpy as np
import random
import time
import copy
from typing import List
from qiskit_aer import AerSimulator

from core.lattice_model import QuantumLattice2D
from core.quantum_physics import HamiltonianTerm, evaluate_lattice_energy
from core.optimizers.gradient_descent import _apply_thetas_to_lattice, _save_optimizer_results

def differential_evolution_lattice(lattice: QuantumLattice2D, simulator: AerSimulator, h_terms: List[HamiltonianTerm], pop_size: int = 20, F: float = 0.7, Cr: float = 0.9, generations: int = 50, experiment_name: str = "experiment_de", qubo_offset: float = 0.0):
    """
    Optimizes the quantum matrix using Differential Evolution (DE).
    The trial vector logic v = a + F * (b - c) and the binomial crossover 
    are applied directly to the rotation angles (theta) in Qiskit.
    """
    print("\n--- STARTING DIFFERENTIAL EVOLUTION ---")
    history_log = []
    start_time = time.time()
    n_qubits = lattice.n_qubits

    # Continuous initial population
    V = [np.random.uniform(0, 2*np.pi, n_qubits) for _ in range(pop_size)]
    
    def evaluate_individual(thetas):
        _apply_thetas_to_lattice(lattice, thetas)
        return evaluate_lattice_energy(lattice, simulator, h_terms)

    fitness = [evaluate_individual(v) for v in V]
    best_idx = np.argmin(fitness)
    x_best = copy.deepcopy(V[best_idx])
    f_best = fitness[best_idx]
    
    history_log.append({
        "generation": 0, "energy": round(f_best, 4), "objective_cx": round(f_best + qubo_offset, 4),
        "status": "START", "time_elapsed_sec": 0.0
    })

    for gen in range(1, generations + 1):
        for i in range(pop_size):
            # Chooses distinct a, b, c for the differential equation
            idxs = list(range(pop_size))
            idxs.remove(i)
            a, b, c = random.sample(idxs, 3)

            # Differential mutation: v_mut = a + F * (b - c)
            v_mut = [V[a][k] + F * (V[b][k] - V[c][k]) for k in range(n_qubits)]

            # Binomial crossover to generate the trial-vector
            j_rand = random.randrange(n_qubits)
            v_trial = np.zeros(n_qubits)
            for k in range(n_qubits):
                if random.random() < Cr or k == j_rand:
                    v_trial[k] = v_mut[k]
                else:
                    v_trial[k] = V[i][k]

            # Evaluates Trial
            f_trial = evaluate_individual(v_trial)

            # Selection Generation vs Trial (Minimizing Energy)
            status = "REJECTED"
            if f_trial <= fitness[i]:
                V[i] = v_trial
                fitness[i] = f_trial
                status = "ACCEPTED"

                if f_trial < f_best:
                    f_best = f_trial
                    x_best = copy.deepcopy(v_trial)
                    status = "NEW_BEST"
                    
        print(f"Gen {gen:02d} | Best Energy: {f_best:+.4f} | Pop Evaluated")
        history_log.append({
            "generation": gen, "energy": round(f_best, 4), "objective_cx": round(f_best + qubo_offset, 4),
            "status": "POP_UPDATED", "time_elapsed_sec": round(time.time() - start_time, 4)
        })

    _apply_thetas_to_lattice(lattice, x_best)
    return _save_optimizer_results(lattice, f_best, history_log, experiment_name, qubo_offset)