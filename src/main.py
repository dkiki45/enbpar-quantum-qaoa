import copy
import random
import numpy as np
from typing import Optional
from qiskit_aer import AerSimulator

from core.real_graph_qubo import get_real_qubo_terms
from core.lattice_model import QuantumLattice2D
from core.quantum_physics import HamiltonianTerm
from core.optimizers.hill_climbing import hill_climbing_lattice
from core.optimizers.simulated_annealing import simulated_annealing_lattice

def set_reproducible_seed(seed: Optional[int] = None) -> int:
    if seed is None:
        seed = int(np.random.SeedSequence().entropy % (2**32 - 1))
    random.seed(seed)
    np.random.seed(seed)
    return seed

if __name__ == "__main__":
    
    experiment_seed = set_reproducible_seed(2)
    print(f"Experiment Seed: {experiment_seed}")
 
    sim = AerSimulator()
    sim.set_options(seed_simulator=experiment_seed)
 
    # ========================================================
    # GLOBAL EXPERIMENT PARAMETERS
    # ========================================================
    N_STREETLIGHTS = 15
    GENERATIONS = 300
    MUTATION_RATE = 0.4
    
    print("\n[PIPELINE] 1. Extracting real topology from IPPUC (Prado Velho)...")
    raw_qubo_terms, qubo_offset = get_real_qubo_terms(limit=N_STREETLIGHTS)
 
    if not raw_qubo_terms:
        print("Pipeline aborted: Failed to extract QUBO terms.")
        exit()
 
    n_qubits_real = len({q for term in raw_qubo_terms if len(term["qubits"]) == 1 for q in term["qubits"]})
    if n_qubits_real != N_STREETLIGHTS:
        print(f"[WARNING] N_STREETLIGHTS={N_STREETLIGHTS} but data returned "
              f"{n_qubits_real} fixtures. Using {n_qubits_real}.")
    
    print(f"\n[PIPELINE] 2. Mapping real streetlights to the Qiskit Lattice...")
    prado_velho_grid = QuantumLattice2D(rows=1, cols=n_qubits_real)

    # Creates a clean, identical copy so Simulated Annealing starts from the same initial state
    prado_velho_grid_sa = copy.deepcopy(prado_velho_grid)
    
    print("\n[PIPELINE] 3. Converting dictionary rules to HamiltonianTerm objects...")
    h_terms_real = []
    for term in raw_qubo_terms:
        h_terms_real.append(
            HamiltonianTerm(
                coefficient=term["coefficient"],
                pauli=term["pauli"],
                qubits=term["qubits"]
            )
        )
        
    print(f"Terms successfully mapped: {len(h_terms_real)} mathematical constraints (Z and ZZ).")
    
    print("\n[PIPELINE] 4. Starting combinatorial optimization...")

    # Runs the Greedy optimizer (Hill Climbing)
    hill_climbing_lattice(
        lattice=prado_velho_grid, 
        simulator=sim, 
        h_terms=h_terms_real, 
        generations=GENERATIONS, 
        mutation_rate=MUTATION_RATE,
        experiment_name=f"prado_velho_{n_qubits_real}_leds",
        qubo_offset=qubo_offset,
    )

    # Runs the Thermal optimizer (Simulated Annealing)
    simulated_annealing_lattice(
        lattice=prado_velho_grid_sa,
        simulator=sim,
        h_terms=h_terms_real,
        generations=GENERATIONS,
        mutation_rate=MUTATION_RATE,
        experiment_name=f"prado_velho_{n_qubits_real}_leds_sa",
        qubo_offset=qubo_offset,
    )