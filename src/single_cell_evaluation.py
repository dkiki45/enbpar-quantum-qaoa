import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# ============================================================
# 1. LATTICE STRUCTURE
# ============================================================
@dataclass
class LatticeCell:
    value: int = 0
    theta: float = 0.0
    phi: float = 0.0
    fitness: float = 0.0
    occupied: bool = True
    qubit_index: Optional[int] = None

    def delete_individual(self):
        self.occupied = False
        self.fitness = 0.0
        self.value = 0
        self.theta = 0.0
        self.phi = 0.0

# ============================================================
# 2. HAMILTONIAN STRUCTURE
# ============================================================
@dataclass
class HamiltonianTerm:
    coefficient: float
    pauli: str
    qubits: Tuple[int, ...]

def z_term_expectation_from_counts(counts: Dict[str, int], qubits: Tuple[int, ...]) -> float:
    shots = sum(counts.values())
    exp_val = 0.0
    for bitstring, count in counts.items():
        prob = count / shots
        product = 1
        for q in qubits:
            bit = bitstring[-1 - q]
            z = +1 if bit == '0' else -1
            product *= z
        exp_val += prob * product
    return exp_val

def hamiltonian_expectation_z_only_from_counts(counts: Dict[str, int], terms: List[HamiltonianTerm]) -> float:
    total = 0.0
    for term in terms:
        exp_val = z_term_expectation_from_counts(counts, term.qubits)
        total += term.coefficient * exp_val
    return total

# ============================================================
# 3. THE EVALUATION ENGINE (Updated for silent execution)
# ============================================================
def evaluate_cell_fitness(cell: LatticeCell, simulator: AerSimulator, shots: int = 4096):
    """
    Evaluates the cell and updates its fitness silently.
    (Removed the print statement so it doesn't spam the terminal during evolution)
    """
    qc = QuantumCircuit(1)
    qc.ry(cell.theta, 0) 
    qc.measure_all()
    
    tqc = transpile(qc, simulator)
    result = simulator.run(tqc, shots=shots).result()
    counts = result.get_counts()
    
    hamiltonian_terms = [HamiltonianTerm(coefficient=1.0, pauli="Z", qubits=(0,))]
    
    energy = hamiltonian_expectation_z_only_from_counts(counts, hamiltonian_terms)
    cell.fitness = energy

# ============================================================
# 4. EVOLUTIONARY ALGORITHM (Hill Climbing)
# ============================================================
def hill_climbing_single_cell(cell: LatticeCell, simulator: AerSimulator, generations: int = 30, mutation_rate: float = 0.2):
    """
    Evolves a single cell towards the minimum energy (Ground State).
    - generations: Number of mutation attempts.
    - mutation_rate: The maximum size of the angle change per step.
    """
    print(f"\n--- STARTING EVOLUTION (Qubit {cell.qubit_index}) ---")
    
    # Initial Evaluation
    evaluate_cell_fitness(cell, simulator)
    best_fitness = cell.fitness
    
    print(f"Gen 0 [START]: Theta = {cell.theta:.4f} rad | Energy = {best_fitness:.4f}")
    
    for gen in range(1, generations + 1):
        # 1. Save the current best state before mutating
        old_theta = cell.theta
        
        # 2. Mutate: Add a random Gaussian noise to the angle
        mutation = np.random.normal(0, mutation_rate)
        cell.theta += mutation
        
        # Optional: Keep theta within 0 to 2*pi for cleaner logs
        cell.theta = cell.theta % (2 * np.pi)
        
        # 3. Evaluate the mutated state
        evaluate_cell_fitness(cell, simulator)
        new_fitness = cell.fitness
        
        # 4. Selection (Minimization)
        if new_fitness < best_fitness:
            # Accepted: The mutation reduced the energy
            best_fitness = new_fitness
            print(f"Gen {gen:02d}: Mutated ({mutation:+.4f}) -> Theta = {cell.theta:.4f} | Energy = {new_fitness:+.4f} | ACCEPTED (Improved)")
        else:
            # Rejected: The mutation increased the energy, revert to old state
            cell.theta = old_theta
            cell.fitness = best_fitness
            print(f"Gen {gen:02d}: Mutated ({mutation:+.4f}) -> Theta = {(old_theta + mutation) % (2*np.pi):.4f} | Energy = {new_fitness:+.4f} | REJECTED (Reverted)")

    print("--- EVOLUTION FINISHED ---")
    print(f"Final Result: Optimal Theta = {cell.theta:.4f} rad | Minimum Energy = {best_fitness:.4f}\n")

# ============================================================
# 5. MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    
    sim = AerSimulator()
    
    # We create a cell with a completely random starting angle
    random_start_angle = np.random.uniform(0, 2 * np.pi)
    evolving_cell = LatticeCell(qubit_index=0, theta=random_start_angle)
    
    # Run the Hill Climbing algorithm for 40 generations
    hill_climbing_single_cell(evolving_cell, sim, generations=40, mutation_rate=0.3)