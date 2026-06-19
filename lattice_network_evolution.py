import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# ============================================================
# 1. LATTICE STRUCTURE (The Neighborhood)
# ============================================================
@dataclass
class LatticeCell:
    value: int = 0
    theta: float = 0.0
    phi: float = 0.0
    fitness: float = 0.0
    occupied: bool = True
    qubit_index: Optional[int] = None

class QuantumLattice2D:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.qubit_map = {}
        self.cells = np.empty((rows, cols), dtype=object)

        q = 0
        for row in range(self.rows):
            for col in range(self.cols):
                self.cells[row, col] = LatticeCell(qubit_index=q)
                # Initialize with a random angle for evolution to work on
                self.cells[row, col].theta = np.random.uniform(0, 2 * np.pi)
                self.qubit_map[q] = (row, col)
                q += 1

    @property
    def n_qubits(self):
        return self.rows * self.cols

    def print_classical_state(self):
        """Prints the grid showing 1 (LED On) or 0 (LED Off) based on Theta"""
        matrix = np.zeros((self.rows, self.cols), dtype=int)
        for i in range(self.rows):
            for j in range(self.cols):
                # If theta is closer to Pi (south pole), it collapses to 1. Otherwise 0.
                theta = self.cells[i, j].theta % (2 * np.pi)
                if np.pi/2 < theta < 3*np.pi/2:
                    matrix[i, j] = 1
                else:
                    matrix[i, j] = 0
        print(matrix)

# ============================================================
# 2. PHYSICS ENGINE (Hamiltonian)
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
# 3. EVALUATION ENGINE (Global Energy)
# ============================================================
def evaluate_lattice_energy(lattice: QuantumLattice2D, simulator: AerSimulator, h_terms: List[HamiltonianTerm], shots=4096) -> float:
    """Evaluates the entire lattice and returns the global energy."""
    qc = QuantumCircuit(lattice.n_qubits)
    
    # Apply each cell's genetics to its corresponding Qubit
    for row in range(lattice.rows):
        for col in range(lattice.cols):
            cell = lattice.cells[row, col]
            if cell.occupied:
                qc.ry(cell.theta, cell.qubit_index)
                
    qc.measure_all()
    
    tqc = transpile(qc, simulator)
    result = simulator.run(tqc, shots=shots).result()
    counts = result.get_counts()
    
    return hamiltonian_expectation_z_only_from_counts(counts, h_terms)

# ============================================================
# 4. EVOLUTIONARY ALGORITHM (Network Hill Climbing)
# ============================================================
def hill_climbing_lattice(lattice: QuantumLattice2D, simulator: AerSimulator, h_terms: List[HamiltonianTerm], generations: int = 100, mutation_rate: float = 0.3):
    print("\n--- STARTING LATTICE EVOLUTION ---")
    
    # 1. Initial Evaluation
    best_energy = evaluate_lattice_energy(lattice, simulator, h_terms)
    print(f"Gen 00 [START] | Global Energy: {best_energy:.4f}")
    lattice.print_classical_state()
    print("-" * 30)
    
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
        if new_energy < best_energy:
            best_energy = new_energy
            print(f"Gen {gen:02d} | Mutated Qubit {target_cell.qubit_index} | Energy: {new_energy:+.4f} | ACCEPTED")
        else:
            # Revert mutation if it didn't help the network
            target_cell.theta = old_theta
            print(f"Gen {gen:02d} | Mutated Qubit {target_cell.qubit_index} | Energy: {new_energy:+.4f} | REJECTED")
            
    print("\n--- EVOLUTION FINISHED ---")
    print(f"Final Minimum Energy (Ground State): {best_energy:.4f}")
    print("Final LED Grid Configuration:")
    lattice.print_classical_state()

# ============================================================
# 5. MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    sim = AerSimulator()
    
    # 1. Create a 2x2 grid (4 LEDs total)
    my_city_grid = QuantumLattice2D(rows=3, cols=3)
    
    # 2. Define the Global Hamiltonian (We want all 4 to reach minimum energy)
    # H = Z_0 + Z_1 + Z_2 + Z_3
    terms = [
        HamiltonianTerm(1.0, "Z", (0,)),
        HamiltonianTerm(1.0, "Z", (1,)),
        HamiltonianTerm(1.0, "Z", (2,)),
        HamiltonianTerm(1.0, "Z", (3,))
    ]
    
    # 3. Run Evolution
    # Expected Ground State Energy for 4 independent terms is -4.00
    hill_climbing_lattice(my_city_grid, sim, terms, generations=150, mutation_rate=0.4)