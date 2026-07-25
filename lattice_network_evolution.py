import random
import numpy as np
import time
import csv
import os
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

def set_reproducible_seed(seed: Optional[int] = None) -> int:
    if seed is None:
        seed = int(np.random.SeedSequence().entropy % (2**32 - 1))
    random.seed(seed)
    np.random.seed(seed)
    return seed

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

def build_z_terms_for_lattice(lattice: "QuantumLattice2D", coefficient: float = 1.0) -> List[HamiltonianTerm]:
    terms = []
    for row in range(lattice.rows):
        for col in range(lattice.cols):
            cell = lattice.cells[row, col]
            if cell.occupied:
                terms.append(HamiltonianTerm(coefficient, "Z", (cell.qubit_index,)))
    return terms

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
def hill_climbing_lattice(lattice: QuantumLattice2D, simulator: AerSimulator, h_terms: List[HamiltonianTerm], generations: int = 100, mutation_rate: float = 0.3, experiment_name: str = "experiment"):
    print("\n--- STARTING LATTICE EVOLUTION ---")
    
    history_log = []
    start_time = time.time()

    # 1. Initial Evaluation
    best_energy = evaluate_lattice_energy(lattice, simulator, h_terms)
    print(f"Gen 00 [START] | Global Energy: {best_energy:.4f}")

    history_log.append({
        "generation": 0,
        "qubit_mutated": "N/A",
        "energy": round(best_energy, 4),
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
        if new_energy < best_energy:
            best_energy = new_energy
            status = "ACCEPTED"
            print(f"Gen {gen:02d} | Mutated Qubit {target_cell.qubit_index} | Energy: {new_energy:+.4f} | ACCEPTED")
        else:
            # Revert mutation if it didn't help the network
            target_cell.theta = old_theta
            status = "REJECTED"
            print(f"Gen {gen:02d} | Mutated Qubit {target_cell.qubit_index} | Energy: {new_energy:+.4f} | REJECTED")

        history_log.append({
            "generation": gen,
            "qubit_mutated": target_cell.qubit_index,
            "energy": round(best_energy, 4),
            "status": status,
            "time_elapsed_sec": round(time.time() - start_time, 4)
        })
            
    print("\n--- EVOLUTION FINISHED ---")

    # ==========================================
    # DATA PERSISTENCE
    # ==========================================
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Export CSV
    csv_filename = f"results/{experiment_name}_history_{timestamp}.csv"
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["generation", "qubit_mutated", "energy", "status", "time_elapsed_sec"])
        writer.writeheader()
        writer.writerows(history_log)
    
    # Optimal Config Export (TXT)
    config_filename = f"results/{experiment_name}_optimal_config_{timestamp}.txt"
    with open(config_filename, 'w') as file:
        file.write(f"Final Minimum Energy: {best_energy:.4f}\n")
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

    print(f"Final Minimum Energy (Ground State): {best_energy:.4f}")
    print("Final LED Grid Configuration:")
    lattice.print_classical_state()

# ============================================================
# 5. MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    # Imports the function from the real_graph_qubo.py file
    from real_graph_qubo import get_real_qubo_terms 
    
    experiment_seed = set_reproducible_seed(1)
    print(f"Experiment Seed: {experiment_seed}")

    sim = AerSimulator()
    sim.set_options(seed_simulator=experiment_seed)

    N_STREETLIGHTS = 3

    # =====================================================================
    # SCIENTIFIC NOTE ON SCALABILITY, ENTANGLEMENT, AND QAOA
    # =====================================================================
    # Currently (in the Hill Climbing phase), the circuit evaluates states 
    # through independent rotations (RY) without entanglement. This would 
    # theoretically allow the use of Tensor Network-based simulators 
    # (e.g., method='matrix_product_state') to bypass hardware limits and 
    # run 150 qubits locally.
    # 
    # HOWEVER, we opted to strictly limit the local simulation to 20 qubits.
    # Reason: The next architectural phase of the project (QAOA) will 
    # introduce dense entanglement (CX/ZZ gates). When this happens, the 
    # state space complexity will revert to 2^N, making 150 qubits 
    # computationally intractable for any classical RAM. Limiting it to 
    # 20 qubits ensures a classical testing environment that accurately 
    # reflects the physical memory limitations that QAOA will demand, 
    # thus preserving the integrity of our baseline benchmark.
    
    print("\n[PIPELINE] 1. Extracting real topology from IPPUC (Prado Velho)...")
    # Pulls the real data directly from the CSV processed by the Haversine formula
    raw_qubo_terms = get_real_qubo_terms(limit=N_STREETLIGHTS)
    
    if not raw_qubo_terms:
        print("Pipeline aborted: Failed to extract QUBO terms.")
        exit()

    n_qubits_real = len({q for term in raw_qubo_terms if len(term["qubits"]) == 1 for q in term["qubits"]})
    if n_qubits_real != N_STREETLIGHTS:
        print(f"[WARNING] N_STREETLIGHTS={N_STREETLIGHTS} but data returned "
              f"{n_qubits_real} fixtures (linear terms). Using {n_qubits_real}.")
    
    print(f"\n[PIPELINE] 2. Mapping real streetlights to the Qiskit Lattice...")
    # Since the positions are now dictated by GPS (real distance),
    # the visual grid doesn't need to be square. We create 1 straight row of 150 cells.
    prado_velho_grid = QuantumLattice2D(rows=3, cols=n_qubits_real)
    
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
    # We pass the 150-cell grid and the real-world terms to the Hill Climbing engine
    hill_climbing_lattice(
        lattice=prado_velho_grid, 
        simulator=sim, 
        h_terms=h_terms_real, 
        generations=300, # Increased to give the algorithm time to explore 150 variables
        mutation_rate=0.4,
        experiment_name=f"prado_velho_{n_qubits_real}_leds"
    )