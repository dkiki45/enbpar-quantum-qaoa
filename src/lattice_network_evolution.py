import random
import copy
import numpy as np
import time
import csv
import os
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector, SparsePauliOp

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

def _h_terms_to_sparse_pauli_op(h_terms: List[HamiltonianTerm], n_qubits: int) -> SparsePauliOp:
    """Converts our HamiltonianTerm list into a qiskit SparsePauliOp so we
    can compute an exact expectation value from a Statevector, without
    shots. Works for both Z (single-qubit) and ZZ (two-qubit) terms, so
    it stays valid once Phase E introduces entangled ZZ terms too."""
    pauli_list = []
    for term in h_terms:
        label = ["I"] * n_qubits
        for q in term.qubits:
            label[q] = "Z"
        # Qiskit's Pauli string convention is little-endian (qubit 0 is the
        # rightmost character), same convention already used elsewhere in
        # this file for bitstrings (see z_term_expectation_from_counts).
        pauli_str = "".join(reversed(label))
        pauli_list.append((pauli_str, term.coefficient))
    return SparsePauliOp.from_list(pauli_list)
 
 
def evaluate_lattice_energy_exact(lattice: QuantumLattice2D, h_terms: List[HamiltonianTerm]) -> float:
    """
    Computes the EXACT <H> from the circuit's statevector -- no shot sampling, so no statistical noise.
    Use this as a diagnostic ground truth to check that the shot-sampled
    energy (evaluate_lattice_energy) is converging to the right value, and
    to sanity-check that epsilon is calibrated correctly
    against the real noise floor.
 
    WARNING: this simulates the full 2^n_qubits statevector classically,
    same scalability limit already documented for AerSimulator in this
    project. Only use it for small lattices (a handful to ~20 qubits) --
    never on the 150-qubit real topology.
    """
    qc = QuantumCircuit(lattice.n_qubits)
    for row in range(lattice.rows):
        for col in range(lattice.cols):
            cell = lattice.cells[row, col]
            if cell.occupied:
                qc.ry(cell.theta, cell.qubit_index)
 
    sv = Statevector.from_instruction(qc)
    op = _h_terms_to_sparse_pauli_op(h_terms, lattice.n_qubits)
    return float(np.real(sv.expectation_value(op)))
 
 
def validate_sampling_vs_statevector(lattice: QuantumLattice2D, simulator: AerSimulator, h_terms: List[HamiltonianTerm], shots: int = 4096, verbose: bool = True):
    """
    Diagnostic helper: compares the shot-sampled energy
    against the exact Statevector energy for the lattice's CURRENT
    configuration. The |diff| returned here is a good empirical reference
    for calibrating `epsilon` in hill_climbing_lattice -- if |diff| is
    consistently larger than the epsilon you're using, mutations may be
    getting accepted/rejected based on noise rather than real improvement.
    """
    sampled = evaluate_lattice_energy(lattice, simulator, h_terms, shots=shots)
    exact = evaluate_lattice_energy_exact(lattice, h_terms)
    diff = abs(sampled - exact)
    if verbose:
        print(f"[VALIDATION] Sampled <H> = {sampled:.4f} | Exact <H> = {exact:.4f} | |diff| = {diff:.4f}")
    return sampled, exact, diff

# ============================================================
# 4. EVOLUTIONARY ALGORITHM (Network Hill Climbing)
# ============================================================
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
            print(f"Gen {gen:02d} | Mutated Qubit {target_cell.qubit_index} | Energy: {new_energy:+.4f} | REJECTED")
 
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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Export CSV
    csv_filename = f"results/{experiment_name}_history_{timestamp}.csv"
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["generation", "qubit_mutated", "energy", "objective_cx", "status", "time_elapsed_sec"])
        writer.writeheader()
        writer.writerows(history_log)
    
    # Optimal Config Export (TXT)
    config_filename = f"results/{experiment_name}_optimal_config_{timestamp}.txt"
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

# ============================================================
# 4b. MULTI-START / RANDOM RESTART
# ============================================================
def multi_start_hill_climbing(
    rows: int,
    cols: int,
    simulator: AerSimulator,
    h_terms: List[HamiltonianTerm],
    generations: int = 100,
    mutation_rate: float = 0.3,
    experiment_name: str = "experiment",
    qubo_offset: float = 0.0,
    epsilon: float = 0.01,
    n_starts: int = 5,
    base_seed: Optional[int] = None,
):
    """
    Runs `n_starts` independent Hill Climbing searches, each one starting from a FRESH,
    randomly re-initialized lattice (new seed), and keeps the best result
    found across all of them.
    """
    best_overall_energy = None
    best_overall_lattice = None
    best_start_index = None
 
    for start_idx in range(n_starts):
        seed = set_reproducible_seed(None if base_seed is None else base_seed + start_idx)
        print(f"\n=== MULTI-START {start_idx + 1}/{n_starts} (seed={seed}) ===")
 
        lattice = QuantumLattice2D(rows=rows, cols=cols)
        energy, result_lattice = hill_climbing_lattice(
            lattice=lattice,
            simulator=simulator,
            h_terms=h_terms,
            generations=generations,
            mutation_rate=mutation_rate,
            experiment_name=f"{experiment_name}_start{start_idx}",
            qubo_offset=qubo_offset,
            epsilon=epsilon,
        )
 
        if best_overall_energy is None or energy < best_overall_energy:
            best_overall_energy = energy
            best_overall_lattice = result_lattice
            best_start_index = start_idx
 
    print(f"\n--- MULTI-START FINISHED: best result from start {best_start_index + 1}/{n_starts} ---")
    print(f"Best Energy (Ising <H>) across all starts: {best_overall_energy:.4f}")
    print(f"Best Objective C(x) across all starts: {best_overall_energy + qubo_offset:.4f}")
 
    return best_overall_energy, best_overall_lattice, best_start_index
 
 
# ============================================================
# 4c. SIMULATED ANNEALING (alternative strategy, Phase C item 3)
# ============================================================
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
 
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
 
    csv_filename = f"results/{experiment_name}_history_{timestamp}.csv"
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["generation", "qubit_mutated", "energy", "objective_cx", "temperature", "status", "time_elapsed_sec"])
        writer.writeheader()
        writer.writerows(history_log)
 
    config_filename = f"results/{experiment_name}_optimal_config_{timestamp}.txt"
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
    # Pulls the real data directly from the CSV processed by the Haversine formula.
    # Get_real_qubo_terms() now formalizes C(x) = -alpha*sum(x_i) + beta*sum(x_i*x_j)
    # explicitly and only then converts it to Ising (Z/ZZ) terms, so it returns the
    # constant `qubo_offset` picked up by the x_i = (1-Z_i)/2 substitution alongside the terms.
    raw_qubo_terms, qubo_offset = get_real_qubo_terms(limit=N_STREETLIGHTS)
 
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
    prado_velho_grid = QuantumLattice2D(rows=1, cols=n_qubits_real)

    # Snapshot the SAME initial random state (thetas) into an independent
    # object BEFORE Hill Climbing mutates prado_velho_grid in place. This is
    # what makes the optional SA comparison below fair: both algorithms
    # start from identical initial conditions, they just evolve separately.
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

    hill_climbing_lattice(
        lattice=prado_velho_grid, 
        simulator=sim, 
        h_terms=h_terms_real, 
        generations=100, 
        mutation_rate=0.4,
        experiment_name=f"prado_velho_{n_qubits_real}_leds",
        qubo_offset=qubo_offset,
    )

    simulated_annealing_lattice(
        lattice=prado_velho_grid_sa,
        simulator=sim,
        h_terms=h_terms_real,
        generations=100,
        mutation_rate=0.4,
        experiment_name=f"prado_velho_{n_qubits_real}_leds_sa",
        qubo_offset=qubo_offset,
    )