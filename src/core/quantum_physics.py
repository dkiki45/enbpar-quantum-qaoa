import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector, SparsePauliOp

from core.lattice_model import QuantumLattice2D

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

def _h_terms_to_sparse_pauli_op(h_terms: List[HamiltonianTerm], n_qubits: int) -> SparsePauliOp:
    """Converts our HamiltonianTerm list into a qiskit SparsePauliOp so we
    can compute an exact expectation value from a Statevector, without
    shots. Works for both Z (single-qubit) and ZZ (two-qubit) terms."""
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
    qc = QuantumCircuit(lattice.n_qubits)
    for row in range(lattice.rows):
        for col in range(lattice.cols):
            cell = lattice.cells[row, col]
            if cell.occupied:
                qc.ry(cell.theta, cell.qubit_index)
 
    sv = Statevector.from_instruction(qc)
    op = _h_terms_to_sparse_pauli_op(h_terms, lattice.n_qubits)
    return float(np.real(sv.expectation_value(op)))