import numpy as np
from qiskit.quantum_info import Statevector
from core.qubo_formalization import *

def test_qubo_equals_ising_for_all_states():
    n, edges = 3, [(0,1), (1,2)]
    lin, quad, off = build_mis_qubo(n, edges)
    model = qubo_to_ising(n, lin, quad, off)
    
    for state in range(2**n):
        displayed = format(state, f"0{n}b")
        bits = [int(c) for c in reversed(displayed)]
        
        energy = float(np.real(
            Statevector.from_label(displayed).expectation_value(model.to_sparse_pauli_op())
        )) + model.offset
        
        assert np.isclose(energy, classical_cost(bits, edges))