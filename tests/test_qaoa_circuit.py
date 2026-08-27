from core.qubo_formalization import build_mis_qubo, qubo_to_ising
from core.qaoa_circuit import build_qaoa_ansatz

def test_qaoa_structure():
    lin, quad, off = build_mis_qubo(3, [(0,1)])
    qc = build_qaoa_ansatz(qubo_to_ising(3, lin, quad, off), reps=2)
    
    ops = qc.count_ops()
    
    assert qc.num_parameters == 4
    assert ops.get("h", 0) == 3 and ops.get("rzz", 0) >= 2 and ops.get("rx", 0) == 6