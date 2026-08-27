from core.qubo_formalization import build_mis_qubo, qubo_to_ising
from core.qaoa_solver import run_qaoa
from core.solution_decoder import decode_distribution, best_feasible_candidate

def test_small_path():
    n, edges = 3, [(0, 1), (1, 2)]
    lin, quad, off = build_mis_qubo(n, edges)
    model = qubo_to_ising(n, lin, quad, off)
    
    result = run_qaoa(model, reps=1, shots=8192, seed=7, maxiter=150)
    best = best_feasible_candidate(decode_distribution(result.distribution, n, edges))
    
    assert not best.violations
    assert best.selected == 2 and best.cost == -2.0