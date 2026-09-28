from itertools import product
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from core.qubo_formalization import classical_cost

def solve_exact_bruteforce(n_vars, edges, alpha=1.0, beta=2.0, max_vars=100):
    if n_vars > max_vars: raise ValueError("Instance for brute force")
    best_bits, best_cost = None, float("inf")
    for bits in product((0,1), repeat=n_vars):
        cost = classical_cost(bits, edges, alpha, beta)
        if cost < best_cost: best_bits, best_cost = list(bits), cost
    return best_bits, best_cost


def solve_exact_ilp(n_vars, edges, alpha=1.0, beta=2.0):
    """
    Exact Maximum Independent Set via Integer Linear Programming (scipy/HiGHS).
        maximize  sum(x_i)
        s.t.      x_i + x_j <= 1  for every edge (i, j),  x_i in {0, 1}
    Branch-and-bound proves optimality, so the result is as exact as brute force,
    but it runs in milliseconds instead of 2^n evaluations.
    Since beta > alpha, the QUBO optimum is exactly the MIS, so the returned cost
    matches solve_exact_bruteforce. With ties, the bits may be a different optimal set.
    """
    A = np.zeros((max(len(edges), 1), n_vars))
    for k, (i, j) in enumerate(edges):
        A[k, i] = A[k, j] = 1
    res = milp(
        c=-np.ones(n_vars),
        constraints=[LinearConstraint(A, -np.inf, 1)],
        integrality=np.ones(n_vars),
        bounds=Bounds(0, 1),
    )
    if not res.success:
        raise RuntimeError(f"ILP solver failed: {res.message}")
    bits = [int(round(v)) for v in res.x]
    return bits, classical_cost(bits, edges, alpha, beta)
