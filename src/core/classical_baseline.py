from itertools import product
from core.qubo_formalization import classical_cost

def solve_exact_bruteforce(n_vars, edges, alpha=1.0, beta=2.0, max_vars=25):
    if n_vars > max_vars: raise ValueError("Instancia grande para forca bruta")
    best_bits, best_cost = None, float("inf")
    for bits in product((0,1), repeat=n_vars):
        cost = classical_cost(bits, edges, alpha, beta)
        if cost < best_cost: best_bits, best_cost = list(bits), cost
    return best_bits, best_cost