from dataclasses import dataclass
from core.qubo_formalization import classical_cost, conflicting_edges

@dataclass(frozen=True)
class Candidate:
    bitstring: str
    bits: list
    probability: float
    cost: float
    selected: int
    violations: list

def integer_to_bits(state, n_vars):
    displayed = format(int(state), f"0{n_vars}b") # q_(n-1)...q_0
    return [int(c) for c in reversed(displayed)] # x_0...x_(n-1)

def decode_distribution(distribution, n_vars, edges, alpha=1.0, beta=2.0):
    output = []
    for state, probability in distribution.items():
        bits = integer_to_bits(state, n_vars)
        output.append(Candidate(
            format(int(state), f"0{n_vars}b"), 
            bits,
            float(probability), 
            classical_cost(bits, edges, alpha, beta),
            sum(bits), 
            conflicting_edges(bits, edges)
        ))
    return sorted(output, key=lambda c: (bool(c.violations), c.cost, -c.probability))

def best_feasible_candidate(candidates):
    feasible = [c for c in candidates if not c.violations]
    if not feasible: 
        raise RuntimeError("Nenhuma amostra factivel")
    return feasible[0]