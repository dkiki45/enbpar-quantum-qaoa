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
    selected_nodes: list

def integer_to_bits(state, n_vars):
    displayed = format(int(state), f"0{n_vars}b") # q_(n-1)...q_0
    return [int(c) for c in reversed(displayed)] # x_0...x_(n-1)

def decode_distribution(distribution, n_vars, edges, nodes=None, alpha=1.0, beta=2.0):
    output = []
    for state, probability in distribution.items():
        bits = integer_to_bits(state, n_vars)

        selected_geo = []
        if nodes:
            selected_geo = [nodes[i] for i, bit in enumerate(bits) if bit == 1]

        output.append(Candidate(
            format(int(state), f"0{n_vars}b"), 
            bits,
            float(probability), 
            classical_cost(bits, edges, alpha, beta),
            sum(bits), 
            conflicting_edges(bits, edges),
            selected_geo
        ))
    return sorted(output, key=lambda c: (bool(c.violations), c.cost, -c.probability))

def best_feasible_candidate(candidates):
    feasible = [c for c in candidates if not c.violations]
    if not feasible: 
        raise RuntimeError("Nenhuma amostra factivel")
    return feasible[0]


def best_feasible_or_none(candidates):
    """Like best_feasible_candidate, but returns None instead of raising when
    no sampled bitstring is a valid independent set (e.g. the optimizer got stuck)."""
    feasible = [c for c in candidates if not c.violations]
    return feasible[0] if feasible else None

def feasible_probability(candidates):
    """Total sampled probability mass on valid independent sets (0.0 to 1.0)."""
    return float(sum(c.probability for c in candidates if not c.violations))
