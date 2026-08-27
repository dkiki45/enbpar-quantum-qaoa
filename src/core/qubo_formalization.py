from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple
import numpy as np
from qiskit.quantum_info import SparsePauliOp

Edge = Tuple[int, int]

@dataclass(frozen=True)
class IsingModel:
    n_vars: int
    h: Dict[int, float]
    j: Dict[Edge, float]
    offset: float

    def to_sparse_pauli_op(self):
        terms = []
        for i, coef in self.h.items():
            label = ["I"] * self.n_vars
            label[i] = "Z"
            terms.append(("".join(reversed(label)), coef))
            
        for (i, j), coef in self.j.items():
            label = ["I"] * self.n_vars
            label[i] = "Z"
            label[j] = "Z"
            terms.append(("".join(reversed(label)), coef))
            
        return SparsePauliOp.from_list(terms or [("I" * self.n_vars, 0.0)]).simplify()


def normalize_edges(n_vars: int, edges: Iterable[Edge]) -> List[Edge]:
    result = set()
    for i, j in edges:
        if i == j: 
            raise ValueError(f"Self-loop: {(i,j)}")
        if not (0 <= i < n_vars and 0 <= j < n_vars): 
            raise IndexError((i,j))
        result.add((min(i,j), max(i,j)))
    return sorted(result)


def build_mis_qubo(n_vars, edges, alpha=1.0, beta=2.0):
    if alpha <= 0 or beta <= alpha: 
        raise ValueError("Exigir beta > alpha > 0")
    edges = normalize_edges(n_vars, edges)
    return ({i: -float(alpha) for i in range(n_vars)},
            {edge: float(beta) for edge in edges}, 0.0)


def qubo_to_ising(n_vars, linear, quadratic, offset0=0.0):
    h, j, offset = {i: 0.0 for i in range(n_vars)}, {}, float(offset0)
    
    for i, a in linear.items():
        offset += a / 2
        h[i] -= a / 2
        
    for (i, k), b in quadratic.items():
        offset += b / 4
        h[i] -= b / 4
        h[k] -= b / 4
        j[(i,k)] = j.get((i,k), 0.0) + b / 4
        
    return IsingModel(n_vars, h, j, offset)


def classical_cost(bits: Sequence[int], edges, alpha=1.0, beta=2.0):
    x = np.asarray(bits, dtype=int)
    if not np.isin(x, [0, 1]).all(): 
        raise ValueError("Vetor nao binario")
    return float(-alpha * x.sum() + beta * sum(x[i] * x[j] for i, j in edges))


def conflicting_edges(bits, edges):
    return [(i,j) for i, j in edges if bits[i] == bits[j] == 1]