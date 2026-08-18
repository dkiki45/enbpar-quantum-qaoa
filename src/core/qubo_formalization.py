# ============================================================
# MATHEMATICAL FORMALIZATION (QUBO -> ISING)
# ============================================================

from typing import Dict, List, Tuple
from dataclasses import dataclass

# Default weights for the classical objective function
ALPHA_COVERAGE = 1.0
BETA_REDUNDANCY = 2.0


@dataclass
class HamiltonianTerm:
    coefficient: float
    pauli: str
    qubits: Tuple[int, ...]


# ============================================================
# 1. CLASSICAL OBJECTIVE FUNCTION (QUBO)
# ============================================================
def build_qubo(
    n_vars: int,
    edges: List[Tuple[int, int]],
    alpha: float = ALPHA_COVERAGE,
    beta: float = BETA_REDUNDANCY,
) -> Tuple[Dict[int, float], Dict[Tuple[int, int], float], float]:
    """
    Explicitly builds C(x) = -alpha * sum_i x_i + beta * sum_{(i,j) in E} x_i x_j

    Returns:
        linear:    {i: a_i}       coefficient of x_i
        quadratic: {(i,j): b_ij}  coefficient of x_i * x_j  (i < j)
        offset0:   classical constant (0.0 here, but kept for generality in
                   case constant terms need to be added later)
    """
    linear: Dict[int, float] = {i: -alpha for i in range(n_vars)}
    quadratic: Dict[Tuple[int, int], float] = {}
    for (i, j) in edges:
        lo, hi = (i, j) if i < j else (j, i)
        quadratic[(lo, hi)] = quadratic.get((lo, hi), 0.0) + beta
    offset0 = 0.0
    return linear, quadratic, offset0


# ============================================================
# 2. QUBO -> ISING CONVERSION 
# ============================================================
def qubo_to_ising(
    linear: Dict[int, float],
    quadratic: Dict[Tuple[int, int], float],
    offset0: float = 0.0,
) -> Tuple[List[HamiltonianTerm], float]:
    """
    Formally applies the substitution  x_i = (1 - Z_i) / 2  to every QUBO
    term and returns the Ising terms (Z and ZZ) together with the global
    constant.
    """
    z_coeff: Dict[int, float] = {}
    offset = offset0

    # linear terms a_i * x_i
    for i, a_i in linear.items():
        offset += a_i / 2.0
        z_coeff[i] = z_coeff.get(i, 0.0) - a_i / 2.0

    # quadratic terms b_ij * x_i * x_j
    zz_terms: List[HamiltonianTerm] = []
    for (i, j), b_ij in quadratic.items():
        offset += b_ij / 4.0
        z_coeff[i] = z_coeff.get(i, 0.0) - b_ij / 4.0
        z_coeff[j] = z_coeff.get(j, 0.0) - b_ij / 4.0
        zz_terms.append(HamiltonianTerm(coefficient=b_ij / 4.0, pauli="ZZ", qubits=(i, j)))

    z_terms = [
        HamiltonianTerm(coefficient=coef, pauli="Z", qubits=(i,))
        for i, coef in z_coeff.items()
    ]

    return z_terms + zz_terms, offset

