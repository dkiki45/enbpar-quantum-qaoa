"""
=============================================================================
MATHEMATICAL FORMALIZATION (QUBO -> ISING)
=============================================================================
This module formulates the optimization problem as a QUBO (Quadratic 
Unconstrained Binary Optimization) and rigorously converts it into an Ising 
Hamiltonian for quantum simulation.

CLASSICAL OBJECTIVE FUNCTION:

    C(x) = -alpha * sum_i x_i  +  beta * sum_{(i,j) in E} x_i * x_j

    - alpha (coverage): the higher it is, the more the model "rewards"
      (lowers the cost of) keeping a streetlight on (x_i = 1). Minimizing
      C(x) pushes x_i -> 1 through this term.
    - beta (redundancy): penalizes (raises the cost) when TWO neighboring
      streetlights (distance < threshold, i.e. an edge in E) are both on
      at the same time -- this characterizes lighting "redundancy".

    x_i = 1  -> streetlight "on"/selected in the solution
    x_i = 0  -> streetlight "off"/not selected

The alpha and beta values preserve the numerical ratio necessary to balance
coverage and redundancy. This formalization ensures that quantum Pauli (Z/ZZ) 
coefficients are mathematically derived rather than arbitrarily assigned.
=============================================================================
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass

# Default weights for the classical objective function (see docstring above)
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

    Derivation (documented, not just implemented):

        a_i * x_i
            = a_i/2 - (a_i/2) Z_i

        b_ij * x_i * x_j
            = (b_ij/4) - (b_ij/4) Z_i - (b_ij/4) Z_j + (b_ij/4) Z_i Z_j

    Therefore:
        offset  = offset0 + sum_i a_i/2 + sum_{i<j} b_ij/4
        h_i     = -a_i/2 - (1/4) * sum_{j neighboring i} b_ij   (coeff. of Z_i)
        J_ij    = b_ij/4                                        (coeff. of Z_i Z_j)
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

