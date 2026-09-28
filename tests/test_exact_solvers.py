import itertools
from core.classical_baseline import solve_exact_bruteforce, solve_exact_ilp
from core.qubo_formalization import conflicting_edges

GRAPHS = [
    (3, [(0, 1), (1, 2)]),
    (5, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]),
    (6, []),
    (8, [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (4, 7)]),
    (10, list(itertools.combinations(range(4), 2)) + [(4, 5), (5, 6), (7, 8), (8, 9), (3, 9)]),
]

def test_ilp_matches_bruteforce_cost():
    for n, edges in GRAPHS:
        _, bf_cost = solve_exact_bruteforce(n, edges)
        ilp_bits, ilp_cost = solve_exact_ilp(n, edges)
        assert ilp_cost == bf_cost
        assert not conflicting_edges(ilp_bits, edges)
