from core.solution_decoder import decode_distribution, best_feasible_or_none, feasible_probability

EDGES = [(0, 1), (1, 2)]

def test_all_infeasible_returns_none():
    # States 0b011 (x0=x1=1) and 0b110 (x1=x2=1) both violate an edge
    cands = decode_distribution({0b011: 0.6, 0b110: 0.4}, 3, EDGES)
    assert best_feasible_or_none(cands) is None
    assert feasible_probability(cands) == 0.0

def test_picks_best_feasible():
    # 0b101 (x0=x2=1) is the optimum; 0b001 is feasible but worse; 0b011 violates
    cands = decode_distribution({0b101: 0.2, 0b001: 0.5, 0b011: 0.3}, 3, EDGES)
    best = best_feasible_or_none(cands)
    assert best.selected == 2 and not best.violations
    assert abs(feasible_probability(cands) - 0.7) < 1e-12
