import pytest
from core.qubo_formalization import build_mis_qubo, qubo_to_ising
from core.qaoa_solver import run_qaoa

def small_model():
    n, edges = 3, [(0, 1), (1, 2)]
    lin, quad, off = build_mis_qubo(n, edges)
    return qubo_to_ising(n, lin, quad, off)

def test_spsa_without_early_stop_runs_all_iterations():
    # No early stop + fixed steps: 2 evaluations per iteration + 1 final evaluation, no calibration
    r = run_qaoa(small_model(), reps=1, shots=256, seed=1, maxiter=5, optimizer_name="SPSA",
                 spsa_patience=None, spsa_learning_rate=0.05, spsa_perturbation=0.05)
    assert len(r.history) == 2 * 5 + 1

def test_spsa_step_sizes_must_be_paired():
    with pytest.raises(ValueError):
        run_qaoa(small_model(), reps=1, shots=256, seed=1, maxiter=5, optimizer_name="SPSA",
                 spsa_learning_rate=0.05)

def test_same_seed_reproduces_run():
    # Same seed -> identical optimizer trajectory and final energy (SPSA and COBYLA)
    for opt in ("SPSA", "COBYLA"):
        a = run_qaoa(small_model(), reps=1, shots=256, seed=7, maxiter=10, optimizer_name=opt)
        b = run_qaoa(small_model(), reps=1, shots=256, seed=7, maxiter=10, optimizer_name=opt)
        assert len(a.history) == len(b.history)
        assert a.expectation_qubo == b.expectation_qubo


def test_transpile_cache_ignores_reused_ids():
    # A cached entry must never be used for a different circuit object (old id() bug)
    from qiskit import QuantumCircuit
    from core.qaoa_solver import FastAerSampler
    sampler = FastAerSampler(default_shots=64, seed=1)
    a = QuantumCircuit(1); a.measure_all()            # always 0
    b = QuantumCircuit(1); b.x(0); b.measure_all()    # always 1
    sampler.run([a]).result()
    sampler._transpiled_cache[id(b)] = sampler._transpiled_cache.pop(id(a))  # simulate id reuse
    counts = sampler.run([b]).result()[0].data.meas.get_counts()
    assert counts == {"1": 64}
