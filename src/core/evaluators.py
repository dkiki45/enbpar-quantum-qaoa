"""
QAOA energy evaluators used by the grid search and the campaign (moved from grid_search_ext.py).

LandscapeEvaluator: exact energy E(betas, gammas) from the statevector (Aer save_expectation_value),
    many angle sets per Aer job.
ShotsEvaluator: energy estimated as the mean QUBO cost of `shots` samples, many angle sets per job.
"""
import numpy as np
from qiskit import transpile
from qiskit.circuit.library import QAOAAnsatz
from qiskit_aer import AerSimulator

from core.qaoa_solver import FastAerSampler

BASIS = ['rx', 'ry', 'rz', 'cx', 'h', 'x', 'y', 'z', 'id']


class LandscapeEvaluator:
    """Exact QAOA energy E(betas, gammas) for many angle sets per Aer job."""

    def __init__(self, model, p, sim_opts):
        self.model, self.p = model, p
        h = model.to_sparse_pauli_op()
        ansatz = QAOAAnsatz(h, reps=p)
        self.circ = transpile(ansatz, basis_gates=BASIS, optimization_level=1)
        self.circ.save_expectation_value(h, list(range(self.circ.num_qubits)), label="E")
        params = list(self.circ.parameters)
        self.betas = sorted([q for q in params if q.name.startswith("β")], key=lambda q: q.index)
        self.gammas = sorted([q for q in params if q.name.startswith("γ")], key=lambda q: q.index)
        assert len(self.betas) == len(self.gammas) == p, "unexpected QAOAAnsatz parameters"
        self.sim = AerSimulator(**sim_opts)

    def energies(self, points):
        """points: list of (betas[p], gammas[p]). Returns the QUBO energies (Ising + offset)."""
        bind = {q: [pt[0][i] for pt in points] for i, q in enumerate(self.betas)}
        bind.update({q: [pt[1][i] for pt in points] for i, q in enumerate(self.gammas)})
        res = self.sim.run(self.circ, parameter_binds=[bind], shots=1).result()
        return np.array([res.data(i)["E"] for i in range(len(points))]) + self.model.offset


class ShotsEvaluator:
    """Sampled QAOA energy (mean QUBO cost of `shots` samples) for many angle sets per job."""

    def __init__(self, model, p, sim_opts, edges, shots=1024, seed=7):
        self.model, self.p = model, p
        circ = QAOAAnsatz(model.to_sparse_pauli_op(), reps=p)
        circ.measure_all()
        self.circ = circ
        params = list(circ.parameters)
        self.order = [("b", q.index) if q.name.startswith("β") else ("g", q.index) for q in params]
        assert len(params) == 2 * p, "unexpected QAOAAnsatz parameters"
        self.sampler = FastAerSampler(default_shots=shots, seed=seed, options={"backend_options": sim_opts})
        self.e = np.asarray(edges, dtype=int).reshape(-1, 2)

    def _mean_cost(self, counts, alpha=1.0, beta=2.0):
        keys = list(counts)
        x = np.array([[int(c) for c in reversed(k)] for k in keys], dtype=float)  # x_0..x_(n-1)
        w = np.array([counts[k] for k in keys], dtype=float)
        cost = -alpha * x.sum(axis=1) + beta * (x[:, self.e[:, 0]] * x[:, self.e[:, 1]]).sum(axis=1)
        return float((cost * w).sum() / w.sum())

    def energies(self, points):
        vals = np.array([[pt[0][i] if kind == "b" else pt[1][i] for kind, i in self.order] for pt in points])
        data = self.sampler.run([(self.circ, vals)]).result()[0].data.meas
        return np.array([self._mean_cost(data.get_counts(loc=k)) for k in range(len(points))])
