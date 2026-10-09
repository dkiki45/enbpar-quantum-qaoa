from dataclasses import dataclass
import numpy as np
from qiskit import transpile
from qiskit_aer.primitives import SamplerV2
from qiskit.primitives.containers.sampler_pub import SamplerPub 
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA, SPSA
from qiskit_algorithms.utils import algorithm_globals

@dataclass
class QAOARunResult:
    optimal_parameters: dict
    expectation_ising: float
    expectation_qubo: float
    distribution: dict
    best_measurement: dict
    history: list
    raw_result: object

class SmartSPSAChecker:
    def __init__(self, tol=0.001, patience=15):
        """
        tol: The minimum improvement threshold to be considered a valid step.
        patience: How many iterations SPSA can run without improvement before early stopping.
        """
        self.tol = tol
        self.patience = patience
        self.best_value = float('inf')
        self.stagnant_count = 0

    def __call__(self, nfev, parameters, value, stepsize, accepted):
        # 'value' is the calculated energy (cost) in the current iteration
        if value < self.best_value - self.tol:
            self.best_value = value
            self.stagnant_count = 0  # Reset the counter, as it found a better path!
        else:
            self.stagnant_count += 1 # It stagnated, so we increase the counter.
        
        # If patience runs out, return True to tell Qiskit to stop SPSA early.
        if self.stagnant_count >= self.patience:
            print(f"    [Smart SPSA] Convergence reached! Stopping at iteration {nfev}.")
            return True
            
        return False

class FastAerSampler(SamplerV2):
    """Aer SamplerV2 with a transpile cache that is safe against id() reuse.

    The old cache was keyed only by id(circ). qiskit-algorithms passes a NEW circuit with the
    angles already bound on every evaluation, and Python reuses the id() of freed objects, so
    sometimes the cache returned the transpiled circuit of an OLDER evaluation (old angles):
    that evaluation got the energy of a different point and runs with the same seed were not
    always reproducible. Now each entry keeps a reference to its original circuit and is only
    reused for that exact object.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._transpiled_cache = {}

    def run(self, pubs, **kwargs):
        # Shots: run(shots=...), else options.default_shots (set by grid_search.py), else the constructor value
        default_shots = kwargs.get("shots") or getattr(self.options, "default_shots", None) or self.default_shots
        new_pubs = []
        for pub in pubs:
            # Normalize any pub format (tuple, SamplerPub, ...) into a SamplerPub
            pub = SamplerPub.coerce(pub, default_shots)
            circ = pub.circuit

            # Reuse the transpiled circuit only if the cached entry belongs to this exact object
            cached = self._transpiled_cache.get(id(circ))
            if cached is None or cached[0] is not circ:
                # Keep the cache small: QAOA creates a new short-lived circuit per evaluation
                if len(self._transpiled_cache) > 64:
                    self._transpiled_cache.clear()
                t_circ = transpile(
                    circ,
                    optimization_level=1,
                    basis_gates=['rx', 'ry', 'rz', 'cx', 'h', 'x', 'y', 'z', 'id'],
                )
                # Store the original circuit alongside the result: the strong reference
                # keeps it alive, so its id() can never be reassigned to another circuit
                self._transpiled_cache[id(circ)] = (circ, t_circ)
            else:
                t_circ = cached[1]

            # Rebuild the pub with the transpiled circuit, keeping parameter values and shots
            new_pubs.append(
                SamplerPub(t_circ, pub.parameter_values, pub.shots, validate=True)
            )

        return super().run(new_pubs, **kwargs)


class GridSearchSampler(FastAerSampler):
    """Kept for grid_search.py: the safe cache now lives in FastAerSampler."""


def run_qaoa(model, reps=1, shots=4096, seed=2, maxiter=300, optimizer_name="COBYLA", initial_point=None, sim_method=None,
             spsa_patience=15, spsa_learning_rate=None, spsa_perturbation=None):
    """
    SPSA options (defaults keep the original behavior):
      spsa_patience: iterations without improvement before early stop (None = no early stop, runs all maxiter)
      spsa_learning_rate / spsa_perturbation: fixed step sizes; both None = Qiskit auto-calibration
    """
    if min(reps, shots, maxiter) < 1: 
        raise ValueError("Parametros invalidos")
        
    history = []
    
    def callback(eval_count, parameters, mean, metadata):
        print(f"Iteração {eval_count} | Energia Quântica: {np.real(mean):.4f}")
        history.append({
            "evaluation": int(eval_count),
            "parameters": np.asarray(parameters).tolist(),
            "expectation_ising": float(np.real(mean)),
            "expectation_qubo": float(np.real(mean)) + model.offset,
            "metadata": dict(metadata or {})
        })

    # qiskit aer
    # Method/device/precision default to config.py (SIMULATION_METHOD, SIMULATION_DEVICE, SIMULATION_PRECISION)
    from core.sim_backend import backend_options
    # Aer's SamplerV2 only accepts shots/seed in the constructor: setting
    # sampler.options.default_shots / seed_simulator afterwards is silently ignored
    sampler = FastAerSampler(default_shots=shots, seed=seed,
                             options={"backend_options": backend_options(sim_method)})

    # Seed Qiskit's global RNG too: it drives the random QAOA initial point and the
    # SPSA perturbations, so the same seed now reproduces the same run
    algorithm_globals.random_seed = seed

    #sampler = StatevectorSampler(default_shots=shots, seed=seed)

    if optimizer_name.upper() == "SPSA":
        if (spsa_learning_rate is None) != (spsa_perturbation is None):
            raise ValueError("Set both spsa_learning_rate and spsa_perturbation, or neither (auto-calibration)")
        checker = SmartSPSAChecker(tol=0.001, patience=spsa_patience) if spsa_patience is not None else None
        optimizer = SPSA(maxiter=maxiter, termination_checker=checker,
                         learning_rate=spsa_learning_rate, perturbation=spsa_perturbation)
    else:
        optimizer = COBYLA(maxiter=maxiter, tol=1e-6)
                 
    qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=reps, 
                initial_point=initial_point, callback=callback)
                
    raw = qaoa.compute_minimum_eigenvalue(model.to_sparse_pauli_op())
    
    distribution = {int(k, 2) if isinstance(k, str) else int(k): float(np.real(v)) for k, v in dict(raw.eigenstate).items()}
    energy = float(np.real(raw.eigenvalue))
    
    return QAOARunResult(
        {str(k): float(v) for k, v in raw.optimal_parameters.items()},
        energy, 
        energy + model.offset, 
        distribution,
        dict(raw.best_measurement or {}), 
        history, 
        raw
    )

