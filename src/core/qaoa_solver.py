from dataclasses import dataclass
import numpy as np
from qiskit import transpile
from qiskit_aer.primitives import SamplerV2
from qiskit.primitives.containers.sampler_pub import SamplerPub 
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA, SPSA

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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._transpiled_cache = {}

    def run(self, pubs, **kwargs):
        new_pubs = []
        for pub in pubs:
            # Extract the quantum circuit
            circ = pub[0] if isinstance(pub, tuple) else pub.circuit
            circ_id = id(circ)
            
            # If it's the first time, transpile and break giant matrices into simple gates
            if circ_id not in self._transpiled_cache:
                self._transpiled_cache[circ_id] = transpile(
                    circ, 
                    optimization_level=1, 
                    basis_gates=['rx', 'ry', 'rz', 'cx', 'h', 'x', 'y', 'z', 'id']
                )
                
            t_circ = self._transpiled_cache[circ_id]
            
            # Rebuild the data payload for Qiskit
            if isinstance(pub, tuple):
                new_pubs.append((t_circ,) + pub[1:])
            else:
                shots = getattr(pub, 'shots', None)
                if shots is not None:
                    new_pubs.append((t_circ, pub.parameter_values, shots))
                else:
                    new_pubs.append((t_circ, pub.parameter_values))
                    
        return super().run(new_pubs, **kwargs)

class GridSearchSampler(FastAerSampler):
    """Sampler used only by grid_search.py: transpile cache that is safe against id() reuse."""

    def run(self, pubs, **kwargs):
        # Fall back to the sampler's default shots when none are passed explicitly
        default_shots = kwargs.get("shots") or self.options.default_shots
        new_pubs = []
        for pub in pubs:
            # Normalize any pub format (tuple, SamplerPub, ...) into a SamplerPub
            pub = SamplerPub.coerce(pub, default_shots)
            circ = pub.circuit

            # Reuse the transpiled circuit only if the cached entry belongs to this exact object
            cached = self._transpiled_cache.get(id(circ))
            if cached is None or cached[0] is not circ:
                # Keep the cache small: grid search creates many short-lived circuits
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

        # Skip FastAerSampler.run and call SamplerV2.run directly, otherwise
        # the original (buggy) caching logic would run a second time
        return SamplerV2.run(self, new_pubs, **kwargs)

def run_qaoa(model, reps=1, shots=4096, seed=2, maxiter=300, optimizer_name="COBYLA", initial_point=None):
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
    sampler = FastAerSampler()
    sampler.options.default_shots = shots
    sampler.options.seed_simulator = seed

    #sampler = StatevectorSampler(default_shots=shots, seed=seed)

    if optimizer_name.upper() == "SPSA":
        checker = SmartSPSAChecker(tol=0.001, patience=15)
        optimizer = SPSA(maxiter=maxiter, termination_checker=checker)
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

