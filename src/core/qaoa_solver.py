from dataclasses import dataclass
import numpy as np
from qiskit.primitives import StatevectorSampler
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
        
    sampler = StatevectorSampler(default_shots=shots, seed=seed)
    optimizer = (SPSA(maxiter=maxiter) if optimizer_name.upper() == "SPSA" 
                 else COBYLA(maxiter=maxiter, tol=1e-6))
                 
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