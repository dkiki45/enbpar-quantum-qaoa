from qiskit.circuit import ParameterVector
from qiskit import QuantumCircuit

def build_qaoa_ansatz(model, reps=1):
    if reps < 1: 
        raise ValueError("reps deve ser >= 1")
        
    gamma, beta = ParameterVector("gamma", reps), ParameterVector("beta", reps)
    qc = QuantumCircuit(model.n_vars, name=f"QAOA_p{reps}")
    
    qc.h(range(model.n_vars)) # |+> ^ n
    
    for layer in range(reps):
        # exp(-i gamma H_C): RZ(phi)=exp(-i phi Z/2)
        for i, coef in model.h.items():
            if coef: 
                qc.rz(2 * gamma[layer] * coef, i)
                
        # RZZ(phi)=exp(-i phi ZZ/2); estes termos geram correlacoes
        for (i,j), coef in model.j.items():
            if coef: 
                qc.rzz(2 * gamma[layer] * coef, i, j)
                
        # exp(-i beta H_M), H_M=sum X_i
        for i in range(model.n_vars): 
            qc.rx(2 * beta[layer], i)
            
    return qc

def bind_by_name(qc, gammas, betas):
    if len(gammas) != len(betas): 
        raise ValueError("Tamanhos diferentes")
        
    lookup = {p.name: p for p in qc.parameters}
    values = {lookup[f"gamma[{k}]"]: gammas[k] for k in range(len(gammas))}
    values.update({lookup[f"beta[{k}]"]: betas[k] for k in range(len(betas))})
    
    return qc.assign_parameters(values, inplace=False)