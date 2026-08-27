import json
import matplotlib.pyplot as plt
from dataclasses import asdict
from pathlib import Path

from core.graph_builder import build_graph_from_csv
from core.qubo_formalization import build_mis_qubo, qubo_to_ising
from core.qaoa_solver import run_qaoa
from core.solution_decoder import decode_distribution, best_feasible_candidate
from core.classical_baseline import solve_exact_bruteforce

def execute(csv_path, output, limit=15, reps=1, shots=8192, seed=2):
    nodes, edges = build_graph_from_csv(csv_path, limit)
    linear, quadratic, off = build_mis_qubo(len(nodes), edges)
    model = qubo_to_ising(len(nodes), linear, quadratic, off)
    
    result = run_qaoa(model, reps, shots, seed, maxiter=300)
    candidates = decode_distribution(result.distribution, len(nodes), edges)
    best = best_feasible_candidate(candidates)
    
    exact_bits, exact_cost = solve_exact_bruteforce(len(nodes), edges)
    
    summary = {
        "n_nodes": len(nodes), 
        "n_edges": len(edges), 
        "reps": reps,
        "shots": shots, 
        "seed": seed, 
        "expectation_ising": result.expectation_ising,
        "expectation_qubo": result.expectation_qubo, 
        "best": asdict(best),
        "exact_bits": exact_bits, 
        "exact_cost": exact_cost,
        "cardinality_ratio": sum(best.bits) / sum(exact_bits) if sum(exact_bits) > 0 else 0.0,
        "parameters": result.optimal_parameters
    }
    
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    
    return summary, result.history

if __name__ == "__main__":
    # Certifique-se de que o caminho do CSV existe no seu projeto!
    pasta_destino = "src/results/qaoa_p1"
    summary, history = execute("src/data/paranainterativo.csv", pasta_destino, limit=10)
    
    # Extrai os dados para o gráfico
    iteracoes = [h["evaluation"] for h in history]
    energias = [h["expectation_qubo"] for h in history]
    
    # Cria o gráfico
    plt.figure(figsize=(10, 6))
    plt.plot(iteracoes, energias, marker='o', linestyle='-', color='#1f77b4', linewidth=2)
    plt.title("Convergência do Algoritmo QAOA (COBYLA)", fontsize=14)
    plt.xlabel("Iterações do Otimizador", fontsize=12)
    plt.ylabel("Energia Esperada (Custo QUBO)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Salva a imagem na mesma pasta do JSON
    caminho_grafico = Path(pasta_destino) / "convergencia_qaoa.png"
    plt.savefig(caminho_grafico, dpi=300, bbox_inches='tight')
    print(f"\nSucesso! Gráfico salvo em: {caminho_grafico}")