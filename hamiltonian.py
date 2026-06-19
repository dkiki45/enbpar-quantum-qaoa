'''
Automação do Hamiltoniano - Pesquisa de Iluminação (10 Circuitos)
Gera uma planilha Excel com resultados empilhados na mesma página.
'''
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from dataclasses import dataclass
from typing import List, Tuple, Dict
import pandas as pd
import numpy as np

# ============================================================
# 1. ESTRUTURA DO HAMILTONIANO
# ============================================================
@dataclass
class HamiltonianTerm:
    coefficient: float
    pauli: str
    qubits: Tuple[int, ...]

def z_term_expectation_from_counts(counts: Dict[str, int], qubits: Tuple[int, ...]) -> float:
    shots = sum(counts.values())
    exp_val = 0.0
    for bitstring, count in counts.items():
        prob = count / shots
        product = 1
        for q in qubits:
            bit = bitstring[-1 - q]
            z = +1 if bit == '0' else -1
            product *= z
        exp_val += prob * product
    return exp_val

def hamiltonian_expectation_z_only_from_counts(counts: Dict[str, int], terms: List[HamiltonianTerm]) -> float:
    total = 0.0
    for term in terms:
        exp_val = z_term_expectation_from_counts(counts, term.qubits)
        total += term.coefficient * exp_val
    return total

# ============================================================
# 2. DEFINIÇÃO DOS 10 CIRCUITOS (A a J)
# ============================================================
def criar_circuitos():
    circuitos = {}
    qc_A = QuantumCircuit(4); qc_A.h(0); qc_A.cx(0, 1); qc_A.ry(0.7, 2); qc_A.cx(2, 3); qc_A.measure_all(); circuitos['Circuito_A'] = qc_A
    qc_B = QuantumCircuit(4); qc_B.h(0); qc_B.cx(0, 1); qc_B.ry(np.pi/2, 2); qc_B.cx(2, 3); qc_B.measure_all(); circuitos['Circuito_B'] = qc_B
    qc_C = QuantumCircuit(4); qc_C.h(0); qc_C.cx(0, 1); qc_C.x(0); qc_C.ry(np.pi, 2); qc_C.cx(2, 3); qc_C.measure_all(); circuitos['Circuito_C'] = qc_C
    qc_D = QuantumCircuit(4); qc_D.measure_all(); circuitos['Circuito_D'] = qc_D
    qc_E = QuantumCircuit(4); qc_E.x([0, 1, 2, 3]); qc_E.measure_all(); circuitos['Circuito_E'] = qc_E
    qc_F = QuantumCircuit(4); qc_F.h([0, 1, 2, 3]); qc_F.measure_all(); circuitos['Circuito_F'] = qc_F
    qc_G = QuantumCircuit(4); qc_G.ry(np.pi/4, [0, 1, 2, 3]); qc_G.measure_all(); circuitos['Circuito_G'] = qc_G
    qc_H = QuantumCircuit(4); qc_H.h(0); qc_H.cx(0, 1); qc_H.h(2); qc_H.cx(2, 3); qc_H.measure_all(); circuitos['Circuito_H'] = qc_H
    qc_I = QuantumCircuit(4); qc_I.ry(np.pi/8, 0); qc_I.ry(np.pi/4, 1); qc_I.ry(np.pi/2, 2); qc_I.ry(np.pi, 3); qc_I.measure_all(); circuitos['Circuito_I'] = qc_I
    qc_J = QuantumCircuit(4); qc_J.x([0, 1]); qc_J.ry(np.pi/2, [2, 3]); qc_J.measure_all(); circuitos['Circuito_J'] = qc_J
    return circuitos

termos_hamiltoniano = [
    HamiltonianTerm(coefficient=1.0, pauli="Z", qubits=(0,)),
    HamiltonianTerm(coefficient=1.0, pauli="Z", qubits=(1,)),
    HamiltonianTerm(coefficient=1.0, pauli="Z", qubits=(2,)),
    HamiltonianTerm(coefficient=1.0, pauli="Z", qubits=(3,)),
]

# ============================================================
# 3. AUTOMAÇÃO E GERAÇÃO DA PLANILHA EXCEL
# ============================================================
num_rodadas = 20
shots_por_rodada = 4096
sim = AerSimulator()
circuitos_dict = criar_circuitos()

nome_arquivo = "Pesquisa_Hamiltoniano_Empilhada.xlsx"
print(f"🚀 Iniciando simulações... Gerando lista empilhada em {nome_arquivo}.")

with pd.ExcelWriter(nome_arquivo, engine='openpyxl') as writer:
    
    # 3.1. Aba de Definições
    definicoes = {
        "Circuito": list(circuitos_dict.keys()),
        "Qubits 0 e 1 (Controle)": [
            "H(0) e CX(0->1)", 
            "H(0) e CX(0->1)", 
            "H(0), CX(0->1) e X(0)", 
            "Livres (|00>)", 
            "X(0) e X(1)", 
            "H(0) e H(1)", 
            "RY(45°)", 
            "H(0) e CX(0->1)", 
            "RY(22.5°) e RY(45°)", 
            "X(0) e X(1)"
        ],
        "Qubits 2 e 3 (Rotação)": [
            "RY(0.7 em Q2) e CX(2->3)", 
            "RY(90° em Q2) e CX(2->3)", 
            "RY(180° em Q2) e CX(2->3)", 
            "Livres (|00>)", 
            "X(2) e X(3)", 
            "H(2) e H(3)", 
            "RY(45°)", 
            "H(2) e CX(2->3)", 
            "RY(90°) e RY(180°)", 
            "RY(90°)"
        ],
        "Objetivo (Efeito nos LEDs)": [
            "Base do professor",
            "Zera a energia/intensidade da metade do circuito",
            "Quebra a paridade e inverte a fase (Luz invertida)",
            "Nenhuma porta (Energia Máxima = Custo Alto)",
            "Todos invertidos (Energia Mínima = Totalmente Otimizado)",
            "Superposição total (Média balanceada em zero)",
            "Rotação suave (Ajuste gradual em todos os LEDs)",
            "Dois pares de LEDs perfeitamente emaranhados",
            "Gradiente: Cada LED com uma intensidade diferente",
            "Inversão parcial: Metade ligada no máximo, metade equilibrada"
        ]
    }
    df_def = pd.DataFrame(definicoes)
    df_def.to_excel(writer, sheet_name="Definicoes_dos_Circuitos", index=False)
    
    # 3.2. Aba de Resultados Empilhados
    nome_aba_resultados = "Resultados_Gerais"
    linha_atual = 0 # O controle de onde o código vai escrever
    
    for nome_aba, qc in circuitos_dict.items():
        print(f"Processando {nome_aba}...")
        resultados = []
        tqc = transpile(qc, sim)
        
        for rodada in range(1, num_rodadas + 1):
            result = sim.run(tqc, shots=shots_por_rodada).result()
            counts = result.get_counts()
            energia = hamiltonian_expectation_z_only_from_counts(counts, termos_hamiltoniano)
            resultados.append(energia)
            
        # Formatar a tabela
        df = pd.DataFrame([resultados], columns=[f"Rodada {i}" for i in range(1, num_rodadas + 1)])
        df.index = ["<H> (Energia Total)"]
        df['Média'] = df.mean(axis=1)
        df['Desvio Padrão'] = df.std(axis=1)
        
        # 1. Escreve o Título do Circuito
        df_titulo = pd.DataFrame([[f"=== {nome_aba} ==="]])
        df_titulo.to_excel(writer, sheet_name=nome_aba_resultados, startrow=linha_atual, index=False, header=False)
        linha_atual += 1 # Pula 1 linha
        
        # 2. Escreve a Tabela de Dados logo abaixo
        df.to_excel(writer, sheet_name=nome_aba_resultados, startrow=linha_atual)
        
        # Pula as linhas da tabela + 3 linhas em branco de respiro para o próximo circuito
        linha_atual += len(df.index) + 3 

print(f"\n🎉 SUCESSO! Abra o Excel e veja tudo organizado na aba 'Resultados_Gerais'.")