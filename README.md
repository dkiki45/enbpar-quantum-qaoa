# 🚀 Refatoração QAOA - PIBIC ENBPar

Esta branch (`feature/refatoracao-qaoa`) é dedicada à migração da arquitetura antiga baseada em VQE heurístico para a implementação formal do algoritmo QAOA, conforme diretrizes do relatório técnico de avaliação.

O cronograma de implementação foi dividido em um sprint intensivo de 3 dias.

## 🏁 Cronograma de Execução (Hackathon 72h)

### ✅ Dia 1: Base Matemática e Validação (Concluído)
- [x] Limpeza da arquitetura heurística (pasta `optimizers` e arquivos legados).
- [x] `src/core/qubo_formalization.py`: Unificação da classe `IsingModel` e conversão exata de QUBO para matrizes de Pauli ($Z$ e $ZZ$) usando `SparsePauliOp`[cite: 7].
- [x] `src/core/classical_baseline.py`: Algoritmo de força bruta implementado para calcular o gabarito absoluto do custo clássico[cite: 7].
- [x] `tests/test_qubo_ising.py`: Teste passando 100%, provando a equivalência matemática entre a energia clássica e a expectativa quântica no Qiskit[cite: 7].

### ⏳ Dia 2: O Motor QAOA e Decodificação (Em Andamento)
O objetivo deste dia é construir o circuito quântico real e configurar os otimizadores nativos.
- [ ] **`src/core/qaoa_circuit.py`**: Construir o ansatz explícito do QAOA, aplicando portas $H$ no estado inicial e alternando as camadas do Hamiltoniano de Custo ($RZ$, $RZZ$) e do Mixer ($RX$)[cite: 7].
- [ ] **`src/core/qaoa_solver.py`**: Configurar a classe nativa `QAOA` do `qiskit_algorithms`, integrando o `StatevectorSampler` e otimizadores como `COBYLA` ou `SPSA`[cite: 7].
- [ ] **`src/core/solution_decoder.py`**: Criar o decodificador para mapear a distribuição de probabilidades das *bitstrings* medidas, separando soluções factíveis das que possuem violações de arestas[cite: 7].
- [ ] **Teste**: Fazer o arquivo `tests/test_qaoa_circuit.py` passar sem erros, atestando a profundidade $p$ e os parâmetros $\gamma$ e $\beta$[cite: 7].

### 📅 Dia 3: Integração, Dados Reais e Resultados
O objetivo final é plugar o mapa geográfico e rodar os testes de ponta a ponta.
- [ ] **`src/core/graph_builder.py`**: Ajustar o parsing do CSV usando a fórmula de Haversine para garantir um grafo de cobertura realista, evitando instâncias triviais[cite: 7].
- [ ] **`src/experiments/run_qaoa.py`**: Finalizar o orquestrador (CLI) que juntará todas as peças para gerar o `summary.json` contendo razões de aproximação, energias esperadas e os melhores candidatos amostrados[cite: 7].
- [ ] **Teste Final**: Fazer o `tests/test_end_to_end.py` passar limpo, simulando um pipeline completo com uma instância pequena controlada[cite: 7].