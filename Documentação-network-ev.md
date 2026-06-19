# Documentação Técnica: Evolução de Rede Lattice (Lattice Network Evolution)

**Objetivo do Script:**
Escalar o algoritmo de minimização quântica de um único indivíduo para uma rede interconectada (Lattice 2D). O código prova que o algoritmo *Hill Climbing* consegue otimizar a Energia Global de um sistema de múltiplos Qubits simultaneamente, encontrando o *Ground State* da rede.

---

## 1. A Estrutura de Rede (`QuantumLattice2D`)
* **O que é:** O "mapa" ou tabuleiro que organiza as células (`LatticeCell`) em coordenadas geográficas (linhas e colunas).
* **Função:** Neste experimento, instanciamos uma grade 2x2 (representando 4 postes de luz). A classe mapeia as coordenadas físicas (ex: linha 0, coluna 1) para os índices lógicos dos Qubits no Qiskit (ex: Qubit 1).
* **Visualização (`print_classical_state`):** Converte os estados quânticos contínuos (ângulos $\theta$) em estados clássicos discretos ($0$ ou $1$) e imprime uma matriz visual representando quais LEDs estão ligados ou desligados.

## 2. O Hamiltoniano Global (Avaliação da Cidade)
* **O que muda:** Em vez de avaliar apenas um Qubit, o Hamiltoniano agora é uma soma das energias de todos os indivíduos da malha. 
* **Equação do Teste:** $H = 1.0 \times Z_0 + 1.0 \times Z_1 + 1.0 \times Z_2 + 1.0 \times Z_3$.
* **Alvo (Ground State):** Como cada porta Z tem um mínimo de $-1.00$, o estado fundamental perfeito para esta rede de 4 postes é uma Energia Global exata de **$-4.0000$**.

## 3. Algoritmo Evolutivo em Rede (`hill_climbing_lattice`)
A dinâmica de evolução agora opera sobre a comunidade, não apenas sobre o indivíduo:
1. **Seleção Aleatória:** A cada geração, a IA sorteia um "poste" (uma célula) qualquer do mapa para sofrer mutação.
2. **Mutação Local:** Aplica-se um ruído gaussiano apenas no ângulo daquela célula escolhida.
3. **Avaliação Global:** O circuito inteiro (4 Qubits) é construído. O Hamiltoniano roda e avalia se a energia *da cidade inteira* melhorou.
4. **Decisão:**
   * Se a alteração naquele poste específico diminuiu o custo total da rede, a mutação é **ACEITA**.
   * Se a alteração piorou o custo total, a mutação é **REJEITADA** e o poste volta ao estado anterior.

---

## Conclusão Experimental
O algoritmo demonstrou capacidade de coordenar múltiplos Qubits simultaneamente. Começando com uma matriz caótica de ângulos aleatórios (energia distante do ideal), a seleção natural iterativa guiou o sistema até a configuração perfeita. Ao final das gerações, a energia global cravou em `-4.0000` e a matriz de LEDs colapsou para o estado clássico ideal, provando a escalabilidade do modelo de otimização combinatória.