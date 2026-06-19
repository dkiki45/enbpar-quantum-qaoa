# Documentação Técnica: Evolução de Indivíduo Quântico (Single Cell Evolution)

**Objetivo do Script:**
Validar a integração entre a estrutura de dados evolutiva (LatticeCell), o avaliador físico (Hamiltoniano) e o algoritmo de minimização (Hill Climbing). O código prova que o sistema consegue treinar um Qubit "cego" a encontrar o Estado Fundamental (*Ground State*) de forma autônoma.

---

## 1. A Estrutura Biológica (`LatticeCell`)
* **O que é:** A planta fundamental de um indivíduo do nosso ecossistema evolutivo.
* **Função:** Armazena o "DNA" do nosso LED, especificamente o ângulo de rotação quântica na variável `theta`. Possui a propriedade `fitness`, que recebe a avaliação da física.

## 2. O Avaliador Físico (Hamiltoniano)
* **O que é:** A função de custo clássica mapeada para o ambiente quântico.
* **Função:** Definimos um Hamiltoniano base ($H = 1.0 \times Z_0$). As funções calculam o **Valor Esperado** do circuito. Energias altas representam "multas" (péssimo fitness), energias baixas representam "economia" (bom fitness).

## 3. O Motor de Integração (`evaluate_cell_fitness`)
Função que conecta os mundos biológico e físico:
1. Extrai o `theta` da célula.
2. Cria um circuito Qiskit de 1 Qubit aplicando uma porta $R_y(\theta)$.
3. Roda o circuito no simulador *Aer*.
4. Calcula a Energia Total via Hamiltoniano.
5. Salva a energia na propriedade `fitness` da célula de forma silenciosa.

---

## 4. Algoritmo Evolutivo: Escalada de Colina (`hill_climbing_single_cell`)
Esta é a IA que minimiza o custo energético iterativamente. 

**O Ciclo de Vida (Gerações):**
1. **Ponto de Partida:** A célula recebe um ângulo `theta` completamente aleatório (Indivíduo cego).
2. **Mutação:** O algoritmo soma um ruído gaussiano (ex: $\pm 0.3$ radianos) ao ângulo atual.
3. **Avaliação:** O Hamiltoniano roda o circuito mutado e devolve a nova energia.
4. **Seleção Natural:**
   * **ACCEPTED (Melhorou):** Se a nova energia for menor (mais próxima de $-1.00$), a mutação é aceita e o DNA da célula é atualizado.
   * **REJECTED (Piorou):** Se a energia aumentou, o algoritmo desfaz a mutação e a célula retorna ao estado anterior seguro.

## Conclusão Experimental
O terminal demonstrou que, não importa quão ruim seja o ângulo aleatório inicial (Geração 0), o algoritmo sempre rejeita caminhos de alta energia e converge perfeitamente para $\theta \approx 3.1415$ ($\pi$). Isso ancora o Qubit no polo sul da Esfera de Bloch ($|1\rangle$), atingindo com sucesso a energia mínima exata do sistema ($-1.0000$), validando a inteligência central do nosso modelo de Otimização Quântica.