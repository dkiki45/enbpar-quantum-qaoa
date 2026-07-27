# Research Project: Quantum Optimization for LED Retrofitting (ENBPar)

**Scientific Initiation Research (PIBIC) - PUCPR**
**Student:** David Bobato Kikina | **Advisors:** Profs. Jonas Krause and Rodrigo Pasti

## Overview
This repository contains the algorithms and technical documentation for the development of a Quantum Combinatorial Optimization model. The objective is to create a system capable of intelligent planning (retrofitting) for ENBPar's public lighting infrastructure, utilizing the **Quantum Approximate Optimization Algorithm (QAOA)** to solve large-scale logistical bottlenecks that are intractable for classical computing.

---

## Project Architecture
```text
├── data/
│   └── paranainterativo.csv         # Real-world IPPUC dataset (GPS coordinates)
├── docs/                            # Theoretical documentation and mathematical justifications
│   ├── network_evolution.md
│   ├── qaoa_justification.md
│   ├── qubo_evolution.md
│   ├── results_and_methodology.md
│   └── single_cell_evolution.md
├── results/                         # Generated outputs (CSVs, TXTs, and PNG graphs)
├── src/                             # Main Python Source Code
│   ├── hamiltonian.py               # Pauli Z/ZZ constraint mapping
│   ├── lattice_network_evolution.py # Optimization Engine (Hill Climbing & Simulated Annealing)
│   ├── plot_convergence.py          # Data Visualization Dashboard (Pandas & Matplotlib)
│   ├── qubo_formalization.py        # Classical C(x) to Ising <H> translation
│   ├── qubo_network_evolution.py    # Legacy evolution tracker
│   ├── real_graph_qubo.py           # Haversine distance and real data extraction pipeline
│   ├── simulated_graph_qubo.py      # Control group mapping
│   ├── single_cell_evaluation.py    # Single qubit quantum gate (RY) testing
│   └── quantum_lattice_2d_template.py # Qiskit lattice object template
├── tests/
│   └── test_quantum_lattice.py      # Unit test suite for lattice structures
├── .gitignore
└── README.md
```

---

## Code and Documentation Index

The project was built in a modular approach. Below are the links to the detailed documentation and source code for each evolutionary stage of the research:

### 1. Single Quantum Individual Evolution
* **Source Code:** `single_cell_evolution.py`
* **Technical Documentation:** [📄 Read docs/single_cell_evolution.md](docs/single_cell_evolution.md)
* **Summary:** Initial Proof of Concept (PoC). It integrates the biological data structure (`LatticeCell`) with a linear physical evaluator (Hamiltonian) guided by a *Hill Climbing* algorithm. It demonstrates that a single "blind" Qubit can minimize its energy and autonomously converge to the *Ground State*.

### 2. Global Optimization in Spatial Grid (2D Lattice)
* **Source Code:** `lattice_network_evolution.py`
* **Technical Documentation:** [📄 Read docs/network_evolution.md](docs/network_evolution.md)
* **Summary:** Expansion of the single-individual minimization algorithm to an interconnected grid (2x2 LED matrix). The evolutionary engine now operates on spatial coordinates, and the Hamiltonian measures the Global Energy of the circuit. It proves that the algorithm can coordinate multiple Qubits simultaneously to find the optimal state of the network.

### 3. Theoretical Foundation: The Classical Bottleneck and the Transition to QAOA
* **Academic Documentation:** [📄 Read docs/qaoa_justification.md](docs/qaoa_justification.md)
* **Summary:** A theoretical essay that exposes the limitations of the classical architecture implemented in the previous steps. The text scientifically justifies the need to adopt QAOA to mitigate "stochastic inefficiency" through massively parallel processing via superposition and phase interference.

### 4. QUBO Model and Neighborhood Interaction
* **Source Code:** `qubo_network_evolution.py`
* **Technical Documentation:** [📄 Read docs/qubo_evolution.md](docs/qubo_evolution.md)
* **Summary:** Upgrades the model to a Quadratic Unconstrained Binary Optimization (QUBO) architecture. Introduces coupled penalty terms ($Z_iZ_j$) for adjacent LEDs, creating spatial frustration. Demonstrates the algorithm's capability to balance linear rewards with quadratic penalties to resolve spatial conflicts and find complex distribution patterns.

### 5. Urban Infrastructure Mapping: Real vs. Simulated Topologies
* **Source Code:** `real_graph_qubo.py` and `simulated_graph_qubo.py`
* **Technical Documentation:** [📄 Read docs/results_and_methodology.md](docs/results_and_methodology.md)
* **Summary:** Bridges the gap between theoretical physics and real-world urban planning. It extracts real geographical coordinates from the IPPUC (GeoCuritiba) dataset and uses the Haversine formula to map the local neighborhood graph of streetlights. To scientifically validate the quantum model, it also generates an artificial control graph (Euclidean distance). Both scripts generate the precise QUBO Hamiltonian constraints ($Z_i$ and $Z_iZ_j$) required to format the problem for QAOA execution on IBM Quantum hardware.

### 6. Advanced Heuristics & Empirical Validation (Code Review Updates)
* **Source Code:** [`src/lattice_network_evolution.py`](src/lattice_network_evolution.py) & [`src/plot_convergence.py`](src/plot_convergence.py)
* **Summary:** Introduces thermal fluctuation strategies (**Simulated Annealing**) and **Random Restarts** to escape local minima in the energy landscape. Implements automated data visualization dashboards (Matplotlib/Pandas) to empirically prove algorithmic convergence and state space exploration.

---

## How to Run the Pipeline
Ensure you run these commands from the **root directory** of the project:

1. **Test Real Data Extraction:** `python src/real_graph_qubo.py`
2. **Run the Hybrid Optimization Engine:** `python src/lattice_network_evolution.py`
3. **Generate Convergence Dashboards:** `python src/plot_convergence.py`
4. **Run the Unit Test Suite:** `python -m unittest discover -s tests -v`

---

## 🗺️ Real-World Data Extraction (Prado Velho / PUCPR)
![Mapa dos postes no Prado Velho](assets/mapa_prado_velho.png)
*Fonte: [Portal Paraná Interativo - Governo do Estado do Paraná](https://paranainterativo.pr.gov.br/portal/apps/webappviewer/index.html?id=f282d1c181c0405789e2f65c17ac274d)*