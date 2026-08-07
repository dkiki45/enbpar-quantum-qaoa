# Research Project: Quantum Optimization for LED Retrofitting (ENBPar)

**Scientific Initiation Research (PIBIC) - PUCPR**
**Student:** David Bobato Kikina | **Advisors:** Profs. Jonas Krause and Rodrigo Pasti

## Overview
This repository contains the algorithms and technical documentation for the development of a Quantum Combinatorial Optimization model. The objective is to create a system capable of intelligent planning (retrofitting) for ENBPar's public lighting infrastructure, utilizing the **Quantum Approximate Optimization Algorithm (QAOA)** to solve large-scale logistical bottlenecks that are intractable for classical computing.

At its core, the retrofitting problem is mapped onto a **Maximal Independent Set (MIS)** problem: each streetlight is a node in a graph, and an edge is drawn between two nodes whenever they are close enough to cause redundant/overlapping illumination. Solving for the MIS of this graph yields the largest possible subset of streetlights that can be kept active with zero spatial conflict between them — maximizing coverage while eliminating redundancy.

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
│   ├── plot_mis_map.py              # Visual geographic proof of the MIS logic
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

## Technical Guide: Parameter Tuning & Experimentation

The repository is designed for modular experimentation. Modify the hyperparameters in the source files below to observe different topological and algorithmic behaviors.

### 1. QUBO Hamiltonian Calibration (`src/qubo_formalization.py`)
Defines the objective function coefficients for the **Maximal Independent Set (MIS)** mapping:

$$C(x) = -\alpha \sum_i x_i + \beta \sum_{(i,j) \in E} x_i x_j$$

* `ALPHA_COVERAGE` (Linear Reward): The numerical incentive for assigning a state $|1\rangle$ (ON) to a geographic node — i.e., for including that streetlight in the independent set.
* `BETA_REDUNDANCY` (Quadratic Penalty): The coupled penalty applied when two connected nodes (an edge $(i,j) \in E$, meaning two streetlights close enough to overlap) share the $|1\rangle$ state — this is exactly what a valid MIS solution forbids.
* **MIS Constraint Rule:** To strictly enforce the independent set (zero overlap), ensure the penalty strictly outweighs the local reward ($\beta > \alpha$). If this rule is violated, the model may prefer a "redundant" solution (two conflicting nodes both ON) over a valid independent one, since the combined linear reward could outweigh a single penalty.

### 2. Topological Graph Generation (`src/real_graph_qubo.py`)
Controls the physical constraints and geographic data extraction via Haversine distance calculations. This is where the **edges of the MIS graph** are actually defined: two streetlights become connected (conflicting) nodes whenever their real-world distance falls under the conflict threshold.

* `led_radius_meters`: Defines the physical coverage radius of a streetlight. Modifying the conflict threshold (e.g., distance < `2 * led_radius_meters` vs `1.0 * led_radius_meters`) alters the strictness of the spatial non-overlap constraint, simulating real-world urban illumination tolerance.
* `start_index` & `limit`: Data slicing parameters. Adjust these to shift the array extraction window within `data/paranainterativo.csv`, enabling topology testing across different urban clusters.

### 3. Stochastic Optimization Engines (`src/lattice_network_evolution.py`)
Configures the classical-hybrid drivers used to minimize the Ising energy landscape and search for the MIS.

* **Simulated Annealing Schedule:** Adjust the initial temperature and cooling rate to control the thermodynamic probability $\exp(-\Delta E / T)$ of accepting higher-energy states, enabling the escape from local minima.
* **Multi-start Heuristics:** Modify the number of random initializations to map broader state spaces and improve global minimum convergence.
* **Shot-Noise Tolerance (`epsilon`):** Calibrate this threshold filter to differentiate genuine state energy improvements from the statistical variance of the quantum simulator's finite sampling.

---

## How to Run the Pipeline
Ensure you run these commands from the **root directory** of the project:

1. **Build QUBO constraints and extract data:** `python src/real_graph_qubo.py`
2. **Execute Optimization Engines:** `python src/lattice_network_evolution.py`
3. **Plot Algorithmic Convergence:** `python src/plot_convergence.py`
4. **Plot MIS Geographic Map:** `python src/plot_mis_map.py`
5. **Run the Unit Test Suite:** `python -m unittest discover -s tests -v`

---

## Technical Documentation Index

Deep dive into the mathematical and theoretical foundations of this research:

* [📄 Single Qubit Evolution](docs/single_cell_evolution.md): PoC for Hamiltonian evaluation and Ground State convergence.
* [📄 2D Lattice Network](docs/network_evolution.md): Expansion to a spatial grid coordinating multiple qubits.
* [📄 QAOA Justification](docs/qaoa_justification.md): Theoretical essay on classical bottlenecks and the transition to quantum superposition.
* [📄 QUBO Formalization](docs/qubo_evolution.md): Detailed explanation of the QUBO-to-MIS mapping and the spatial exclusion (redundancy) penalties that enforce it.
* [📄 Results & Methodology](docs/results_and_methodology.md): Haversine distance application and topological validation (Real vs. Simulated).

---

## Real-World Data Extraction (Prado Velho / PUCPR)
![Mapa dos postes no Prado Velho](assets/mapa_prado_velho.png)
*Fonte: [Portal Paraná Interativo - Governo do Estado do Paraná](https://paranainterativo.pr.gov.br/portal/apps/webappviewer/index.html?id=f282d1c181c0405789e2f65c17ac274d)*