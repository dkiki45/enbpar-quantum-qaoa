# Research Project: Quantum Optimization for LED Retrofitting (ENBPar)

**Scientific Initiation Research (PIBIC) - PUCPR**
**Student:** David Bobato Kikina | **Advisors:** Profs. Jonas Krause and Rodrigo Pasti

## Overview
This repository contains the algorithms and technical documentation for the development of a Quantum Combinatorial Optimization model. The objective is to create a system capable of intelligent planning (retrofitting) for ENBPar's public lighting infrastructure, utilizing the **Quantum Approximate Optimization Algorithm (QAOA)** to solve large-scale logistical bottlenecks that are intractable for classical computing.

---

## 📂 Code and Documentation Index

The project was built in a modular approach. Below are the links to the detailed documentation and source code for each evolutionary stage of the research:

### 1. Single Quantum Individual Evolution
* **Source Code:** `single_cell_evolution.py`
* **Technical Documentation:** [📄 Read doc_single_cell.md](Documentação-single-cell-ev.md)
* **Summary:** Initial Proof of Concept (PoC). It integrates the biological data structure (`LatticeCell`) with a linear physical evaluator (Hamiltonian) guided by a *Hill Climbing* algorithm. It demonstrates that a single "blind" Qubit can minimize its energy and autonomously converge to the *Ground State*.

### 2. Global Optimization in Spatial Grid (2D Lattice)
* **Source Code:** `lattice_network_evolution.py`
* **Technical Documentation:** [📄 Read doc_lattice_network.md](Documentação-network-ev.md)
* **Summary:** Expansion of the single-individual minimization algorithm to an interconnected grid (2x2 LED matrix). The evolutionary engine now operates on spatial coordinates, and the Hamiltonian measures the Global Energy of the circuit. It proves that the algorithm can coordinate multiple Qubits simultaneously to find the optimal state of the network.

### 3. Theoretical Foundation: The Classical Bottleneck and the Transition to QAOA
* **Academic Documentation:** [📄 Read doc_justificacao_qaoa.md](doc_justificacao_qaoa.md)
* **Summary:** A theoretical essay that exposes the limitations of the classical architecture implemented in the previous steps. The text scientifically justifies the need to adopt QAOA to mitigate "stochastic inefficiency" through massively parallel processing via superposition and phase interference.

### 4. QUBO Model and Neighborhood Interaction
* **Source Code:** `qubo_network_evolution.py`
* **Technical Documentation:** [📄 Read doc_qubo_evolution.md](doc_qubo_ev.md)
* **Summary:** Upgrades the model to a Quadratic Unconstrained Binary Optimization (QUBO) architecture. Introduces coupled penalty terms ($Z_iZ_j$) for adjacent LEDs, creating spatial frustration. Demonstrates the algorithm's capability to balance linear rewards with quadratic penalties to resolve spatial conflicts and find complex distribution patterns.

