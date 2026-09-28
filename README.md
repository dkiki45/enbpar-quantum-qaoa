# Research Project: Quantum Optimization for LED Retrofitting (ENBPar)
*PIBIC PUCPR - Student: David Bobato Kikina | Advisors: Profs. Jonas Krause and Rodrigo Pasti*

> *Note: The `main` branch contains the formal QAOA architecture (Phases P0 & P1). The legacy classical heuristic optimizers are archived in the `v1-otimizadores-heuristicos` branch.*

This repository maps public lighting planning as a **Maximal Independent Set (MIS)** mathematical problem. The goal is to use the **QAOA** quantum algorithm to find the maximum number of streetlights that can remain active without their illumination areas overlapping spatially.

**Phase P0: Formal QAOA Implementation**
* **QUBO to Ising:** Exact mathematical conversion using `SparsePauliOp` ($Z$ and $ZZ$ matrices).
* **Quantum Circuit:** Explicit Ansatz alternating Cost ($RZZ$) and Mixer ($RX$) gates.
* **Hybrid Engine:** Integration of the `COBYLA` classical optimizer with the `StatevectorSampler` quantum simulator.
* **Geographic Data:** Reading real CSV coordinates and calculating physical distance constraints via the Haversine formula.
* **Absolute Validation:** Solution decoder paired with an exact brute-force algorithm to certify the accuracy rate.

**Phase P1: Scale, Metrics & Geospatial Integration**
* **Geospatial Payload:** Upgraded the solution decoder to map the optimal quantum bitstring back to the real-world IPPUC geographic coordinates (`id`, `latitude`, `longitude`), outputting a JSON ready for map plotting.
* **Optimizer Benchmark:** Executed a comparative analysis between `COBYLA` (gradient-free) and `SPSA` (stochastic). Proved that `COBYLA` strictly dominates in ideal, noiseless local simulations.
* **Circuit Depth Analysis:** Evaluated quantum depths ($p=1, 2, 3$) across multiple random seeds. Confirmed that shallow circuits ($p=1$) converge optimally under current noiseless constraints.
* **Graph Density Control:** Scaled up problem complexity by manipulating the Haversine tolerance factor, verifying the QAOA solver's efficacy on non-trivial, highly connected interference graphs.

**Phase P2: Scaling Beyond 30 Qubits**
* **Exact ILP Baseline:** Replaced the $O(2^n)$ brute-force validator with an exact Integer Linear Programming solver (`scipy.optimize.milp`, HiGHS branch-and-bound): maximize $\sum_i x_i$ subject to $x_i + x_j \le 1$ for every edge $(i, j)$, $x_i \in \{0, 1\}$. It proves optimality like brute force (same `exact_cost`, checked in `tests/test_exact_solvers.py`) but solves the full 125-node connected component in milliseconds. With ties, `exact_bits` may be a different optimal set with the same cardinality.
* **Connected Subgraph Selection:** The interference graph is a BFS-connected subgraph with `LIMIT_NODES` vertices starting from `START_NODE`.
* **Aer Simulation Backend:** `qiskit-aer` `SamplerV2` with a configurable `SIMULATION_METHOD`: `statevector` (exact, $16 \cdot 2^n$ bytes of RAM, up to 32 qubits on a 128 GB machine) or `matrix_product_state` (exact, memory scales with entanglement instead of qubit count).

**Configuration (`src/config.py`)**
* `LIMIT_NODES`: number of vertices in the subgraph = number of qubits.
* `START_NODE`: CSV index where the BFS subgraph selection starts.
* `SIMULATION_METHOD`: `"statevector"` or `"matrix_product_state"`.
* `SEEDS_TO_TEST`, `DEPTHS_STANDARD`, `DEPTHS_WARM_START`, `SHOTS_PER_EVAL`: experiment grid.

**Directory Structure**
* `src/core/`: All quantum physics logic, graph processing, and solution decoding.
* `src/experiments/run_qaoa.py`: Main orchestrator file to run the simulation and generate JSON/graph reports.
* `tests/`: Rigorous test suite ensuring mathematical and structural parity of the project.

**How to Run**

Install the dependencies (Python 3.11 recommended):
```bash
pip install -r requirements.txt
```

Run the end-to-end main simulation (optimizer: `COBYLA` by default, or `SPSA`):
```bash
PYTHONPATH=src python src/experiments/run_qaoa.py COBYLA
```

Run the validation test suite:
```bash
PYTHONPATH=src pytest
```