# Research Project: Quantum Optimization for LED Retrofitting (ENBPar)
*PIBIC PUCPR - Student: David Bobato Kikina | Advisors: Profs. Jonas Krause and Rodrigo Pasti*

> *Note: The `main` branch contains the formal QAOA architecture (Phase P0). The legacy classical heuristic optimizers are archived in the `v1-otimizadores-heuristicos` branch.*

This repository maps public lighting planning as a **Maximal Independent Set (MIS)** mathematical problem. The goal is to use the **QAOA** quantum algorithm to find the maximum number of streetlights that can remain active without their illumination areas overlapping spatially.

**Phase P0: Formal QAOA Implementation**
* **QUBO to Ising:** Exact mathematical conversion using `SparsePauliOp` ($Z$ and $ZZ$ matrices).
* **Quantum Circuit:** Explicit Ansatz alternating Cost ($RZZ$) and Mixer ($RX$) gates.
* **Hybrid Engine:** Integration of the `COBYLA` classical optimizer with the `StatevectorSampler` quantum simulator.
* **Geographic Data:** Reading real CSV coordinates and calculating physical distance constraints via the Haversine formula.
* **Absolute Validation:** Solution decoder paired with an exact brute-force algorithm to certify the accuracy rate.

**Directory Structure**
* `src/core/`: All quantum physics logic, graph processing, and solution decoding.
* `src/experiments/run_qaoa.py`: Main orchestrator file to run the simulation and generate JSON/graph reports.
* `tests/`: Rigorous test suite ensuring mathematical and structural parity of the project.

**How to Run**

Run the end-to-end main simulation:
```bash
PYTHONPATH=src python src/experiments/run_qaoa.py
```

Run the validation test suite:
```bash
PYTHONPATH=src pytest
```