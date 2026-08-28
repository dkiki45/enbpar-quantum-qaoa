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