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

**Phase P3: GPU Acceleration & Scaling Tools**
* **GPU Simulation:** `SIMULATION_DEVICE = "GPU"` runs the Aer `statevector` method on an NVIDIA GPU (`qiskit-aer-gpu-cu11`, same Aer 0.17.2 as the CPU build, installed in a separate env from `requirements-gpu.txt`). `matrix_product_state` has no GPU implementation. A 30-qubit statevector needs 16 GB in double precision (8 GB in single).
* **Separate Result Folders:** any simulator other than MPS + CPU + double writes to folders with a suffix (e.g. `warm_start_n30_COBYLA_statevector_GPU`), so `SKIP_COMPLETED_RUNS` never mixes or skips runs from different simulators. Every `summary.json` records the simulator used.
* **CPU vs GPU Benchmark (`bench_devices.py`):** time and memory of one QAOA evaluation for MPS-CPU, statevector-CPU and statevector-GPU (with/without cuStateVec, single precision), same circuit, angles and shots, for $p = 1 \dots 6$.
* **ILP Limits (`ilp_limits.py`):** how far the exact baseline scales on the real streetlight graph, geometric graphs and dense random graphs, reporting proven optimality or the remaining gap at a time limit.
* **Deeper Circuits:** warm-start now runs $p = 3, 4, 5, 6$.
* **Extended Grid Search (`grid_search_ext.py`):** exact (shot-free) QAOA energy landscape on fine grids, layer by layer up to $p$ = `--max-p`, with zoom refinement, sampling at the best angles (P(feasible), P(optimum) vs ILP) and resume support. Works with `statevector` or `tensor_network` on the GPU.
* **Tensor Network:** `SIMULATION_METHOD = "tensor_network"` (GPU only, NVIDIA cuTensorNet). `bench_devices.py` reports the exact energy of every method at the same angles, so tensor network and statevector can be compared directly.
* **Standalone GPU Test (`gpu_test.py`):** checks the GPU and compares CPU vs GPU time for 20-30 qubits, no project code needed.

**Branch `motor-quantico-led`: the quantum engine applied to the LED (MIS) problem**
* `main` keeps the reference code; this branch carries the LED-specific work.
* **Warm-start from the extended grid:** `WARM_START_SOURCE = "ext"` makes `run_warm_start.py` start from the best angles of every layer found by `grid_search_ext.py` (e.g. $p=1..3$); deeper layers start at 0. Results go to `warm_start_n<N>_<optimizer>..._gridext`, separate from the `"v3"` runs.

**Problem-specific (LED / MIS) parts of the code** — what changes to apply the engine to another binary problem:
* `src/core/graph_builder.py`: reads the streetlight CSV, builds the conflict graph (Haversine distance) and the BFS subgraph.
* `src/core/qubo_formalization.py`: `build_mis_qubo`, `classical_cost`, `conflicting_edges` (MIS objective and penalty $\alpha=1$, $\beta=2$). `IsingModel` and `qubo_to_ising` are generic.
* `src/core/classical_baseline.py`: exact MIS solvers (ILP $x_i + x_j \le 1$, brute force).
* `src/core/solution_decoder.py`: feasibility = no conflicting edge, mapping back to streetlight coordinates (`integer_to_bits` and the probability helpers are generic).
* `src/core/graph_visualization.py` and `plot_analysis.py`: geographic plots of the streetlights.
* `src/config.py`: `CSV_PATH`, `LIMIT_NODES`, `START_NODE`.
* Experiment scripts (`run_qaoa.py`, `run_warm_start.py`, `grid_search*.py`, `spsa_study.py`, `validate_samplers.py`, `bench_devices.py`) call the functions above to build the problem; `ilp_limits.py` is MIS-only.
* Generic engine (no change needed): `qaoa_solver.py`, `sim_backend.py`, `qaoa_circuit.py`, the GPU/benchmark tooling and the warm-start/grid machinery.

**Configuration (`src/config.py`)**
* `LIMIT_NODES`: number of vertices in the subgraph = number of qubits.
* `START_NODE`: CSV index where the BFS subgraph selection starts.
* `SIMULATION_METHOD`: `"statevector"` or `"matrix_product_state"`.
* `SIMULATION_DEVICE`: `"CPU"` or `"GPU"` (GPU requires `"statevector"`).
* `SIMULATION_PRECISION`: `"double"` or `"single"` (statevector only; single halves the memory).
* `SIMULATION_CUSTATEVEC`: GPU only, use NVIDIA cuStateVec kernels.
* `SKIP_COMPLETED_RUNS`: resume interrupted experiments by skipping runs that already have a `summary.json`.
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

Warm-start pipeline (grid search for $p=1,2$ angles, then deeper circuits):
```bash
PYTHONPATH=src python src/experiments/grid_search.py
PYTHONPATH=src python src/experiments/run_warm_start.py COBYLA
```

**GPU (NVIDIA, Linux only)**

Create a separate env so CPU runs are not affected, then check the GPU:
```bash
conda create -n qaoa-gpu --clone qaoa -y
conda activate qaoa-gpu
pip uninstall -y qiskit-aer
pip install -r requirements-gpu.txt
nvidia-smi    # must show the GPU table, not an error
```

CPU vs GPU benchmark, then a GPU warm-start run (set `SIMULATION_METHOD = "statevector"` and `SIMULATION_DEVICE = "GPU"` in `src/config.py` first):
```bash
PYTHONPATH=src python -u src/experiments/bench_devices.py
PYTHONPATH=src python -u src/experiments/run_warm_start.py COBYLA
```

Extended grid search on the GPU (check the time first with `--estimate`):
```bash
PYTHONPATH=src python -u src/experiments/grid_search_ext.py --method statevector --device GPU --estimate
PYTHONPATH=src python -u src/experiments/grid_search_ext.py --method statevector --device GPU --max-p 3
```

Report 04 feedback campaign (GPU, ~52 h, resumable): depth with 20 seeds, grid effects, layer expansion and generalization, each run re-evaluated with exact energy and 10k independent shots (P(optimum), P(feasible), valid size, evaluations, time):
```bash
bash scripts/campaign.sh smoke   # ~5 min check first
bash scripts/campaign.sh all     # exp1 -> exp2 -> exp3 -> exp4 -> report (src/results/campaign/report.md)
```

ILP scaling limits:
```bash
PYTHONPATH=src python -u src/experiments/ilp_limits.py --kinds csv,geo,er3,er6 --sizes 30,60,125,250,500,1000
```