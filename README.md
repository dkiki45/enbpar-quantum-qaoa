# Research Project: Quantum Optimization for LED Retrofitting (ENBPar)
*PIBIC PUCPR - Student: David Bobato Kikina | Advisors: Profs. Jonas Krause and Rodrigo Pasti*

This repository maps public lighting planning as a **Maximum Independent Set (MIS)** problem: choose the largest set of streetlights with no two closer than a conflict distance. It is solved with **QAOA** (Qiskit) and checked against an exact classical solver (ILP).

> **Branches**
> * `main`: the current pipeline only (extended grid search, campaign runner, reports).
> * `arquivo-campanha-gpu`: frozen snapshot of the code that produced Reports 04 to 06, including the older scripts removed from `main` (`grid_search.py`, `run_warm_start.py`, `run_qaoa.py`, `spsa_study.py`, `plot_analysis.py`, `validate_samplers.py`, `qaoa_circuit.py`, `graph_visualization.py`).
> * `v1-otimizadores-heuristicos`: the legacy classical heuristic optimizers.

## Pipeline

1. **Problem:** `src/core/graph_builder.py` reads the streetlight CSV, builds the conflict graph (Haversine distance, 20 m × tolerance) and takes a BFS-connected subgraph of `LIMIT_NODES` vertices from `START_NODE`. `src/core/qubo_formalization.py` builds the MIS QUBO (penalty α = 1, β = 2) and its Ising form.
2. **Exact reference:** `src/core/classical_baseline.py` solves the MIS with ILP (`scipy.optimize.milp`, HiGHS); a min-degree greedy heuristic is recorded too.
3. **Initial angles:** `src/experiments/grid_search_ext.py` searches the exact QAOA energy landscape layer by layer (50×50 grid + zoom, up to p = 3) and saves the best angles (`summary.json`).
4. **Campaign:** `src/experiments/run_campaign.py` starts QAOA from the grid angles (per-seed perturbation of ±0.05 rad) and runs one of three modes:
   * `--mode none` (A): no optimization, evaluate the grid point;
   * `--mode shots` (B): COBYLA on the energy estimated with `--shots` shots;
   * `--mode exact` (C): COBYLA on the exact statevector energy.

   Every run ends with the same evaluation: exact energy, 10k independent shots, P(optimum), P(feasible), approximation ratios, TTS99 and times.
5. **Reports:** `src/experiments/campaign_report.py` (tables of every campaign) and `src/experiments/abc_report.py` (paired A/B/C comparison with 95% CIs and plot).

Shared engine: `src/core/qaoa_solver.py` (`FastAerSampler`, `run_qaoa`), `src/core/evaluators.py` (exact and shot-based energy evaluators), `src/core/sim_backend.py` (Aer method/device options, GPU check), `src/core/solution_decoder.py` (feasibility and probabilities).

Tools: `src/experiments/bench_devices.py` (CPU vs GPU vs tensor network timing), `src/experiments/ilp_limits.py` (how far the ILP scales), `scripts/gpu_test.py` (standalone GPU check).

## Configuration (`src/config.py`)

* `LIMIT_NODES`, `START_NODE`, `CSV_PATH`: the instance (number of qubits and where the subgraph starts).
* `SIMULATION_METHOD`, `SIMULATION_DEVICE`, `SIMULATION_PRECISION`, `SIMULATION_CUSTATEVEC`: Aer defaults. The campaign and the grid search take `--method` / `--device` instead.
* `SHOTS_PER_EVAL`: default shots of `bench_devices.py`.
* `RESULTS_BASE_DIR`: results folder (`src/results/`, ignored by git, like `logs/`).

## How to Run

CPU install (Python 3.11):
```bash
pip install -r requirements.txt
PYTHONPATH=src pytest
```

GPU (NVIDIA, Linux), in a separate env:
```bash
conda create -n qaoa-gpu --clone qaoa -y
conda activate qaoa-gpu
pip uninstall -y qiskit-aer
pip install -r requirements-gpu.txt
python scripts/gpu_test.py      # checks the GPU and compares CPU vs GPU time
```

Extended grid search (check the time first with `--estimate`):
```bash
PYTHONPATH=src python -u src/experiments/grid_search_ext.py --method statevector --device GPU --estimate
PYTHONPATH=src python -u src/experiments/grid_search_ext.py --method statevector --device GPU --max-p 3
```

Campaigns (run inside tmux; every step resumes after a crash):
```bash
bash scripts/campaign.sh smoke       # ~5 min check with 12 qubits
bash scripts/campaign.sh all         # Report 05: depth, grid effects, extra layers, generalization, report
bash scripts/campaign.sh smoke_abc   # ~5 min check of the A/B/C modes
bash scripts/campaign.sh abc         # Report 06: A/B/C at p = 3, 20 seeds, paired report
```

Benchmarks:
```bash
PYTHONPATH=src python -u src/experiments/bench_devices.py
PYTHONPATH=src python -u src/experiments/ilp_limits.py --kinds csv,geo,er3,er6 --sizes 30,60,125,250,500,1000
```

Note: `grid_search_ext.py` always builds the graph from `START_NODE` with tolerance 2.0 (40 m). The generalization runs (`--start-node`, `--tolerance` in `run_campaign.py`) reuse those angles on other graphs on purpose.
