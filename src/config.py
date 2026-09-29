from pathlib import Path

# ==========================================
# GLOBAL CONFIGURATIONS FOR QAOA PROJECT
# ==========================================

# 1. GRAPH PARAMETERS (Memory bound)
LIMIT_NODES = 30        # Number of nodes in the subgraph = number of qubits
START_NODE = 30         # CSV index of the node where the connected subgraph selection (BFS) starts
CSV_PATH = "src/data/paranainterativo.csv"

# 2. STATISTICAL PARAMETERS 
SHOTS_PER_EVAL = 1024
SEEDS_TO_TEST = list(range(1,5))  

# 3. CIRCUIT PARAMETERS
# Depths for the blind test (Random initialization)
DEPTHS_STANDARD = [1, 2, 3]
# Depths for the guided test (Warm-Start)
DEPTHS_WARM_START = [3, 4]

# Aer simulation method: "statevector" (exact, RAM = 16 bytes * 2^n -> up to 32 qubits on 128 GB)
# or "matrix_product_state" (exact, memory depends on entanglement -> can go beyond 32 qubits)
SIMULATION_METHOD = "matrix_product_state"

# Skip (seed, depth) runs that already have a summary.json, so an interrupted
# experiment resumes where it stopped. Delete the results folder to rerun from scratch.
SKIP_COMPLETED_RUNS = True

# 4. OPTIMIZERS
AVAILABLE_OPTIMIZERS = ["COBYLA", "SPSA"]

# 5. BASE DIRECTORIES
RESULTS_BASE_DIR = Path("src/results")
