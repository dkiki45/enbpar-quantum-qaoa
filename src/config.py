from pathlib import Path

# ==========================================
# GLOBAL CONFIGURATIONS FOR QAOA PROJECT
# ==========================================

# 1. GRAPH PARAMETERS (Memory bound)
LIMIT_NODES = 30        # Number of nodes in the subgraph = number of qubits
START_NODE = 30         # CSV index of the node where the connected subgraph selection (BFS) starts
CSV_PATH = "src/data/paranainterativo.csv"

# 2. SHOTS (default of bench_devices.py; the campaign sets its own with --shots)
SHOTS_PER_EVAL = 1024

# 3. SIMULATOR (defaults; run_campaign.py, grid_search_ext.py and scripts/campaign.sh pass --method/--device)
# Aer simulation method: "statevector" (exact, RAM = 16 bytes * 2^n -> up to 32 qubits on 128 GB)
# or "matrix_product_state" (exact, memory depends on entanglement -> can go beyond 32 qubits)
SIMULATION_METHOD = "matrix_product_state"

# Aer device: "CPU" or "GPU". GPU needs the qaoa-gpu env (requirements-gpu.txt) and
# SIMULATION_METHOD = "statevector" or "tensor_network" (Aer has no GPU version of
# matrix_product_state; tensor_network runs only on the GPU).
SIMULATION_DEVICE = "CPU"
# "double" (16 bytes/amplitude: 30 qubits = 16 GB) or "single" (8 bytes: 31 qubits fit in a 24 GB GPU).
# Only used with "statevector".
SIMULATION_PRECISION = "double"
# GPU only: use NVIDIA cuStateVec (cuQuantum) kernels instead of Aer's own. Compare both with bench_devices.py
SIMULATION_CUSTATEVEC = False

# 4. BASE DIRECTORIES
RESULTS_BASE_DIR = Path("src/results")
