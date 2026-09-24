from pathlib import Path

# ==========================================
# GLOBAL CONFIGURATIONS FOR QAOA PROJECT
# ==========================================

# 1. GRAPH PARAMETERS (Memory bound)
LIMIT_NODES = 35  
CSV_PATH = "src/data/paranainterativo.csv"

# 2. STATISTICAL PARAMETERS 
SHOTS_PER_EVAL = 1024
SEEDS_TO_TEST = list(range(1, 16))  

# 3. CIRCUIT PARAMETERS
# Depths for the blind test (Random initialization)
DEPTHS_STANDARD = [1, 2, 3]
# Depths for the guided test (Warm-Start)
DEPTHS_WARM_START = [3, 4, 5]

# 4. OPTIMIZERS
AVAILABLE_OPTIMIZERS = ["COBYLA", "SPSA"]

# 5. BASE DIRECTORIES
RESULTS_BASE_DIR = Path("src/results")