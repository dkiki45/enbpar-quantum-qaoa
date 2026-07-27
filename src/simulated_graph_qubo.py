"""
=============================================================================
SCRIPT: Simulated City QUBO Mapping (Control Environment)
=============================================================================
DESCRIPTION:
Generates an artificial geometric graph representing a "perfect" neighborhood 
with 150 streetlights. This serves as a control group to compare the quantum 
algorithm's performance against the noisy, real-world data from the IPPUC.
Coordinates are generated procedurally on a 2D Cartesian plane (meters).
=============================================================================
"""

import numpy as np

# ============================================================
# 1. GENERATE ARTIFICIAL STREETLIGHTS
# ============================================================
print("\n--- STARTING QUBO MAPPING (SIMULATED CITY) ---")
NUM_QUBITS = 150
AREA_SIZE = 500.0 # 500x500 meters area

# Procedurally generate random (X, Y) coordinates in meters
np.random.seed(42) # Seed ensures the "random" city is the same every time we run
simulated_coords = np.random.rand(NUM_QUBITS, 2) * AREA_SIZE
print(f"Successfully generated artificial streetlights: {NUM_QUBITS}")

# ============================================================
# 2. GRAPH CONSTRUCTION (Mapping Neighbors using Euclidean Distance)
# ============================================================
# Since coordinates are already in meters, we use simple Euclidean distance
DISTANCE_THRESHOLD = 45.0 
edges = []

def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

for i in range(NUM_QUBITS):
    for j in range(i + 1, NUM_QUBITS):
        dist = euclidean_distance(simulated_coords[i], simulated_coords[j])
        if dist < DISTANCE_THRESHOLD:
            edges.append((i, j))

print(f"Connections (streets/edges) found: {len(edges)}")

# ============================================================
# 3. GENERATING THE QUBO HAMILTONIAN
# ============================================================
qubo_terms = []

# Linear Terms (Z_i)
for i in range(NUM_QUBITS):
    qubo_terms.append({"coefficient": 1.0, "pauli": "Z", "qubits": (i,)})

# Quadratic Terms (Z_i * Z_j)
for edge in edges:
    qubo_terms.append({"coefficient": 2.0, "pauli": "ZZ", "qubits": edge})

print(f"\nTotal mathematical constraints generated: {len(qubo_terms)}")
print("\nExample of the first 5 spatial conflict rules (Penalties):")
for t in qubo_terms[150:155]: 
    print(f"-> Penalty generated between Qubit {t['qubits'][0]} and Qubit {t['qubits'][1]}")