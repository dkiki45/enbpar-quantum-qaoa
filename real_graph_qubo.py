"""
=============================================================================
SCRIPT: Real City QUBO Mapping (Prado Velho / PUCPR)
=============================================================================
DESCRIPTION:
This script translates real geographical coordinates (Latitude/Longitude) into 
a Quantum Unconstrained Binary Optimization (QUBO) mathematical model. 
It calculates the physical distance between streetlights using the Haversine 
formula to establish a neighborhood graph (nodes and edges).

NOTE ON CLASSICAL SIMULATION:
This script DOES NOT execute the Hill Climbing algorithm or the Qiskit Aer 
simulator. Simulating 150 entangled qubits classically requires computing 
2^150 simultaneous states, which would instantly crash any classical RAM. 
Instead, this script mathematically prepares the exact constraints (Linear 
and Quadratic terms) to be sent and processed by a real Quantum Processing 
Unit (QPU) via the QAOA algorithm on IBM Quantum hardware.
=============================================================================
"""

import pandas as pd
import numpy as np

# ============================================================
# 1. MATHEMATICAL ENGINE: DISTANCE CALCULATION (HAVERSINE)
# ============================================================
def calculate_distance_meters(lat1, lon1, lat2, lon2):
    """Calculates the real distance in meters between two GPS coordinates."""
    R = 6371000  # Earth's radius in meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    
    a = np.sin(dphi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

# ============================================================
# 2. DATA PREPARATION (The 150 Qubit Cut)
# ============================================================
print("\n--- STARTING QUBO MAPPING (PRADO VELHO) ---")

# Loads the CSV and selects only the first 150 streetlights
df = pd.read_csv('paranainterativo.csv')
df_ibm = df.head(150).copy()
print(f"Successfully loaded streetlights: {len(df_ibm)}")

# ============================================================
# 3. GRAPH CONSTRUCTION (Mapping Neighbors)
# ============================================================
# If the distance between streetlights is less than 45 meters, they are neighbors (edges)
DISTANCE_THRESHOLD = 45.0 
edges = []
coords = df_ibm[['latitude', 'longitude']].values

# Compares each streetlight with all others to find neighbors
for i in range(len(coords)):
    for j in range(i + 1, len(coords)):
        dist = calculate_distance_meters(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
        if dist < DISTANCE_THRESHOLD:
            edges.append((i, j))

print(f"Connections (streets/edges) found: {len(edges)}")

# ============================================================
# 4. GENERATING THE QUBO HAMILTONIAN FOR IBM
# ============================================================
qubo_terms = []

# 4.1. LINEAR TERMS: Reward for turning the streetlight on (Z_i)
for i in range(len(coords)):
    qubo_terms.append({"coefficient": 1.0, "pauli": "Z", "qubits": (i,)})

# 4.2. QUADRATIC TERMS: Penalty for turning two neighbors on together (Z_i * Z_j)
for edge in edges:
    qubo_terms.append({"coefficient": 2.0, "pauli": "ZZ", "qubits": edge})

print(f"\nTotal mathematical constraints generated: {len(qubo_terms)}")
print("\nExample of the first 5 spatial conflict rules (Penalties):")
for t in qubo_terms[150:155]: 
    print(f"-> Penalty generated between Qubit {t['qubits'][0]} and Qubit {t['qubits'][1]}")