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
# 2. INTEGRATION FUNCTION (Exporting to the Pipeline)
# ============================================================
def get_real_qubo_terms(csv_path='paranainterativo.csv', limit=150, threshold=45.0):
    """
    Reads real data, builds the neighborhood graph, and returns the QUBO terms.
    This allows other files (like the quantum simulation) to import the data.
    """
    print(f"\n--- EXTRACTING REAL DATA ({limit} streetlights) ---")
    
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"ERROR: File {csv_path} not found in the folder.")
        return []

    # Former "Data Preparation" section
    df_ibm = df.head(limit).copy()
    coords = df_ibm[['latitude', 'longitude']].values
    
    # Former "Graph Construction" section
    edges = []
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            dist = calculate_distance_meters(coords[i][0], coords[i][1], coords[j][0], coords[j][1])
            if dist < threshold:
                edges.append((i, j))
                
    # QUBO Generation
    qubo_terms = []
    # LINEAR TERMS (Z_i)
    for i in range(len(coords)):
        qubo_terms.append({"coefficient": 1.0, "pauli": "Z", "qubits": (i,)})

    # QUADRATIC TERMS (Z_i * Z_j)
    for edge in edges:
        qubo_terms.append({"coefficient": 2.0, "pauli": "ZZ", "qubits": edge})
        
    return qubo_terms

# ============================================================
# 3. STANDALONE EXECUTION (If running this file directly)
# ============================================================
if __name__ == "__main__":
    # Tests the extraction in isolation to ensure it is working
    terms = get_real_qubo_terms()
    print(f"\nTotal mathematical constraints generated: {len(terms)}")
    
    if len(terms) > 150:
        print("\nExample of the first 5 spatial conflict rules (Penalties):")
        for t in terms[150:155]: 
            print(f"-> Penalty between Qubit {t['qubits'][0]} and Qubit {t['qubits'][1]}")