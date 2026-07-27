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
import os
import pandas as pd
import numpy as np

from qubo_formalization import build_qubo, qubo_to_ising, ALPHA_COVERAGE, BETA_REDUNDANCY

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
def get_real_qubo_terms(
    csv_path=None,
    limit=150,
    threshold=45.0,
    alpha=ALPHA_COVERAGE,
    beta=BETA_REDUNDANCY,
):
    """
    Reads real data, builds the neighborhood graph, and returns the Ising
    terms derived from the formalized QUBO objective:
 
        C(x) = -alpha * sum_i x_i + beta * sum_{(i,j) in E} x_i * x_j
 
    where x_i = 1 means streetlight i is ON. The QUBO is built explicitly
    (see qubo_formalization.build_qubo) and only then converted to Ising
    (Z / ZZ Pauli terms) via qubo_formalization.qubo_to_ising, instead of
    writing Z/ZZ coefficients by hand as before.
 
    Returns:
        qubo_terms: list of dicts {"coefficient", "pauli", "qubits"} -
                    same shape consumed by the rest of the pipeline.
        offset:     constant term picked up by the x_i -> Z_i substitution.
                    It doesn't change WHERE the minimum is, but it does
                    change the reported numeric value of the energy, so
                    callers that want the true C(x) value must add it back.
    """

    if csv_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(base_dir, "data", "paranainterativo.csv")


    print(f"\n--- EXTRACTING REAL DATA ({limit} streetlights) ---")
 
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"ERROR: File {csv_path} not found in the folder.")
        return [], 0.0
 
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
 
    # Formulate the QUBO first, then convert it to Ising.
    linear, quadratic, offset0 = build_qubo(n_vars=len(coords), edges=edges, alpha=alpha, beta=beta)
    ising_terms, offset = qubo_to_ising(linear, quadratic, offset0)
 
    qubo_terms = [
        {"coefficient": t.coefficient, "pauli": t.pauli, "qubits": t.qubits}
        for t in ising_terms
    ]
 
    return qubo_terms, offset
 
# ============================================================
# 3. STANDALONE EXECUTION (If running this file directly)
# ============================================================
if __name__ == "__main__":
    # Tests the extraction in isolation to ensure it is working
    terms, offset = get_real_qubo_terms()
    print(f"\nTotal mathematical constraints generated: {len(terms)}")
    print(f"Constant offset (from x_i = (1-Z_i)/2 substitution): {offset:+.4f}")
 
    zz_terms = [t for t in terms if t["pauli"] == "ZZ"]
    if zz_terms:
        print("\nExample of the first 5 spatial conflict rules (Penalties):")
        for t in zz_terms[:5]:
            print(f"-> Penalty between Qubit {t['qubits'][0]} and Qubit {t['qubits'][1]} (J={t['coefficient']:+.4f})")