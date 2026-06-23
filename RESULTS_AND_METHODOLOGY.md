# QUBO Mapping: Real vs. Simulated Topology

This repository contains the data preparation and mathematical modeling (QUBO Hamiltonian) step for the public lighting optimization problem using quantum computing. 

To validate the robustness of the QAOA algorithm that will be executed on IBM Quantum hardware, the modeling was divided into two distinct scenarios (a test group and a control group). Both scenarios use the 150 Qubit limit to fit the capacity of the quantum processor.

## 1. Real Topology (Prado Velho / PUCPR)
* **File:** `real_graph_qubo.py`
* **Data Source:** IPPUC Open Data Portal (GeoCuritiba).
* **Mathematics:** Distance calculation via Haversine Formula using GPS coordinates (Latitude/Longitude).
* **Results:** We identified **746 connections (edges)** with a distance of less than 45 meters. This generated a dense Hamiltonian with **896 mathematical constraints** (150 linear $Z$ terms + 746 quadratic $ZZ$ penalties).
* **Characteristic:** Represents real-world "noise". The streetlights are grouped irregularly along the organic urban layout.

## 2. Simulated Topology (Control Group)
* **File:** `simulated_graph_qubo.py`
* **Data Source:** Random procedural generation on a 500x500 meter Cartesian plane.
* **Mathematics:** Euclidean distance calculation.
* **Results:** We identified **161 connections (edges)** with a distance of less than 45 meters. This generated a sparser Hamiltonian with **311 mathematical constraints** (150 linear $Z$ terms + 161 quadratic $ZZ$ penalties).
* **Characteristic:** Represents a mathematically perfect and uniform environment. 

## Why two approaches? (Scientific Justification)
In network science and quantum engineering, proving that an algorithm works requires isolating variables. 

1. **Hardware Agnosticism:** By sending completely different density matrices (896 constraints vs. 311 constraints) to the IBM QPU, we prove that our QUBO model and optimization algorithm are agnostic to geography. They solve the problem purely based on the adjacency matrix.
2. **Impact of Density:** The simulated model acts as a baseline. It will allow us to compare how quantum noise (decoherence) affects the results when the matrix is simple (simulated) versus when the matrix is highly connected and complex (real).