# Technical Documentation: QUBO Lattice Evolution (Neighborhood Interaction)

**Script Objective:**
To upgrade the Quantum Optimization model from a linear evaluator to a **QUBO (Quadratic Unconstrained Binary Optimization)** model. This script introduces neighborhood constraints, proving the algorithm can resolve spatial conflicts (frustration) and find optimal distribution patterns in a 2D grid.

---

## 1. The Mathematical Shift: From Linear to QUBO
In previous iterations, the Hamiltonian evaluated each Qubit in isolation. The system learned to turn all LEDs on because there was no spatial restriction. 

In real-world infrastructure (like ENBPar's street lighting), having adjacent lights at maximum power is a waste of energy. To model this, we transitioned to a QUBO architecture, which maps the problem into an Ising-like model grid .

## 2. The New Hamiltonian Structure
The Hamiltonian is now divided into two competing forces:

* **Linear Terms (The Reward):** $H_{linear} = 1.0 \times Z_i$
  * The system is rewarded (energy decreases) when it turns an LED on ($|1\rangle$). This prevents the algorithm from simply turning off the entire city to avoid penalties.
* **Quadratic Terms (The Penalty / Edges):** $H_{quadratic} = 2.0 \times Z_i Z_j$
  * This is the core of QUBO. If Qubit $i$ and its neighbor Qubit $j$ are both turned on at the same time, their multiplication triggers a massive penalty (coefficient of $2.0$), overriding the linear reward and causing a spike in Global Energy.

## 3. Evolutionary Dynamics and "Frustration"
By implementing competing rules, we introduce **frustration** into the system. The Hill Climbing algorithm can no longer rely on a simple "turn everything on" strategy. 
1. If it turns all LEDs on: It gets linear rewards but suffers catastrophic quadratic penalties.
2. If it turns all LEDs off: It avoids penalties but gets zero linear rewards (sub-optimal energy).

## Experimental Conclusion
Faced with the QUBO constraints, the evolutionary algorithm navigates the complex energy landscape and naturally converges to an **alternating pattern** (a checkerboard grid). This ensures maximum illumination coverage without any adjacent overlaps. 

This experiment successfully validates the model's ability to handle complex, coupled spatial restrictions, which is the foundational requirement for routing and retrofitting large-scale smart city grids.