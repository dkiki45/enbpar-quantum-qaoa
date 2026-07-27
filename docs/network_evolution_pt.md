# Technical Documentation: Lattice Network Evolution

**Script Objective:**
To scale the single-individual quantum minimization algorithm to an interconnected network (2D Lattice). The code proves that the *Hill Climbing* algorithm can optimize the Global Energy of a multiple-Qubit system simultaneously, finding the network's *Ground State*.

---

## 1. The Network Structure (`QuantumLattice2D`)
* **What it is:** The "map" or board that organizes the cells (`LatticeCell`) into geographical coordinates (rows and columns).
* **Function:** In this experiment, we instantiated a 2x2 grid (representing 4 streetlights). The class maps physical coordinates (e.g., row 0, column 1) to the logical Qubit indices in Qiskit (e.g., Qubit 1).
* **Visualization (`print_classical_state`):** Converts continuous quantum states ($\theta$ angles) into discrete classical states ($0$ or $1$) and prints a visual matrix representing which LEDs are turned on or off.

## 2. The Global Hamiltonian (City Evaluation)
* **What changes:** Instead of evaluating just one Qubit, the Hamiltonian is now a sum of the energies of all individuals in the grid.
* **Test Equation:** $H = 1.0 \times Z_0 + 1.0 \times Z_1 + 1.0 \times Z_2 + 1.0 \times Z_3$.
* **Target (Ground State):** Since each Z gate has a minimum of $-1.00$, the perfect ground state for this 4-light network is an exact Global Energy of **$-4.0000$**.

## 3. Network Evolutionary Algorithm (`hill_climbing_lattice`)
The evolutionary dynamics now operate on the community, not just the individual:
1. **Random Selection:** At each generation, the AI randomly selects any "streetlight" (a cell) on the map to undergo mutation.
2. **Local Mutation:** A Gaussian noise is applied only to the angle of that chosen cell.
3. **Global Evaluation:** The entire circuit (4 Qubits) is built. The Hamiltonian runs and evaluates if the energy of the *entire city* has improved.
4. **Decision:**
   * If the change in that specific streetlight decreased the total cost of the network, the mutation is **ACCEPTED**.
   * If the change worsened the total cost, the mutation is **REJECTED**, and the streetlight returns to its previous state.

---

## Experimental Conclusion
The algorithm demonstrated the ability to coordinate multiple Qubits simultaneously. Starting with a chaotic matrix of random angles (energy far from ideal), iterative natural selection guided the system to the perfect configuration. At the end of the generations, the global energy locked at `-4.0000`, and the LED matrix collapsed to the ideal classical state, proving the scalability of the combinatorial optimization model.
