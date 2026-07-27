# Technical Documentation: Single Cell Evolution

**Script Objective:**
To validate the integration between the evolutionary data structure (LatticeCell), the physical evaluator (Hamiltonian), and the minimization algorithm (Hill Climbing). The code proves that the system can train a "blind" Qubit to find the *Ground State* autonomously.

---

## 1. The Biological Structure (`LatticeCell`)
* **What it is:** The fundamental blueprint of an individual in our evolutionary ecosystem.
* **Function:** It stores the "DNA" of our LED, specifically the quantum rotation angle in the `theta` variable. It has the `fitness` property, which receives the physics evaluation.

## 2. The Physical Evaluator (Hamiltonian)
* **What it is:** The classical cost function mapped to the quantum environment.
* **Function:** We define a base Hamiltonian ($H = 1.0 \times Z_0$). The functions calculate the **Expectation Value** of the circuit. High energies represent "fines" (poor fitness), while low energies represent "savings" (good fitness).

## 3. The Integration Engine (`evaluate_cell_fitness`)
A function that connects the biological and physical worlds:
1. Extracts the `theta` from the cell.
2. Creates a 1-Qubit Qiskit circuit by applying an $R_y(\theta)$ gate.
3. Runs the circuit on the *Aer* simulator.
4. Calculates the Total Energy via the Hamiltonian.
5. Saves the energy to the cell's `fitness` property silently.

---

## 4. Evolutionary Algorithm: Hill Climbing (`hill_climbing_single_cell`)
This is the AI that iteratively minimizes the energy cost.

**The Life Cycle (Generations):**
1. **Starting Point:** The cell receives a completely random `theta` angle (Blind individual).
2. **Mutation:** The algorithm adds a Gaussian noise (e.g., $\pm 0.3$ radians) to the current angle.
3. **Evaluation:** The Hamiltonian runs the mutated circuit and returns the new energy.
4. **Natural Selection:**
   * **ACCEPTED (Improved):** If the new energy is lower (closer to $-1.00$), the mutation is accepted, and the cell's DNA is updated.
   * **REJECTED (Worsened):** If the energy increased, the algorithm undoes the mutation, and the cell returns to its previous safe state.

## Experimental Conclusion
The terminal demonstrated that no matter how bad the initial random angle is (Generation 0), the algorithm always rejects high-energy paths and converges perfectly to $\theta \approx 3.1415$ ($\pi$). This anchors the Qubit at the south pole of the Bloch Sphere ($|1\rangle$), successfully achieving the system's exact minimum energy ($-1.0000$), validating the core intelligence of our Quantum Optimization model.
