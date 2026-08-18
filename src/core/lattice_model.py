import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class LatticeCell:
    value: int = 0
    theta: float = 0.0
    phi: float = 0.0
    fitness: float = 0.0
    occupied: bool = True
    qubit_index: Optional[int] = None

class QuantumLattice2D:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.qubit_map = {}
        self.cells = np.empty((rows, cols), dtype=object)

        q = 0
        for row in range(self.rows):
            for col in range(self.cols):
                self.cells[row, col] = LatticeCell(qubit_index=q)
                # Initialize with a random angle for evolution to work on
                self.cells[row, col].theta = np.random.uniform(0, 2 * np.pi)
                self.qubit_map[q] = (row, col)
                q += 1

    @property
    def n_qubits(self):
        return self.rows * self.cols

    def print_classical_state(self):
        """Prints the grid showing 1 (LED On) or 0 (LED Off) based on Theta"""
        matrix = np.zeros((self.rows, self.cols), dtype=int)
        for i in range(self.rows):
            for j in range(self.cols):
                # If theta is closer to Pi (south pole), it collapses to 1. Otherwise 0.
                theta = self.cells[i, j].theta % (2 * np.pi)
                if np.pi/2 < theta < 3*np.pi/2:
                    matrix[i, j] = 1
                else:
                    matrix[i, j] = 0
        print(matrix)