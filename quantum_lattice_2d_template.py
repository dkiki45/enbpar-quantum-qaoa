from dataclasses import dataclass, field
from typing import Optional
from qiskit import QuantumCircuit
import numpy as np
import random

@dataclass
class LatticeCell:

    # Classical value
    value: int = 0

    # Quantum information
    theta: float = 0.0
    phi: float = 0.0

    # Evolutionary information
    fitness: float = 0.0
    occupied: bool = True
    #age: int = 0
    #species: int = 0

    # Qiskit mapping
    qubit_index: Optional[int] = None

    # User-defined information
    #metadata: dict = field(default_factory=dict)

    #def __post_init__(self):
    #    if self.metadata is None:
    #        self.metadata = {}


    def delete_individual(self):
        """
        Remove the individual from this cell.
        The qubit still exists physically, but the cell becomes empty
        in the evolutionary lattice.
        """
        self.occupied = False
        self.fitness = 0.0
        self.value = 0
        self.theta = 0.0
        self.phi = 0.0




class QuantumLattice2D:

    def __init__(self, rows, cols):

        self.rows = rows
        self.cols = cols
        self.qubit_map = {}
        self.metadata = {}
        self.cells = np.empty((rows, cols), dtype=object)

        q = 0
        for row in range(self.rows):
            for col in range(self.cols):

                self.cells[row, col] = LatticeCell(
                    qubit_index=q
                )

                self.qubit_map[q] = (row, col)

                q += 1


            
    @property
    def n_qubits(self):
        return self.rows * self.cols
    
    def pos_to_qubit(self, row: int, col: int) -> int:
        return self.cells[row, col].qubit_index


    def qubit_to_pos(self, qubit: int) -> tuple[int, int]:
        return self.qubit_map[qubit]

    '''
    def pos_to_qubit(self, row: int, col: int) -> int:
        """Map lattice position to Qiskit qubit index."""
        return row * self.cols + col

    def qubit_to_pos(self, qubit: int) -> tuple[int, int]:
        """Map Qiskit qubit index back to lattice position."""
        return divmod(qubit, self.cols)
    
    '''

    def set_value(self, row: int, col: int, value: int):
        self.values[row, col] = value

    def get_value(self, row: int, col: int) -> int:
        return int(self.values[row, col])
    
    
    def print_occupancy(self):

        matrix = np.zeros(
            (self.rows, self.cols),
            dtype=int
        )

        for i in range(self.rows):
            for j in range(self.cols):

                if self.cells[i, j].occupied:
                    matrix[i, j] = 1

        print(matrix)

