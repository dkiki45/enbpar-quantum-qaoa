import sys
import os
import unittest
import numpy as np

# Aponta para a nova pasta 'src'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Importa da nova arquitetura limpa
from core.lattice_model import QuantumLattice2D, LatticeCell

class TestQuantumLattice2D(unittest.TestCase):

    def setUp(self):
        """Sets up a clean environment before each test runs."""
        self.rows = 2
        self.cols = 2
        self.lattice = QuantumLattice2D(self.rows, self.cols)

    def test_initialization(self):
        """Tests if the lattice was created with the correct size and attributes."""
        self.assertEqual(self.lattice.n_qubits, 4)
        self.assertEqual(self.lattice.cells.shape, (2, 2))
        
        # Checks if the cells were instantiated as LatticeCell objects
        cell = self.lattice.cells[0, 0]
        self.assertIsInstance(cell, LatticeCell)
        self.assertEqual(cell.qubit_index, 0)
        
        # Verifica se a IA inicializou as mutações (theta) de forma aleatória
        self.assertTrue(0.0 <= cell.theta <= 2 * np.pi)

    def test_cell_attributes(self):
        """Tests if cell attributes can be accessed and modified directly."""
        cell = self.lattice.cells[1, 1]
        
        # Agora alteramos os atributos diretamente via POO em vez de usar set_value()
        cell.value = 1
        cell.fitness = 10.5
        cell.occupied = False
        
        # Verifica se o salvamento ocorreu perfeitamente
        self.assertEqual(self.lattice.cells[1, 1].value, 1)
        self.assertEqual(self.lattice.cells[1, 1].fitness, 10.5)
        self.assertFalse(self.lattice.cells[1, 1].occupied)
        
        # Verifica se as outras células permaneceram intactas
        self.assertEqual(self.lattice.cells[0, 0].value, 0)

    def test_qubit_mapping(self):
        """Tests if the coordinate to Qubit mapping dictionary is correct."""
        # Em uma matriz 2x2 (rodando linha por linha), a posição (1, 1) deve ser o Qubit 3
        self.assertEqual(self.lattice.cells[1, 1].qubit_index, 3)

        # Checa o dicionário reverso qubit_map criado no __init__
        pos = self.lattice.qubit_map[2]
        self.assertEqual(pos, (1, 0))

if __name__ == '__main__':
    unittest.main()