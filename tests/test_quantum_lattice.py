import unittest
from quantum_lattice_2d_template import QuantumLattice2D, LatticeCell

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
        self.assertIsInstance(self.lattice.cells[0, 0], LatticeCell)
        self.assertEqual(self.lattice.cells[0, 0].qubit_index, 0)

    def test_set_and_get_value(self):
        """Tests the set_value and get_value"""
        # Sets a classical value (e.g., turns on the streetlight at row 1, col 1)
        self.lattice.set_value(1, 1, 1)
        
        # Checks if the value was saved and read correctly
        self.assertEqual(self.lattice.get_value(1, 1), 1)
        
        # Checks if the other cells remain turned off (0)
        self.assertEqual(self.lattice.get_value(0, 0), 0)

    def test_qubit_mapping(self):
        """Tests if the coordinate to Qubit mapping conversion is correct."""
        # In a 2x2 lattice, position (1, 1) should be Qubit 3
        qubit = self.lattice.pos_to_qubit(1, 1)
        self.assertEqual(qubit, 3)

        # Qubit 2 should be at position (1, 0)
        pos = self.lattice.qubit_to_pos(2)
        self.assertEqual(pos, (1, 0))

    def test_delete_individual(self):
        """Tests if the cell deletion/reset function works correctly."""
        self.lattice.cells[0, 0].fitness = 10.5
        self.lattice.cells[0, 0].value = 1
        
        self.lattice.cells[0, 0].delete_individual()
        
        self.assertFalse(self.lattice.cells[0, 0].occupied)
        self.assertEqual(self.lattice.cells[0, 0].fitness, 0.0)
        self.assertEqual(self.lattice.cells[0, 0].value, 0)

if __name__ == '__main__':
    unittest.main()