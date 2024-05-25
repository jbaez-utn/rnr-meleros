import sys
import unittest

# Add the relative path to sys.path
sys.path.append('../../src')

import dataset_manipulation

class TestDatasetManipulation(unittest.TestCase):
    def test_comportamientos_translation(self):
        expected_dict = {
            "D": "descanso",
            "IM": "inmovil",
            "A": "alerta",
            "AA": "auto-acicalamiento",
            "AL": "alimentacion",
            "E": "exploracion",
            "L": "locomocion",
            "LR": "locomocion-repetitiva",
            "LA": "locomocion-ascenso",
            "LD": "locomocion-descenso",
            "LI": "locomocion-invertida",
            "O": "otros"
        }
        self.assertEqual(dataset_manipulation.comportamientos_translation, expected_dict)

if __name__ == '__main__':
    unittest.main()