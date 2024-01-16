import unittest
from src.rnn.model import RNN

class TestRNNModel(unittest.TestCase):
    def setUp(self):
        self.model = RNN()

    def test_model_build(self):
        self.assertIsNotNone(self.model.build())

    def test_model_compile(self):
        self.assertIsNotNone(self.model.compile())

if __name__ == '__main__':
    unittest.main()