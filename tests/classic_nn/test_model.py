import unittest
from src.classic_nn.model import ClassicNN

class TestClassicNN(unittest.TestCase):

    def setUp(self):
        self.model = ClassicNN()

    def test_model_creation(self):
        self.assertIsNotNone(self.model)

    def test_model_compile(self):
        try:
            self.model.compile()
        except Exception as e:
            self.fail(f"Compilation failed with error {e}")

if __name__ == '__main__':
    unittest.main()