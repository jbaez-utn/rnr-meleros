import unittest
from src.rnn import train, model, data

class TestTrain(unittest.TestCase):
    def setUp(self):
        self.model = model.RNN()
        self.data = data.load_data()

    def test_train_model(self):
        initial_weights = self.model.get_weights()
        train.train_model(self.model, self.data)
        final_weights = self.model.get_weights()

        # Check that the model's weights have been updated during training
        self.assertNotEqual(initial_weights, final_weights)

if __name__ == '__main__':
    unittest.main()