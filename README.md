# rnr-meleros
Red Neuronal Recurrente (Recurrent Neural Network RNN) aplicada sobre lecturas de acelerometro tomadas de Osos Meleros para predecir el comportamiento en tiempo real a partir de los valores medidos. 

---
Generado copilot

# Neural Network Application

This application trains two different types of neural networks: a classic neural network classifier and a recurrent neural network.

## Structure

The application is structured into two main parts: `classic_nn` and `rnn`. Each part has its own model definition, data processing, and training scripts.

## Setup

To set up the application, follow these steps:

1. Clone the repository.
2. Install the required Python packages using the command `pip install -r requirements.txt`.

## Usage

To run the application, use the command `python src/main.py`. This will initialize and train both the classic neural network and the recurrent neural network.

## Testing

Unit tests for both the classic neural network and the recurrent neural network are included in the `tests/` directory. To run the tests, use the command `python -m unittest discover tests`.

## Data

The data used to train the neural networks is stored in the `data/` directory. The `classic_nn` directory contains data for the classic neural network, and the `rnn` directory contains data for the recurrent neural network.

## Contributing

Contributions are welcome. Please submit a pull request with your changes.

## License

This project is licensed under the MIT License.