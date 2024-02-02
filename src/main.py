from classic_nn.model import ClassicNN
from data_ingestion.data import load_data as load_classic_data
from classic_nn.train import train_model as train_classic_model
from rnn.model import RNN
from rnn.data import load_data as load_rnn_data
from rnn.train import train_model as train_rnn_model

def main():
    # Load and preprocess data for classic neural network
    classic_data = load_classic_data()

    # Initialize and compile classic neural network
    classic_nn = ClassicNN()
    classic_nn.build_model()
    classic_nn.compile_model()

    # Train classic neural network
    train_classic_model(classic_nn, classic_data)

    # Load and preprocess data for recurrent neural network
    rnn_data = load_rnn_data()

    # Initialize and compile recurrent neural network
    rnn = RNN()
    rnn.build_model()
    rnn.compile_model()

    # Train recurrent neural network
    train_rnn_model(rnn, rnn_data)

if __name__ == "__main__":
    main()