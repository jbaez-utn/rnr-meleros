from model import ClassicNN
from data import load_data_single_row
from tensorflow.keras.optimizers import Adam

def train_classic_nn_model(csv_path, input_columns, output_columns):
    # Load the data
    X_train, X_test, X_val, y_train, y_test, y_val = load_data_single_row(csv_path, input_columns, output_columns)

    # Initialize the model
    model = ClassicNN()

    # Compile the model
    model.compile_model()

    # Train the model
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    return model

if __name__ == "__main__":

    # test with data from estudio3.csv
    csv_path = "/home/jbaez/Documents/utn/gintea/rnr-meleros/data/estudio3.csv"
    input_columns = ["X", "Y", "Z","ODBA"]
    output_columns = ["Comportamiento"]

    train_classic_nn_model(csv_path, input_columns, output_columns)