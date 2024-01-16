from .model import RNN
from .data import load_data
from tensorflow.keras.optimizers import Adam

def train_model():
    # Load the data
    X_train, X_test, y_train, y_test = load_data()

    # Initialize the model
    model = RNN()

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    return model