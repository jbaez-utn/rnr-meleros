from libs import utils
from libs import utils4e
from libs import learning4e
from libs import probabilistic_learning

class ClassicBaseNN:
    def __init__(self):
        self.model = Sequential()

    # Build base model 4 input nodes, 1 hidden layer with 4 nodes, 1 output node
    def build_model(self):
        self.model.add(Dense(4, input_dim=4, activation='relu'))
        self.model.add(Dense(2, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))

    def compile_model(self):
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Train model method
    def fit(self, X_train, y_train, epochs, validation_data):
        self.model.fit(X_train, y_train, epochs=epochs, validation_data=validation_data)

    def get_model(self):
        return self.model