import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def create_model(input_shape, output_shape, topology):
    model = Sequential()
    model.add(Dense(topology[0], activation='relu', input_shape=(input_shape,)))
    
    for units in topology[1:]:
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(0.2))
    
    model.add(Dense(output_shape, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_models(datasets, topologies):
    for i, dataset in enumerate(datasets):
        X, y = load_dataset(dataset)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        for j, topology in enumerate(topologies):
            model = create_model(X_train.shape[1], y_train.shape[1], topology)
            
            history = model.fit(X_train_scaled, y_train, 
                                epochs=100, 
                                batch_size=32, 
                                validation_split=0.2, 
                                callbacks=[tf.keras.callbacks.EarlyStopping(patience=10)])
            
            test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test)
            print(f"Dataset {i+1}, Topology {j+1} - Test accuracy: {test_accuracy:.4f}")
            
            model.save(f"model_dataset{i+1}_topology{j+1}.h5")

# Define the three different topologies
topologies = [
    [20],              # One hidden layer
    [30, 15],          # Two hidden layers
    [35, 25, 15]       # Three hidden layers
]

# Define the three datasets
datasets = ['general_dataset.csv', 'male_dataset.csv', 'female_dataset.csv']

# Train the models
train_models(datasets, topologies)