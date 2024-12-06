import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
import numpy as np
import json
import mlflow
import mlflow.keras
from mlflow.keras import MLflowCallback

# Dataset generico 1 segundo
estudio3_1seg_train = pd.read_csv('../../data/2-procesada/estudio3_normalizado_1seg_train.csv')
estudio3_1seg_test = pd.read_csv('../../data/2-procesada/estudio3_normalizado_1seg_test.csv')
estudio3_1seg_val = pd.read_csv('../../data/2-procesada/estudio3_normalizado_1seg_val.csv')

topologies = [
    [20],              # One hidden layer
    [30, 15],          # Two hidden layers
    [35, 25, 15]       # Three hidden layers
]

def create_model(input_shape, output_shape, topology, optimizer='Adam'):
    model = Sequential()
    model.add(Dense(topology[0], activation='relu', input_shape=(input_shape,)))
    
    for units in topology[1:]:
        model.add(Dense(units, activation='relu'))
        model.add(Dropout(0.2))
    
    model.add(Dense(output_shape, activation='softmax'))
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_models(dataset_name: str, train_df, test_df, topologies):
    
    # Start MLflow experiment
    experiment_name = f"{dataset_name}_experiment"
    mlflow.set_experiment(experiment_name)

    X_train = train_df.drop(columns=['Comportamiento', 'Sexo'])
    y_train = pd.get_dummies(train_df['Comportamiento'])
    X_test = test_df.drop(columns=['Comportamiento', 'Sexo'])
    y_test = pd.get_dummies(test_df['Comportamiento'])

    # Check the label mappings are the same
    assert all(y_train.columns == y_test.columns)
    # Save the label mappings as json file
    labels_mapping = y_train.columns.to_list()
    with open(f"{dataset_name}_labels_mapping.json", "w") as f:
        json.dump(labels_mapping, f)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    for j, topology in enumerate(topologies):
        with mlflow.start_run(run_name=f"topology_{j+1}_lr002"):
            # Log topology parameters
            # mlflow.log_param("topology", topology)
            # mlflow.log_param("num_layers", len(topology))
            mlflow.tensorflow.autolog()

            optimizer = Adam(learning_rate=0.002)

            model = create_model(X_train.shape[1], y_train.shape[1], topology, optimizer)
            
            # # Custom callback to log metrics to MLflow
            # class MetricsLogger(tf.keras.callbacks.Callback):
            #     def on_epoch_end(self, epoch, logs=None):
            #         for metric_name, metric_value in logs.items():
            #             mlflow.log_metric(metric_name, metric_value, step=epoch)
          
            history = model.fit(X_train_scaled, y_train, 
                              epochs=10, 
                              batch_size=32, 
                              validation_split=0.2, 
                            #   callbacks=[
                            #       tf.keras.callbacks.EarlyStopping(patience=10),
                            #       MetricsLogger()
                            #   ],
                              verbose=1,
                              )
            
            test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test)
            
            # Log final metrics
            # mlflow.log_metric("test_accuracy", test_accuracy)
            # mlflow.log_metric("test_loss", test_loss)
            
            # Log training history
            # for metric_name, metric_values in history.history.items():
            #     mlflow.log_metric(f"final_{metric_name}", metric_values[-1])
            
            # Save model artifact
            # mlflow.keras.log_model(model, f"model_topology{j+1}")
            
            print(f"Dataset {dataset_name}, Topology {j+1} - Test accuracy: {test_accuracy:.4f}")
            
            # Save training history locally
            with open(f"history_{dataset_name}_topology{j+1}.json", "w") as f:
                json.dump(history.history, f)
            
            # model.save(f"model_{dataset_name}_topology{j+1}.h5")


train_models("estudio3_1seg", estudio3_1seg_train, estudio3_1seg_test, topologies)