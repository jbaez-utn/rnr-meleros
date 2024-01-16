import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)

    # Preprocess the data
    # Here we simply scale the data to range between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Split the data into training and testing sets
    train_data, test_data = train_test_split(scaled_data, test_size=0.2)

    return train_data, test_data