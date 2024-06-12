import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Diccionario de valores de "Comportamiento" en el dataset
comportamientos_translation = {
    "D": "descanso",
    "IM": "inmovil",
    "A": "alerta",
    "AA": "auto-acicalamiento",
    "AL": "alimentacion",
    "E": "exploracion",
    "L": "locomocion",
    "LR": "locomocion-repetitiva",
    "LA": "locomocion-ascenso",
    "LD": "locomocion-descenso",
    "LI": "locomocion-invertida",
    "O": "otros"
}

def generate_dataset_single_row(csv_path, input_columns, output_columns) -> str:
    """
    Loads and transforms the csv file to a format suitable to train a neural network. 

    Args:
        csv_path (str): Path to the CSV file.
        input_columns (list): List of column names to be extracted as inputs. Objects in the column must be float numbers.
        output_columns (list): List of column names to be extracted as outputs for the dataset.

    Returns:
        dataset(str): Path to the converted CSV dataset.
    """
    # Load the data from the CSV file
    data = pd.read_csv(csv_path)

    # Extract the input and output columns from the data
    inputs = data[input_columns]
    outputs = data[output_columns]

    # Convert inputs data to numpy array. Exit if the input colums are not floating point numbers.
    try:
        inputs = inputs.to_numpy().astype(float)
    except ValueError:
        print("Error: Input columns must be floating point numbers.")
        exit(1)

    # Convert inputs back to dataframe
    inputs = pd.DataFrame(inputs)
    # Change input column names back to x, y, z, ODBA
    inputs.rename(columns={0:"x", 1:"y", 2:"z", 3:"ODBA"}, inplace=True)

    # Convert outputs data according to the comportamientos_translation dictionary
    outputs = outputs.replace(comportamientos_translation)

    #Join the input and output dataframes
    df = pd.concat([inputs, outputs], axis=1)

    # Save the dataframe as a csv file with the original plus "_dataset_single_row" without column names
    df.to_csv(csv_path[:-4]+"_dataset_single_row.csv", index=False)

    return csv_path[:-4]+"_dataset_single_row.csv"

def generate_dataset_per_second(csv_path, input_columns, output_columns, rows_per_second=10, output_path="./") -> str:
    """
    Loads and transforms the csv file to a format suitable to train a neural network converting a fixed number of rows as different secuential inputs.
    It assumes the data is ordered sequentially by second and that the rows per second are equal for every second
    Args:
        csv_path (str): Path to the CSV file.
        input_columns (list): List of column names to be extracted as inputs.
        output_columns (list): List of column names to be extracted as outputs.

    Returns:
        tuple: A tuple containing the train-test-validation split of the inputs and outputs selected.
    """
    # Load the data from the CSV file with pandas
    data = pd.read_csv(csv_path)
    # Extract file name from csv_path
    file_name = csv_path.split("/")[-1]
    # Extract the input and output columns from the data
    inputs = data[input_columns]
    outputs = data[output_columns]

    # Print first 3 rows of each dataframe
    print(inputs.head(3))
    print(outputs.head(3))

    # Convert input dataframe to numpy array as float
    inputs = inputs.to_numpy().astype(float)

    # Print the shape of the data
    print(inputs.shape)
    print(outputs.shape)

    # Create empty numpy array 10 times smaller than the original data and 10 times more columns for inputs
    inputs_per_second = np.empty((int(len(inputs)/rows_per_second), len(inputs[0])*rows_per_second))
    # Create output per second dataframe with the original row size divided by rows_per_second
    outputs_per_second = outputs.iloc[::rows_per_second, :]

    # Print the shape of the new data
    print(inputs_per_second.shape)
    print(outputs_per_second.shape)
    
    # Extract the inputs and outputs per second
    for i in range (0, len(inputs),rows_per_second):
        # Loop over the following rows_per_second rows
        for j in range (0,rows_per_second):
            for k in range (0, len(inputs[0])):
                # Copy item from inputs to inputs_per_second
                # print(f"Assignig item {inputs[i+j][k]} from inputs[{i+j}][{k}]")
                inputs_per_second[int(i/rows_per_second)][j*len(inputs[0])+k] = inputs[i+j][k]
                # Print values copied
                # print(f"to inputs_per_second[{int(i/10)}][{j*len(inputs[0])+k}] -> {inputs_per_second[int(i/10)][j*len(inputs[0])+k]}")
                # Copy item from original outputs[i] to outputs_per_second[int(i/rows_per_second)]
        # Fill outputs_per_second with the original outputs
        # print(f"Assignig item {outputs.iloc[i]} from outputs.iloc[{i}]")
        outputs_per_second.iloc[int(i/rows_per_second)] = outputs.iloc[i]

    # Print single row of inputs_per_second and outputs_per_second
    # print(inputs_per_second[0])
    # print(outputs_per_second.iloc[0])
    
    # Shape after conversion
    # print(f"inputs_per_second shape: {inputs_per_second.shape} type: {type(inputs_per_second)}")
    # print(f"outputs_per_second shape: {outputs_per_second.shape} type: {type(outputs_per_second)}")

    # Create new dataframe from inputs_per_second
    df = pd.DataFrame(inputs_per_second)
    # Add column names to the dataframe from adding _0 to _9 to the input_columns
    for i in range (0,10):
        for j in range (0, len(input_columns)):
            df.rename(columns={i*len(input_columns)+j:input_columns[j]+"_"+str(i)}, inplace=True)

    # print(f"df shape: {df.shape} type: {type(df)}")
    # print(f"outputs_per_second shape: {outputs_per_second.shape} type: {type(outputs_per_second)}")

    # Reset the index of outputs_per_second to be sure the resulting dataframe has the same index
    outputs_per_second.reset_index(drop=True, inplace=True)
    
    # Join the two dataframes
    df3 = pd.concat([df, outputs_per_second], axis=1)

    # print(f"df3 shape: {df3.shape} type: {type(df3)}")


    # Save the dataframe as a csv file with the original plus "_1seg"
    df3.to_csv(output_path + file_name +"_1seg.csv", index=False)

    # Return the location of the new csv file
    return output_path + file_name +"_1seg.csv"

# Function to split the data per value in the specified column
def split_csv_per_column(csv_path, column, max_unique_values=10) -> None:
    """
    Split the data from a CSV file into train, test, and validation sets.

    Args:
        csv_path (str): Path to the CSV file.
        column (str): Column name to be used to split the data.

    Returns:
        tuple: A tuple containing the train-test-validation split of the inputs and outputs selected.
    """
    # Load the data from the CSV file with pandas
    data = pd.read_csv(csv_path)

    # Create a dictionary for the values in the column
    unique_values = data[column].unique()
    # Exit with error for unique values greater than max_unique_values
    if len(unique_values) > max_unique_values:
        print(f"Error: Too many unique values in column {column}.")
        exit(1)
    
    # Save a csv file for each dataframe with the name of the unique value
    for value in unique_values:
        # Create a dataframe for each unique value
        df = data.loc[data[column] == value]
        # Save the dataframe as a csv file with the original name plus the unique value
        df.to_csv(csv_path[:-4]+"_"+str(value)+".csv", index=False)

# Function to load the data from the CSV file, remove nan values, and return the dataframe
def dataframe_from_csv(csv_path):
    """
    Load the data from a CSV file and return the dataframe.

    Args:
        csv_path (str): Path to the CSV file.

    Returns:
        DataFrame: A pandas DataFrame containing the data from the CSV file.
    """
    # Load the data from the CSV file with pandas
    data = pd.read_csv(csv_path)
    # Remove the rows with NaN values
    data = data.dropna()
    # Return the dataframe
    return data

# Function to remove outliers
def remove_outliers_single_column(data, column, threshold=3)->pd.DataFrame:
    """
    Remove the outliers from a column in a dataframe.

    Args:
        data (DataFrame): A pandas DataFrame containing the data.
        column (str): The column name to remove the outliers.
        threshold (int): The threshold to consider a value an outlier.

    Returns:
        DataFrame: A pandas DataFrame with the outliers removed.
    """
    # Calculate the z-score for the column
    z = np.abs((data[column] - data[column].mean()) / data[column].std())
    # Remove the rows with z-score greater than the threshold
    data = data[z < threshold]
    # Return the dataframe
    return data

# Function to remove outliers from multiple columns
def remove_outliers_multiple_columns(data, columns, threshold=3)->pd.DataFrame:
    """
    Remove the outliers from multiple columns in a dataframe.

    Args:
        data (DataFrame): A pandas DataFrame containing the data.
        columns (list): A list of column names to remove the outliers.
        threshold (int): The threshold to consider a value an outlier.

    Returns:
        DataFrame: A pandas DataFrame with the outliers removed.
    """
    # Loop over the columns
    for column in columns:
        # Remove the outliers from the column
        data = remove_outliers_single_column(data, column, threshold)
    # Return the dataframe
    return data

# Function to scale the data in a single colum with the MinMaxScaler
def scale_single_column(data, column)->pd.DataFrame:
    """
    Scale the data in a single column with the MinMaxScaler.

    Args:
        data (DataFrame): A pandas DataFrame containing the data.
        column (str): The column name to scale.

    Returns:
        DataFrame: A pandas DataFrame with the column scaled.
    """
    # Create the MinMaxScaler object
    scaler = MinMaxScaler()
    # Fit the scaler to the data
    scaler.fit(data[[column]])
    # Scale the data
    data[column] = scaler.transform(data[[column]])
    # Return the dataframe
    return data

# Function to scale the data in multiple columns with the MinMaxScaler
def scale_multiple_columns(data, columns)->pd.DataFrame:
    """
    Scale the data in multiple columns with the MinMaxScaler.

    Args:
        data (DataFrame): A pandas DataFrame containing the data.
        columns (list): A list of column names to scale.

    Returns:
        DataFrame: A pandas DataFrame with the columns scaled.
    """
    # Loop over the columns
    for column in columns:
        # Scale the data in the column
        data = scale_single_column(data, column)
    # Return the dataframe
    return data

# Function to encode data from single column using label encoder
def encode_single_column(data, column)->pd.DataFrame:
    """
    Encode the data from a single column using label encoder.

    Args:
        data (DataFrame): A pandas DataFrame containing the data.
        column (str): The column name to encode.

    Returns:
        DataFrame: A pandas DataFrame with the column encoded.
    """
    # Create the label encoder object
    le = LabelEncoder()
    # Fit the label encoder to the data
    data[column] = le.fit_transform(data[column])
    # Return the dataframe
    return data

# Function to encode data from columns with categorical values
def encode_categorical_columns(data, columns)->pd.DataFrame:
    """
    Encode the data from columns with categorical values using label encoder.

    Args:
        data (DataFrame): A pandas DataFrame containing the data.
        columns (list): A list of column names to encode.

    Returns:
        DataFrame: A pandas DataFrame with the columns encoded.
    """
    # Loop over the columns
    for column in columns:
        # Encode the data in the column
        data = encode_single_column(data, column)
    # Return the dataframe
    return data

# Function to split the data into train, test, and validation sets
def split_data_train_test_validation(data, input_columns, output_columns, test_size=0.2, val_size=0.1):
    """
    Split the data into train, test, and validation sets.

    Args:
        data (DataFrame): A pandas DataFrame containing the data.
        input_columns (list): A list of column names to be used as inputs.
        output_columns (list): A list of column names to be used as outputs.
        test_size (float): The size of the test set.
        val_size (float): The size of the validation set.

    Returns:
        tuple: A tuple containing the train-test-validation split of the inputs and outputs selected.
    """
    # Split the data into inputs and outputs
    X = data[input_columns]
    y = data[output_columns]
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    # Split the data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size)
    # Return the train-test-validation split
    return X_train, X_test, X_val, y_train, y_test, y_val

#TODO: Dataset per last 5 seconds

if __name__ == "__main__":
    # test with data from estudio3.csv
    csv_path = "../../data/0-cruda/estudio3.csv"
    # input_columns = ["X", "Y", "Z","ODBA"]
    input_columns = ["X", "Y", "Z","ODBA"]
    output_columns = ["Comportamiento", "Sexo"]
    # print(f"Generate dataset single row - Dataset: {generate_dataset_single_row(csv_path, input_columns, output_columns)}")
    print(f"Generate dataset per second - Dataset: {1}")
    generate_dataset_per_second(csv_path, input_columns, output_columns, 10, "../../data/1-intermedia/")
    split_csv_per_column("../../data/1-intermedia/estudio3_1seg.csv", 'Sexo')
    input_columns = ["x", "y", "z","ODBA"]
    csv_path = "../../data/0-cruda/tabla-resumen.csv"
    print(f"Generate dataset per second - Dataset: {2}")
    generate_dataset_per_second(csv_path, input_columns, output_columns, 10, "../../data/1-intermedia/")
    split_csv_per_column("../../data/1-intermedia/tabla-resumen_1seg.csv", 'Sexo')
    