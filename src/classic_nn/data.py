import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Dictionary to translate "Comportamiento" values
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

def generate_dataset_single_row(csv_path, input_columns, output_columns):
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

def generate_dataset_per_second(csv_path, input_columns, output_columns, rows_per_second=10):
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
        # Assign corresponding output to outputs_per_second
        outputs_per_second.loc[int(i/rows_per_second), output_columns] = outputs.loc[i, output_columns]

    # Print single row of inputs_per_second and outputs_per_second
    print(inputs_per_second[0])
    print(outputs_per_second.iloc[0])
    
    # Create new dataframe from inputs_per_second
    df = pd.DataFrame(inputs_per_second)
    # Add column names to the dataframe from adding _0 to _9 to the input_columns
    for i in range (0,10):
        for j in range (0, len(input_columns)):
            df.rename(columns={i*len(input_columns)+j:input_columns[j]+"_"+str(i)}, inplace=True)
    
    # Rename the output column according to the comportamientos_translation dictionary
    outputs_per_second = outputs_per_second.replace(comportamientos_translation)
    
    # Join the two dataframes
    df3 = pd.concat([df, outputs_per_second], axis=1)

    # Save the dataframe as a csv file with the original plus "_dataset_per_second"
    df3.to_csv(csv_path[:-4]+"_dataset_per_second.csv", index=False)

    # Return the location of the new csv file
    return csv_path[:-4]+"_dataset_per_second.csv"

# Function to split the data per value in the specified column
def split_csv_per_column(csv_path, column, max_unique_values=10):
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

if __name__ == "__main__":
    # test with data from estudio3.csv
    # csv_path = "../../data/base/resumen-comportamientos.csv"
    # split_csv_per_column(csv_path, "Nombre", 15)
    csv_path = "../../data/individual/resumen-comportamientos_Matilda.csv"
    # input_columns = ["X", "Y", "Z","ODBA"]
    input_columns = ["x", "y", "z","ODBA"]
    output_columns = ["Comportamiento"]
    print(f"Generate dataset single row - Dataset: {generate_dataset_single_row(csv_path, input_columns, output_columns)}")
    # print(f"Generate dataset per second - Dataset: {generate_dataset_per_second(csv_path, input_columns, output_columns)}")