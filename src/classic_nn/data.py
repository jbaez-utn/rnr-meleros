import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data_single_row(csv_path, input_columns, output_columns):
    """
    Split the data from a CSV file into train, test, and validation sets.

    Args:
        csv_path (str): Path to the CSV file.
        input_columns (list): List of column names to be extracted as inputs.
        output_columns (list): List of column names to be extracted as outputs.

    Returns:
        tuple: A tuple containing the train-test-validation split of the inputs and outputs selected.
    """
    # Load the data from the CSV file
    data = pd.read_csv(csv_path)

    # Extract the input and output columns from the data
    inputs = data[input_columns]
    outputs = data[output_columns]

    # Extract output column names
    output_names = outputs.columns
    # For each output name, assign a code to the values
    for output_name in output_names:
        # Get the unique values in the output column
        unique_values = outputs[output_name].unique()
        # Create a dictionary to map the unique values to the codes
        value_to_code = {value: code for code, value in enumerate(unique_values)}
        # Replace the values in the output column with the codes
        outputs[output_name] = outputs[output_name].map(value_to_code)

    #Print dictionary
    print(value_to_code)

    # Convert dataframes to numpy arrays
    inputs = inputs.to_numpy()
    outputs = outputs.to_numpy()

    # Print the shape of the data
    print(inputs.shape)
    print(outputs.shape)

    # Split the data into train, test, and validation sets
    X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    return X_train, X_test, X_val, y_train, y_test, y_val

# Function that will add extract batches of 10 rows from the data requested based of the counts 1 to ten per minute in the column "Hora" in the csv
def load_data_per_minute(csv_path, input_columns, output_columns):
    """
    Split the data from a CSV file into train, test, and validation sets.

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

    # Extract output column names
    output_names = outputs.columns
    # For each output name, assign a code to the values
    for output_name in output_names:
        # Get the unique values in the output column
        unique_values = outputs[output_name].unique()
        # Create a dictionary to map the unique values to the codes
        value_to_code = {value: code for code, value in enumerate(unique_values)}
        # Replace the values in the output column with the codes
        outputs[output_name] = outputs[output_name].map(value_to_code)

    #Print dictionary
    print(value_to_code)

    # Convert dataframes to numpy arrays
    inputs = inputs.to_numpy()
    outputs = outputs.to_numpy()

    # Print the shape of the data
    print(inputs.shape)
    print(outputs.shape)

    # Create empty numpy array 10 times smaller than the original data and 10 times more columns for inputs
    # Create the structured array, dtype=dt
    inputs_per_minute = np.empty((int(len(inputs)/10), len(inputs[0])*10))
    outputs_per_minute = np.empty((int(len(outputs)/10),len(outputs[0])))

    # Print the shape of the new data
    print(inputs_per_minute.shape)
    print(outputs_per_minute.shape)
    
    # Extract the inputs and outputs per minute
    for i in range (0, len(inputs),10):
        for j in range (0,10):
            for k in range (0, len(inputs[0])):
                # Copy item from inputs to inputs_per_minute
                # print(f"Assignig item {inputs[i+j][k]} from inputs[{i+j}][{k}]")
                inputs_per_minute[int(i/10)][j*len(inputs[0])+k] = inputs[i+j][k]
                # Print values copied
                # print(f"to inputs_per_minute[{int(i/10)}][{j*len(inputs[0])+k}] -> {inputs_per_minute[int(i/10)][j*len(inputs[0])+k]}")
        # Populate the output for the current row
        outputs_per_minute[int(i/10)] = outputs[i]

    # Print single row of inputs_per_minute and outputs_per_minute
    print(inputs_per_minute[0])
    print(outputs_per_minute[0])

    # Create new dataframe from inputs_per_minute and outputs_per_minute
    df = pd.DataFrame(inputs_per_minute)
    # Add column names to the dataframe from adding _0 to _9 to the input_columns
    for i in range (0,10):
        for j in range (0, len(input_columns)):
            df.rename(columns={i*len(input_columns)+j:input_columns[j]+"_"+str(i)}, inplace=True)
    
    df2 = pd.DataFrame(outputs_per_minute)
    # Add only the original column names to the dataframe
    for i in range (0, len(output_columns)):
        df2.rename(columns={i:output_columns[i]}, inplace=True)
    
    # Join the two dataframes
    df3 = pd.concat([df, df2], axis=1)
    # Save the dataframe as a csv file
    df3.to_csv("data_per_minute.csv", index=False)


    # Split the data into train, test, and validation sets
    X_train, X_test, y_train, y_test = train_test_split(inputs_per_minute, outputs_per_minute, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    return X_train, X_test, X_val, y_train, y_test, y_val

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
        # Save the dataframe as a csv file inside a folder named "split"
        df.to_csv(f"/home/jbaez/Documents/utn/gintea/rnr-meleros/data/split/{value}.csv", index=False)


if __name__ == "__main__":
    # test with data from estudio3.csv
    csv_path = "/home/jbaez/Documents/utn/gintea/rnr-meleros/data/Matilda.csv"
    input_columns = ["X", "Y", "Z","ODBA"]
    output_columns = ["Comportamiento"]
    # split_csv_per_column(csv_path, "Nombre", 10)
    X_train, X_test, X_val, y_train, y_test, y_val = load_data_per_minute(csv_path, input_columns, output_columns)
    # X_train, X_test, X_val, y_train, y_test, y_val = load_data_single_row(csv_path, input_columns, output_columns)
    # print(X_train.shape)
    # print(X_test.shape)
    # print(X_val.shape)
    # print(y_train.shape)
    # print(y_test.shape)
    # print(y_val.shape)
