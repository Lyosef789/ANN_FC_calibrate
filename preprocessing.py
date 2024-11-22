import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_data(filepath):
    """
    Preprocess data dynamically based on the provided file path.
    """
    # Read the file
    data = pd.read_csv(
        filepath,
        delimiter=',',
        comment='#',
        parse_dates=[0],
        infer_datetime_format=True,
        na_values='-1.00000e+31'
    )

    # Rename columns based on your convention
    data = data.rename(columns={
        data.columns[0]: "Epoch",
        data.columns[1]: "PROTON_BULK_SPEED",  # Speed
        data.columns[2]: "P+_W_NONLIN",       # Temp
        data.columns[3]: "P+_DENSITY",        # Density
        data.columns[4]: "BX",
        data.columns[5]: "BY",
        data.columns[6]: "BZ"
    })

    # Drop rows with missing values
    data = data.dropna()

    # Apply filtering for known parameter ranges
    data = data[data["PROTON_BULK_SPEED"].between(200, 900)]  # Speed
    data = data[data["P+_W_NONLIN"].between(10, 150)]         # Temperature
    data = data[data["P+_DENSITY"].between(0.1, 50)]          # Density
    data = data[data["BX"].between(-100, 100)]               # Magnetic field X
    data = data[data["BY"].between(-100, 100)]               # Magnetic field Y
    data = data[data["BZ"].between(-100, 100)]               # Magnetic field Z

    return data

def prepare_inputs(target_data, warped_reference, start_idx, end_idx, parameter):
    """
    Prepare input data for the BNN model based on the parameter.
    """
    # Select relevant columns for inputs and targets
    inputs = target_data.iloc[:, 3:53].values  # Assuming columns 3:53 contain features
    outputs = warped_reference[start_idx:end_idx]
    return inputs[start_idx:end_idx, :], outputs.reshape(-1, 1)

def standardize_data(inputs_reference, outputs_target):
    """
    Standardize input and target data for training.
    """
    # Initialize scalers
    predictor_scaler = StandardScaler()
    target_scaler = StandardScaler()

    # Standardize inputs and outputs
    inputs_reference_scaled = predictor_scaler.fit_transform(inputs_reference)
    outputs_target_scaled = target_scaler.fit_transform(outputs_target)

    # Split into training and testing datasets
    reference_train, reference_test, target_train, target_test = train_test_split(
        inputs_reference_scaled, outputs_target_scaled, test_size=0.3, random_state=42
    )

    return reference_train, target_train, predictor_scaler, target_scaler
