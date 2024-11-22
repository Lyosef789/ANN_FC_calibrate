import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def preprocess_reference(filepath):
    """
    Preprocess reference (Wind) data with filtering, resampling, and interpolation.
    """
    # Load the original Wind data
    data = pd.read_csv(filepath, delimiter=',', parse_dates=[0], infer_datetime_format=True, na_values='-1.00000e+31')

    # Rename columns
    data = data.rename(columns={
        data.columns[0]: "Epoch",
        data.columns[1]: "Speed",
        data.columns[2]: "Temp",
        data.columns[3]: "Density",
        data.columns[4]: "bx",
        data.columns[5]: "by",
        data.columns[6]: "bz"
    })

    # Drop missing values
    data = data.dropna()

    # Apply filtering for known ranges
    data = data[data["Speed"].between(200, 900)]
    data = data[data["Temp"].between(10, 150)]
    data = data[data["Density"].between(0.1, 50)]
    data = data[data["bx"].between(-100, 100)]
    data = data[data["by"].between(-100, 100)]
    data = data[data["bz"].between(-100, 100)]

    # Save the cleaned data to a new CSV file
    cleaned_filepath = filepath.replace(".csv", "_final.csv")
    data.to_csv(cleaned_filepath, index=False)

    # Reload the cleaned data
    data = pd.read_csv(
        cleaned_filepath, 
        delimiter=',', 
        parse_dates=[0], 
        infer_datetime_format=True, 
        date_parser=lambda col: pd.to_datetime(col, utc=True),
        na_values='-1.00000e+31'
    )
    data = data.rename(columns={data.columns[0]: "Epoch_time"})

    # Resample and interpolate to fill gaps
    data = data.resample('160s', on='Epoch_time').median()
    data = data.interpolate(method="linear")

    return data


def preprocess_target(filepath):
    """
    Preprocess target (DSCOVR) data with filtering, resampling, and interpolation.
    """
    # Load the DSCOVR data
    data = pd.read_csv(filepath, delimiter=',', parse_dates=[0], infer_datetime_format=True, na_values='-1.00000e+31')

    # Remove rows where all columns (4:54) are zero
    zero_mask = (((data.iloc[:, 4:54]).values) == 0).all(axis=1)
    data = data[~zero_mask]

    # Drop missing values
    data = data.dropna()

    # Rename columns
    data = data.rename(columns={
        data.columns[0]: "t",
        data.columns[1]: "bx",
        data.columns[2]: "by",
        data.columns[3]: "bz"
    })

    # Resample and interpolate
    data = data.resample('160s', on='t').median()
    data = data.interpolate(method="linear")

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
