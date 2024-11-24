import tensorflow as tf
from model import improved_penalized_nll

 # Register the custom loss function
tf.keras.utils.get_custom_objects()['improved_penalized_nll'] = improved_penalized_nll

    
def run_experiment(model, reference_train, target_train, max_value, num_epochs=50, batch_size=20):
    """
    Train the Bayesian Neural Network model.
    """
    
   
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss='improved_penalized_nll', metrics=['mae','mse']
    )

    # Train the model
    print("Training the model...")
    model.fit(reference_train, target_train, epochs=num_epochs, batch_size=batch_size, verbose=1)
    print("Training complete.")

def evaluate_non_warped(reference_data, start_idx, end_idx, parameter):
    """
    Extract non-warped values for the first month.
    """
    return reference_data[parameter].values[start_idx:end_idx]

def evaluate_month(model, reference_scaler, target_scaler, reference_data, start_idx, end_idx, parameter):
    """
    Evaluate the model for a given month using warped indices.
    """
    from dtw_utils import DTW_function

    # Perform DTW for the given month
    query, template, alignment, indices = DTW_function(reference_data, target_data, parameter, start_idx, end_idx)
    warped_values = reference_data[parameter].values[indices]

    # Standardize the inputs
    standardized_inputs = reference_scaler.transform(reference_data.iloc[start_idx:end_idx, 3:53].values)

    # Predict using the trained model
    predictions_distribution = model(standardized_inputs)
    predictions_mean = predictions_distribution.mean().numpy().tolist()
    predictions_stdv = predictions_distribution.stddev().numpy()

    # Inverse scale predictions
    predictions_final = target_scaler.inverse_transform(predictions_mean)

    return predictions_final
