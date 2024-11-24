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

def evaluate_non_warped(reference_data, parameter):
    """
    Extract non-warped values for the first month.
    """
    return reference_data[parameter].values

