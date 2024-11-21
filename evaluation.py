#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
    query, template, alignment, indices = DTW_function(reference_data, reference_data, parameter, start_idx, end_idx)
    warped_values = reference_data[parameter].values[indices]

    # Standardize the inputs
    standardized_inputs = reference_scaler.transform(reference_data.iloc[start_idx:end_idx, 3:53].values)

    # Predict using the trained model
    predictions_distribution = model(standardized_inputs)
    predictions_mean = predictions_distribution.mean().numpy()

    # Inverse scale predictions
    predictions_final = target_scaler.inverse_transform(predictions_mean).flatten()

    return predictions_final

