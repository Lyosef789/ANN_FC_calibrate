from preprocessing import preprocess_reference, preprocess_target, prepare_inputs, standardize_data
from dtw_utils import DTW_function, warp_function
from model import create_bnn_model
from evaluation import run_experiment, evaluate_non_warped, evaluate_month
from plotting import plot_results, plot_residuals, plot_confidence_intervals
from utils import prompt_user_for_inputs

# Max value dictionary for parameters
parameter_max_values = {
    "Speed": 850,
    "Density": 50,
    "Temperature": 120
}

def main():
    # Step 1: User Prompt for Inputs
    wind_file, dscovr_file, parameter, year, train_start, train_end, eval_start, eval_end, future_month_indices = prompt_user_for_inputs()

    # Step 2: Get max_value for the parameter
    max_value = parameter_max_values.get(parameter, 1)

    # Step 3: Data Preprocessing
    reference_data = preprocess_reference(wind_file)
    target_data = preprocess_target(dscovr_file)

    print(f"Processing year {year}, parameter '{parameter}'")

    # Step 4: Dynamic Time Warping
    query, template, alignment, indices = DTW_function(reference_data, target_data, parameter, train_start, train_end)
    warped_reference = warp_function(reference_data, indices, parameter)

    # Step 5: Data Preparation for BNN
    inputs_reference, outputs_target = prepare_inputs(target_data, warped_reference, train_start, train_end, parameter)

    # Step 6: Standardize Data
    reference_train, target_train, reference_scaler, target_scaler = standardize_data(inputs_reference, outputs_target)

    # Step 7: Model Training
    train_size = reference_train.shape[0]
    bnn_model = create_bnn_model(train_size)
    run_experiment(bnn_model, reference_train, target_train, max_value)

    # Step 8: Evaluation - Non-warped (First Month)
    non_warped_values = evaluate_non_warped(reference_data, eval_start, eval_end, parameter)
    actual_first_month = reference_data[parameter].values[eval_start:eval_end]
    plot_results(non_warped_values, actual_first_month, parameter, f"Year {year} - First Month")
    residuals_first_month = actual_first_month - non_warped_values
    plot_residuals(residuals_first_month, parameter, f"Year {year} - First Month")
    plot_confidence_intervals(non_warped_values, actual_first_month, parameter, f"Year {year} - First Month")

    # Step 9: Evaluation - Future Months
    for idx, (start, end) in enumerate(future_month_indices):
        predictions_month = evaluate_month(
            bnn_model, reference_scaler, target_scaler, reference_data, start, end, parameter
        )
        actual = reference_data[parameter].values[start:end]
        plot_results(predictions_month, actual, parameter, f"Year {year} - Month {idx + 2}")
        residuals = actual - predictions_month
        plot_residuals(residuals, parameter, f"Year {year} - Month {idx + 2}")
        plot_confidence_intervals(predictions_month, actual, parameter, f"Year {year} - Month {idx + 2}")

if __name__ == "__main__":
    main()
