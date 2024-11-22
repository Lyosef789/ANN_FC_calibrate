import matplotlib.pyplot as plt
import numpy as np

def plot_results(predictions, actual, parameter, dataset_label):
    """
    Plot predictions and actual values for the parameter.
    """
    plt.plot(predictions, label=f'{parameter} NN Prediction')
    plt.plot(actual, label=f'{parameter} Ground Truth')
    plt.title(f'{parameter} Predictions ({dataset_label})')
    plt.ylabel(f'{parameter} (units)')
    plt.legend()
    plt.show()

def plot_residuals(residuals, parameter, dataset_label):
    """
    Plot residuals for the parameter.
    """
    plt.hist(residuals, bins=50, density=True, alpha=0.7, label='Residuals')
    plt.title(f'Residuals for {parameter} ({dataset_label})')
    plt.legend()
    plt.show()

def plot_confidence_intervals(predictions, actual, parameter, dataset_label, std_dev_factor=1.96):
    """
    Plot predictions with confidence intervals (CI) for the parameter.
    """
    std_dev = np.std(predictions - actual)
    upper_bound = predictions + (std_dev_factor * std_dev)
    lower_bound = predictions - (std_dev_factor * std_dev)

    plt.plot(predictions, label=f'{parameter} NN Prediction')
    plt.fill_between(range(len(predictions)), upper_bound, lower_bound, alpha=0.2, label='95% CI')
    plt.plot(actual, label=f'{parameter} Ground Truth', linestyle='--')
    plt.title(f'{parameter} Predictions with 95% CI ({dataset_label})')
    plt.ylabel(f'{parameter} (units)')
    plt.legend()
    plt.show()

