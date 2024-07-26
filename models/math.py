import pandas as pd

import numpy as np

def mape_exponential(expected, predicted):
    """
    Calculate the Mean Absolute Percentage Error (MAPE) using exponential values.
    
    Parameters:
    expected (float): The expected value.
    predicted (float): The predicted value.
    
    Returns:
    float: The MAPE using exponential values.
    """
    exp_expected = np.exp(expected*.01)
    exp_predicted = np.exp(predicted*.01)
    mape = np.abs(exp_expected - exp_predicted) / np.abs(exp_expected)
    return mape


def Accuracy_score(orig, pred):
    exp_orig = np.exp(orig * 0.01)
    exp_pred = np.exp(pred * 0.01)
    
    mape = np.abs(exp_orig - exp_pred) / np.abs(exp_orig)
    return np.mean(mape) 

# Example usage
expected_value = 49.3917465
predicted_value = 49.2525101
result = mape_exponential(expected_value, predicted_value)
print(f"MAPE using exponential values: {result}")

orig = 49.3917465
pred = 49.2525101
result2 = Accuracy_score(orig, pred)
print(f"MAPE using exponential values: {result2}")



