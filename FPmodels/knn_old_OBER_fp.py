import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from scipy.optimize import differential_evolution
from scipy.io import loadmat

# Load data
mat_data = loadmat('All the Data.mat')
array_data = mat_data['result_array']

# Define column names
column_names = ['PilotLength', 'PilotSpacing', 'SymbolRate', 'PhaseNoise', 'BER', 'OBER']

# Create DataFrame
df = pd.DataFrame(array_data, columns=column_names)

# Split data into features and target
X = df[['PhaseNoise', 'PilotLength', 'PilotSpacing', 'SymbolRate']]
y = df['OBER']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=8)

# Train the KNN model
knn = KNeighborsRegressor(n_neighbors=1, weights='distance')
knn.fit(X_train, y_train)

# Evaluate the model
y_pred = knn.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Function to predict BER using the trained model
def predict_ber(params):
    phase_noise, pilot_length, pilot_spacing, symbol_rate = params
    data = pd.DataFrame([[phase_noise, pilot_length, pilot_spacing, symbol_rate]],
                        columns=['PhaseNoise', 'PilotLength', 'PilotSpacing', 'SymbolRate'])
    prediction = knn.predict(data)[0]
    print(f"Predicted BER for params {params}: {prediction}")
    return prediction

# Objective function to minimize (BER)
def objective(params):
    return predict_ber(params)

# Function to find optimal parameters given a phase noise value
def find_optimal_parameters(phase_noise_value):
    # Define bounds, keeping phase_noise fixed
    bounds = [(phase_noise_value, phase_noise_value), (8, 64), (32, 1024), (1e6, 3e10)]
    
    # Define the function for differential evolution
    def de_objective(params):
        params = [phase_noise_value] + list(params)
        return objective(params)
    
    # Optimization using differential evolution
    result = differential_evolution(de_objective, bounds[1:], strategy='best1bin', maxiter=1000, tol=1e-7)
    
    optimal_pilot_length, optimal_pilot_spacing, optimal_symbol_rate = result.x
    
    # Print the BER after optimization
    final_ber = objective([phase_noise_value, optimal_pilot_length, optimal_pilot_spacing, optimal_symbol_rate])
    print(f"Final BER: {final_ber}")

    return optimal_pilot_length, optimal_pilot_spacing, optimal_symbol_rate

# Example usage
phase_noise_value = 0  # Example phase noise value
optimal_pilot_length, optimal_pilot_spacing, optimal_symbol_rate = find_optimal_parameters(phase_noise_value)

print(f"Optimal Pilot Length: {optimal_pilot_length}")
print(f"Optimal Pilot Spacing: {optimal_pilot_spacing}")
print(f"Optimal Symbol Rate: {optimal_symbol_rate}")