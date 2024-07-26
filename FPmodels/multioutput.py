import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Load the data
mat_data = loadmat('All the Data.mat')
array_data = mat_data['result_array']

# Define column names
column_names = ['PilotLength', 'PilotSpacing', 'SymbolRate', 'PhaseNoise', 'BER', 'OBER']

# Convert to DataFrame
df = pd.DataFrame(array_data, columns=column_names)

# Define features and targets
X = df[['PhaseNoise']]
y = df[['PilotLength', 'PilotSpacing', 'SymbolRate']]

# Normalize features
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)

# Define and train models
models = {
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=17),
    'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=17),
    'AdaBoost': AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=8), n_estimators=100, learning_rate=0.01, random_state=17)
}

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'{name} MSE: {mse}')

# Predict for a given PhaseNoise value
phase_noise_to_predict = np.array([[25]])
phase_noise_to_predict_scaled = scaler.transform(phase_noise_to_predict)

# Using the best model (e.g., GradientBoosting)
best_model = models['GradientBoosting']
predicted_values = best_model.predict(phase_noise_to_predict_scaled)

predicted_symbol_rate = predicted_values[0][0]
predicted_pilot_length = predicted_values[0][1]
predicted_pilot_spacing = predicted_values[0][2]

print(f"\nPredicted values for PhaseNoise = {phase_noise_to_predict[0][0]}:")
print(f"Predicted SymbolRate: {predicted_symbol_rate}")
print(f"Predicted PilotLength: {predicted_pilot_length}")
print(f"Predicted PilotSpacing: {predicted_pilot_spacing}")
