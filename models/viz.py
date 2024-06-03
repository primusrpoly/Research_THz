import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from scipy.io import loadmat
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression

#%% Initilize
mat_data = loadmat('All the Data.mat')

#print(mat_data)

array_data = mat_data['result_array']

print(array_data)

column_names = ['PilotLength', 'PilotSpacing', 'SymbolRate', 'PhaseNoise', 'BER', 'OBER']

# Convert the numpy array to a pandas DataFrame with the specified column names
df = pd.DataFrame(array_data, columns=column_names)

# Now you can access the columns by their names
X = df[['PilotLength', 'PilotSpacing', 'SymbolRate', 'PhaseNoise']]
#print("X:", X)
y = df['BER']
#print("y:\n", y)

#Normalize
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

selector = SelectKBest(f_regression, k=4)
X = selector.fit_transform(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)


# Generate synthetic data (replace with your actual dataset)
X_train = np.random.rand(100, 5)  # Features: Pilot Spacing, Pilot Length, Symbol Rate, Phase Noise, BER
y_train = np.random.rand(100)  # Target: BER

# Fit SVR with a linear kernel
svr_linear = SVR(kernel='linear')
svr_linear.fit(X_train, y_train)

# Predictions
X_test = np.random.rand(50, 5)  # New data points for prediction
y_pred = svr_linear.predict(X_test)

# Visualize feature weights
feature_weights = svr_linear.coef_[0]

equation = "BER = "
for i, weight in enumerate(feature_weights):
    feature_name = f"Feature {i+1}"
    equation += f"{weight:.4f} * {feature_name}"
    if i < len(feature_weights) - 1:
        equation += " + "

# Plot results
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], y, color='blue', label='Actual BER')
plt.scatter(X_test[:, 0], y_pred, color='red', label='Predicted BER')
plt.xlabel('Pilot Spacing')
plt.ylabel('BER')
plt.title('SVR with Linear Kernel')
plt.legend()
plt.show()

print("Feature Weights:")
for i, weight in enumerate(feature_weights):
    print(f"Feature {i+1}: {weight:.4f}")

print("\nEquation:")
print(equation)