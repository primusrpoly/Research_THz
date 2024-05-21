import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression

mat_data = loadmat('All the Data.mat')

#print(mat_data)

array_data = mat_data['result_array']

#print(array_data)

column_names = ['PilotLength', 'PilotSpacing', 'SymbolRate', 'PhaseNoise', 'BER', 'OBER']

# Convert the numpy array to a pandas DataFrame with the specified column names
df = pd.DataFrame(array_data, columns=column_names)

# Now you can access the columns by their names
X = df[['PilotLength', 'PilotSpacing', 'SymbolRate', 'PhaseNoise']]
#print("X:", X)
y = df['BER']
#print("y:\n", y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict on the testing set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Print coefficients (optional)
print("Coefficients:", model.coef_)




