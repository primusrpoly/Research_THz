import numpy as np
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

# Load the .mat file
mat_data = loadmat('All the Data.mat')
array_data = mat_data['result_array']
column_names = ['PilotLength', 'PilotSpacing', 'SymbolRate', 'PhaseNoise', 'BER', 'OBER']

# Convert the numpy array to a pandas DataFrame with the specified column names
df = pd.DataFrame(array_data, columns=column_names)

# One input variable and four target variables
X = df[['PhaseNoise']]
y = df[['PilotLength', 'PilotSpacing', 'SymbolRate', 'BER', 'OBER']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)

# Initialize KNeighborsRegressor
knn = KNeighborsRegressor(n_neighbors=1, weights='distance')

# Fit the model
knn_model = knn.fit(X_train, y_train)

# Predictions
knn_preds = knn_model.predict(X_test)

# Create DataFrame for predictions
df_metrics = pd.DataFrame(knn_preds, columns=['PilotLength', 'PilotSpacing', 'SymbolRate', 'BER', 'OBER'])

# Add the input PhaseNoise column to the DataFrame
df_metrics.insert(0, 'Input PhaseNoise', X_test.values)

# Sort the DataFrame by 'Input PhaseNoise' in ascending order
df_metrics = df_metrics.sort_values(by='Input PhaseNoise')

file_path = "C:/Users/ryanj/Code/Research_THz/excel/Feature Prediction.xlsx"

# Export the DataFrame to an Excel file on a specific sheet
with pd.ExcelWriter(file_path, mode='a', engine='openpyxl') as writer:
    df_metrics.to_excel(writer, sheet_name='KNN_Pred_PN_Sorted', index=False, startrow=0, startcol=0)
