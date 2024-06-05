import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.inspection import permutation_importance
from openpyxl import load_workbook

# Reading the input file
mat_data = loadmat('All the Data.mat')

array_data = mat_data['result_array']
column_names = ['PilotLength', 'PilotSpacing', 'SymbolRate', 'PhaseNoise', 'BER', 'OBER']

# Convert the numpy array to a pandas DataFrame with the specified column names
df = pd.DataFrame(array_data, columns=column_names)

# Now you can access the columns by their names
X = df[['PilotLength', 'PilotSpacing', 'SymbolRate', 'PhaseNoise']]
y = df['OBER']

# Normalize
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Select top 4 best features
selector = SelectKBest(f_regression, k=4)
X_new = selector.fit_transform(X, y)
selected_features = df.columns[selector.get_support(indices=True)]  # Storing the selected feature names

# Splitting dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.3, random_state=17)
feature_importances_list = []

# Regressor
knn = KNeighborsRegressor(n_neighbors=1, weights='distance')

# Feature Importance Calculation using Permutation Importance
for state in range(20):
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.3, random_state=state)
    knn.fit(X_train, y_train)
    result = permutation_importance(knn, X_test, y_test, n_repeats=10, random_state=state)
    feature_importances_list.append(result.importances_mean)

# Convert the list to a DataFrame and add the feature names
feature_importances_df = pd.DataFrame(feature_importances_list, columns=selected_features).transpose()

# Adding the feature names as the first column
feature_importances_df.insert(0, 'Feature', feature_importances_df.index)

# Reset index
feature_importances_df.reset_index(drop=True, inplace=True)

# Apply MinMax scaling to the feature importances
scaler = MinMaxScaler()
scaled_importances = scaler.fit_transform(feature_importances_df.iloc[:, 1:])

# Update the DataFrame with scaled values
scaled_importances_df = pd.DataFrame(scaled_importances, columns=feature_importances_df.columns[1:])
scaled_importances_df.insert(0, 'Feature', feature_importances_df['Feature'])

# Export the DataFrame to an Excel file
file_path = "C:/Users/ryanj/Code/Research_THz/excel/Book1.xlsx"
with pd.ExcelWriter(file_path, mode='a', engine='openpyxl') as writer:
    scaled_importances_df.to_excel(writer, sheet_name='FeatureImportanceskNN2', index=False)
