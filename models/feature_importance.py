import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression

#%% Initilize
# Reading the input file
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
y = df['OBER']
#print("y:\n", y)

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


#%% Regressors
"""
kNN - k_neighbors = 1, weights = 'distance'				
ABR - estimators = 500, lr = 0.01, max_depth = 8 				
RFR - estimators = 5, max_depth = 9				
GBR - estimators = 75, lr = 0.1, max_depth = 8				
"""

lnr = LinearRegression()
svr = SVR(kernel='rbf', C=4,epsilon=0.09)
knn = KNeighborsRegressor(n_neighbors=1, weights='distance')
dtr = DecisionTreeRegressor(max_depth=8,random_state=17)
rfr = RandomForestRegressor(n_estimators=5, random_state=17, max_depth=9,max_features=4)
gbr = GradientBoostingRegressor(max_depth=8, random_state=17, n_estimators= 75, learning_rate= 0.1)
abr = AdaBoostRegressor(estimator=dtr,random_state=17,n_estimators=500,learning_rate=0.01)

#%%  Feature Importance

# Feature Importance Calculation
for state in range(20):
    X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.3, random_state=state)
    rfr.fit(X_train, y_train)
    try:
        feature_importances_list.append(rfr.feature_importances_)
    except AttributeError:
        # Handle the case where the feature_importances_ attribute is missing
        pass

# Convert the list to a DataFrame and add the feature names
feature_importances_df = pd.DataFrame(feature_importances_list, columns=selected_features).transpose()

# Adding the feature names as the first column
feature_importances_df.insert(0, 'Feature', feature_importances_df.index)

# Reset index
feature_importances_df.reset_index(drop=True, inplace=True)

# Export the DataFrame to an Excel file
file_path = "C:/Users/ryanj/Code/Research_THz/excel/Book1.xlsx"
with pd.ExcelWriter(file_path, mode='a', engine='openpyxl') as writer:
    feature_importances_df.to_excel(writer, sheet_name='FeatureImportancesRFR', index=False)
    
#%% Correlation Coefficent


# correlation_matrix = Book1.corr()


# print(correlation_matrix)


# # To export this correlation matrix to an Excel file
# file_path = "C:/Users/ryanj/Code/Research_THz/excel/Book1.xlsx"

# with pd.ExcelWriter(file_path, mode='a', engine='openpyxl') as writer:
#     correlation_matrix.to_excel(writer, sheet_name='Correlation', index=False, startrow=0, startcol=0)

