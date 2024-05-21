# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 19:09:42 2024

@author: kevry
"""



import numpy as np
import pandas as pd
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
# Reading the input file
file_path = "C:/spydertest/csv/Cum_Rot_Data_7_Features_Log.csv"
EDData = pd.read_csv(file_path)
#print(EDData.head())

X = EDData[['A/Ac', 'h/B', 'Cr', 'Sand/Clay','amax', 'Ia','CAV']]
# print(X)
y = EDData['CR']
# print(y)

# Normalize
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Select top 4 best features
selector = SelectKBest(f_regression, k=4)
X_new = selector.fit_transform(X, y)
selected_features = EDData.columns[selector.get_support(indices=True)]  # Storing the selected feature names

# Splitting dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.3, random_state=17)
feature_importances_list = []


#%% Regressors

lnr = LinearRegression()
svr = SVR(kernel='rbf', C=4,epsilon=0.09)
knn = KNeighborsRegressor(n_neighbors=3, weights='distance')
dtr = DecisionTreeRegressor(max_depth=4,random_state=17)
rfr = RandomForestRegressor(n_estimators=50, random_state=17, max_depth=8,max_features=5)
gbr = GradientBoostingRegressor(max_depth= 3, random_state=17, n_estimators= 1000, learning_rate= 0.25)
abr = AdaBoostRegressor(estimator=dtr,random_state=17,n_estimators=1000,learning_rate=0.5)

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
file_path = "C:/spydertest/csv/CumulRot.xlsx"
with pd.ExcelWriter(file_path, mode='a', engine='openpyxl') as writer:
    feature_importances_df.to_excel(writer, sheet_name='FeatureImportancesRFR2', index=False)
    
#%% Correlation Coefficent


# correlation_matrix = CumRot.corr()


# print(correlation_matrix)


# # To export this correlation matrix to an Excel file
# file_path = "C:/spydertest/csv/ED_Data.xlsx"

# with pd.ExcelWriter(file_path, mode='a', engine='openpyxl') as writer:
#     correlation_matrix.to_excel(writer, sheet_name='Correlation', index=False, startrow=0, startcol=0)

