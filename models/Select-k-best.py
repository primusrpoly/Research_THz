# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 12:54:25 2024

@author: kevry
"""

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

#%% Initilize
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

# Normalize
scaler = MinMaxScaler()
X_normalized = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

k = 1

selector = SelectKBest(f_regression, k=k)
X_new = selector.fit_transform(X_normalized, y) 

selected_features_mask = selector.get_support()
selected_features = X_normalized.columns[selected_features_mask]

# print(f"Selected features for k={k}: {selected_features[::-1].tolist()}")

# Since X_new is a numpy array without column names, for the train-test split, you should use it directly.
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.3, random_state=17)

#log base 10
#log base 10 produces large mape, can produce numbers above 1 which doens't make sense
def Accuracy_score(orig,pred):
    numerator = np.abs(pred - orig)
    denominator = (np.abs(orig) + np.abs(pred)) / 2
    smape = np.mean(numerator / denominator) * 100
    return smape

def Accuracy_score3(orig,pred):
    orig = 10 ** np.array(orig)
    pred = 10 ** np.array(pred)
    
    count = 0
    for i in range(len(orig)):
        if(pred[i] <= 1.2*orig[i]) and (pred[i] >= 0.8*orig[i]):
            count = count +1
    a_20 = count/len(orig)
    return(a_20)


custom_Scoring = make_scorer(Accuracy_score,greater_is_better = True)
#%% Regressors

lnr = LinearRegression()
svr = SVR(kernel='rbf', C=600,epsilon=0.25)
knn = KNeighborsRegressor(n_neighbors=2, weights='distance')
dtr = DecisionTreeRegressor(max_depth=4,random_state=17, criterion='squared_error')
rfr = RandomForestRegressor(n_estimators=250, random_state=17, max_depth=8,max_features=5)
gbr = GradientBoostingRegressor(max_depth= 2, random_state=17, n_estimators= 1000, learning_rate= 0.15)
abr = AdaBoostRegressor(estimator=dtr,random_state=17,n_estimators=50,learning_rate=5)
#bag = BaggingRegressor(estimator=dtr,n_estimators=500)

#%% MAPE

cv_scores = RepeatedKFold(n_splits=5, n_repeats = 3, random_state = 8)
Accuracy_Values = cross_val_score(knn, X_train, y_train, cv = cv_scores, scoring = custom_Scoring)

print('\nAccuracy values for k-fold Cross Validation:\n', Accuracy_Values)
print('\nFinal Average Accuracy of the model:', round(Accuracy_Values.mean(), 2))

selected_features_mask = selector.get_support()
selected_features = X.columns[selected_features_mask]

print(f"Selected features for k={k}: {selected_features[::-1].tolist()}")

#%% A-20
#custom scoring a_20 calulation
custom_Scoring3 = make_scorer(Accuracy_score3,greater_is_better=True)

#Running cross validation
CV = RepeatedKFold(n_splits = 5, n_repeats=3, random_state = 8)
Accuracy_Values3 = cross_val_score(knn,X_train,y_train,\
                                   cv=CV,scoring=custom_Scoring3)

print('\n"a_20 index" for 5-fold Cross Validation:\n', Accuracy_Values3)
print('\nFinal Average Accuracy a_20 index of the model:', round(Accuracy_Values3.mean(),4))
#%% Export

file_path = "C:/Users/ryanj/Code/Research_THz/excel/Book1.xlsx"

# DataFrame for accuracy values
df_metrics = pd.DataFrame(Accuracy_Values, index=range(1, len(Accuracy_Values) + 1), columns=['Accuracy'])

# Reverse the selected features list if it's in descending order
df_selected_features = pd.DataFrame(selected_features.tolist()[::-1], columns=['Selected Features'])

#Export the DataFrame to an Excel file on a specific sheet
with pd.ExcelWriter(file_path, mode='a', engine='openpyxl') as writer:
    df_metrics.to_excel(writer, sheet_name='KNN_kBestAcc1', index=False, startrow=0, startcol=0)
    df_selected_features.to_excel(writer, sheet_name='KNN_kBestSF1', index=False, startrow=0, startcol=0)

