# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 13:06:02 2024

@author: kevry
"""

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
#%% Initialize
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


#Normalize
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

selector = SelectKBest(f_regression, k=4)
X = selector.fit_transform(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)


#%% Regressors

lnr = LinearRegression()
svr = SVR(kernel='rbf', C=600,epsilon=0.25)
knn = KNeighborsRegressor(n_neighbors=2, weights='distance')
dtr = DecisionTreeRegressor(max_depth=4,random_state=17,criterion='squared_error')
rfr = RandomForestRegressor(n_estimators=50, random_state=17, max_depth=8,max_features=5)
gbr = GradientBoostingRegressor(max_depth= 3, random_state=17, n_estimators= 1000, learning_rate= 0.25)
abr = AdaBoostRegressor(estimator=dtr,random_state=17,n_estimators=1000,learning_rate=0.5)
#bag = BaggingRegressor(estimator=dtr,n_estimators=500)

#%% Test and analysis

# Train the model
lnr.fit(X_train, y_train)

# Make predictions
y_pred = lnr.predict(X_test)

# y_test = 10 ** y_test 
# y_pred = 10 ** y_pred 

#RSME Calculations
def rsm_error(actual, predicted):
    mse = np.mean((actual - predicted) ** 2)
    rmse = np.sqrt(mse)
    return rmse

#Smape calculations
def Accuracy_score(orig,pred):
    numerator = np.abs(pred - orig)
    denominator = (np.abs(orig) + np.abs(pred)) / 2
    smape = np.mean(numerator / denominator)
    return smape

#a20 calculations
def Accuracy_score3(orig,pred):
    orig = 10 ** np.array(orig)
    pred = 10 ** np.array(pred)
    
    count = 0
    for i in range(len(orig)):
        if(pred[i] <= 1.2*orig[i]) and (pred[i] >= 0.8*orig[i]):
            count = count +1
    a_20 = count/len(orig)
    return(a_20)

#Statistical calculations
#SHOULD BE TEST NOT TRAIN
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2=r2_score(y_test,y_pred)
mape=Accuracy_score(y_test, y_pred)
rmse=rsm_error(y_test,y_pred)
a_20 = Accuracy_score3(y_test,y_pred)

#Print statements
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R^2:",r2)
print("MAPE:",mape)
print("RSME:", rmse)
print("A_20:", a_20)

# print("Y_test\n",y_test)
# print("The number of values in y_test",len(y_test))
# print("Y_pred",y_pred)

#%% Export to excel Actual vs expected


 ## Expected vs avtual dataframe
results_df = pd.DataFrame({
    'Expected/Test': y_test,
    'Predicted': y_pred
})

file_path = "C:/Users/ryanj/Code/Research_THz/excel/Book1.xlsx"

# # Export to Excel
with pd.ExcelWriter(file_path, mode='a', engine='openpyxl') as writer:
    results_df.to_excel(writer, sheet_name='LNR_TvP', index=False, startrow=0, startcol=0)
# %%
