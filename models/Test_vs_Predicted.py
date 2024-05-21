# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 13:06:02 2024

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
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
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

#Normalize
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

selector = SelectKBest(f_regression, k=4)
X = selector.fit_transform(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)

#%% Regressors

lnr = LinearRegression()
svr = SVR(kernel='rbf', C=600,epsilon=0.25)
knn = KNeighborsRegressor(n_neighbors=3, weights='distance')
dtr = DecisionTreeRegressor(max_depth=4,random_state=17,criterion='squared_error')
rfr = RandomForestRegressor(n_estimators=50, random_state=17, max_depth=8,max_features=5)
gbr = GradientBoostingRegressor(max_depth= 3, random_state=17, n_estimators= 1000, learning_rate= 0.25)
abr = AdaBoostRegressor(estimator=dtr,random_state=17,n_estimators=1000,learning_rate=0.5)
#bag = BaggingRegressor(estimator=dtr,n_estimators=500)

#%% Test and analysis

# Train the model
abr.fit(X_train, y_train)

# Make predictions
y_pred = abr.predict(X_test)

# y_test = 10 ** y_test 
# y_pred = 10 ** y_pred 

#RSME Calculations
def rsm_error(actual,predicted):
    mse = np.mean((actual - predicted) ** 2)
    rmse = np.sqrt(mse)
    return rmse

#Mape calculations
def Accuracy_score(orig,pred):
    orig = 10.0 ** orig
    pred = 10.0 **  pred
    MAPE = np.mean((np.abs(orig-pred))/orig)
    return(MAPE)

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
mse = mean_squared_error(y_train, y_pred)
mae = mean_absolute_error(y_train, y_pred)
r2=r2_score(y_train,y_pred)
mape=Accuracy_score(y_train, y_pred)
rmse=rsm_error(y_train,y_pred)
a_20 = Accuracy_score3(y_train,y_pred)

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
    'Expected/Test': y_train,
    'Predicted': y_pred
})

file_path = "C:/spydertest/csv/CumulRot.xlsx"

# # Export to Excel
results_df.to_excel("results.xlsx", index=False)
with pd.ExcelWriter(file_path, mode='a', engine='openpyxl') as writer:
    results_df.to_excel(writer, sheet_name='ABR4Feats', index=False, startrow=0, startcol=0)