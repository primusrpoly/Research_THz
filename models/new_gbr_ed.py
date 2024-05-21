# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 15:08:29 2024

@author: kevry
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import MinMaxScaler

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=8)

#5,10,15,20,25,30,35,40,45,50,75,100,250,500,750,1000
#5,15,25,30,40,50,75,100,250,500,750,1000,1250,1500,1750,2000
#5,25,35,50,75,100,250,500,750,1000,1250,1750,2000,2500,3000,5000
# 5,25,35,50,75,100,250,500,750,1000,1250,1750,2000,2500,3000,5000,6000,7500,8500,10000
#0.01,0.05,0.06,0.07,0.08,0.09,0.1,0.11,.12,.13,.14,.15,.25,.5,1,2

trees = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
l_rate = [0.1]

def Accuracy_score(orig,pred):
    orig = 10.0 ** orig
    pred = 10.0 ** pred
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


#def Accuracy_score(orig, pred):
  #  orig = np.exp(orig)  # Convert original values back from log scale
  #  pred = np.exp(pred)  # Convert predicted values back from log scale
  #  MAPE = np.mean((np.abs(orig - pred)) / orig)
   # return MAPE

custom_Scoring = make_scorer(Accuracy_score,greater_is_better = True)

#%% Training
cv_folds = 15
num_depths = len(trees)
num_epi = len(l_rate)

rows = num_depths * num_epi

all_cv_scores = np.zeros((rows, cv_folds))
all_cv_scores2 = np.zeros((rows, cv_folds))
current_row = 0

# Loop over depths and trees to fill the array
for depth_idx, rate in enumerate(l_rate):
    for tree_idx, n_trees in enumerate(trees):
        gbr = GradientBoostingRegressor(max_depth=n_trees,random_state=17,n_estimators=50,learning_rate=rate)
        
        cv_scores = RepeatedKFold(n_splits=5, n_repeats=3, random_state=8)
        Accuracy_Values = cross_val_score(gbr, X_train, y_train, cv=cv_scores, scoring=custom_Scoring)
        
        print('Trial #:',current_row)
        print('\nAccuracy values for k-fold Cross Validation:\n', Accuracy_Values)
        print('\nFinal Average Accuracy of the model:', round(Accuracy_Values.mean(), 2))
        
        all_cv_scores[current_row, :] = Accuracy_Values
        
        
        #custom scoring a_20 calulation
        custom_Scoring3 = make_scorer(Accuracy_score3,greater_is_better=True)

        #Running cross validation
        CV = RepeatedKFold(n_splits = 5, n_repeats=3, random_state = 8)
        Accuracy_Values3 = cross_val_score(gbr,X_train ,y_train,\
                                           cv=CV,scoring=custom_Scoring3)
        
        print('\n"a_20 index" for 5-fold Cross Validation:\n', Accuracy_Values3)
        print('\nFinal Average Accuracy a_20 index of the model:', round(Accuracy_Values3.mean(),4))
        
        all_cv_scores2[current_row, :] = Accuracy_Values3
        current_row += 1
#%% Export

#MAPE
# df_metrics = pd.DataFrame(all_cv_scores, columns=[f'Fold {i+1}' for i in range(cv_folds)])
# A20
df_metrics = pd.DataFrame(all_cv_scores2, columns=[f'Fold {i+1}' for i in range(cv_folds)])
# 

file_path = "C:/spydertest/csv/CumulRot.xlsx"

with pd.ExcelWriter(file_path, mode='a', engine='openpyxl') as writer:
    df_metrics.to_excel(writer, sheet_name='GBRa20', index_label='Depth')