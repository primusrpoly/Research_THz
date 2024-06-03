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

#Normalize
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

selector = SelectKBest(f_regression, k=4)
X = selector.fit_transform(X, y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=8)
# [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12,0.13,0.14,0.15,0.25]
#0.01,0.05,0.1,0.25,0.5,0.75,1,2.5,5
#1,5,10,15,20,25,30,35,40,45,50,75,100,250,500,1000
# 95,96,97,98,99,100,101,102,103,104,105
# 0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5
# 0.25,0.5,0.75,1,5,10,15,20,25,30,35,40,45,50,75,100,250

epi_values = [0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]
C_value = [1000]

def Accuracy_score(orig,pred):
    numerator = np.abs(pred - orig)
    denominator = (np.abs(orig) + np.abs(pred)) / 2
    smape = np.mean(numerator / denominator)
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

#def Accuracy_score(orig, pred):
  #  orig = np.exp(orig)  # Convert original values back from log scale
  #  pred = np.exp(pred)  # Convert predicted values back from log scale
  #  MAPE = np.mean((np.abs(orig - pred)) / orig)
   # return MAPE

custom_Scoring = make_scorer(Accuracy_score,greater_is_better = True)


#%% Training
cv_folds = 15
num_depths = len(C_value)
num_epi = len(epi_values)

rows = num_depths * num_epi

all_cv_scores = np.zeros((rows, cv_folds))
all_cv_scores2 = np.zeros((rows, cv_folds))
current_row = 0

# Loop over depths and trees to fill the array
for depth_idx, depth in enumerate(C_value):
    for tree_idx, n_epi in enumerate(epi_values):
        svr = SVR(kernel='rbf', C=depth, epsilon = n_epi)
        
        cv_scores = RepeatedKFold(n_splits=5, n_repeats=3, random_state=8)
        Accuracy_Values = cross_val_score(svr, X_train, y_train, cv=cv_scores, scoring=custom_Scoring)
        
        print('Trial #:',current_row)
        print('\nAccuracy values for k-fold Cross Validation:\n', Accuracy_Values)
        print('\nFinal Average Accuracy of the model:', round(Accuracy_Values.mean(), 2))
        
        all_cv_scores[current_row, :] = Accuracy_Values
        
        
        #custom scoring a_20 calulation
        custom_Scoring3 = make_scorer(Accuracy_score3,greater_is_better=True)

        #Running cross validation
        CV = RepeatedKFold(n_splits = 5, n_repeats=3, random_state = 8)
        Accuracy_Values3 = cross_val_score(svr,X_train ,y_train,\
                                           cv=CV,scoring=custom_Scoring3)
        
        print('\n"a_20 index" for 5-fold Cross Validation:\n', Accuracy_Values3)
        print('\nFinal Average Accuracy a_20 index of the model:', round(Accuracy_Values3.mean(),4))
        
        all_cv_scores2[current_row, :] = Accuracy_Values3
        current_row += 1
        
#%% Export

#MAPE
df_metrics = pd.DataFrame(all_cv_scores, columns=[f'Fold {i+1}' for i in range(cv_folds)])
#A20
df_metrics_A20 = pd.DataFrame(all_cv_scores2, columns=[f'Fold {i+1}' for i in range(cv_folds)])


file_path = "C:/Users/ryanj/Code/Research_THz/excel/Book1.xlsx"

with pd.ExcelWriter(file_path, mode='a', engine='openpyxl') as writer:
    df_metrics.to_excel(writer, sheet_name='SVR_epi2_SMAPE', index_label='Depth')
    df_metrics_A20.to_excel(writer, sheet_name='SVR_epi2_A20', index_label='Depth')