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

print(array_data)

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

smape_values = []
trial = 0
a_20values = []

#log base 10
#log base 10 produces large mape, can produce numbers above 1 which doens't make sense
def transform_ber(ber):
    return np.minimum(ber, 100 - ber)

def Accuracy_score(orig, pred):
    if orig.name == 'BER':  # Apply transformation only for BER
        orig = transform_ber(orig)
        pred = transform_ber(pred)
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


#natural log
#Natural log produces smaller mape, but potentially incorrect
#def Accuracy_score(orig, pred):
  #  orig = np.exp(orig)  # Convert original values back from log scale
  #  pred = np.exp(pred)  # Convert predicted values back from log scale
  #  MAPE = np.mean((np.abs(orig - pred)) / orig)
  # return MAPE
    
custom_Scoring = make_scorer(Accuracy_score,greater_is_better = True)

#%% Test
for neighbor in range(1,16):
    knn = KNeighborsRegressor(n_neighbors=neighbor, weights='distance')
    cv_scores = RepeatedKFold(n_splits=5, n_repeats = 3, random_state = 8)
    Accuracy_Values = cross_val_score(knn, X_train, y_train, cv = cv_scores, scoring = custom_Scoring)
    trial += 1
        
    smape_values.append(Accuracy_Values)
    print('Trial #:',trial)
    print('\nAccuracy values for k-fold Cross Validation:\n', Accuracy_Values)
    print('\nFinal Average Accuracy of the model:', round(Accuracy_Values.mean(), 2))
    
    #custom scoring a_20 calulation
    custom_Scoring3 = make_scorer(Accuracy_score3,greater_is_better=True)

    #Running cross validation
    CV = RepeatedKFold(n_splits = 5, n_repeats=3, random_state = 8)
    Accuracy_Values3 = cross_val_score(knn,X_train ,y_train,\
                                       cv=CV,scoring=custom_Scoring3)
    a_20values.append(Accuracy_Values3)
    print('\n"a_20 index" for 5-fold Cross Validation:\n', Accuracy_Values3)
    print('\nFinal Average Accuracy a_20 index of the model:', round(Accuracy_Values3.mean(),4))
    
#%% Exporting to Excel
#df_metrics = pd.DataFrame({
   # 'Mean Squared Error': mse_list,
  #  'Mean Absolute Error': mae_list,
  #  'R^2 Score': r2_list,
  #  'MAPE': mape_values,
#    'RMSE': rmse_list
#})
df_metrics = pd.DataFrame(smape_values, index=range(1, 16), columns=range(1, 16))   
df_metrics_a20 = pd.DataFrame(a_20values, index=range(1, 16), columns=range(1, 16))   
    
file_path = "C:/Users/ryanj/Code/Research_THz/excel/Book1.xlsx"

#Export the DataFrame to an Excel file on a specific sheet
with pd.ExcelWriter(file_path, mode='a', engine='openpyxl') as writer:
    df_metrics.to_excel(writer, sheet_name='KNN_SMAPE_17', index=False, startrow=0, startcol=0)
    df_metrics_a20.to_excel(writer, sheet_name='KNN_A20_17', index=False, startrow=0, startcol=0)

# %%
