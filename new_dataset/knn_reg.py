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
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression

#%% Initilize
#load
mat_data = loadmat('full_Results.mat')

#extract results struct
array_data = mat_data['Results']

#intialize lists
modulation_order = []
snr = []
pilot_length = []
pilot_spacing = []
symbol_rate = []
phase_noise = []
ber = []
cber = []

#extract data
for result in array_data[0]:
    modulation_order.append(result['Modulation_order'][0][0])
    snr.append(result['SNR'][0][0])
    pilot_length.append(result['pilot_length'][0][0])
    pilot_spacing.append(result['symbols_between_pilot'][0][0])
    symbol_rate.append(result['symbol_rate'][0][0])
    phase_noise.append(result['phase_noise'][0][0])
    ber.append(result['BER'][0][0])
    cber.append(result['CBER'][0][0])

# data frame
column_names = ['ModulationOrder', 'SNR', 'PilotLength', 'PilotSpacing', 'SymbolRate', 'PhaseNoise', 'BER', 'CBER']
df = pd.DataFrame({
    'ModulationOrder': modulation_order,
    'SNR': snr,
    'PilotLength': pilot_length,
    'PilotSpacing': pilot_spacing,
    'SymbolRate': symbol_rate,
    'PhaseNoise': phase_noise,
    'BER': ber,
    'CBER': cber
})

#x and y split
X = df[['PhaseNoise', 'PilotLength', 'PilotSpacing', 'SymbolRate', 'SNR']]
#print("X:", X)
y = df['CBER']
#print("y:\n", y)

#Normalize
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)

smape_values = []
trial = 0
a_20values = []
mae_values = []

#log base 10
#returning the minimum value of our BER ex. 53->47, 58->42...

def Accuracy_score(orig, pred):
    exp_orig = np.exp(orig)
    exp_pred = np.exp(pred)
    
    mape = np.abs(exp_orig - exp_pred) / np.abs(exp_orig)
    return np.mean(mape) 

def Accuracy_score3(orig,pred):
    orig = np.array(orig)
    pred = np.array(pred)

    count = 0
    for i in range(len(orig)):
        if(pred[i] <= 1.4*orig[i]) and (pred[i] >= 0.6*orig[i]):
            count += 1
    a_20 = count/len(orig)
    return a_20

#Exp MAPE    
custom_Scoring = make_scorer(Accuracy_score,greater_is_better=True)
#A20
custom_Scoring3 = make_scorer(Accuracy_score3,greater_is_better=True)
#MAE
custom_Scoring5 = make_scorer(mean_absolute_error, greater_is_better=True)


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
    

    #Running cross validation
    CV = RepeatedKFold(n_splits = 5, n_repeats=3, random_state = 8)
    Accuracy_Values3 = cross_val_score(knn,X_train ,y_train,\
                                       cv=CV,scoring=custom_Scoring3)
    a_20values.append(Accuracy_Values3)
    print('\n"a_20 index" for 5-fold Cross Validation:\n', Accuracy_Values3)
    print('\nFinal Average Accuracy a_20 index of the model:', round(Accuracy_Values3.mean(),4))


    CV = RepeatedKFold(n_splits = 5, n_repeats=3, random_state = 8)
    Accuracy_Values5 = cross_val_score(knn,X_train ,y_train,\
                                       cv=CV,scoring=custom_Scoring5)
    mae_values.append(Accuracy_Values5)
    print('\nMAE for 5-fold Cross Validation:\n', Accuracy_Values3)
    print('\nFinal Average Accuracy MAE of the model:', round(Accuracy_Values5.mean(),4))
    
# %% Exporting to Excel
# df_metrics = pd.DataFrame({
#    'Mean Squared Error': mse_list,
#    'Mean Absolute Error': mae_list,
#    'R^2 Score': r2_list,
#    'MAPE': mape_values,
#    'RMSE': rmse_list
# })

df_metrics = pd.DataFrame(smape_values, index=range(1, 16), columns=range(1, 16))   
df_metrics_a20 = pd.DataFrame(a_20values, index=range(1, 16), columns=range(1, 16))   
df_metrics_mae = pd.DataFrame(mae_values, index=range(1, 16), columns=range(1, 16))
    
file_path = "C:/Users/ryanj/Code/Research_THz/excel/NewDataset.xlsx"

#Export the DataFrame to an Excel file on a specific sheet
with pd.ExcelWriter(file_path, mode='a', engine='openpyxl') as writer:
    df_metrics.to_excel(writer, sheet_name='KNN_SMAPE_new', index=False, startrow=0, startcol=0)
    #df_metrics_a20.to_excel(writer, sheet_name='KNN_A20_standar', index=False, startrow=0, startcol=0)
    #df_metrics_mae.to_excel(writer, sheet_name='KNN_MAE_standared', index=False, startrow=0, startcol=0)

# %%

