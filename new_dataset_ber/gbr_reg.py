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
y = df['BER']
#print("y:\n", y)

#Normalize
# scaler = MinMaxScaler()
# X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)

# Number of estimators for AdaBoost

#trees: 5,10,15,20,25,30,35,40,45,50,75,100,250,500,750,1000
#5,15,25,30,40,50,75,100,250,500,750,1000,1250,1500,1750,2000
#estimators: 5,25,35,50,75,100,250,500,750,1000,1250,1750,2000,2500,3000
#5,25,35,50,75,100,250,500,750,1000,1250,1750,2000,2500,3000,5000,6000,7500,8500,10000
#learning rate: 0.01,0.05,0.06,0.07,0.08,0.09,0.1,0.11,.12,.13,.14,.15,.25,.5,1,2
#depth: 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

trees = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
l_rate = [0.1]

def transform_ber(ber):
    return np.minimum(ber, 1 - ber)
    
def Accuracy_score(orig, pred):
    orig = transform_ber(orig)
    pred = transform_ber(pred)
    
    exp_orig = np.exp(orig)
    exp_pred = np.exp(pred)
    
    mape = np.abs(exp_orig - exp_pred) / np.abs(exp_orig)
    return np.mean(mape) 

def Accuracy_score3(orig, pred):
    orig = np.array(orig)
    pred = np.array(pred)
    
    count = 0
    for i in range(len(orig)):
        if(pred[i] <= 1.2 * orig[i]) and (pred[i] >= 0.8 * orig[i]):
            count += 1
    a_20 = count / len(orig)
    return a_20

custom_Scoring = make_scorer(Accuracy_score, greater_is_better=True)


#%% Training
cv_folds = 15
num_depths = len(trees)
num_epi = len(l_rate)

rows = num_depths * num_epi

all_cv_scores = np.zeros((rows, cv_folds))
all_cv_scores2 = np.zeros((rows, cv_folds))
all_cv_scores3 = np.zeros((rows, cv_folds))
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

        #MAE
        custom_Scoring5 = make_scorer(mean_absolute_error, greater_is_better=True)

        # Running cross validation
        Accuracy_Values5 = cross_val_score(gbr, X_train, y_train, cv=CV, scoring=custom_Scoring5)
        
        print('\nMAE for 5-fold Cross Validation:\n', Accuracy_Values5)
        print('\nFinal Average Accuracy MAE index of the model:', round(Accuracy_Values5.mean(), 4))
        
        all_cv_scores3[current_row, :] = Accuracy_Values5

        current_row += 1
#%% Export

#MAPE
df_metrics = pd.DataFrame(all_cv_scores, columns=[f'Fold {i+1}' for i in range(cv_folds)])
#A20
df_metricsA20 = pd.DataFrame(all_cv_scores2, columns=[f'Fold {i+1}' for i in range(cv_folds)])
#MAE
df_metrics_MAE = pd.DataFrame(all_cv_scores3, columns=[f'Fold {i+1}' for i in range(cv_folds)])

file_path = "C:/Users/ryanj/Code/Research_THz/excel/BERAgain.xlsx"

with pd.ExcelWriter(file_path, mode='a', engine='openpyxl') as writer:
    df_metrics.to_excel(writer, sheet_name='GBR_SMAPE_depth', index_label='Depth')
    df_metricsA20.to_excel(writer, sheet_name='GBR_A20_depth', index_label='Depth')
    df_metrics_MAE.to_excel(writer, sheet_name='GBR_MAE_depth', index_label='Depth')
# %%
