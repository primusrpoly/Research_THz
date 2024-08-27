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

#Normalize FOR kNN ONLY
# scaler = MinMaxScaler()
# X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)

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
        if(pred[i] <= 1.2*orig[i]) and (pred[i] >= 0.8*orig[i]):
            count += 1
    a_20 = count/len(orig)
    return a_20

custom_Scoring = make_scorer(Accuracy_score,greater_is_better = True)
#%% Regressors

lnr = LinearRegression()
knn = KNeighborsRegressor(n_neighbors=1, weights='distance')
dtr = DecisionTreeRegressor(max_depth=15,random_state=17, criterion='squared_error')
rfr = RandomForestRegressor(n_estimators=5, random_state=17, max_depth=15)
gbr = GradientBoostingRegressor(max_depth=15, random_state=17, n_estimators=100, learning_rate=0.1)
abr = AdaBoostRegressor(estimator=dtr,random_state=17,n_estimators=25,learning_rate=1)

#%% SMAPE

cv_scores = RepeatedKFold(n_splits=5, n_repeats = 3, random_state = 17)
Accuracy_Values = cross_val_score(knn, X, y, cv = cv_scores, scoring = custom_Scoring)

print('\nAccuracy values for k-fold Cross Validation:\n', Accuracy_Values)
print('\nFinal Average Accuracy of the model:', round(Accuracy_Values.mean(), 2))


#%% A-20

#custom scoring a_20 calulation
custom_Scoring3 = make_scorer(Accuracy_score3,greater_is_better=True)

#Running cross validation
CV = RepeatedKFold(n_splits = 5, n_repeats=3, random_state = 17)
Accuracy_Values3 = cross_val_score(knn,X ,y,\
                                   cv=CV,scoring=custom_Scoring3)

print('\n"a_20 index" for 5-fold Cross Validation:\n', Accuracy_Values3)
print('\nFinal Average Accuracy a_20 index of the model:', round(Accuracy_Values3.mean(),4))

#MAE
custom_Scoring5 = make_scorer(mean_absolute_error, greater_is_better=True)

CV = RepeatedKFold(n_splits = 5, n_repeats=3, random_state = 17)
Accuracy_Values5 = cross_val_score(knn,X ,y,\
                                   cv=CV,scoring=custom_Scoring5)

print('\n"MAE" for 5-fold Cross Validation:\n', Accuracy_Values5)
print('\nFinal Average Accuracy MAE of the model:', round(Accuracy_Values5.mean(),4))


#%% Export

#SMAPE
df_metrics_SMAPE = pd.DataFrame(Accuracy_Values, index=range(1, 16), columns=range(1, 2))   
#A20
df_metrics_A20 = pd.DataFrame(Accuracy_Values3, index=range(1, 16), columns=range(1, 2))   
#MAE
df_metrics_MAE = pd.DataFrame(Accuracy_Values5, index=range(1, 16), columns=range(1, 2))   

file_path = "C:/Users/ryanj/Code/Research_THz/excel/NewDataset.xlsx"

#Export the DataFrame to an Excel file on a specific sheet
with pd.ExcelWriter(file_path, mode='a', engine='openpyxl') as writer:
    df_metrics_SMAPE.to_excel(writer, sheet_name='knn_Final_SMAPE', index=False, startrow=0, startcol=0)
    df_metrics_A20.to_excel(writer, sheet_name='knn_Final_A20', index=False, startrow=0, startcol=0)
    df_metrics_MAE.to_excel(writer, sheet_name='knn_Final_MAE', index=False, startrow=0, startcol=0)