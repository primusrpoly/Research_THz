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
# X = df[['PhaseNoise', 'PilotLength', 'PilotSpacing', 'SymbolRate', 'SNR']]
# #print("X:", X)
# y = df['CBER']
# #print("y:\n", y)

#x and y split
X = df[['PhaseNoise', 'SymbolRate', 'SNR']]
#print("X:", X)
y = df['CBER']
#print("y:\n", y)


#Normalize for kNN ONLY
# scaler = MinMaxScaler()
# X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)


#%% Regressors

lnr = LinearRegression()
knn = KNeighborsRegressor(n_neighbors=1, weights='distance')
dtr = DecisionTreeRegressor(max_depth=12,random_state=17,criterion='squared_error')
rfr = RandomForestRegressor(n_estimators=5, random_state=17, max_depth=15)
gbr = GradientBoostingRegressor(max_depth=4, random_state=17, n_estimators=100, learning_rate=0.1)
abr = AdaBoostRegressor(estimator=dtr,random_state=17,n_estimators=5,learning_rate=0.01)

#%% Test and analysis

# Train the model
abr.fit(X_train, y_train)

# Make predictions
y_pred = abr.predict(X_test)
#print(X_test)

#RMSE Calculations
def rms_error(actual, predicted):
    mse = np.mean((actual - predicted) ** 2)
    rmse = np.sqrt(mse)
    return rmse

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

#Statistical calculations
#SHOULD BE TEST NOT TRAIN
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2=r2_score(y_test,y_pred)
mape=Accuracy_score(y_test, y_pred)
rmse=rms_error(y_test,y_pred)
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

#knn_X_test_df = pd.DataFrame(scaler.inverse_transform(X_test), columns=['PhaseNoise', 'PilotLength', 'PilotSpacing', 'SymbolRate', 'SNR'])
X_test_df = pd.DataFrame((X_test), columns=['PhaseNoise', 'SymbolRate', 'SNR'])


# Expected vs actual dataframe
results_df = pd.DataFrame({
    'Expected/Test': y_test,
    'Predicted': y_pred
})

file_path = "C:/Users/ryanj/Code/Research_THz/excel/NewDataset.xlsx"

# Export to Excel
with pd.ExcelWriter(file_path, mode='a', engine='openpyxl') as writer:
    #X_test_df.to_excel(writer, sheet_name='ABR_Features', index=False, startrow=0, startcol=0)
    results_df.to_excel(writer, sheet_name='ABR_TvP12', index=False, startrow=0, startcol=0)
# %%
