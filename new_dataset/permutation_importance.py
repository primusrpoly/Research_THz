import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.inspection import permutation_importance

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
# X = df[['PhaseNoise', 'PilotLength', 'PilotSpacing', 'SymbolRate', 'SNR']]
# #print("X:", X)
# y = df['CBER']
# #print("y:\n", y)
X = df[['PhaseNoise', 'SymbolRate', 'SNR']]
#print("X:", X)
y = df['CBER']
#print("y:\n", y)

# Normalize FOR kNN ONLY
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=17)
# knn.fit(X_train, y_train)
# knn_predictions = knn.predict(X_test)

# Initialize list for feature importances
feature_importances_list = []

#%% Regressors
"""
kNN - k_neighbors = 1, weights = 'distance'				
ABR - estimators = 25, lr = 1, max_depth = 15				
RFR - estimators = 5, max_depth = 15			
GBR - estimators = 75, lr = 0.11, max_depth = 12				
"""

knn = KNeighborsRegressor(n_neighbors=1, weights='distance')
dtr = DecisionTreeRegressor(max_depth=15, random_state=17)
rfr = RandomForestRegressor(n_estimators=5, random_state=17, max_depth=15)
gbr = GradientBoostingRegressor(max_depth=12, random_state=17, n_estimators=75, learning_rate=0.11)
abr = AdaBoostRegressor(estimator=dtr, random_state=17, n_estimators=5, learning_rate=0.01)

#%%Permutation Importance

#PI Calculation
for state in range(20):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=state)
    abr.fit(X_train, y_train)
    result = permutation_importance(abr, X_test, y_test, n_repeats=10, random_state=state)
    #summing to 1
    normalized_importances = result.importances_mean / np.sum(result.importances_mean)
    feature_importances_list.append(normalized_importances)

#DataFrame for feature importances
feature_importances_df = pd.DataFrame(feature_importances_list, columns=['PhaseNoise', 'SymbolRate', 'SNR']).transpose()

feature_importances_df.insert(0, 'Feature', feature_importances_df.index)
feature_importances_df.reset_index(drop=True, inplace=True)

#Export
file_path = "C:/Users/ryanj/Code/Research_THz/excel/NewDataset.xlsx"
with pd.ExcelWriter(file_path, mode='a', engine='openpyxl') as writer:
    feature_importances_df.to_excel(writer, sheet_name='ABRPermutationImportances', index=False)


