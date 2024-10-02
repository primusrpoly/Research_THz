import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from scipy.optimize import differential_evolution
from scipy.io import loadmat

#%% Initilize

# load
mat_data = loadmat('full_Results.mat')

# extract
array_data = mat_data['Results']

# Initialize lists to store each column of the DataFrame
modulation_order = []
snr = []
pilot_length = []
pilot_spacing = []
symbol_rate = []
phase_noise = []
ber = []
cber = []

# Loop through each element in array and extract data
for result in array_data[0]:
    modulation_order.append(result['Modulation_order'][0][0])
    snr.append(result['SNR'][0][0])
    pilot_length.append(result['pilot_length'][0][0])
    pilot_spacing.append(result['symbols_between_pilot'][0][0])
    symbol_rate.append(result['symbol_rate'][0][0])
    phase_noise.append(result['phase_noise'][0][0])
    ber.append(result['BER'][0][0])
    cber.append(result['CBER'][0][0])

#creating Dataframe
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

X = df[['PhaseNoise', 'PilotLength', 'PilotSpacing', 'SymbolRate', 'SNR']]
#print("X:", X)
y = df['CBER']
#print("y:\n", y)

#Normalize
# scaler = MinMaxScaler()
# X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)

# training abr
abr = AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=15,random_state=17,criterion='squared_error'),
                        random_state=17,n_estimators=250,learning_rate=0.1)
abr.fit(X_train, y_train)

#abr predicting/getting accuracy of abr
y_pred = abr.predict(X_test)
#mse = mean_squared_error(y_test, y_pred)
#print(f'ABR Mean Squared Error: {mse}')

#[5, 2.000000884519867, 1023.9971124558101, 300000675.7525711, 12.500013410679896]: 0.0
#[5, 2, 1024, 300000676, 12.5]: 0.00224

def predict_ber(abr):
    phase_noise = 5
    pilot_length = 2.000000884519867
    pilot_spacing = 1023.9971124558101
    symbol_rate = 300000675.7525711
    snr = 12.500013410679896
    
    data = pd.DataFrame([[phase_noise, pilot_length, pilot_spacing, symbol_rate, snr]],
                        columns=['PhaseNoise', 'PilotLength', 'PilotSpacing', 'SymbolRate', 'SNR'])
    params = phase_noise, pilot_length, pilot_spacing, symbol_rate, snr
    
    prediction = abr.predict(data)[0]
    #print([params])
    
    print(f"Predicted BER for params {params}: {prediction}")
    return prediction

predict_ber(abr)

#5, 2.000000884519867, 1023.9971124558101, 300000675.7525711, 12.500013410679896
