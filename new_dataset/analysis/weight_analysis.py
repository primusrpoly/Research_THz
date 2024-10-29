import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from scipy.optimize import differential_evolution
from scipy.io import loadmat
import itertools


# Load the .mat file
mat_data = loadmat('full_Results.mat')

# Extract the 'Results' struct
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

# Loop through each element in the struct array and extract the data
for result in array_data[0]:
    modulation_order.append(result['Modulation_order'][0][0])
    snr.append(result['SNR'][0][0])
    pilot_length.append(result['pilot_length'][0][0])
    pilot_spacing.append(result['symbols_between_pilot'][0][0])
    symbol_rate.append(result['symbol_rate'][0][0])
    phase_noise.append(result['phase_noise'][0][0])
    ber.append(result['BER'][0][0])
    cber.append(result['CBER'][0][0])

# Create a DataFrame from the extracted data
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

# Extract features (X) and target (y) variables
X = df[['PhaseNoise', 'PilotLength', 'PilotSpacing', 'SymbolRate', 'SNR']]
#print("X:", X)
y = df['CBER']
#print("y:\n", y)


#Normalize
# scaler = MinMaxScaler()
# X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)


knn = KNeighborsRegressor(n_neighbors=1, weights='distance')
abr = AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=15,random_state=17,criterion='squared_error'),
                        random_state=17,n_estimators=250,learning_rate=0.1)
abr.fit(X_train, y_train)

y_pred = abr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

#Function to predict BER using trained regressor
def predict_ber(params):
    phase_noise, pilot_length, pilot_spacing, symbol_rate, snr = params
    data = pd.DataFrame([[phase_noise, pilot_length, pilot_spacing, symbol_rate, snr]],
                        columns=['PhaseNoise', 'PilotLength', 'PilotSpacing', 'SymbolRate', 'SNR'])
    prediction = abr.predict(data)[0]
    #print(f"Predicted BER for params {params}: {prediction}")
    return prediction

def penalty_pilot_spacing(pilot_spacing, max_spacing=1024):
    return (max_spacing - pilot_spacing) / max_spacing

def penalty_pilot_length(pilot_length, min_length=8):
    return (pilot_length - min_length) / min_length

def penalty_symbol_rate(symbol_rate, min_rate=1e6):
    return (symbol_rate - min_rate) / min_rate

def penalty_snr(snr, min_snr=1):
    return (snr - min_snr) / min_snr

def combined_objective(params, weight_ber, weight_penalty):
    phase_noise, pilot_length, pilot_spacing, symbol_rate, snr = params
    ber = predict_ber(params)
    penalty = penalty_pilot_spacing(pilot_spacing) + penalty_pilot_length(pilot_length) + penalty_symbol_rate(symbol_rate) + penalty_snr(snr)
    return weight_ber * ber + weight_penalty * penalty

def find_optimal_parameters(phase_noise_value, weight_ber, weight_penalty):
    bounds = [(phase_noise_value, phase_noise_value), (8, 64), (32, 1024), (1e6, 3e10), (0, 50)]
    
    def de_objective(params):
        params = [phase_noise_value] + list(params)
        return combined_objective(params, weight_ber, weight_penalty)
    
    result = differential_evolution(de_objective, bounds[1:], strategy='best1bin', maxiter=1000, tol=1e-7)
    
    optimal_pilot_length, optimal_pilot_spacing, optimal_symbol_rate, optimal_snr = result.x
    
    final_ber = predict_ber([phase_noise_value, optimal_pilot_length, optimal_pilot_spacing, optimal_symbol_rate, optimal_snr])
    print(f"Final BER: {final_ber}")

    return final_ber, optimal_pilot_length, optimal_pilot_spacing, optimal_symbol_rate, optimal_snr

#different phase noises
phase_noise_values = [10,15,20,25,30,35]  

#different weight combinations
weights_ber = [0.5, 1.0, 1.25, 1.5, 1.75, 2.0]  #weights for BER
weights_penalty = [0.00001, 0.0001, 0.001, 0.01, 0.05, 0.1]  #  weights for penalties

results = []

for phase_noise_value in phase_noise_values:
    for weight_ber, weight_penalty in itertools.product(weights_ber, weights_penalty):
        final_ber, optimal_pilot_length, optimal_pilot_spacing, optimal_symbol_rate, optimal_snr = find_optimal_parameters(phase_noise_value, weight_ber, weight_penalty)
        results.append({
            'Phase Noise': phase_noise_value,
            'BER Weight': weight_ber,
            'Penalty Weight': weight_penalty,
            'Final BER': final_ber,
            'Optimal Pilot Length': optimal_pilot_length,
            'Optimal Pilot Spacing': optimal_pilot_spacing,
            'Optimal Symbol Rate': optimal_symbol_rate,
            'Optimal SNR': optimal_snr
        })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

file_path = "C:/Users/ryanj/Code/Research_THz/excel/NewDataset.xlsx"

# Export to Excel
with pd.ExcelWriter(file_path, mode='a', engine='openpyxl') as writer:
    results_df.to_excel(writer, sheet_name='Total Weight Analysis', index=False, startrow=0, startcol=0)

# Print results
# for result in results:
#     print(f"Weight BER: {result[0]}, Weight Penalty: {result[1]}, Final BER: {result[2]}, Optimal Pilot Length: {result[3]}, Optimal Pilot Spacing: {result[4]}, Optimal Symbol Rate: {result[5]}")
