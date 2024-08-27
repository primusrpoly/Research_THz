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

# training abr
abr = AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=15,random_state=17,criterion='squared_error'),
                        random_state=17,n_estimators=250,learning_rate=0.1)
abr.fit(X_train, y_train)

#abr predicting/getting accuracy of abr
y_pred = abr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f'ABR Mean Squared Error: {mse}')

def Accuracy_score3(orig, pred):
    orig = np.array(orig)
    pred = np.array(pred)
    
    count = 0
    for i in range(len(orig)):
        if(pred[i] <= 1.2 * orig[i]) and (pred[i] >= 0.8 * orig[i]):
            count += 1
    a_20 = count / len(orig)
    return a_20

a_20 = Accuracy_score3(y_test, y_pred)
print(f'ABR A20: {a_20}')


### INPUT PHASE NOISE ###
phase_noise_value = 50

# ### Desired BER target ###
# desired_ber_target = 0


print(f"Finding optimal parameters for {phase_noise_value} GHz using ABR...")

#Function to predict BER using trained regressor
def predict_ber(params):
    phase_noise, pilot_length, pilot_spacing, symbol_rate, snr = params
    data = pd.DataFrame([[phase_noise, pilot_length, pilot_spacing, symbol_rate, snr]],
                        columns=['PhaseNoise', 'PilotLength', 'PilotSpacing', 'SymbolRate', 'SNR'])
    prediction = abr.predict(data)[0]
    #print(f"Predicted BER for params {params}: {prediction}")
    return prediction

##Penalties for PL, PS, SR

#PL -> low as possible
def penalty_pilot_length(pilot_length, min_length=2):
    return (pilot_length - min_length) / min_length

#PS -> high as possible
def penalty_pilot_spacing(pilot_spacing, max_spacing=1024):
    return (max_spacing - pilot_spacing) / max_spacing

#SR -> low as possible
def penalty_symbol_rate(symbol_rate, min_rate=1e6):
    return (symbol_rate - min_rate) / min_rate

#SNR -> low as possible
def penalty_snr(snr, min_snr=1):
    return (snr - min_snr) / min_snr

#Minimizing all our parameters
def combined_objective(params):
    phase_noise, pilot_length, pilot_spacing, symbol_rate, snr = params
    ber = predict_ber(params)
    penalty = (penalty_pilot_length(pilot_length) +
               penalty_pilot_spacing(pilot_spacing) +
               penalty_symbol_rate(symbol_rate) +
               penalty_snr(snr))
    weight_ber = 2  #BER weight
    weight_penalty = 0.00001 #penalty weight
    return weight_ber * ber + weight_penalty * penalty

    #penalty for BER deviation from the desired target
    # ber_penalty = abs(ber - desired_ber_target)

    # return weight_ber * ber_penalty + weight_penalty * penalty

#Entire function for optimal parameters from a phase noise value
def find_optimal_parameters(phase_noise_value):
    # Bounds, phase noise is fixed
    bounds = [(phase_noise_value, phase_noise_value), (2, 64), (16, 1024), (1e6, 3e10), (0, 50)]
    
    #Differential evolution
    def de_objective(params):
        params = [phase_noise_value] + list(params)
        return combined_objective(params)
    
    #Optimization using differential evolution
    result = differential_evolution(de_objective, bounds[1:], strategy='best1bin', maxiter=1000, tol=1e-7)
    optimal_pilot_length, optimal_pilot_spacing, optimal_symbol_rate, optimal_snr = result.x
    
    #BER after optimizing
    final_ber = predict_ber([phase_noise_value, optimal_pilot_length, optimal_pilot_spacing, optimal_symbol_rate, optimal_snr])
    print(f"Final BER: {final_ber}")

    return optimal_pilot_length, optimal_pilot_spacing, optimal_symbol_rate, optimal_snr

#Print
optimal_pilot_length, optimal_pilot_spacing, optimal_symbol_rate, optimal_snr = find_optimal_parameters(phase_noise_value)

print(f"Optimal Pilot Length: {optimal_pilot_length}")
print(f"Optimal Pilot Spacing: {optimal_pilot_spacing}")
print(f"Optimal Symbol Rate: {optimal_symbol_rate}")
print(f"Optimal SNR: {optimal_snr}")

