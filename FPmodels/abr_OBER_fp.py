import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from scipy.optimize import differential_evolution
from scipy.io import loadmat

#Load
mat_data = loadmat('All the Data.mat')
array_data = mat_data['result_array']

column_names = ['PilotLength', 'PilotSpacing', 'SymbolRate', 'PhaseNoise', 'BER', 'OBER']

#dataframe
df = pd.DataFrame(array_data, columns=column_names)

X = df[['PhaseNoise', 'PilotLength', 'PilotSpacing', 'SymbolRate']]
y = df['OBER']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=8)

# training abr
abr = AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=13,random_state=17,criterion='squared_error'),
                        random_state=17,n_estimators=35,learning_rate=0.15)
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
phase_noise_value = 0


print(f"Finding optimal parameters for {phase_noise_value} GHz using ABR...")

#Function to predict BER using trained regressor
def predict_ber(params):
    phase_noise, pilot_length, pilot_spacing, symbol_rate = params
    data = pd.DataFrame([[phase_noise, pilot_length, pilot_spacing, symbol_rate]],
                        columns=['PhaseNoise', 'PilotLength', 'PilotSpacing', 'SymbolRate'])
    prediction = abr.predict(data)[0]
    #print(f"Predicted BER for params {params}: {prediction}")
    return prediction

##Penalties for PL, PS, SR

#PL -> low as possible
def penalty_pilot_length(pilot_length, min_length=8):
    return (pilot_length - min_length) / min_length

#PS -> high as possible
def penalty_pilot_spacing(pilot_spacing, max_spacing=1024):
    return (max_spacing - pilot_spacing) / max_spacing

#SR -> low as possible
def penalty_symbol_rate(symbol_rate, min_rate=1e6):
    return (symbol_rate - min_rate) / min_rate

#Minimizing all our parameters
def combined_objective(params):
    phase_noise, pilot_length, pilot_spacing, symbol_rate = params
    ber = predict_ber(params)
    penalty = penalty_pilot_spacing(pilot_spacing) + penalty_pilot_length(pilot_length) + penalty_symbol_rate(symbol_rate)
    weight_ber = 2  #BER weight
    weight_penalty = 0.00001 #penalty weight
    return weight_ber * ber + weight_penalty * penalty

#Entire function for optimal parameters from a phase noise value
def find_optimal_parameters(phase_noise_value):
    # Bounds, phase noise is fixed
    bounds = [(phase_noise_value, phase_noise_value), (8, 64), (32, 1024), (1e6, 3e10)]
    
    #Differential evolution
    def de_objective(params):
        params = [phase_noise_value] + list(params)
        return combined_objective(params)
    
    #Optimization using differential evolution
    result = differential_evolution(de_objective, bounds[1:], strategy='best1bin', maxiter=1000, tol=1e-7)
    optimal_pilot_length, optimal_pilot_spacing, optimal_symbol_rate = result.x
    
    #BER after optimizing
    final_ber = predict_ber([phase_noise_value, optimal_pilot_length, optimal_pilot_spacing, optimal_symbol_rate])
    print(f"Final BER: {final_ber}")

    return optimal_pilot_length, optimal_pilot_spacing, optimal_symbol_rate

#Print
optimal_pilot_length, optimal_pilot_spacing, optimal_symbol_rate = find_optimal_parameters(phase_noise_value)

print(f"Optimal Pilot Length: {optimal_pilot_length}")
print(f"Optimal Pilot Spacing: {optimal_pilot_spacing}")
print(f"Optimal Symbol Rate: {optimal_symbol_rate}")

