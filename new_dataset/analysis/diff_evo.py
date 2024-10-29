import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from scipy.optimize import differential_evolution
from scipy.io import loadmat
import openpyxl

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
y = df['CBER']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)

abr = AdaBoostRegressor(
    estimator=DecisionTreeRegressor(max_depth=15, random_state=17, criterion='squared_error'),
    random_state=17, n_estimators=250, learning_rate=0.1
)
abr.fit(X_train, y_train)

def find_optimal_parameters(phase_noise_value, de_params):
    def round_to_nearest_5(value):
        return round(value * 2) / 2

    if 0 <= phase_noise_value < 5:
        symbol_rate_min = 300e6
    elif 5 <= phase_noise_value < 10:
        symbol_rate_min = 1e9
    elif 10 <= phase_noise_value < 15:
        symbol_rate_min = 10e9
    else:
        symbol_rate_min = 30e9
    symbol_rate_max = 40e9
    snr_min = snr_max = 12.5

    # Penalty 
    def penalty_pilot_length(pilot_length, min_length=2):
        return (pilot_length - min_length) / min_length
    
    def penalty_pilot_spacing(pilot_spacing, max_spacing=1024):
        return (max_spacing - pilot_spacing) / max_spacing
    
    def penalty_symbol_rate(symbol_rate):
        return (symbol_rate - symbol_rate_min) / symbol_rate_min
    
    def penalty_snr(snr):
        return (snr - snr_min) / snr_min

    #redict BER 
    def predict_ber(params):
        phase_noise, pilot_length, pilot_spacing, symbol_rate, snr = params
        data = pd.DataFrame([[phase_noise, pilot_length, pilot_spacing, symbol_rate, snr]],
                            columns=['PhaseNoise', 'PilotLength', 'PilotSpacing', 'SymbolRate', 'SNR'])
        prediction = abr.predict(data)[0]
        if prediction < 1e-9:
            prediction = 0
        else:
            prediction = round(prediction, 8)
        return prediction
    
    #combined objective function
    def combined_objective(params):
        phase_noise, pilot_length, pilot_spacing, symbol_rate, snr = params
        ber = predict_ber(params)
        penalty = (penalty_pilot_length(pilot_length) +
                   penalty_pilot_spacing(pilot_spacing) +
                   penalty_symbol_rate(symbol_rate) +
                   penalty_snr(snr))
        weight_ber = 2
        weight_penalty = 0.1
        return weight_ber * ber + weight_penalty * penalty

    bounds = [
        (phase_noise_value, phase_noise_value),
        (2, 64),  # Pilot Length
        (16, 1024),  # Pilot Spacing
        (symbol_rate_min, symbol_rate_max),  # Symbol Rate
        (snr_min, snr_max)  # SNR
    ]

    # Diff evo optimization
    def de_objective(params):
        rounded_params = [
            round(params[0]),                   # Pilot Length
            round(params[1]),                   # Pilot Spacing
            round(params[2]),                   # Symbol Rate
            round_to_nearest_5(params[3])       # SNR
        ]
        return combined_objective([phase_noise_value] + rounded_params)

    result = differential_evolution(
        de_objective, bounds[1:], **de_params
    )
    
    optimal_pilot_length, optimal_pilot_spacing, optimal_symbol_rate, optimal_snr = result.x
    final_ber = predict_ber([
        phase_noise_value, round(optimal_pilot_length), round(optimal_pilot_spacing),
        round(optimal_symbol_rate), round_to_nearest_5(optimal_snr)
    ])
    return round(optimal_pilot_length), round(optimal_pilot_spacing), round(optimal_symbol_rate), round_to_nearest_5(optimal_snr), final_ber

# phase noise values and hyperparameters
phase_noise_values = [30]
de_strategies = ['best1bin', 'rand1bin', 'randtobest1bin']
max_iters = [25, 50, 100, 200]
pop_sizes = [5, 10, 15, 20]

results = []

# Loop pn values and hyperparameters
for phase_noise in phase_noise_values:
    for strategy in de_strategies:
        for max_iter in max_iters:
            for pop_size in pop_sizes:
                de_params = {
                    'strategy': strategy,
                    'maxiter': max_iter,
                    'popsize': pop_size,
                    'tol': 1e-6
                }
                print(f"Phase Noise: {phase_noise}, Strategy: {strategy}, Max Iter: {max_iter}, Pop Size: {pop_size}")
                optimal_pilot_length, optimal_pilot_spacing, optimal_symbol_rate, optimal_snr, final_ber = find_optimal_parameters(phase_noise, de_params)
                results.append([phase_noise, strategy, max_iter, pop_size, optimal_pilot_length, optimal_pilot_spacing, optimal_symbol_rate, optimal_snr, final_ber])

# df
results_df = pd.DataFrame(results, columns=[
    'PhaseNoise', 'Strategy', 'MaxIter', 'PopSize', 'PilotLength', 'PilotSpacing', 'SymbolRate', 'SNR', 'FinalBER'
])

# Export
file_path = "C:/Users/ryanj/Code/Research_THz/excel/NewDataset.xlsx"

with pd.ExcelWriter(file_path, mode='a', engine='openpyxl') as writer:
    results_df.to_excel(writer, sheet_name='Diff Evo 30', index_label='Depth')