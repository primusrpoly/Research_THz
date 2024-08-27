from scipy.io import loadmat
import pandas as pd

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
file_path = "C:/Users/ryanj/Code/Research_THz/excel/NewDataset.xlsx"

with pd.ExcelWriter(file_path, mode='a', engine='openpyxl') as writer:
    df.to_excel(writer, sheet_name='NewDataset')
