import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from scipy.optimize import differential_evolution
from scipy.io import loadmat
import seaborn as sns
import matplotlib.pyplot as plt
import shap

#%% Initialize

# Load data from MAT file
mat_data = loadmat('full_Results.mat')
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

# Extract data from MAT file
for result in array_data[0]:
    modulation_order.append(result['Modulation_order'][0][0])
    snr.append(result['SNR'][0][0])
    pilot_length.append(result['pilot_length'][0][0])
    pilot_spacing.append(result['symbols_between_pilot'][0][0])
    symbol_rate.append(result['symbol_rate'][0][0])
    phase_noise.append(result['phase_noise'][0][0])
    ber.append(result['BER'][0][0])
    cber.append(result['CBER'][0][0])

# Create DataFrame
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

# Plot pairwise relationships
# sns.pairplot(df[['SNR', 'SymbolRate', 'PhaseNoise', 'CBER']])
# plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Assuming df is your DataFrame and contains 'SNR', 'SymbolRate', 'PhaseNoise', and 'BER'

# Create a 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the points in 3D
sc = ax.scatter(df['SNR'], df['SymbolRate'], df['PhaseNoise'], c=df['CBER'], cmap='coolwarm')

# Add labels for each axis
ax.set_xlabel('SNR')
ax.set_ylabel('SymbolRate')
ax.set_zlabel('PhaseNoise')

# Set title
ax.set_title('3D Scatter Plot: SNR, SymbolRate, PhaseNoise vs BER')

# Add color bar to show the color scale of BER
cbar = plt.colorbar(sc)
cbar.set_label('BER')

# Show the plot
plt.show()
