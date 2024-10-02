import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
from scipy.io import loadmat

# Load and prepare the data
def load_data(file_path):
    # Load .mat file
    file_path = 'full_Results.mat'
    mat_data = loadmat(file_path)

    # Extract
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

    # Create a DataFrame
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

    return df

# Train the AdaBoost Regressor model
def train_model(df):
    X = df[['PhaseNoise', 'PilotLength', 'PilotSpacing', 'SymbolRate', 'SNR']]
    y = df['CBER']

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)

    # Training AdaBoost Regressor
    abr = AdaBoostRegressor(
        estimator=DecisionTreeRegressor(max_depth=15, random_state=17, criterion='squared_error'),
        random_state=17, n_estimators=250, learning_rate=0.1
    )
    abr.fit(X_train, y_train)
    
    return abr

# Define function to predict CBER
def calculate_cber(abr, phase_noise, pilot_length, pilot_spacing, symbol_rate, snr):
    """
    Function to calculate the predicted CBER using the trained AdaBoost model.

    Args:
    - abr (AdaBoostRegressor): Trained AdaBoost regressor model
    - phase_noise (float): Phase noise value
    - pilot_length (float): Pilot length value
    - pilot_spacing (float): Pilot spacing value
    - symbol_rate (float): Symbol rate value
    - snr (float): Signal-to-noise ratio value

    Returns:
    - cber (float): Predicted CBER value
    """
    return abr.predict([[phase_noise, pilot_length, pilot_spacing, symbol_rate, snr]])[0]
