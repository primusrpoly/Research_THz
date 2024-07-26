import pandas as pd
from scipy.io import loadmat

# Load the MATLAB .mat file
mat_data = loadmat('All the Data.mat')

array_data = mat_data['result_array']

# Define column names
column_names = ['PilotLength', 'PilotSpacing', 'SymbolRate', 'PhaseNoise', 'BER', 'CBER']

# Create a pandas DataFrame
df = pd.DataFrame(array_data, columns=column_names)

# Export sorted DataFrame to Excel
file_path = "C:/Users/ryanj/Code/Research_THz/excel/CBER.xlsx"

with pd.ExcelWriter(file_path, mode='a', engine='openpyxl') as writer:
    df.to_excel(writer, sheet_name='Dataset2', index_label='Depth')