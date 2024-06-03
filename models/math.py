import pandas as pd

# Input data
data = {
    "Feature": ["PilotLength", "PilotSpacing", "SymbolRate", "PhaseNoise"],
    0: [0.0, 0.003646349, 0.051574657, 1.0],
    1: [0.0, 0.001263418, 0.039687748, 1.0],
    2: [0.001308438, 0.0, 0.043333551, 1.0],
    3: [0.0, 0.008446461, 0.042134317, 1.0],
    4: [0.0, 0.003108927, 0.046944893, 1.0],
    5: [0.0, 0.002956909, 0.055370469, 1.0],
    6: [0.0, 0.005825211, 0.046479301, 1.0],
    7: [0.0, 0.003637675, 0.043206714, 1.0],
    8: [0.0, 0.00279633, 0.052344665, 1.0],
    9: [0.0, 0.000386155, 0.049421447, 1.0],
    10: [0.0, 0.005546799, 0.048408159, 1.0],
    11: [0.0, 0.002064876, 0.042911613, 1.0],
    12: [0.0, 0.004132638, 0.034430416, 1.0],
    13: [0.0, 0.002102181, 0.042354457, 1.0],
    14: [0.0, 0.004673009, 0.048481309, 1.0],
    15: [0.0, 0.000360894, 0.048190524, 1.0],
    16: [0.0, 0.002372619, 0.045960979, 1.0],
    17: [0.0, 0.000116452, 0.051698526, 1.0],
    18: [0.0, 0.002881157, 0.03517014, 1.0],
    19: [0.0, 0.001072018, 0.045262411, 1.0]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Set 'Feature' as index
df.set_index("Feature", inplace=True)

# Normalize the data
df_normalized = df.div(df.sum(axis=0), axis=1)

# Print the normalized DataFrame
print(df_normalized)

file_path = "C:/Users/ryanj/Code/Research_THz/excel/Book1.xlsx"
with pd.ExcelWriter(file_path, mode='a', engine='openpyxl') as writer:
    df_normalized.to_excel(writer, sheet_name='FeatureImportanceskNN2', index=False)


