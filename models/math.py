import pandas as pd

# Input data
data = {
    "Feature": ["PilotLength", "PilotSpacing", "SymbolRate", "PhaseNoise"],
    0: [0.0, 0.05881331, 0.604170369, 1.0],
    1: [0.0, 0.036047271, 0.63820126, 1.0],
    2: [0.0, 0.03586657, 0.629657289, 1.0],
    3: [0.0, 0.04830742, 0.58548178, 1.0],
    4: [0.0, 0.042441668, 0.611178277, 1.0],
    5: [0.0, 0.060051794, 0.680372818, 1.0],
    6: [0.0, 0.043773163, 0.68087626, 1.0],
    7: [0.0, 0.052556772, 0.595876297, 1.0],
    8: [0.0, 0.053921855, 0.598812884, 1.0],
    9: [0.0, 0.044639421, 0.616218009, 1.0],
    10: [0.0, 0.043364337, 0.623104207, 1.0],
    11: [0.0, 0.022005339, 0.609435051, 1.0],
    12: [0.0, 0.032731972, 0.617693682, 1.0],
    13: [0.0, 0.030038649, 0.612012113, 1.0],
    14: [0.0, 0.041520197, 0.701333729, 1.0],
    15: [0.0, 0.032172855, 0.604990027, 1.0],
    16: [0.0, 0.023141082, 0.64756127, 1.0],
    17: [0.0, 0.041409703, 0.721437303, 1.0],
    18: [0.0, 0.048672086, 0.611001788, 1.0],
    19: [0.0, 0.039570209, 0.630175298, 1.0]
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
    df_normalized.to_excel(writer, sheet_name='FeatureImportanceskNN3', index=False)


