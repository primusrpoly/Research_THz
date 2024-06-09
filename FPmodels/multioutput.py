import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

#%% Initialize
mat_data = loadmat('All the Data.mat')
array_data = mat_data['result_array']
column_names = ['PilotLength', 'PilotSpacing', 'SymbolRate', 'PhaseNoise', 'BER', 'OBER']

# Convert the numpy array to a pandas DataFrame with the specified column names
df = pd.DataFrame(array_data, columns=column_names)

# One input variable and four target variables
X = df[['PhaseNoise']]
y = df[['PilotLength', 'PilotSpacing', 'SymbolRate', 'BER']]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=17)

# Calculate the correlation matrix for the entire DataFrame
correlation_matrix = df.corr()

# Plot a heatmap of the correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

#%% Plots
# Plot the distribution of the target variables
plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1)
sns.histplot(df['PhaseNoise'], kde=True, color='green')
plt.title('Distribution of PhaseNoise')

plt.subplot(1, 3, 2)
sns.histplot(df['BER'], kde=True, color='green')
plt.title('Distribution of BER')

plt.subplot(1, 3, 3)
sns.histplot(df['OBER'], kde=True, color='green')
plt.title('Distribution of OBER')

plt.tight_layout()
plt.show()

# %%
