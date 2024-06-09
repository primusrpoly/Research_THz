import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

# Load the .mat file
mat_data = loadmat('All the Data.mat')
array_data = mat_data['result_array']
column_names = ['PilotLength', 'PilotSpacing', 'SymbolRate', 'PhaseNoise', 'BER', 'OBER']

# Convert the numpy array to a pandas DataFrame with the specified column names
df = pd.DataFrame(array_data, columns=column_names)

# One input variable and four target variables
X = df[['PhaseNoise']]
y = df[['PilotLength', 'PilotSpacing', 'SymbolRate', 'BER', 'OBER']]

# Scale the data
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.3, random_state=17)

# Build the neural network
def build_model(layers=2, neurons=64, activation='relu', optimizer='adam'):
    model = Sequential()
    model.add(Dense(neurons, input_dim=1, activation=activation))
    for _ in range(layers - 1):
        model.add(Dense(neurons, activation=activation))
    model.add(Dense(y.shape[1], activation='linear'))
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    return model

# Hyperparameters
layers = 3
neurons = 64
activation = 'relu'
optimizer = Adam(learning_rate=0.001)
epochs = 50
batch_size = 16

# Build and train the model
model = build_model(layers=layers, neurons=neurons, activation=activation, optimizer=optimizer)
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

# Make predictions
y_preds_scaled = model.predict(X_test)

# Inverse scale the predictions
y_preds = scaler_y.inverse_transform(y_preds_scaled)

# Plot training history
plt.figure(figsize=(12, 6))

# Plot training & validation loss values
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

# Plot training & validation MAE values
plt.subplot(1, 2, 2)
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])
plt.title('Model MAE')
plt.ylabel('Mean Absolute Error')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')

plt.tight_layout()
plt.show()

# Create DataFrame for predictions
df_metrics = pd.DataFrame(y_preds, columns=['PilotLength', 'PilotSpacing', 'SymbolRate', 'BER', 'OBER'])

# Add the input PhaseNoise column to the DataFrame
df_metrics.insert(0, 'Input PhaseNoise', scaler_X.inverse_transform(X_test))

file_path = "C:/Users/ryanj/Code/Research_THz/excel/Feature Prediction.xlsx"

# Export the DataFrame to an Excel file on a specific sheet
with pd.ExcelWriter(file_path, mode='a', engine='openpyxl') as writer:
    df_metrics.to_excel(writer, sheet_name='kNN_NN_Pred', index=False, startrow=0, startcol=0)
