import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from scipy.optimize import differential_evolution
from scipy.io import loadmat
from sklearn.feature_selection import SelectKBest, f_regression

mat_data = loadmat('full_Results.mat')
array_data = mat_data['Results']

modulation_order = []
snr = []
pilot_length = []
pilot_spacing = []
symbol_rate = []
phase_noise = []
ber = []
cber = []

for result in array_data[0]:
    modulation_order.append(result['Modulation_order'][0][0])
    snr.append(result['SNR'][0][0])
    pilot_length.append(result['pilot_length'][0][0])
    pilot_spacing.append(result['symbols_between_pilot'][0][0])
    symbol_rate.append(result['symbol_rate'][0][0])
    phase_noise.append(result['phase_noise'][0][0])
    ber.append(result['BER'][0][0])
    cber.append(result['CBER'][0][0])

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
#X = df[['PhaseNoise', 'SymbolRate', 'SNR']]
#print("X:", X)
y = df['CBER']

abr = AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=15,random_state=17, criterion='squared_error'),
                                n_estimators=25, learning_rate=1, random_state=17)

abr.fit(X, y)

y_pred = abr.predict(X)

def Accuracy_score3(orig, pred):
    orig = np.array(orig)
    pred = np.array(pred)
    
    count = 0
    for i in range(len(orig)):
        if(pred[i] <= 1.2 * orig[i]) and (pred[i] >= 0.8 * orig[i]):
            count += 1
    a_20 = count / len(orig)
    return a_20

a_20 = Accuracy_score3(y, y_pred)
print(f'ABR A20: {a_20}')

df['Predicted BER'] = y_pred 

# plt.figure(figsize=(8, 6))
# plt.scatter(df['CBER'], df['SNR'], c='blue', alpha=0.6)
# plt.title('CBER vs SNR')
# plt.xlabel('CBER')
# plt.ylabel('SNR')
# plt.grid(True)
# plt.show()

# plt.figure(figsize=(8, 6))
# plt.scatter(df['CBER'], df['SymbolRate'], c='green', alpha=0.6)
# plt.title('CBER vs Symbol Rate')
# plt.xlabel('CBER')
# plt.ylabel('Symbol Rate')
# plt.grid(True)
# plt.show()

# plt.figure(figsize=(8, 6))
# plt.scatter(df['CBER'], df['PhaseNoise'], c='red', alpha=0.6)
# plt.title('CBER vs Phase Noise')
# plt.xlabel('CBER')
# plt.ylabel('Phase Noise')
# plt.grid(True)
# plt.show()

# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# sc = ax.scatter(df['CBER'], df['SymbolRate'], df['SNR'], c=df['SNR'], cmap='coolwarm')
# ax.set_xlabel('CBER')
# ax.set_ylabel('Symbol Rate')
# ax.set_zlabel('SNR')
# ax.set_title('3D Scatter Plot: CBER + Symbol Rate vs SNR')
# cbar = plt.colorbar(sc)
# cbar.set_label('SNR')
# plt.show()

# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# sc = ax.scatter(df['CBER'], df['PhaseNoise'], df['SNR'], c=df['SNR'], cmap='coolwarm')
# ax.set_xlabel('CBER')
# ax.set_ylabel('Phase Noise')
# ax.set_zlabel('SNR')
# ax.set_title('3D Scatter Plot: CBER + Phase Noise vs SNR')
# cbar = plt.colorbar(sc)
# cbar.set_label('SNR')
# plt.show()

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(df['PhaseNoise'], df['SymbolRate'], df['SNR'], c=df['CBER'], cmap='coolwarm')
ax.set_xlabel('Phase Noise')
ax.set_ylabel('Symbol Rate')
ax.set_zlabel('SNR')
# print("labels")
# plt.tight_layout()
# cbar = plt.colorbar(sc, pad=0.1, shrink=0.8)
# cbar.set_label('BER')
# print("color bar")
plt.show(block=True)
print("nonononS")
plt.close()
print("not showing")
plt.savefig('C:/Users/ryanj/Code/Research_THz/new_dataset/viz/3d_plot.png', bbox_inches='tight', dpi=300)



