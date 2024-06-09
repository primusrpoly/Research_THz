import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold
from sklearn.metrics import make_scorer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression

#%% Initialize
mat_data = loadmat('All the Data.mat')
array_data = mat_data['result_array']
column_names = ['PilotLength', 'PilotSpacing', 'SymbolRate', 'PhaseNoise', 'BER', 'OBER']

# Convert the numpy array to a pandas DataFrame with the specified column names
df = pd.DataFrame(array_data, columns=column_names)

# One input variable and four target variables
X = df[['PhaseNoise']]
y = df[['PilotLength', 'PilotSpacing', 'SymbolRate', 'BER']]

# Normalize
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)

smape_values = []
a_20values = []

# Log base 10
def transform_ber(ber):
    return np.minimum(ber, 100 - ber)


def smape_score_calculator(y1_true, y1_pred, y2_true, y2_pred, y3_true, y3_pred, y4_true, y4_pred):
    def calculate_smape(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred)))

    smape_y1 = calculate_smape(y1_true, y1_pred)
    smape_y2 = calculate_smape(y2_true, y2_pred)
    smape_y3 = calculate_smape(y3_true, y3_pred)
    smape_y4 = calculate_smape(y4_true, y4_pred)

    return {
        smape_y1,
        smape_y2,
        smape_y3,
        smape_y4
    }

# Example usage
y1_true = [10, 20, 30, 40, 50]
y1_pred = [12, 18, 29, 41, 48]
y2_true = [15, 25, 35, 45, 55]
y2_pred = [14, 27, 33, 44, 56]
y3_true = [20, 30, 40, 50, 60]
y3_pred = [19, 31, 39, 51, 59]
y4_true = [25, 35, 45, 55, 65]
y4_pred = [23, 34, 46, 54, 67]

smape_scores = smape_score_calculator(y1_true, y1_pred, y2_true, y2_pred, y3_true, y3_pred, y4_true, y4_pred)
print(smape_scores)


def Accuracy_score(orig, ber):
    if orig.name == 'BER':  # Apply transformation only for BER
        orig = transform_ber(orig)
        pred = transform_ber(pred)

    numerator = np.abs(pred - orig)
    denominator = (np.abs(orig) + np.abs(pred)) / 2
    smape = np.mean(numerator / denominator)
    return smape

def Accuracy_score3(orig, pred):
    orig = 10 ** np.array(orig)
    pred = 10 ** np.array(pred)
    
    count = 0
    for i in range(len(orig)):
        if (pred[i] <= 1.2 * orig[i]) and (pred[i] >= 0.8 * orig[i]):
            count += 1
    a_20 = count / len(orig)
    return a_20

custom_Scoring = make_scorer(Accuracy_score, greater_is_better=True)
custom_Scoring3 = make_scorer(Accuracy_score3, greater_is_better=True)

#%% Test
for neighbor in range(1, 16):
    knn = KNeighborsRegressor(n_neighbors=neighbor, weights='distance')
    cv_scores = RepeatedKFold(n_splits=5, n_repeats=3, random_state=8)
    Accuracy_Values = cross_val_score(knn, X_train, y_train, cv=cv_scores, scoring=custom_Scoring)
        
    smape_values.append(Accuracy_Values)
    print('Trial #:', neighbor)
    print('\nAccuracy values for k-fold Cross Validation:\n', Accuracy_Values)
    print('\nFinal Average Accuracy of the model:', round(Accuracy_Values.mean(), 2))
    
    Accuracy_Values3 = cross_val_score(knn, X_train, y_train, cv=cv_scores, scoring=custom_Scoring3)
    a_20values.append(Accuracy_Values3)
    print('\n"a_20 index" for 5-fold Cross Validation:\n', Accuracy_Values3)
    print('\nFinal Average Accuracy a_20 index of the model:', round(Accuracy_Values3.mean(), 4))

#%% Exporting to Excel
df_metrics = pd.DataFrame(smape_values, index=range(1, 16), columns=range(1, 16))   
df_metrics_a20 = pd.DataFrame(a_20values, index=range(1, 16), columns=range(1, 16))   

file_path = "C:/Users/ryanj/Code/Research_THz/excel/Feature Prediction.xlsx"

# Export the DataFrame to an Excel file on a specific sheet
with pd.ExcelWriter(file_path, mode='a', engine='openpyxl') as writer:
    df_metrics.to_excel(writer, sheet_name='KNN_SMAPE_17', index=False, startrow=0, startcol=0)
    df_metrics_a20.to_excel(writer, sheet_name='KNN_A20_17', index=False, startrow=0, startcol=0)