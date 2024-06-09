import numpy as np
import pandas as pd
import time
from scipy.io import loadmat
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression

#%% Initilize
mat_data = loadmat('All the Data.mat')

#print(mat_data)

array_data = mat_data['result_array']

print(array_data)

column_names = ['PilotLength', 'PilotSpacing', 'SymbolRate', 'PhaseNoise', 'BER', 'OBER']

# Convert the numpy array to a pandas DataFrame with the specified column names
df = pd.DataFrame(array_data, columns=column_names)

# Now you can access the columns by their names
X = df[['PilotLength', 'PilotSpacing', 'SymbolRate', 'PhaseNoise']]
#print("X:", X)
y = df['BER']
#print("y:\n", y)

#Normalize
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

selector = SelectKBest(f_regression, k=4)
X = selector.fit_transform(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)

# Number of estimators for AdaBoost

#trees: 5,10,15,20,25,30,35,40,45,50,75,100,250,500,750,1000
#5,15,25,30,40,50,75,100,250,500,750,1000,1250,1500,1750,2000
#estimators: 5,25,35,50,75,100,250,500,750,1000,1250,1750,2000,2500,3000
#5,25,35,50,75,100,250,500,750,1000,1250,1750,2000,2500,3000,5000,6000,7500,8500,10000
#learning rate: 0.01,0.05,0.06,0.07,0.08,0.09,0.1,0.11,.12,.13,.14,.15,.25,.5,1,2
#depth: 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

trees = [100]
l_rate = [0.01,0.05,0.06,0.07,0.08,0.09,0.1,0.11,.12,.13,.14,.15,.25,.5,1,2]

def transform_ber(ber):
    return np.minimum(ber, 100 - ber)

def Accuracy_score(orig, pred):
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
        if(pred[i] <= 1.2 * orig[i]) and (pred[i] >= 0.8 * orig[i]):
            count += 1
    a_20 = count / len(orig)
    return a_20

custom_Scoring = make_scorer(Accuracy_score, greater_is_better=True)

#%% Training
cv_folds = 15
num_depths = len(trees)
num_epi = len(l_rate)

rows = num_depths * num_epi

all_cv_scores = np.zeros((rows, cv_folds))
all_cv_scores2 = np.zeros((rows, cv_folds))
current_row = 0

# Loop over depths and trees to fill the array
for depth_idx, rate in enumerate(l_rate):
    for tree_idx, n_trees in enumerate(trees):
        abr = AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=8,random_state=17, criterion='squared_error'),
                                n_estimators=100, learning_rate=rate, random_state=17)
        
        cv_scores = RepeatedKFold(n_splits=5, n_repeats=3, random_state=8)
        Accuracy_Values = cross_val_score(abr, X_train, y_train, cv=cv_scores, scoring=custom_Scoring)
        
        print('Trial #:', current_row)
        print('\nAccuracy values for k-fold Cross Validation:\n', Accuracy_Values)
        print('\nFinal Average Accuracy of the model:', round(Accuracy_Values.mean(), 2))
        
        all_cv_scores[current_row, :] = Accuracy_Values
        
        # Custom scoring a_20 calculation
        custom_Scoring3 = make_scorer(Accuracy_score3, greater_is_better=True)

        # Running cross validation
        CV = RepeatedKFold(n_splits=5, n_repeats=3, random_state=8)
        Accuracy_Values3 = cross_val_score(abr, X_train, y_train, cv=CV, scoring=custom_Scoring3)
        
        print('\n"a_20 index" for 5-fold Cross Validation:\n', Accuracy_Values3)
        print('\nFinal Average Accuracy a_20 index of the model:', round(Accuracy_Values3.mean(), 4))
        
        all_cv_scores2[current_row, :] = Accuracy_Values3
        current_row += 1

#%% Export

# SMAPE
df_metrics = pd.DataFrame(all_cv_scores, columns=[f'Fold {i+1}' for i in range(cv_folds)])
# a_20
df_metricsA20 = pd.DataFrame(all_cv_scores2, columns=[f'Fold {i+1}' for i in range(cv_folds)])

file_path = "C:/Users/ryanj/Code/Research_THz/excel/Book1.xlsx"

with pd.ExcelWriter(file_path, mode='a', engine='openpyxl') as writer:
    df_metrics.to_excel(writer, sheet_name='ABR_SMAPE_lr', index_label='Depth')
    df_metricsA20.to_excel(writer, sheet_name='ABR_A20_lr', index_label='Depth')
