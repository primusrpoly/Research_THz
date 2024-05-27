import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression

mat_data = loadmat('All the Data.mat')

#print(mat_data)

array_data = mat_data['result_array']

#print(array_data)

column_names = ['PilotLength', 'PilotSpacing', 'SymbolRate', 'PhaseNoise', 'BER', 'OBER']

# Convert the numpy array to a pandas DataFrame with the specified column names
df = pd.DataFrame(array_data, columns=column_names)

# Now you can access the columns by their names
X = df[['PilotLength', 'PilotSpacing', 'SymbolRate', 'PhaseNoise']]
#print("X:", X)
y = df['BER']
#print("y:\n", y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17)

trial = 0
a_20values = []

#Scoring
def calculate_smape(orig,pred):
    """
    Calculates the Symmetric Mean Absolute Percentage Error (SMAPE).

    Args:
        actual_values (numpy array or list): Actual target values.
        predicted_values (numpy array or list): Predicted target values.

    Returns:
        float: SMAPE value (expressed as a percentage).
    """
    orig = np.array(orig)
    predicted_values = np.array(predicted_values)

    numerator = np.abs(actual_values - predicted_values)
    denominator = (np.abs(actual_values) + np.abs(predicted_values)) / 2

    smape = np.mean(numerator / denominator) * 100
    return smape

# Example usage:
actual_values = [10, 20, 30, 0, 15]
predicted_values = [12, 18, 28, 5, 14]

smape_value = calculate_smape(actual_values, predicted_values)
print(f"SMAPE: {smape_value:.2f}%")



#A_20
def Accuracy_score3(orig,pred):
    orig = 10 ** np.array(orig)
    pred = 10 ** np.array(pred)
    
    count = 0
    for i in range(len(orig)):
        if(pred[i] <= 1.2*orig[i]) and (pred[i] >= 0.8*orig[i]):
            count = count +1
    a_20 = count/len(orig)
    return(a_20)

# Train the model

for neighbor in range(1,16):
    knn_model = KNeighborsRegressor(n_neighbors=neighbor, weights='distance')  # You can adjust the number of neighbors (k) as needed
    knn_model.fit(X_train, y_train)
    cv_scores = RepeatedKFold(n_splits=5, n_repeats = 3, random_state = 8)
    trial += 1

    print('Trial #:',trial)
    
    #custom scoring a_20 calulation
    custom_Scoring3 = make_scorer(Accuracy_score3,greater_is_better=True)

    #Running cross validation
    CV = RepeatedKFold(n_splits = 5, n_repeats=3, random_state = 8)
    Accuracy_Values3 = cross_val_score(knn_model,X_train ,y_train,\
                                       cv=CV,scoring=custom_Scoring3)
    a_20values.append(Accuracy_Values3)

# Evaluate the model
    y_pred_knn = knn_model.predict(X_test)
    mae_knn = mean_absolute_percentage_error(y_test, y_pred_knn)
    mse_knn = mean_squared_error(y_test, y_pred_knn)
    print("kNN Mean Absolute Error:", mae_knn)
    print("kNN Mean Squared Error:", mse_knn)
    print('\n"a_20 index" for 5-fold Cross Validation:\n', Accuracy_Values3)
    print('\nFinal Average Accuracy a_20 index of the model:', round(Accuracy_Values3.mean(),4))




