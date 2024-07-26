import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression

import numpy as np

def mape_exponential(expected, predicted):
    """
    Calculate the Mean Absolute Percentage Error (MAPE) using exponential values.
    
    Parameters:
    expected (float): The expected value.
    predicted (float): The predicted value.
    
    Returns:
    float: The MAPE using exponential values.
    """
    if expected == 0:
        return 0
    exp_expected = np.exp(expected)
    exp_predicted = np.exp(predicted)
    mape = np.abs(exp_expected - exp_predicted) / np.abs(exp_expected)
    return mape

# Example usage
expected_value = 49.3917465209961
predicted_value = 49.2525100708008
result = mape_exponential(expected_value, predicted_value)
print(f"MAPE using exponential values: {result}")




#%% Initialize
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
y = df['OBER']
#print("y:\n", y)

#Normalize
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17)


#%% Regressors

lnr = LinearRegression()
knn = KNeighborsRegressor(n_neighbors=1, weights='distance')
dtr = DecisionTreeRegressor(max_depth=13,random_state=17,criterion='squared_error')
rfr = RandomForestRegressor(n_estimators=5, random_state=17, max_depth=12,max_features=4)
gbr = GradientBoostingRegressor(max_depth=12, random_state=17, n_estimators=100, learning_rate=0.09)
abr = AdaBoostRegressor(estimator=dtr,random_state=17,n_estimators=35,learning_rate=0.15)

#%% Test and analysis

# Train the model
rfr.fit(X_train, y_train)

# Make predictions
y_pred = rfr.predict(X_test)

#RMSE Calculations
def rms_error(actual, predicted):
    mse = np.mean((actual - predicted) ** 2)
    rmse = np.sqrt(mse)
    return rmse

# def transform_ber(ber):
#     return np.minimum(ber, 100 - ber)

# def Accuracy_score(orig, pred):
#     orig = transform_ber(orig)
#     pred = transform_ber(pred)
    
#     numerator = np.abs(pred - orig)
#     denominator = (np.abs(orig) + np.abs(pred)) / 2
#     smape = np.mean(numerator / denominator)
#     return smape

# #a20 calculations
# def Accuracy_score3(orig,pred):
#     orig = np.array(orig)
#     pred = np.array(pred)
    
#     count = 0
#     for i in range(len(orig)):
#         if(pred[i] <= 1.2*orig[i]) and (pred[i] >= 0.8*orig[i]):
#             count = count +1
#     a_20 = count/len(orig)
#     return(a_20)

def Accuracy_score(orig, pred):
    exp_orig = np.exp(orig * 0.01)
    exp_pred = np.exp(pred * 0.01)
    
    mape = np.abs(exp_orig - exp_pred) / np.abs(exp_orig)
    return np.mean(mape) 

def Accuracy_score3(orig,pred):
    orig = np.array(orig)
    pred = np.array(pred)

    count = 0
    for i in range(len(orig)):
        if(pred[i] <= 1.4*orig[i]) and (pred[i] >= 0.6*orig[i]):
            count += 1
    a_20 = count/len(orig)
    return a_20

#Statistical calculations
#SHOULD BE TEST NOT TRAIN
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2=r2_score(y_test,y_pred)
mape=Accuracy_score(y_test, y_pred)
rmse=rms_error(y_test,y_pred)
a_20 = Accuracy_score3(y_test,y_pred)

#Print statements
print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mae)
print("R^2:",r2)
print("MAPE:",mape)
print("RSME:", rmse)
print("A_20:", a_20)

# print("Y_test\n",y_test)
# print("The number of values in y_test",len(y_test))
# print("Y_pred",y_pred)

#%% Export to excel Actual vs expected

X_test_df = pd.DataFrame(scaler.inverse_transform(X_test), columns=['PilotLength', 'PilotSpacing', 'SymbolRate', 'PhaseNoise'])

# Expected vs actual dataframe
results_df = pd.DataFrame({
    'Expected/Test': y_test*.01,
    'Predicted': y_pred*.01
})

file_path = "C:/Users/ryanj/Code/Research_THz/excel/CBER.xlsx"

# # Export to Excel
with pd.ExcelWriter(file_path, mode='a', engine='openpyxl') as writer:
    #X_test_df.to_excel(writer, sheet_name='RFR12_TvP_CBER3', index=False, startrow=0, startcol=0)
    results_df.to_excel(writer, sheet_name='RFR_TvP_CBER4', index=False, startrow=0, startcol=0)
# %%
