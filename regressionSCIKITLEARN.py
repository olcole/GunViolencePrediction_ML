import numpy as np
import pandas as pd
import math
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from baseline import baselineModel

original = pd.read_csv("census_county_counts.csv")

dataframe = original
dataframe["PercentMen"] = dataframe["Men"] / dataframe["TotalPop"]
dataframe["PercentEmployed"] = dataframe["Employed"] / dataframe["TotalPop"]
#dataframe["IncidentsPerCap"] = dataframe["incident_counts"] / dataframe["TotalPop"]

dataframe = dataframe.drop(dataframe.columns[0], axis = 1)
dataframe = dataframe.dropna()

features = (dataframe.drop(['CensusId', 'State', 'County', 'incident_counts', 'n_killed', 'n_injured'], axis = 1))
column_names = features.columns
features = features.values
targets = (dataframe["incident_counts"]).values

X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=0)

scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

regressor = LinearRegression()
#regressor = LogisticRegression()
regressor.fit(X_train_scaled, y_train)
#print(regressor.coef_)

#Comment this line out if you're doing Logistic Regression
print(pd.DataFrame(regressor.coef_, column_names, columns=['Coefficient']))

y_pred = regressor.predict(X_test_scaled)
i = 0
for x in y_pred:
    print("PREDICTION: " + str(x) + "     REAL: " + str(y_test[i]))
    i += 1

rmse = math.sqrt(metrics.mean_squared_error(y_test, y_pred))
print('RMSE for raw incident counts: %.3f' % rmse)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

#baselineModel(targets.mean(), y_test)