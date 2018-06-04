import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import math
from sklearn import metrics
from sklearn.model_selection import train_test_split


dataframe = pd.read_csv("census_county_counts.csv")
dataframe["PercentMen"] = dataframe["Men"] / dataframe["TotalPop"]
dataframe["PercentEmployed"] = dataframe["Employed"] / dataframe["TotalPop"]
dataframe["IncidentsPerCap"] = dataframe["incident_counts"] / dataframe["TotalPop"]

dataframe = dataframe.drop(dataframe.columns[0], axis = 1)
dataframe = dataframe.dropna()

features = (dataframe.drop(['CensusId', 'State', 'County', 'incident_counts', 'n_killed', 'n_injured'], axis = 1))
num_cols = len(features.columns)
features = features.values
targets = (dataframe["incident_counts"]).values

def baseline_model():
	model = Sequential()
	model.add(Dense(num_cols, input_dim=num_cols, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=0)
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=1)
estimator.fit(X_train_scaled, y_train)
y_pred = estimator.predict(X_test_scaled)
rmse = math.sqrt(metrics.mean_squared_error(y_test, y_pred))
print('RMSE for raw incident counts: %.3f' % rmse)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))