from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
import math
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# "PercentMen", "Hispanic", "White", "Black", "Native", "Asian", "Pacific", "Income", "IncomePerCap", "Poverty",
#      "ChildPoverty", "Professional", "Service", "Office", "Construction", "Production", "Drive", "Carpool", "Transit",
#      "Walk", "OtherTransp", "WorkAtHome", "MeanCommute", "PercentEmployed", "PrivateWork", "PublicWork",
#      "SelfEmployed", "FamilyWork", "Unemployment"

# fix random seed for reproducibility
np.random.seed(7)

dataframe = pd.read_csv("census_county_counts.csv")

# If you need to create new features, do that here in dataframe
dataframe["PercentMen"] = dataframe["Men"] / dataframe["TotalPop"]
dataframe["PercentEmployed"] = dataframe["Employed"] / dataframe["TotalPop"]
dataframe["IncidentsPerCap"] = dataframe["incident_counts"] / dataframe["TotalPop"]

# Should contain [[feature1, feature2, feature3, ..., target]]
features = (dataframe[['PercentMen', 'Poverty', 'PercentEmployed']]).values
targets = (dataframe['IncidentsPerCap']).values

#Split into train and test
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=0)

# Can also try StandardScaler here
scaler = MinMaxScaler(feature_range=(-1,1))
scaler = scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = Sequential()
model.add(Dense(12, input_dim=3, activation='linear'))
# Add more layers here by adding more lines like the one commented out below. First parameter is the number of neurons
# in the layer. Can play around with activation functions too
model.add(Dense(8, activation='linear'))
model.add(Dense(1, activation='linear'))

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# Don't know what batch_size does...we should probably play around with it. Epochs is the number of iterations.
model.fit(X_train_scaled, y_train, epochs=20, batch_size=10)

predictions = model.predict(X_test_scaled)
rmse = math.sqrt(metrics.mean_squared_error(y_test, predictions))
print('RMSE for raw incident counts: %.3f' % rmse)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, predictions))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, predictions))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
