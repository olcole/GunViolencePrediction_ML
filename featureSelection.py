import numpy as np
import pandas as pd
import math
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
from baseline import baselineModel
import matplotlib.pyplot as plt
import operator
import random

import itertools

#change this to change change the dataset we are using
original = pd.read_csv("census_county_counts.csv")
#original = pd.read_csv("census_data_filtered_injuries_only.csv")


#to remove 0 incident coutns metrics
dataframe = original
#dataframe = dataframe[dataframe["incident_counts"] != 0]
dataframe = dataframe[dataframe["State"] != "Puerto Rico"]

#add new columns
dataframe["PercentMen"] = dataframe["Men"] / dataframe["TotalPop"]
dataframe["PercentEmployed"] = dataframe["Employed"] / dataframe["TotalPop"]

#normalizing based on population
dataframe["incident_counts_per_cap"] = dataframe["incident_counts"] / dataframe["TotalPop"] * 1000
dataframe['log_incident_counts_per_cap']  = np.log((1 + dataframe['incident_counts_per_cap']))

#remove rows with missing data
dataframe = dataframe.drop(dataframe.columns[0], axis = 1)
dataframe = dataframe.dropna()

#features = (dataframe.drop(['CensusId', 'State', 'County', 'incident_counts', 'n_killed', 'n_injured'], axis = 1))

#columns that we will not use as features
columns_to_remove = ["TotalPop", 'IncomePerCapErr', 'CensusId', 'log_incident_counts_per_cap', 'State', 'County', 'incident_counts', 'incident_counts_per_cap', 'n_killed', 'n_injured', 'Men', 'Women', 'Citizen', 'Income', 'IncomeErr', 'Employed']
#columns_to_remove = ['log_incident_counts_per_cap', 'IncomePerCapErr', 'State', 'County', 'CensusTract', 'incident_counts', 'incident_counts_per_cap', 'n_killed', 'n_injured', 'Men', 'Women', 'Citizen', 'Income', 'IncomeErr', 'Employed']

#get list of features that we are using
used_columns = np.setdiff1d(dataframe.columns, columns_to_remove)

features = (dataframe.drop(columns_to_remove, axis = 1))
#features = (dataframe.drop(['n_killed', 'n_injured'], axis = 1))
#used_columns = np.setdiff1d(dataframe.columns, ['log_incident_counts','incident_counts', 'n_killed', 'n_injured'])

def regress(dataframe, target_column, feature_columns, random_seed, kFold = True, normalize = False, logBaseline = True, generateAccuracyBreakdown = False):
	
	#convert target to log or create feature of just ones if we don't have any feature columns
	if (len(feature_columns) == 0):
		features = np.ones((len((dataframe[target_column]).values), 1))
	else:
		features = dataframe[np.array(feature_columns)]
		features = features.values

	#get our target and total population so that we can generate per capita and raw coutns
	targets = (dataframe[target_column]).values
	totalPop = (dataframe["TotalPop"]).values

	#error measurements
	train_rmse = 0
	test_rmse = 0
	train_baseline_rmse = 0
	baseline_rmse = 0

	#used to generate accuracy breakdown by quartiles
	if (generateAccuracyBreakdown):
		log_quartiles = np.array(dataframe['log_incident_counts_per_cap'].quantile([0, .25, .5, .75, 1]))
		per_cap_quartiles = np.array(dataframe['incident_counts_per_cap'].quantile([0, .25, .5, .75, 1]))

		log_quartiles[0] = -.00001
		log_quartiles[4] += .00001
		per_cap_quartiles[0] = -.00001
		per_cap_quartiles[4] += .00001

		quartile_total =[0,0,0,0]
		quartile_correct =[0,0,0,0]
		quartile_accuracy =[0,0,0,0]


	#k-fold
	if (kFold):
		num_folds = 3

		kf = KFold(n_splits=num_folds, shuffle=True, random_state=random_seed)
		
		#perform kFold validation
		for train_index, test_index in kf.split(features):

			#get train and test data
			X_train, X_test = features[train_index], features[test_index]
			y_train, y_test = targets[train_index], targets[test_index]
			population_train, population_test = totalPop[train_index], totalPop[test_index]


			y_train.shape= (len(y_train), 1)
			y_test.shape = (len(y_test), 1)

			#normalize output feature or just our original target values
			if (normalize):
				y_mean  = np.mean(y_train)
				y_std = np.std(y_train)
				y_train_scaled = (y_train - y_mean)/y_std
				y_test_scaled = (y_test - y_mean)/y_std
			else:
				y_train_scaled = y_train
				y_test_scaled = y_test


			#scale x between 0 and 1
			scaler = MinMaxScaler().fit(X_train)
			X_train_scaled = scaler.transform(X_train)
			X_test_scaled = scaler.transform(X_test)

			#scaling y
			#y_train.shape= (len(y_train), 1)
			#y_test.shape = (len(y_test), 1)

			#scaler_y = MinMaxScaler().fit(y_train)
			#y_train_scaled = scaler_y.transform(y_train)
			#y_test_scaled = scaler_y.transform(y_test)

			#y_train_scaled.shape= (len(y_train_scaled))
			#y_test_scaled.shape = (len(y_test_scaled))

			regressor = LinearRegression()
			regressor.fit(X_train_scaled, y_train_scaled)
			#print(regressor.coef_)

			#generate prediction
			y_pred = regressor.predict(X_test_scaled)
			y_train_pred = regressor.predict(X_train_scaled)

			y_test_scaled.shape= (len(y_test_scaled),)
			y_train_scaled.shape = (len(y_train_scaled),)
			y_pred.shape= (len(y_test_scaled),)
			y_train_pred.shape = (len(y_train_scaled),)

			#convert back to raw incident counts
			y_test_raw = ((np.exp(y_test_scaled) - 1) / 1000) * population_test
			y_train_raw = ((np.exp(y_train_scaled) - 1) / 1000) * population_train

			y_test_pred_raw = ((np.exp(y_pred) - 1) / 1000) * population_test
			y_train_pred_raw = ((np.exp(y_train_pred) - 1) / 1000) * population_train

			#for generating our different error measure (log vs raw RMSE)
			if (logBaseline):
				train_baseline_rmse += baselineModel(np.mean(y_train_scaled), y_train_scaled)
				baseline_rmse += baselineModel(np.mean(y_test_scaled), y_test_scaled)

				train_rmse += math.sqrt(metrics.mean_squared_error(y_train_scaled, y_train_pred))
				test_rmse += math.sqrt(metrics.mean_squared_error(y_test_scaled, y_pred))

				#convert actual and predicted output into categories by quartile values computer earlier, generate accuracies
				if (generateAccuracyBreakdown):
					log_categories_actual = pd.cut(y_test_scaled, per_cap_quartiles, labels = [0, 1, 2, 3])
					log_categories_pred = pd.cut(y_pred, per_cap_quartiles, labels = [0, 1, 2, 3])

					for i in range(len(log_categories_actual)):
						quartile_total[log_categories_actual[i]] += 1

						if (log_categories_actual[i] == log_categories_pred[i]):
							quartile_correct[log_categories_actual[i]] += 1

			#raw values
			else:
				train_baseline_rmse += baselineModel(np.mean(y_train_raw), y_train_raw)
				baseline_rmse += baselineModel(np.mean(y_test_raw), y_test_raw)

				train_rmse += math.sqrt(metrics.mean_squared_error(y_train_raw, y_train_pred_raw))
				test_rmse += math.sqrt(metrics.mean_squared_error(y_test_raw, y_test_pred_raw))

		#print("DONE")

		#plot quartile accuracy values
		if (generateAccuracyBreakdown):
			for i in range(len(quartile_correct)):
				quartile_accuracy[i] = quartile_correct[i]/quartile_total[i]

			plt.figure()
			plt.title("Accuracy By Quartile Value")
			x = ["0-25%", "25-50%", "50-75%", "75-100%"]
			plt.bar(x,quartile_accuracy)
			plt.xlabel("Incident Counts Per Capita Percentile Range")
			plt.ylabel("Predicted Quartile Accuracy")
			plt.show()

		#return baseline and model error metrics along with weights for the model
		return regressor.coef_, baseline_rmse/num_folds, train_baseline_rmse/num_folds, test_rmse/num_folds, train_rmse/num_folds


	#non kFold regression
	else:
		X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state =random_seed)

		scaler = MinMaxScaler().fit(X_train)
		X_train_scaled = scaler.transform(X_train)
		X_test_scaled = scaler.transform(X_test)

		if (normalize):
			y_mean  = np.mean(y_train)
			y_std = np.std(y_train)
			y_train_scaled = (y_train-y_mean)/y_std
			y_test_scaled = (y_test-y_mean)/y_std
		else:
			y_train_scaled = y_train
			y_test_scaled = y_test

		#scaling y
		#y_train.shape= (len(y_train), 1)
		#y_test.shape = (len(y_test), 1)

		#scaler_y = MinMaxScaler().fit(y_train)
		#y_train_scaled = scaler_y.transform(y_train)
		#y_test_scaled = scaler_y.transform(y_test)

		#y_train_scaled.shape= (len(y_train_scaled))
		#y_test_scaled.shape = (len(y_test_scaled))

		regressor = LinearRegression()		
		regressor.fit(X_train_scaled, y_train_scaled)

		y_pred = regressor.predict(X_test_scaled)
		y_train_pred = regressor.predict(X_train_scaled)

		train_baseline_rmse = baselineModel(np.mean(y_train_scaled), y_train_scaled)
		baseline_rmse = baselineModel(np.mean(y_test_scaled), y_test_scaled)
		train_rmse = math.sqrt(metrics.mean_squared_error(y_train_scaled, y_train_pred))
		test_rmse = math.sqrt(metrics.mean_squared_error(y_test_scaled, y_pred))
		
		return regressor.coef_, baseline_rmse, train_baseline_rmse, test_rmse, train_rmse


def get_best_attributes(dataframe, used_columns, target_column, combineFeatures = False):
	all_columns = used_columns

	best_total_accuracy = float('inf')
	columns_added = []

	max_size = 16
	best_subsets = []

	baseline_test_hist = []
	baseline_train_hist = []
	test_hist = []
	train_hist = []

	test_hist_raw = []
	train_hist_raw = []

	seed = random.randint(0,100)

	#use combinations of features
	if (combineFeatures):
		i = len(all_columns)
		for column1 in used_columns:
			for column2 in used_columns:
				columns = [column1, column2]
				columns = sorted(columns)
				new_column = columns[0] + '-' + columns[1]
				
				if (not new_column in all_columns):
					dataframe[new_column] = dataframe[column1]*dataframe[column1]
					all_columns = np.append(all_columns,np.array([new_column]))

	#iterate from 1 to all possible features
	for i in range(len(all_columns)):

		best_accuracy = float('inf')

		#iterate over all columns that haven't been added to the subset
		for next_column in np.setdiff1d(all_columns, columns_added):

			columns_subset = np.append(np.array(columns_added), np.array([next_column]))
			
			weights, baseline_rmse, baseline_train, test_rmse, train_rmse = regress(dataframe, target_column, columns_subset, seed)
			weights2, baseline_rmse_raw, baseline_train_raw, test_rmse_raw, train_rmse_raw = regress(dataframe, target_column, columns_subset, seed, logBaseline = False)

			accuracy = train_rmse/baseline_train

			#save the subset that has the best log RMSE accuracy
			if (accuracy < best_accuracy):
				best_train = train_rmse/baseline_train
				best_test  = test_rmse/baseline_rmse
				best_train_raw = train_rmse_raw/baseline_train_raw
				best_test_raw = test_rmse_raw/baseline_rmse_raw
				best_column = next_column
				best_accuracy = accuracy

		columns_added.append(best_column)

		print("\nAdding column:", best_column)
		print("New Percentage:", best_accuracy)
		#columns_subset.append(best_column)
		print("New list:", columns_added)
		print("Train: ", best_train)
		print("Test: ", best_test)
		#print("\n")

		#weights, accuracy = regress(dataframe, "incident_counts_per_cap", columns_subset)
		#print(pd.DataFrame(weights.T, columns_added, columns=['Coefficient']))

		#save to history so that we generate accuracy plots
		best_list = columns_added.copy()
		best_total_accuracy = best_accuracy
		baseline_test_hist.append(baseline_rmse)
		baseline_train_hist.append(baseline_train)
		test_hist.append(1-best_test)
		train_hist.append(1-best_train)
		test_hist_raw.append(1-best_test_raw)
		train_hist_raw.append(1-best_train_raw)

	return best_list, baseline_test_hist, train_hist, test_hist, train_hist_raw, test_hist_raw


#perform feature selection
best_attributes, baseline_hist, training_hist, testing_hist, training_hist_raw, testing_hist_raw = get_best_attributes(dataframe, used_columns, "log_incident_counts_per_cap")


seed = random.randint(0,100)

#generate accuracy breakdowns for our best set of features
weights, baseline_train_rmse, baseline_test_rmse, train_rmse, test_rmse = regress(dataframe, "log_incident_counts_per_cap", best_attributes[:12], seed, generateAccuracyBreakdown = True)


#generate RMSE compared to baseline plots against number of features 
plt.figure(1)
num_attributes = np.arange(1,len(best_attributes) + 1)

#base, = plt.plot(num_attributes, baseline_hist, 'b')
valid, = plt.plot(num_attributes, testing_hist, 'r--')
train, = plt.plot(num_attributes, training_hist, 'g--')
plt.title('Log(incident per capita) Accuracy ')
plt.legend(handles=[valid, train], labels=['Validation RMSE', 'Training RMSE'], loc='upper left')
plt.xlabel('Number of features')
plt.ylabel('1 - (Model RMSE)/(ZeroR RMSE)')
#print(pd.DataFrame(weights.T, best_attributes, columns=['Coefficient']))

plt.figure(2)
num_attributes = np.arange(1,len(best_attributes) + 1)

#base, = plt.plot(num_attributes, baseline_hist, 'b')
valid, = plt.plot(num_attributes, testing_hist_raw, 'r--')
train, = plt.plot(num_attributes, training_hist_raw, 'g--')
plt.title('Raw incident accuracy')
plt.legend(handles=[valid, train], labels=['Validation RMSE', 'Training RMSE'], loc='upper left')
plt.xlabel('Number of features')
plt.ylabel('1 - (Model RMSE)/(ZeroR RMSE)')
plt.show()
print(pd.DataFrame(weights.T, best_attributes, columns=['Coefficient']))

# 	