import math
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def baselineModel(mean, test_data):
    #Y = dataframe[target_col]
    #X_train, X_test, y_train, y_test = train_test_split(Y, Y, test_size=0.2, random_state=0)
    #mean = Y.mean()
    means = list()

    for i in range((len(test_data))):
        means.append(mean)

    i = 0
    for x in test_data:
        print("PREDICTION: " + str(means[i]) + "     REAL: " + str(x))
        i += 1

    rmse = math.sqrt(mean_squared_error(test_data, means))
    print('RMSE for raw incident counts: %.3f' % rmse)
    return rmse