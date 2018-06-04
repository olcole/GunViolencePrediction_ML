import matplotlib.pyplot as plt
import pandas
import numpy


dataframe = pandas.read_csv("census_county_counts.csv")

dataframe["PercentMen"] = dataframe["Men"] / dataframe["TotalPop"]
dataframe["PercentEmployed"] = dataframe["Employed"] / dataframe["TotalPop"]
dataframe["IncidentsPerCap"] = dataframe["incident_counts"] / dataframe["TotalPop"]

correlations = dataframe.corr()
print(correlations)

key = []
i = 0
for label in data.columns.values:
    key.append(i)
    print(str(i) + ": " + data.columns.values[i])
    i += 1

# plot correlation matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = numpy.arange(0,len(data.columns),1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(key)
ax.set_yticklabels(key)
plt.show()