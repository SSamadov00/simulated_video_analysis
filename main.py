# Numpy deals with large arrays and linear algebra
# Library for data manipulation and analysis
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
from scipy import stats
from sklearn.metrics import r2_score
from sklearn import linear_model

# Create the variable to retract the csv file
# read_csv function of pandas reads the data in CSV format
# from path given and stores in the variable named train
# the data type of train is DataFrame

# 2 escape characters needed "\\"
df = pd.read_csv("C:\\Users\\samir\\Desktop\\CFSC-2023\\CFSC-2023\\cfsc2023\\examples\\video_behavior_analysis.csv")

# first we split our data into input and output
# y is the output and is stored in "Class" column of dataframe
# X contains the other columns and are features or input
# y = df.MotionEnergy
# df.drop(['MotionEnergy'], axis=1, inplace=True)
# X = df
#
# z = df.Behavior
# df.drop(['Behavior'], axis=1, inplace=True)

# print(np.sum(df['Behavior'] == "walk"))

# To print out every data
string_of_data = df.to_string()
motion_energy = df["MotionEnergy"]
behavior_motion = df["Behavior"]
area = df["Area"]
mean_motion = np.mean(motion_energy)
median_motion = np.median(motion_energy)
# mod_motion = stats.mode(motion_energy)
sd_motion = np.std(motion_energy)
var_motion = np.var(motion_energy)
percentile_motion = np.percentile(motion_energy, 75)

# Scatter plot motion energy vs area
slope, intercept, r, p, std_err = stats.linregress(motion_energy, area)


def myfunc(x):
    return slope * motion_energy + intercept


mymodel = list(map(myfunc, motion_energy))

plt.scatter(motion_energy, area)
plt.plot(motion_energy, mymodel)
plt.show()



# Test and train
train_x = motion_energy[:80]
train_y = area[:80]

test_x = motion_energy[80:]
test_y = area[80:]

mymodel = np.poly1d(np.polyfit(train_x, train_y, 4))

r2 = r2_score(test_y, mymodel(test_x))

print(r2)



# Correlation
slope, intercept, r, p, std_err = stats.linregress(motion_energy, area)


def myfunc(x):
    return slope * motion_energy + intercept


speed = myfunc(10)

print(speed)


# Convert the categorical to numeric value

# d = {'still': 0, 'groom': 1, 'paw_lick': 2, 'walk': 3}
# behavior_motion = behavior_motion.map(d)
#
# print(d)
#



# MotionEnergy x mean, y median
# x = df["MotionEnergy"].mean()
# c = df["MotionEnergy"].mode()

# # Plotting the data
# df.plot()
# df.plot(kind = 'scatter', x = 'MotionEnergy', y = 'Area')
# plt.show()


# ----------------------------------------------------------------------------------------------------------------------
