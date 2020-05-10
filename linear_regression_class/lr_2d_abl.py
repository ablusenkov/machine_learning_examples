#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 19:05:22 2020

@author: ablusenk
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import mpl_toolkits.mplot3d as Axes3D # for 3d scaterrplot

data_set = pd.read_csv("/Users/ablusenk/GitHub/udemy/machine_learning_examples/linear_regression_class/data_2d.csv",
                       names=["X1","X2","Y"], sep=",")   

# ========================================================================== #
# will need to inflate DS with column where set to 1, to respect to W0X0
# Can be done by calling DS on new column: 
#data_set['X0'] = 1
# or 
#data_set.assign(X0 = 1.0)
# but this will define location of the column 
data_set.insert(0, 'X0', 1.0)


# ========================================================================== #
# plot original data
fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')
ax.scatter(data_set.X1, data_set.X2, data_set.Y, c='r', marker='o')

ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')

plt.show()


# ========================================================================== #
# calculate weights
w = np.linalg.solve(np.dot(data_set[['X0', 'X1', 'X2']].T, data_set[['X0', 'X1', 'X2']]), np.dot(data_set[['X0', 'X1', 'X2']].T, data_set[['Y']]))
Y_hat = np.dot(data_set[['X0', 'X1', 'X2']], w)


# ========================================================================== #
# calculate r-square (y-y_hat)^2/ 
d1 = data_set[['Y']] - Y_hat
d2 = data_set[['Y']] - data_set[['Y']].mean()


# with math given above, I need to convert d1/d2 data type from DataFrame to Series
d1 = d1.Y 
d2 = d2.Y
r2 = 1 - d1.dot(d1) / d2.dot(d2)
print('R-squared is: ', r2)

