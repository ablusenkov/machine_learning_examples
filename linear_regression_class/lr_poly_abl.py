#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 15:56:43 2020

@author: ablusenk
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

# followin needed for 3d plotting
import mpl_toolkits.mplot3d as Axes3D

df = pd.read_csv("/Users/ablusenk/GitHub/udemy/machine_learning_examples/linear_regression_class/data_poly.csv",
                 names = ["X", "Y"], sep = ",")
                       

# ========================================================================== #
# here manipulating with X to have quadratic function ax^2 + bx + c
# 
df.insert(0, "X0", 1)
df.insert(2, "X2", df.X**2)

#np.array(df[["X0","X","X2"]])

df_x = df[["X0","X","X2"]] 
df_y = df[["Y"]]



# ========================================================================== #
# plot original data only as well as polinom we made 
# 
plt.scatter(df_x.X, df_y)
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.view_init(200, 70)
ax.scatter(df.X, df.X2, df.Y)
ax.set_xlabel('X')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
plt.show()

'''
# rotation shoudl be done with ax.draw() function, 
# but somehow does not work here

for angle in range(0, 360):
    ax.view_init(30, angle)
    plt.draw()
    plt.pause(.001)

'''    

# ========================================================================== #
# calculate weight w in a similar manner as lr_2d_abl
# 

w = np.linalg.solve(np.dot(df_x.T, df_x), 
                    np.dot(df_x.T, df_y))
Y_hat = np.dot(df_x, w)

# ========================================================================== #
# plot all togather: scaterplot and fit line 
# line plot will work if sorted only. Same time we sort both X and Y, since 
# quadratic function monotonicly increasing

plt.scatter(df_x.X, df_y)
plt.plot(sorted(df_x.X), sorted(Y_hat))
plt.show()

# ========================================================================== #
# R-suqred
# 

d1 = df_y - Y_hat 
d2 = df_y - df_y.Y.mean() 

# convert DF to Series
d1 = d1.Y 
d2 = d2.Y

r_sq = 1 - d1.dot(d1)/d2.dot(d2)
print("R-squared is ", r_sq)