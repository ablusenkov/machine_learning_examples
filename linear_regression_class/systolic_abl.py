#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 23:47:04 2020

@author: ablusenk
"""
# ========================================================================== #
# https://deeplearningcourses.com/c/data-science-linear-regression-in-python
# need to sudo pip install xlrd to use pd.read_excel
# data is from:
# http://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/mlr/frames/mlr02.html

# The data (X1, X2, X3) are for each patient.
# X1 = systolic blood pressure
# X2 = age in years
# X3 = weight in pounds


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
# followin needed for 3d plotting
import mpl_toolkits.mplot3d as Axes3D

df = pd.read_excel("http://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/mlr/excel/mlr02.xls")

# ========================================================================== #
# plot as 3d 
# 

plt.title("Pressure vs Age")
plt.scatter(df.X1, df.X2)
plt.show()
plt.title("Pressure vs Weight")
plt.scatter(df.X1, df.X3)
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df.X1, df.X2, df.X3, c='r', marker='o')

ax.view_init(10,120)
ax.set_xlabel("Systolic BP")
ax.set_ylabel("Age")
ax.set_zorder("Weight")

plt.show()

# ========================================================================== #
# iltimate goal is to predict pressure based on Age+Weight 
# for that we need to take X1 as Y and X2,X3 as X. Based on that find w 
# to solve Y = Xw we'll used no.linalg.solve() function
# same time df should be inflated with X0 = 1, since y = x0*1 + x1w1 + x2w2
# 

df.insert(0, "X0", 1)
w = np.linalg.solve(np.dot(df[["X0","X2", "X3"]].T, df[["X0","X2", "X3"]]), 
                    np.dot(df[["X0","X2", "X3"]].T, df[["X1"]]))

print(w)
# 
# now we look for Y_hat (predicted with w value)
# 
Y_hat = np.dot(df[["X0","X2", "X3"]], w)

               
# ========================================================================== #
# R-squared
# 

d1 = df[["X1"]] - Y_hat
d2 = df[["X1"]] - df[["X1"]].mean()

d1 = d1.X1 
d2 = d2.X1

r_sq = 1 - d1.dot(d1)/d2.dot(d2)
print("R-squared is", r_sq)




# ========================================================================== #
# prediction based on input
# 

X2_read = [1]

X2_read.append(float(input("What is your age? - ")))
X2_read.append(float(input("What is your weigth /kg/? - ")) * 2.205)

X2_read = np.array(X2_read)

Y_hat_pred = np.dot(X2_read, w)

print("In your age and with your weight your pressure should be ", Y_hat_pred[0])

# ========================================================================== #
# EXTRA drqawing to `best` fit line and prediction compared to data we have
# 

plt.title("Pressure vs Age + Predicted")
plt.scatter(Y_hat_pred, X2_read[1], c='b', marker='X')
plt.scatter(df.X1, df.X2)
plt.plot(sorted(Y_hat), sorted(df.X2), c='r')
plt.show()

plt.title("Pressure vs Weight + Predicted")
plt.scatter(Y_hat_pred, X2_read[2], c='b', marker='X')
plt.scatter(df.X1, df.X3)
plt.plot(sorted(Y_hat), sorted(df.X3), c='r')
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(Y_hat_pred, X2_read[1], X2_read[2], c='b', marker='X')
ax.scatter(df.X1, df.X2, df.X3, c='r', marker='o')
ax.scatter(Y_hat, df.X2, df.X3, c='g', marker='v')


ax.view_init(10,120)
ax.set_xlabel("Systolic BP")
ax.set_ylabel("Age")
ax.set_zorder("Weight")
plt.show()


