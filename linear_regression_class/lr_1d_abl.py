#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 22:38:42 2020

@author: ablusenk
"""



import pandas as pd
import matplotlib.pyplot as plt

ds = pd.read_csv("/Users/ablusenk/GitHub/udemy/machine_learning_examples/linear_regression_class/data_1d.csv", names=["X","Y"])


"""
# alternative method of fetching data 
import numpy as np
X = []
Y = []

for i in open("/Users/ablusenk/GitHub/udemy/machine_learning_examples/linear_regression_class/data_1d.csv"):
    x,y = i.split(",")
    X.append(float(x))
    Y.append(float(y))

X = np.Series(X)
Y = np.Series(Y)
"""


# split in Series
X = ds.X    
Y = ds.Y

# plot for first time
#plt.scatter(X,Y)
#plt.show()


# calculus based on lecture #7 
denominator = X.dot(X) - X.mean() * X.sum()
a = (X.dot(Y) - Y.mean() * X.sum()) / denominator
b = (Y.mean() * X.dot(X) - X.mean() * X.dot(Y) ) / denominator

# predicted values used in linear 
Y_pred = a * X + b

#plot all together
plt.scatter(X,Y)
plt.plot(X, Y_pred)
plt.show()



# R_squared

d1 = Y - Y_pred
d2 = Y - Y.mean()
R_s = 1 - d1.dot(d1) / d2.dot(d2)
print("R-squared is: ", R_s)