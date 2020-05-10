#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Mon May  4 12:41:53 2020

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


# ========================================================================== #
# Strategy: 
# Apply Gradient Descent on Training data to find corresponding `weights`
# Data should be standardized before applying GD (by mean of scikit-learn)
# W_0 value (W0*X0) should be included after normalization
# As sc.fit() calculates and saves centerline and SD for normalization, it can 
#    be used for `input` data transformation. This done in INPUT section
# Vizualization brings numbers to original scales with sc.inverse_transform() 
# Note: Standardization is not Normalization. First transforms data to have 
#       mean zero and SD one (z-score), while second is bringing data to range 
#       between zero and one. 


import numpy as np 
import pandas as pd 

# ========================================================================== #
# Plotting infra with respect to 3d graph (due to multivariative linear reg  #
#

import matplotlib.pyplot as plt 
# library needed for 3d plotting
import mpl_toolkits.mplot3d as Axes3D
# Following to ebalen inline graphs (default)
#%matplotlib inline
# Following to enable graphs in a separate window
#%matplotlib qt


# ========================================================================== #
# uploding data and converting them in to np.array
#

#df = pd.read_excel("http://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/mlr/excel/mlr02.xls")
df = pd.read_excel("/Users/ablusenk/GitHub/udemy/machine_learning_examples/linear_regression_class/mlr02.xls")
X = np.array(df[["X2", "X3"]])
Y = np.array(df[["X1"]])

# ========================================================================== #
# Origianl X standardized with fit_transform() 
# afterwards array extended with W0X0, i.e. vector of ones or Bias
#

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
X = np.insert(X, 0, 1, axis=1)

# ========================================================================== #
# so far not clear how to take initial `weight` for GD, though some 
# recomendations are pointing on random from Gaussian (bell curve)
# centered at zero and variance equal 1/D, where D is a number of experements 
# Note: variance is SD**2
#
w = 1/len(df) * np.random.randn(3,1)
print("Initial weight is:", w.T[0])


# ========================================================================== #
# Defining MSE/Cost and GD function
#

def cost(X,Y,weight): 
    """
    This will calculate MSE or cost function for given weight, X and Y: 
        Y = w0*X0 + w1*X1 + .. + wj*Xj
    
    MSE function defined as: 
        MSE = 1/D * sum(Y - Y_predicted)**2 
    input:  X,Y(x) data sets as well as vector/array of weights 
    output: Returns MSE        
    """
    Y_hat =  np.dot(X, weight)
    MSE = np.sum(np.square(Y_hat - Y)) / (2*len(Y))
    return(MSE)

def gradient_descent(X,Y,weight,learn_rate = 0.003,step=100):
    """ 
    This will calculate weight with gradient descent, by doing: 
    w = w - learn_rate * X.T * (Y - Y_hat)    
    
    Calculation (descent) will take number of steps as defined by `step`
    
    input:  X,Y(x) data sets, weights, learning rate and number of steps
    output: Returns Weights, Cost change History, Weight change History      
    """
    
    m = len(Y)   
    cost_history   = np.zeros(step)
    weight_histroy = np.zeros((step,3))  
 
    for i in range(step): 
        
        Y_hat  = np.dot(X,weight)
        gradient = np.dot(X.T, (Y_hat - Y))
        weight = weight - (1/m) * learn_rate * gradient
        weight_histroy[i,:] = weight.T
        mse = cost(X,Y,weight)
        cost_history[i] = mse

    return(weight,cost_history, weight_histroy)
 

lr=0.3
steps = 100
w_n, cost, w_history = gradient_descent(X,Y,w, lr, steps)


Y_hat = np.dot(X,w_n)
plt.scatter(range(steps), cost)
plt.show()

'''
# ========================================================================== #
# Following to show how we converge 
# 

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df.X1, df.X2, df.X3, c='r', marker='o')
ax.view_init(10,120)
ax.set_xlabel("Systolic BP")
ax.set_ylabel("Age")
ax.set_zorder("Weight")

for i in range(steps): 
    ax.plot(sorted(np.dot(X,w_history[i])), sorted(df.X2), sorted(df.X3), c='g', marker='v')
    plt.draw()

'''

print("Final W_0:         {:0.3f}\nFinal W_1:         {:0.3f}\nFinal W_2:         {:0.3f}"
      .format(w_n[0][0],w_n[1][0],w_n[2][0]))
print("\nFinal_MSE:         {:0.3f}".format(cost[-1]))



# ========================================================================== #
# INPUT section
# Here call for input data followed by plotting 
# NOTE: data trasnformation is needed as well as inverse_transformation before 
#       plotting
#

X2_read = np.zeros(2)
X2_read[0] = (float(input("What is your age? - ")))
X2_read[1] = (float(input("What is your weigth /kg/? - ")) * 2.205)


X2_read = sc.transform(X2_read.reshape(1,-1))
back_transofrm = X2_read

X2_read = np.insert(X2_read, 0, 1, axis = 1)
Y_hat_pred = np.dot(X2_read, w_n)
print("In your age and with your weight your pressure should be ", Y_hat_pred[0])




X2_read_inversed = sc.inverse_transform(back_transofrm)

plt.title("Pressure vs Age + Predicted")
plt.scatter(Y_hat_pred, X2_read_inversed[0][0], c='b', marker='X')
plt.scatter(df.X1, df.X2)
plt.plot(sorted(Y_hat), sorted(df.X2), c='r')
plt.show()

plt.title("Pressure vs Weight + Predicted")
plt.scatter(Y_hat_pred, X2_read_inversed[0][1], c='b', marker='X')
plt.scatter(df.X1, df.X3)
plt.plot(sorted(Y_hat), sorted(df.X3), c='r')
plt.show()



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Y_hat_pred, X2_read_inversed[0][0], X2_read_inversed[0][1], c='b', marker='X')
ax.scatter(df.X1, df.X2, df.X3, c='r', marker='o')
ax.plot(sorted(Y_hat), sorted(df.X2), sorted(df.X3), c='g', marker='v')

"""
# rotation shoudl be done with ax.draw() function, 
# but somehow does not work here

for angle in range(0, 360):
    ax.view_init(30, angle)
    plt.draw()
    plt.pause(.001)
"""
ax.view_init(10,120)
ax.set_xlabel("Systolic BP")
ax.set_ylabel("Age")
ax.set_zorder("Weight")
plt.show()
