#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 00:36:34 2020

@author: ablusenk
"""

import numpy as np 
import matplotlib.pyplot as plt

# ========================================================================== #
# number of data points 
#
N = 50 

X = np.linspace(0, 10, N)
Y = 0.5*X + np.random.randn(N)

# ========================================================================== #
# adding outlier, last two values are +30 
#
Y[-1] += 30 
Y[-2] += 30 

plt.scatter(X,Y)
plt.show()

# ========================================================================== #
# modifying X, by adding Bias
# in Pandas DF I used to use np.insert(loc, name, value), here though
# need to rely on np.vstack
# 

X = np.vstack([np.ones(N), X]).T

# ========================================================================== #
# calculating maximum likelihood solution
# 
w_ml = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
Y_hat_ml = np.dot(X, w_ml)

# ========================================================================== #
# plot both data sets - original and calculated
#
plt.scatter(X[:,1], Y, c='g')
plt.plot(X[:,1], Y_hat_ml, c='r')
plt.show() 

# ========================================================================== #
# L2 regularization solution with l2 penalty set to 1000
# MAP for maximizing the posterior or this called RIDGE

l2 = 1000
w_map = np.linalg.solve(l2*np.eye(2) + X.T.dot(X), X.T.dot(Y))
Y_hat_map = np.dot(X,w_map)


plt.scatter(X[:,1], Y, c='g')
plt.plot(X[:,1], Y_hat_ml, c='r', label = 'Maximum Likelihood')
plt.plot(X[:,1], Y_hat_map, c='b', label = 'MAP/Ridge')
plt.show()