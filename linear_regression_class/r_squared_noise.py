#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 01:00:51 2020

@author: ablusenk
"""

# ========================================================================== #
# Will R-suare be improved with random noise added to feature? 
# 

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

df = pd.read_excel("http://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/mlr/excel/mlr02.xls")

df.insert(0, "X0", 1)

df.insert(4, "R", np.random.randint(0, 300, size = (11, 1)))


X = df[["X0", "X2", "X3"]]
Y = df[["X1"]]

X1 = df[["X0", "X2", "X3", "R"]]
Y1 = df[["X1"]]



w = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
Y_heat = np.dot(X, w)

w1 = np.linalg.solve(X1.T.dot(X1), X1.T.dot(Y1))
Y_heat1 = np.dot(X1, w1)


d1 = Y - Y_heat 
d2 = Y - Y.mean()

d1 = d1.X1
d2 = d2.X1

r_s = 1 - d1.dot(d1)/d2.dot(d2)



d1_1 = Y1 - Y_heat1 
d2_1 = Y1 - Y1.mean()

d1_1 = d1_1.X1
d2_1 = d2_1.X1

r_s_1  = 1 - d1_1.dot(d1_1)/d2_1.dot(d2_1)

print("R-squared REG vs RAND. Where RAND is greater:", r_s_1 > r_s)
print(r_s)
print(r_s_1)


# ========================================================================== #
# answer is Yes. Nomrally we expect random input to be uncorrelated with 
# target.
# Same time random sample mean is not equal true mean, as result of that R 
# improved.
# ========================================================================== #