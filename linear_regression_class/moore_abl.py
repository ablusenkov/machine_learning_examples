#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 16:58:02 2020

@author: ablusenk
"""

import re 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

non_decimat = re.compile(r'[^\d]+')

ds = pd.read_csv("/Users/ablusenk/GitHub/udemy/machine_learning_examples/linear_regression_class/moore.csv",
                 names = ["Processor", "Transistors", "Year", "Vendor", "Technology", "Footprint"], sep = "\t", error_bad_lines=False)


# ========================================================================== #
# cleaning data to keep only digits 
# converting each value in int
Y = ds.Transistors
pat = r"\[.*\]|\D"
Y = Y.str.replace(pat, '', regex = True)
Y = pd.to_numeric(Y)


# cleaning data same way
X = ds.Year
X = X.str.replace('\[.*\]', '', regex = True)
X = pd.to_numeric(X) 


# ========================================================================== #
# Year/Transistors scatterplot
# ticks removed not to crowd plot

#THIS PLOT REMOVED, otherwise two plotted in one... shitty
plt.scatter(X, Y)
plt.xlabel("Year")
plt.ylabel("Transistors")
plt.xticks([])
plt.yticks([])
plt.show()

# converting Y (Transistor) to log to get linear regression
Y = np.log(Y)

plt.scatter(X,Y)
plt.xlabel("Year")
plt.ylabel("Transistors")
plt.title("Linear Reg of Moore")
plt.xticks([])
plt.yticks([])
plt.show()


# ========================================================================== #
# Solving linear regression problem for logarithmic Y
denominator = X.dot(X) - X.mean() * X.sum()
a = ( X.dot(Y) - Y.mean() * X.sum() ) / denominator
b = ( Y.mean() * X.dot(X) - X.mean() * X.dot(Y) ) / denominator

Y_pred = a * X + b


# ========================================================================== #
# calculating R-squared for linear regression
d1 = Y - Y_pred
d2 = Y - Y.mean()
R_sq = 1 - (d1.dot(d1) / d2.dot(d2))


plt.scatter(X,Y)
plt.plot(X, Y_pred)
plt.xlabel("Year")
plt.ylabel("Transistors")
plt.title("Linear Reg of Moore")
plt.xticks([])
plt.yticks([])
plt.show()

print("R-squared for particular problem is:", R_sq)


# ========================================================================== #
# find time to double (in accordance with M oore's law) number of transistors
# this will be our X2
# log(Transistors) = a*X + b 
# ...now taking log of both sides 
# Transistors = exp(b) * exp(a * X)
# 2*Transistors = 2 * exp(b) * exp(a * X) = exp(ln(2)) * exp(b) * exp(a * X)
#               = exp(b) * exp(a * X + ln(2))
# ...or 
# exp(b) * exp(a * X2) = exp(b) * exp(a * X + ln(2))
# ...or 
# a * X2 = a * X + ln(2)
# ...or 
# X2 = X + ln(2)/a
print("Time to double number of transistors is:", np.log(2)/a, "years") 

