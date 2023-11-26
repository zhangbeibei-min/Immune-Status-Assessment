#!/usr/bin/env python
# encoding: utf-8

"""
@version: 3.5.2
@author: beibei zhang
@license: Apache Licence 
@contact: 695193839@qq.com
@site: http://www.baidu.com
@software: PyCharm
@file: Cubic_polynomial_fitting.py
@time: 2023/11/26 20:33
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
# Read Data
data = pd.read_csv('age-score.csv')
x = data['age'].values.reshape(-1, 1)
y = data['score'].values

# Define a range of orders
#degrees = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
degrees = [1, 2, 3, 4]
# Initializes the minimum MSE and the corresponding order
best_mse = float('inf')
best_degree = None
best_model = None

# Example Initialize the MSE list
mse_list = []

# Traverse different orders
for degree in degrees:
    # Create polynomial features
    polynomial_features = PolynomialFeatures(degree=degree)
    #x_poly = polynomial_features.fit_transform(x)
    x_poly = polynomial_features.fit_transform(x.reshape(-1, 1))
    # Fit a linear regression model
    model = LinearRegression()
    model.fit(x_poly, y)
    # prediction
    y_pred = model.predict(x_poly)
    # Calculate the mean square error
    mse = mean_squared_error(y, y_pred)
    # Puts the current MSE in the list
    mse_list.append(mse)
    # Update the minimum MSE and corresponding order
    if mse < best_mse:
        best_mse = mse
        best_degree = degree
        best_model = model
# Output MSE for each order
for i in range(len(degrees)):
    print(f"Degree {degrees[i]} MSE: {mse_list[i]}")

# Draw the fit results
plt.scatter(x, y, color='blue', label='Actual')
plt.plot(x, best_model.predict(polynomial_features.fit_transform(x.reshape(-1, 1))), color='red', label='Fit')
plt.xlabel('Age')
plt.ylabel('Score')
plt.legend()
plt.show()

# The relationship between MSE and order is plotted
plt.plot(degrees, mse_list, marker='o')
plt.xticks(degrees)
plt.xlabel('Degree')
plt.ylabel('MSE')
plt.title("MSE vs Degree Relationship")

plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

plt.savefig('MSE.jpg', dpi=300, format='jpg')
plt.show()


######************************2.Draw a third-order polynomial fit*******************************
data = pd.read_csv('age-score.csv')
#print(data)
x = data.age
x = np.array(x)
print('x is :\n', x)
y = data['score'].values
print('y is :\n', y)
# Fit with a 3rd degree polynomial
f1 = np.polyfit(x, y, 3)
print('f1 is :\n', f1)

p1 = np.poly1d(f1)
print('p1 is :\n', p1)

yvals = p1(x)  #
print('yvals is :\n', yvals)

# plot
plot1 = plt.plot(x, y ,"o",alpha = 0.7,markersize=16,label='original values')
plot2 = plt.plot(x, yvals, 'r', label='y=-6.580e-07*x^3 + 9.662e-05*x^2-5.468e-03*x+ 6.177e-01')
plt.xlabel('age',fontsize=14)
plt.ylabel('score',fontsize=14)
plt.legend(loc=4)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
#plt.title('polyfitting')
plt.savefig('age-score.jpg', dpi=300, format='jpg', bbox_inches = 'tight')
plt.show()
