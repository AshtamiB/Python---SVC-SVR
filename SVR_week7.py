# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 03:17:46 2018

@author: Ashtami
"""
#EXAMPLE 1
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
#---------------------------- Generate Data ----------------------------#
X = np.sort(5 * np.random.rand(200, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - np.random.rand(40))
#-----------------------------------------------------------------------#
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3, degree=3)
y_rbf = svr_rbf.fit(X, y).predict(X)
y_lin = svr_lin.fit(X, y).predict(X)
y_poly = svr_poly.fit(X, y).predict(X)
lw = 2
plt.figure(figsize=(12, 7))
plt.scatter(X, y, color='darkorange', label='data')
plt.plot(X, y_rbf, color='navy', lw=lw, label='RBF model')
plt.plot(X, y_lin, color='c', lw=lw, label='Linear model')
plt.plot(X, y_poly, color='cornflowerblue', lw=lw, label='Polynomial
model') # ’poly’ Sometimes takes time to perform, pls be patient !!
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()


#EXAMPLE 2 SVR vs OLS

import sklearn.preprocessing as skp
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
X = np.linspace(0,100,101)
y = np.array([(100*np.random.rand(1)+num) for num in (5*X+10)])
X = skp.scale(X, axis=0)
y = skp.scale(y, axis=0)
X_train, X_test, y_train, y_test = train_test_split(X, y)
svr = SVR(kernel='linear')
ols = LinearRegression()
svr.fit(X_train.reshape(-1,1),y_train.flatten())
ols.fit(X_train.reshape(-1,1), y_train.flatten())
pred_SVR = svr.predict(X_test.reshape(-1,1))
pred_OLS = ols.predict(X_test.reshape(-1,1))
print(np.sqrt(mean_squared_error(y_test, pred_SVR)))
print(np.sqrt(mean_squared_error(y_test, pred_OLS)))
plt.plot(X, y, 'kv',label='True data')
plt.plot(X_test, pred_SVR,'ro' ,label='SVR')
plt.plot(X_test, pred_OLS, label='Linear Reg')
plt.legend(loc='upper left')
plt.show()
