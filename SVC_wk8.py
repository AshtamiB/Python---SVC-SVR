# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 03:51:10 2018

@author: Ashtami
"""

 SUPPORT VECTOR MACHINE (SVC)
import numpy as np
from sklearn import svm, datasets
import matplotlib.pyplot as plt
iris = datasets.load_iris()
X = iris.data[:, :2] # we only take the first two features.
y = iris.target
# ------------- create a mesh to plot in ----------------------
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
h = (x_max / x_min)/100
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
np.arange(y_min, y_max, h))
X_plot = np.c_[xx.ravel(), yy.ravel()]
# ------------- Create the linear SVC model object -------------
C = 1.0 # SVM regularization parameter
svc = svm.SVC(kernel='linear', C=C)
svc.fit(X, y)
Z = svc.predict(X_plot)
Z = Z.reshape(xx.shape)
plt.figure(figsize=(15, 5))
plt.subplot(121)
plt.contourf(xx, yy, Z, cmap=plt.cm.tab10, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.title('SVC with linear kernel')
# --------------- Create the rbf SVC model object --------------
C = 1.0 # SVM regularization parameter
svc = svm.SVC(kernel='rbf', C=C)
svc.fit(X, y)
Z = svc.predict(X_plot)
Z = Z.reshape(xx.shape)
plt.subplot(122)
plt.contourf(xx, yy, Z, cmap=plt.cm.tab10, alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Set1)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.xlim(xx.min(), xx.max())
plt.title('SVC with RBF kernel')
plt.show()


###############EXAMPLE 2########################
#==================== Import & Trim the Dataset ==================#
df= pd.read_csv('https://archive.ics.uci.edu/ml/machine-learningdatabases/00267/data_banknote_authentication.txt', header=None)
X = df[df.columns[0:2]]
y = df[df.columns[4]]
X = np.asarray(X.values)
y = np.asarray(y.values)
#================= Processing the Classification =================#
C = 1.0
lin_svc = svm.SVC(kernel='linear', C=C).fit(X, y) # SVC with linear
kernel
rbf_svc = svm.SVC(kernel='rbf', gamma=0.9, C=C).fit(X, y) # SVC with RBF
kernel
poly_svc = svm.SVC(kernel='poly', degree=2, C=C).fit(X, y) # SVC with
polynomial (degree 3) kernel
lin_test = lin_svc.predict(X)
rbf_test = rbf_svc.predict(X)
pol_test = poly_svc.predict(X)
#======================= Confusion Matrix ========================#
cnf_matrix_lin = confusion_matrix(y, lin_test)
cnf_matrix_rbf = confusion_matrix(y, rbf_test)
cnf_matrix_pol = confusion_matrix(y, pol_test)
print(cnf_matrix_lin)
print(cnf_matrix_rbf)
print(cnf_matrix_pol)
#===================== Plotting Data & a Mesh ====================#
h = .02 # step size in the mesh
# create a mesh to plot
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
titles = ['SVC with linear kernel', 'SVC with RBF kernel', 'SVC with polynomial
(degree 3) kernel']
for i, clf in enumerate((lin_svc , rbf_svc , poly_svc)):
 plt.subplot(2, 3, i + 1)
 Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# # Put the result into a color plot
 Z = Z.reshape(xx.shape)
 plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm)
 # Plot also the training points
 plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
 plt.xlabel('INPUT 1')
 plt.ylabel('INPUT 2')
 plt.xticks(())
 plt.yticks(())
 plt.title(titles[i])
 plt.subplot(2, 3, i+4)
 if i==0:
 sns.heatmap(cnf_matrix_lin.T, square=True, annot=True, fmt='d', cbar=False)
 if i==1:
 sns.heatmap(cnf_matrix_rbf.T, square=True, annot=True, fmt='d', cbar=False)
 if i==2:
 sns.heatmap(cnf_matrix_pol.T, square=True, annot=True, fmt='d',
cbar=False)##
 plt.xlabel('true label')
 plt.ylabel('predicted label')
 plt.title(titles[i])
plt.show()
