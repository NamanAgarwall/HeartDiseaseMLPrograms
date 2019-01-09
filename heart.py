# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 20:06:07 2018

@author: nagarwal
"""
import numpy as np
import pandas as pd
from class_algo import KNN
from class_algo import DecisionTree
from class_algo import Perceptron

df = pd.read_csv("C:\\Users\\nagar\\Machine Learning For SF\\heart.txt", sep=" ")

#Seperate into X and Y
data = df.astype(float).as_matrix()
np.random.shuffle(data)
X = data[:,:-1]
Y = data[:,-1]

# Changing to -1
Y_P = np.copy(Y)
Y_P[Y_P == 2] = -1

#Feature Scaling
for i in [0,3,4,7,9,11]:
	X[:,i] = (X[:,i]-X[:,i].mean()) / X[:,i].std()
	
#Categorical Data
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [2,6,10,12])
X = onehotencoder.fit_transform(X).toarray()

# Split in train and test 
Ntrain = len(Y) // 2
Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

# Spliting for Perceptron
Ytrain_P = Y_P[:Ntrain]
Ytest_P = Y_P[Ntrain:]

knn = KNN(5)
knn.fit(Xtrain, Ytrain)
print("Train Accuracy (KNN):", knn.score(Xtrain, Ytrain))
print("Test Accuracy (KNN):", knn.score(Xtest, Ytest))

decision_tree = DecisionTree(max_depth=4)
decision_tree.fit(Xtrain, Ytrain)
print("Train Accuracy (Decision Tree):", decision_tree.score(Xtrain, Ytrain))
print("Test Accuracy (Decision Tree):", decision_tree.score(Xtest, Ytest))

perceptron = Perceptron()
perceptron.fit(Xtrain, Ytrain_P)
print("Train Accuracy (Perceptron):", perceptron.score(Xtrain, Ytrain_P))
print("Test Accuracy (Perceptron):", perceptron.score(Xtest, Ytest_P))
