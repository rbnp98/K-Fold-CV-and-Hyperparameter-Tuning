# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 21:32:28 2019

@author: User

"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC()
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)

# K-fold CV
from sklearn.model_selection import cross_val_score as cvs
score = cvs(classifier, X_train, y_train, cv=10)
score.mean()
score.std()

# Tuning hyperparameters
parameters = {'C':[10,1,0.1,0.01],
              'kernel':['linear', 'rbf'], 
              'gamma':[0.1,0.2,0.3,0.4]}
# here 
from sklearn.model_selection import GridSearchCV
gs = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy', cv =10)
gs.fit(X_train,y_train)
bests = gs.best_score_#average score of the model over the cv=10 with best combination of parameters
best = gs.best_params_#best combination of parameters 
