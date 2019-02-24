# -*- coding: utf-8 -*-
"""

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

# Fitting classifier to the Training set
# Create your classifier here
from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state=0)
# kernel : rbf, linear, poly, sigmoid, precomputed
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# -------------------------apply k-fold cross validation----------
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
# 得到10次交叉验证的准确率
acc_mean = accuracies.mean()

# -----------apply grid search to find the best model and the best paramters
#paramters = [
#        {'C':[1,10,100,1000], 'kernel':['linear']},
#        {'C':[1,10,100,1000], 'kernel':['rbf'], 'gamma':[0.5, 0.1,0.01,0.001,0.0001]}
#        ]
# best:   C:1, gamma:0.5, kernel:rbf
# 由上测试可知，C的最佳参数在1附近，gamma的最佳值在0.5附近，那么缩小范围查找最佳参数
paramters = [
        {'C':[1,10,100,1000], 'kernel':['linear']},
        {'C':[1,2,3,4,5,6], 'kernel':['rbf'], 'gamma':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]}
        ]
# best:   C:1, gamma:0.7

from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator = SVC(), 
             param_grid = paramters, 
             scoring = 'accuracy',
             cv = 10,
             n_jobs = -1,   
             )

grid_search.fit(X_train, y_train)   # 得到的是估计器的集合
best_accuracy = grid_search.best_score_   # 最好的分数
best_classifier = grid_search.best_estimator_   # 最好的估计器
best_params = grid_search.best_params_   # 最好的参数


# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('orange', 'blue'))(i), label = j)
plt.title = 'Classifier (Training set)'
plt.xlabel = 'Age'
plt.ylabel = 'Estimated Salary'
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('orange', 'blue'))(i), label = j)
plt.title = 'Classifier (Test set)'
plt.xlabel = 'Age'
plt.ylabel = 'Estimated Salary'
plt.legend()
plt.show()






