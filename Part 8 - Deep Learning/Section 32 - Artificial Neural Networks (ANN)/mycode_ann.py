# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 18:51:11 2018

@author: AlanP
"""

# import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as ply

# ----------------------part1 data preprocessing------------------
# import the dataset
dataset = pd.read_csv("Churn_Modelling.csv")
X = dataset.iloc[:,3:-1].values # 序号，id，姓名，是无用数据
y = dataset.iloc[:,-1].values

# encode category data
from sklearn.preprocessing import LabelEncoder
labelEncode = LabelEncoder()
X[:,1] = labelEncode.fit_transform(X[:,1])
X[:,2] = labelEncode.fit_transform(X[:,2])
# one-hot   # 对于国家来说是必要的，但对于性别来说是不必要的，因为性别就两类
from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder(categorical_features=[1])
X = onehot_encoder.fit_transform(X).toarray()
# 处理虚拟编码陷阱
X = X[:,1:]

# split the dataset into the training set and test sset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size =0.2 ,random_state = 0)

# feature scale
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

# ---------------------part2 let's make the ANN-----------------------
# import the keras libraries and the packages
import keras
from keras.models import Sequential  # 用于初始化神经网络
from keras.layers import Dense   # 用于添加层

# Initialising the ANN
classifier = Sequential()

# add the input layer and the first hidden layer
# units是隐藏层的神经元个数，根据经验：（输入层的神经元个数+输出层的神经元个数）/ 2
# 激活函数：relu， 初始化方式：uniform， 输入层维度：11
classifier.add(Dense(units = 6,activation ='relu',
                     kernel_initializer = 'uniform'))

# add the second hidden layer
classifier.add(Dense(units = 6,activation ='relu',
                     kernel_initializer = 'uniform'))

# add the output layer
# 这里处理的是二维分类，sigmoid函数是处理二维分类的，如果是多维，可以用softmax，可以想象成多维的sigmoid
classifier.add(Dense(units = 1,activation ='sigmoid',
                     kernel_initializer = 'uniform'))

# compiling the ANN
# 编译神经网络
# optimizer:优化器，loss:损失函数,metrics:性能评估器
classifier.compile(optimizer='adam', loss='binary_crossentropy',metrics = ['accuracy'])

# Fitting the ANN to the Training set
# 每10个样本求一次损失函数，做100期前向反向传播
classifier.fit(X_train, y_train, batch_size=10, epochs=100)

# --------------------part3 make the predictions and evaluating model--------------
# predict the test set result
y_pred = classifier.predict(X_test)
# y_pred_binary = (y_pred > 0.5)
from sklearn.preprocessing import binarize   # 二值化
y_pred_binary = binarize(y_pred, threshold = 0.5)

# make a confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_binary)    # 0.8415 与测试集的准确率差不多，说明没有过拟合  



