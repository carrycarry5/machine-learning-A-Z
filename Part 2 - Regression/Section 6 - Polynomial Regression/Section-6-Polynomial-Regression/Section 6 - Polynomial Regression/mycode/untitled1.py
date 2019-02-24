# -*- coding: utf-8 -*-

# 导入标准库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 加载数据集
dataset = pd.read_csv("../Position_Salaries.csv")
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

# 不做分割和特征缩放

# 用简单线性模型预测
from sklearn.linear_model import LinearRegression
lin_reg_1 = LinearRegression()
lin_reg_1.fit(X, y)
lin_y_pred = lin_reg_1.predict(X)

# 将集合变成多项式
from sklearn.preprocessing import PolynomialFeatures
polynomial_feature = PolynomialFeatures(degree = 4)  # 设置维度为4
X_poly = polynomial_feature.fit_transform(X)
# 用多项式模型进行拟合
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# 可视化线性模型
plt.scatter(X,y,c="red")
plt.plot(X, lin_y_pred)
plt.show()

# 可视化多项式模型
# 将X的维度变大
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid),1)
poly_y_pred = lin_reg_2.predict(polynomial_feature.fit_transform(X_grid))
plt.scatter(X,y, c="red")
plt.plot(X_grid, poly_y_pred)
plt.show()

# 用简单线性模型和多项式回归模型做预测
lin_reg_1.predict(6.5)  # 33w
lin_reg_2.predict(polynomial_feature.fit_transform(6.5))  # 15.88w


