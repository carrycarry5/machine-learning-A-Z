# -*- coding: utf-8 -*-

# 导入标准库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 导入数据集
dataset = pd.read_csv("../50_Startups.csv")
X = dataset.iloc[:,:-1].values    
y = dataset.iloc[:,-1].values

# 标签分类
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder = LabelEncoder()
X[:,3] = label_encoder.fit_transform(X[:,3])

# 虚拟编码
onehot_encoder = OneHotEncoder(categorical_features=[3])
X = onehot_encoder.fit_transform(X).toarray()

# 处理虚拟编码陷阱
X = X[:,1:]

# 分割训练集和测试
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size = 0.2, random_state=0)

# 创建模型，用训练集拟合模型
from sklearn.linear_model import LinearRegression
linear_regression = LinearRegression()
linear_regression.fit(X_train, y_train)
# 用创建好的模型开始预测
y_pred = linear_regression.predict(X_test)
# 显示预测结果和实际结果
plt.plot(y_pred)
plt.plot(y_test)
plt.show()

# 给训练集填上一项全1列
X_train = np.append(arr = np.ones((40,1)),values=X_train, axis=1)
# 运用反向淘汰算法
import statsmodels.formula.api as sm
X_opt = X_train[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog = y_train, exog=X_opt).fit()
#print(regressor_OLS.summary())
X_opt = X_train[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog = y_train, exog=X_opt).fit()
#print(regressor_OLS.summary())
X_opt = X_train[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog = y_train, exog=X_opt).fit()
#print(regressor_OLS.summary())
X_opt = X_train[:,[0,3,5]]
regressor_OLS = sm.OLS(endog = y_train, exog=X_opt).fit()
#print(regressor_OLS.summary())
X_opt = X_train[:,[0,3]]
regressor_OLS = sm.OLS(endog = y_train, exog=X_opt).fit()
#print(regressor_OLS.summary())




