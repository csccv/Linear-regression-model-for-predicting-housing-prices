# 首先，导入必要的包：
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 接着，读取数据集并进行数据清洗：
data = pd.read_csv('housing.csv')
data = data.dropna()

# 对数据进行可视化
fig = plt.figure(figsize=(20, 5))
fig1 = plt.subplot(131)
plt.scatter(data.loc[:, '面积'], data.loc[:, '价格'])
plt.title('Price VS Size')

fig2 = plt.subplot(132)
plt.scatter(data.loc[:, '人均收入'], data.loc[:, '价格'])
plt.title('Price VS Income')

fig3 = plt.subplot(133)
plt.scatter(data.loc[:, '平均房龄'], data.loc[:, '价格'])
plt.title('Price VS House_age')
plt.show()
# 选择自变量和因变量：
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
# 对自变量进行特征缩放：
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
X = (X - mean) / std


# 定义代价函数和梯度下降函数：
def compute_cost(X, y, theta):
    m = len(y)
    h = X.dot(theta)
    J = 1 / (2 * m) * np.sum((h - y) ** 2)
    return J


def gradient_descent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = np.zeros(num_iters)
    for i in range(num_iters):
        h = X.dot(theta)
        theta = theta - alpha * (1 / m) * (X.T.dot(h - y))
        J_history[i] = compute_cost(X, y, theta)
    return theta, J_history


# 设置初始参数并运行梯度下降算法：

theta = np.zeros(X.shape[1])
alpha = 0.01
num_iters = 1000
theta, J_history = gradient_descent(X, y, theta, alpha, num_iters)
# 绘制代价函数的变化：
plt.plot(J_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost function using Gradient Descent')
plt.show()
# 最后，预测一个新的房价：
a = input("房屋面积：")
b = input("人均收入：")
c = input("房屋年龄：")
test = np.array([a, b, c])
test = test.astype('float64')
test = (test - mean) / std
price = test.dot(theta)
print('Predicted price:', price)
