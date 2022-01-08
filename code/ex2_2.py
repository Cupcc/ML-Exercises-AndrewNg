import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt


# plot data
def draw_data(data):
    positive = data[data.Accepted == 1]
    negative = data[data.Accepted == 0]
    plt.figure(figsize=(12, 8))
    plt.scatter(positive.Test1, positive.Test2, marker='o', s=50, c='b', label='Accepted', )
    plt.scatter(negative.Test1, negative.Test2, marker='x', s=50, c='r', label='Rejected', )
    plt.xlabel('Test1 score')
    plt.ylabel('Test1 score')
    plt.legend(loc=0)
    return plt


# feature mapping
def feature_mapping(x1, x2, degree):
    datamap = {}
    # 从0阶扩展到degree阶，就不用插入偏置项了
    for i in range(0, degree + 1):
        for j in range(i + 1):
            datamap['f{}{}'.format(i - j, j)] = np.power(x1, i - j) * np.power(x2, j)
    return pd.DataFrame(datamap)


# sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 检验sigmoid函数
# nums = np.arange(-10, 10, 0.1)
# fig, ax = plt.subplots(figsize=(12,8))
# ax.plot(nums, sigmoid(nums), 'r')
# plt.show()
# regularization cost function


def regularization_cost(theta, X, y, lam):
    m = len(X)
    h = sigmoid(X @ theta.T)
    # !!!!记得加括号！！！
    j = -1 / m * (y @ np.log(h) + (1 - y) @ np.log(1 - h))
    penalty = lam / 2 * m * (theta @ theta)
    return j + penalty


# gradient
def regularized_gradient(theta, X, y, lam):
    m = len(X)
    partial_j = 1 / m * ((sigmoid(X @ theta.T) - y).T @ X)
    partial_penalty = lam / m * theta
    partial_penalty[0] = 0
    return partial_j + partial_penalty


# desicion boundary
def boundary(theta, degree):
    axis = np.linspace(-1, 1.2, 100)
    x1, x2 = np.meshgrid(axis, axis)
    z = feature_mapping(x1.ravel(), x2.ravel(), degree)
    z = z.values
    z = z @ theta
    z = z.reshape(x1.shape)
    plt.contour(x1, x2, z, 0, colors='#41bc60')
    plt.title('Boundary')


# read data
init_data = "data/ex2data2.txt"
init_data = pd.read_csv(init_data, header=None, names=['Test1', 'Test2', 'Accepted'])
# print(init_data.info)
# data processing
degree = 5
data = feature_mapping(init_data['Test1'], init_data['Test2'], degree)
# X = data.values
X = data.to_numpy()
print(X.shape)
y = init_data['Accepted'].to_numpy()
print(X.shape[1])
# theta initialization
theta = np.zeros(X.shape[1])
print(theta)
result = opt.minimize(fun=regularization_cost, x0=theta, args=(X, y, 1),
                      method='tnc', jac=regularized_gradient)
print(regularization_cost(theta, X, y, 1))
min_theta = result.x
print(min_theta)
draw_data(init_data)
boundary(min_theta, degree)
plt.show()
