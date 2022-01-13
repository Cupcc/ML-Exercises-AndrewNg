import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt
from sklearn.metrics import classification_report

raw_data = loadmat('../data/ex4data1.mat')
weight = loadmat('../data/ex4weights.mat')
theta1, theta2 = weight['Theta1'], weight['Theta2']

X = raw_data['X']
y = raw_data['y']
# print(X.shape)
# print(y.shape)
print(theta1.shape)
print(theta2.shape)
X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)


# 扁平化并拼接
def serialize(a, b):
    return np.concatenate((np.ravel(a), np.ravel(b)))


theta = serialize(theta1, theta2)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# sigmoid求导并化简
def sigmoid_gradient(z):
    return sigmoid(z) * (1 - sigmoid(z))


print(sigmoid_gradient(0))


def random_init(size):
    return np.random.uniform(-0.12, 0.12, size)


def expand_y(y):
    res = []
    for i in y:
        y_array = np.zeros(10)
        y_array[i - 1] = 1
        res.append(y_array)
    return np.array(res)


y = expand_y(y)


# print(y)
def plot_images(X):
    index = np.random.choice(range(len(X)), 100)
    images = X[index]
    fig, ax_array = plt.subplots(10, 10, sharey=True, sharex=True, figsize=(10, 10))
    for r in range(10):
        for c in range(10):
            ax_array[r, c].matshow(images[r * 10 + c].reshape(20, 20), cmap="gray_r")
    plt.xticks([])
    plt.yticks([])
    plt.show()


# plot_images(X)

def deserialize(seq):
    return seq[:25 * 401].reshape(25, 401), seq[25 * 401:].reshape(10, 26)


def feed_forward(theta, X):
    t1, t2 = deserialize(theta)
    m = X.shape[0]
    a1 = X
    z2 = a1 @ t1.T
    a2 = np.insert(sigmoid(z2), 0, np.ones(m), axis=1)
    z3 = a2 @ t2.T
    h = sigmoid(z3)
    return a1, z2, a2, z3, h


def cost(theta, X, y):
    m = len(X)
    _, _, _, _, h = feed_forward(theta, X)
    J = -y * np.log(h) - (1 - y) * np.log(1 - h)
    return J.sum() / m


def regularized_cost(theta, X, y, l=1):
    t1, t2 = deserialize(theta)
    m = X.shape[0]
    reg_t1 = (l / (2 * m)) * np.power(t1[:, 1:], 2).sum()
    reg_t2 = (l / (2 * m)) * np.power(t2[:, 1:], 2).sum()
    return cost(theta, X, y) + reg_t1 + reg_t2


def gradient(theta, X, y):
    t1, t2 = deserialize(theta)
    a1, z2, a2, z3, h = feed_forward(theta, X)
    d3 = h - y
    d2 = d3 @ t2[:, 1:] * sigmoid_gradient(z2)
    D2 = d3.T @ a2
    D1 = d2.T @ a1
    D = (1 / len(X)) * serialize(D1, D2)
    return D


# 正则化梯度
def regualrized_gradient(theta, X, y, l=1):
    t1, t2 = deserialize(theta)
    D1, D2 = deserialize(gradient(theta, X, y))
    t1[:, 0] = 0
    t2[:, 0] = 0
    m = len(X)
    reg_D1 = D1 + (1 / m) * t1
    reg_D2 = D2 + (1 / m) * t2
    return serialize(reg_D1, reg_D2)


# 梯度检测
def gradient_checking(theta, X, y, e):
    def a_numeric_grad(plus, minus):
        return (regularized_cost(plus, X, y) - regularized_cost(minus, X, y))
    
    numeric_grad = []
    for i in range(len(theta)):
        plus = theta.copy()
        minus = theta.copy()
        plus[i] += e
        minus[i] -= e
        grad_i = a_numeric_grad(plus, minus)
        numeric_grad.append(grad_i)
    
    numeric_grad = np.array(numeric_grad)
    analytic_grad = regularized_cost(theta, X, y)
    diff = np.linalg.norm(numeric_grad - analytic_grad) \
           / np.linalg.norm(numeric_grad + analytic_grad)
    
    print('If your backpropagation implementation is correct,\n'
          'the relative will be smaller than 10e-9 (assume epsilon=0.0001).\n'
          'Relative Difference:{}\n'.format(diff))


def nn_training(X, y):
    init_theta = random_init(10285)  # 25*401+10*26
    res = opt.minimize(fun=regularized_cost,
                       x0=init_theta,
                       args=(X, y, 1),
                       method='TNC',
                       jac=regualrized_gradient,
                       options={'maxiter': 250})
    return res


# 数据预处理

# print(theta.shape)
# _, _, _, _, h = feed_forward(theta, X)
# print(h)
# print(cost(theta, X, y))
# print(regularized_cost(theta, X, y))


# print(gradient_checking(theta,X,y,0.0001))

# 训练网络
res = nn_training(X, y)
final_theta=res.x

# print(nn_training(X,y))

# 计算准确率
def accuracy(theta, X, y):
    _, _, _, _, h = feed_forward(theta, X)
    y_pred = np.argmax(h, axis=1) + 1
    print(classification_report(y, y_pred))


y = raw_data['y']
y = y.reshape(y.shape[0])
# print(y.shape)
# accuracy(final_theta, X, y)


# 显示隐藏层
def plot_hidden_layer(theta):
    t1, _ = deserialize(theta)
    t1 = t1[:, 1:]
    fig, ax_array = plt.subplots(5, 5, sharex=True, sharey=True, figsize=(6, 6))
    
    for r in range(5):
        for c in range(5):
            ax_array[r, c].matshow(t1[r * 5 + c].reshape(20, 20), cmap='gray_r')
            plt.xticks([])
            plt.yticks([])
    
    plt.show()

plot_hidden_layer(final_theta)