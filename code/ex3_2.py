import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.metrics import classification_report


# neural network


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def predictNN(theta1, theta2, a1):
    z = sigmoid(a1 @ theta1.T)
    # 写成元组形式：()
    a2 = np.column_stack((np.ones(z.shape[0]), z))
    a3 = sigmoid(a2 @ theta2.T)
    p = np.argmax(a3, axis=1)
    p += 1
    return p


# read data
weights = loadmat('../data/ex3weights.mat')
data = loadmat('../data/ex3data1.mat')
X = data['X']
y = data['y']
theta1 = weights['Theta1']
theta2 = weights['Theta2']

# 添加偏置单元
X = np.column_stack((np.ones(X.shape[0]), X))
print(X.shape)
X
p = predictNN(theta1, theta2, X)
print("Training accuracy:{}%".format(
    np.mean((y.flatten() == p) * 100)
))

# report = classification_report(y, p)
# print(report)
