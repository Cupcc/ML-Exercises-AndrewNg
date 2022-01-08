import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
from scipy.optimize import minimize



# read data
from sklearn.metrics import classification_report

path = "../data/ex3data1.mat"
data = scio.loadmat(path)
# print(data)
raw_X = data['X']
raw_y = data['y']


# print(np.unique(data['y']))

# define sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# define regularized Cost Function
def regularized_cost(theta, X, y, lam):
    m = len(X)
    h = sigmoid(X @ theta.T)
    j = 1 / m * (-y @ np.log(h) - (1 - y) @ np.log(1 - h))
    penalty = lam * theta.T @ theta / (2 * m)
    return j + penalty


# regularized gradient
def regularized_gradient(theta, X, y, lam):
    m = len(X)
    h = sigmoid(X @ theta.T)
    partial_j = (h - y).T @ X / m
    partial_penalty = lam / m * theta
    partial_penalty[0] = 0
    return partial_j + partial_penalty


# classifier
# def one_vs_all(X, y, rate, labels):
#     all_theta = np.zeros((labels, X.shape[1]))
#     for i in range(1, labels + 1):
#         part_theta = np.zeros(X.shape[1])
#         result_y = np.array([1 if label == i else 0 for
#                              label in y])
#
#
#         part_theta = minimize(fun=regularized_cost, x0=part_theta,
#                               args=(X, result_y, rate), method='TNC',
#                               jac=relarized_gradient, options={'disp': True})
#         all_theta[i - 1, :] = part_theta.x
#     return all_theta

def one_vs_all(x, y, lam, k):
    all_theta = np.zeros((k, x.shape[1]))
    # (10,401)一共十类，加入一个x0,参数矩阵多一列theta0,所以共401*10个参数
    for i in range(1, k + 1):
        part_theta = np.zeros(x.shape[1])
        resu_y = np.array([1 if label == i else 0 for label in y])
        part_theta = minimize(fun=regularized_cost, x0=part_theta, args=(x, resu_y, lam), method='TNC',
                              jac=regularized_gradient)
        all_theta[i - 1, :] = part_theta.x  # i从1到10，而下标是从0到9
    
    return all_theta


def predict_all(X, all_theta):
    h = sigmoid(X @ all_theta.T)
    # return max of indices in axis 1.
    # the maximal indice in column
    h_argmax = np.argmax(h, axis=1) + 1
    return h_argmax


def plot_an_image(X):
    pick_one = np.random.randint(0, 4999)
    image = X[pick_one, :]
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.matshow(image.reshape((20, 20)), cmap='gray_r')
    plt.title('{}'.format(raw_y[pick_one]))
    plt.xticks([])
    plt.yticks([])
    plt.show()

def show_pictures(data):
    sample_idx = np.random.choice(np.arange(data['X'].shape[0]), 100)
    sample_images = data['X'][sample_idx, :]
    fig, ax_array = plt.subplots(nrows=10, ncols=10, sharex=True, sharey=True, figsize=(12, 12))
    for r in range(10):
        for c in range(10):
            ax_array[r, c].matshow(np.array(sample_images[10 * r + c].reshape((20, 20))).T, cmap=matplotlib.cm.binary)
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
    plt.show()
    
# show pictures
# show_pictures(data)
#
# plot_an_image(raw_X)
X = np.insert(raw_X, 0, 1, axis=1)
# y = raw_y.ravel()
y = raw_y.flatten()
# y = y.reshape((y.shape[0], 1))
print(y.shape)
# print(type(X))
# print(X.shape)
# print(type(y))
theta = np.zeros(X.shape[1])
print(theta.shape)
# g = regularized_gradient(theta, X, y, 1)
# c = regularized_cost(theta, X, y, 1)
# print(g)
# print(c)


# compute theta
all_theta = one_vs_all(X, y, 1, 10)
print(all_theta.shape)

def predict_all(X, all_theta):
    h = sigmoid(X @ all_theta.T)
    h_argmax = np.argmax(h, axis=1)
    h_argmax += 1
    return h_argmax

y_predict = predict_all(X, all_theta)
accuracy = np.mean(y_predict == y)
print("accuracy: {}%".format(accuracy*100))
print(classification_report(data['y'], y_predict))