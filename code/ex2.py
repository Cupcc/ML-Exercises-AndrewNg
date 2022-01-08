import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from sklearn import linear_model
from sklearn.metrics import classification_report

# **************
# logistic regression
# **************

# read data

path = "../data/ex2data1.txt"
data = pd.read_csv(path, header=None, names=['Exam1', 'Exam2', 'Admitted'])
# print(data.head())

# process data
data.insert(0, "Ones", 1)
X = data.iloc[:, :-1].to_numpy()
y = data.Admitted.to_numpy()

theta = np.zeros(X.shape[1])


# print(theta.shape)
# print(X.shape, y.shape)
def draw_data(data):
    """
    :param data:
    :return:
    """
    positive = data[data.Admitted == 1]
    negative = data[data.Admitted == 0]
    plt.scatter(positive.Exam1, positive.Exam2, marker='^', c='blue', label='Admitted')
    plt.scatter(negative.Exam1, negative.Exam2, marker='x', c='r', label='Not Admitted')
    plt.title('Admission')
    plt.xlabel('score1')
    plt.ylabel('score2')
    plt.legend(loc=0, fontsize='small')


# draw_data(data)


# sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# cost fuction
def compute_cost(theta, X, y):
    m = len(X)
    h = sigmoid(X @ theta.T)
    j = -1 / m * ((y @ np.log(h)) + (1 - y) @ np.log(1 - h))
    # h = sigmoid(X.dot(theta))
    # j = (y.dot(np.log(h)) + (1 - y).dot(np.log(1 - h))) / (-m)
    return j


# gradient descent
def gradient(theta, X, y, ):
    return ((sigmoid(X @ theta.T)) - y).T @ X


# prediction
def predict(theta, X):
    h = sigmoid(X @ theta.T)
    return [1 if X >= 0.5 else 0 for X in h]


# decision boundary
def boundary(theta):
    x1 = np.arange(20, 100, 0.01)
    x2 = (theta[0] + theta[1] * x1) / -theta[2]
    plt.plot(x1, x2)


# verify cost and gradient
cost = compute_cost(theta, X, y)
g = gradient(theta, X, y)
print(cost)
print(g)


# compute last theta
# result = opt.fmin_tnc(func=compute_cost, x0=theta, fprime=gradient, args=(X, y))
result2 = opt.minimize(fun=compute_cost, x0=theta, args=(X, y), method='tnc', jac=gradient)
# print(result)
# print(result2)
# new_theta = result[0]
new_theta = result2.x
print(compute_cost(new_theta, X, y))


# compute score of precision, recall, f1-score and support.
# print(classification_report(predict(new_theta, X), y))
# model = linear_model.LogisticRegression()
# model.fit(X, y)
# print(model.score(X, y))

# plot data and boundary
# draw_data(data)
# boundary(new_theta)
# plt.show()


# admitted probability whose has scores of 45 and 85.
def hfunc(theta, X):
    return sigmoid(theta.T @ X)


probability = hfunc(new_theta, [1, 45, 85])
print(probability)
prediction = predict(new_theta, X)

# model accuracy
correct = [1 if ((a == 1 and b == 1) or (a == 0 and b == 0)) else 0
           for (a, b) in zip(prediction, y)]
accuracy = (sum(map(int, correct)) % len(correct))
print("accuracy={0}%".format(accuracy))
