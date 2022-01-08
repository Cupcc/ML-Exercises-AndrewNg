import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['Arial Unicode MS']

path = "../data/ex1data1.txt"
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
# print(data.shape)
# print(data.head())

data.plot(kind='scatter', x='Population', y='Profit', figsize=(8, 5))
# plt.show()

# define Cost function
def computeCost(X, y, theta):
    # Cost = 1/(2m) sum[(X * theta.T - y)^2]
    inner = np.power((X @ theta.T - y), 2)
    return np.sum(inner) / (2 * len(X))
# increase column 0, total data are 3 columns
data.insert(0, 'Ones', 1)
# print(data.shape)
cols = data.shape[1] # columns/ length for dimension 2 of data
X = data.iloc[:, :-1]
# print(X.head())
y = data.iloc[:, cols - 1:cols]
# print(X.head())
# print(y.head())

# remove labels
# X = np.array(y.values) recommend using to_numpy()
X = X.to_numpy()
# y = np.array(y.values) recommend using to_numpy()
y = y.to_numpy()
theta = np.array([[0, 0]])
#  compute Cost
print(computeCost(X, y, theta))


# -----
# Gradient Descent
def gradientDescent(X, y, theta, alpha=0.01, iters=100):
    # matrix theta initialization
    cost = np.zeros(iters)
    m = len(X)
    for i in range(iters):
        # simultaneously update all theta
        error = X @ theta.T - y
        theta = theta - alpha/m * error.T @ X
        cost[i] = computeCost(X, y, theta)
    return theta, cost


# assignment
alpha = 0.01
iters = 1500

final_theta, final_cost = gradientDescent(X, y, theta, alpha, iters)
# print("parameters theta are:")
print(final_theta)
# final_cost = computeCost(X, y, final_theta)
# print("final cost:")
print(final_cost)
predict1 = np.array([1, 7]).dot(final_theta.T)
print("predict1:", predict1)


# equally generate n dots from min to max of "data X"
x = np.linspace(data.Population.min(), data.Population.max(), 100)
# hypothesis function
hypothesis = final_theta[0, 0] + (final_theta[0, 1] * x)

# plot pictures
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(x, hypothesis, 'red', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Training data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Prediction Profit vs. Population Size')
plt.show()

