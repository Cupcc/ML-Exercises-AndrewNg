import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

# read data
path = "../data/ex1data2.txt"
data = pd.read_csv(path, header=None, names=["Size", "Bedrooms", "Price"])

# Mean normalization
data = (data - data.mean()) / data.std()  # std() standard deviation
# print(data.head())
# separate varies
data.insert(0, "Ones", 1)
X = data.iloc[:, : -1].to_numpy()
y = data.iloc[:, -1:].to_numpy()
theta = np.zeros((1, 3))
m = len(X)


# print(theta)
# print(X)
# print(y)


# define Cost function
def computeCost(X, y, theta):
    inner = np.power(X @ theta.T - y, 2)
    return np.sum(inner) / (2 * m)


# define gradient descent function
def gradientDescent(X, y, theta, alpha=0.27, iters=1000):
    cost = np.zeros(iters)
    for i in range(iters):
        theta = theta - (alpha / m) * ((X @ theta.T - y).T @ X)
        cost[i] = computeCost(X, y, theta)
    return theta, cost


# parameters
alpha = 0.02
iters = 1000
new_theta, new_cost = gradientDescent(X, y, theta, alpha, iters)
print("theta:", new_theta)
print("minimal cost:", new_cost[-1])

# plot
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(np.arange(iters), new_cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show()


# Normal Equation
def normalEquation(X, y):
    theta = np.linalg.inv(X.T @ X) @ X.T @ y
    return theta


theta2 = normalEquation(X, y)
print("normal equation method:")
print("theta:", theta2)
