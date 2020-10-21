from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

# Read in data
df = pd.read_csv("week3.csv", comment='#')
X1 = df.iloc[:, 0]
X2 = df.iloc[:, 1]
X = np.column_stack((X1, X2))
y = np.array(df.iloc[:, 2])

# Plot Q1 a - 3D figure.
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], y)
ax.set_xlabel('X1', color='blue', size=15)
ax.set_ylabel('X2', color='blue', size=15)
ax.set_zlabel('Y', color='blue', size=15)
ax.set_title("3D Plot of training data as scatter")

# Q1 b training lasso models
# Grab the powers up to 5.
poly = PolynomialFeatures(5)
X_poly = poly.fit_transform(X)
# Generate the [-5,5] grid for predictions.
Xtest = []
grid = np.linspace(-5, 5)
for i in grid:
    for j in grid:
        Xtest.append([i, j])
# Grab the powers up to 5 for the grid data.
Xtest = np.array(Xtest)
Xtest_poly = poly.fit_transform(Xtest)

# Plot the plan with training data alone
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(X[:, 0], X[:, 1], y)
ax.set_xlabel('X1', color='blue', size=15)
ax.set_ylabel('X2', color='blue', size=15)
ax.set_zlabel('Y', color='blue', size=15)
ax.set_title("Training data as a plane")

# Question 1 b and c - using C = 1 200 and 500
for C in [1, 200, 5000]:
    # Note that alpha = 1 / C
    model = linear_model.Lasso(alpha=(1/C))
    model.fit(X_poly, y)
    ypred = model.predict(Xtest_poly)
    # Print model parameters
    print("c = ", C)
    print(model.intercept_)
    print(model.coef_)
    # Plot model predictions.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(X[:, 0], X[:, 1], y)
    ax.scatter(Xtest[:, 0], Xtest[:, 1], ypred, color="red")
    ax.set_xlabel('X1', color='blue', size=15)
    ax.set_ylabel('X2', color='blue', size=15)
    ax.set_zlabel('Y', color='blue', size=15)
    ax.set_title("Lasso Predictions over [-5, 5] Grid, C = " + str(C))

# Question e - Ridge Model
for C in [1, 200, 5000]:
    # Note that alpha = 1 / 2C for Ridge regression
    model = Ridge(alpha=(1/(2 * C)))
    model.fit(X_poly, y)
    ypred = model.predict(Xtest_poly)
    # Print model parameters.
    print("c = ", C)
    print(model.intercept_)
    print(model.coef_)
    # Plot model predictions.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Xtest[:, 0], Xtest[:, 1], ypred, color="red")
    ax.plot_trisurf(X[:, 0], X[:, 1], y)
    ax.set_xlabel('X1', color='blue', size=15)
    ax.set_ylabel('X2', color='blue', size=15)
    ax.set_zlabel('Y', color='blue', size=15)
    ax.set_title("Ridge Predictions over [-5, 5] Grid, C = " + str(C))
plt.show()
