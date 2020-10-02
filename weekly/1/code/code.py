import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from util import get_median, scaling_factor, cost_function, gradient_descent
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing

# Read data from CSV
df = pd.read_csv("dataset.csv", comment='#')
# Extract columns from data
x = np.array(df.iloc[:, 0])
x = x.reshape(-1, 1)
y = np.array(df.iloc[:, 1])
y = y.reshape(-1, 1)
# Convert x to floats
x = x.astype(float)

# normalise x data
median = get_median(x)
scaling = scaling_factor(x)
length = len(x)
for i in range(length):
    value = x[i][0]
    normalised = (value - median) / scaling
    x[i][0] = normalised

# normalise y data
median = get_median(y)
scaling = scaling_factor(y)
length = len(y)
for i in range(length):
    value = y[i][0]
    normalised = (value - median) / scaling
    y[i][0] = normalised

# For Q1 use 200 iterations and 0.1 learning rates
nb_iterations = 200
coefs = gradient_descent(x, y, 0.1, nb_iterations)

print("Q1, coefficients: ", coefs["coef_1"], coefs["coef_2"])

# Q2 i: use gradient descent on learning rates 0.1, 0.01, 0.001.
nb_iterations = 1000
xaxis = range(nb_iterations)
coefs_001 = gradient_descent(x, y, 0.001, nb_iterations)
coefs_01 = gradient_descent(x, y, 0.01, nb_iterations)
coefs_1 = gradient_descent(x, y, 0.1, nb_iterations)

# Plot the cost function evolution
plt.rc('font', size=30)
plt.rcParams['figure.constrained_layout.use'] = True
plt.plot(xaxis, coefs_001["deltas"], color='green', linewidth=3)
plt.plot(xaxis, coefs_01["deltas"], color='blue', linewidth=3)
plt.plot(xaxis, coefs_1["deltas"], color='orange', linewidth=3)
plt.xlabel("iteration")
plt.ylabel("cost function")
plt.legend(["learning rate: 0.001", "learning rate: 0.01", "learning rate: 0.1"])
# Plot is shown at the end of program executiong as it blocks execution otherwise.

# Q2 ii: use learning rate of 0.5
coefs_05 = gradient_descent(x, y, 0.5, nb_iterations)

print("Q2 ii parameter values after training: ",
      coefs_05["coef_1"], coefs_05["coef_2"])

print("Q2 iii cost function for trained model: ", coefs_05["deltas"][199])
print("Q2 iii cost function for baseline model (using mean): ",
      cost_function(x, y, get_median(y), 0))

# Q3, reload values from dataset to start over
# Read data from CSV
df = pd.read_csv("dataset.csv", comment='#')
# Extract columns from data
x = np.array(df.iloc[:, 0])
x = x.reshape(-1, 1)
y = np.array(df.iloc[:, 1])
y = y.reshape(-1, 1)
# Convert x to floats
x = x.astype(float)

# Standardise data x and y.
x = preprocessing.scale(x)
y = preprocessing.scale(y)

# Separate features into training / test sets.
Xtrain = x[:-20]
Xtest = x[-20:]

# Separate targets as well.
Ytrain = y[:-20]
Ytest = y[-20:]

# Train the model
regr = LinearRegression()
regr.fit(Xtrain, Ytrain)

# Predict using the test set
y_pred = regr.predict(Xtest)

# Print coefficients and cost function output.
print("Q3")
print('Sklearn model Coefficients: ', regr.coef_, regr.intercept_)
print('Sklearn model Cost function: ',
      cost_function(x, y, regr.intercept_, regr.coef_))


# Show plot for Q2 i.
plt.show()
