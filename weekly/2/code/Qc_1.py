import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from util import get_data
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

# Read in data
X1, X2, X, y, ytrain = get_data()

X1 = np.array(X1)
X2 = np.array(X2)
# Square both features
X1_square = X1.reshape(-1, 1)
X2_square = X2.reshape(-1, 1)
X1_square = np.square(X1)
X2_square = np.square(X2)

X = np.column_stack((X1, X2, X1_square, X2_square))

model = LogisticRegression(solver='lbfgs', penalty="none")
model.fit(X, ytrain)

# Print model values
print("Logistic Regression Classifier")
print("intercept:", model.intercept_)
print("slope:", model.coef_)

# Predict values
ypred = np.sign(model.predict(X))

# Plot the preditions
plt.rc('font', size=20)
pos = plt.scatter(X1[ypred > 0], X2[ypred > 0],
                  color='red', marker="+")
pos = plt.scatter(X1[ypred < 0], X2[ypred < 0],
                  color='blue', marker=".")

plt.rcParams['figure.constrained_layout.use'] = True
plt.show()
