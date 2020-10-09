import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from util import get_data
import numpy as np

# Read in data
X1, X2, X, y, ytrain = get_data()


# Grab square of the features
X1 = X1 + X1 ** 2
X2 = X2 + X2 ** 2
X1 = np.array(X1).reshape(-1, 1)
X2 = np.array(X2).reshape(-1, 1)
X = np.column_stack((X1, X2))


# Train the model
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
pos = plt.scatter(X1[y > 0], X2[y > 0],
                  color='green', marker="+")
pos = plt.scatter(X1[y < 0], X2[y < 0],
                  color='blue', marker=".")

plt.rcParams['figure.constrained_layout.use'] = True
plt.show()
