import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from util import get_data
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics
import statistics

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

# Train our model
model = LogisticRegression(solver='lbfgs', penalty="none")
model.fit(X, ytrain)

# Print model values
print("Logistic Regression Classifier")
print("intercept:", model.intercept_)
print("slope:", model.coef_)

# Predict values
ypred = np.sign(model.predict(X))

# Create a baseline model which predicts the most common value
# Find the most common value by getting the sign of the mean.
baseline_model = np.sign(statistics.mean(y))
# generate an array of same size as y filled with out baseline predictions
y_baseline_pred = np.full((len(y), 1), baseline_model)

# Compute percentage of accurace for each predictions
accuracy_model = metrics.accuracy_score(y, ypred)
accuracy_base = metrics.accuracy_score(y, y_baseline_pred)
# Print outputs
print("base model accuracy score: ", accuracy_base,
      " - trained model accuracy score: ", accuracy_model)

# Plot the preditions
plt.rc('font', size=20)
pos = plt.scatter(X1[y > 0], X2[y > 0],
                  color='black', marker=".")
neg = plt.scatter(X1[y < 0], X2[y < 0],
                  color='green', marker=".")
pos_pred = plt.scatter(X1[ypred > 0], X2[ypred > 0],
                       color='red', marker="+")
neg_pred = plt.scatter(X1[ypred < 0], X2[ypred < 0],
                       color='blue', marker="+")
plt.rcParams['figure.constrained_layout.use'] = True
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend((neg, pos, pos_pred, neg_pred), ["positive", "negative", "positive predictions", "negative predictions"],
           scatterpoints=1,
           loc='upper right',
           ncol=2,
           fontsize=18)
plt.show()
