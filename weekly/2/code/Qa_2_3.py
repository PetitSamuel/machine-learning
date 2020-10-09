from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from util import get_data
import numpy as np

# Read in data
X1, X2, X, y, ytrain = get_data()

# Train the model
model = LogisticRegression(solver='lbfgs', penalty="none")
model.fit(X, ytrain)

# Print model values
print("Logistic Regression Classifier")
print("intercept:", model.intercept_)
print("slope:", model.coef_)

# Predict values
ypred = np.sign(model.predict(X))

# Extract the slope to display from model coefs
line_bias = model.intercept_
line_w = model.coef_.T
points_y = [(line_w[0]*x+line_bias)/(-1*line_w[1]) for x in X1]

# Plot the preditions
plt.rc('font', size=20)
pos = plt.scatter(X1[ypred > 0], X2[ypred > 0],
                  color='black', marker="+")
neg = plt.scatter(X1[ypred < 0], X2[ypred < 0],
                  color='orange', marker=".")
plt.rcParams['figure.constrained_layout.use'] = True
plt.xlabel("X1 (first feature)")
plt.ylabel("X2 (second feature)")
plt.plot(X1, points_y)
plt.legend((neg, pos), ["positive", "negative"],
           scatterpoints=1,
           loc='lower right',
           ncol=2,
           fontsize=12)
plt.show()
