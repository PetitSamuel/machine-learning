from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from util import get_data
import numpy as np

# Read in data
X1, X2, X, y, ytrain = get_data()

# Train Linear SVC for C=0.001, 1 and 1000
for C_val in [0.001, 1, 1000]:
    # max iter lower than 100 000 makes C=1000 not converge
    model = LinearSVC(C=C_val, max_iter=100000).fit(X, ytrain)
    print("C, intercept, slope", C_val, model.intercept_, model.coef_)

# Plot one of the 3 models (change C value to plot the other ones)
model = LinearSVC(C=1000, max_iter=100000).fit(X, ytrain)
line_bias = model.intercept_
line_w = model.coef_.T
points_y = [(line_w[0]*x+line_bias)/(-1*line_w[1]) for x in X1]

# Predict values
ypred = np.sign(model.predict(X))

# Plot the preditions
plt.rc('font', size=20)
pos = plt.scatter(X1[y > 0], X2[y > 0],
                  color='black', marker=".", linewidths=10)
neg = plt.scatter(X1[y < 0], X2[y < 0],
                  color='green', marker=".", linewidths=10)
pos_pred = plt.scatter(X1[ypred > 0], X2[ypred > 0],
                       color='red', marker="+", linewidths=7)
neg_pred = plt.scatter(X1[ypred < 0], X2[ypred < 0],
                       color='blue', marker="+", linewidths=7)
plt.rcParams['figure.constrained_layout.use'] = True
plt.xlabel("X1")
plt.ylabel("X2")
plt.plot(X1, points_y, color='black', linewidth=3)
plt.legend((neg, pos, pos_pred, neg_pred), ["positive", "negative", "positive predictions", "negative predictions"],
           scatterpoints=1,
           loc='upper right',
           ncol=2,
           fontsize=18)
plt.show()
