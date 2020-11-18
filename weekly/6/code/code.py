from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read in data
df = pd.read_csv("dummy.csv")
x = np.array([0, 1, 0]).reshape(-1, 1)
y = np.array([-1, 0, 1])


def gaussian_kernel(distances):
    weights = np.exp(-V * (distances**2))
    return weights/np.sum(weights)


# Generate [-3,3] for predictions.
x_pred = np.linspace(-3, 3).reshape(-1, 1)

k = 3
V_range = [0, 1, 5, 10, 25]
predictions = []
for V in V_range:
    model = KNeighborsClassifier(
        n_neighbors=k, weights=gaussian_kernel).fit(x, y)
    predictions.append(model.predict(x_pred))


# Plot predictions
for i in range(len(V_range)):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    neg = plt.scatter(x_pred, predictions[i], color='black', marker=".")
    ax.set_ylabel("Y predictions", fontsize=20)
    ax.set_xlabel("X", fontsize=20)
    ax.set_title(
        "Predictions with a kNN model using a gaussian kernel with V =" + str(V_range[i]), fontsize=20)
    plt.rc('font', size=20)

plt.xticks(np.arange(min(x_pred), max(x_pred)+1, 0.5))
plt.show()
