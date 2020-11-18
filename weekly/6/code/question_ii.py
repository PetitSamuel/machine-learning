from sklearn.kernel_ridge import KernelRidge
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import pandas as pd

df = pd.read_csv("week6.csv", comment='#')
Xtrain = np.array(df.iloc[:, 0])
ytrain = df.iloc[:, 1]
Xtrain = Xtrain.reshape(-1, 1)


def gaussian_kernel(distances):
    weights = np.exp(-V * (distances ** 2))
    return weights/np.sum(weights)


Xtest = np.linspace(min(Xtrain) * 3, max(Xtrain) * 3, num=300).reshape(-1, 1)
k = len(ytrain)
V_range = [0, 1, 5, 10, 25]
predictions = []
for V in V_range:
    model = KNeighborsRegressor(
        n_neighbors=k, weights=gaussian_kernel).fit(Xtrain, ytrain)
    predictions.append(model.predict(Xtest))

# Plot predictions
for i in range(len(V_range)):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    red = plt.scatter(Xtrain, ytrain, color='red', marker='+')
    preds = plt.plot(Xtest, predictions[i], color='green')
    ax.set_ylabel("Y predictions", fontsize=20)
    ax.set_xlabel("X", fontsize=20)
    ax.set_title(
        "kNN | gamma =" + str(V_range[i]), fontsize=20)

V_range = [0, 1, 5, 10, 25]
predictions = []
for V in V_range:
    model = KernelRidge(kernel='rbf',
                        gamma=V).fit(Xtrain, ytrain)
    ypred = model.predict(Xtest)
    predictions.append(ypred)
    # plot predictions
    fig = plt.figure()
    ax = fig.add_subplot(111)
    red = plt.scatter(Xtrain, ytrain, color='red', marker='+')
    preds = plt.plot(Xtest, ypred, color='green')
    ax.set_ylabel("output Y", fontsize=20)
    ax.set_xlabel("input X", fontsize=20)
    ax.set_title(
        "KernelRidge | gamma =" + str(V), fontsize=20)

plt.rcParams['figure.constrained_layout.use'] = True
plt.rc('font', size=18)
plt.show()
