from sklearn.kernel_ridge import KernelRidge
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import pandas as pd

# Read and format data from our dataset
df = pd.read_csv("week6.csv", comment='#')
Xtrain = np.array(df.iloc[:, 0])
ytrain = df.iloc[:, 1]
Xtrain = Xtrain.reshape(-1, 1)


def gaussian_kernel(distances):
    # Gaussian kernel function, V (gamma) is a global variable
    weights = np.exp(-V * (distances ** 2))
    return weights/np.sum(weights)


# Generate the Xtest line. Over the range [3 * min, 3 * min]
Xtest = np.linspace(min(Xtrain) * 3, max(Xtrain) * 3, num=300).reshape(-1, 1)

# Train KNeighborsRegressor models with a range of V (gamma values)
k = len(ytrain)
V_range = [0, 1, 5, 10, 25]
predictions = []
for V in V_range:
    model = KNeighborsRegressor(
        n_neighbors=k, weights=gaussian_kernel).fit(Xtrain, ytrain)
    predictions.append(model.predict(Xtest))

# Plot predictions for the kNN model
for i in range(len(V_range)):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(Xtrain, ytrain, color='red', marker='+')
    ax.plot(Xtest, predictions[i], color='green')
    ax.set_ylabel("Y output", fontsize=15)
    ax.set_xlabel("X", fontsize=15)
    ax.set_title(
        "kNN | gamma =" + str(V_range[i]), fontsize=18)
    fig.legend(["kNN", "train"],
               scatterpoints=1,
               loc='lower right',
               ncol=2,
               fontsize=15)

# Train some KernelRidge models using a range of V (gamma) values
# & the default alpha value (1).
V_range = [0, 1, 5, 10, 25]
for V in V_range:
    model = KernelRidge(kernel='rbf',
                        gamma=V).fit(Xtrain, ytrain)
    ypred = model.predict(Xtest)
    # plot predictions
    fig = plt.figure()
    ax = fig.add_subplot(111)
    red = plt.scatter(Xtrain, ytrain, color='red', marker='+')
    preds = plt.plot(Xtest, ypred, color='green')
    ax.set_ylabel("output Y", fontsize=20)
    ax.set_xlabel("input X", fontsize=20)
    ax.set_title(
        "KernelRidge | gamma =" + str(V), fontsize=20)
    fig.legend(["KernelRidge", "train"],
               scatterpoints=1,
               loc='lower right',
               ncol=2,
               fontsize=15)

plt.rcParams['figure.constrained_layout.use'] = True
plt.rc('font', size=18)
plt.show()
