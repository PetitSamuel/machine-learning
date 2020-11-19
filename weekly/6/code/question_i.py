from sklearn.kernel_ridge import KernelRidge
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
import numpy as np

# Generate our data points
m = 3
Xtrain = np.linspace(-1.0, 1.0, num=m)
ytrain = np.sign(np.array([0, 1, 0]))
Xtrain = Xtrain.reshape(-1, 1)


# Gaussian kernel uses the V global variable
def gaussian_kernel(distances):
    weights = np.exp(-V * (distances ** 2))
    return weights/np.sum(weights)


# Generate the [-3, 3] line for testing our predictions
Xtest = np.linspace(-3.0, 3.0, num=100).reshape(-1, 1)

k = 3
V_range = [0, 1, 5, 10, 25]
predictions = []
# Train a bunch of KNeighborsRegressor models with a set of V values
for V in V_range:
    # Note that V is a global variable for the Gaussian kernel method
    model = KNeighborsRegressor(
        n_neighbors=k, weights=gaussian_kernel).fit(Xtrain, ytrain)
    predictions.append(model.predict(Xtest))

# Plot predictions
for i in range(len(V_range)):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(Xtrain, ytrain, color='red', marker='+', linewidths=3)
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

# Train a bunch of KernelRidge models using a range of V (gamma) and C values
V_range = [0, 1, 5, 10, 25]
C_range = [0.1, 1, 1000]
for V in V_range:
    for C in C_range:
        model = KernelRidge(alpha=1.0/C, kernel='rbf',
                            gamma=V).fit(Xtrain, ytrain)
        ypred = model.predict(Xtest)
        # plot predictions
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(Xtrain, ytrain, color='red', marker='+', linewidths=3)
        ax.plot(Xtest, ypred, color='green')
        ax.set_ylabel("output Y", fontsize=20)
        ax.set_xlabel("input X", fontsize=20)
        ax.set_title(
            "KernelRidge | gamma =" + str(V) +
            " | C = " + str(C) + " | theta = "
            + str(model.dual_coef_), fontsize=15)
        fig.legend(["KernelRidge", "train"],
                   scatterpoints=1,
                   loc='lower right',
                   ncol=2,
                   fontsize=15)

plt.rcParams['figure.constrained_layout.use'] = True
plt.rc('font', size=18)
plt.show()
