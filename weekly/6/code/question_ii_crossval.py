from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error
from sklearn.kernel_ridge import KernelRidge
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import pandas as pd
import statistics
from sklearn import metrics

df = pd.read_csv("week6.csv", comment='#')
Xtrain = np.array(df.iloc[:, 0])
ytrain = df.iloc[:, 1]
Xtrain = Xtrain.reshape(-1, 1)


def gaussian_kernel(distances):
    weights = np.exp(-V * (distances ** 2))
    return weights/np.sum(weights)


Xtest = np.linspace(min(Xtrain) * 3, max(Xtrain) * 3, num=300).reshape(-1, 1)
k = len(ytrain)
V_range = range(0, 12, 1)
scores = []
temp = []
for V in V_range:
    model = KNeighborsRegressor(
        n_neighbors=k, weights=gaussian_kernel).fit(Xtrain, ytrain)
    scores.append(model.score(Xtrain, ytrain))
    temp.append(mean_squared_error(ytrain, model.predict(Xtrain)))

# plot the CV
fig = plt.figure()
plt.rc('font', size=20)
ax = fig.add_subplot(111)
ax.errorbar(V_range, scores, label='Mean accuracy', yerr=temp)
ax.set_ylabel("Score (mean accuracy) of the model")
ax.set_xlabel("V (gamma)")
ax.set_title(
    "Gamma Cross-validation | kNN model")
ax.legend(loc='lower right')
plt.rcParams['figure.constrained_layout.use'] = True
plt.rc('font', size=18)

V_range = range(0, 30, 2)
scores = []
temp = []
for V in V_range:
    model = KernelRidge(kernel='rbf', gamma=V).fit(Xtrain, ytrain)
    scores.append(model.score(Xtrain, ytrain))
    temp.append(mean_squared_error(ytrain, model.predict(Xtrain)))

fig = plt.figure()
plt.rc('font', size=20)
ax = fig.add_subplot(111)
ax.errorbar(V_range, scores, label='Mean accuracy', yerr=temp)
ax.set_ylabel("Score (mean accuracy) of the model")
ax.set_xlabel("V (gamma)")
ax.set_title(
    "Gamma Cross-validation | KernelRidge model")
ax.legend(loc='lower right')
plt.rcParams['figure.constrained_layout.use'] = True
plt.rc('font', size=18)


V = 4
C_range = np.linspace(0.1, 15, num=10)
scores = []
temp = []
for C in C_range:
    model = KernelRidge(kernel='rbf', gamma=V, alpha=1.0/C).fit(Xtrain, ytrain)
    scores.append(model.score(Xtrain, ytrain))
    temp.append(mean_squared_error(ytrain, model.predict(Xtrain)))

fig = plt.figure()
plt.rc('font', size=20)
ax = fig.add_subplot(111)
ax.errorbar(C_range, scores, label='Mean accuracy', yerr=temp)
ax.set_ylabel("Score (mean accuracy) of the model")
ax.set_xlabel("C")
ax.set_title(
    "C Cross-validation | KernelRidge model using V = " + str(V))
ax.legend(loc='lower right')
plt.rcParams['figure.constrained_layout.use'] = True
plt.rc('font', size=18)

# Training optimised models
V = 8
k = len(ytrain)
knn = KNeighborsRegressor(
    n_neighbors=k, weights=gaussian_kernel).fit(Xtrain, ytrain)
V = 4
C = 3
kridge = KernelRidge(kernel='rbf', gamma=V, alpha=1.0/C).fit(Xtrain, ytrain)

dummy = DummyRegressor().fit(Xtrain, ytrain)

# TODO : plot predictions for both
# compare scorewith baseline model
fig = plt.figure()
ax = fig.add_subplot(111)
red = ax.scatter(Xtrain, ytrain, color='red', marker='+')
knn_plot = ax.plot(Xtest, knn.predict(Xtest), color='green')
kridge_plot = ax.plot(Xtest, kridge.predict(Xtest), color='blue')
base = ax.plot(Xtest, dummy.predict(Xtest), color='orange', linestyle='--')
ax.set_ylabel("output Y", fontsize=20)
ax.set_xlabel("input X", fontsize=20)
fig.legend(["kNN", "KernelRidge", "baseline", "train"],
           scatterpoints=1,
           loc='right',
           ncol=2,
           fontsize=15)
ax.set_title(
    "kNN & KernelRidge Predictions", fontsize=20)


# Compute percentage of accuracy for each predictions
knn_accuracy = knn.score(Xtrain, ytrain)
kridge_accuracy = kridge.score(Xtrain, ytrain)
baseline_accuracy = dummy.score(Xtrain, ytrain)


# Print outputs
print("base model accuracy score: ", baseline_accuracy,
      " - knn model accuracy score: ", knn_accuracy,
      " - kridge accuracy: ", kridge_accuracy)

plt.show()
