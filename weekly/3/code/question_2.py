from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics
from sklearn.linear_model import Ridge

# Read in data
df = pd.read_csv("week3.csv", comment='#')
X1 = df.iloc[:, 0]
X2 = df.iloc[:, 1]
X = np.column_stack((X1, X2))
y = np.array(df.iloc[:, 2])

# Get polynomials for features
poly = PolynomialFeatures(5)
X_poly = poly.fit_transform(X)

# For each fold, keep track of means, variances and folds
means = []
variances = []
folds = []
C = 1
all_folds = [2, 5, 10, 25, 50, 100]
for f in all_folds:
    kf = KFold(n_splits=f)
    mse = []
    for train, test in kf.split(X_poly):
        model = linear_model.Lasso(alpha=(1/C))
        model.fit(X_poly[train], y[train])
        ypred = model.predict(X_poly[test])
        mse.append(mean_squared_error(y[test], ypred))
    means.append(statistics.mean(mse))
    variances.append(statistics.variance(mse))
    folds.append(f)

# Plot the means and variances
fig = plt.figure()
ax = fig.add_subplot(111)
ax.errorbar(folds, means, label='Mean',
            yerr=all_folds, uplims=True, lolims=True)
ax.errorbar(folds, variances, label='Variance',
            yerr=all_folds, uplims=True, lolims=True)
ax.set_xlabel("number of folds")
ax.set_title(
    "Variance and Mean against number of folds used. Lasso Model with C = 1")
ax.legend(loc='lower right')

means = []
variances = []
Cs = []
num_folds = 10
C_range = np.arange(1, 75, 5)
for C in C_range:
    kf = KFold(n_splits=num_folds)
    mse = []
    for train, test in kf.split(X_poly):
        model = linear_model.Lasso(alpha=(1/C))
        model.fit(X_poly[train], y[train])
        ypred = model.predict(X_poly[test])
        mse.append(mean_squared_error(y[test], ypred))
    means.append(statistics.mean(mse))
    variances.append(statistics.variance(mse))

# Plot the means and variances
fig = plt.figure()
ax = fig.add_subplot(111)
ax.errorbar(C_range, means, label='Mean',
            yerr=C_range, uplims=True, lolims=True)
ax.errorbar(C_range, variances, label='Variance',
            yerr=C_range, uplims=True, lolims=True)
ax.set_xlabel("mean/variance")
ax.set_xlabel("C")
ax.set_title(
    "Variance and Mean against C value. Lasso Model with folds=10")
ax.legend(loc='lower right')

means = []
variances = []
Cs = []
num_folds = 10
for C in C_range:
    kf = KFold(n_splits=num_folds)
    mse = []
    for train, test in kf.split(X_poly):
        model = Ridge(alpha=(1/(2 * C)))
        model.fit(X_poly[train], y[train])
        ypred = model.predict(X_poly[test])
        mse.append(mean_squared_error(y[test], ypred))
    means.append(statistics.mean(mse))
    variances.append(statistics.variance(mse))

# Plot the means and variances
fig = plt.figure()
ax = fig.add_subplot(111)
ax.errorbar(C_range, means, label='Mean',
            yerr=C_range, uplims=True, lolims=True)
ax.errorbar(C_range, variances, label='Variance',
            yerr=C_range, uplims=True, lolims=True)
ax.set_xlabel("mean/variance")
ax.set_xlabel("C")
ax.set_title(
    "Variance and Mean against C value. Ridge Model with folds=10")
ax.legend(loc='lower right')


plt.show()
