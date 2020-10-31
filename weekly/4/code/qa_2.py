from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read in data
df = pd.read_csv("week4_2.csv", comment='#')
X1 = df.iloc[:, 0]
X2 = df.iloc[:, 1]
X = np.column_stack((X1, X2))
y = np.array(df.iloc[:, 2])

# Plotting the data alone
fig = plt.figure()
ax = fig.add_subplot(111)
neg = plt.scatter(X1[y < 0], X2[y < 0], color='red', marker=".")
pos = plt.scatter(X1[y > 0], X2[y > 0], color='blue', marker=".")
ax.set_ylabel("X2", fontsize=20)
ax.set_xlabel("X1", fontsize=20)
ax.set_title(
    "Plot of the provided data, colour represents the y output", fontsize=20)
plt.rc('font', size=20)
plt.legend((neg, pos), ["positive", "negative"],
           scatterpoints=1,
           loc='lower right',
           ncol=2,
           fontsize=15)

# Compute a simple logistic regression with varying degrees of polynomials.
degrees = range(1, 10)
scores = []
temp = []
for degree in degrees:
    # Generate new features
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)
    # Train the model
    model = LogisticRegression(penalty='l2').fit(X_poly, y)
    # Keep the score of the model
    scores.append(model.score(X_poly, y))
    temp.append(mean_squared_error(y, model.predict(X_poly)))


# Plot the scores for each model.
fig = plt.figure()
plt.rc('font', size=20)
ax = fig.add_subplot(111)
ax.errorbar(degrees, scores, label='Mean accuracy', yerr=temp)
ax.set_ylabel("Score (mean accuracy) of the model")
ax.set_xlabel("Maximum degree of polynomials generated")
ax.set_title(
    "Cross-validation of the degree of polynomials generated for a Logistic Regression model")
ax.legend(loc='lower right')

# Use interesting degrees to plot the predictions
for degree in [1]:
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)
    model = LogisticRegression(solver='liblinear', penalty='l2').fit(X_poly, y)
    y_pred = model.predict(X_poly)
    # Plt the predictions
    fig = plt.figure()
    ax = fig.add_subplot(111)
    neg = plt.scatter(X1[y_pred < 0], X2[y_pred < 0], color='red', marker=".")
    pos = plt.scatter(X1[y_pred > 0], X2[y_pred > 0], color='blue', marker=".")
    ax.set_ylabel("X2", fontsize=20)
    ax.set_xlabel("X1", fontsize=20)
    ax.set_title(
        "Plot of the predictions with polynomial features up to " + str(degree), fontsize=20)
    plt.rc('font', size=20)
    plt.legend((neg, pos), ["positive", "negative"],
               scatterpoints=1,
               loc='lower right',
               ncol=2,
               fontsize=15)


# cross validate C value
C_values = range(1, 30, 1)
scores = []
temp = []
for C in C_values:
    poly = PolynomialFeatures(1)
    X_poly = poly.fit_transform(X)
    model = LogisticRegression(C=C).fit(X_poly, y)
    scores.append(model.score(X_poly, y))
    temp.append(mean_squared_error(y, model.predict(X_poly)))

# Plot the scores for each model.
fig = plt.figure()
plt.rc('font', size=20)
ax = fig.add_subplot(111)
ax.errorbar(C_values, scores, label='Mean accuracy', yerr=temp)
ax.set_ylabel("Score (mean accuracy) of the model")
ax.set_xlabel("C")
ax.set_title(
    "Cross-validation of the l2 penalty term C for a Logistic Regression model")
ax.legend(loc='lower right')

# Plot predictions with some C values
C_values = [1]
scores = []
for C in C_values:
    poly = PolynomialFeatures(1)
    X_poly = poly.fit_transform(X)
    model = LogisticRegression(
        solver='liblinear', penalty='l2', C=C).fit(X_poly, y)
    # Plt the predictions
    fig = plt.figure()
    ax = fig.add_subplot(111)
    neg = plt.scatter(X1[y_pred < 0], X2[y_pred < 0], color='red', marker=".")
    pos = plt.scatter(X1[y_pred > 0], X2[y_pred > 0], color='blue', marker=".")
    ax.set_ylabel("X2", fontsize=20)
    ax.set_xlabel("X1", fontsize=20)
    ax.set_title(
        "Plot of the predictions with C =" + str(C), fontsize=20)
    plt.rc('font', size=20)
    plt.legend((neg, pos), ["positive", "negative"],
               scatterpoints=1,
               loc='lower right',
               ncol=2,
               fontsize=15)
plt.show()
