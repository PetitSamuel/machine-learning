from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

# Read in data
df = pd.read_csv("week4_2.csv", comment='#')
X1 = df.iloc[:, 0]
X2 = df.iloc[:, 1]
X = np.column_stack((X1, X2))
y = np.array(df.iloc[:, 2])

neighbors_range = range(1, 25, 1)
scores = []
for n in neighbors_range:
    model = KNeighborsClassifier(
        n_neighbors=n, weights='uniform').fit(X, y)
    scores.append(model.score(X, y))

# Plot the scores for each model.
fig = plt.figure()
plt.rc('font', size=20)
ax = fig.add_subplot(111)
ax.errorbar(neighbors_range, scores, label='Mean accuracy',
            yerr=neighbors_range, uplims=True, lolims=True)
ax.set_ylabel("Score (mean accuracy) of the model")
ax.set_xlabel("neighbors range")
ax.set_title(
    "Cross-validation of the neighbors range for a kNN model")
ax.legend(loc='lower right')

# Plot predictions with some k values
K_values = [1, 10, 60]
scores = []
for K in K_values:
    model = KNeighborsClassifier(
        n_neighbors=K, weights='uniform').fit(X, y)
    y_pred = model.predict(X)
    # Plt the predictions
    fig = plt.figure()
    ax = fig.add_subplot(111)
    neg = plt.scatter(X1[y_pred < 0], X2[y_pred < 0], color='red', marker=".")
    pos = plt.scatter(X1[y_pred > 0], X2[y_pred > 0], color='blue', marker=".")
    ax.set_ylabel("X2", fontsize=20)
    ax.set_xlabel("X1", fontsize=20)
    ax.set_title(
        "Plot of the predictions with k =" + str(K), fontsize=20)
    plt.rc('font', size=20)
    plt.legend((neg, pos), ["positive", "negative"],
               scatterpoints=1,
               loc='lower right',
               ncol=2,
               fontsize=15)


# Compute a simple knn model with varying degrees of polynomials.
degrees = range(1, 100, 5)
scores = []
K = 3
for degree in degrees:
    poly = PolynomialFeatures(degree)
    X_poly = poly.fit_transform(X)
    model = KNeighborsClassifier(
        n_neighbors=K, weights='uniform').fit(X_poly, y)
    scores.append(model.score(X_poly, y))
print(scores)
# Plot the scores for each model.
fig = plt.figure()
plt.rc('font', size=20)
ax = fig.add_subplot(111)
ax.errorbar(degrees, scores, label='Mean accuracy',
            yerr=degrees, uplims=True, lolims=True)
ax.set_ylabel("Score (mean accuracy) of the model")
ax.set_xlabel("Maximum degree of polynomials generated")
ax.set_title(
    "Cross-validation of the degree of polynomials generated for a kNN model")
ax.legend(loc='lower right')

plt.show()
