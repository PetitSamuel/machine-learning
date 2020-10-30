from sklearn.metrics import roc_curve
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import statistics
import random
from sklearn.dummy import DummyClassifier

# Read in data
df = pd.read_csv("week4.csv", comment='#')
X1 = df.iloc[:, 0]
X2 = df.iloc[:, 1]
X = np.column_stack((X1, X2))
y = np.array(df.iloc[:, 2])
# Compute polynomials
poly = PolynomialFeatures(4)
X_poly = poly.fit_transform(X)

# Train our models
knnModel = KNeighborsClassifier(
    n_neighbors=1, weights='uniform').fit(X, y)
knn_pred = knnModel.predict(X)
tn, fp, fn, tp = confusion_matrix(y, knn_pred).ravel()
print(tn, fp, fn, tp)

lgModel = LogisticRegression(solver='liblinear', C=18).fit(X_poly, y)
lg_pred = lgModel.predict(X_poly)
tn, fp, fn, tp = confusion_matrix(y, lg_pred).ravel()
print(tn, fp, fn, tp)

randomModel = DummyClassifier(strategy="uniform")
mostFreqModel = DummyClassifier(strategy="most_frequent")

randomModel.fit(X, y)
mostFreqModel.fit(X, y)

# Random
rqndom_pred = randomModel.predict(X)
tn, fp, fn, tp = confusion_matrix(y, rqndom_pred).ravel()
print(tn, fp, fn, tp)

# Uniform
most_freq_pred = mostFreqModel.predict(X)
tn, fp, fn, tp = confusion_matrix(y, most_freq_pred).ravel()
print(tn, fp, fn, tp)

# knn roc
fpr, tpr, _ = roc_curve(y, lgModel.decision_function(X_poly))
knn_proba = knnModel.predict_proba(X)
knn_fpr, knn_tpr, thresh = roc_curve(y, knn_proba[:, 1])

# most freq val roc
most_freq_proba = mostFreqModel.predict_proba(X)
most_freq_fpr, most_freq_tpr, thresh = roc_curve(y, most_freq_proba[:, 1])

# random roc
rand_proba = randomModel.predict_proba(X)
rand_fpr, rand_tpr, thresh = roc_curve(y, rand_proba[:, 1])


fig = plt.figure()
plt.rc('font', size=20)
ax = fig.add_subplot(111)
ax.plot(fpr, tpr, color='cyan')
ax.plot(knn_fpr, knn_tpr, color='orange')
ax.plot(most_freq_fpr, most_freq_tpr, color='blue')
ax.plot(rand_fpr, rand_tpr, color='red')
ax.plot([0, 1], [0, 1], color='green', linestyle='--')
ax.set_ylabel('True positive rate')
ax.set_xlabel('False positive rate')
ax.set_title(
    "Cross-validation of the neighbors range for a kNN model")
plt.legend(["Logistic Regression", "kNN",
            "Most Frequent value", "Random"])
plt.show()
