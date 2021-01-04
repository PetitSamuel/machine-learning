from sklearn.utils import shuffle
from nltk.stem import PorterStemmer
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, precision_recall_fscore_support, roc_auc_score
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
import json_lines
import numpy as np
import matplotlib.pyplot as plt
plt.rcdefaults()

X = []
y = []
z = []

with open('final_data', 'rb') as f:
    for item in json_lines.reader(f):
        X.append(item['text'])
        y.append(item['voted_up'])
        z.append(item['early_access'])

# Dataset has first 5000 reviews positive and next 5000 negative, shuffle values in unison
X, y, z = shuffle(X, y, z, random_state=0)

# Take a sample 500 reviews such as to avoid too high computing times when testing approaches
X = X[:1000]
y = y[:1000]
z = z[:1000]

stemmer = PorterStemmer()
analyzer = CountVectorizer().build_analyzer()


# Returns the stems from words
def stemmed_words(doc):
    return (stemmer.stem(w) for w in analyzer(doc))


stem_vectorizer = CountVectorizer(analyzer=stemmed_words, ngram_range=(2, 2))
tokens = stem_vectorizer.fit_transform(X)

# Using stems reduces number of words from 177883 to 44853
print(len(stem_vectorizer.get_feature_names()))

X = np.array(tokens.toarray())
y = np.array(y)
# use this line instead of the above one for early access models
# y = np.array(z)

model_names = []
model_accuracies = []

# Decision Tree Model
accuracies = []
kf = KFold(n_splits=5)
for train, test in kf.split(X):
    model = DecisionTreeClassifier().fit(X[train], y[train])
    ypred = model.predict(X[test])
    accuracies.append(accuracy_score(y[test], ypred))
avg_accuracy = sum(accuracies) / len(accuracies)
print("Decision Tree Model average accuracy: ", avg_accuracy)

model_names.append("DecisionTree")
model_accuracies.append(avg_accuracy)

# Dummy Model
accuracies = []
kf = KFold(n_splits=5)
for train, test in kf.split(X):
    model = DummyClassifier(strategy="most_frequent").fit(X[train], y[train])
    ypred = model.predict(X[test])
    accuracies.append(accuracy_score(y[test], ypred))
avg_accuracy = sum(accuracies) / len(accuracies)
print("Dummy Model average accuracy: ", avg_accuracy)

model_names.append("Dummy")
model_accuracies.append(avg_accuracy)

# RidgeClassifier Model
best_ridge_accuracy = (-1, -1)
mean_error = []
std_error = []
alpha_range = [0.0001, 0.01, 0.1, 0, 1, 5, 10, 20, 30, 40, 50, 75, 100]
for alpha in alpha_range:
    accuracies = []
    kf = KFold(n_splits=5)
    for train, test in kf.split(X):
        model = RidgeClassifier(alpha=alpha).fit(X[train], y[train])
        ypred = model.predict(X[test])
        accuracies.append(accuracy_score(y[test], ypred))
    avg_accuracy = sum(accuracies) / len(accuracies)
    print("Ridge Model, alpha = ", alpha,
          " - average accuracy: ", avg_accuracy)
    mean_error.append(np.array(accuracies).mean())
    std_error.append(np.array(accuracies).std())
    if avg_accuracy > best_ridge_accuracy[0]:
        best_ridge_accuracy = (avg_accuracy, alpha)

# plot the CV
fig = plt.figure()
plt.rc('font', size=20)
ax = fig.add_subplot(111)
plt.errorbar(alpha_range, mean_error, yerr=std_error)
ax.set_ylabel("Mean accuracy")
ax.set_xlabel("Alpha")
ax.set_title("Alpha Cross-validation | Ridge model")

model_names.append("Ridge (alpha=" + str(best_ridge_accuracy[1]) + ")")
model_accuracies.append(best_ridge_accuracy[0])

# Logistic Model
best_logistic_accuracy = (-1, -1)
mean_error = []
std_error = []
C_range = [0.0001, 0.1, 1, 10, 20, 30, 60]
for C in C_range:
    accuracies = []
    kf = KFold(n_splits=5)
    for train, test in kf.split(X):
        model = LogisticRegression(C=C, max_iter=10000).fit(X[train], y[train])
        ypred = model.predict(X[test])
        accuracies.append(accuracy_score(y[test], ypred))
    avg_accuracy = sum(accuracies) / len(accuracies)
    print("LogisticRegression Model, C = ", C,
          " - average accuracy: ", avg_accuracy)
    mean_error.append(np.array(accuracies).mean())
    std_error.append(np.array(accuracies).std())
    if avg_accuracy > best_logistic_accuracy[0]:
        best_logistic_accuracy = (avg_accuracy, C)

# plot the CV
fig = plt.figure()
plt.rc('font', size=20)
ax = fig.add_subplot(111)
plt.errorbar(C_range, mean_error, yerr=std_error)
ax.set_ylabel("Mean accuracy")
ax.set_xlabel("C")
ax.set_title("C Cross-validation | LogisticRegression model")

model_names.append("Logistic (C=" + str(best_logistic_accuracy[1]) + ")")
model_accuracies.append(best_logistic_accuracy[0])

# kNN Model
best_knn_accuracy = (-1, -1)
mean_error = []
std_error = []
k_range = [1, 3, 5, 10, 20, 35, 50, 100, 250]
for k in k_range:
    accuracies = []
    kf = KFold(n_splits=5)
    for train, test in kf.split(X):
        model = KNeighborsClassifier(n_neighbors=k).fit(X[train], y[train])
        ypred = model.predict(X[test])
        accuracies.append(accuracy_score(y[test], ypred))
    avg_accuracy = sum(accuracies) / len(accuracies)
    mean_error.append(np.array(accuracies).mean())
    std_error.append(np.array(accuracies).std())
    print("kNN Model, k = ", k,
          " - average accuracy: ", avg_accuracy)
    if avg_accuracy > best_knn_accuracy[0]:
        best_knn_accuracy = (avg_accuracy, k)

# plot the CV
fig = plt.figure()
plt.rc('font', size=20)
ax = fig.add_subplot(111)
plt.errorbar(k_range, mean_error, yerr=std_error)
ax.set_ylabel("Mean accuracy")
ax.set_xlabel("k")
ax.set_title("k Cross-validation | kNN model")

model_names.append("kNN (k=" + str(best_knn_accuracy[1]) + ")")
model_accuracies.append(best_knn_accuracy[0])

# SVC Model
best_svc_accuracy = (-1, -1)
mean_error = []
std_error = []
C_range = [0.0001, 0.1, 1, 10, 20, 30, 60]
for C in C_range:
    accuracies = []
    kf = KFold(n_splits=5)
    for train, test in kf.split(X):
        model = SVC(C=C).fit(X[train], y[train])
        ypred = model.predict(X[test])
        accuracies.append(accuracy_score(y[test], ypred))
    avg_accuracy = sum(accuracies) / len(accuracies)
    mean_error.append(np.array(accuracies).mean())
    std_error.append(np.array(accuracies).std())
    print("SVC Model, C = ", C, " - average accuracy: ", avg_accuracy)
    if avg_accuracy > best_svc_accuracy[0]:
        best_svc_accuracy = (avg_accuracy, C)

# plot the CV
fig = plt.figure()
plt.rc('font', size=20)
ax = fig.add_subplot(111)
plt.errorbar(C_range, mean_error, yerr=std_error)
ax.set_ylabel("Mean accuracy")
ax.set_xlabel("C")
ax.set_title("C Cross-validation | SVC model")

model_names.append("SVC (C=" + str(best_svc_accuracy[1]) + ")")
model_accuracies.append(best_svc_accuracy[0])

# Plot bar chart to compare performance of all different models
fig = plt.figure()
ax = fig.add_subplot(111)
ax.bar(model_names, model_accuracies, color='green')
ax.set_xlabel("Model")
ax.set_ylabel("Best Score (accuracy)")

plt.show()

# both logistic & svc get to 0.907 for the early access, dummy too --> not feasible with this approach
