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

# Logistic Model
C = 1
accuracies = []
kf = KFold(n_splits=5)
for train, test in kf.split(X):
    print("Starting fold")
    model = LogisticRegression(C=C, max_iter=10000).fit(X[train], y[train])
    ypred = model.predict(X[test])
    accuracies.append(accuracy_score(y[test], ypred))
avg_accuracy = sum(accuracies) / len(accuracies)
print("LogisticRegression Model, C = ", C,
      " - average accuracy: ", avg_accuracy)
mean_error = np.array(accuracies).mean()
std_error = np.array(accuracies).std()
print("mean error: ", mean_error, " - std error: ", std_error)

# AVG Accuracy 0.72
