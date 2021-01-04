from nltk.stem import PorterStemmer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
import json_lines
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier

X = []
y = []
z = []
with open('final_data', 'rb') as f:
    for item in json_lines.reader(f):
        X.append(item['text'])
        y.append(item['voted_up'])
        z.append(item['early_access'])

X = X[:1000]
y = y[:1000]
z = z[:1000]

print(len(X))
stemmer = PorterStemmer()
analyzer = CountVectorizer().build_analyzer()


def stemmed_words(doc):
    return (stemmer.stem(w) for w in analyzer(doc))


stem_vectorizer = CountVectorizer(analyzer=stemmed_words, ngram_range=(2, 2))
tokens = stem_vectorizer.fit_transform(X)

# Using stems reduces number of words from 177883 to 44853
print(len(stem_vectorizer.get_feature_names()))

# print("lets get it")
# knn_model = KNeighborsClassifier(
#     n_neighbors=1).fit(X, y)
# print("lets get it")
# print(knn_model.score(X, y))

X = np.array(tokens.toarray())
y = np.array(y)


kf = KFold(n_splits=5)
for train, test in kf.split(X):
    print("Going through fold")
    # LinearRegression().fit(X[train], y[train])
    model = KNeighborsClassifier().fit(X[train], y[train])
    ypred = model.predict(X[test])
    print("intercept % f, slope % f, square error % f")
    acc = accuracy_score(y[test], ypred)
    print(acc)
