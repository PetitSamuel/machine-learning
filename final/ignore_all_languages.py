from sklearn.utils import shuffle
from nltk.stem import PorterStemmer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
import json_lines
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, precision_recall_fscore_support
from sklearn.dummy import DummyClassifier

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

X = X[:500]
y = y[:500]
z = z[:500]

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


kf = KFold(n_splits=5)
for train, test in kf.split(X):
    print("Going through fold")

    # KNeighborsClassifier().fit(X[train], y[train])
    model = LogisticRegression().fit(X[train], y[train])
    ypred = model.predict(X[test])
    print("intercept % f, slope % f, square error % f")
    acc = accuracy_score(y[test], ypred)
    print(acc)

dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X, y)
ypred_dummy = dummy.predict(X)
acc = accuracy_score(y, ypred_dummy)
print("dummy: ", acc)
