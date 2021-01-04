import matplotlib.pyplot as plt
import numpy as np
import json_lines
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, precision_recall_fscore_support, roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
from sklearn.utils import shuffle
from langdetect import detect


# Map a detected language from langdetect into a Stemmer language code (only supported ones)
def get_stemmer_lang(key_lang):
    return {
        'ru': 'russian',
        'en': 'english',
        'de': 'german',
        'pt': 'portuguese',
        'es': 'spanish',
        'fr': 'french',
        'sw': 'swedish',
        'ro': 'romanian',
        'hu': 'hungarian',
        'no': 'norwegian',
        'it': 'italian',
        'fi': 'finnish',
        'da': 'danish',
        'nl': 'dutch',
        'sv': 'swedish'
    }.get(key_lang, None)  # None is default if lang not found


stemmer = PorterStemmer()
analyzer = CountVectorizer().build_analyzer()


# Returns the stems from words
def stemmed_words(doc):
    return (stemmer.stem(w) for w in analyzer(doc))


X = []
y = []
z = []

# Read data from file
with open('final_data', 'rb') as f:
    for item in json_lines.reader(f):
        X.append(item['text'])
        y.append(item['voted_up'])
        z.append(item['early_access'])

# Dataset has first positive reviews and then negative (ordered), shuffle values in unison
X, y, z = shuffle(X, y, z, random_state=0)

# Use a sample of the data for faster tests
X = X[:1000]
y = y[:1000]
z = z[:1000]

# Split reviews into language groups
reviews_by_language = {}
for index in range(len(X)):
    item = X[index]
    try:
        lang = detect(item)
    except:
        # When language not detectable, use "other"
        lang = 'other'
    if lang not in reviews_by_language:
        reviews_by_language[lang] = {
            "x": [],
            "y": [],
            "z": []
        }
    reviews_by_language[lang]["x"].append(X[index])
    reviews_by_language[lang]["y"].append(y[index])
    reviews_by_language[lang]["z"].append(z[index])

# Values are now stored in reviews_by_language - clear some memory space
X = []
y = []
z = []

best_models = []
dummy_models = []
model = None  # Place holder
# Process each language indepedently
for lang in reviews_by_language.keys():
    print("PROCESSING ", lang)
    stem_lang = get_stemmer_lang(lang)
    if stem_lang is None:
        # Use default analyser if there is no matching stemmer for this language
        analyzer_for_lang = 'word'
    else:
        # Language has a stemmer
        analyzer_for_lang = stemmed_words
        # Redefine stemmer with specified language
        stemmer = SnowballStemmer(stem_lang)
    stem_vectorizer = CountVectorizer(
        analyzer=analyzer_for_lang, ngram_range=(2, 2))
    try:
        tokens = stem_vectorizer.fit_transform(reviews_by_language[lang]["x"])
    except:
        # On tokeniser error, skip the language
        continue
    X = np.array(tokens.toarray())
    y = np.array(reviews_by_language[lang]["y"])
    # use this line instead of the above one for early access models
    # y = np.array(reviews_by_language[lang]["z"])

    # Skip languages with less than 5 reviews (not possible with k-fold)
    # this may trigger depending on sampling size used
    if len(X) < 5:
        continue

    print("training models now...")
    model_names = []
    model_accuracies = []

    # Keep track of the best performing model / accuracy for the current language
    curr_best = (None, -1, 'undefined language')
    # Decision Tree Model
    accuracies = []
    kf = KFold(n_splits=5)
    for train, test in kf.split(X):
        model = DecisionTreeClassifier().fit(X[train], y[train])
        ypred = model.predict(X[test])
        accuracies.append(accuracy_score(y[test], ypred))
    avg_accuracy = sum(accuracies) / len(accuracies)
    print("Decision Tree Model average accuracy: ", avg_accuracy)

    if curr_best[0] is None or curr_best[1] < avg_accuracy:
        curr_best = (model, avg_accuracy, lang)

    model_names.append("DecisionTree")
    model_accuracies.append(avg_accuracy)

    # Dummy Model
    accuracies = []
    kf = KFold(n_splits=5)
    for train, test in kf.split(X):
        model = DummyClassifier(
            strategy="most_frequent").fit(X[train], y[train])
        ypred = model.predict(X[test])
        accuracies.append(accuracy_score(y[test], ypred))
    avg_accuracy = sum(accuracies) / len(accuracies)
    dummy_models.append((avg_accuracy, lang))
    print("Dummy Model average accuracy: ", avg_accuracy)

    if curr_best[0] is None or curr_best[1] < avg_accuracy:
        curr_best = (model, avg_accuracy, lang)

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
        if curr_best[0] is None or curr_best[1] < avg_accuracy:
            curr_best = (model, avg_accuracy, lang)
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
        try:
            for train, test in kf.split(X):
                model = LogisticRegression(
                    C=C, max_iter=10000).fit(X[train], y[train])
                ypred = model.predict(X[test])
                accuracies.append(accuracy_score(y[test], ypred))
        except:
            # Logistic regression fails when there is a single class (ex all are true) in the sample data
            break
        avg_accuracy = sum(accuracies) / len(accuracies)
        if curr_best[0] is None or curr_best[1] < avg_accuracy:
            curr_best = (model, avg_accuracy, lang)
        print("LogisticRegression Model, C = ", C,
              " - average accuracy: ", avg_accuracy)
        mean_error.append(np.array(accuracies).mean())
        std_error.append(np.array(accuracies).std())
        if avg_accuracy > best_logistic_accuracy[0]:
            best_logistic_accuracy = (avg_accuracy, C)

    if len(mean_error) > 0 and len(std_error) > 0:
        # plot the CV
        fig = plt.figure()
        plt.rc('font', size=20)
        ax = fig.add_subplot(111)
        plt.errorbar(C_range, mean_error, yerr=std_error)
        ax.set_ylabel("Mean accuracy")
        ax.set_xlabel("C")
        ax.set_title("C Cross-validation | LogisticRegression model")

        model_names.append(
            "Logistic (C=" + str(best_logistic_accuracy[1]) + ")")
        model_accuracies.append(best_logistic_accuracy[0])

        # kNN Model
    best_knn_accuracy = (-1, -1)
    mean_error = []
    std_error = []
    k_range = [1, 3, 5, 10, 20, 35, 50, 100, 250]
    actual_k_range = []
    for k in k_range:
        accuracies = []
        kf = KFold(n_splits=5)
        try:
            for train, test in kf.split(X):
                model = KNeighborsClassifier(
                    n_neighbors=k).fit(X[train], y[train])
                ypred = model.predict(X[test])
                accuracies.append(accuracy_score(y[test], ypred))
            actual_k_range.append(k)
        except:
            # We need n_neighborhs <= n_samples. Stop kNN execution on exception raised here
            break
        avg_accuracy = sum(accuracies) / len(accuracies)
        if curr_best[0] is None or curr_best[1] < avg_accuracy:
            curr_best = (model, avg_accuracy, lang)
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
    plt.errorbar(actual_k_range, mean_error, yerr=std_error)
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
        try:
            for train, test in kf.split(X):
                model = SVC(C=C).fit(X[train], y[train])
                ypred = model.predict(X[test])
                accuracies.append(accuracy_score(y[test], ypred))
        except:
            # SVC regression fails when there is a single class (ex all are true) in the sample data
            break
        avg_accuracy = sum(accuracies) / len(accuracies)
        if curr_best[0] is None or curr_best[1] < avg_accuracy:
            curr_best = (model, avg_accuracy, lang)
        mean_error.append(np.array(accuracies).mean())
        std_error.append(np.array(accuracies).std())
        print("SVC Model, C = ", C, " - average accuracy: ", avg_accuracy)
        if avg_accuracy > best_svc_accuracy[0]:
            best_svc_accuracy = (avg_accuracy, C)

    if len(mean_error) > 0 and len(std_error) > 0:
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

    # Plot comparison bar chart of all models for current language
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(model_names, model_accuracies, color='green')
    ax.set_xlabel("Model")
    ax.set_ylabel("Best Score (accuracy)")
    ax.set_title("Language: " + str(lang))

    best_models.append(curr_best)
    print("BEST MODEL WAS: ", curr_best, " -- lang: ", lang)


dummy_accuracy, dummy_langs = zip(*dummy_models)

# Plot performance of dummy model for all languages
fig = plt.figure()
ax = fig.add_subplot(111)
ax.bar(dummy_langs, dummy_accuracy, color='green')
ax.set_xlabel("Language")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracies of Dummy models split per language")

# Plot performance of best models for all languages
models, accuracies, langs = zip(*best_models)
fig = plt.figure()
ax = fig.add_subplot(111)
ax.bar(langs, accuracies, color='green')
ax.set_xlabel("Language")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracies of Best models split per language")

print(best_models)
print(dummy_models)
plt.show()
