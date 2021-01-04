from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import FrenchStemmer
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
analyzer = CountVectorizer().build_analyzer()


def stemmed_words(doc):
    return (stemmer.stem(w) for w in analyzer(doc))


stem_vectorizer = CountVectorizer(analyzer=stemmed_words, ngram_range=(2, 2))
print(stem_vectorizer.fit_transform(['Tu marches dans la rue']))
print(stem_vectorizer.get_feature_names())
