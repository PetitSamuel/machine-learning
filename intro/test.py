from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import load_files
d = load_files("txt_sentoken", shuffle=False)
x = d.data
y = d.target
print(y)
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.2)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)
Xtrain = vectorizer.fit_transform(xtrain)
Xtest = vectorizer.transform(xtest)
model = LogisticRegression()
model.fit(Xtrain, ytrain)
preds = model.predict(Xtest)
print(classification_report(ytest, preds))
