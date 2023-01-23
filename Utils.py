from typing import Any
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def vectorize(data_samples: list, vectorizer_type: str, ngram_range: tuple, n_features: int = 1000):
    global vectorizer
    if vectorizer_type == "tfidf":
        vectorizer = TfidfVectorizer(max_features=n_features, ngram_range=ngram_range)
    elif vectorizer_type == "count":
        vectorizer = CountVectorizer(max_features=n_features, ngram_range=ngram_range)
    else:
        print("wrong vectorizer_type, try to use vectorizer_type from this list [tfidf,count]")
        return
    x_vector =vectorizer.fit_transform(data_samples)
    return x_vector, vectorizer


def log_reg(x_vector, y_train, x_test, y_test):

    clf = LogisticRegression(random_state=42)
    clf.fit(x_vector, y_train)

    pred = clf.predict(x_test)
    print(classification_report(pred, y_test))
    return pred



