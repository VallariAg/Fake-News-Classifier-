# Importing the libraries
import pandas as pd
import sklearn.metrics as metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
# import pickle
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from collections import Counter
import joblib
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from processing import *


def get_vocab(file):
    with open(file) as f:
        vocab = f.readlines()
    return [x.strip() for x in vocab]


if __name__ == "__main__":
    df = pd.read_csv('final_news_dataset.csv', usecols=[
                     'title', 'content', 'label'], encoding='latin1')
    df.dropna(inplace=True)
    df['text'] = df['title'] + df['content']
    X = df['text']
    y = df['label']

    # vocabulary
    vocab = get_vocab('vocabulary')
    print("vocab imported")

    # Splitting the data into train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # ML Models
    svm_cf = svm.SVC(kernel='rbf')
    lr_cf = LogisticRegression(
        solver='sag', random_state=0, C=5, max_iter=1000)
    mnb_cf = MultinomialNB()
    avg_cf = VotingClassifier(
        estimators=[('lr', lr_cf), ('svm', svm_cf), ('mnb', mnb_cf)], voting='hard')

    # X_train = convert_to_tokens(X_train)

    vectorizer = TfidfVectorizer(
        use_idf=True, max_df=200, tokenizer=convert_to_tokens, vocabulary=vocab)
    print("initalized vectorizer")

    vectorizer.fit_transform(df['text'])
    print("fit_transfomed-ed vectorizer")

    pipeline = Pipeline([('preprocessing', InputTransformer(vectorizer)),
                         ('average_model', avg_cf)])

    pipeline.fit(X_train, y_train)
    print("pipeline fit-ed")

    y_pred = pipeline.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print(classification_report(y_test, y_pred))
    print("\nAccuracy - ", accuracy)

    joblib.dump(pipeline, 'model.pkl')
