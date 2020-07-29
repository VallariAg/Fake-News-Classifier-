# Importing the libraries
import pandas as pd
import sklearn.metrics as metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from collections import Counter
import joblib
from processing import *


def extract_features(df, field, training_data, testing_data, vocabulary):
    vectorizer = TfidfVectorizer(
        use_idf=True, max_df=200, vocabulary=vocabulary, tokenizer=convert_to_tokens)
    vectorizer.fit_transform(training_data[field].values)
    train_feature_set = vectorizer.transform(training_data[field].values)
    test_feature_set = vectorizer.transform(testing_data[field].values)
    return train_feature_set, test_feature_set, vectorizer


def pre_training(df, field, vocabulary):
    training_data, testing_data = train_test_split(
        df, test_size=0.30, random_state=2000)
    Y_train = training_data['label'].values
    Y_test = testing_data['label'].values
    X_train, X_test, feature_transformer = extract_features(
        df, field, training_data, testing_data, vocabulary)
    return X_train, X_test, Y_train, Y_test, feature_transformer


def train_SVM_model(df, field, vocab):
    svm_cf = svm.SVC(kernel='rbf')
    X_train, X_test, Y_train, Y_test, feature_transformer = pre_training(
        df, field, vocab)
    svm_cf.fit(X_train, Y_train)
    y_pred = svm_cf.predict(X_test)
    return feature_transformer, y_pred, Y_test


def train_LR_model(df, field, vocabulary):
    X_train, X_test, Y_train, Y_test, feature_transformer = pre_training(
        df, field, vocabulary)
    log_reg = LogisticRegression(
        solver='sag', random_state=0, C=5, max_iter=1000)
    log_reg.fit(X_train, Y_train)
    y_pred = log_reg.predict(X_test)
    return feature_transformer, y_pred, Y_test


if __name__ == "__main__":
    df = pd.read_csv('final_news_dataset.csv', usecols=[
                     'title', 'content', 'label'], encoding='latin1')
    df.dropna(inplace=True)
    df['text'] = df['title'] + df['content']
    X = df['text']
    y = df['label']

    # Splitting the data into train
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    vectorizer = TfidfVectorizer(
        use_idf=True, max_df=200, tokenizer=convert_to_tokens)

    vectorizer.fit_transform(df['text'])
    pipeline = Pipeline([('preprocessing', InputTransformer(vectorizer)),
                         ('nbmodel', MultinomialNB())])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print(classification_report(y_test, y_pred))
    print("\nAccuracy - ", accuracy)

    joblib.dump(pipeline, 'model.pkl')
