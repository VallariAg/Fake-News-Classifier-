# Importing the libraries
import pandas as pd
import numpy as np
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
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

df = pd.read_csv('final_news_dataset.csv', usecols=[
    'title', 'content', 'label'])
df.dropna(inplace=True)

# ----------------- Our ML model --------------------


def convert_to_tokens(text) -> [str]:

    try:
        words = nltk.word_tokenize(" ".join(text.tolist()))
    except:
        words = nltk.word_tokenize(" ".join(text.split()))

    stop = stopwords.words('english')

    lemmatized_words = []
    for word in words:
        if word not in stop and word.isalpha() and len(word) > 2:
            lemmatized_words.append(WordNetLemmatizer().lemmatize(word))

    return lemmatized_words


def make_vocab(text, freq_limit: int) -> {}:
    c = Counter(convert_to_tokens(text))
    all_tokens = [word for word, count in c.items() if count > freq_limit]

    return all_tokens


# title_limit = 10
# content_limit = 200

# vocab_title = make_vocab(df.title, title_limit)
# vocab_content = make_vocab(df.content, content_limit)

# df['all_text'] = df.title + df.content
# vocab_all = set(vocab_title + vocab_content)


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


def train_bayes_model(df, field, vocabulary):

    X_train, X_test, Y_train, Y_test, feature_transformer = pre_training(
        df, field, vocabulary)
    mnb = MultinomialNB()
    mnb.fit(X_train, Y_train)
    y_pred = mnb.predict(X_test)

    return feature_transformer, y_pred, Y_test


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

# ---------------------------------------------------

# --------- Model used to test website --------------


df['text'] = df['title'] + df['content']
X = df['text']
y = df['label']

# Splitting the data into train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Creating a pipeline that first creates bag of words(after applying stopwords) & then applies Multinomial Naive Bayes model
pipeline = Pipeline([('tfidf', TfidfVectorizer(stop_words='english')),
                     ('nbmodel', MultinomialNB())])
# Training our data
pipeline.fit(X_train, y_train)
# Predicting the label for the test data
pred = pipeline.predict(X_test)
# Checking the performance of our model
print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))
# ---------------------------------------------------

# Serialising the file
with open('model.pickle', 'wb') as handle:
    pickle.dump(pipeline, handle, protocol=pickle.HIGHEST_PROTOCOL)
