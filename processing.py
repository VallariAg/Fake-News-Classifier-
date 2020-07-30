from sklearn.base import BaseEstimator, TransformerMixin
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
import pandas as pd


def convert_to_tokens(text) -> [str]:
    try:
        words = nltk.word_tokenize(" ".join(text.tolist()))
    except:
        words = nltk.word_tokenize(" ".join(text.split()))
    stop = stopwords.words('english')
    lemmatized_words = []
    for word in words:
        word = word.lower()
        if word not in stop and word.isalpha() and len(word) > 2:
            lemmatized_words.append(WordNetLemmatizer().lemmatize(word))
    return lemmatized_words


def make_vocab(text, freq_limit: int) -> {}:
    c = Counter(convert_to_tokens(text))
    vocab = [word for word, count in c.items() if count > freq_limit]
    return vocab


class InputTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, vectorizer):
        self.vectorizer = vectorizer
        print("initalized InputTransformer")

    def fit(self, X, y=None):
        print('fit')
        return self

    def transform(self, X, y=None):
        X_ = X.copy()
        print('trasform')
        transformedX = self.vectorizer.transform(X_)
        return transformedX


if __name__ == '__main__':
    '''
    Make vocabulary with make_vocab() and store it in `vocabulary` text file
    '''
    TITLE_LIMIT = 10
    CONTENT_LIMIT = 200
    df = pd.read_csv('final_news_dataset.csv', usecols=[
                     'title', 'content', 'label'], encoding='latin1')
    df.dropna(inplace=True)
    df['text'] = df['title'] + df['content']

    print("making vocab...")
    vocab = set(make_vocab(df['title'], TITLE_LIMIT) +
                make_vocab(df['content'], CONTENT_LIMIT))

    print("dumping into file...")
    with open('vocabulary', 'w+') as f:
        f.write("\n".join(vocab))
