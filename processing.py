from sklearn.base import BaseEstimator, TransformerMixin
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


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
        # vocabulary = vocab[field]
        transformedX = self.vectorizer.transform(X_)
        return transformedX


# title_limit = 10
# content_limit = 200
# vocab_title = make_vocab(df.title, title_limit)
# vocab_content = make_vocab(df.content, content_limit)
# df['all_text'] = df.title + df.content
# vocab_all = set(vocab_title + vocab_content)
