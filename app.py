from flask_cors import CORS
import os
import joblib
import pickle
import flask
import os
import newspaper
from newspaper import Article
from flask import Flask, request, render_template
import urllib
from processing import *

pipeline = joblib.load('model.pkl')

# Loading Flask and assigning the model variable
app = Flask(__name__)
CORS(app)
app = flask.Flask(__name__, template_folder='templates')


@app.route('/')
def main():
    return render_template('main.html')

# Receiving the input url from the user and using Web Scrapping to extract the news content


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    url = request.get_data(as_text=True)[5:]
    url = urllib.parse.unquote(url)
    article = Article(str(url))
    article.download()
    article.parse()
    article.nlp()
    news = [article.summary]
    pred = pipeline.predict(news)
    # print(pred)
    return render_template('main.html', prediction_text='The news is {}'.format(pred))


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(port=port, debug=True, use_reloader=False)
