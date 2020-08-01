from flask_cors import CORS
import os
import joblib
import pickle
import flask
import os
import newspaper
from newspaper import Article
from flask import Flask, request, render_template, jsonify, make_response
import urllib
import processing

pipeline = joblib.load('model.pkl')

# Loading Flask and assigning the model variable
app = Flask(__name__)
CORS(app)
app = flask.Flask(__name__)


@app.route('/')
def main():
    return 'This is Fake News Detector Server.'

# Receiving the input url from the user and using Web Scrapping to extract the news content


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    url = request.get_data(as_text=True)
    url = urllib.parse.unquote(url)
    article = Article(str(url))
    article.download()
    article.parse()
    article.nlp()
    news = [article.summary]
    pred = pipeline.predict(news)
    print(pred)

    response = make_response(jsonify({"result": pred[0]}))
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response


if __name__ == "__main__":
    # port = int(os.environ.get('PORT', 4000))
    app.run(debug=True, use_reloader=False)
