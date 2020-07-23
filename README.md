# Fake-News-Classifier-
A website to classify fake/real news by a ML model.  

## Prerequisites

Install `python3` and `pip`.

Download dataset from [here](https://drive.google.com/file/d/1eVYHo4sowg0NBiX8aJmG8-4p_pZAXlVQ/view?usp=sharing).

## Installing

1. Set up virtual [virtual environment](https://docs.python.org/3/tutorial/venv.html) by running the following commands:
    ```
    python3 -m venv fakeNews
    ```
    On windows, run: 
    ```
    fakeNews\Scripts\activate.bat
    ```
    On Unix or MacOS, run:
    ```
    source fakeNews/bin/activate
    ```

2. Install required packages:
    ```
    pip install -r requirements.txt
    ```

3. To build the ML model:
    ```
    python3 fake_news_detection.py 
    ```
    This would create a serialized version of our model into a file model.pkl
4. To test the project on locally hosted URL http://127.0.0.1:5000/ run:
    ```
    python3 app.py
    ```

## Authors

- [Tania Malhotra](github.com/TaniaMalhotra)
- [Malaika Agrawal]()
- [Vallari Agrawal](github.com/vallariag)
