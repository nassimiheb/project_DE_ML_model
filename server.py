from flask import Flask
import pandas as pd

app = Flask(__name__)

@app.route("/", methods=['GET'])
def load():
    return "Server running on port 3000 !"

@app.route("/preprocess", methods=['POST'])
def preprocess():
    pass

@app.route("/fit", methods=['POST'])
def fit():
    pass

@app.route("/predict", methods=['POST'])
def predict():
    pass

if __name__ == "__main__":
    app.run(host='0.0.0.0')