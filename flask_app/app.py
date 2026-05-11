from flask import Flask, render_template, request
import numpy as np
import mlflow
import pickle
import pandas as pd
from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CollectorRegistry,
    CONTENT_TYPE_LATEST,
)
import time
import nltk
import string
import re
import dagshub
import warnings

# ---------------- NLTK Downloads ----------------
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("stopwords")

from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer

# Ensure wordnet fully loaded
wordnet.ensure_loaded()

# Global lemmatizer
lemmatizer = WordNetLemmatizer()

# ---------------- Warning Settings ----------------
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# ---------------- Text Preprocessing Functions ----------------
def lemmatization(text):
    """Lemmatize the text."""
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)


def remove_stop_words(text):
    """Remove stop words from the text."""
    stop_words = set(stopwords.words("english"))
    text = [word for word in str(text).split() if word not in stop_words]
    return " ".join(text)


def removing_numbers(text):
    """Remove numbers from the text."""
    text = ''.join([char for char in text if not char.isdigit()])
    return text


def lower_case(text):
    """Convert text to lower case."""
    text = text.split()
    text = [word.lower() for word in text]
    return " ".join(text)


def removing_punctuations(text):
    """Remove punctuations from the text."""
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace('؛', "")
    text = re.sub('\s+', ' ', text).strip()
    return text


def removing_urls(text):
    """Remove URLs from the text."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)


def normalize_text(text):
    """Apply all preprocessing steps."""
    text = lower_case(text)
    text = remove_stop_words(text)
    text = removing_numbers(text)
    text = removing_punctuations(text)
    text = removing_urls(text)
    text = lemmatization(text)

    return text


# ---------------- MLflow + DagsHub Setup ----------------
mlflow.set_tracking_uri(
    'https://dagshub.com/deepaku0222/mlops-project.mlflow'
)

dagshub.init(
    repo_owner='deepaku0222',
    repo_name='mlops-project',
    mlflow=True
)

# ---------------- Flask App ----------------
app = Flask(__name__)

# ---------------- Prometheus Metrics ----------------
registry = CollectorRegistry()

REQUEST_COUNT = Counter(
    "app_request_count",
    "Total number of requests to the app",
    ["method", "endpoint"],
    registry=registry,
)

REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds",
    "Latency of requests in seconds",
    ["endpoint"],
    registry=registry,
)

PREDICTION_COUNT = Counter(
    "model_prediction_count",
    "Count of predictions for each class",
    ["prediction"],
    registry=registry,
)

# ---------------- Model Loading ----------------
model_name = "my_model"


def get_latest_model_version(model_name):
    client = mlflow.MlflowClient()

    latest_versions = client.search_model_versions(
        f"name='{model_name}'"
    )

    if not latest_versions:
        return None

    latest_version = max(
        latest_versions,
        key=lambda x: int(x.version)
    )

    return latest_version.version


model_version = get_latest_model_version(model_name)

if model_version is None:
    raise Exception(f"No versions found for model: {model_name}")

model_uri = f"models:/{model_name}/{model_version}"

print(f"Fetching model from: {model_uri}")

model = mlflow.pyfunc.load_model(model_uri)

# ---------------- Load Vectorizer ----------------
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

# ---------------- Routes ----------------
@app.route("/")
def home():
    REQUEST_COUNT.labels(
        method="GET",
        endpoint="/"
    ).inc()

    start_time = time.time()

    response = render_template(
        "index.html",
        result=None
    )

    REQUEST_LATENCY.labels(
        endpoint="/"
    ).observe(time.time() - start_time)

    return response


@app.route("/predict", methods=["POST"])
def predict():

    REQUEST_COUNT.labels(
        method="POST",
        endpoint="/predict"
    ).inc()

    start_time = time.time()

    try:
        # ---------------- Input ----------------
        text = request.form["text"]

        # ---------------- Preprocessing ----------------
        processed_text = normalize_text(text)

        print("\n----------------------------")
        print("Original Text:", text)
        print("Processed Text:", processed_text)

        # ---------------- Feature Extraction ----------------
        features = vectorizer.transform([processed_text])

        features_df = pd.DataFrame(
            features.toarray(),
            columns=[
                str(i)
                for i in range(features.shape[1])
            ]
        )

        print("Features Shape:", features_df.shape)

        # ---------------- Prediction ----------------
        result = model.predict(features_df)

        raw_prediction = result[0]

        print("Raw Prediction:", raw_prediction)

        # ---------------- Label Mapping ----------------
        if raw_prediction == 0:
            prediction = "😊 Positive Sentiment"
        else:
            prediction = "😞 Negative Sentiment"

        print("Final Prediction:", prediction)
        print("----------------------------\n")

        # ---------------- Metrics ----------------
        PREDICTION_COUNT.labels(
            prediction=str(prediction)
        ).inc()

        REQUEST_LATENCY.labels(
            endpoint="/predict"
        ).observe(time.time() - start_time)

        # ---------------- Return Result ----------------
        return render_template(
            "index.html",
            result=prediction
        )

    except Exception as e:

        print("Prediction Error:", str(e))

        return render_template(
            "index.html",
            result=f"Error: {str(e)}"
        )


@app.route("/metrics", methods=["GET"])
def metrics():
    """Expose Prometheus metrics."""
    return (
        generate_latest(registry),
        200,
        {"Content-Type": CONTENT_TYPE_LATEST},
    )


# ---------------- Main ----------------
if __name__ == "__main__":
    app.run(
        debug=False,
        host="0.0.0.0",
        port=5000
    )