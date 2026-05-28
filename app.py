
# =========================
# app.py
# Advanced AI Spam Detector
# =========================

from flask import Flask, render_template, request

import pandas as pd

# NLP
from sklearn.feature_extraction.text import TfidfVectorizer

# ML Model
from sklearn.linear_model import LogisticRegression

# Accuracy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Charts
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

# Database
import sqlite3

# Time
from datetime import datetime


# =========================
# Flask App
# =========================

app = Flask(__name__)


# =========================
# Database Connection
# =========================

conn = sqlite3.connect(
    "history.db",
    check_same_thread=False
)

cursor = conn.cursor()

cursor.execute("""

CREATE TABLE IF NOT EXISTS predictions (

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    message TEXT,

    prediction TEXT,

    confidence REAL,

    time TEXT

)

""")

conn.commit()


# =========================
# Load Dataset
# =========================

data = pd.read_csv("spam.csv")

# Input and Output
X = data["text"]

y = data["label"]


# =========================
# TF-IDF Vectorization
# =========================

vectorizer = TfidfVectorizer()

X_vector = vectorizer.fit_transform(X)


# =========================
# Split Dataset
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X_vector,
    y,
    test_size=0.2,
    random_state=42
)


# =========================
# Train Logistic Regression
# =========================

model = LogisticRegression()

model.fit(X_train, y_train)


# =========================
# Model Accuracy
# =========================

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)


# =========================
# Pie Chart
# =========================

spam_count = len(
    data[data["label"] == "spam"]
)

ham_count = len(
    data[data["label"] == "ham"]
)

labels = ["Spam", "Ham"]

sizes = [spam_count, ham_count]

plt.figure(figsize=(4,4))

plt.pie(
    sizes,
    labels=labels,
    autopct='%1.1f%%'
)

plt.title("Spam vs Ham Emails")

plt.savefig("static/chart.png")

plt.close()


# =========================
# Home Page
# =========================

@app.route("/")
def home():

    # Fetch history
    cursor.execute("""

    SELECT * FROM predictions
    ORDER BY id DESC

    """)

    history = cursor.fetchall()

    return render_template(
        "index.html",
        accuracy=round(accuracy * 100, 2),
        history=history
    )


# =========================
# Prediction Route
# =========================

@app.route("/predict", methods=["POST"])
def predict():

    # Get user message
    message = request.form["message"]

    data_input = [message]

    # Convert text into vector
    vector = vectorizer.transform(data_input)

    # Prediction
    prediction = model.predict(vector)

    # Probability
    probability = model.predict_proba(vector)

    confidence = max(probability[0]) * 100

    # Current Time
    current_time = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    # Save to Database
    cursor.execute("""

    INSERT INTO predictions (

        message,
        prediction,
        confidence,
        time

    )

    VALUES (?, ?, ?, ?)

    """, (

        message,
        prediction[0],
        round(confidence, 2),
        current_time

    ))

    conn.commit()

    # Fetch Updated History
    cursor.execute("""

    SELECT * FROM predictions
    ORDER BY id DESC

    """)

    history = cursor.fetchall()

    return render_template(

        "index.html",

        prediction=prediction[0],

        confidence=round(confidence, 2),

        accuracy=round(accuracy * 100, 2),

        history=history
    )


# =========================
# Run Flask App
# =========================

if __name__ == "__main__":

    app.run(debug=True)

