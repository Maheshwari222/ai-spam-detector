
from flask import Flask, render_template, request

import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

import sqlite3

from datetime import datetime


# =========================
# Flask App
# =========================

app = Flask(__name__)


# =========================
# Database
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

X = data["text"]

y = data["label"]


# =========================
# Text Vectorization
# =========================

vectorizer = CountVectorizer()

X_vector = vectorizer.fit_transform(X)


# =========================
# Train Test Split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X_vector,
    y,
    test_size=0.2,
    random_state=42
)


# =========================
# Train Model
# =========================

model = MultinomialNB()

model.fit(X_train, y_train)


# =========================
# Accuracy
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

    # Get message
    message = request.form["message"]

    data_input = [message]

    # Convert to vector
    vector = vectorizer.transform(data_input)

    # Prediction
    prediction = model.predict(vector)

    # Confidence
    probability = model.predict_proba(vector)

    confidence = max(probability[0]) * 100

    # Current Time
    current_time = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    # Save to database
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

    # Fetch history
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
# Run App
# =========================

if __name__ == "__main__":

    app.run(debug=True)
