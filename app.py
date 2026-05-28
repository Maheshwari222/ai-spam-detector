
# =========================
# app.py
# Full AI Spam Detector
# With Login System
# =========================

from flask import Flask, render_template, request
from flask import session, redirect, url_for

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

app.secret_key = "secret123"


# =========================
# Database Connection
# =========================

conn = sqlite3.connect(
    "history.db",
    check_same_thread=False
)

cursor = conn.cursor()


# =========================
# Prediction Table
# =========================

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
# Users Table
# =========================

cursor.execute("""

CREATE TABLE IF NOT EXISTS users (

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    username TEXT,

    password TEXT

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
# TF-IDF Vectorizer
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
# Train Model
# =========================

model = LogisticRegression()

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

    if "user" not in session:

        return redirect(url_for("login"))

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
# Register Route
# =========================

@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":

        username = request.form["username"]

        password = request.form["password"]

        cursor.execute("""

        INSERT INTO users (
            username,
            password
        )

        VALUES (?, ?)

        """, (

            username,
            password

        ))

        conn.commit()

        return redirect(url_for("login"))

    return render_template("register.html")


# =========================
# Login Route
# =========================

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        username = request.form["username"]

        password = request.form["password"]

        cursor.execute("""

        SELECT * FROM users
        WHERE username=? AND password=?

        """, (

            username,
            password

        ))

        user = cursor.fetchone()

        if user:

            session["user"] = username

            return redirect(url_for("home"))

    return render_template("login.html")


# =========================
# Logout Route
# =========================

@app.route("/logout")
def logout():

    session.pop("user", None)

    return redirect(url_for("login"))


# =========================
# Prediction Route
# =========================

@app.route("/predict", methods=["POST"])
def predict():

    if "user" not in session:

        return redirect(url_for("login"))

    # Get message
    message = request.form["message"]

    data_input = [message]

    # Convert to vector
    vector = vectorizer.transform(data_input)

    # Prediction
    prediction = model.predict(vector)

    # Probability
    probability = model.predict_proba(vector)

    confidence = max(probability[0]) * 100

    # Current time
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
# Run Flask App
# =========================

if __name__ == "__main__":

    app.run(debug=True)

