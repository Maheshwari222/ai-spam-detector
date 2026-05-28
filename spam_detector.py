import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Read CSV file
data = pd.read_csv("spam.csv")

# Inputs and outputs
X = data["text"]
y = data["label"]

# Convert words into numbers
vectorizer = CountVectorizer()
X_vector = vectorizer.fit_transform(X)

# Train model
model = MultinomialNB()
model.fit(X_vector, y)

# Test message
msg = ["Congratulations! You won free cash"]

# Convert test message
msg_vector = vectorizer.transform(msg)

# Predict
result = model.predict(msg_vector)

print("Prediction:", result[0])
