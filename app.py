from flask import Flask, request, jsonify, render_template_string, redirect, url_for
import pickle
import re
import nltk
import webbrowser
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("wordnet")

app = Flask(__name__)

# Load trained model and vectorizer
model, tfidf = pickle.load(open("final_sentiment_model.pkl", "rb"))

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

# 🔹 Default route → redirects to prediction page
@app.route("/")
def home():
    return redirect(url_for("ui"))

# 🔹 Prediction UI
@app.route("/ui")
def ui():
    return render_template_string("""
        <h2>Sentiment Analysis of Product Reviews</h2>
        <form action="/predict_ui" method="post">
            <textarea name="review" rows="5" cols="60"
            placeholder="Enter your review here"></textarea><br><br>
            <input type="submit">
        </form>
    """)

# 🔹 UI Prediction
@app.route("/predict_ui", methods=["POST"])
def predict_ui():
    review = request.form["review"]
    cleaned = clean_text(review)
    vector = tfidf.transform([cleaned])
    prediction = model.predict(vector)[0]
    return f"<h3>Predicted Sentiment: {prediction}</h3>"

# 🔹 API Prediction (for Postman / evaluation)
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    review = data["review"]
    cleaned = clean_text(review)
    vector = tfidf.transform([cleaned])
    prediction = model.predict(vector)[0]
    return jsonify({"sentiment": prediction})

if __name__ == "__main__":
    webbrowser.open("http://127.0.0.1:5000/")
    app.run(host="0.0.0.0", port=5000)
