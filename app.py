import streamlit as st
import pickle
import re
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -----------------------------
# SAFE NLTK SETUP FOR HF
# -----------------------------
NLTK_DATA_DIR = "/tmp/nltk_data"
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DATA_DIR)

@st.cache_resource
def setup_nltk():
    try:
        stopwords.words("english")
    except LookupError:
        nltk.download("stopwords", download_dir=NLTK_DATA_DIR)
        nltk.download("wordnet", download_dir=NLTK_DATA_DIR)

setup_nltk()

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="Flipkart Sentiment Analysis", layout="centered")
st.title("Sentiment Analysis of Flipkart Product Reviews")

# -----------------------------
# LOAD MODEL
# -----------------------------
with open("final_sentiment_model.pkl", "rb") as f:
    model, tfidf = pickle.load(f)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# -----------------------------
# TEXT CLEANING
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

# -----------------------------
# USER INPUT & PREDICTION
# -----------------------------
review = st.text_area("Enter your review")

if st.button("Predict Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        cleaned = clean_text(review)
        vector = tfidf.transform([cleaned])
        prediction = model.predict(vector)[0]

        if prediction.lower() == "positive":
            st.success("Predicted Sentiment: POSITIVE 😊")
        else:
            st.error("Predicted Sentiment: NEGATIVE 😞")
