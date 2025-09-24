# test_app.py
import streamlit as st

st.title("✅ Streamlit is working")
st.write("If you see this, Streamlit is set up correctly.")

import joblib
import re
import string

# -------------------------------
# Load model and vectorizer
# -------------------------------
lr_model = joblib.load("LR_model.joblib")
tfidf_vectorizer = joblib.load("vectorizer.joblib")

# -------------------------------
# Text Cleaning Function
# -------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('\n', ' ', text)
    text = re.sub('[^a-zA-Z]', ' ', text)
    return text

def predict_news(news_text):
    cleaned = clean_text(news_text)
    vect = tfidf_vectorizer.transform([cleaned])
    pred = lr_model.predict(vect)[0]
    return "🟢 Real News" if pred == 1 else "🔴 Fake News"

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Fake News Detection", layout="centered")

st.title("📰 Fake News Detection App")
st.write("Paste a news article below and check if it's **Real** or **Fake**.")

news_input = st.text_area("Enter News Text Here:")

if st.button("Predict"):
    if news_input.strip():
        result = predict_news(news_input)
        st.subheader("Prediction Result:")
        st.success(result) if "Real" in result else st.error(result)
    else:
        st.warning("⚠️ Please enter some text before predicting.")
