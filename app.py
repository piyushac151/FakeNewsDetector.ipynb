import streamlit as st
import joblib
import numpy as np
import re

model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def clean_text(text):
    return re.sub(r'\s+', ' ', text).strip()

def predict_news(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)[0]
    prob = model.decision_function(vec)[0]
    confidence = 1 / (1 + np.exp(-abs(prob)))
    return prediction, confidence

st.title("📰 Fake News Detector")

user_input = st.text_area("Paste a news article or headline:")

if st.button("Check"):
    if user_input:
        label, confidence = predict_news(user_input)
        st.markdown(f"### 🔎 Prediction: {'Real' if label else 'Fake'}")
        st.markdown(f"**Confidence**: {confidence * 100:.2f}%")
