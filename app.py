# This code was written by Vignesh Pandiya G from Sri Eshwar College AI-DS Department

import re
import string
import joblib
import streamlit as st

def preprocess_text(text: str) -> str:
    """Cleans input text for prediction."""
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    return text

def load_model(model_path: str, vectorizer_path: str):
    """Loads the trained model and vectorizer."""
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

def predict_sentiment(model, vectorizer, new_text: str) -> tuple:
    """Predicts the sentiment and confidence score of a given review."""
    new_text = preprocess_text(new_text)
    X_new = vectorizer.transform([new_text])
    prediction = model.predict(X_new)
    confidence = model.predict_proba(X_new).max()  
    sentiment = "positive" if prediction[0] == 1 else "negative"
    return sentiment, round(confidence * 100, 2)  

def main():
    """Streamlit Web App for Sentiment Analysis"""
    st.title("Sentiment Analysis for Airline Reviews")

   
    model_path = "sentiment_model.pkl"
    vectorizer_path = "vectorizer.pkl"
    model, vectorizer = load_model(model_path, vectorizer_path)

    review_text = st.text_area("Enter an airline review:", height=150)

    if st.button("Predict Sentiment"):
        if review_text.strip():
            sentiment, confidence = predict_sentiment(model, vectorizer, review_text)
            st.write(f"**Predicted Sentiment:** {sentiment}")
            st.write(f"**Confidence Score:** {confidence}%")
        else:
            st.write("Please enter a review.")

    st.markdown("---")
    st.write("### Example Reviews")
    st.write("1. The flight was amazing, I loved the service!")
    st.write("2. Worst experience ever, I will never book again.")

if __name__ == "__main__":
    main()
