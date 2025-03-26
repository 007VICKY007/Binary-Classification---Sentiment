# This code was written by Vignesh Pandiya G from Sri Eshwar College AI-DS Department

import re
import string
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

def preprocess_text(text: str) -> str:
    """Cleans and tokenizes text data."""
    text = str(text).lower() 
    text = re.sub(f"[{string.punctuation}]", "", text) 
    return text

def train_sentiment_model(csv_path: str, model_path: str, vectorizer_path: str) -> None:
    """Trains a sentiment classification model and saves it."""
    df = pd.read_csv(csv_path)

   
    if "Review" not in df.columns or "Recommended" not in df.columns:
        raise ValueError("CSV file must contain 'Review' and 'Recommended' columns")


    df = df[["Review", "Recommended"]].dropna()
    df["Review"] = df["Review"].apply(preprocess_text)
    df["Recommended"] = df["Recommended"].map(lambda x: 1 if x.lower() == "yes" else 0)  

  
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df["Review"])
    y = np.array(df["Recommended"])


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

 
    model = LogisticRegression()
    model.fit(X_train, y_train)

  
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")


    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Model saved to {model_path} and vectorizer saved to {vectorizer_path}")

if __name__ == "__main__":  
    csv_path = "AirlineReviews.csv" 
    model_path = "sentiment_model.pkl"
    vectorizer_path = "vectorizer.pkl"
    train_sentiment_model(csv_path, model_path, vectorizer_path)
