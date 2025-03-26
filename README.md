# Problem 2 Binary Classification for Customer Sentiment For Goml

## Project Overview
This project implements a sentiment analysis model to classify airline reviews as **positive** or **negative** based on textual input. It uses **Logistic Regression** with **TF-IDF vectorization** for text preprocessing. The model is trained using airline review data and can be accessed via a **Streamlit web interface**.

## Features
- Train a sentiment classification model using airline reviews.
- Preprocesses text data by removing punctuation and converting to lowercase.
- Predicts sentiment (**positive/negative**) from user-inputted reviews.
- Displays model confidence scores.
- Interactive web app built with **Streamlit**.

## Technologies Used
- Python
- Scikit-learn (Machine Learning Model)
- Pandas & NumPy (Data Processing)
- Joblib (Model Serialization)
- Streamlit (Frontend for UI)

## Installation & Setup
### Prerequisites
Ensure you have **Python 3.8+** installed. Install required dependencies:
```sh
pip install -r requirements.txt
```

### Running the Model Training Script
To train the sentiment model, place your dataset (CSV file) in the project directory and run:
```sh
python train.py
```
This will create:
- `sentiment_model.pkl` (Trained Model)
- `vectorizer.pkl` (TF-IDF Vectorizer)

### Running the Streamlit Web App
Once the model is trained, launch the Streamlit web app:
```sh
streamlit run app.py
```

## File Structure
```
📂 Problem 2
├── app.py              # Streamlit web app for sentiment prediction
├── train.py            # Script to train and save the model
├── sentiment_model.pkl # Saved trained model
├── vectorizer.pkl      # Saved TF-IDF vectorizer
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
└── AirlineReviews.csv  # Dataset (ensure this is in the directory)
```

## Example Usage
Once the Streamlit app is running:
1. Enter a review in the text box.
2. Click **Predict Sentiment**.
3. The app will display:
   - **Sentiment (Positive/Negative)**
   - **Confidence Score (%)**

## Sample Output
**Example Input:**
```
"The flight was amazing, I loved the service!"
```
**Output:**
```
Predicted Sentiment: Positive
Confidence Score: 92.45%
```

## Author
This project was developed by **Vignesh Pandiya G** from **Sri Eshwar College AI-DS Department**.

