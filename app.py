import streamlit as st
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
from gnews import GNews
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from transformers import AutoTokenizer, BertForSequenceClassification
import torch
from tensorflow.keras.models import load_model
import requests
import os

# --- Helper Function for RSI ---
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# --- File Downloader ---
def download_file_from_url(url, save_path):
    if not os.path.exists(save_path):
        st.info(f"Downloading {os.path.basename(save_path)}... Please wait.")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status() # Raise an exception for bad status codes
            with open(save_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            st.success(f"Downloaded {os.path.basename(save_path)} successfully.")
        except requests.exceptions.RequestException as e:
            st.error(f"Error downloading {os.path.basename(save_path)}: {e}")
            return False
    return True

# --- Load Saved Objects ---
@st.cache_resource
def load_app_models():
    # --- YOUR GITHUB RELEASE URLs ---
    MODEL_URL = "https://github.com/dev-kumar12/Stock-Trend-Prediction-with-NLP/releases/download/v1.0/lstm_model.keras"
    SCALER_URL = "https://github.com/dev-kumar12/Stock-Trend-Prediction-with-NLP/releases/download/v1.0/scaler.pkl"

    MODEL_PATH = "lstm_model.keras"
    SCALER_PATH = "scaler.pkl"

    # Download files if they don't exist
    model_exists = download_file_from_url(MODEL_URL, MODEL_PATH)
    scaler_exists = download_file_from_url(SCALER_URL, SCALER_PATH)

    if model_exists and scaler_exists:
        model = load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    else:
        st.error("Could not load model or scaler. Please check the URLs and refresh.")
        st.stop()


@st.cache_data
def get_stock_data(ticker):
    return yf.download(ticker, period="6mo")

@st.cache_data
def get_news_data(query):
    google_news = GNews(language='en', country='IN', period='2d')
    return pd.DataFrame(google_news.get_news(query))

@st.cache_resource
def get_finbert_model():
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = BertForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return tokenizer, model

def clean_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    if not isinstance(text, str): return ""
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    words = text.lower().split()
    cleaned_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(cleaned_words)

def get_finbert_sentiment(text_list, tokenizer, model):
    inputs = tokenizer(text_list, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return [model.config.id2label[label_id] for label_id in torch.argmax(predictions, dim=1).tolist()]

# --- Streamlit App ---
st.title("Reliance Industries Stock Trend Predictor")
st.write("This app uses a Deep Learning (LSTM) model to predict the next day's stock price based on historical prices, technical indicators, and the sentiment of recent news.")

if st.button("Predict Tomorrow's Trend"):
    with st.spinner("Running prediction pipeline... This may take a few minutes the first time."):
        
        st.info("Step 1: Fetching latest stock