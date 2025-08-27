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

# --- Helper Function for RSI ---
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# --- Load Saved Objects ---
@st.cache_resource
def load_app_models():
    model = load_model('lstm_model.keras')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

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
        
        st.info("Step 1: Fetching latest stock and news data...")
        stock_df = get_stock_data("RELIANCE.NS")
        model, scaler = load_app_models()
        
        stock_df.columns = stock_df.columns.get_level_values(0)
        news_df = get_news_data('Reliance Industries')

        st.info("Step 2: Calculating technical indicators and news sentiment...")
        # --- FIX: Calculate indicators manually with pandas ---
        stock_df['SMA_14'] = stock_df['Close'].rolling(window=14).mean()
        stock_df['RSI_14'] = calculate_rsi(stock_df)
        
        news_df['cleaned_title'] = news_df['title'].apply(clean_text)
        tokenizer, finbert_model = get_finbert_model()
        sentiments = get_finbert_sentiment(news_df['cleaned_title'].tolist(), tokenizer, finbert_model)
        news_df['finbert_sentiment'] = sentiments
        
        sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
        news_df['sentiment_score'] = news_df['finbert_sentiment'].map(sentiment_map)
        today_sentiment = news_df['sentiment_score'].mean()
        if pd.isna(today_sentiment): today_sentiment = 0.0

        st.info("Step 3: Preparing data sequence for the LSTM model...")
        
        stock_df_with_sentiment = stock_df.copy()
        stock_df_with_sentiment['sentiment_score'] = today_sentiment
        stock_df_with_sentiment.dropna(inplace=True)
        
        model_features = ['Close', 'Volume', 'SMA_14', 'RSI_14', 'sentiment_score']
        last_60_days = stock_df_with_sentiment[model_features].tail(60).values
        
        scaled_sequence = scaler.transform(last_60_days)
        X_pred = np.array([scaled_sequence])

        st.info("Step 4: Making the final prediction...")
        prediction_scaled = model.predict(X_pred)

        dummy_pred = np.zeros((1, len(model_features)))
        dummy_pred[:, 0] = prediction_scaled
        prediction_actual = scaler.inverse_transform(dummy_pred)[0, 0]

        st.success("Prediction Complete!")
        last_close_price = stock_df['Close'].iloc[-1]
        
        col1, col2 = st.columns(2)
        col1.metric(label="Last Closing Price", value=f"₹{last_close_price:.2f}")
        col2.metric(label="Predicted Price for Tomorrow", value=f"₹{prediction_actual:.2f}", delta=f"₹{prediction_actual - last_close_price:.2f}")

        if prediction_actual > last_close_price:
            st.write("### Conclusion: The model predicts the stock price will **GO UP** tomorrow. 📈")
        else:
            st.write("### Conclusion: The model predicts the stock price will **GO DOWN** tomorrow. 📉")