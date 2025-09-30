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
from datetime import date, timedelta
import nltk
import os

# --- ALL HELPER FUNCTIONS ---
@st.cache_resource
def load_app_models():
    # --- FINAL FIX: Load the .h5 file ---
    model = load_model('lstm_model.h5')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

@st.cache_data
def get_live_stock_data(ticker):
    end_date = date.today() + timedelta(days=1)
    start_date = end_date - timedelta(days=365)
    return yf.download(ticker, start=start_date, end=end_date)

@st.cache_data
def get_live_news_data(query):
    google_news = GNews(language='en', country='IN', period='3d')
    news = google_news.get_news(query)
    return pd.DataFrame(news) if news else pd.DataFrame()

@st.cache_resource
def get_finbert_model():
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = BertForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return model, tokenizer

def clean_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    if not isinstance(text, str): return ""
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    words = text.lower().split()
    cleaned_words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(cleaned_words)

def get_finbert_sentiment(text_list, _tokenizer, _model):
    inputs = _tokenizer(text_list, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        outputs = _model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return [_model.config.id2label[label_id] for label_id in torch.argmax(predictions, dim=1).tolist()]

def calculate_rsi(data, window=14):
    close_prices = data['Close']
    delta = close_prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# --- Main Streamlit App Logic ---
st.title("Reliance Industries Stock Trend Predictor")
st.write("This app uses a pre-trained LSTM model to predict the next day's trend based on live market data and recent news.")

try:
    model, scaler = load_app_models()

    if st.button("Predict Tomorrow's Trend"):
        with st.spinner("Running live prediction pipeline..."):
            
            st.info("Attempting to fetch live market data...")
            live_data_fetched = False
            try:
                stock_df = get_live_stock_data("RELIANCE.NS")
                if not stock_df.empty:
                    live_data_fetched = True
                    st.success("Successfully fetched live market data.")
            except Exception as e:
                st.warning(f"Could not fetch live market data (Error: {e}).")

            if not live_data_fetched:
                st.warning("Falling back to last known historical data for prediction.")
                stock_df = pd.read_csv("final_master_dataset.csv", index_col='Date', parse_dates=True)
            
            news_df = get_live_news_data('Reliance Industries')
            
            last_trading_day = stock_df.index.max()
            if live_data_fetched and last_trading_day.date() < date.today():
                 st.warning(f"Market appears closed. Using last available data from {last_trading_day.date()}.")

            st.info("Analyzing news sentiment...")
            if news_df.empty:
                st.warning("Could not find recent news. Using a neutral sentiment score (0.0).")
                today_sentiment = 0.0
            else:
                news_df['cleaned_title'] = news_df['title'].apply(clean_text)
                tokenizer, finbert_model = get_finbert_model()
                sentiments = get_finbert_sentiment(news_df['cleaned_title'].tolist(), tokenizer, finbert_model)
                news_df['finbert_sentiment'] = sentiments
                
                sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
                news_df['sentiment_score'] = news_df['finbert_sentiment'].map(sentiment_map)
                today_sentiment = news_df['sentiment_score'].mean()
                if pd.isna(today_sentiment): today_sentiment = 0.0

            st.info("Calculating technical indicators...")
            stock_df['SMA_14'] = stock_df['Close'].rolling(window=14).mean()
            stock_df['RSI_14'] = calculate_rsi(stock_df)
            
            st.info("Preparing data sequence for the model...")
            stock_df_with_sentiment = stock_df.copy()
            stock_df_with_sentiment['sentiment_score'] = today_sentiment
            stock_df_with_sentiment.dropna(inplace=True)
            
            model_features = ['Close', 'Volume', 'SMA_14', 'RSI_14', 'sentiment_score']
            last_60_days = stock_df_with_sentiment[model_features].tail(60).values
            
            if last_60_days.shape[0] < 60:
                st.error(f"Error: Not enough historical data to make a prediction after processing. Needed 60 days, but found only {last_60_days.shape[0]}.")
            else:
                scaled_sequence = scaler.transform(last_60_days)
                X_pred = np.array([scaled_sequence])

                st.info("Making the final prediction...")
                prediction_scaled = model.predict(X_pred)

                dummy_pred = np.zeros((1, len(model_features)))
                dummy_pred[:, 0] = prediction_scaled
                prediction_actual = scaler.inverse_transform(dummy_pred)[0, 0]

                st.success("Prediction Complete!")
                
                last_close_price = float(stock_df['Close'].iloc[-1])
                prediction_actual_float = float(prediction_actual)
                
                col1, col2 = st.columns(2)
                col1.metric(label=f"Last Close ({last_trading_day.date()})", value=f"₹{last_close_price:.2f}")
                col2.metric(label="Predicted Price for Next Trading Day", value=f"₹{prediction_actual_float:.2f}", delta=f"₹{prediction_actual_float - last_close_price:.2f}")

                if prediction_actual_float > last_close_price:
                    st.write("### Conclusion: The model predicts the stock price will **GO UP**. 📈")
                else:
                    st.write("### Conclusion: The model predicts the stock price will **GO DOWN**. 📉")

except Exception as e:
    st.error(f"An unexpected error occurred.")
    st.error(f"Error details: {e}")