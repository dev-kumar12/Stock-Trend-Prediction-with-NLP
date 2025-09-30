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
from datetime import date

# --- Helper Function for RSI ---
def calculate_rsi(data, window=14):
    close_prices = data['Close']
    delta = close_prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# --- Load Local Files ---
@st.cache_resource
def load_app_models():
    model = load_model('lstm_model.keras')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

# --- Live Data Fetching Functions ---
@st.cache_data
def get_live_stock_data(ticker):
    return yf.download(ticker, period="1y")

@st.cache_data
def get_live_news_data(query):
    google_news = GNews(language='en', country='IN', period='3d')
    news = google_news.get_news(query)
    return pd.DataFrame(news) if news else pd.DataFrame()

# --- NLP Functions ---
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

# --- Main Streamlit App Logic ---
st.title("Reliance Industries Stock Trend Predictor")
st.write("This app uses a pre-trained LSTM model to predict the next day's trend based on live market data and recent news.")

try:
    model, scaler = load_app_models()

    if st.button("Predict Tomorrow's Trend"):
        with st.spinner("Running live prediction pipeline..."):
            
            st.info("Fetching live stock data and recent news...")
            stock_df = get_live_stock_data("RELIANCE.NS")

            # --- THE FINAL FIX IS HERE ---
            # Add a check to ensure stock data was downloaded successfully
            if stock_df.empty:
                st.error("Could not download stock data. The market may be closed or data source is temporarily unavailable. Please try again on a trading day.")
                st.stop() # Stop the app execution

            news_df = get_live_news_data('Reliance Industries')
            
            last_trading_day = stock_df.index.max()
            if last_trading_day.date() < date.today():
                 st.warning(f"Market is closed. Using last available data from {last_trading_day.date()} to predict for the next trading day.")

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
                st.error(f"Error: Not enough historical data to make a prediction. Needed 60 days, but found only {last_60_days.shape[0]}. Please try again on a trading day.")
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
    st.error(f"An unexpected error occurred. Please ensure all model and data files are in the folder.")
    st.error(f"Error details: {e}")