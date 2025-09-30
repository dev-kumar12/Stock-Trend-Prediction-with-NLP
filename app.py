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
import plotly.graph_objects as go

# --- Page Configuration (Set at the very top) ---
st.set_page_config(layout="wide", page_title="Stock Trend Predictor")

# --- ALL HELPER FUNCTIONS ---
# (These functions are the same as before)
@st.cache_resource
def load_app_models():
    model = load_model('lstm_model.keras')
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
    return tokenizer, model

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

# --- UI & Main App Logic ---
st.title("📈 Stock Trend Predictor")
st.write("This app uses a Deep Learning (LSTM) model to predict the next day's trend based on live market data and recent news.")

# --- Sidebar for User Inputs ---
st.sidebar.header("User Input")
ticker_symbol = st.sidebar.text_input("Enter Stock Ticker (e.g., RELIANCE.NS, TATAMOTORS.NS)", "RELIANCE.NS").upper()
predict_button = st.sidebar.button("Predict Trend")

if predict_button:
    try:
        model, scaler = load_app_models()

        with st.spinner("Running live prediction pipeline..."):
            
            st.subheader(f"Analysis for {ticker_symbol}")
            tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Historical Data", "📰 News Sentiment"])

            # --- Data Fetching and Processing ---
            stock_df = get_live_stock_data(ticker_symbol)
            if stock_df.empty:
                st.error("Could not download stock data. Please check the ticker symbol or try again later.")
                st.stop()
            
            query = ticker_symbol.split('.')[0] # Use the company name for news search
            news_df = get_live_news_data(query)
            
            last_trading_day = stock_df.index.max()

            # --- News Sentiment Analysis ---
            if news_df.empty:
                today_sentiment = 0.0
            else:
                tokenizer, finbert_model = get_finbert_model()
                news_df['cleaned_title'] = news_df['title'].apply(clean_text)
                sentiments = get_finbert_sentiment(news_df['cleaned_title'].tolist(), tokenizer, finbert_model)
                news_df['finbert_sentiment'] = sentiments
                sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
                news_df['sentiment_score'] = news_df['finbert_sentiment'].map(sentiment_map)
                today_sentiment = news_df['sentiment_score'].mean()
                if pd.isna(today_sentiment): today_sentiment = 0.0
            
            # --- Technical Indicator Calculation ---
            stock_df['SMA_14'] = stock_df['Close'].rolling(window=14).mean()
            stock_df['RSI_14'] = calculate_rsi(stock_df)

            # --- Prediction Logic ---
            stock_df_with_sentiment = stock_df.copy()
            stock_df_with_sentiment['sentiment_score'] = today_sentiment
            stock_df_with_sentiment.dropna(inplace=True)
            
            model_features = ['Close', 'Volume', 'SMA_14', 'RSI_14', 'sentiment_score']
            last_60_days = stock_df_with_sentiment[model_features].tail(60)

            if len(last_60_days) < 60:
                st.error("Not enough historical data to make a prediction after processing. Please try again later.")
            else:
                scaled_sequence = scaler.transform(last_60_days)
                X_pred = np.array([scaled_sequence])
                prediction_scaled = model.predict(X_pred)

                dummy_pred = np.zeros((1, len(model_features)))
                dummy_pred[:, 0] = prediction_scaled
                prediction_actual = scaler.inverse_transform(dummy_pred)[0, 0]
                last_close_price = float(stock_df['Close'].iloc[-1])

                # --- Display in Tabs ---
                with tab1:
                    st.header("Prediction for the Next Trading Day")
                    col1, col2 = st.columns(2)
                    col1.metric(label=f"Last Close ({last_trading_day.date()})", value=f"₹{last_close_price:.2f}")
                    col2.metric(label="Predicted Price", value=f"₹{prediction_actual:.2f}", delta=f"₹{prediction_actual - last_close_price:.2f}")
                    if prediction_actual > last_close_price:
                        st.write("### Conclusion: The model predicts the stock price will **GO UP**. 📈")
                    else:
                        st.write("### Conclusion: The model predicts the stock price will **GO DOWN**. 📉")

                with tab2:
                    st.header("Historical Data & Technical Indicators")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df['Close'], mode='lines', name='Closing Price'))
                    fig.add_trace(go.Scatter(x=stock_df.index, y=stock_df['SMA_14'], mode='lines', name='14-Day SMA'))
                    fig.update_layout(title='Closing Price and Moving Average', xaxis_title='Date', yaxis_title='Price (INR)')
                    st.plotly_chart(fig, use_container_width=True)
                    st.write("Last 10 days of data:")
                    st.dataframe(stock_df.tail(10))

                with tab3:
                    st.header("Recent News Sentiment")
                    if not news_df.empty:
                        st.write(f"Average sentiment from the last 3 days: **{today_sentiment:.2f}**")
                        st.dataframe(news_df[['published date', 'title', 'finbert_sentiment']])
                    else:
                        st.warning("No recent news found to display.")

    except Exception as e:
        st.error("An unexpected error occurred.")
        st.error(f"Error details: {e}")