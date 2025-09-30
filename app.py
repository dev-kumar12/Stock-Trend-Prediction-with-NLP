import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# --- Load Local Files ---
@st.cache_resource
def load_app_models():
    """Loads the pre-trained model and scaler from local files."""
    model = load_model('lstm_model.h5')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

@st.cache_data
def load_master_data():
    """Loads the single, clean, pre-processed master dataset."""
    df = pd.read_csv("final_master_dataset.csv")
    # Robustly handle the date column
    if 'Unnamed: 0' in df.columns:
        df = df.rename(columns={'Unnamed: 0': 'Date'})
    elif 'date' in df.columns:
         df = df.rename(columns={'date': 'Date'})
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    return df

# --- Main Streamlit App Logic ---
st.title("Reliance Industries Stock Trend Predictor")
st.write("This app uses a pre-trained LSTM model and a pre-processed historical dataset to demonstrate a prediction for the next trend.")

try:
    model, scaler = load_app_models()
    final_df = load_master_data()
    st.success("Model and historical data loaded successfully!")
    
    st.write("### Last 5 Days of Available Data:")
    st.dataframe(final_df.tail())

    if st.button("Predict Trend Using Latest Available Data"):
        with st.spinner("Running prediction..."):
            
            st.info("Preparing final data sequence...")
            
            # Define the features the model was trained on
            model_features = ['Close', 'Volume', 'SMA_14', 'RSI_14', 'sentiment_score']
            
            # Get the last 60 days from our clean dataset
            last_60_days = final_df[model_features].tail(60).values
            
            if last_60_days.shape[0] < 60:
                st.error(f"Error: The master dataset has less than 60 days of data. Found only {last_60_days.shape[0]} days.")
            else:
                # Scale the sequence
                scaled_sequence = scaler.transform(last_60_days)
                X_pred = np.array([scaled_sequence])

                st.info("Making the prediction...")
                prediction_scaled = model.predict(X_pred)

                # Inverse transform the prediction
                dummy_pred = np.zeros((1, len(model_features)))
                dummy_pred[:, 0] = prediction_scaled
                prediction_actual = scaler.inverse_transform(dummy_pred)[0, 0]

                st.success("Prediction Complete!")
                last_close_price = final_df['Close'].iloc[-1]
                
                col1, col2 = st.columns(2)
                col1.metric(label="Last Available Closing Price", value=f"₹{last_close_price:.2f}")
                col2.metric(label="Predicted Price for Next Day", value=f"₹{prediction_actual:.2f}", delta=f"₹{prediction_actual - last_close_price:.2f}")

                if prediction_actual > last_close_price:
                    st.write("### Conclusion: The model predicts the stock price will **GO UP**. 📈")
                else:
                    st.write("### Conclusion: The model predicts the stock price will **GO DOWN**. 📉")

except FileNotFoundError as e:
    st.error(f"A required file was not found. Please make sure `final_master_dataset.csv`, `lstm_model.h5`, and `scaler.pkl` are in the same folder as the app. Missing file: {e.filename}")
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")