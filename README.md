# 📈 Reliance Industries Stock Trend Predictor

This is an advanced data science capstone project that predicts the stock trend for Reliance Industries. It uses a hybrid model that combines time-series analysis with Natural Language Processing (NLP) of financial news.

---

## 🚀 Live Application

You can use the live, deployed version of this app right here:

**[https://stock-trend-prediction-with-nlp-hfjz2upal8yszcfa8ck2by.streamlit.app/](https://stock-trend-prediction-with-nlp-hfjz2upal8yszcfa8ck2by.streamlit.app/)**



---

## 🛠️ Key Features & Technologies

This project demonstrates a complete, end-to-end data science pipeline:

* **Deep Learning Model:** A **Long Short-Term Memory (LSTM)** network is used to analyze patterns in 60 days of historical stock data.
* **NLP Sentiment Analysis:** The model analyzes real-time financial news using **FinBERT**, a state-of-the-art AI model trained on financial text.
* **Technical Indicators:** The model is enriched with technical indicators (SMA and RSI) to capture market momentum.
* **Interactive Dashboard:** The entire application is built with **Streamlit** to be a user-friendly, interactive web app.
* **Core Stack:** Python, TensorFlow/Keras, Pandas, NumPy, Scikit-learn, and Joblib.

---

## ⚙️ How It Works

The app runs on a pre-trained model and a processed dataset (`final_master_dataset.csv`). When the user clicks "Predict," the app does the following:

1.  **Loads Data:** It loads the pre-processed historical dataset, which includes stock prices, technical indicators, and sentiment scores.
2.  **Loads Models:** It loads the pre-trained `lstm_model.h5` and the `scaler.pkl` files.
3.  **Prepares Sequence:** It takes the last 60 days of data from the dataset.
4.  **Generates Prediction:** It feeds this 60-day sequence into the LSTM model to generate a prediction for the next day's price.
5.  **Displays Results:** It shows the user the last closing price and the model's prediction, along with a simple "GO UP" 📈 or "GO DOWN" 📉 conclusion.

---

## 💻 How to Run This Project Locally

If you want to run this project on your own machine, you can follow these steps:

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/dev-kumar12/Stock-Trend-Prediction-with-NLP.git](https://github.com/dev-kumar12/Stock-Trend-Prediction-with-NLP.git)
    cd Stock-Trend-Prediction-with-NLP
    ```

2.  **Create a Conda Environment:**
    * This project uses Python 3.9.
    * You will need to install the libraries in the `requirements.txt` file.
    ```bash
    conda create -n stock_app python=3.9
    conda activate stock_app
    pip install -r requirements.txt
    ```
    
3.  **(Optional) Re-run the Notebook:**
    * To generate fresh data and re-train the model, run the `Data_Collection.ipynb` notebook from top to bottom. This will take a long time (30+ minutes).

4.  **Run the Streamlit App:**
    ```bash
    streamlit run app.py
    ```