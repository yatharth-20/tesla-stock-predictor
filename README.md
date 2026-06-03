# ⚡ Tesla (TSLA) Stock Price forecasting & EDA Dashboard

A deep learning time-series forecasting system and exploratory stock analysis dashboard for Tesla, Inc. (TSLA) stock prices, built using **Long Short-Term Memory (LSTM)** networks, **Simple Recurrent Neural Networks (SimpleRNN)**, **Plotly**, and **Streamlit**.

This project provides a robust framework for sequence-based financial forecasting, shifting from standard regression models to gated recurrent neural networks capable of modeling temporal dependencies in highly volatile financial markets.

---

## 🚀 Key Features

* **Exploratory Data Analysis (EDA):** Over 15 distinct visualizations in [Tesla_EDA_Completed.ipynb](Tesla_EDA_Completed.ipynb) analyzing stock trends, rolling volatility, volume-price breakouts, day-of-week seasonality, and autocorrelation profiles (ACF/PACF).
* **Deep Learning Forecasters:** Recurrent sequence models implemented in Keras ([Tesla_ML_Completed.ipynb](Tesla_ML_Completed.ipynb)) using a sliding window of the past **30 days** to predict closing prices.
* **Multi-Horizon Forecasting:** Separate, optimized LSTM networks trained for:
  - **1-Day Look-Ahead** (`tesla_lstm_1day.keras`)
  - **5-Day Look-Ahead** (`best_lstm_model_5day.keras`)
  - **10-Day Look-Ahead** (`best_lstm_model_10day.keras`)
* **Interactive Streamlit App:** A dark-themed, glassmorphism-styled dashboard (`tesla_app.py`) allowing users to:
  - Select prediction horizons (1, 5, or 10 days).
  - Calculate forecasts in real-time using actual or custom-simulated prices.
  - Interact with Plotly charts showing SMA crossover events, daily returns, and historical volatility trends.
* **Google Colab Support:** Notebooks are fully equipped with automatic file uploaders for seamless execution in the Colab cloud environment.

---

## 📊 Exploratory Data Analysis Overview

The dataset contains daily stock quotes of **TSLA** from its initial public offering in **2010 through early 2020** (2,416 trading records). Key findings from the EDA include:
1. **Right-Skewed Distribution:** Tesla's stock spent the majority of its first decade trading under $100 before experiencing exponential expansion starting in late 2019.
2. **Volatity Spikes:** The 20-day rolling volatility is highly non-constant, showing clustering behavior during company announcements and market stresses.
3. **Leptokurtic Returns:** Daily returns follow a distribution with extremely fat tails, meaning tail risks are highly prevalent.
4. **Strong Path-Dependency:** The ACF/PACF plots show high autocorrelation decay, justifying recurrent sequence modeling instead of simple static regressions.

---

## 🧠 Deep Learning Architecture

Recurrent architectures capture sequence memory by passing hidden states step-by-step.
* **SimpleRNN:** Leverages simple feedback units, but suffers from vanishing gradients over a 30-day window, creating a prediction lag.
* **LSTM:** Incorporates input, forget, and output gates along with a cell state to regulate information flow. This allows it to retain long-term dependencies, achieving an $R^2$ score of **0.985** on the 1-day forecast test split.

---

## 🛠️ Installation & Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yatharth-20/tesla-stock-predictor.git
   cd tesla-stock-predictor
   ```

2. **Install Dependencies:**
   Make sure you have Python 3.10+ installed. Install required packages:
   ```bash
   pip install tensorflow pandas numpy scikit-learn matplotlib seaborn plotly streamlit statsmodels
   ```

3. **Run the Jupyter Notebooks:**
   Open the notebooks to review the EDA visualizations or run the model training scripts:
   ```bash
   jupyter notebook Tesla_EDA_Completed.ipynb
   jupyter notebook Tesla_ML_Completed.ipynb
   ```

4. **Launch the Streamlit Dashboard:**
   Start the interactive web application locally:
   ```bash
   streamlit run tesla_app.py
   ```

---

## 📁 Repository Structure

```
├── TSLA.csv                      # Historical stock prices dataset
├── Tesla_EDA_Completed.ipynb     # Exploratory Data Analysis notebook (Pre-run)
├── Tesla_ML_Completed.ipynb      # Deep Learning model training notebook (Pre-run)
├── tesla_app.py                  # Interactive Streamlit application
├── tesla_lstm_1day.keras         # Saved 1-day forecast LSTM model checkpoint
├── best_lstm_model_5day.keras    # Saved 5-day forecast LSTM model checkpoint
├── best_lstm_model_10day.keras   # Saved 10-day forecast LSTM model checkpoint
├── LICENSE                       # Project license
└── README.md                     # Project documentation
```

---

## 📝 Submission Details
* **Project Name:** Tesla Stock Price Analysis & Forecasting
* **Project Type:** EDA / Deep Learning Regression
* **Contribution:** Individual
* **Team Member:** Yatharth Verma
* **GitHub Link:** [yatharth-20/tesla-stock-predictor](https://github.com/yatharth-20/tesla-stock-predictor)

---

*Disclaimer: This project is for educational purposes only. Stock trading involves substantial financial risk.*