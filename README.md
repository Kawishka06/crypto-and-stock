# ⚙️ Stock & Crypto Forecasting API (Server App)

A **production-style machine learning inference service** that exposes trained LSTM models as RESTful APIs for real-time stock and cryptocurrency forecasting.

This service acts as the **core ML backend** for both the dashboard and the chatbot.

---

## 🔹 Features

- LSTM-based time series forecasting
- Supports:
  - BTC-USD
  - ETH-USD
  - SL20 Synthetic Index (Sri Lankan market use case)
- Configurable forecast horizon
- Clean REST API design
- Model + scaler loading at runtime
- Designed for cloud deployment

---

## 🧠 Why Synthetic SL20?

Due to limited availability of public historical data for the Sri Lankan stock market, a **synthetic SL20 dataset** is used to:
- Preserve realistic market dynamics
- Enable experimentation with local market behavior
- Focus on system design and deployment rather than data access

---

## 🛠️ Tech Stack

- Python
- FastAPI
- TensorFlow / Keras (LSTM)
- Pandas, NumPy
- Joblib
- Render (deployment)

---

##▶️ Run Locally

pip install -r requirements.txt
uvicorn api:app --reload

## ▶️ API Endpoints

### Health Check
