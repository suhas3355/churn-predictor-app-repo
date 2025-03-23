import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model and expected feature list
model = joblib.load("churn_model.pkl")
feature_columns = joblib.load("model_features.pkl")

st.title("🛍️🛒 Customer Churn Predictor")
st.write("Provide customer info below to predict churn. Some fields are optional.")

# --- Required Inputs ---
tenure = st.slider("Tenure (Months)", 0, 60, 12)
order_count = st.slider("Order Count", 0, 20, 5)
hours = st.slider("Hours Spent on App", 0.0, 10.0, 3.5)
satisfaction = st.slider("Satisfaction Score", 1, 5, 3)
cashback = st.slider("Cashback Amount ($)", 0.0, 100.0, 20.0)

# --- Optional Inputs ---
with st.expander("🔧 Advanced Options (Optional Features)", expanded=False):
    optional_values = {}
    for col in feature_columns:
        if col not in ["Tenure", "OrderCount", "HourSpendOnApp", "SatisfactionScore", "CashbackAmount"]:
            if "_M" in col or "_F" in col or "True" in col:
                optional_values[col] = st.selectbox(col, [0, 1], index=0)
            else:
                optional_values[col] = st.number_input(col, value=0.0)

# --- Build input dict with default 0s ---
input_dict = dict.fromkeys(feature_columns, 0)

# Fill in the required values
input_dict["Tenure"] = tenure
input_dict["OrderCount"] = order_count
input_dict["HourSpendOnApp"] = hours
input_dict["SatisfactionScore"] = satisfaction
input_dict["CashbackAmount"] = cashback

# Fill optional values
input_dict.update(optional_values)

# Predict
if st.button("Predict Churn"):
    input_df = pd.DataFrame([input_dict])
    prediction = model.predict(input_df)[0]
    result = "🚨 Likely to Churn" if prediction == 1 else "✅ Likely to Stay"
    st.success(f"Prediction: {result}")
