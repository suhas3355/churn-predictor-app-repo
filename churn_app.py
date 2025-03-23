import pandas as pd
import joblib
import streamlit as st

st.title("🛒🛍️ Customer Churn Predictor")

# Load model and expected feature columns
model = joblib.load("churn_model.pkl")
features = joblib.load("model_features.pkl")

# === Step 1: Capture required numeric inputs ===
st.header("🔹 Required Inputs")
tenure = st.slider("Tenure", 0, 60, 12, key="tenure")
order_count = st.slider("Order Count", 0, 20, 5, key="order_count")
hours = st.slider("Hours Spent on App", 0.0, 10.0, 3.5, key="hours")
satisfaction = st.slider("Satisfaction Score", 1, 5, 3, key="satisfaction")
cashback = st.slider("Cashback Amount", 0.0, 100.0, 20.0, key="cashback")


# === Step 2: Optional one-hot encoded (dummy) features ===

dummy_inputs = {}

with st.expander("🔧 Advanced Options", expanded=False):
    for col in features:
        if col not in ["Tenure", "OrderCount", "HourSpendOnApp", "SatisfactionScore", "CashbackAmount"]:
            dummy_inputs[col] = st.selectbox(f"{col}", [0, 1], index=0, key=col)


optional_values = {}



# === Step 3: Create the full input dict ===
input_dict = dict.fromkeys(features, 0)
input_dict["Tenure"] = tenure
input_dict["OrderCount"] = order_count
input_dict["HourSpendOnApp"] = hours
input_dict["SatisfactionScore"] = satisfaction
input_dict["CashbackAmount"] = cashback

# Optional dummy vars
for key, value in dummy_inputs.items():
    if key in input_dict:
        input_dict[key] = value

# Fill in optional dummy variables
input_dict.update(optional_values)

# Convert to DataFrame
input_df = pd.DataFrame([input_dict])

# Show what the model sees
with st.expander("📋 Final Input to Model (click to view)", expanded=False):
    st.table(input_df.T.rename(columns={0: "Value"}))


# === Step 4: Predict ===
if st.button("Predict Churn"):
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    result = "🚨 Likely to Churn" if proba >= 0.5 else "✅ Likely to Stay"
    st.success(f"Prediction: {result}")
    st.write(f"🧠 Churn Probability: {proba:.2f}")



