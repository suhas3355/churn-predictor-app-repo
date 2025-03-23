import pandas as pd
import joblib
import streamlit as st

st.title("🛒🛍️ Customer Churn Predictor")

# Load model and expected feature columns
model = joblib.load("churn_model.pkl")
features = joblib.load("model_features.pkl")

# === Step 1: Capture required numeric inputs ===
st.header("Required Inputs")

tenure = st.slider("🗓️Tenure (Months)", 0, 60, 12, help="How long the customer has been active, in months.")
order_count = st.slider("📦Number of Orders (Last 12 Months)", 0, 50, 5, help="Total orders placed by the customer in the past year.")
hours = st.slider("⏱️Hours Spent on App (Monthly Avg)", 0.0, 10.0, 3.5, help="Average number of hours spent on the app per month.")
satisfaction = st.slider("🎯Satisfaction Score (1 = Low, 5 = High)", 1, 5, 3, help="Customer-reported satisfaction score.")
cashback = st.slider("💵Cashback Amount (USD per Order)", 0.0, 100.0, 20.0, help="Average cashback the customer receives per order in USD.")


# Map original categorical fields to their encoded dummy columns
original_categoricals = {
    "PreferredLoginDevice": ["Computer", "Mobile Phone", "Tablet"],
    "PreferredPaymentMode": ["Credit Card", "Debit Card", "NetBanking", "UPI", "E wallet"],
    "Gender": ["F", "M"],
    "MaritalStatus": ["Married", "Single"],
    "PreferredOrderCat": ["Grocery", "Laptop & Accessory", "Mobile", "Mobile Phone", "Others"],
    "CityTier": ["1", "2", "3"],
    "Complain": ["No", "Yes"]
}
# Identify dummy columns from your model features
top_dummy_cols = []
for k, v_list in original_categoricals.items():
    top_dummy_cols.extend([f"{k}_{v}" for v in v_list if f"{k}_{v}" in features])

# Remaining dummy features not covered above
remaining_dummy_cols = [col for col in features
                        if col not in top_dummy_cols
                        and col not in ["Tenure", "OrderCount", "HourSpendOnApp", "SatisfactionScore", "CashbackAmount"]]


# === Step 2: Optional one-hot encoded (dummy) features ===

dummy_inputs = {}

optional_values = {}

# Friendly labels
label_map = {
    "Gender_M": "Gender",
    "PreferredLoginDevice_Mobile Phone": "Login Device",
    "PreferredPaymentMode_Credit Card": "Payment Mode",
    "PreferredOrderCat_Mobile": "Order Category: Mobile",
    "MaritalStatus_Single": "Marital Status",
    "Complain": "Filed a Complaint?",
    "CityTier_3": "City Tier"
}

# Top 10 influential variables (only dummy ones shown here)
top_features_order = [
    "PreferredLoginDevice_Mobile Phone",
    "PreferredPaymentMode_Credit Card",
    "Gender_M",
    "PreferredOrderCat_Mobile",
    "Complain",
]

# Collect dummy features only (excluding numeric inputs)
dummy_feature_cols = [
    col for col in features
    if col not in ["Tenure", "OrderCount", "HourSpendOnApp", "SatisfactionScore", "CashbackAmount"]
]

# Sort: top features first, then remaining alphabetically
ordered_dummies = top_features_order + sorted(
    [col for col in dummy_feature_cols if col not in top_features_order]
)

# Build form
# Collect user selection in a friendly way
user_inputs = {}
optional_values = {}

with st.expander("🔧 Advanced variables (Top Features)", expanded=True):
    for feature, options in original_categoricals.items():
        selected = st.selectbox(f"{feature.replace('_', ' ')}", options, index=0)
        user_inputs[feature] = selected

with st.expander("🔧 Additional variables", expanded=False):
    for col in sorted(remaining_dummy_cols):
        label = col.replace("_", " ")
        val = st.selectbox(f"{label}", ["No", "Yes"], index=0, key=f"extra_{col}")
        optional_values[col] = 1 if val == "Yes" else 0



# === Step 3: Create the full input dict ===
# Start with all zeros
input_dict = dict.fromkeys(features, 0)

# Required numeric inputs
input_dict["Tenure"] = tenure
input_dict["OrderCount"] = order_count
input_dict["HourSpendOnApp"] = hours
input_dict["SatisfactionScore"] = satisfaction
input_dict["CashbackAmount"] = cashback

# Top categorical features
for original, selected in user_inputs.items():
    dummy_col = f"{original}_{selected}"
    if dummy_col in input_dict:
        input_dict[dummy_col] = 1

# Additional dummy variables
for col, val in optional_values.items():
    input_dict[col] = val


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



