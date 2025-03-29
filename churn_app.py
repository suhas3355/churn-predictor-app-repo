def preprocess_uploaded_data(df, required_features):
    # Drop label column if present
    if "Churn" in df.columns:
        df = df.drop(columns=["Churn"])

    # One-hot encode all object (categorical) columns
    df = pd.get_dummies(df, drop_first=False)

    # Add missing columns (expected by model) with 0s
    for col in required_features:
        if col not in df.columns:
            df[col] = 0

    # Ensure same column order
    df = df[required_features]

    return df

import streamlit as st
import pandas as pd
import joblib

# Load model and features
model = joblib.load("churn_model.pkl")
features = joblib.load("model_features.pkl")

st.set_page_config(page_title="Churn Predictor", layout="wide")
st.title("📊 Churn Predictor - Upload & Score Customers")

# Upload CSV
st.subheader("1. Upload Customer File")
uploaded_file = st.file_uploader("Upload a CSV file with your customer data", type=["csv"])

if uploaded_file:
    try:
        raw_df = pd.read_csv(uploaded_file)

        # Clean and prepare input
        input_df = preprocess_uploaded_data(raw_df.copy(), features)

        # Predict
        raw_df["ChurnScore"] = model.predict_proba(input_df)[:, 1]
        raw_df["RiskLevel"] = raw_df["ChurnScore"].apply(
            lambda x: "High" if x > 0.75 else "Medium" if x > 0.4 else "Low"
        )

        st.success("✅ Churn predictions generated!")
        st.subheader("Preview")
        st.dataframe(raw_df[["CustomerID", "ChurnScore", "RiskLevel"]].head())

        st.metric("🔁 Average Churn Score", round(raw_df["ChurnScore"].mean(), 2))
        st.metric("🚨 High-Risk Customers", (raw_df["ChurnScore"] > 0.75).sum())

        # Download button
        csv = raw_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Full Results CSV", csv, file_name="churn_predictions.csv")

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")

# Optional Customer ID Lookup
st.markdown("---")
st.subheader("2. (Optional) Predict Single Customer by ID")

customer_id = st.text_input("Enter Customer ID")

if st.button("Predict Churn for This Customer") and uploaded_file and customer_id:
    if "raw_df" in locals():
        match = raw_df[raw_df["CustomerID"].astype(str) == str(customer_id)]
        if not match.empty:
            score = match["ChurnScore"].values[0]
            risk = match["RiskLevel"].values[0]
            st.success(f"Churn Score: {round(score, 2)} — Risk Level: {risk}")
        else:
            st.warning("Customer ID not found in uploaded file.")
    else:
        st.warning("Please upload a CSV file first.")


# Footer
st.markdown("---")
st.markdown("🔐 [Privacy Policy](#) | 📫 [Contact](mailto:your@email.com) | 💻 [GitHub](https://github.com/suhas3355/churn-predictor-app-repo)")
