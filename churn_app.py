# At the top of churn_app.py
import os
import pandas as pd
import joblib
from train_utils import train_model_for_business  # ⬅️ Ensure this is imported

st.set_page_config(page_title="Churn Predictor", layout="wide")
st.title("📊 Churn Predictor - Upload & Score Customers")

# Add sidebar toggle
selected_tab = st.sidebar.radio("Choose Action", ["🔍 Predict Churn", "📈 Train Business Model"])

if selected_tab == "🔍 Predict Churn":
    st.subheader("Churn Prediction - Upload & Score")

# (Your existing prediction logic goes here — file upload, processing, scoring, metrics, charts, etc.)

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

        if "ChurnScore" in raw_df.columns:
            # Metrics
            avg_score = round(raw_df["ChurnScore"].mean(), 2)
            high_risk = (raw_df["ChurnScore"] > 0.75).sum()
            total_rows = len(raw_df)
            churned = (raw_df["ChurnScore"] >= 0.5).sum()
            not_churned = (raw_df["ChurnScore"] < 0.5).sum()

            # Display in horizontal blocks
            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.metric("🔁 Average Churn Score", avg_score)

            with col2:
                st.metric("🚨 High-Risk Customers", high_risk)

            with col3:
                st.metric("🧾 Rows Processed", total_rows)

            with col4:
                st.metric("❌ Likely to Churn", churned)

            with col5:
                st.metric("✅ Likely to Stay", not_churned)
        else:
            st.warning("⛔ChurnScore column not found — unable to show stats.")
        
        # Metrics (rows processed, churned, etc.)

        st.subheader("📊 Visual Summary")

        import matplotlib.pyplot as plt

        # Churned vs Not Churned
        churned = (raw_df["ChurnScore"] >= 0.5).sum()
        not_churned = (raw_df["ChurnScore"] < 0.5).sum()

        # Risk level count
        risk_counts = raw_df["RiskLevel"].value_counts()

        # Set up horizontal layout
        col1, col2, col3 = st.columns(3)

        # Pie Chart
        with col1:
            st.markdown("#### 🥧 Churn Breakdown")
            fig1, ax1 = plt.subplots(figsize=(2, 2))
            ax1.pie([churned, not_churned],
                    labels=["Churn", "No Churn"],
                    autopct="%1.1f%%",
                    startangle=90)
            ax1.axis("equal")
            st.pyplot(fig1, use_container_width=False)

        # Bar Chart
        with col2:
            st.markdown("#### 📊 Risk Level Distribution")
            fig2, ax2 = plt.subplots(figsize=(2, 2))
            risk_counts.plot(kind="bar", color=["red", "orange", "green"], ax=ax2)
            ax2.set_ylabel("Customers")
            fig2.tight_layout()
            st.pyplot(fig2, use_container_width=False)

        # Histogram
        with col3:
            st.markdown("#### 📈 Churn Score Distribution")
            fig3, ax3 = plt.subplots(figsize=(2, 2))
            ax3.hist(raw_df["ChurnScore"], bins=10, color="skyblue", edgecolor="black")
            ax3.set_xlabel("Churn Score")
            ax3.set_ylabel("Customers")
            fig3.tight_layout()
            st.pyplot(fig3, use_container_width=False)



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

elif selected_tab == "📈 Train Business Model":
    st.subheader("Train a Churn Model for Your Business")

    business_id = st.text_input("Enter Your Business or Client Name", placeholder="e.g., acme_co")

    uploaded_file = st.file_uploader(
        "Upload historical churn data (CSV with a 'Churn' column)", 
        type=["csv"], 
        key="train_file"
    )

    if uploaded_file and business_id:
        df = pd.read_csv(uploaded_file)

        if st.button("🚀 Train Model for This Business"):
            with st.spinner(f"Training churn model for '{business_id}'..."):
                try:
                    result = train_model_for_business(df, business_id)
                    st.success(f"✅ Model successfully trained and saved for **{business_id}**!")

                    st.markdown(f"📁 Model Path: `models/{business_id}/churn_model.pkl`")
                    st.markdown(f"🧬 Features Used: `{len(result['features'])}`")
                    st.markdown(f"📊 Total Rows After Balancing: `{result['rows_used']}`")
                    st.markdown(f"📉 Churn Rate in Uploaded Data: `{result['churn_rate']}`")

                except Exception as e:
                    st.error(f"❌ Training failed: {e}")
    else:
        st.info("Please enter a business name and upload a valid CSV file.")


# Footer
st.markdown("---")
st.markdown("🔐 [Privacy Policy](#) | 📫 [Contact](mailto:suhas3355@gmail.com) | 💻 [GitHub](https://github.com/suhas3355/churn-predictor-app-repo)")

#from train_utils import train_model_for_business
#import os

#st.header("📈 Train Churn Model for Your Business")

business_id = st.text_input("Enter Your Business ID (e.g., biz123)")

uploaded_training = st.file_uploader("Upload historical customer data (with 'Churn' column)", type=["csv"], key="train_upload")

if uploaded_training and business_id:
    df_train = pd.read_csv(uploaded_training)

    if st.button("🚀 Train My Model"):
        with st.spinner("Training your model..."):
            try:
                result = train_model_for_business(df_train, business_id)
                st.success(f"Model trained successfully for '{business_id}'")
                st.write(f"✅ Features used: {len(result['features'])}")
                st.write(f"📊 Rows after SMOTE: {result['rows_used']}")
                st.write(f"💡 Churn Rate: {result['churn_rate']}")
            except Exception as e:
                st.error(f"Training failed: {e}")
