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

        # Pie Chart: Churn vs. No Churn
        fig1, ax1 = plt.subplots(figsize=(4, 4))
        ax1.pie([churned, not_churned],
                labels=["Likely to Churn", "Likely to Stay"],
                autopct="%1.1f%%",
                startangle=90)
        ax1.axis("equal")

        # Bar Chart: Risk Level Breakdown
        risk_counts = raw_df["RiskLevel"].value_counts()
        fig2, ax2 = plt.subplots(figsize=(4, 4))
        risk_counts.plot(kind="bar", color=["red", "orange", "green"], ax=ax2)
        ax2.set_title("Customer Risk Levels")
        ax2.set_ylabel("Count")

        # Histogram: Churn Score Distribution
        fig3, ax3 = plt.subplots(figsize=(4, 4))
        ax3.hist(raw_df["ChurnScore"], bins=10, color="skyblue", edgecolor="black")
        ax3.set_title("Distribution of Churn Scores")
        ax3.set_xlabel("Churn Score")
        ax3.set_ylabel("Customers")

        # Layout in columns
        col1, col2, col3 = st.columns(3)

        with col1:
            st.pyplot(fig1)

        with col2:
            st.pyplot(fig2)

        with col3:
            st.pyplot(fig3)


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
st.markdown("🔐 [Privacy Policy](#) | 📫 [Contact](mailto:suhas3355@gmail.com) | 💻 [GitHub](https://github.com/suhas3355/churn-predictor-app-repo)")
