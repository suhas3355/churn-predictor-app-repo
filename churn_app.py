import os
os.makedirs("models", exist_ok=True)
import pandas as pd
import joblib
import streamlit as st
from train_utils import train_model_for_business
import matplotlib.pyplot as plt

# --------------------- PAGE SETUP ---------------------
st.set_page_config(page_title="Retention Intelligence Tool", layout="wide")
st.markdown("## 📊 Retention Intelligence Tool")
st.markdown("---")

# --------------------- SIDEBAR NAV ---------------------
st.sidebar.markdown("### 🧭 Select Mode", unsafe_allow_html=True)

tab_options = {
    "📈 Train Business Model": "📈 Train Business Model",
    "🔍 Predict Churn": "🔍 Predict Churn"
}

selected_tab = st.sidebar.radio(
    "Navigation",
    list(tab_options.keys()),
    index=0,
    format_func=lambda x: f"{tab_options[x]}",
    label_visibility="collapsed"
)

# --------------------- TRAIN TAB ---------------------
if selected_tab == "📈 Train Business Model":
    st.subheader("🛠️ Train Churn Model for Your Business")

    business_id = st.text_input("Enter Business or Client Name", placeholder="e.g., acme_co")
    uploaded_file = st.file_uploader("Upload historical churn data (CSV with a 'Churn' column)", type=["csv"], key="train_csv")

    if uploaded_file and business_id:
        df = pd.read_csv(uploaded_file)

        if st.button("🚀 Train Model"):
            with st.spinner(f"Training churn model for '{business_id}'..."):
                try:
                    result = train_model_for_business(df, business_id)
                    st.success(f"✅ Model successfully trained for **{business_id}**!")
                except Exception as e:
                    st.error(f"❌ Training failed: {e}")
    else:
        st.info("Please enter a business name and upload a valid CSV file.")

# --------------------- PREDICT TAB ---------------------
elif selected_tab == "🔍 Predict Churn":
    st.subheader("📉 Churn Score Calculator")

    available_models = [d for d in os.listdir("models") if os.path.isdir(f"models/{d}")]

    if not available_models:
        st.warning("No trained models found. Please train one first.")
        st.stop()

    selected_model = st.selectbox("Select your model to predict churn", available_models)

    # Load model and features
    model_path = f"models/{selected_model}/churn_model.pkl"
    features_path = f"models/{selected_model}/model_features.pkl"

    try:
        model = joblib.load(model_path)
        features = joblib.load(features_path)
    except:
        st.error("Unable to load the selected model.")
        st.stop()

    # CSV Upload
    uploaded_csv = st.file_uploader("Upload new customer data for churn prediction", type=["csv"], key="predict_csv")

    if uploaded_csv:
        try:
            raw_df = pd.read_csv(uploaded_csv)

            def preprocess_uploaded_data(df, required_features):
                if "Churn" in df.columns:
                    df = df.drop(columns=["Churn"])
                df = pd.get_dummies(df, drop_first=False)
                for col in required_features:
                    if col not in df.columns:
                        df[col] = 0
                df = df[required_features]
                return df

            input_df = preprocess_uploaded_data(raw_df.copy(), features)

            # Predict
            raw_df["ChurnScore"] = model.predict_proba(input_df)[:, 1]
            raw_df["RiskLevel"] = raw_df["ChurnScore"].apply(
                lambda x: "High" if x > 0.75 else "Medium" if x > 0.4 else "Low"
            )

            st.success("✅ Churn predictions generated!")
            st.dataframe(raw_df[["CustomerID", "ChurnScore", "RiskLevel"]].head())

            # Metrics
            avg_score = round(raw_df["ChurnScore"].mean(), 2)
            high_risk = (raw_df["ChurnScore"] > 0.75).sum()
            total_rows = len(raw_df)
            churned = (raw_df["ChurnScore"] >= 0.5).sum()
            not_churned = (raw_df["ChurnScore"] < 0.5).sum()

            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("🔁 Avg Score", avg_score)
            col2.metric("🚨 High Risk", high_risk)
            col3.metric("🧾 Total Rows", total_rows)
            col4.metric("❌ Churn", churned)
            col5.metric("✅ No Churn", not_churned)

            st.subheader("📊 Visual Summary")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("#### 🥧 Churn Breakdown")
                fig1, ax1 = plt.subplots(figsize=(2, 2))
                ax1.pie([churned, not_churned], labels=["Churn", "No Churn"], autopct="%1.1f%%", startangle=90)
                ax1.axis("equal")
                st.pyplot(fig1, use_container_width=False)

            with col2:
                st.markdown("#### 📊 Risk Level Distribution")
                risk_counts = raw_df["RiskLevel"].value_counts()
                fig2, ax2 = plt.subplots(figsize=(2, 2))
                risk_counts.plot(kind="bar", color=["red", "orange", "green"], ax=ax2)
                ax2.set_ylabel("Customers")
                fig2.tight_layout()
                st.pyplot(fig2, use_container_width=False)

            with col3:
                st.markdown("#### 📈 Churn Score Histogram")
                fig3, ax3 = plt.subplots(figsize=(2, 2))
                ax3.hist(raw_df["ChurnScore"], bins=10, color="skyblue", edgecolor="black")
                ax3.set_xlabel("Score")
                ax3.set_ylabel("Count")
                fig3.tight_layout()
                st.pyplot(fig3, use_container_width=False)

            # Download
            csv = raw_df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Full Results CSV", csv, file_name="churn_predictions.csv")

        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")

# --------------------- FOOTER ---------------------
st.markdown("---")
st.markdown("🔐 [Privacy Policy](#) | 📫 [Contact](mailto:suhas3355@gmail.com) | 💻 [GitHub](https://github.com/suhas3355/churn-predictor-app-repo)")
