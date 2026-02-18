import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt

from utils import preprocess_input, categorize_risk, recommended_action

st.set_page_config(page_title="AI Fraud Detection", layout="wide")

st.title("💳 AI Fraud Detection System")

# ================= LOAD FILES =================
model = pickle.load(open("app/model.pkl", "rb"))
scaler = pickle.load(open("app/scaler.pkl", "rb"))
trained_features = pickle.load(open("app/features.pkl", "rb"))
explainer = pickle.load(open("app/shap_explainer.pkl", "rb"))

# ================= MENU =================
menu = st.sidebar.selectbox("Navigation", [
    "🏠 Dashboard",
    "🔍 Single Prediction",
    "📂 Batch Prediction",
    "📊 Fraud Analytics"
])

# ================= DASHBOARD =================
if menu == "🏠 Dashboard":

    st.subheader("System Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Model Type", "XGBoost")
    col2.metric("Features Used", len(trained_features))
    col3.metric("Status", "Active")

# ================= SINGLE PREDICTION =================
elif menu == "🔍 Single Prediction":

    st.subheader("Enter Transaction Details")

    col1, col2 = st.columns(2)

    with col1:
        step = st.number_input("Step", 1, 1000, 1)
        amount = st.number_input("Amount", 0.0, 1000000.0, 1000.0)
        oldbalanceOrg = st.number_input("Old Balance Origin", 0.0, 1000000.0, 5000.0)
        newbalanceOrig = st.number_input("New Balance Origin", 0.0, 1000000.0, 4000.0)

    with col2:
        oldbalanceDest = st.number_input("Old Balance Dest", 0.0, 1000000.0, 0.0)
        newbalanceDest = st.number_input("New Balance Dest", 0.0, 1000000.0, 1000.0)
        transaction_type = st.selectbox("Transaction Type",
            ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"])

    if st.button("Predict Risk"):

        input_df = pd.DataFrame({
            "step":[step],
            "type":[transaction_type],
            "amount":[amount],
            "oldbalanceOrg":[oldbalanceOrg],
            "newbalanceOrig":[newbalanceOrig],
            "oldbalanceDest":[oldbalanceDest],
            "newbalanceDest":[newbalanceDest]
        })

        processed = preprocess_input(input_df, trained_features)
        scaled = scaler.transform(processed)

        risk_score = model.predict_proba(scaled)[0][1] * 100
        risk_label = categorize_risk(risk_score)

        st.metric("Risk Score", f"{risk_score:.2f}%")
        st.success(f"Risk Category: {risk_label}")
        st.warning(recommended_action(risk_score))

        # SHAP
        st.subheader("🔎 SHAP Explanation")

        shap_values = explainer.shap_values(scaled)

        fig, ax = plt.subplots()
        shap.plots._waterfall.waterfall_legacy(
            explainer.expected_value,
            shap_values[0],
            feature_names=trained_features,
            show=False
        )
        st.pyplot(fig)

# ================= BATCH =================
elif menu == "📂 Batch Prediction":

    st.subheader("Upload CSV")

    file = st.file_uploader("Upload CSV file", type=["csv"])

    if file:

        df = pd.read_csv(file)

        processed = preprocess_input(df, trained_features)
        scaled = scaler.transform(processed)

        df["Risk Score"] = model.predict_proba(scaled)[:,1] * 100
        df["Risk Category"] = df["Risk Score"].apply(categorize_risk)

        st.success("Prediction completed")
        st.dataframe(df.head())

        st.download_button(
            "Download Results",
            df.to_csv(index=False),
            file_name="fraud_predictions.csv"
        )

# ================= ANALYTICS =================
elif menu == "📊 Fraud Analytics":

    st.subheader("📊 Advanced Fraud Monitoring Dashboard")

    file = st.file_uploader("Upload Transaction CSV", type=["csv"])

    if file is not None:

        df = pd.read_csv(file)

        processed = preprocess_input(df, trained_features)
        scaled = scaler.transform(processed)

        df["Risk Score"] = model.predict_proba(scaled)[:,1] * 100
        df["Risk Category"] = df["Risk Score"].apply(categorize_risk)

        st.success("Analysis Completed")

        # ===================== BASIC METRICS =====================
        total = len(df)
        avg_risk = df["Risk Score"].mean()
        high_risk = (df["Risk Category"] == "HIGH RISK").sum()

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Transactions", total)
        col2.metric("Average Risk Score", f"{avg_risk:.2f}%")
        col3.metric("High Risk Transactions", high_risk)

        # ===================== 1️⃣ RISK TREND OVER TIME =====================
        if "step" in df.columns:
            st.subheader("📈 Risk Trend Over Time")
            trend = df.groupby("step")["Risk Score"].mean()
            st.line_chart(trend)

        # ===================== 2️⃣ ROLLING RISK INDEX =====================
        st.subheader("📊 Fraud Pressure Index (Rolling Average)")
        df["Rolling Risk"] = df["Risk Score"].rolling(window=20, min_periods=1).mean()
        st.line_chart(df["Rolling Risk"])

        # ===================== 3️⃣ TRANSACTION TYPE RISK HEATMAP =====================
        if "type" in df.columns:
            st.subheader("🔥 Risk by Transaction Type")
            type_risk = df.groupby("type")["Risk Score"].mean()
            st.bar_chart(type_risk)

        # ===================== 4️⃣ HIGH RISK SPIKE DETECTION =====================
        st.subheader("⚠️ High Risk Spike Detection")
        threshold_spike = avg_risk + 2 * df["Risk Score"].std()
        spikes = df[df["Risk Score"] > threshold_spike]

        st.write(f"Dynamic Spike Threshold: {threshold_spike:.2f}%")
        st.write(f"Spike Transactions Detected: {len(spikes)}")
        st.dataframe(spikes.head(5))

        # ===================== 5️⃣ EXPECTED VS HIGH RISK =====================
        st.subheader("📊 Expected vs High Risk Comparison")

        expected_normal = total - high_risk
        comparison_df = pd.DataFrame({
            "Category": ["Expected Normal", "High Risk"],
            "Count": [expected_normal, high_risk]
        })

        st.bar_chart(comparison_df.set_index("Category"))

        # ===================== 6️⃣ RISK THRESHOLD SIMULATOR =====================
        st.subheader("🎯 Risk Threshold Simulator")

        threshold = st.slider("Set Custom Risk Threshold", 0, 100, 70)

        flagged = (df["Risk Score"] >= threshold).sum()

        st.write(f"Transactions flagged at {threshold}% threshold: {flagged}")

        # ===================== DOWNLOAD =====================
        st.download_button(
            "Download Full Analysis",
            df.to_csv(index=False),
            file_name="advanced_fraud_analysis.csv"
        )


