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

    st.subheader("Fraud Analytics Overview")

    st.info("Analytics dashboard will display insights from live predictions.")

    st.write("Model Type: XGBoost")
    st.write("Number of Features:", len(trained_features))
    st.write("SHAP Enabled: Yes")

