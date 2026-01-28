import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Load model & feature list
# -----------------------------
model = joblib.load("loan_risk_model.pkl")
feature_names = joblib.load("model_features.pkl")

st.set_page_config(page_title="Loan Risk Scoring", layout="centered")

st.title("🏦 Explainable Loan Risk Scoring App")
st.write("Enter customer details to predict loan default risk")

# -----------------------------
# Input form
# -----------------------------
st.subheader("🧾 Customer Information")

input_data = {}

for feature in feature_names:
    input_data[feature] = st.number_input(
        label=feature,
        value=0.0
    )

input_df = pd.DataFrame([input_data])

# -----------------------------
# Prediction
# -----------------------------
if st.button("🔍 Predict Risk"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    # Risk buckets
    if probability < 0.33:
        risk = "LOW RISK"
        color = "🟢"
    elif probability < 0.66:
        risk = "MEDIUM RISK"
        color = "🟡"
    else:
        risk = "HIGH RISK"
        color = "🔴"

    st.markdown("---")
    st.subheader("📊 Risk Assessment Result")
    st.write(f"### {color} **{risk}**")
    st.write(f"**Default Probability:** {probability:.2f}")

    st.subheader("📄 Entered Customer Data")
    st.dataframe(input_df)
