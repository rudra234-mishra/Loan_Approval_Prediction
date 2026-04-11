# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 11:34:08 2026

@author: rudra
"""

import pickle
import streamlit as st
import pandas as pd
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="💰",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = pickle.load(open("C:/Users/rudra/Downloads/Loan_Model.pkl", "rb"))
    return model

pipe = load_model()

# ---------------- HEADER ----------------
st.title("💰 Loan Approval Prediction System")
st.markdown("### Machine Learning Voting Classifier Model")

st.info("Fill applicant details from the sidebar and click **Predict Loan Status**")

# ---------------- SIDEBAR INPUT ----------------
st.sidebar.header("📋 Applicant Details")

Gender = st.sidebar.radio("Gender", ['Male','Female'])
Married = st.sidebar.radio("Marital Status", ['No','Yes'])
Dependents = st.sidebar.selectbox("Dependents", [0,1,2,4])
Education = st.sidebar.radio("Education", ['Not Graduate','Graduate'])
Self_Employed = st.sidebar.radio("Self Employed", ['Yes','No'])

ApplicantIncome = st.sidebar.slider("Applicant Income", 0, 100000, 5000)
CoapplicantIncome = st.sidebar.slider("Coapplicant Income", 0, 50000, 2000)
LoanAmount = st.sidebar.slider("Loan Amount (in thousands)", 0, 700, 150)

Loan_Amount_Term = st.sidebar.selectbox(
    "Loan Term (months)",
    [12,36,60,84,120,180,240,300,360]
)

Credit_History = st.sidebar.radio(
    "Credit History",
    [1,0],
    help="1 = Good history, 0 = Bad history"
)

Property_Area = st.sidebar.selectbox(
    "Property Area",
    ['Rural','Urban','Semiurban']
)

# ---------------- TABS ----------------
tab1, tab2 = st.tabs(["🔍 Prediction", "📊 Input Summary"])

# Create dataframe
input_df = pd.DataFrame({
    "Gender":[Gender],
    "Married":[Married],
    "Dependents":[Dependents],
    "Education":[Education],
    "Self_Employed":[Self_Employed],
    "ApplicantIncome":[ApplicantIncome],
    "CoapplicantIncome":[CoapplicantIncome],
    "LoanAmount":[LoanAmount],
    "Loan_Amount_Term":[Loan_Amount_Term],
    "Credit_History":[Credit_History],
    "Property_Area":[Property_Area]
})

# ---------------- PREDICTION TAB ----------------
with tab1:

    if st.button("🚀 Predict Loan Status"):

        with st.spinner("Analyzing applicant profile..."):
            time.sleep(2)

            prediction = pipe.predict(input_df)[0]
            probability = pipe.predict_proba(input_df)[0][1]

        col1, col2 = st.columns(2)

        # RESULT COLUMN
        with col1:
            if prediction == 1:
                st.success("✅ Loan Approved")
                st.balloons()
            else:
                st.error("❌ Loan Not Approved")

                # Main reason message
                st.warning("⚠️ Your details do not fulfill our loan approval criteria.")

                # -------- SMART REASON LOGIC --------
                reasons = []

                if Credit_History == 0:
                    reasons.append("Poor credit history")

                if ApplicantIncome < 2500:
                    reasons.append("Low applicant income")

                if LoanAmount > (ApplicantIncome / 10):
                    reasons.append("Requested loan amount is too high compared to income")

                if Loan_Amount_Term < 60:
                    reasons.append("Loan term is too short")

                if Self_Employed == "Yes" and ApplicantIncome < 3000:
                    reasons.append("Unstable income for self-employed applicant")

                # Show reasons
                if len(reasons) > 0:
                    st.markdown("### 🔍 Possible Reasons:")
                    for r in reasons:
                        st.write(f"- {r}")
                else:
                    st.write("Your profile does not match the model's approval pattern.")

        # PROBABILITY COLUMN
        with col2:
            st.metric("Approval Probability", f"{probability*100:.2f}%")
            st.progress(int(probability*100))

# ---------------- SUMMARY TAB ----------------
with tab2:
    st.subheader("Entered Applicant Information")
    st.dataframe(input_df, use_container_width=True)

    st.markdown("### Income Visualization")

    chart_data = pd.DataFrame({
        'Income Type':['Applicant','Coapplicant'],
        'Income':[ApplicantIncome, CoapplicantIncome]
    })

    st.bar_chart(chart_data.set_index('Income Types'))