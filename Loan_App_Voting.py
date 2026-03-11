# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 21:34:44 2026

@author: rudra
"""

import pickle
import streamlit as st
import pandas as pd
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Smart Loan Advisor", page_icon="🏦", layout="wide")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return pickle.load(open("C:/Users/rudra/Downloads/Voting_Classifier.pkl","rb"))

pipe = load_model()

# ---------------- HEADER ----------------
st.title("🏦 Smart Loan Approval Advisor")
st.caption("AI powered loan risk evaluation system")

# ---------------- SIDEBAR ----------------
st.sidebar.header("🧾 Applicant Profile")

Gender = st.sidebar.radio("Gender", ['Male','Female'])
Married = st.sidebar.radio("Marital Status", ['No','Yes'])
Dependents = st.sidebar.selectbox("Dependents", [0,1,2,3])
Education = st.sidebar.radio("Education", ['Not Graduate','Graduate'])
Self_Employed = st.sidebar.radio("Self Employed", ['Yes','No'])

st.sidebar.divider()
st.sidebar.subheader("💰 Financial Details")

ApplicantIncome = st.sidebar.number_input("Applicant Monthly Income (₹)", 0, 200000, 15000)
CoapplicantIncome = st.sidebar.number_input("Co-Applicant Monthly Income (₹)", 0, 200000, 5000)
LoanAmount = st.sidebar.slider("Loan Amount (₹ in thousands)", 0, 700, 150)
Loan_Amount_Term = st.sidebar.selectbox("Loan Term (months)", [12,36,60,84,120,180,240,300,360])
Credit_History = st.sidebar.radio("Credit History", [1,0], format_func=lambda x: "Good" if x==1 else "Bad")
Property_Area = st.sidebar.selectbox("Property Area", ['Rural','Urban','Semiurban'])

st.sidebar.divider()

# RESET BUTTON
if st.sidebar.button("🔄 Reset Form"):
    st.rerun()

# ---------------- VALIDATION ----------------
if LoanAmount == 0:
    st.warning("⚠️ Loan amount cannot be zero")
    st.stop()

# ---------------- DATAFRAME ----------------
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

# ---------------- AUTO PREDICTION ----------------
prediction = pipe.predict(input_df)[0]
probability = pipe.predict_proba(input_df)[0][1]

# ---------------- MAIN LAYOUT ----------------
col1, col2, col3 = st.columns([1.2,1,1])

# ---------- RESULT ----------
with col1:
    st.subheader("📊 Decision")

    if prediction == 1:
        st.success("Loan Approved")
        st.balloons()
    else:
        st.error("Loan Rejected")

    st.metric("Approval Probability", f"{probability*100:.2f}%")

# ---------- RISK METER ----------
with col2:
    st.subheader("⚠️ Risk Level")

    risk = 1 - probability
    st.progress(int(risk*100))

    if risk < 0.3:
        st.success("Low Risk Customer")
    elif risk < 0.6:
        st.warning("Medium Risk Customer")
    else:
        st.error("High Risk Customer")

# ---------- EMI CALCULATOR ----------
with col3:
    st.subheader("💳 EMI Estimator")

    loan_rupees = LoanAmount * 1000
    r = 0.09/12  # 9% interest
    n = Loan_Amount_Term

    emi = (loan_rupees*r*(1+r)**n)/((1+r)**n - 1)

    st.metric("Estimated EMI", f"₹{emi:,.0f}/month")

# ---------------- RECOMMENDATION ENGINE ----------------
st.divider()
st.subheader("🤖 AI Recommendation")

if prediction == 0:
    suggestions = []

    if Credit_History == 0:
        suggestions.append("Improve your credit history")
    if ApplicantIncome < 8000:
        suggestions.append("Increase monthly income")
    if LoanAmount > 300:
        suggestions.append("Apply for smaller loan amount")
    if Dependents >= 3:
        suggestions.append("High dependents reduce approval chances")

    for s in suggestions:
        st.info("👉 " + s)
else:
    st.success("Your profile looks financially healthy. High chance of bank approval!")

# ---------------- USER INPUT VIEW ----------------
with st.expander("📋 View Submitted Application"):
    st.dataframe(input_df, use_container_width=True)