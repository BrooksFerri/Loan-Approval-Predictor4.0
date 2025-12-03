# -*- coding: utf-8 -*-
import streamlit as st
import pickle
import pandas as pd

# Load the trained model
with open("BUS458Final.pkl", "rb") as file:
    model = pickle.load(file)

# Title for the app
st.markdown(
    "<h1 style='text-align: center; background-color: #ffcccc; padding: 10px; color: #cc0000;'><b>Loan Approval Predictor</b></h1>",
    unsafe_allow_html=True
)

# Input fields
st.header("Enter Loan Applicant's Details")

# Numeric inputs
requested_loan = st.slider("Requested Loan Amount", min_value=0, max_value=50000, step=500)
fico_score = st.slider("FICO Score", min_value=300, max_value=850, step=5)
monthly_income = st.slider("Monthly Gross Income", min_value=0, max_value=20000, step=100)
housing_payment = st.slider("Monthly Housing Payment", min_value=0, max_value=5000, step=50)
bankrupt = st.selectbox("Ever Bankrupt or Foreclosure?", [0, 1])

# Categorical inputs
reason = st.selectbox("Reason for Loan", ["DebtConsolidation", "HomeImprovement", "Other", "Business", "Major_purchase", "Medical"])
fico_group = st.selectbox("FICO Score Group", ["Poor", "Fair", "Good", "Very Good", "Exceptional"])
employment_status = st.selectbox("Employment Status", ["employed", "self-employed", "unemployed"])
employment_sector = st.selectbox("Employment Sector", ["Healthcare", "Technology", "Finance", "Education", "Retail", "Other"])
lender = st.selectbox("Lender", ["Lender_A", "Lender_B", "Lender_C"])

# Create the input data as a DataFrame
input_data = pd.DataFrame({
    "Requested_Loan_Amount": [requested_loan],
    "FICO_score": [fico_score],
    "Monthly_Gross_Income": [monthly_income],
    "Monthly_Housing_Payment": [housing_payment],
    "Ever_Bankrupt_or_Foreclose": [bankrupt],
    "Reason": [reason],
    "Fico_Score_group": [fico_group],
    "Employment_Status": [employment_status],
    "Employment_Sector": [employment_sector],
    "Lender": [lender]
})

# --- Prepare Data for Prediction ---
# 1. One-hot encode the user's input
input_data_encoded = pd.get_dummies(input_data, columns=['Reason', 'Fico_Score_group', 'Employment_Status', 'Employment_Sector', 'Lender'], drop_first=True)

# 2. Add any "missing" columns the model expects (fill with 0)
model_columns = model.feature_names_in_
for col in model_columns:
    if col not in input_data_encoded.columns:
        input_data_encoded[col] = 0

# 3. Reorder/filter columns to exactly match the model's training data
input_data_encoded = input_data_encoded[model_columns]

# Predict button
if st.button("Evaluate Loan Application"):
    # Predict using the loaded model
    prediction = model.predict(input_data_encoded)[0]
    probability = model.predict_proba(input_data_encoded)[0]

    # Display result
    if prediction == 1:
        st.success(f"✅ **Loan APPROVED** (Confidence: {probability[1]:.1%})")
    else:
        st.error(f"❌ **Loan DENIED** (Confidence: {probability[0]:.1%})")
