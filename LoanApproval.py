# -*- coding: utf-8 -*-
import streamlit as st
import pickle
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Loan Approval Predictor",
    page_icon="💰",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
    <style>
    /* White background and clean styling */
    .stApp {
        background-color: #ffffff;
    }
    
    /* Modern title styling */
    .main-title {
        text-align: center;
        font-size: 3rem;
        font-weight: 700;
        color: #0a2463;;
        margin-bottom: 0.5rem;
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    
    /* Column headers */
    .column-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1f4788;
        margin-bottom: 1rem;
        border-bottom: 3px solid #1f4788;
        padding-bottom: 0.5rem;
    }
    
    /* Summary card styling */
    .summary-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        border-left: 5px solid #1f4788;
        margin: 1rem 0;
    }
    
    .summary-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1f4788;
        margin-bottom: 1rem;
    }
    
    .summary-section {
        margin-bottom: 1rem;
    }
    
    .summary-section-title {
        font-weight: 600;
        color: #333;
        margin-bottom: 0.5rem;
    }
    
    /* Gauge styling */
    .gauge-container {
        text-align: center;
        margin: 2rem 0;
    }
    
    /* Footer styling */
    .footer {
        background-color: #f8f9fa;
        padding: 2rem;
        margin-top: 3rem;
        border-top: 2px solid #e0e0e0;
        text-align: center;
        color: #666;
        font-size: 0.9rem;
    }
    
    .footer-section {
        margin: 0.5rem 0;
    }
    
    /* Button styling */
    div.stButton > button {
        background-color: #1f4788;
        color: white;
        font-size: 1.1rem;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        border: none;
        width: 100%;
        transition: all 0.3s;
    }
    
    div.stButton > button:hover {
        background-color: #163a6d;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    </style>
""", unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    with open("BUS458Final.pkl", "rb") as file:
        return pickle.load(file)

model = load_model()

# Title
st.markdown("<h1 class='main-title'>💰 Loan Approval Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Advanced ML-powered loan decision support system</p>", unsafe_allow_html=True)

# Create two columns for organized input
col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='column-header'>💵 Financial Information</div>", unsafe_allow_html=True)
    
    requested_loan = st.slider("Requested Loan Amount", min_value=0, max_value=50000, step=500, value=10000)
    fico_score = st.slider("FICO Score", min_value=300, max_value=850, step=5, value=650)
    fico_group = st.selectbox("FICO Score Group", ["Poor", "Fair", "Good", "Very Good", "Exceptional"], index=2)
    monthly_income = st.slider("Monthly Gross Income", min_value=0, max_value=20000, step=100, value=4000)
    housing_payment = st.slider("Monthly Housing Payment", min_value=0, max_value=5000, step=50, value=1000)
    
with col2:
    st.markdown("<div class='column-header'>👤 Personal Information</div>", unsafe_allow_html=True)
    
    reason = st.selectbox("Reason for Loan", ["DebtConsolidation", "HomeImprovement", "Other", "Business", "Major_purchase", "Medical"])
    employment_status = st.selectbox("Employment Status", ["employed", "self-employed", "unemployed"])
    employment_sector = st.selectbox("Employment Sector", ["Healthcare", "Technology", "Finance", "Education", "Retail", "Other"])
    lender = st.selectbox("Lender", ["Lender_A", "Lender_B", "Lender_C"])
    bankrupt = st.selectbox("Ever Bankrupt or Foreclosure?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

# Add some spacing
st.markdown("<br>", unsafe_allow_html=True)

# Predict button - centered
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button("🔍 Evaluate Loan Application")

# Prediction logic
if predict_button:
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
    
    # One-hot encode the user's input
    input_data_encoded = pd.get_dummies(input_data, columns=['Reason', 'Fico_Score_group', 'Employment_Status', 'Employment_Sector', 'Lender'], drop_first=True)
    
    # Add any "missing" columns the model expects (fill with 0)
    model_columns = model.feature_names_in_
    for col in model_columns:
        if col not in input_data_encoded.columns:
            input_data_encoded[col] = 0
    
    # Reorder/filter columns to exactly match the model's training data
    input_data_encoded = input_data_encoded[model_columns]
    
    # Make prediction
    prediction = model.predict(input_data_encoded)[0]
    probability = model.predict_proba(input_data_encoded)[0]
    
    # Calculate debt-to-income ratio
    dti_ratio = (housing_payment / monthly_income * 100) if monthly_income > 0 else 0
    
    # Application Summary Card
    st.markdown("""
        <div class='summary-card'>
            <div class='summary-title'>📋 Application Summary</div>
            <div style='display: flex; justify-content: space-between;'>
                <div style='flex: 1;'>
                    <div class='summary-section'>
                        <div class='summary-section-title'>Financial Details:</div>
                        <ul style='margin: 0; padding-left: 1.5rem;'>
                            <li>Requested Amount: ${:,}</li>
                            <li>FICO Score: {} ({})</li>
                            <li>Monthly Income: ${:,}</li>
                            <li>Housing Payment: ${:,}</li>
                            <li>Debt-to-Income: {:.1f}%</li>
                        </ul>
                    </div>
                </div>
                <div style='flex: 1;'>
                    <div class='summary-section'>
                        <div class='summary-section-title'>Personal Details:</div>
                        <ul style='margin: 0; padding-left: 1.5rem;'>
                            <li>Reason: {}</li>
                            <li>Employment: {} - {}</li>
                            <li>Lender: {}</li>
                            <li>Bankruptcy History: {}</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
    """.format(
        requested_loan, 
        fico_score, 
        fico_group,
        monthly_income, 
        housing_payment,
        dti_ratio,
        reason.replace("_", " "),
        employment_status.capitalize(),
        employment_sector,
        lender.replace("_", " "),
        "No" if bankrupt == 0 else "Yes"
    ), unsafe_allow_html=True)
    
    # Display result with gauge
    st.markdown("<br>", unsafe_allow_html=True)
    
    if prediction == 1:
        confidence = probability[1] * 100
        st.markdown(f"""
            <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                 border-radius: 15px; color: white; margin: 2rem 0;'>
                <h2 style='margin: 0; font-size: 2.5rem;'>✅ LOAN APPROVED</h2>
                <p style='font-size: 1.2rem; margin-top: 1rem;'>Model Confidence: {confidence:.1f}%</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Progress bar for confidence
        st.progress(confidence / 100)
        
    else:
        confidence = probability[0] * 100
        st.markdown(f"""
            <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                 border-radius: 15px; color: white; margin: 2rem 0;'>
                <h2 style='margin: 0; font-size: 2.5rem;'>❌ LOAN DENIED</h2>
                <p style='font-size: 1.2rem; margin-top: 1rem;'>Model Confidence: {confidence:.1f}%</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Progress bar for confidence
        st.progress(confidence / 100)

# Footer
st.markdown("""
    <div class='footer'>
        <div class='footer-section' style='font-weight: 600; font-size: 1rem; color: #333;'>
            ⚠️ Important Disclaimer
        </div>
        <div class='footer-section'>
            This tool is for educational purposes only. Predictions are based on historical data and machine learning models. 
            This should NOT be considered financial advice or used for actual lending decisions.
        </div>
        <div class='footer-section' style='margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #ddd;'>
            <strong>Created by Brooks Ferri & Nathan Olson</strong> | BUS 458 Final Project | NC State University
        </div>
        <div class='footer-section' style='font-size: 0.85rem; color: #999;'>
            Model: Logistic Regression | Last Updated: December 2024
        </div>
    </div>
""", unsafe_allow_html=True)
