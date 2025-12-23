import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# 1. SETUP & STYLE
st.set_page_config(page_title="Credit Risk AI", page_icon="🏦")
st.title("🏦 AI Credit Risk Analyzer")
st.write("This tool helps loan officers make data-driven decisions using **LightGBM** & **Bureau History**.")

# 2. LOAD MODEL
@st.cache_resource
def load_model():
    model = joblib.load('models/final_model_lgbm_boosted.joblib')
    return model

model = load_model()

# 3. SIDEBAR: INPUTS
st.sidebar.header("1. Applicant Details")
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.slider("Age", 20, 70, 30)

st.sidebar.header("2. Financial Info")
income = st.sidebar.number_input("Annual Income ($)", min_value=10000, value=50000, step=5000)
loan_amount = st.sidebar.number_input("Loan Amount Requested ($)", min_value=10000, value=250000, step=10000)
annuity = st.sidebar.number_input("Loan Annuity (Monthly Payment)", min_value=1000, value=15000, step=1000)

st.sidebar.header("3. Credit History (The 'Bureau Boost')")
bureau_count = st.sidebar.slider("Number of Past Loans (Bureau)", 0, 20, 2)
ext_source_2 = st.sidebar.slider("Credit Score (External Source 2)", 0.0, 1.0, 0.5)
ext_source_3 = st.sidebar.slider("Credit Score (External Source 3)", 0.0, 1.0, 0.5)

# 4. PREDICTION LOGIC
if st.button("Analyze Risk Profile"):
    
    # A. Create the "Ghost Frame" (All features set to 0 initially)
    features = model.feature_name()
    input_df = pd.DataFrame(0, index=[0], columns=features)
    
    # B. Overwrite with User Inputs
    # Note: We mapped the math to your exact column names
    input_df['AMT_INCOME_TOTAL'] = income
    input_df['AMT_CREDIT'] = loan_amount
    input_df['AMT_ANNUITY'] = annuity
    input_df['DAYS_BIRTH'] = -age * 365  # Convert age to days
    input_df['EXT_SOURCE_2'] = ext_source_2
    input_df['EXT_SOURCE_3'] = ext_source_3
    input_df['BUREAU_LOAN_COUNT'] = bureau_count  # <-- Your new feature!
    
    # C. Feature Engineering (On the fly!)
    # We must recreate the custom features we made in training
    input_df['CREDIT_TERM'] = input_df['AMT_CREDIT'] / (input_df['AMT_ANNUITY'] + 1)
    
    # D. Predict
    prob = model.predict(input_df)[0]
    
    # 5. DISPLAY RESULTS
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(label="Default Probability", value=f"{prob:.1%}")
    
    with col2:
        if prob > 0.45: # Threshold
            st.error("❌ REJECT APPLICATION")
            st.write(f"Risk Level: **High**")
        else:
            st.success("✅ APPROVE APPLICATION")
            st.write(f"Risk Level: **Low**")

# 6. EXPLAINABILITY (The "Why" & The "How to Fix")
    st.divider()
    st.subheader("📝 Decision Explanation")
    
    # Calculate "Debt-to-Income" (DTI)
    # Note: We add +1 to income to avoid division by zero errors
    debt_to_income = input_df['AMT_CREDIT'][0] / (input_df['AMT_INCOME_TOTAL'][0] + 1)
    
    # --- IF REJECTED (Red Section) ---
    if prob > 0.45:
        st.error("❌ Reasons for Rejection")
        
        # 1. Diagnose the specific problems
        if input_df['EXT_SOURCE_2'][0] < 0.5:
            st.write(f"• **Credit Score is too low** ({input_df['EXT_SOURCE_2'][0]:.2f}). Banks usually require at least 0.5.")
        
        if input_df['BUREAU_LOAN_COUNT'][0] == 0:
            st.write(f"• **No Credit History:** You have 0 past loans. Banks prefer customers with a track record.")
        elif input_df['BUREAU_LOAN_COUNT'][0] > 5:
            st.write(f"• **Too many active loans** ({input_df['BUREAU_LOAN_COUNT'][0]}). This indicates high existing debt.")
            
        if debt_to_income > 3.0:
            st.write(f"• **Loan amount is too high** compared to income ({debt_to_income:.1f}x your salary).")
            
        # ... inside the "if prob > 0.45" block ...

        # 2. The "Fixer" Feature (Smarter Version)
        st.subheader("💡 How to get Approved")
        
        safe_loan = input_df['AMT_INCOME_TOTAL'][0] * 2.5
        current_loan = input_df['AMT_CREDIT'][0]
        
        # SCENARIO 1: Rich but Bad Score (The "Rich Ghost")
        if input_df['AMT_INCOME_TOTAL'][0] > 100000 and input_df['EXT_SOURCE_2'][0] < 0.6:
            st.warning("💰 You have high income, but your **Credit Score** is hurting you.")
            st.write("• The model relies 70% on Credit Score and only 10% on Income.")
            st.write("• **Solution:** Wait 6 months to improve your score before applying.")
            
        # SCENARIO 2: Loan too big
        elif current_loan > safe_loan:
            st.info(f"To improve your chances, consider reducing the loan amount to **${safe_loan:,.0f}**.")
            
        # SCENARIO 3: Generic Advice
        else:
            st.info("Your financial ratios look okay. The issue is likely your **Credit History**.")
            st.write("• **Solution:** Add a co-signer with a better Credit Score (Ext_Source_2 > 0.6).")

    # --- IF APPROVED (Green Section) ---
    else:
        st.success("✅ Application Strengths")
        st.write("Your profile looks solid. Here is what helped you:")
        
        if input_df['EXT_SOURCE_2'][0] >= 0.5:
            st.write(f"• **Strong Credit Score** ({input_df['EXT_SOURCE_2'][0]:.2f}).")
        if debt_to_income < 2.0:
            st.write(f"• **Healthy Debt Ratio:** The loan is affordable for your income.")
        if input_df['BUREAU_LOAN_COUNT'][0] <= 2:
            st.write("• **Low Debt Burden:** You don't have many other loans.")