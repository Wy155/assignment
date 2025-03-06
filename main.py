import streamlit as st
import joblib
import numpy as np

# Load the model
model = joblib.load(r"C:\Users\Wei Ying\AI\credit_risk.joblib")

# Streamlit app
st.title("Credit Risk Prediction Dashboard")

st.sidebar.header("User Input Features")

# Collecting user inputs (customize these fields based on your model's features)

age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
income = st.sidebar.number_input("Monthly Income ($)", min_value=0, value=3000)
loan_amount = st.sidebar.number_input("Loan Amount ($)", min_value=0, value=10000)
loan_term = st.sidebar.slider("Loan Term (months)", 6, 60, 36)
credit_score = st.sidebar.slider("Credit Score", 300, 850, 650)

# Prediction
if st.sidebar.button("Predict Credit Risk"):
    features = np.array([[age, income, loan_amount, loan_term, credit_score]])
    prediction = model.predict(features)
    
    st.subheader("Prediction Result")
    if prediction[0] == 1:
        st.success("High Risk")
    else:
        st.success("Low Risk")

st.markdown("\n**Note:** This is a demo app. The prediction is based on the input features and model logic.")
