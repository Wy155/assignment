import streamlit as st
import joblib
import numpy as np

# Load the model
model = joblib.load('credit_risk.joblib')

# Streamlit app
st.title("Credit Risk Prediction Dashboard")

st.sidebar.header("User Input Features")

# Define features in the correct format
features = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length"
]

# Collect user inputs in the sidebar
person_age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30, step=1)
person_income = st.sidebar.number_input("Income ($)", min_value=0.0, value=50000.0)
person_emp_length = st.sidebar.number_input("Employment Length (Years)", min_value=0, value=5, step=1)
loan_amnt = st.sidebar.number_input("Loan Amount ($)", min_value=0.0, value=10000.0)
loan_int_rate = st.sidebar.number_input("Loan Interest Rate (%)", min_value=0.0, max_value=100.0, value=5.0)
loan_percent_income = st.sidebar.number_input("Loan Percent Income (%)", min_value=0.0, max_value=100.0, value=20.0)
cb_person_cred_hist_length = st.sidebar.number_input("Credit History Length (Years)", min_value=0, value=10, step=1)

# Prepare features for prediction
feature_inputs = [
    person_age,
    person_income,
    person_emp_length,
    loan_amnt,
    loan_int_rate,
    loan_percent_income,
    cb_person_cred_hist_length
]

# Prediction
if st.sidebar.button("Predict Credit Risk"):
    features_array = np.array([feature_inputs])
    st.write(f"Input features: {feature_inputs}")
    try:
        prediction = model.predict(features_array)
        
        st.subheader("Prediction Result")
        if prediction[0] == 1:
            st.success("High Risk")
        else:
            st.success("Low Risk")
    except ValueError as e:
        st.error(f"Prediction failed: {e}")

st.markdown("\n**Note:** This is a demo app. The prediction is based on the input features and model logic.")
