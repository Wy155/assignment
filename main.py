import streamlit as st
import joblib
import numpy as np

# Load the model
model = joblib.load('credit_risk.joblib')

# Streamlit app
st.title("Credit Risk Prediction Dashboard")

st.sidebar.header("User Input Features")

# Collecting user inputs (match the number of features expected by the model)
feature_names = [
    "person_age",
    "person_income",
    "person_home_ownership",
    "person_emp_length",
    "loan_intent",
    "loan_amnt",
    "loan_int_rate",
    "loan_status",
    "loan_percent_income",
    "cb_person_default_on_file",
    "cb_person_cred_hist_length"
]

feature_inputs = []
for feature_name in feature_names:
    feature = st.sidebar.number_input(feature_name, value=0.0)
    feature_inputs.append(feature)

# Prediction
if st.sidebar.button("Predict Credit Risk"):
    features = np.array([feature_inputs])
    st.write(f"Input shape: {features.shape}")  # Debugging line
    try:
        prediction = model.predict(features)
        
        st.subheader("Prediction Result")
        if prediction[0] == 1:
            st.success("High Risk")
        else:
            st.success("Low Risk")
    except ValueError as e:
        st.error(f"Prediction failed: {e}")

st.markdown("\n**Note:** This is a demo app. The prediction is based on the input features and model logic.")

# Let me know if you want to tweak anything! 🚀
