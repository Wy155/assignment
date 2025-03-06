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
    "loan_percent_income",
    "loan_int_rate",
    "person_income",
    "loan_amnt",
    "person_home_ownership_RENT",
    "person_emp_length",
    "person_age",
    "cb_person_cred_hist_length",
    "cb_person_default_on_file_Y",
    "person_home_ownership_OWN",
    "loan_intent_HOMEIMPROVEMENT",
    "loan_intent_MEDICAL",
    "loan_intent_EDUCATION",
    "loan_intent_PERSONAL",
    "loan_intent_VENTURE",
    "person_home_ownership_OTHER"
]

# Separate important and optional features
important_features = [
    ("loan_percent_income", float),
    ("loan_int_rate", float),
    ("person_income", float),
    ("loan_amnt", float),
    ("person_age", int),
    ("cb_person_cred_hist_length", int)
]

optional_features = [
    ("person_home_ownership_RENT", int),
    ("person_emp_length", int),
    ("cb_person_default_on_file_Y", int),
    ("person_home_ownership_OWN", int),
    ("loan_intent_HOMEIMPROVEMENT", int),
    ("loan_intent_MEDICAL", int),
    ("loan_intent_EDUCATION", int),
    ("loan_intent_PERSONAL", int),
    ("loan_intent_VENTURE", int),
    ("person_home_ownership_OTHER", int)
]

feature_inputs = []

st.subheader("Key Information")
for feature_name, feature_type in important_features:
    feature = st.number_input(feature_name, value=0, format="%d") if feature_type == int else st.number_input(feature_name, value=0.0)
    feature_inputs.append(feature)

with st.expander("Additional Information (Optional)"):
    for feature_name, feature_type in optional_features:
        feature = st.number_input(feature_name, value=0, format="%d") if feature_type == int else st.number_input(feature_name, value=0.0)
        feature_inputs.append(feature)

# Prediction
if st.button("Predict Credit Risk"):
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
