import streamlit as st
import joblib
import numpy as np

# Load the model
model = joblib.load('credit_risk.joblib')

# Streamlit app
st.title("Credit Risk Prediction Dashboard")

st.sidebar.header("User Input Features")

# Focus on the most important features
important_features = [
    "loan_percent_income",
    "loan_int_rate",
    "person_income",
    "loan_amnt",
    "person_age",
    "cb_person_cred_hist_length"
]

optional_features = [
    "person_home_ownership_RENT",
    "person_home_ownership_OTHER",
    "person_home_ownership_OWN",
    "cb_person_default_on_file_Y",
    "loan_intent_HOMEIMPROVEMENT",
    "loan_intent_MEDICAL",
    "loan_intent_EDUCATION",
    "loan_intent_PERSONAL",
    "loan_intent_VENTURE"
]

feature_inputs = []

st.subheader("Key Information")
for feature_name in important_features:
    feature = st.number_input(feature_name, value=0.0)
    feature_inputs.append(feature)

with st.expander("Additional Information (Optional)"):
    for feature_name in optional_features:
        feature = st.number_input(feature_name, value=0.0)
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

# Let me know if you want to tweak anything! 🚀
