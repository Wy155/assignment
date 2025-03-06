import streamlit as st
import joblib
import numpy as np

# Load the model
model = joblib.load('credit_risk.joblib')

# Streamlit app
st.title("Credit Risk Prediction Dashboard")

st.sidebar.header("User Input Features")

# Collecting user inputs (match the number of features expected by the model)

feature_inputs = []
for i in range(16):
    feature = st.sidebar.number_input(f"Feature {i+1}", value=0.0)
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
